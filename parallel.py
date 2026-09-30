"""Fork-pool implementations of the morloc `parallel` module.

Workers are processes, not threads: a morloc manifold is ordinary Python and
the GIL would serialize it. Workers are forked so they inherit the manifolds
exec'd into the pool's globals, which is what lets a mapped function be a
morloc closure -- including one that calls into another language.

For the list functions, work units are index ranges rather than element
lists. A range is a pair of ints whatever the elements weigh, so the schedule
is decided without pickling anything, and a worker slices the input it already
inherited. Results are pickled back.
"""

import itertools
import multiprocessing
import os
import queue

# Tag order follows the `data ParChunking` declaration in parallel/main.loc.
EVEN_CHUNKS = 0
SHRINKING_CHUNKS = 1
FIXED_CHUNKS = 2

# Tag order follows `data ParOrder`.
INPUT_ORDER = 0
ARRIVAL_ORDER = 1


def _int_opt(opts, key, fallback):
    v = opts.get(key)
    return fallback if v is None or v <= 0 else int(v)


def _workers(opts):
    return _int_opt(opts, "workers", os.cpu_count() or 1)


def _ranges(n, opts):
    """Split [0, n) into work units per the requested schedule."""
    if n <= 0:
        return []
    w = max(1, _workers(opts))
    mode = int(opts.get("chunking", SHRINKING_CHUNKS))

    if mode == EVEN_CHUNKS:
        # One contiguous block per worker; no coordination beyond the dispatch.
        size = -(-n // w)
        return [(i, min(i + size, n)) for i in range(0, n, size)]

    if mode == FIXED_CHUNKS:
        size = max(1, _int_opt(opts, "chunkSize", 1))
        return [(i, min(i + size, n)) for i in range(0, n, size)]

    # Shrinking (factoring): each round deals w units of half the remaining
    # work. Early units are large, so a run of cheap elements pays almost no
    # coordination; the tail is single elements, so one expensive element
    # cannot strand a worker holding a large unaccounted block.
    out = []
    lo = 0
    while lo < n:
        remaining = n - lo
        size = max(1, remaining // (2 * w))
        for _ in range(w):
            if lo >= n:
                break
            hi = min(lo + size, n)
            out.append((lo, hi))
            lo = hi
    return out


# The mapped function and the input list reach workers by fork inheritance
# rather than by pickling. Registering them here before the pool is created
# means each child snapshots them at fork, so a task carries only a token and a
# (lo, hi) pair -- otherwise every unit would re-pickle the entire input, which
# for a list of sequences costs more than the work being parallelized.
#
# Each dispatch has its own token, so concurrent dispatches (the thread-model
# pool) and nested ones (a mapped function that itself calls pmap) never share
# an entry. No lock is held across the fork: a child would inherit it held.
_CONTEXTS = {}
_TOKENS = itertools.count()

# `fork` is required, not preferred: a morloc manifold lives in the pool's
# globals, not in an importable module, so a spawned or forkserver child could
# not find it. Named explicitly because the platform default is not `fork`
# everywhere.
_FORK = multiprocessing.get_context("fork")


def _apply_range(task):
    token, lo, hi = task
    fn, xs = _CONTEXTS[token]
    return [fn(x) for x in xs[lo:hi]]


def _apply_range_concat(task):
    token, lo, hi = task
    fn, xs = _CONTEXTS[token]
    out = []
    for x in xs[lo:hi]:
        out.extend(fn(x))
    return out


def _in_worker():
    """True inside a pool worker. Workers are daemonic, and a daemonic process
    may not have children, so a nested dispatch runs inline -- the outer level
    already occupies every core."""
    return multiprocessing.current_process().daemon


def _run(opts, worker, fn, xs):
    """Dispatch `worker` over the schedule and return per-unit results in input
    order. With one effective worker the units run inline: forking to hand one
    process the whole list buys nothing and costs a process."""
    spans = _ranges(len(xs), opts)
    if not spans:
        return []
    nproc = min(_workers(opts), len(spans))
    token = next(_TOKENS)
    _CONTEXTS[token] = (fn, xs)
    try:
        tasks = [(token, lo, hi) for lo, hi in spans]
        if nproc == 1 or _in_worker():
            return [worker(t) for t in tasks]
        with _FORK.Pool(processes=nproc) as pool:
            # chunksize=1 because the schedule IS the chunking; letting the
            # pool batch units on top of it would undo the shrinking tail.
            return pool.map(worker, tasks, chunksize=1)
    finally:
        del _CONTEXTS[token]


def morloc_pmap_with(opts, fn, xs):
    parts = _run(opts, _apply_range, fn, xs)
    return [y for part in parts for y in part]


def morloc_pconcat_map_with(opts, fn, xs):
    parts = _run(opts, _apply_range_concat, fn, xs)
    return [y for part in parts for y in part]


def morloc_pfilter_with(opts, pred, xs):
    parts = _run(opts, _apply_range, pred, xs)
    keep = [k for part in parts for k in part]
    return [x for x, k in zip(xs, keep) if k]


# Streams
# -------
#
# A stream stage is a pipeline. This process pulls batches and sinks results,
# because that is where the source and sink handles live; a pool forked once
# per stage maps work units. A unit's elements are pickled to a worker (a batch
# pulled after the fork cannot be inherited), while the mapped function is
# inherited through the token-keyed context like the list functions.
#
# `inflight` counts units dispatched and not yet delivered to the sink, so it
# bounds the reorder buffer as well as the units queued at the pool.


MAP, CONCAT, FILTER = 0, 1, 2


def _apply_unit(mode, fn, chunk):
    if mode == MAP:
        return [fn(x) for x in chunk]
    if mode == FILTER:
        return [x for x in chunk if fn(x)]
    out = []
    for x in chunk:
        out.extend(fn(x))
    return out


def _stream_unit(task):
    token, mode, chunk = task
    return _apply_unit(mode, _CONTEXTS[token], chunk)


def _inflight(opts, w):
    return max(1, _int_opt(opts, "inflight", 2 * w))


def _ordered(opts):
    return int(opts.get("order", INPUT_ORDER)) == INPUT_ORDER


def _pipeline(opts, fn, mode, pull, deliver, ordered):
    """Run a stream stage. `deliver` receives each unit's results in the order
    the stage promises. Returns (True, "") at end of stream, or (False, msg)
    after a read failure, once every dispatched unit has been delivered. A
    failure in a worker or in `deliver` propagates."""
    w = _workers(opts)
    if w == 1 or _in_worker():
        return _pipeline_inline(opts, fn, mode, pull, deliver)

    token = next(_TOKENS)
    _CONTEXTS[token] = fn
    done = queue.Queue()
    try:
        with _FORK.Pool(processes=w) as pool:
            return _pipeline_pool(opts, w, token, mode, pull, deliver, ordered, pool, done)
    finally:
        del _CONTEXTS[token]


def _pipeline_pool(opts, w, token, mode, pull, deliver, ordered, pool, done):
    limit = _inflight(opts, w)
    spans = []           # units of the current batch not yet dispatched
    batch = None
    reading = True       # False once the source is exhausted or failed
    status = (True, "")
    failure = None       # first worker failure
    next_seq = 0         # sequence number of the next unit dispatched
    next_out = 0         # sequence number of the next unit delivered (in order)
    running = 0          # dispatched and not yet completed
    undelivered = 0      # dispatched and not yet delivered
    held = {}            # completed units waiting for an earlier one

    def dispatch(seq, chunk):
        pool.apply_async(
            _stream_unit,
            ((token, mode, chunk),),
            callback=lambda r, seq=seq: done.put((seq, True, r)),
            error_callback=lambda e, seq=seq: done.put((seq, False, e)),
        )

    while True:
        while reading and failure is None and undelivered < limit:
            if not spans:
                ok, msg, batch = pull()
                if not ok:
                    reading, status = False, (False, msg)
                    break
                if not batch:
                    reading = False
                    break
                spans = _ranges(len(batch), opts)
                spans.reverse()
            lo, hi = spans.pop()
            dispatch(next_seq, batch[lo:hi])
            next_seq += 1
            running += 1
            undelivered += 1

        if running == 0:
            break

        seq, ok, result = done.get()
        running -= 1
        if not ok:
            if failure is None:
                failure = result
            continue
        if failure is not None:
            continue
        if ordered:
            held[seq] = result
            while next_out in held:
                ys = held.pop(next_out)
                next_out += 1
                undelivered -= 1
                if ys:
                    deliver(ys)
        else:
            undelivered -= 1
            if result:
                deliver(result)

    if failure is not None:
        raise failure
    return status


def _pipeline_inline(opts, fn, mode, pull, deliver):
    """The same contract with one worker, or inside a pool worker (which may
    not fork): units run in this process in input order."""
    while True:
        ok, msg, batch = pull()
        if not ok:
            return (False, msg)
        if not batch:
            return (True, "")
        for lo, hi in _ranges(len(batch), opts):
            ys = _apply_unit(mode, fn, batch[lo:hi])
            if ys:
                deliver(ys)


def morloc_psconcat_map_native(opts, fn, pull, sink):
    return _pipeline(opts, fn, CONCAT, pull, sink, _ordered(opts))


def morloc_psmap_native(opts, fn, pull, sink):
    return _pipeline(opts, fn, MAP, pull, sink, _ordered(opts))


def morloc_psfilter_native(opts, pred, pull, sink):
    return _pipeline(opts, pred, FILTER, pull, sink, _ordered(opts))


def morloc_psfold_native(opts, combine, identity, fn, pull):
    # The fold always consumes results in input order, whatever `order` asks:
    # a fixed fold order is what makes the answer schedule-independent.
    acc = [identity]

    def deliver(ys):
        a = acc[0]
        for y in ys:
            a = combine(a, y)
        acc[0] = a

    ok, msg = _pipeline(opts, fn, MAP, pull, deliver, True)
    return (ok, msg, acc[0])
