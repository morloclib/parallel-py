"""Fork-pool implementations of the morloc `parallel` module.

Workers are processes, not threads: a morloc manifold is ordinary Python and
the GIL would serialize it. `pool.py` forces the `fork` start method on every
platform precisely so a worker inherits the manifolds exec'd into the pool's
globals, which is what lets a mapped function be a morloc closure -- including
one that calls into another language.

Work units are index ranges rather than element lists. A range is a pair of
ints whatever the elements weigh, so the schedule is decided without pickling
anything, and a worker slices the input it already inherited.
"""

import multiprocessing
import threading
import os

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
# rather than by pickling. Setting them here before the pool is created means
# each child snapshots them at fork, so a task carries only its (lo, hi) pair
# -- otherwise every unit would re-pickle the entire input, which for a list of
# sequences costs more than the work being parallelized.
#
# The lock makes the handoff safe under the thread-model pool (macOS), where
# two dispatches can be in flight in one process.
_CTX = None
_CTX_LOCK = threading.Lock()


def _apply_range(span):
    fn, xs = _CTX
    lo, hi = span
    return [fn(x) for x in xs[lo:hi]]


def _apply_range_concat(span):
    fn, xs = _CTX
    lo, hi = span
    out = []
    for x in xs[lo:hi]:
        out.extend(fn(x))
    return out


def _run(opts, worker, fn, xs):
    """Dispatch `worker` over the schedule and return per-unit results in input
    order. A single unit runs inline: forking to hand one worker the whole list
    buys nothing and costs a process."""
    global _CTX
    spans = _ranges(len(xs), opts)
    if not spans:
        return []
    with _CTX_LOCK:
        _CTX = (fn, xs)
        try:
            if len(spans) == 1:
                return [worker(spans[0])]
            with multiprocessing.Pool(processes=_workers(opts)) as pool:
                # chunksize=1 because the schedule IS the chunking; letting the
                # pool batch units on top of it would undo the shrinking tail.
                return pool.map(worker, spans, chunksize=1)
        finally:
            _CTX = None


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


# ── Streams ────────────────────────────────────────────────────────────────
#
# The loop lives here rather than in morloc because a suspended computation
# passed as a parameter is serialized under its result schema when a recursive
# call crosses a manifold boundary, so the natural recursive stage does not
# survive. See /work/plans/parallel-issues.md.
#
# A `Try e a` value crosses as a ("constructor", (fields...)) pair, so `Ok xs`
# arrives as ("Ok", (xs,)).

def _unwrap_try(t):
    tag, fields = t
    if tag == "Err":
        raise RuntimeError(fields[0])
    return fields[0]


def _ordered(opts):
    return int(opts.get("order", INPUT_ORDER)) == INPUT_ORDER


def morloc_psconcat_map_with(opts, fn, pull, sink):
    """Pull batches, map each batch's elements in parallel, push the results.

    A batch is the unit the source already chose (a stream sub-packet), so it
    is also the unit of parallelism: the pool is built per batch and the sink
    runs in this process, which is where the output handle lives.
    """
    while True:
        batch = _unwrap_try(pull())
        if not batch:
            return None
        sink(morloc_pconcat_map_with(opts, fn, batch))


def morloc_psfold_with(opts, combine, identity, fn, pull):
    """Reduce a stream as batches arrive.

    Partials are combined in batch order, so the answer does not depend on
    which worker finished first. That is what makes a run reproducible for a
    non-associative `combine` such as floating-point addition -- provided the
    chunking is itself deterministic (see ParChunking).
    """
    acc = identity
    while True:
        batch = _unwrap_try(pull())
        if not batch:
            return ("Ok", (acc,))
        for y in morloc_pmap_with(opts, fn, batch):
            acc = combine(acc, y)
