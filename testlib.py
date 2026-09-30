"""Synthetic stream source and sink for the shared `parallel` test suite.

A source is a list of batch sizes; its elements count up from 0 across
batches, and a negative size is a read failure at that point. A sink records
every batch it receives.

State lives in files, not module globals: the Python pool serves a re-entrant
call (a pull passed into a stage, for instance) from a sibling worker process,
which does not share this process's memory. Sibling workers share a parent, so
the parent's pid names the directory.
"""

import json
import os
import tempfile


def _dir():
    d = os.path.join(tempfile.gettempdir(), "morloc-parallel-test-%d" % os.getppid())
    os.makedirs(d, exist_ok=True)
    return d


def _claim(prefix):
    """Create a fresh numbered file and return its number."""
    d = _dir()
    k = 0
    while True:
        try:
            fd = os.open(os.path.join(d, "%s-%d" % (prefix, k)), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return k
        except FileExistsError:
            k += 1


def _path(prefix, k):
    return os.path.join(_dir(), "%s-%d" % (prefix, k))


def synth_open(sizes):
    k = _claim("src")
    with open(_path("src", k), "w") as fh:
        json.dump({"sizes": list(sizes), "i": 0, "start": 0}, fh)
    return k


def synth_next(k):
    path = _path("src", k)
    with open(path) as fh:
        state = json.load(fh)
    sizes, i, start = state["sizes"], state["i"], state["start"]
    if i >= len(sizes):
        return (True, [])
    n = sizes[i]
    state["i"] = i + 1
    if n >= 0:
        state["start"] = start + n
    with open(path, "w") as fh:
        json.dump(state, fh)
    if n < 0:
        return (False, [])
    return (True, list(range(start, start + n)))


def sink_open():
    return _claim("sink")


def sink_put(k, xs):
    with open(_path("sink", k), "a") as fh:
        fh.write(json.dumps(list(xs)) + "\n")
    return None


def sink_batches(k):
    with open(_path("sink", k)) as fh:
        return [json.loads(line) for line in fh if line.strip()]
