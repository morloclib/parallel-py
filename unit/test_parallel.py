"""Unit tests of parallel.py that a morloc program cannot express: the worker
count. Run with `make unit`."""

import enum
import os
import sys
import types
import unittest

# parallel.py imports pymorloc for failure messages and the CPU count; the
# real one is used when it can be imported, so the count is the runtime's.
try:
    import pymorloc  # noqa: F401
except ImportError:
    sys.modules["pymorloc"] = types.SimpleNamespace(
        failure_message=str, available_cpus=lambda: len(os.sched_getaffinity(0)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import parallel  # noqa: E402

SHRINKING = {"chunking": parallel.SHRINKING_CHUNKS}


class WorkerCount(unittest.TestCase):
    def setUp(self):
        self.saved = os.environ.pop("MORLOC_THREADS", None)

    def tearDown(self):
        os.environ.pop("MORLOC_THREADS", None)
        if self.saved is not None:
            os.environ["MORLOC_THREADS"] = self.saved

    def test_precedence(self):
        cpus = parallel._available_cpus()
        self.assertGreaterEqual(cpus, 1)
        self.assertEqual(parallel._workers({}), cpus)
        self.assertEqual(parallel._workers({"workers": None}), cpus)
        self.assertEqual(parallel._workers({"workers": 0}), cpus)
        self.assertEqual(parallel._workers({"workers": -3}), cpus)
        self.assertEqual(parallel._workers({"workers": 3}), 3)
        os.environ["MORLOC_THREADS"] = "5"
        self.assertEqual(parallel._workers({}), 5)
        self.assertEqual(parallel._workers({"workers": 0}), 5)
        self.assertEqual(parallel._workers({"workers": 3}), 3)
        for v in ("4 ", " 4", "  4  "):
            os.environ["MORLOC_THREADS"] = v
            self.assertEqual(parallel._workers({}), 4)
        for v in ("0", "", " ", "+4", "-4", "4_0", "1e2", "4x", "\t4", "99999999999999999999", "\u0664"):
            os.environ["MORLOC_THREADS"] = v
            self.assertEqual(parallel._workers({}), cpus)

    @unittest.skipUnless(os.environ.get("MLCPAR_EXPECT_CPUS"), "run under taskset")
    def test_affinity(self):
        self.assertEqual(parallel._workers({}), int(os.environ["MLCPAR_EXPECT_CPUS"]))


class EnumTags(unittest.TestCase):
    """A `data` field arrives as the generated enum.Enum."""

    def test_enum_options(self):
        Chunking = enum.Enum("ParChunking", [("EvenChunks", 0), ("ShrinkingChunks", 1), ("FixedChunks", 2)])
        Order = enum.Enum("ParOrder", [("InputOrder", 0), ("ArrivalOrder", 1)])
        for c in Chunking:
            o = {"workers": 3, "chunkSize": 7, "chunking": c, "order": Order.ArrivalOrder}
            self.assertEqual(parallel._ranges(100, o, 3), parallel._ranges(100, dict(o, chunking=c.value), 3))
        self.assertFalse(parallel._ordered({"order": Order.ArrivalOrder}))
        self.assertTrue(parallel._ordered({"order": Order.InputOrder}))
        self.assertTrue(parallel._ordered({}))


FOLD_SIZES = [0, 1, 2, 255, 256, 257, 511, 1000, 5000]


def all_opts():
    for w in (1, 2, 7, 12):
        for c in (parallel.EVEN_CHUNKS, parallel.SHRINKING_CHUNKS, parallel.FIXED_CHUNKS):
            for k in ((3, 100000) if c == parallel.FIXED_CHUNKS else (None,)):
                yield {"workers": w, "chunking": c, "chunkSize": k}


def big_small(i):
    """Terms whose sum depends on the grouping: 1e16 + 1 is 1e16."""
    return 1.0e16 if i % 4 == 0 else -1.0e16 if i % 4 == 2 else 1.0


def fold_sum(v):
    s = 0.0
    for x in v:
        s += x
    return s


def merge_counts(ds):
    out = {}
    for d in ds:
        for k, n in d.items():
            out[k] = out.get(k, 0) + n
    return out


class FoldBlocks(unittest.TestCase):
    def test_exact(self):
        cat = lambda ls: [y for l in ls for y in l]
        for n in FOLD_SIZES:
            xs = [i * 7919 % 1009 for i in range(n)]
            for o in all_opts():
                pf = lambda f, c: parallel.morloc_pfold_map_blocks(o, f, c, xs)
                self.assertEqual(pf(lambda x: 3 * x - 1000, sum), sum(3 * x - 1000 for x in xs))
                self.assertEqual(pf(lambda x: [x, -x], cat), cat([x, -x] for x in xs))
                self.assertEqual(pf(lambda x: "%d," % x, "".join), "".join("%d," % x for x in xs))
                self.assertEqual(pf(lambda x: {x % 13: 1}, merge_counts), merge_counts({x % 13: 1} for x in xs))

    def test_real_depends_on_n_only(self):
        any_differs = False
        for n in FOLD_SIZES:
            size = max(1, -(-n // 256))
            blocks = [fold_sum(big_small(i) for i in range(lo, min(lo + size, n))) for lo in range(0, n, size)]
            self.assertLessEqual(len(blocks), 256)
            ref = fold_sum(blocks)
            any_differs = any_differs or fold_sum(big_small(i) for i in range(n)) != ref
            for o in all_opts():
                self.assertEqual(parallel.morloc_pfold_map_blocks(o, big_small, fold_sum, list(range(n))), ref)
        self.assertTrue(any_differs)

    def test_one_mconcat_per_block(self):
        for n in FOLD_SIZES:
            calls = []

            def mconcat(ms):
                calls.append(len(ms))
                return sum(ms)

            got = parallel.morloc_pfold_map_blocks({"workers": 1}, lambda x: x, mconcat, [1] * n)
            self.assertEqual(got, n)
            size = max(1, -(-n // 256))
            nblocks = -(-n // size)
            self.assertEqual(len(calls), 1 if n == 0 else nblocks + 1)
            self.assertLessEqual(max(calls), max(nblocks, size))

    def test_failure_reaches_caller(self):
        def f(x):
            if x == 2345:
                raise ValueError("boom-at-2345")
            return x

        for o in ({"workers": 1}, {"workers": 4}):
            with self.assertRaises(Exception) as e:
                parallel.morloc_pfold_map_blocks(o, f, sum, list(range(3000)))
            self.assertIn("boom-at-2345", str(e.exception))


if __name__ == "__main__":
    unittest.main()
