"""Tests for ``_BoundedAttrCache`` — the LRU helper backing the
``JVMView`` / ``JavaPackage`` / ``JavaClass`` attribute caches added
for issue #557.

Pure Python unit tests; no JVM needed.
"""
import threading
import time
import unittest

from py4j.java_gateway import _BoundedAttrCache


class BoundedAttrCacheTest(unittest.TestCase):

    def test_get_miss_returns_none(self):
        cache = _BoundedAttrCache(maxsize=4)
        self.assertIsNone(cache.get("absent"))

    def test_put_then_get_returns_value(self):
        cache = _BoundedAttrCache(maxsize=4)
        sentinel = object()
        cache.put("k", sentinel)
        self.assertIs(cache.get("k"), sentinel)

    def test_put_replaces_existing_value(self):
        cache = _BoundedAttrCache(maxsize=4)
        cache.put("k", "first")
        cache.put("k", "second")
        self.assertEqual(cache.get("k"), "second")

    def test_eviction_drops_oldest(self):
        # maxsize=3, insert 4 distinct keys; the first inserted should
        # be evicted because nothing promoted it.
        cache = _BoundedAttrCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("d", 4)
        self.assertIsNone(cache.get("a"))
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)
        self.assertEqual(cache.get("d"), 4)

    def test_get_promotes_to_most_recent(self):
        # After touching ``a``, the next eviction should drop ``b``
        # (now the oldest), not ``a``.
        cache = _BoundedAttrCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.get("a")  # promote a
        cache.put("d", 4)
        self.assertEqual(cache.get("a"), 1)
        self.assertIsNone(cache.get("b"))
        self.assertEqual(cache.get("c"), 3)
        self.assertEqual(cache.get("d"), 4)

    def test_put_existing_promotes(self):
        # Re-putting the same key should also promote it past the
        # eviction threshold.
        cache = _BoundedAttrCache(maxsize=3)
        cache.put("a", 1)
        cache.put("b", 2)
        cache.put("c", 3)
        cache.put("a", 10)  # promote + replace
        cache.put("d", 4)
        self.assertEqual(cache.get("a"), 10)
        self.assertIsNone(cache.get("b"))

    def test_concurrent_get_put_does_not_raise(self):
        # Smoke test for the race fix in get()/put(): two threads
        # hammering the cache near its eviction boundary should never
        # raise (KeyError on the second op of a non-atomic pair would
        # be the regression). 50 keys, maxsize 10 → constant churn.
        cache = _BoundedAttrCache(maxsize=10)
        errors = []
        stop = threading.Event()

        def writer():
            try:
                i = 0
                while not stop.is_set():
                    cache.put("k{0}".format(i % 50), i)
                    i += 1
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                while not stop.is_set():
                    for j in range(50):
                        cache.get("k{0}".format(j))
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        # Run for ~250 ms — long enough to hit the race in practice
        # but short enough to keep the test fast.
        time.sleep(0.25)
        stop.set()
        for t in threads:
            t.join(timeout=5.0)
        self.assertEqual(errors, [])

    def test_concurrent_clear_during_get_put_does_not_raise(self):
        # Phase 2 of the cache PR makes ``JavaGateway.shutdown()`` call
        # ``_attr_cache._cache.clear()``. That clear can race with
        # readers and writers from other threads. Each individual
        # OrderedDict op is atomic under the GIL, but the chained ops
        # in ``get`` (lookup + ``move_to_end``) and ``put`` (insert +
        # ``popitem``) can intersect with a ``clear()`` between their
        # two steps. The race must never raise — the ``try/except
        # KeyError`` in ``get`` and ``put`` is the safety net.
        cache = _BoundedAttrCache(maxsize=10)
        errors = []
        stop = threading.Event()

        def writer():
            try:
                i = 0
                while not stop.is_set():
                    cache.put("k{0}".format(i % 50), i)
                    i += 1
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                while not stop.is_set():
                    for j in range(50):
                        cache.get("k{0}".format(j))
            except Exception as e:
                errors.append(e)

        def clearer():
            try:
                while not stop.is_set():
                    cache._cache.clear()
                    # Tiny sleep so clearer doesn't dominate scheduler.
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
            threading.Thread(target=clearer),
        ]
        for t in threads:
            t.start()
        time.sleep(0.25)
        stop.set()
        for t in threads:
            t.join(timeout=5.0)
        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
