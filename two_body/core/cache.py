"""
Infraestructura de cache para las evaluaciones del optimizador.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional


class LRUCache:
    """Implementacion minima de un cache LRU."""

    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = max(1, int(capacity))
        self.store: "OrderedDict[Any, Any]" = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: Any) -> Any:
        if key in self.store:
            self.store.move_to_end(key)
            self.hits += 1
            return self.store[key]
        self.misses += 1
        return None

    def set(self, key: Any, value: Any) -> None:
        if key in self.store:
            self.store.move_to_end(key)
        self.store[key] = value
        if len(self.store) > self.capacity:
            self.store.popitem(last=False)

    def compact(self, keep: Optional[int] = None) -> None:
        limit = min(self.capacity, keep) if keep is not None else self.capacity
        while len(self.store) > limit:
            self.store.popitem(last=False)


class HierarchicalCache:
    """Dos niveles de cache: aproximado y exacto."""

    def __init__(self, approx_max: int = 5000, exact_max: int = 2000) -> None:
        self.approx = LRUCache(approx_max)
        self.exact = LRUCache(exact_max)

    def get_approx(self, key: Any) -> Any:
        return self.approx.get(key)

    def set_approx(self, key: Any, value: Any) -> None:
        self.approx.set(key, value)

    def get_exact(self, key: Any) -> Any:
        return self.exact.get(key)

    def set_exact(self, key: Any, value: Any) -> None:
        self.exact.set(key, value)

    def stats(self) -> Dict[str, Any]:
        hits = self.approx.hits + self.exact.hits
        misses = self.approx.misses + self.exact.misses
        return {
            "approx": {"hits": self.approx.hits, "misses": self.approx.misses},
            "exact": {"hits": self.exact.hits, "misses": self.exact.misses},
            "hit_rate": hits / (hits + misses) if (hits + misses) > 0 else 0.0,
        }

    def compact(self) -> None:
        self.approx.compact()
        self.exact.compact()


if __name__ == "__main__":
    cache = HierarchicalCache(approx_max=3, exact_max=2)
    for i in range(5):
        cache.set_exact(i, i * i)
    print("Estado cache exacto:", list(cache.exact.store.items()))
