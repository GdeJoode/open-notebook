"""
Union-Find (Disjoint Set Union) data structure.

Efficiently tracks which entities should be merged together,
handling transitive relationships (if A=B and B=C, then A=C).
Uses path compression and union by rank for near-constant time operations.
"""

from typing import Dict, List


class UnionFind:
    """Union-Find with path compression and union by rank.

    Lazily initializes elements on first access so no upfront
    population is required.
    """

    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}
        self._rank: Dict[str, int] = {}

    def find(self, x: str) -> str:
        """Find the root representative of *x* with path compression."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0
            return x
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: str, y: str) -> None:
        """Merge the sets containing *x* and *y* (union by rank)."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self._rank[px] < self._rank[py]:
            px, py = py, px
        self._parent[py] = px
        if self._rank[px] == self._rank[py]:
            self._rank[px] += 1

    def get_groups(self, items: List[str]) -> Dict[str, List[str]]:
        """Group *items* by their root representative.

        Only items present in *items* appear in the output, even if
        the internal structure contains additional elements.
        """
        groups: Dict[str, List[str]] = {}
        for item in items:
            root = self.find(item)
            groups.setdefault(root, []).append(item)
        return groups

    def connected(self, x: str, y: str) -> bool:
        """Check if *x* and *y* are in the same set."""
        return self.find(x) == self.find(y)
