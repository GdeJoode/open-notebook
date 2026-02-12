"""Tests for the UnionFind (Disjoint Set Union) data structure.

Covers lazy initialization, path compression, union by rank,
transitive closure, group extraction, and edge cases.
"""

from entity_filtering.deduplication.union_find import UnionFind


class TestUnionFindBasics:
    def test_find_creates_element_on_first_access(self):
        uf = UnionFind()
        root = uf.find("a")
        assert root == "a"

    def test_find_returns_same_root_for_existing_element(self):
        uf = UnionFind()
        root1 = uf.find("a")
        root2 = uf.find("a")
        assert root1 == root2

    def test_find_different_elements_have_different_roots(self):
        uf = UnionFind()
        assert uf.find("a") != uf.find("b")


class TestUnionFindUnion:
    def test_union_merges_two_separate_sets(self):
        uf = UnionFind()
        uf.union("a", "b")
        assert uf.find("a") == uf.find("b")

    def test_union_is_idempotent(self):
        uf = UnionFind()
        uf.union("a", "b")
        root_after_first = uf.find("a")
        uf.union("a", "b")
        root_after_second = uf.find("a")
        assert root_after_first == root_after_second
        assert uf.find("a") == uf.find("b")

    def test_union_by_rank_keeps_higher_rank_as_root(self):
        """When merging trees of different rank, the higher-rank root wins.

        union("a","b") -> rank 1 at root "a"
        union("c","d") -> rank 1 at root "c"
        union("a","c") -> equal rank, "a" stays root (rank becomes 2)
        union("e","f") -> rank 1 at root "e"

        Now merging "e" into the a-group: rank-2 "a" beats rank-1 "e".
        """
        uf = UnionFind()
        # Build a rank-1 tree
        uf.union("a", "b")
        # Build another rank-1 tree
        uf.union("c", "d")
        # Merge two rank-1 trees -> root "a" becomes rank 2
        uf.union("a", "c")
        root_big = uf.find("a")  # rank-2 root

        # Build a rank-1 tree
        uf.union("e", "f")
        root_small = uf.find("e")  # rank-1 root

        # Merge rank-2 with rank-1: rank-2 root should win
        uf.union("e", "a")
        assert uf.find("e") == root_big
        assert uf.find("f") == root_big
        assert uf.find("e") != root_small


class TestUnionFindPathCompression:
    def test_path_compression_on_long_chain(self):
        """After path compression, find() should give direct parent link."""
        uf = UnionFind()
        # Build a linear chain: 0 -> 1 -> 2 -> ... -> 9
        items = [str(i) for i in range(10)]
        for i in range(len(items) - 1):
            uf.union(items[i], items[i + 1])

        root = uf.find(items[0])
        # After find with path compression, every element should
        # point directly to root.
        for item in items:
            assert uf.find(item) == root


class TestUnionFindConnected:
    def test_connected_returns_true_after_union(self):
        uf = UnionFind()
        uf.union("x", "y")
        assert uf.connected("x", "y") is True

    def test_connected_returns_false_for_unrelated_elements(self):
        uf = UnionFind()
        uf.find("x")
        uf.find("y")
        assert uf.connected("x", "y") is False

    def test_connected_handles_never_seen_elements(self):
        """Calling connected on new elements initializes them as singletons."""
        uf = UnionFind()
        assert uf.connected("new_a", "new_b") is False

    def test_transitive_closure(self):
        """union(A,B) + union(B,C) implies A, B, C are all connected."""
        uf = UnionFind()
        uf.union("A", "B")
        uf.union("B", "C")
        assert uf.connected("A", "C") is True
        assert uf.connected("A", "B") is True
        assert uf.connected("B", "C") is True

    def test_transitive_closure_long_chain(self):
        """union across a chain: 1-2, 2-3, 3-4, 4-5 -> all connected."""
        uf = UnionFind()
        uf.union("1", "2")
        uf.union("2", "3")
        uf.union("3", "4")
        uf.union("4", "5")
        assert uf.connected("1", "5") is True
        assert uf.connected("2", "4") is True


class TestUnionFindGetGroups:
    def test_get_groups_correct_groupings(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        groups = uf.get_groups(["a", "b", "c", "d"])
        # Should have exactly 2 groups
        assert len(groups) == 2
        # a and b should be in the same group
        group_values = list(groups.values())
        group_sets = [set(g) for g in group_values]
        assert {"a", "b"} in group_sets
        assert {"c", "d"} in group_sets

    def test_get_groups_singletons_for_unmerged_items(self):
        uf = UnionFind()
        groups = uf.get_groups(["x", "y", "z"])
        assert len(groups) == 3
        for group_members in groups.values():
            assert len(group_members) == 1

    def test_get_groups_empty_items_list(self):
        uf = UnionFind()
        uf.union("a", "b")
        groups = uf.get_groups([])
        assert groups == {}

    def test_get_groups_subset_of_known_items(self):
        """Only items in the provided list appear, even if more exist internally."""
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        # Only ask for a and c (not b)
        groups = uf.get_groups(["a", "c"])
        assert len(groups) == 1
        group_members = list(groups.values())[0]
        assert set(group_members) == {"a", "c"}

    def test_get_groups_large_group(self):
        """A group with 15 elements should all appear under one root."""
        uf = UnionFind()
        items = [f"item_{i}" for i in range(15)]
        for i in range(len(items) - 1):
            uf.union(items[i], items[i + 1])
        groups = uf.get_groups(items)
        assert len(groups) == 1
        group_members = list(groups.values())[0]
        assert set(group_members) == set(items)

    def test_get_groups_mixed_merged_and_singletons(self):
        uf = UnionFind()
        uf.union("a", "b")
        groups = uf.get_groups(["a", "b", "c"])
        assert len(groups) == 2
        group_values = list(groups.values())
        group_sets = [set(g) for g in group_values]
        assert {"a", "b"} in group_sets
        assert {"c"} in group_sets
