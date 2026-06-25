"""Unit tests for the pure mean-pool helper (Track R.0)."""

from __future__ import annotations

import pytest

from shared.utils.vectors import mean_pool


def test_mean_pool_basic():
    result = mean_pool([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    assert result == [2.0, 3.0, 4.0]


def test_mean_pool_single_vector_is_identity():
    assert mean_pool([[0.5, -1.0, 2.0]]) == [0.5, -1.0, 2.0]


def test_mean_pool_empty_returns_none():
    assert mean_pool([]) is None


def test_mean_pool_all_empty_vectors_returns_none():
    assert mean_pool([[], []]) is None


def test_mean_pool_skips_mismatched_dimension():
    # First vector pins dim=3; the 2-d vector is skipped, not crashed on.
    result = mean_pool([[1.0, 1.0, 1.0], [9.0, 9.0]])
    assert result == [1.0, 1.0, 1.0]


def test_mean_pool_preserves_dimension():
    dim = 1024
    vecs = [[float(i)] * dim for i in (2, 4, 6)]
    pooled = mean_pool(vecs)
    assert pooled is not None
    assert len(pooled) == dim
    assert pooled == pytest.approx([4.0] * dim)
