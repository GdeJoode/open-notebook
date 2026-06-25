"""Small, dependency-free vector helpers.

Kept pure (stdlib only, no numpy) so every workspace member — the embeddings
pipeline, the app, and the R.0 backfill script — can mean-pool chunk vectors
into a source-level aggregate without taking a heavy dependency.
"""

from __future__ import annotations

from typing import List, Optional, Sequence


def mean_pool(vectors: Sequence[Sequence[float]]) -> Optional[List[float]]:
    """Element-wise mean of equal-length vectors.

    Returns ``None`` for an empty input (the honest "no aggregate" state R.0's
    ``source.embedding`` stores as NONE). Vectors whose length differs from the
    first vector's length are skipped — a defensive guard against a mixed-model
    corpus silently averaging incompatible dimensions into garbage; if every
    vector is skipped the result is ``None``.

    Args:
        vectors: Sequence of equal-length float vectors (e.g. chunk embeddings).

    Returns:
        The element-wise mean vector, or ``None`` when there is nothing poolable.
    """
    usable: List[Sequence[float]] = []
    expected_dim: Optional[int] = None
    for vec in vectors:
        if not vec:
            continue
        if expected_dim is None:
            expected_dim = len(vec)
        if len(vec) != expected_dim:
            # Skip mismatched-dimension vectors rather than corrupt the pool.
            continue
        usable.append(vec)

    if not usable or expected_dim is None:
        return None

    n = len(usable)
    sums = [0.0] * expected_dim
    for vec in usable:
        for i in range(expected_dim):
            sums[i] += float(vec[i])
    return [s / n for s in sums]
