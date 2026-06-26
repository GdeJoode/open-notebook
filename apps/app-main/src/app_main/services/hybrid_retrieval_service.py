"""Hybrid source-retrieval service (Track R.3).

Fuses the two source↔source retrieval signals into ONE ranking:

- the **dense** signal (R.1): ``SourceService.find_related`` — cosine kNN over
  aggregate source embeddings, and
- the **KG** signal (R.2): ``KGRetrievalService.find_related_by_kg`` — shared
  active canonical entities weighted by type salience × IDF rarity,

via the pure RRF combiner :func:`shared.retrieval.fuse_rankings`. The fusion is
**KG-prominent by default** (the locked Track R steer — "the KG must play a
large role in search"): the default preset is ``kg-heavy`` (KG weight 3× dense).

This service is the only place that does I/O for the hybrid path; the ranking
math lives in the pure fusion module so it stays unit-testable and ablatable
offline. The service fetches BOTH signals over-deep (a candidate pool larger
than ``k``) so the fusion sees enough of each signal's tail to reorder properly
before truncating to ``k`` — fusing two already-truncated top-``k`` lists would
hide cross-signal disagreement past the cutoff.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from shared.retrieval import (
    FusionWeights,
    fuse_rankings,
    get_preset,
)

from app_main.services.kg_retrieval_service import KGRetrievalService
from app_main.services.source_service import SourceService

# How deep to pull each signal before fusing, relative to the requested k. A
# candidate from one signal can be reordered by the other signal only if it is
# present in the fetched pool — so we pull a wider pool than k from each signal,
# fuse, then truncate. Bounded to keep the two DB round-trips cheap.
_POOL_MULTIPLIER = 4
_MIN_POOL = 25
_MAX_POOL = 100


def _pool_size(k: int) -> int:
    """Per-signal candidate-pool depth for a requested ``k`` (bounded)."""
    return max(_MIN_POOL, min(int(k) * _POOL_MULTIPLIER, _MAX_POOL))


class HybridRetrievalService:
    """Rank sources related to a given source by fused dense + KG signals (R.3)."""

    def __init__(
        self,
        source_service: SourceService,
        kg_service: KGRetrievalService,
    ):
        self.source_service = source_service
        self.kg_service = kg_service

    async def find_related_hybrid(
        self,
        source_id: str,
        k: int,
        *,
        weights: Optional[FusionWeights] = None,
        preset: Optional[str] = None,
        expand_relations: bool = False,
    ) -> List[Dict[str, Any]]:
        """Return the top-``k`` related sources, fused from dense + KG (R.3).

        Args:
            source_id: The source to find related sources for.
            k: Max number of fused results to return.
            weights: Explicit RRF weights, overriding ``preset`` when given.
                ``None`` resolves ``preset`` (default ``kg-heavy``).
            preset: Named weight preset (``kg-heavy`` | ``balanced``). Ignored
                when ``weights`` is passed. Unknown/empty -> the KG-heavy default.
            expand_relations: Forwarded to the KG signal's opt-in 1-hop relation
                expansion (OFF by default; runs through R.6 noise when on).

        Returns:
            A list of fused result dicts ranked by ``fused_score`` descending,
            each carrying per-signal provenance::

                {
                  "id", "title", "fused_score",
                  "dense": {"present", "score", "rank", "contribution"},
                  "kg":    {"present", "score", "rank", "contribution"},
                  "kg_entities": [ ...R.2 driving-entities lineage... ],
                }

            A source present in only ONE signal still appears (never dropped).
            Empty when neither signal returns anything.
        """
        resolved = weights if weights is not None else get_preset(preset)

        pool = _pool_size(k)

        # Pull both signals over-deep so the fusion can reorder across the tail.
        # The KG signal is given the TRUE corpus source count for its IDF (the
        # KG service fetches it via source_repo.count() when n_sources is None).
        dense_rows = await self.source_service.find_related(source_id, pool)
        kg_rows = await self.kg_service.find_related_by_kg(
            source_id, pool, expand_relations=expand_relations
        )

        dense_signal = [
            (str(r["id"]), float(r["score"])) for r in dense_rows
        ]
        kg_signal = [
            (
                str(r["id"]),
                float(r["score"]),
                list(r.get("explanation") or []),
            )
            for r in kg_rows
        ]

        # Hydrate titles from whichever signal carried them (KG hydrates; dense
        # also returns title). Build one map, dense first so KG can fill gaps.
        titles: Dict[str, Optional[str]] = {}
        for r in dense_rows:
            titles[str(r["id"])] = r.get("title")
        for r in kg_rows:
            titles.setdefault(str(r["id"]), r.get("title"))

        fused = fuse_rankings(
            dense_signal,
            kg_signal,
            weights=resolved,
            titles=titles,
            limit=k,
        )

        results = [
            {
                "id": f.source_id,
                "title": f.title,
                "fused_score": f.fused_score,
                "dense": {
                    "present": f.dense.present,
                    "score": f.dense.score,
                    "rank": f.dense.rank,
                    "contribution": f.dense.contribution,
                },
                "kg": {
                    "present": f.kg.present,
                    "score": f.kg.score,
                    "rank": f.kg.rank,
                    "contribution": f.kg.contribution,
                },
                "kg_entities": f.kg_entities,
            }
            for f in fused
        ]
        logger.debug(
            "find_related_hybrid: source={src} k={k} weights={w} -> {n} results",
            src=source_id,
            k=k,
            w=resolved,
            n=len(results),
        )
        return results
