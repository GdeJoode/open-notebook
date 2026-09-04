"""
Knowledge-graph entity resolver.

Matches extracted entities against an existing knowledge graph using a
three-tier cascade:

1. **Alias table lookup** -- exact match against known surface-form
   aliases (fast path).
2. **Fuzzy matching** -- Levenshtein similarity against type-filtered
   candidates from the KG.
3. **Semantic matching** -- cosine similarity of embedding vectors
   against the same candidates.

On match the entity's ``properties`` dict is enriched with:

* ``kg_entity_id`` -- the ID of the matched KG entity.
* ``kg_match_type`` -- ``"alias"``, ``"fuzzy"``, or ``"semantic"``.
* ``kg_similarity_score`` -- the similarity value that triggered the
  match (1.0 for alias matches).
* ``is_new`` -- ``False``.

When no match is found and *mark_new_entities* is enabled the entity
receives ``is_new = True``.

Optionally, matched surface forms are registered back into the alias
table for future fast-path lookups.

The resolver communicates with the database exclusively through the
``EntityRepositoryProtocol``.  When no repository is provided, all
entities are simply marked as new (no-op mode).
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from loguru import logger
from shared.utils.text_folding import fold_for_comparison

from entity_filtering.resolution.embedding_resolver import (
    EmbeddingResolver,
)


@runtime_checkable
class EntityRepositoryProtocol(Protocol):
    """Protocol for entity resolution repository operations.

    Implementations live in the ``surrealdb-service`` package (or any
    other persistence layer).  The resolver never imports concrete
    repository classes.
    """

    async def find_by_alias(
        self, alias_text: str
    ) -> Optional[Dict[str, Any]]:
        """Look up an entity by exact alias text.

        Returns the matched entity dict or ``None``.
        """
        ...

    async def find_by_type(
        self, entity_type: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve candidate entities of the given type.

        Returns up to *limit* entity dicts for fuzzy/semantic comparison.
        """
        ...

    async def register_alias(
        self,
        canonical_entity_id: str,
        alias_text: str,
        match_type: str,
        similarity_score: float,
        method: str = "",
    ) -> bool:
        """Store a new alias mapping.

        Returns ``True`` on success, ``False`` on failure.
        """
        ...


class KGResolver:
    """Match entities against existing knowledge graph entities.

    Implements a three-tier cascade (alias -> fuzzy -> semantic).  Every
    entity is enriched with resolution metadata; none are removed.

    Args:
        entity_repo: Repository implementing ``EntityRepositoryProtocol``.
            When ``None`` the resolver operates in no-op mode (all entities
            marked as new).
        fuzzy_threshold: Minimum Levenshtein similarity for a fuzzy match.
        semantic_threshold: Minimum cosine similarity for a semantic match.
        max_candidates: Upper bound on KG candidates evaluated per type.
        register_aliases: Whether to write new aliases back to the store.
            Defaults to False here as well as in `KGResolutionConfig`, so a
            caller that constructs the resolver directly cannot get the opposite
            policy from the one the pipeline runs (PC.2). An alias is a
            deliberate act; the curator queue is where one is made.
        mark_new_entities: Whether to set ``is_new=True`` on unmatched
            entities.
        use_alias_table: Whether to attempt alias-table lookup (tier 1).
        centrality_aware: Raise matching thresholds for important KG
            entities (those with high ``weight``).
        centrality_strictness: Amount to add to thresholds for important
            entities.
        importance_threshold: Entity ``weight`` at or above which the
            stricter thresholds apply.
    """

    def __init__(
        self,
        entity_repo: Optional[EntityRepositoryProtocol] = None,
        fuzzy_threshold: float = 0.85,
        semantic_threshold: float = 0.90,
        max_candidates: int = 100,
        register_aliases: bool = False,
        mark_new_entities: bool = True,
        use_alias_table: bool = True,
        centrality_aware: bool = False,
        centrality_strictness: float = 0.05,
        importance_threshold: float = 5.0,
    ) -> None:
        self._repo = entity_repo
        self._fuzzy_threshold = fuzzy_threshold
        self._semantic_threshold = semantic_threshold
        self._max_candidates = max_candidates
        self._register_aliases = register_aliases
        self._mark_new_entities = mark_new_entities
        self._use_alias_table = use_alias_table
        self._centrality_aware = centrality_aware
        self._centrality_strictness = centrality_strictness
        self._importance_threshold = importance_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def resolve(
        self, entities: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Resolve entities against the knowledge graph.

        Returns:
            A tuple of ``(enriched_entities, resolution_report)``.

            ``resolution_report`` contains:

            * ``matched_count`` -- entities matched to existing KG nodes.
            * ``new_count`` -- entities not found in the KG.
            * ``aliases_registered`` -- number of new aliases stored.
            * ``match_type_counts`` -- breakdown by match type
              (``alias``, ``fuzzy``, ``semantic``).
        """
        report: Dict[str, Any] = {
            "matched_count": 0,
            "new_count": 0,
            "aliases_registered": 0,
            "match_type_counts": {
                "alias": 0,
                "fuzzy": 0,
                "semantic": 0,
            },
        }

        if not entities:
            return entities, report

        # No-op mode: no repository available
        if self._repo is None:
            logger.debug(
                "KGResolver: no entity_repo provided, operating in "
                "no-op mode"
            )
            if self._mark_new_entities:
                for entity in entities:
                    entity.setdefault("properties", {})["is_new"] = True
            report["new_count"] = len(entities)
            return entities, report

        # Resolve each entity through the cascade
        for entity in entities:
            try:
                matched = await self._resolve_single(entity, report)
                if not matched and self._mark_new_entities:
                    entity.setdefault("properties", {})["is_new"] = True
                    report["new_count"] += 1
            except Exception:
                logger.warning(
                    "KGResolver: error resolving entity '{}', marking "
                    "as new",
                    entity.get("text", "<unknown>"),
                    exc_info=True,
                )
                if self._mark_new_entities:
                    entity.setdefault("properties", {})["is_new"] = True
                report["new_count"] += 1

        logger.info(
            "KGResolver: {} matched, {} new, {} aliases registered "
            "(alias={}, fuzzy={}, semantic={})",
            report["matched_count"],
            report["new_count"],
            report["aliases_registered"],
            report["match_type_counts"]["alias"],
            report["match_type_counts"]["fuzzy"],
            report["match_type_counts"]["semantic"],
        )

        return entities, report

    # ------------------------------------------------------------------
    # Single-entity resolution cascade
    # ------------------------------------------------------------------

    async def _resolve_single(
        self,
        entity: Dict[str, Any],
        report: Dict[str, Any],
    ) -> bool:
        """Run the three-tier cascade for a single entity.

        Returns ``True`` if a match was found, ``False`` otherwise.
        """
        entity_text = entity.get("text", "").strip()
        if not entity_text:
            return False

        # ---- Tier 1: Alias table lookup ----
        if self._use_alias_table:
            try:
                alias_match = await self._repo.find_by_alias(entity_text)  # type: ignore[union-attr]
                if alias_match is not None:
                    self._enrich_entity(
                        entity,
                        kg_entity_id=alias_match.get("id", ""),
                        match_type="alias",
                        similarity_score=1.0,
                    )
                    report["matched_count"] += 1
                    report["match_type_counts"]["alias"] += 1
                    return True
            except Exception:
                logger.debug(
                    "KGResolver: alias lookup failed for '{}'",
                    entity_text,
                    exc_info=True,
                )

        # ---- Fetch candidates for tier 2 & 3 ----
        entity_type = entity.get("label", "MISC")
        try:
            candidates = await self._repo.find_by_type(  # type: ignore[union-attr]
                entity_type, limit=self._max_candidates
            )
        except Exception:
            logger.debug(
                "KGResolver: candidate fetch failed for type '{}'",
                entity_type,
                exc_info=True,
            )
            return False

        if not candidates:
            return False

        # ---- Tier 2: Fuzzy matching ----
        best_fuzzy_score = 0.0
        best_fuzzy_candidate: Optional[Dict[str, Any]] = None
        normalized_text = self._normalize(entity_text)

        for candidate in candidates:
            # DB candidates use "name"; batch entities use "text"
            candidate_text = candidate.get("name") or candidate.get(
                "text", ""
            )
            if not candidate_text:
                continue
            normalized_candidate = self._normalize(candidate_text)
            score = self._levenshtein_similarity(
                normalized_text, normalized_candidate
            )
            if score > best_fuzzy_score:
                best_fuzzy_score = score
                best_fuzzy_candidate = candidate

        if best_fuzzy_candidate is not None:
            effective_fuzzy = self._effective_threshold(
                self._fuzzy_threshold, best_fuzzy_candidate
            )
            if best_fuzzy_score >= effective_fuzzy:
                candidate_id = best_fuzzy_candidate.get("id", "")
                self._enrich_entity(
                    entity,
                    kg_entity_id=candidate_id,
                    match_type="fuzzy",
                    similarity_score=best_fuzzy_score,
                )
                report["matched_count"] += 1
                report["match_type_counts"]["fuzzy"] += 1
                await self._maybe_register_alias(
                    candidate_id, entity_text, "fuzzy", best_fuzzy_score, report
                )
                return True

        # ---- Tier 3: Semantic matching ----
        entity_embedding = entity.get("properties", {}).get("embedding")
        if entity_embedding and len(entity_embedding) > 0:
            best_sem_score = 0.0
            best_sem_candidate: Optional[Dict[str, Any]] = None

            for candidate in candidates:
                # DB candidates have embedding at top level; batch
                # entities nest it in properties.
                candidate_emb = candidate.get(
                    "embedding"
                ) or candidate.get("properties", {}).get("embedding")
                if not candidate_emb or len(candidate_emb) == 0:
                    continue
                score = EmbeddingResolver._cosine_similarity(
                    entity_embedding, candidate_emb
                )
                if score > best_sem_score:
                    best_sem_score = score
                    best_sem_candidate = candidate

            if best_sem_candidate is not None:
                effective_semantic = self._effective_threshold(
                    self._semantic_threshold, best_sem_candidate
                )
                if best_sem_score >= effective_semantic:
                    candidate_id = best_sem_candidate.get("id", "")
                    self._enrich_entity(
                        entity,
                        kg_entity_id=candidate_id,
                        match_type="semantic",
                        similarity_score=best_sem_score,
                    )
                    report["matched_count"] += 1
                    report["match_type_counts"]["semantic"] += 1
                    await self._maybe_register_alias(
                        candidate_id,
                        entity_text,
                        "semantic",
                        best_sem_score,
                        report,
                    )
                    return True

        return False

    # ------------------------------------------------------------------
    # Entity enrichment
    # ------------------------------------------------------------------

    @staticmethod
    def _enrich_entity(
        entity: Dict[str, Any],
        *,
        kg_entity_id: str,
        match_type: str,
        similarity_score: float,
    ) -> None:
        """Add KG resolution metadata to an entity's properties."""
        props = entity.setdefault("properties", {})
        props["kg_entity_id"] = kg_entity_id
        props["kg_match_type"] = match_type
        props["kg_similarity_score"] = round(similarity_score, 6)
        props["is_new"] = False

    # ------------------------------------------------------------------
    # Centrality-aware threshold adjustment
    # ------------------------------------------------------------------

    def _effective_threshold(
        self, base_threshold: float, candidate: Dict[str, Any]
    ) -> float:
        """Raise the matching threshold for important KG entities.

        When centrality_aware is enabled, candidates with weight >=
        importance_threshold get their threshold raised by
        centrality_strictness, protecting important hub entities
        from borderline matches.
        """
        if not self._centrality_aware:
            return base_threshold
        weight = candidate.get("weight", 0.0)
        if weight >= self._importance_threshold:
            return min(base_threshold + self._centrality_strictness, 1.0)
        return base_threshold

    # ------------------------------------------------------------------
    # Alias registration
    # ------------------------------------------------------------------

    async def _maybe_register_alias(
        self,
        canonical_entity_id: str,
        alias_text: str,
        match_type: str,
        similarity_score: float,
        report: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a new alias if the feature is enabled.

        Returns nothing; the outcome is recorded on ``report`` when one is given.
        ``aliases_registered`` was initialised and logged but never incremented,
        so the INFO line printed `0` for every run — including runs that had
        written aliases. A counter that cannot move is not observability, it is a
        claim the code contradicts (PC.2).
        """
        if not self._register_aliases or self._repo is None:
            return
        try:
            success = await self._repo.register_alias(
                canonical_entity_id=canonical_entity_id,
                alias_text=alias_text,
                match_type=match_type,
                similarity_score=similarity_score,
                method="kg_resolver",
            )
            if success and report is not None:
                report["aliases_registered"] += 1
            if not success:
                logger.debug(
                    "KGResolver: alias registration returned False "
                    "for '{}' -> '{}'",
                    alias_text,
                    canonical_entity_id,
                )
        except Exception:
            logger.debug(
                "KGResolver: failed to register alias '{}' -> '{}'",
                alias_text,
                canonical_entity_id,
                exc_info=True,
            )

    # ------------------------------------------------------------------
    # String utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize text for comparison (lowercase, strip, collapse ws).

        PC.2: one shared fold, so a fifth copy cannot drift from the
        other four. Kept as a thin shim rather than deleted because this method is
        called from several places and the class has its own suite; the guard in
        `tests/test_one_comparison_fold.py` sees through it.
        """
        return fold_for_comparison(text)

    @staticmethod
    def _levenshtein_similarity(s1: str, s2: str) -> float:
        """Calculate Levenshtein similarity (0-1, where 1 = identical).

        Uses a single-row dynamic programming approach for
        O(min(m, n)) space complexity.
        """
        if not s1 or not s2:
            return 0.0
        if s1 == s2:
            return 1.0

        len1, len2 = len(s1), len(s2)
        if len1 > len2:
            s1, s2 = s2, s1
            len1, len2 = len2, len1

        current_row = list(range(len1 + 1))
        for i in range(1, len2 + 1):
            previous_row, current_row = current_row, [i] + [0] * len1
            for j in range(1, len1 + 1):
                add = previous_row[j] + 1
                delete = current_row[j - 1] + 1
                change = previous_row[j - 1]
                if s1[j - 1] != s2[i - 1]:
                    change += 1
                current_row[j] = min(add, delete, change)

        distance = current_row[len1]
        max_len = max(len1, len2)
        return 1 - (distance / max_len) if max_len > 0 else 1.0
