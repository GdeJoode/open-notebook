"""Fuzzy/embedding candidate dedup over the persisted KG (K.5).

K.1/K.2 collapse entities the DETERMINISTIC normalizer can prove equal (article
strip, curated alias). K.3 merges those collisions retroactively. But typos, OCR
noise, and near-duplicate phrasings escape a deterministic rule — ``Koninkrijks``
**relaties** vs an OCR-mangled ``Koninkrijks``**reiaties** never normalize equal,
yet they are the same entity. This service catches that class by running the
entity-filtering pipeline's existing fuzzy + embedding resolvers over the
persisted entities and PROPOSING merges — it never silently writes.

The over-merge guards (this is the ×2.0 risk surface):

* **Type-aware** — entities are bucketed by ``entity_type`` and only compared
  WITHIN a bucket. A person and an org with similar names are never proposed
  (the recurring Track-K lesson; mirrors K.3's ``(name, type)`` discipline).
* **Review band, nothing silent** — every pair is partitioned into
  ``auto_merge`` (score ≥ auto-threshold), ``review``
  (review-threshold ≤ score < auto-threshold → queued for a human, NEVER
  auto-applied), and ``reject`` (below review-threshold, dropped).
* **force-split overlay = hard veto** — an ``alias_overlay`` split rule removes a
  pair from EVERY band even if its similarity is 1.0 (the ultimate backstop).
* **force-merge overlay = hard include** — a merge rule injects a pair as an
  ``auto_merge`` candidate regardless of similarity, scoped per the rule (a
  notebook merge rule only fires within that notebook).

``propose_candidates`` is read-only. Applying an ``auto_merge`` candidate is done
through K.3's reviewable ``RecanonicalizationService.apply_merge`` (reused via
:meth:`to_merge_cluster`), so the destructive path keeps the K.3 idempotency /
relation-repointing guarantees. Embedding dedup degrades to fuzzy-only when an
entity has no vector (``embedding=[]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from entity_filtering.config import EmbeddingDedupConfig, FuzzyDedupConfig
from entity_filtering.deduplication.fuzzy_resolver import FuzzyResolver
from loguru import logger
from surrealdb_service.repositories.entity import EntityRepository

from app_main.services.entity_resolution.overlay_service import OverlayService
from app_main.services.entity_resolution.recanonicalization_service import (
    MergeCluster,
)

# Band labels.
AUTO_MERGE = "auto_merge"
REVIEW = "review"

# Forced (overlay) candidates report a sentinel score so the UI can flag them as
# a user rule rather than a matcher score.
FORCED_SCORE = 1.0


@dataclass
class MergeCandidate:
    """One proposed merge between two same-typed persisted entities.

    A candidate is a PROPOSAL — nothing is written by producing it. ``band``
    decides downstream handling: ``auto_merge`` can be applied (via K.3),
    ``review`` is queued for a human and never auto-applied.
    """

    id_a: str
    id_b: str
    name_a: str
    name_b: str
    entity_type: str
    score: float
    band: str  # AUTO_MERGE | REVIEW
    method: str  # "fuzzy" | "embedding" | "overlay"
    # The winner/loser split for the apply path (winner = higher confidence,
    # tie-break id for stability). Populated by the service.
    winner_id: str = ""
    loser_id: str = ""

    def to_merge_cluster(self) -> MergeCluster:
        """Project an auto-merge candidate onto a K.3 ``MergeCluster``.

        Reuses the K.3 reviewable-merge machinery for the destructive apply
        (relation repointing, provenance fold, alias rows, soft status) — this
        service never re-implements the merge.
        """
        return MergeCluster(
            new_canonical=self.name_a,
            entity_type=self.entity_type,
            winner_id=self.winner_id or self.id_a,
            loser_ids=[self.loser_id or self.id_b],
            member_surface_forms=[self.name_a, self.name_b],
            total_source_docs=0,
        )


@dataclass
class CandidateReport:
    """The output of a proposal run: bands + counts."""

    auto_merge: List[MergeCandidate] = field(default_factory=list)
    review: List[MergeCandidate] = field(default_factory=list)
    scope: str = "global"
    total_active_entities: int = 0

    @property
    def auto_merge_count(self) -> int:
        return len(self.auto_merge)

    @property
    def review_count(self) -> int:
        return len(self.review)


class CandidateDedupService:
    """Propose fuzzy/embedding merges with strong over-merge guards (K.5)."""

    def __init__(
        self,
        entity_repo: Optional[EntityRepository] = None,
        overlay_service: Optional[OverlayService] = None,
        fuzzy_config: Optional[FuzzyDedupConfig] = None,
        embedding_config: Optional[EmbeddingDedupConfig] = None,
    ) -> None:
        self.entity_repo = entity_repo or EntityRepository()
        self.overlay_service = overlay_service or OverlayService()
        self.fuzzy_config = fuzzy_config or FuzzyDedupConfig(enabled=True)
        self.embedding_config = embedding_config or EmbeddingDedupConfig(
            enabled=True
        )
        # Reuse the entity-filtering resolver's scoring (do NOT reimplement
        # matching) — we drive its pairwise similarity, not its union-find merge.
        self._fuzzy = FuzzyResolver(
            algorithm=self.fuzzy_config.algorithm,
            similarity_threshold=self.fuzzy_config.similarity_threshold,
            phonetic_algorithm=self.fuzzy_config.phonetic_algorithm,
            phonetic_weight=self.fuzzy_config.phonetic_weight,
            max_candidates_per_entity=self.fuzzy_config.max_candidates_per_entity,
        )

    # ------------------------------------------------------------------
    # Threshold resolution (review-band split)
    # ------------------------------------------------------------------

    def _fuzzy_bands(self) -> Tuple[float, float]:
        """(review_floor, auto_threshold) for fuzzy, with safe fallbacks."""
        auto = (
            self.fuzzy_config.auto_merge_threshold
            if self.fuzzy_config.auto_merge_threshold is not None
            else self.fuzzy_config.similarity_threshold
        )
        review = (
            self.fuzzy_config.review_threshold
            if self.fuzzy_config.review_threshold is not None
            else self.fuzzy_config.similarity_threshold
        )
        return min(review, auto), auto

    def _embedding_bands(self) -> Tuple[float, float]:
        """(review_floor, auto_threshold) for embeddings, with safe fallbacks."""
        auto = (
            self.embedding_config.auto_merge_threshold
            if self.embedding_config.auto_merge_threshold is not None
            else self.embedding_config.similarity_threshold
        )
        review = (
            self.embedding_config.review_threshold
            if self.embedding_config.review_threshold is not None
            else self.embedding_config.similarity_threshold
        )
        return min(review, auto), auto

    @staticmethod
    def _band(score: float, review_floor: float, auto: float) -> Optional[str]:
        """Map a similarity score to a band, or ``None`` to reject."""
        if score >= auto:
            return AUTO_MERGE
        if score >= review_floor:
            return REVIEW
        return None

    # ------------------------------------------------------------------
    # Propose
    # ------------------------------------------------------------------

    async def propose_candidates(
        self, notebook_id: Optional[str] = None
    ) -> CandidateReport:
        """Propose merges over the active KG. **Read-only — zero writes.**

        Args:
            notebook_id: scope the proposals to one notebook's entities; the
                overlay rules resolved are global + that notebook's. ``None`` =
                global (global overlay rules only).

        Returns:
            A :class:`CandidateReport` with the ``auto_merge`` / ``review`` bands.
            Force-split pairs appear in NEITHER band (hard veto). Force-merge
            pairs always land in ``auto_merge``.
        """
        source_ids: Optional[List[str]] = None
        if notebook_id is not None:
            source_ids = await self._sources_for_notebook(notebook_id)
            if not source_ids:
                return CandidateReport(
                    scope=notebook_id, total_active_entities=0
                )

        rows = await self.entity_repo.list_active_entities_with_embeddings(
            source_ids=source_ids
        )

        # Overlay rules in effect for this scope.
        split_pairs = await self.overlay_service.split_pairs(
            notebook_id=notebook_id
        )
        merge_rules = await self.overlay_service.merge_rules(
            notebook_id=notebook_id
        )

        # Index entities by id and bucket by entity_type (type-aware guard).
        by_id: Dict[str, Dict[str, Any]] = {}
        buckets: Dict[str, List[Dict[str, Any]]] = {}
        for row in rows:
            rid = str(row.get("id"))
            name = str(row.get("canonical_name") or "")
            if not name:
                continue
            etype = str(row.get("entity_type") or "")
            rec = {
                "id": rid,
                "name": name,
                "entity_type": etype,
                "confidence": float(row.get("confidence") or 0.0),
                "embedding": list(row.get("embedding") or []),
            }
            by_id[rid] = rec
            buckets.setdefault(etype, []).append(rec)

        # Pair -> best (score, method). Dedup so a pair caught by both fuzzy and
        # embedding reports once, at its highest score.
        best: Dict[Tuple[str, str], Tuple[float, str]] = {}

        f_review, f_auto = self._fuzzy_bands()
        e_review, e_auto = self._embedding_bands()

        for etype, group in buckets.items():
            self._score_fuzzy(group, best)
            self._score_embedding(group, best)

        # Build candidates, applying the force-split veto.
        report = CandidateReport(
            scope=notebook_id or "global",
            total_active_entities=len(rows),
        )
        seen_pairs: set[Tuple[str, str]] = set()

        for (id_a, id_b), (score, method) in best.items():
            rec_a = by_id[id_a]
            rec_b = by_id[id_b]
            norm_pair = frozenset({rec_a["name"], rec_b["name"]})
            # Hard veto: a force-split pair is never proposed, any score.
            if self._vetoed(norm_pair, rec_a["name"], rec_b["name"], split_pairs):
                continue

            review_floor = f_review if method == "fuzzy" else e_review
            auto = f_auto if method == "fuzzy" else e_auto
            band = self._band(score, review_floor, auto)
            if band is None:
                continue

            candidate = self._make_candidate(
                rec_a, rec_b, score, band, method
            )
            seen_pairs.add(self._pair_key(id_a, id_b))
            if band == AUTO_MERGE:
                report.auto_merge.append(candidate)
            else:
                report.review.append(candidate)

        # Force-merge overlays: inject as auto-merge (hard include), unless a
        # split rule contradicts it (split always wins — a self-inconsistent
        # overlay set should not over-merge).
        self._apply_force_merge(
            merge_rules, by_id, split_pairs, seen_pairs, report
        )

        logger.info(
            "propose_candidates: scope={s} active={n} auto={a} review={r}",
            s=notebook_id or "global",
            n=len(rows),
            a=report.auto_merge_count,
            r=report.review_count,
        )
        return report

    # ------------------------------------------------------------------
    # Scoring (reuse the entity-filtering resolvers)
    # ------------------------------------------------------------------

    def _score_fuzzy(
        self,
        group: List[Dict[str, Any]],
        best: Dict[Tuple[str, str], Tuple[float, str]],
    ) -> None:
        """Score every same-type pair with the fuzzy resolver's similarity."""
        n = len(group)
        for i in range(n):
            norm_i = self._fuzzy._normalize(group[i]["name"])
            if not norm_i:
                continue
            for j in range(i + 1, n):
                norm_j = self._fuzzy._normalize(group[j]["name"])
                if not norm_j:
                    continue
                if norm_i == norm_j:
                    # Deterministic-equal — already K.1/K.3 territory, skip.
                    continue
                score = self._fuzzy._compute_similarity(norm_i, norm_j)
                self._record(best, group[i]["id"], group[j]["id"], score, "fuzzy")

    def _score_embedding(
        self,
        group: List[Dict[str, Any]],
        best: Dict[Tuple[str, str], Tuple[float, str]],
    ) -> None:
        """Score same-type pairs by cosine similarity of their embeddings.

        Falls back silently (skips a pair) when either entity has no vector —
        embedding dedup degrades to fuzzy-only rather than fabricating a score.
        """
        with_emb = [e for e in group if e["embedding"]]
        if len(with_emb) < 2:
            return
        try:
            import numpy as np
        except ImportError:  # pragma: no cover - numpy is a hard dep in app
            return

        n = len(with_emb)
        for i in range(n):
            vi = np.asarray(with_emb[i]["embedding"], dtype="float64")
            ni = float(np.linalg.norm(vi))
            if ni == 0.0:
                continue
            for j in range(i + 1, n):
                vj = np.asarray(with_emb[j]["embedding"], dtype="float64")
                nj = float(np.linalg.norm(vj))
                if nj == 0.0 or vi.shape != vj.shape:
                    continue
                score = float(np.dot(vi, vj) / (ni * nj))
                self._record(
                    best, with_emb[i]["id"], with_emb[j]["id"], score, "embedding"
                )

    @staticmethod
    def _pair_key(id_a: str, id_b: str) -> Tuple[str, str]:
        """Order-insensitive pair key."""
        return (id_a, id_b) if id_a <= id_b else (id_b, id_a)

    def _record(
        self,
        best: Dict[Tuple[str, str], Tuple[float, str]],
        id_a: str,
        id_b: str,
        score: float,
        method: str,
    ) -> None:
        """Keep the highest score per pair (method follows the winning score)."""
        key = self._pair_key(id_a, id_b)
        prior = best.get(key)
        if prior is None or score > prior[0]:
            best[key] = (score, method)

    # ------------------------------------------------------------------
    # Overlay application
    # ------------------------------------------------------------------

    @staticmethod
    def _vetoed(
        norm_pair: "frozenset[str]",
        name_a: str,
        name_b: str,
        split_pairs: "set[frozenset[str]]",
    ) -> bool:
        """True when a force-split rule covers this pair (hard veto)."""
        from shared.utils.name_normalizer import normalize_entity_name

        normalized = frozenset(
            {normalize_entity_name(name_a), normalize_entity_name(name_b)}
        )
        return normalized in split_pairs or norm_pair in split_pairs

    def _make_candidate(
        self,
        rec_a: Dict[str, Any],
        rec_b: Dict[str, Any],
        score: float,
        band: str,
        method: str,
    ) -> MergeCandidate:
        """Assemble a candidate, fixing winner/loser by confidence then id."""
        if (rec_a["confidence"], rec_b["id"]) >= (
            rec_b["confidence"],
            rec_a["id"],
        ):
            winner, loser = rec_a, rec_b
        else:
            winner, loser = rec_b, rec_a
        return MergeCandidate(
            id_a=rec_a["id"],
            id_b=rec_b["id"],
            name_a=rec_a["name"],
            name_b=rec_b["name"],
            entity_type=rec_a["entity_type"],
            score=round(score, 4),
            band=band,
            method=method,
            winner_id=winner["id"],
            loser_id=loser["id"],
        )

    def _apply_force_merge(
        self,
        merge_rules: List[Any],
        by_id: Dict[str, Dict[str, Any]],
        split_pairs: "set[frozenset[str]]",
        seen_pairs: "set[Tuple[str, str]]",
        report: CandidateReport,
    ) -> None:
        """Inject force-merge rule pairs as auto-merge candidates.

        A rule names two surface forms (and a type). We resolve them to the
        matching active entities IN THE SCOPE (type-aware), and add an
        auto-merge candidate for the realized id pair if not already proposed.
        A contradicting force-split rule still wins (no inject).
        """
        from shared.utils.name_normalizer import normalize_entity_name

        # Build a (normalized_name, type) -> [entity rec] index for resolution.
        index: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for rec in by_id.values():
            key = (normalize_entity_name(rec["name"]), rec["entity_type"])
            index.setdefault(key, []).append(rec)

        for rule in merge_rules:
            norm_a = normalize_entity_name(rule.name_a)
            norm_b = normalize_entity_name(rule.name_b)
            if norm_a == norm_b:
                continue
            if frozenset({norm_a, norm_b}) in split_pairs:
                # A split rule on the same pair vetoes a contradictory merge.
                continue
            etype = rule.entity_type
            # When the rule carries a type, only that bucket resolves; otherwise
            # try every type but keep both endpoints in the SAME type (no
            # cross-type force-merge — the type guard holds even for overrides).
            type_candidates = (
                [etype]
                if etype
                else sorted({t for (_, t) in index.keys()})
            )
            for t in type_candidates:
                recs_a = index.get((norm_a, t), [])
                recs_b = index.get((norm_b, t), [])
                for rec_a in recs_a:
                    for rec_b in recs_b:
                        if rec_a["id"] == rec_b["id"]:
                            continue
                        key = self._pair_key(rec_a["id"], rec_b["id"])
                        if key in seen_pairs:
                            continue
                        seen_pairs.add(key)
                        report.auto_merge.append(
                            self._make_candidate(
                                rec_a, rec_b, FORCED_SCORE, AUTO_MERGE, "overlay"
                            )
                        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _sources_for_notebook(self, notebook_id: str) -> List[str]:
        """Resolve source record ids linked to a notebook (mirrors K.3)."""
        from surrealdb_service.connection import execute_query

        try:
            rows = await execute_query(
                "SELECT VALUE in FROM reference "
                "WHERE out = type::thing($notebook_id)",
                {"notebook_id": notebook_id},
            )
        except Exception as e:
            logger.error(
                f"_sources_for_notebook failed for '{notebook_id}': {e}"
            )
            return []
        return [str(r) for r in (rows or []) if r]


__all__ = [
    "CandidateDedupService",
    "CandidateReport",
    "MergeCandidate",
    "AUTO_MERGE",
    "REVIEW",
]
