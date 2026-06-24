"""Triage signals: effective structural degree, cross-doc recurrence, affected set (Q.2).

This service turns the merged KG (Q.2a gave each ``(in, out, relation_type)``
edge a cumulative ``source_documents`` union) into the three signals the status
layer (Q.3) and queue/backbone layers (Q.4) consume:

* **Effective structural degree** — count of an entity's ``structural`` edges
  plus its PROMOTED weak edges. A weak (``RELATED``) edge promotes only once the
  SAME edge has recurred across ``weak_promotion_min_docs`` documents (its
  per-edge ``source_documents`` count). A single-doc ``RELATED`` does NOT count
  toward connectivity (and is backbone-flagged in Q.4); a multi-doc ``RELATED``
  DOES — that is the operator's promotion guarantee.
* **Cross-doc recurrence** — the distinct document count from an entity's own
  ``source_documents`` provenance bag.
* **Affected set** — new nodes plus nodes whose effective-degree or doc-count
  changed this batch, computed against a snapshot captured BEFORE the merge.
  Only the affected set is re-triaged downstream, keeping re-runs idempotent.

Predicate strength and the promotion threshold are sourced exclusively from the
Q.1 triage config (``cfg.predicate_strength`` / ``cfg.weak_promotion_min_docs``);
nothing here hardcodes the predicate lists or the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Set

from loguru import logger

from app_main.services.triage.triage_config import TriageConfig, load_triage_config

# Degree bucket the UI/queue render. Bucketing collapses raw degree into the
# three bands the status rules care about: isolated (0), lightly connected
# (1-2), and well-connected (>= well_connected_threshold, default 3).
DegreeBucket = Literal["0", "1-2", "3+"]

# Snapshot shape captured by the caller BEFORE the merge (per id).
#   {entity_id: {"degree": int, "doc_count": int}}
PreBatchSnapshot = Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class EntitySignals:
    """Per-entity triage signals for one batch.

    ``is_new`` is True when the entity was absent from the pre-batch snapshot.
    ``degree_changed`` / ``doc_count_changed`` are True when the post-batch
    value differs from the snapshot (always True for a new entity, since it has
    no prior value).
    """

    entity_id: str
    effective_degree: int
    bucket: DegreeBucket
    doc_count: int
    is_new: bool
    degree_changed: bool
    doc_count_changed: bool

    @property
    def changed(self) -> bool:
        """True when the entity is new OR any tracked signal changed."""
        return self.is_new or self.degree_changed or self.doc_count_changed


@dataclass(frozen=True)
class SignalsReport:
    """Signals for a whole batch plus the derived affected set.

    ``by_id`` carries one :class:`EntitySignals` per batch entity; ``affected``
    is exactly the ids whose ``changed`` flag is set (new + changed-degree +
    changed-doc-count). Downstream re-triages only ``affected``.
    """

    by_id: Mapping[str, EntitySignals]
    affected: Set[str] = field(default_factory=set)


def _doc_count_from_sources(source_documents: Optional[Iterable[Any]]) -> int:
    """Distinct-document count from a ``source_documents`` provenance bag.

    Q.2a unions provenance per entity/edge, but this de-dups defensively (stable
    via stringification) so a snapshot row that was NOT built through the union
    path still yields a correct distinct count.
    """
    if not source_documents:
        return 0
    return len({str(d) for d in source_documents if d is not None})


class TriageSignalsService:
    """Compute degree / recurrence / affected-set signals for a triage batch.

    Degree is read from the persisted ``relation`` table via the batched
    :meth:`EntityRepository.effective_degree_for_entities` helper (ONE round-trip
    for the affected set). Doc-count is read off each entity's own
    ``source_documents``. The structural/weak predicate split and the promotion
    threshold come from the injected (or default-loaded) Q.1 triage config.
    """

    def __init__(
        self,
        entity_repository: Any,
        config: Optional[TriageConfig] = None,
    ) -> None:
        """Wire the repository and resolve the triage config.

        Args:
            entity_repository: An ``EntityRepository`` (the relation-degree seam).
            config: An already-loaded :class:`TriageConfig`. Defaults to the
                cached committed config via :func:`load_triage_config`.
        """
        self._repo = entity_repository
        self._cfg = config if config is not None else load_triage_config()
        # Materialise the configured predicate lists once from the frozen reverse
        # map so the batched query can pass them as ``INSIDE`` sets. The
        # STRUCTURAL list is the gating set: the query counts structural edges
        # unconditionally and any non-structural edge only once promoted, so an
        # unmapped predicate (config-default weak) promotes exactly like an
        # explicit ``RELATED``. The weak list is passed through for interface
        # symmetry / future weak-only signals.
        self._structural_predicates = sorted(
            p
            for p, strength in self._cfg._strength_by_predicate.items()
            if strength == "structural"
        )
        self._weak_predicates = sorted(
            p
            for p, strength in self._cfg._strength_by_predicate.items()
            if strength == "weak"
        )

    # ------------------------------------------------------------------
    # Pure helpers
    # ------------------------------------------------------------------

    def degree_bucket(self, degree: int) -> DegreeBucket:
        """Bucket an effective degree into ``"0"`` / ``"1-2"`` / ``"3+"``.

        ``3+`` is keyed on the config's ``well_connected_threshold`` (default 3)
        so the well-connected band tracks the status rules rather than a magic
        number.
        """
        if degree <= 0:
            return "0"
        if degree < self._cfg.well_connected_threshold:
            return "1-2"
        return "3+"

    def doc_count(self, entity: Any) -> int:
        """Distinct-document count for ``entity``.

        Accepts an ``Entity`` model, a raw row dict, or anything exposing
        ``source_documents`` (attribute or key).
        """
        sources: Optional[Iterable[Any]]
        if isinstance(entity, Mapping):
            sources = entity.get("source_documents")
        else:
            sources = getattr(entity, "source_documents", None)
        return _doc_count_from_sources(sources)

    # ------------------------------------------------------------------
    # DB-backed degree
    # ------------------------------------------------------------------

    async def effective_degree(self, entity_id: str) -> int:
        """Effective structural degree for a single entity (convenience).

        Delegates to the batched repo helper with a one-element batch. Prefer
        :meth:`compute_report` for a real batch — it makes ONE call for the whole
        affected set rather than one per node.
        """
        degrees = await self._repo.effective_degree_for_entities(
            [entity_id],
            self._structural_predicates,
            self._weak_predicates,
            self._cfg.weak_promotion_min_docs,
        )
        return int(degrees.get(entity_id, 0))

    async def _batch_effective_degree(self, ids: List[str]) -> Dict[str, int]:
        """ONE batched degree call for ``ids`` (the affected-set query)."""
        if not ids:
            return {}
        degrees = await self._repo.effective_degree_for_entities(
            ids,
            self._structural_predicates,
            self._weak_predicates,
            self._cfg.weak_promotion_min_docs,
        )
        return {i: int(degrees.get(i, 0)) for i in ids}

    # ------------------------------------------------------------------
    # Affected set / report
    # ------------------------------------------------------------------

    def compute_affected_set(
        self,
        batch_entity_ids: Iterable[str],
        pre_batch_snapshot: PreBatchSnapshot,
        post_batch_signals: Mapping[str, EntitySignals],
    ) -> Set[str]:
        """The affected set = new + changed-effective-degree + changed-doc-count.

        **Snapshot contract (load-bearing for idempotency):** ``pre_batch_snapshot``
        MUST be captured by the caller BEFORE the merge (Q.2a) runs, keyed by
        entity id with ``{"degree": int, "doc_count": int}`` values. Capturing it
        AFTER the merge would make every node look "unchanged" (degree/doc-count
        already at their post-merge values) and break change detection. Ids absent
        from the snapshot are treated as NEW (no prior value to compare).

        A weak edge crossing the promotion threshold this batch bumps its
        endpoints' effective degree, so those endpoints land in the affected set
        even if their own ``source_documents`` did not grow.

        Args:
            batch_entity_ids: Every entity touched this batch.
            pre_batch_snapshot: Pre-merge degree/doc-count per id.
            post_batch_signals: Post-merge :class:`EntitySignals` per id
                (from :meth:`compute_report`).

        Returns:
            The set of changed/new ids. Unchanged ids are excluded.
        """
        affected: Set[str] = set()
        for entity_id in batch_entity_ids:
            sig = post_batch_signals.get(entity_id)
            if sig is None:
                # No post-batch signal computed (e.g. id dropped) → skip.
                continue
            if sig.changed:
                affected.add(entity_id)
        return affected

    async def compute_report(
        self,
        batch_entities: Mapping[str, Any],
        pre_batch_snapshot: PreBatchSnapshot,
    ) -> SignalsReport:
        """Compute per-id signals + the affected set for a batch.

        Args:
            batch_entities: ``{entity_id: entity}`` for every entity touched this
                batch, where ``entity`` is an ``Entity`` / row exposing
                ``source_documents`` (for doc-count). The keys drive the single
                batched degree query.
            pre_batch_snapshot: Pre-merge ``{id: {"degree", "doc_count"}}`` per
                the snapshot contract on :meth:`compute_affected_set`.

        Returns:
            A :class:`SignalsReport` with one :class:`EntitySignals` per id and
            the derived affected set. Degree is computed in ONE batched DB call.
        """
        ids = list(batch_entities.keys())
        degrees = await self._batch_effective_degree(ids)

        by_id: Dict[str, EntitySignals] = {}
        for entity_id in ids:
            degree = degrees.get(entity_id, 0)
            doc_count = self.doc_count(batch_entities[entity_id])
            prior = pre_batch_snapshot.get(entity_id)
            is_new = prior is None
            if prior is None:
                degree_changed = True
                doc_count_changed = True
            else:
                degree_changed = int(prior.get("degree", 0)) != degree
                doc_count_changed = int(prior.get("doc_count", 0)) != doc_count
            by_id[entity_id] = EntitySignals(
                entity_id=entity_id,
                effective_degree=degree,
                bucket=self.degree_bucket(degree),
                doc_count=doc_count,
                is_new=is_new,
                degree_changed=degree_changed,
                doc_count_changed=doc_count_changed,
            )

        affected = self.compute_affected_set(ids, pre_batch_snapshot, by_id)
        logger.debug(
            "TriageSignals: batch={n} affected={a} (new/changed-degree/changed-docs)",
            n=len(ids),
            a=len(affected),
        )
        return SignalsReport(by_id=by_id, affected=affected)
