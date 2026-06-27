"""Pure `mentions` (source → entity) edge projection (Track U.2).

This module turns the canonical ``entity.source_documents`` provenance bags into
a **document↔entity bipartite edge list** — the materialized ``mentions`` graph
the KG visualization / export / traversal layer draws on. It is deliberately
PURE (no I/O): given the active entity rows and the R.6-normalized concept set it
returns a list of ``(source_id, entity_id, weight)`` edges, so it is unit-testable
in isolation and the regenerator service (U.2 repo method) just persists what this
returns.

Why this is a *projection*, not new data
========================================
``mentions`` adds nothing to search — R.2 already computes the same shared-entity
signal on the fly. Its value is making the document↔entity graph **real**
(traversable / drawable / exportable). So the edge weight here is the SAME
per-entity weight R.2 uses (``entity_weight`` = type-salience × IDF rarity) and
the entities kept are the SAME ones R.6 keeps (drop df==1 singletons, unify
case/type duplicates, down-weight generics) — the drawn graph therefore matches
the search signal exactly, by construction (no second weighting model).

The bipartite concept→entity mapping
====================================
R.6 collapses raw entity rows into search-facing *concepts*
(``concept::<name>|<type-slot>``): ``Energietransitie`` / ``energietransitie``
become ONE concept, the ``programme``/``administrative_area`` "Regio" split
unifies, etc. But a ``mentions`` edge endpoint must be a real ``entity`` record
(the table is ``TYPE RELATION FROM source TO entity``). So each surviving concept
is mapped back to ONE **representative entity id** — the same MAX-salience member
R.6 picks as the concept's name/type — and one ``mentions`` edge is emitted per
(source, representative-entity) membership of that concept. The weight uses the
CONCEPT's unified type + document-frequency, so ``Energietransitie`` appearing in
two sources weighs as df=2 even though it is two raw rows.

This keeps the edge count at the U.1 measured figure (the concept→source count:
~67 with R.6 on across the 4 convenanten) while every edge still anchors at a real
entity row that the viz can click through to.

Filters / presets (U.1 Measurement 4)
=====================================
- **R.6 normalization + drop df==1 singletons** is the MANDATORY legibility step
  (466 active raw edges → 67). It is what kills the single-document leaf spokes;
  it is on by default and is what makes the graph drawable rather than a hairball.
- **min_weight** (default 0.0) is an OPTIONAL per-edge weight cutoff. The graph is
  already small so the default keeps every df≥2 edge and carries the weight as an
  attribute (for render thickness/opacity). The **0.3 "named-only" preset**
  (:data:`NAMED_ONLY_MIN_WEIGHT`) collapses to the named anchors ("Regio Deal" /
  "Regio") for a clean overview.
- **named_only** is a convenience flag equivalent to ``min_weight=0.3``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from shared.retrieval.kg_signal_normalizer import (
    concept_id,
    normalize_entities_for_signal,
)
from shared.retrieval.kg_source_scorer import (
    EntityRecord,
    entity_weight,
    type_salience,
)

# The 0.3 "named-only" preset from U.1 Measurement 4: at this cutoff only the two
# high-salience anchors survive (the "Regio Deal" programme / "Regio" area
# skeleton), every generic topic drops out. Exposed as a named constant so the
# service / endpoint can offer it without re-deriving the number.
NAMED_ONLY_MIN_WEIGHT = 0.3


@dataclass(frozen=True)
class MentionEdge:
    """One projected ``mentions`` edge (source → entity), pre-weighted.

    Attributes:
        source_id: The document (``source:...``) endpoint — the ``in`` of the
            RELATE edge.
        entity_id: The representative canonical entity (``entity:...``) endpoint —
            the ``out`` of the RELATE edge. This is the MAX-salience member of the
            normalized concept (the same row R.6 picks as the concept's
            name/type).
        weight: ``type_salience(concept_type) × IDF(df, n_sources)`` — the R.2
            per-entity contribution unit, computed on the UNIFIED concept's type
            and document frequency so case/type duplicates count as one.
        concept_name: The concept's surface name — the per-edge "why" the viz
            shows (e.g. "Regio Deal").
        concept_type: The concept's unified (max-salience) entity type.
        document_frequency: How many distinct sources the concept appears in (df);
            ``>= 2`` for every emitted edge when singletons are dropped.
    """

    source_id: str
    entity_id: str
    weight: float
    concept_name: str
    concept_type: str
    document_frequency: int


@dataclass(frozen=True)
class MentionsProjectionStats:
    """Before/after counts for the projection (measurement / acceptance).

    Attributes:
        input_entities: Raw active entity rows passed in.
        emitted_concepts: Concepts that survived R.6 (post case/type merge,
            post singleton-drop).
        edges_before_min_weight: Edges from the surviving concepts before the
            ``min_weight`` cutoff (the U.1 "67" figure with R.6 on).
        edges: Final emitted edges after the ``min_weight`` cutoff.
        unmapped_concepts: Surviving concepts whose representative entity id could
            NOT be recovered (defensive; expected 0 — every concept maps back).
        min_weight: The cutoff actually applied.
    """

    input_entities: int
    emitted_concepts: int
    edges_before_min_weight: int
    edges: int
    unmapped_concepts: int
    min_weight: float


def _representative_entity_ids(
    raw_entities: Sequence[EntityRecord],
) -> Dict[str, str]:
    """Map each R.6 concept id → the MAX-salience member's entity id.

    Mirrors the normalizer's "keep the most specific type" tie-break so the chosen
    representative is the SAME row R.6 surfaces as the concept's name/type. On a
    salience tie the first-seen member wins (deterministic given the input order;
    the caller passes ``source_documents`` sorted upstream for stability).
    """
    best: Dict[str, Tuple[float, str]] = {}
    for ent in raw_entities:
        name = (ent.canonical_name or "").strip()
        if not name or not ent.source_documents:
            continue
        cid = concept_id(ent.canonical_name, ent.entity_type)
        sal = type_salience(ent.entity_type)
        cur = best.get(cid)
        if cur is None or sal > cur[0]:
            best[cid] = (sal, str(ent.entity_id))
    return {cid: eid for cid, (_, eid) in best.items()}


def project_mentions_edges(
    active_entities: Sequence[EntityRecord],
    *,
    n_sources: Optional[int] = None,
    drop_singletons: bool = True,
    min_weight: float = 0.0,
    named_only: bool = False,
) -> Tuple[List[MentionEdge], MentionsProjectionStats]:
    """Project ``mentions`` (source → entity) edges from active entity rows.

    Pure: takes the active canonical entities (each with its
    ``source_documents`` provenance bag) and returns the weighted edge list +
    stats. No DB access — the regenerator persists the result.

    The pipeline:
      1. R.6 ``normalize_entities_for_signal`` — unify case/type-duplicate
         concepts, union their source sets, keep the max-salience type, and
         (default) drop df==1 singletons that cannot link two sources.
      2. For each surviving concept, recover its representative entity id (the
         max-salience member) and emit one edge per (source, representative)
         membership, weighted by ``entity_weight`` on the concept's unified
         type + df.
      3. Apply the optional ``min_weight`` cutoff.

    Args:
        active_entities: The ``status='active'`` canonical entities (the same
            input the R.2 scorer / U.1 measurement consume). The caller is
            responsible for the active filter (archived/merged/reference are
            excluded upstream at the query).
        n_sources: Corpus source count for the IDF rarity denominator. When
            ``None`` it is derived from the distinct sources observed across the
            passed entities; pass the TRUE corpus count (incl. entity-less
            sources) for a faithful IDF, exactly as R.2 does.
        drop_singletons: When True (default, MANDATORY for legibility) a concept
            appearing in only one source after grouping is dropped — it is a leaf
            spoke that cannot link two documents (the 466→67 step). Set False only
            to measure the pre-suppression set.
        min_weight: Per-edge weight cutoff (default 0.0 — keep every df≥2 edge and
            carry weight as a render attribute). Edges with ``weight < min_weight``
            are excluded.
        named_only: Convenience for the U.1 "named-only" preset; when True forces
            ``min_weight = max(min_weight, 0.3)`` so only the named anchors
            survive.

    Returns:
        ``(edges, stats)``. ``edges`` is sorted deterministically by
        ``(-weight, source_id, entity_id)`` so a re-run yields an identical
        ordering (idempotent regeneration). ``stats`` carries the before/after
        counts for the acceptance check.
    """
    effective_min_weight = min_weight
    if named_only:
        effective_min_weight = max(effective_min_weight, NAMED_ONLY_MIN_WEIGHT)

    raw = list(active_entities)

    # 1. R.6 concept projection (case/type unification + singleton drop).
    concepts, norm_stats = normalize_entities_for_signal(
        raw, drop_singletons=drop_singletons
    )

    # The true corpus source count for IDF — derived from observed sources when
    # not supplied (same fallback as the R.2 scorer).
    if n_sources is None:
        observed = {
            s for ent in raw for s in ent.source_documents if s
        }
        n_sources = len(observed)
    n_sources = max(int(n_sources), 1)

    # 2. Map each surviving concept back to a real representative entity id.
    rep_by_concept = _representative_entity_ids(raw)

    edges: List[MentionEdge] = []
    edges_before_cut = 0
    unmapped = 0
    for concept in concepts:
        rep_entity_id = rep_by_concept.get(concept.entity_id)
        if rep_entity_id is None:
            # Defensive: a concept emitted by the normalizer with no recoverable
            # member (should never happen — both pass over the same rows). Skip
            # rather than fabricate an endpoint.
            unmapped += 1
            continue
        df = len(concept.source_documents)
        weight = entity_weight(concept.entity_type, df, n_sources)
        for source_id in concept.source_documents:
            edges_before_cut += 1
            if weight < effective_min_weight:
                continue
            edges.append(
                MentionEdge(
                    source_id=str(source_id),
                    entity_id=rep_entity_id,
                    weight=weight,
                    concept_name=concept.canonical_name,
                    concept_type=concept.entity_type,
                    document_frequency=df,
                )
            )

    # Deterministic order so regeneration is byte-stable (idempotency at the
    # projection level; the persister also clears+recreates so the DB matches).
    edges.sort(key=lambda e: (-e.weight, e.source_id, e.entity_id))

    stats = MentionsProjectionStats(
        input_entities=norm_stats.input_entities,
        emitted_concepts=norm_stats.emitted_concepts,
        edges_before_min_weight=edges_before_cut,
        edges=len(edges),
        unmapped_concepts=unmapped,
        min_weight=effective_min_weight,
    )
    return edges, stats
