"""Pure KG-proximity scorer for source↔source retrieval (Track R.2).

This module is the **KG signal** of the hybrid retriever: it ranks OTHER
sources by *shared knowledge* (shared canonical entities, optionally extended
one hop through relations), NOT by dense text similarity (that is R.1). The
user steer is load-bearing — "the KG must play a large role in search" — so this
scorer is a first-class signal, deliberately kept pure (no I/O) so it is
unit-testable in isolation and reusable by the service layer and any offline
ablation harness (R.3).

Scoring model
-------------
For a query source ``Q`` and a candidate source ``B``::

    score(Q, B) = Σ over entities e shared by Q and B   of   weight(e)
                + (optional) Σ over 1-hop relation paths of   relation_weight

where each shared entity's contribution is::

    weight(e) = type_salience(e.entity_type) × rarity(e)

- **type_salience** (:data:`TYPE_SALIENCE`): named/specific canonical types
  (person, organization, government_organization, administrative_area,
  programme, technology, …) weigh HIGH; the generic LLM-noise buckets
  (``other``, bare ``topic`` / ``concept``) weigh LOW. The map is an explicit,
  documented table over the Track L canonical type set
  (``entity_persistence_service._ALLOWED_ENTITY_TYPES``) so the down-weight is
  inspectable and reviewable rather than buried in a heuristic.

- **rarity** (inverse source frequency, IDF-like): an entity appearing in many
  sources is less discriminating, so it contributes less. We use a smoothed IDF
  ``log((N_sources + 1) / (df + 1)) + 1`` where ``df`` is the number of distinct
  sources the entity appears in and ``N_sources`` is the corpus source count.
  The ``+1`` smoothing keeps the factor strictly positive (so a corpus-wide
  entity still counts a little, just much less than a unique one) and avoids a
  divide-by-zero / log(0). A ``programme`` unique to two convenanten therefore
  outweighs a ``government_organization`` present in every source, even though
  the latter has the higher raw salience — rarity and salience compose.

Only ``status='active'`` entities are scored; the caller is responsible for
passing in the active set (archived / merged / reference entities are excluded
at the query, see :class:`EntityRepository` loaders). The generic-bucket
down-weight in :data:`TYPE_SALIENCE` is what keeps ``topic``/``other`` noise
from dominating; archived entities are filtered out entirely upstream.

Explanation / lineage
---------------------
Every ranked result carries the list of shared entities (name, type, df, and
the exact weight each contributed) that drove its score — the "why matched"
lineage (the Purview linking-explanation lesson; feeds R.5). Results are sorted
by score descending with a deterministic ``source id`` tie-break.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Type salience — the explicit, documented down-weight table (Track R.2).
# ---------------------------------------------------------------------------
#
# Keyed on the canonical ``entity_type`` set from Track L
# (``entity_persistence_service._ALLOWED_ENTITY_TYPES``). The value is a
# multiplier on a shared entity's contribution. The split encodes the core R.2
# decision: a SHARED NAMED entity (a specific org / programme / place / person)
# is strong evidence two sources are about the same thing; a shared GENERIC
# bucket (``topic`` / ``concept`` / ``other``) is weak — those are exactly the
# 81%-of-active LLM noise the KG carries, so they are down-weighted, not
# excluded (exclusion would discard a real-but-coarse signal; the IDF rarity
# factor and this low salience together keep them from dominating).
#
# HIGH (1.0) — named, specific, identity-bearing types.
# MEDIUM (0.5) — typed but less identity-bearing (events, products, works,
#                bibliographic types, public consultations, social profiles).
# LOW (0.15) — generic catch-all buckets that carry little discriminative
#              signal on their own (``topic``, ``concept``).
# FLOOR (0.05) — ``other``: the residual "we could not type this" bucket; kept
#                non-zero so a shared ``other`` is a faint tie-breaker, never a
#                driver.
#
# An unknown / not-yet-in-enum type falls back to :data:`_DEFAULT_SALIENCE`
# (LOW) rather than crashing — the scorer must never fail on schema drift.
TYPE_SALIENCE: Dict[str, float] = {
    # --- HIGH: named, specific, identity-bearing ---
    "person": 1.0,
    "organization": 1.0,
    "government_organization": 1.0,
    "administrative_area": 1.0,
    "location": 1.0,
    "programme": 1.0,
    "technology": 1.0,
    "legislation": 1.0,
    "policy_document": 1.0,
    "dataset": 1.0,
    "grant": 1.0,
    "research_project": 1.0,
    "scholarly_article": 1.0,
    # --- MEDIUM: typed but less identity-bearing ---
    "event": 0.5,
    "product": 0.5,
    "creative_work": 0.5,
    "periodical": 0.5,
    "public_consultation": 0.5,
    "social_profile": 0.5,
    # --- LOW: generic catch-all buckets (the LLM-noise majority) ---
    "topic": 0.15,
    "concept": 0.15,
    # --- FLOOR: residual untyped ---
    "other": 0.05,
}

# Salience for an entity_type not present in TYPE_SALIENCE (schema drift / a
# future canonical type the scorer has not been taught yet). Treated as a
# generic bucket — conservative, never crashes.
_DEFAULT_SALIENCE = 0.15


@dataclass(frozen=True)
class EntityRecord:
    """One active canonical entity, projected for scoring.

    Attributes:
        entity_id: The entity record id (used only for de-dup / explanation).
        canonical_name: Surface name, surfaced in the explanation.
        entity_type: Canonical ``entity_type`` driving the salience lookup.
        source_documents: The distinct source ids this entity appears in
            (the provenance bag ``upsert_entity`` maintains). Stringified ids;
            the scorer dedups defensively.
    """

    entity_id: str
    canonical_name: str
    entity_type: str
    source_documents: Tuple[str, ...]


@dataclass(frozen=True)
class RelationRecord:
    """One active relation edge, projected for 1-hop expansion (optional).

    Attributes:
        in_id: Source-endpoint entity id (SurrealDB ``in``).
        out_id: Target-endpoint entity id (SurrealDB ``out``).
        relation_type: The predicate (used only for explanation today).
    """

    in_id: str
    out_id: str
    relation_type: str


@dataclass(frozen=True)
class SharedEntityContribution:
    """A single shared entity's contribution to a pair score (the "why").

    Attributes:
        entity_id: The shared entity's id.
        canonical_name: Its surface name (for display).
        entity_type: Its canonical type.
        document_frequency: How many distinct sources it appears in (df).
        weight: ``type_salience × rarity`` — the exact amount it added.
        via_relation: ``False`` for a directly-shared entity; ``True`` when the
            contribution came from a 1-hop relation path (the entity is on the
            candidate side, reached from a query-side entity through an edge).
    """

    entity_id: str
    canonical_name: str
    entity_type: str
    document_frequency: int
    weight: float
    via_relation: bool = False


@dataclass
class SourceKGScore:
    """A ranked related-source result with its lineage.

    Attributes:
        source_id: The related (candidate) source id.
        score: Total KG-proximity score (sum of contribution weights).
        contributions: The shared entities (+ optional 1-hop paths) that drove
            the score, sorted by descending weight — the "why matched" lineage.
    """

    source_id: str
    score: float
    contributions: List[SharedEntityContribution] = field(default_factory=list)


def type_salience(entity_type: Optional[str]) -> float:
    """Return the salience multiplier for a canonical ``entity_type``.

    Unknown / missing types fall back to :data:`_DEFAULT_SALIENCE` (generic),
    so the scorer is robust to schema drift and never raises.
    """
    if not entity_type:
        return _DEFAULT_SALIENCE
    return TYPE_SALIENCE.get(str(entity_type).strip().lower(), _DEFAULT_SALIENCE)


def inverse_source_frequency(document_frequency: int, n_sources: int) -> float:
    """Smoothed IDF rarity factor for an entity.

    ``log((n_sources + 1) / (df + 1)) + 1``. The ``+1`` smoothing keeps the
    factor strictly positive even for a corpus-wide entity (``df == n_sources``)
    and avoids ``log(0)`` / divide-by-zero, while still strongly favouring rare
    entities. ``df`` is clamped to ``[1, n_sources]`` defensively.

    Args:
        document_frequency: Distinct sources the entity appears in.
        n_sources: Total source count in the scored corpus.

    Returns:
        A strictly-positive rarity multiplier (larger = rarer = more
        discriminative).
    """
    n = max(int(n_sources), 1)
    df = min(max(int(document_frequency), 1), n)
    return math.log((n + 1) / (df + 1)) + 1.0


def entity_weight(
    entity_type: Optional[str], document_frequency: int, n_sources: int
) -> float:
    """Per-entity weight ``type_salience × rarity`` (the R.2 contribution unit).

    This is the single composition point for the two signals so the unit tests
    and the scorer agree by construction. See the module docstring for the
    rationale of each factor.
    """
    return type_salience(entity_type) * inverse_source_frequency(
        document_frequency, n_sources
    )


def _dedup_sources(source_documents: Iterable[str]) -> Tuple[str, ...]:
    """Stringify + dedup a source_documents bag, preserving first-seen order."""
    seen: List[str] = []
    seen_set = set()
    for raw in source_documents or ():
        sid = str(raw)
        if sid and sid not in seen_set:
            seen_set.add(sid)
            seen.append(sid)
    return tuple(seen)


def score_related_sources(
    query_source_id: str,
    entities: Sequence[EntityRecord],
    *,
    n_sources: Optional[int] = None,
    relations: Optional[Sequence[RelationRecord]] = None,
    expand_relations: bool = False,
    limit: Optional[int] = None,
) -> List[SourceKGScore]:
    """Rank sources related to ``query_source_id`` by KG proximity (pure).

    Args:
        query_source_id: The source to find related sources for.
        entities: The ACTIVE canonical entities to score over (each carrying its
            ``source_documents`` provenance bag). The caller is responsible for
            filtering to ``status='active'`` — archived/merged/reference
            entities must not be passed in (they are excluded at the query).
        n_sources: Corpus source count used for the IDF rarity denominator. When
            ``None`` it is derived from the distinct sources observed across the
            passed entities — a safe default, though passing the true corpus
            count (incl. sources with no entities) gives a more faithful IDF.
        relations: Active relation edges, required only when
            ``expand_relations`` is True. Each edge connects two entity ids.
        expand_relations: When True, also credit a candidate source for entities
            it owns that are ONE relation hop from a query-source entity
            (``Q has X, X--rel-->Y or Y--rel-->X, B has Y``). Gated OFF by
            default: the 1-hop path multiplies through the duplicate-predicate
            noise R.6 has not yet trimmed, so shared-entity scoring ships as the
            reliable core and relation expansion is opt-in. The hop is bounded
            (exactly one edge; no transitive walk) and each candidate entity is
            credited at most once via the strongest path.
        limit: Optional cap on the number of ranked results returned (the
            highest-scoring ``limit``). ``None`` returns all related sources.

    Returns:
        A list of :class:`SourceKGScore`, sorted by ``score`` descending with a
        deterministic ``source_id`` ascending tie-break, excluding the query
        source itself and any source with a zero/empty overlap. Empty when the
        query source shares no active entities with any other source.
    """
    query_source_id = str(query_source_id)

    # --- Index pass: df per entity + which entities the query source owns. ----
    # Build, for every entity, its deduped source set; this gives df (rarity)
    # and lets us find the query source's own entities in one sweep.
    entity_by_id: Dict[str, EntityRecord] = {}
    entity_sources: Dict[str, Tuple[str, ...]] = {}
    observed_sources: set = set()
    for ent in entities:
        srcs = _dedup_sources(ent.source_documents)
        if not srcs:
            continue
        entity_by_id[ent.entity_id] = ent
        entity_sources[ent.entity_id] = srcs
        observed_sources.update(srcs)

    corpus_n = int(n_sources) if n_sources is not None else len(observed_sources)
    corpus_n = max(corpus_n, 1)

    # Entities owned by the query source (direct membership).
    query_entity_ids = {
        eid
        for eid, srcs in entity_sources.items()
        if query_source_id in srcs
    }

    # Accumulate per candidate-source contributions, deduping by entity so the
    # same shared entity is never double-counted for one pair.
    # candidate_source_id -> {entity_id -> SharedEntityContribution}
    per_source: Dict[str, Dict[str, SharedEntityContribution]] = {}

    def _credit(
        candidate_source: str,
        ent: EntityRecord,
        df: int,
        via_relation: bool,
    ) -> None:
        if candidate_source == query_source_id:
            return
        weight = entity_weight(ent.entity_type, df, corpus_n)
        if weight <= 0.0:
            return
        bucket = per_source.setdefault(candidate_source, {})
        existing = bucket.get(ent.entity_id)
        # A direct share always beats a relation-path credit for the same
        # entity; otherwise keep the stronger (they share df/weight, so the
        # first direct one wins and we never inflate by counting both).
        if existing is None or (existing.via_relation and not via_relation):
            bucket[ent.entity_id] = SharedEntityContribution(
                entity_id=ent.entity_id,
                canonical_name=ent.canonical_name,
                entity_type=ent.entity_type,
                document_frequency=df,
                weight=weight,
                via_relation=via_relation,
            )

    # --- Direct shared-entity scoring (the reliable core). --------------------
    for eid in query_entity_ids:
        ent = entity_by_id[eid]
        srcs = entity_sources[eid]
        df = len(srcs)
        for candidate in srcs:
            _credit(candidate, ent, df, via_relation=False)

    # --- Optional 1-hop relation expansion (gated). ---------------------------
    if expand_relations and relations:
        # Adjacency over entity ids (undirected — a relation links the two
        # endpoints regardless of direction for the purpose of "these two
        # entities are connected").
        neighbours: Dict[str, set[str]] = {}
        for rel in relations:
            a, b = str(rel.in_id), str(rel.out_id)
            if a == b:
                continue
            neighbours.setdefault(a, set()).add(b)
            neighbours.setdefault(b, set()).add(a)

        for eid in query_entity_ids:
            for nbr_id in neighbours.get(eid, ()):  # one hop only
                if nbr_id in query_entity_ids:
                    # Already a direct query entity — covered above.
                    continue
                nbr = entity_by_id.get(nbr_id)
                if nbr is None:
                    continue
                nbr_srcs = entity_sources.get(nbr_id, ())
                df = len(nbr_srcs)
                for candidate in nbr_srcs:
                    _credit(candidate, nbr, df, via_relation=True)

    # --- Materialise + rank. --------------------------------------------------
    results: List[SourceKGScore] = []
    for candidate_source, contrib_map in per_source.items():
        contributions = sorted(
            contrib_map.values(),
            key=lambda c: (-c.weight, c.canonical_name),
        )
        total = sum(c.weight for c in contributions)
        if total <= 0.0:
            continue
        results.append(
            SourceKGScore(
                source_id=candidate_source,
                score=total,
                contributions=contributions,
            )
        )

    results.sort(key=lambda r: (-r.score, r.source_id))
    if limit is not None:
        results = results[: max(int(limit), 0)]
    return results
