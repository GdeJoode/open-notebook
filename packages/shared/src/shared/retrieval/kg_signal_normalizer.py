"""Search-facing KG-signal normalization (Track R.6).

This module is a **search-facing normalization layer** that sharpens the R.2 KG
proximity signal by trimming extraction noise — WITHOUT mutating the canonical
``entity`` / ``relation`` rows. It is purely an in-memory projection applied
inside the retrieval path (the R.2 service, before the pure scorer runs); the
exporters, entity-resolution (Track K), and triage (Track Q) all keep reading
the canonical data unchanged. Track K owns real merges; R.6 only changes how the
*ranking signal* counts shared concepts.

The problem it fixes
====================
R.2's KG signal counts SHARED entities: two sources both linking the same active
entity *row* reinforce each other. But the LLM extraction emits the same concept
under different casings (``Energietransitie`` / ``energietransitie``) and, worse,
the SAME surface form under different types (a ``programme`` "Regio" vs an
``administrative_area`` "Regio"). Those are DISTINCT entity rows with distinct
ids, so the scorer treats them as *different* concepts — the case/type split
SILENTLY WEAKENS the signal (source A's ``Energietransitie`` and source B's
``energietransitie`` no longer count as shared). This module groups such variants
into ONE search-facing concept so they count as shared again.

Three knobs (all additive, reversible, config-driven)
=====================================================
1. **Entity normalization** (:func:`normalize_entities_for_signal`): group active
   entities by a normalized concept key (case-folded + whitespace-trimmed name;
   for NAMED/high-salience types the same surface form across types is ONE
   concept, so the ``programme``/``administrative_area`` "Regio" split unifies).
   The grouped concept's source set = the UNION of its members'
   ``source_documents``; its type = the member with the MAX (most specific)
   salience, so the merged concept keeps the strongest typing. No entity row is
   rewritten — this is an in-memory projection of the retrieval input.
2. **Singleton suppression** (same function, ``drop_singletons=True``): a concept
   that, after grouping, appears in only ONE source (df == 1) cannot link any two
   sources, so it contributes nothing to cross-source ranking and is dropped from
   the signal. This is signal-only; the row stays in the KG.
3. **Predicate canonicalization** (:data:`PREDICATE_CANON` +
   :func:`canonicalize_relations`): a reviewable EN/NL + typo → canonical
   predicate map for the relation projection used by 1-hop expansion + telemetry.
   It is a config table, not a data migration — fully reversible by editing the
   map. Data-free while expansion stays off, but the map exists + is unit-tested
   so turning expansion on (R.4/R.6) does not multiply through the duplicate
   predicate noise.

Generic-bucket suppression is already handled by the scorer's ``TYPE_SALIENCE``
down-weight (``other`` / bare ``topic`` weigh low); this module confirms +
extends it by also dropping post-grouping singletons. The salience table is the
single source of truth for "named vs generic" so the two layers stay consistent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from shared.retrieval.kg_source_scorer import (
    EntityRecord,
    RelationRecord,
    type_salience,
)

# ---------------------------------------------------------------------------
# Named-type threshold: which entity types unify across types on surface form.
# ---------------------------------------------------------------------------
#
# The entity-normalization key has two regimes (see _concept_key):
#  - NAMED / identity-bearing types (salience >= _NAMED_SALIENCE_FLOOR): the same
#    surface form is treated as ONE concept REGARDLESS of type, so the
#    cross-typed split (``programme "Regio"`` vs ``administrative_area "Regio"``)
#    collapses to a single concept. Two sources each naming "Regio" — even under
#    different types — count as sharing it.
#  - GENERIC buckets (``topic`` / ``concept`` / ``other``, salience below the
#    floor): the type IS part of the key, so a generic ``topic`` "Regio" is NOT
#    folded onto a named "Regio" (a coarse LLM topic must not absorb a real named
#    entity). Generic concepts only merge with same-type, same-name variants
#    (the pure case-fold dedup ``Energietransitie`` / ``energietransitie``).
#
# Threshold is the MEDIUM salience tier (0.5) — HIGH (1.0) and MEDIUM (0.5)
# types are "named"; LOW (0.15 topic/concept) and FLOOR (0.05 other) are generic.
_NAMED_SALIENCE_FLOOR = 0.5

# Sentinel used as the "type slot" of the unified key for named concepts, so all
# named variants of one surface form share a key irrespective of their real type.
_NAMED_TYPE_SLOT = "\x00named"


# ---------------------------------------------------------------------------
# Predicate canonicalization map (Track R.6) — reviewable + reversible.
# ---------------------------------------------------------------------------
#
# A curated EN/NL-variant + typo -> canonical predicate table for the RELATION
# projection (1-hop expansion + telemetry). This is a CONFIG table, not a data
# migration: nothing is rewritten on disk; the canon is applied in-memory to the
# relation projection only, and is reversed by editing this map. Keys are matched
# case-insensitively after whitespace/hyphen->underscore normalization (see
# :func:`canonical_predicate`), so ``is pijler van`` / ``IS-PIJLER-VAN`` /
# ``IS_PIJLER_VAN`` all hit the same entry.
#
# Grouping rationale (each row is reviewable):
#  - Typo fixes: ``ACEPTS`` -> ``ACCEPTS`` (single-char LLM typo).
#  - EN/NL synonyms collapsed onto ONE canonical so the same semantic edge from
#    an English-prompted and a Dutch-prompted extraction counts once:
#      * accepts / aanvaardt / accepteert            -> ACCEPTS
#      * is_pillar_of / is_pijler_van                 -> IS_PILLAR_OF
#      * leads_to / leidt_tot / resulteert_in         -> LEADS_TO
#      * part_of / onderdeel_van / maakt_deel_uit_van -> PART_OF
#      * works_at / werkt_bij                         -> WORKS_AT
#      * located_in / gevestigd_in / bevindt_zich_in  -> LOCATED_IN
#      * funds / financiert / subsidieert             -> FUNDS
#      * collaborates_with / werkt_samen_met          -> COLLABORATES_WITH
# Unknown predicates pass through canonicalized-but-unmapped (upper-snake), never
# dropped — an untaught edge stays in the projection rather than vanishing.
PREDICATE_CANON: Dict[str, str] = {
    # typo
    "ACEPTS": "ACCEPTS",
    # accepts (EN/NL)
    "ACCEPTS": "ACCEPTS",
    "AANVAARDT": "ACCEPTS",
    "ACCEPTEERT": "ACCEPTS",
    # is pillar of (EN/NL) — IS_PIJLER_VAN is the observed NL form
    "IS_PILLAR_OF": "IS_PILLAR_OF",
    "IS_PIJLER_VAN": "IS_PILLAR_OF",
    "PIJLER_VAN": "IS_PILLAR_OF",
    # leads to (EN/NL) — LEIDT_TOT is the observed NL form
    "LEADS_TO": "LEADS_TO",
    "LEIDT_TOT": "LEADS_TO",
    "RESULTEERT_IN": "LEADS_TO",
    # part of (EN/NL)
    "PART_OF": "PART_OF",
    "ONDERDEEL_VAN": "PART_OF",
    "MAAKT_DEEL_UIT_VAN": "PART_OF",
    # works at (EN/NL)
    "WORKS_AT": "WORKS_AT",
    "WERKT_BIJ": "WORKS_AT",
    # located in (EN/NL)
    "LOCATED_IN": "LOCATED_IN",
    "GEVESTIGD_IN": "LOCATED_IN",
    "BEVINDT_ZICH_IN": "LOCATED_IN",
    # funds (EN/NL)
    "FUNDS": "FUNDS",
    "FINANCIERT": "FUNDS",
    "SUBSIDIEERT": "FUNDS",
    # collaborates with (EN/NL)
    "COLLABORATES_WITH": "COLLABORATES_WITH",
    "WERKT_SAMEN_MET": "COLLABORATES_WITH",
}


@dataclass
class _ConceptAccumulator:
    """Mutable accumulator for one grouped concept (internal to grouping)."""

    name: str
    type: str
    salience: float
    sources: Set[str] = field(default_factory=set)
    members: int = 0


@dataclass(frozen=True)
class NormalizationStats:
    """Before/after counts for the entity-normalization projection (measurement).

    Attributes:
        input_entities: Raw active entity rows passed in.
        grouped_concepts: Distinct concepts after case/type grouping (the merge
            count is ``input_entities - grouped_concepts`` net of empties).
        merged_groups: Concepts whose group has >1 member entity row (i.e. a
            case/type duplicate was unified — the noise this fixes).
        singletons_dropped: Concepts dropped because they appear in only one
            source after grouping (df == 1, can't link sources).
        emitted_concepts: Concepts emitted to the scorer (grouped minus dropped).
        empty_skipped: Rows skipped for having no name or no source documents.
    """

    input_entities: int
    grouped_concepts: int
    merged_groups: int
    singletons_dropped: int
    emitted_concepts: int
    empty_skipped: int


def _collapse_ws(text: str) -> str:
    """Case-fold + collapse all interior whitespace + strip ends."""
    return " ".join(str(text or "").split()).lower()


def _concept_key(canonical_name: str, entity_type: str) -> Tuple[str, str]:
    """Normalized grouping key for one entity (the search-facing concept id).

    The first slot is the case-folded, whitespace-collapsed surface form. The
    second slot is the *type slot*:

    - For NAMED types (salience >= the named floor) it is the shared
      :data:`_NAMED_TYPE_SLOT` sentinel, so the SAME surface form under ANY named
      type maps to ONE key (the ``programme``/``administrative_area`` "Regio"
      unification).
    - For GENERIC buckets the slot is the real (lowercased) type, so a generic
      ``topic`` "Regio" stays distinct from a named "Regio" — a coarse LLM topic
      must never absorb a real named entity.
    """
    name_key = _collapse_ws(canonical_name)
    if type_salience(entity_type) >= _NAMED_SALIENCE_FLOOR:
        return (name_key, _NAMED_TYPE_SLOT)
    return (name_key, str(entity_type or "").strip().lower())


def normalize_entities_for_signal(
    entities: Sequence[EntityRecord],
    *,
    drop_singletons: bool = True,
) -> Tuple[List[EntityRecord], NormalizationStats]:
    """Project active entities into the search-facing concept set (R.6).

    Groups entities by :func:`_concept_key` (case/type unification), unions each
    group's ``source_documents``, keeps the MAX-salience (most specific) type and
    a representative name, and optionally drops post-grouping singletons (df == 1)
    that cannot link two sources. The returned :class:`EntityRecord` list is what
    the R.2 scorer consumes; the original rows are untouched.

    Args:
        entities: The ACTIVE canonical entities (the same input the R.2 scorer
            takes — case/type duplicates and all).
        drop_singletons: When True (default) a concept appearing in only one
            source after grouping is excluded from the signal (it can't link
            sources). Set False to keep all concepts (e.g. for measurement of the
            pre-suppression set).

    Returns:
        ``(projected_entities, stats)``. ``projected_entities`` is one
        :class:`EntityRecord` per surviving concept, with a synthetic stable
        ``entity_id`` (``concept::<name>|<type-slot>``) so the scorer's per-entity
        de-dup keys on the concept, not the original row id. ``stats`` carries the
        before/after counts for measurement.
    """
    # group key -> mutable accumulator
    groups: Dict[Tuple[str, str], _ConceptAccumulator] = {}
    empty_skipped = 0
    input_count = 0

    for ent in entities:
        input_count += 1
        name_key = _collapse_ws(ent.canonical_name)
        srcs = _dedup_sources(ent.source_documents)
        if not name_key or not srcs:
            empty_skipped += 1
            continue
        key = _concept_key(ent.canonical_name, ent.entity_type)
        acc = groups.get(key)
        if acc is None:
            acc = _ConceptAccumulator(
                name=ent.canonical_name,
                type=ent.entity_type,
                salience=type_salience(ent.entity_type),
                sources=set(srcs),
                members=1,
            )
            groups[key] = acc
        else:
            acc.members += 1
            sal = type_salience(ent.entity_type)
            # Keep the MAX-salience (most specific) type as the concept's type.
            if sal > acc.salience:
                acc.salience = sal
                acc.type = ent.entity_type
                acc.name = ent.canonical_name
            acc.sources.update(srcs)

    grouped_concepts = len(groups)
    merged_groups = sum(1 for a in groups.values() if a.members > 1)

    projected: List[EntityRecord] = []
    singletons_dropped = 0
    for (name_key, type_slot), acc in groups.items():
        if drop_singletons and len(acc.sources) <= 1:
            singletons_dropped += 1
            continue
        projected.append(
            EntityRecord(
                entity_id=f"concept::{name_key}|{type_slot}",
                canonical_name=acc.name,
                entity_type=acc.type,
                # Deterministic order so the scorer's df is stable across runs.
                source_documents=tuple(sorted(acc.sources)),
            )
        )

    stats = NormalizationStats(
        input_entities=input_count,
        grouped_concepts=grouped_concepts,
        merged_groups=merged_groups,
        singletons_dropped=singletons_dropped,
        emitted_concepts=len(projected),
        empty_skipped=empty_skipped,
    )
    return projected, stats


def concept_id(canonical_name: str, entity_type: str) -> str:
    """Return the synthetic concept id an entity normalizes to.

    The same id :func:`normalize_entities_for_signal` assigns the grouped
    concept (``concept::<name-key>|<type-slot>``). Lets callers remap relation
    endpoints (raw entity ids) onto concept ids so 1-hop expansion runs in the
    SAME id space as the normalized entity projection.
    """
    name_key, type_slot = _concept_key(canonical_name, entity_type)
    return f"concept::{name_key}|{type_slot}"


def remap_relations_to_concepts(
    relations: Iterable[RelationRecord],
    entities: Sequence[EntityRecord],
    *,
    canonicalize_predicates: bool = True,
) -> List[RelationRecord]:
    """Re-key relation endpoints onto concept ids (for normalized expansion).

    When the entity signal is normalized, entities carry synthetic
    ``concept::...`` ids, but relation endpoints still reference the original
    entity row ids — so 1-hop expansion would find no adjacency. This maps each
    endpoint to its concept id (via the raw ``entities`` it knows the
    name/type for), optionally canonicalizes the predicate, and drops self-loops
    that collapse when two endpoints normalize to the same concept. Endpoints
    with no known entity (e.g. dangling) are passed through unchanged so nothing
    silently vanishes.
    """
    id_to_concept: Dict[str, str] = {
        ent.entity_id: concept_id(ent.canonical_name, ent.entity_type)
        for ent in entities
    }
    out: List[RelationRecord] = []
    for rel in relations or ():
        in_c = id_to_concept.get(rel.in_id, rel.in_id)
        out_c = id_to_concept.get(rel.out_id, rel.out_id)
        if in_c == out_c:
            # Both endpoints collapsed to one concept — no longer an edge.
            continue
        pred = (
            canonical_predicate(rel.relation_type)
            if canonicalize_predicates
            else rel.relation_type
        )
        out.append(RelationRecord(in_id=in_c, out_id=out_c, relation_type=pred))
    return out


def canonical_predicate(relation_type: Optional[str]) -> str:
    """Map a raw predicate to its canonical form via :data:`PREDICATE_CANON`.

    Normalizes the raw label first (trim, collapse whitespace + hyphens to
    underscores, uppercase) so casing/spacing/hyphen variants all hit the same
    map entry, then looks it up. An unmapped predicate returns its normalized
    upper-snake form (never dropped) — so an untaught edge survives in the
    projection rather than vanishing.

    Examples:
        >>> canonical_predicate("ACEPTS")
        'ACCEPTS'
        >>> canonical_predicate("is pijler van")
        'IS_PILLAR_OF'
        >>> canonical_predicate("Leidt-Tot")
        'LEADS_TO'
        >>> canonical_predicate("some_new_edge")
        'SOME_NEW_EDGE'
        >>> canonical_predicate("")
        ''
    """
    if not relation_type:
        return ""
    norm = "_".join(str(relation_type).replace("-", " ").split()).upper()
    if not norm:
        return ""
    return PREDICATE_CANON.get(norm, norm)


def canonicalize_relations(
    relations: Iterable[RelationRecord],
) -> List[RelationRecord]:
    """Project relation edges with canonicalized predicates (R.6).

    Rewrites each edge's ``relation_type`` through :func:`canonical_predicate`,
    leaving the endpoints untouched. In-memory only — the ``relation`` rows are
    not mutated. Used for the 1-hop expansion projection + telemetry so EN/NL +
    typo predicate variants collapse onto one canonical edge label.
    """
    out: List[RelationRecord] = []
    for rel in relations or ():
        out.append(
            RelationRecord(
                in_id=rel.in_id,
                out_id=rel.out_id,
                relation_type=canonical_predicate(rel.relation_type),
            )
        )
    return out


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
