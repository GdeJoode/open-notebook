"""Unit tests for the search-facing KG-signal normalizer (Track R.6).

No DB. These assert the normalization contract in isolation:
  - case-fold dedup (``Energietransitie`` / ``energietransitie`` -> one concept);
  - cross-type unification for NAMED types (``programme "Regio"`` /
    ``administrative_area "Regio"`` -> one concept) and the deliberate
    NON-unification of a generic ``topic`` onto a named concept;
  - source-set UNION + MAX-salience type retention on merge;
  - singleton (df == 1) suppression;
  - predicate canonicalization (typo + EN/NL variants -> canonical, reversible
    map, unmapped pass-through);
  - the before/after measurement stats.
"""

from __future__ import annotations

from shared.retrieval.kg_signal_normalizer import (
    PREDICATE_CANON,
    canonical_predicate,
    canonicalize_relations,
    normalize_entities_for_signal,
)
from shared.retrieval.kg_source_scorer import (
    EntityRecord,
    RelationRecord,
    score_related_sources,
)


def _ent(eid, name, etype, sources):
    return EntityRecord(
        entity_id=eid,
        canonical_name=name,
        entity_type=etype,
        source_documents=tuple(sources),
    )


# ---------------------------------------------------------------------------
# Entity normalization — case-fold dedup
# ---------------------------------------------------------------------------
class TestCaseFoldDedup:
    def test_case_variants_merge_to_one_concept(self):
        ents = [
            _ent("e1", "Energietransitie", "topic", ["s:a"]),
            _ent("e2", "energietransitie", "topic", ["s:b"]),
        ]
        projected, stats = normalize_entities_for_signal(
            ents, drop_singletons=False
        )
        assert len(projected) == 1
        # Source sets unioned across the two casings.
        assert set(projected[0].source_documents) == {"s:a", "s:b"}
        assert stats.merged_groups == 1

    def test_whitespace_variants_merge(self):
        ents = [
            _ent("e1", "Brede  welvaart", "topic", ["s:a"]),
            _ent("e2", " brede welvaart ", "topic", ["s:b"]),
        ]
        projected, _ = normalize_entities_for_signal(ents, drop_singletons=False)
        assert len(projected) == 1
        assert set(projected[0].source_documents) == {"s:a", "s:b"}


# ---------------------------------------------------------------------------
# Entity normalization — cross-type unification (the Regio split)
# ---------------------------------------------------------------------------
class TestCrossTypeUnification:
    def test_named_same_surface_form_across_types_unifies(self):
        """programme "Regio" + administrative_area "Regio" -> one concept."""
        ents = [
            _ent("e1", "Regio", "programme", ["s:a"]),
            _ent("e2", "Regio", "administrative_area", ["s:b"]),
        ]
        projected, stats = normalize_entities_for_signal(
            ents, drop_singletons=False
        )
        assert len(projected) == 1
        assert set(projected[0].source_documents) == {"s:a", "s:b"}
        assert stats.merged_groups == 1
        # MAX-salience type retained (both are 1.0 here; either is fine).
        assert projected[0].entity_type in {"programme", "administrative_area"}

    def test_generic_topic_does_not_absorb_named_concept(self):
        """A coarse ``topic`` "Regio" must NOT fold onto the named "Regio"."""
        ents = [
            _ent("e1", "Regio", "programme", ["s:a"]),
            _ent("e2", "Regio", "topic", ["s:b"]),
        ]
        projected, _ = normalize_entities_for_signal(ents, drop_singletons=False)
        # Two distinct concepts: named vs generic.
        names_types = {(p.canonical_name.lower(), p.entity_type) for p in projected}
        assert ("regio", "programme") in names_types
        assert ("regio", "topic") in names_types
        assert len(projected) == 2

    def test_max_salience_type_wins_on_named_merge(self):
        """When named variants differ in salience, the MAX type is kept."""
        # organization (HIGH 1.0) vs event (MEDIUM 0.5) on the same surface form.
        ents = [
            _ent("e1", "Regio Deal", "event", ["s:a"]),
            _ent("e2", "Regio Deal", "organization", ["s:b"]),
        ]
        projected, _ = normalize_entities_for_signal(ents, drop_singletons=False)
        assert len(projected) == 1
        assert projected[0].entity_type == "organization"


# ---------------------------------------------------------------------------
# Singleton suppression
# ---------------------------------------------------------------------------
class TestSingletonSuppression:
    def test_singleton_dropped_by_default(self):
        ents = [
            _ent("e1", "Lonely Concept", "topic", ["s:a"]),
            _ent("e2", "Shared Concept", "organization", ["s:a", "s:b"]),
        ]
        projected, stats = normalize_entities_for_signal(ents)
        names = {p.canonical_name for p in projected}
        assert "Lonely Concept" not in names
        assert "Shared Concept" in names
        assert stats.singletons_dropped == 1

    def test_singleton_kept_when_flag_false(self):
        ents = [_ent("e1", "Lonely", "topic", ["s:a"])]
        projected, _ = normalize_entities_for_signal(ents, drop_singletons=False)
        assert len(projected) == 1

    def test_case_merge_rescues_a_would_be_singleton(self):
        """Two singletons under different casings unify into a df=2 concept,
        so neither is dropped — the core R.6 insight."""
        ents = [
            _ent("e1", "Energietransitie", "topic", ["s:a"]),
            _ent("e2", "energietransitie", "topic", ["s:b"]),
        ]
        projected, stats = normalize_entities_for_signal(ents)  # drop on
        assert len(projected) == 1
        assert stats.singletons_dropped == 0
        assert set(projected[0].source_documents) == {"s:a", "s:b"}


# ---------------------------------------------------------------------------
# Empty / robustness
# ---------------------------------------------------------------------------
class TestRobustness:
    def test_empty_name_or_no_sources_skipped(self):
        ents = [
            _ent("e1", "", "topic", ["s:a"]),
            _ent("e2", "Real", "topic", []),
            _ent("e3", "Real", "topic", ["s:a", "s:b"]),
        ]
        projected, stats = normalize_entities_for_signal(
            ents, drop_singletons=False
        )
        assert stats.empty_skipped == 2
        assert len(projected) == 1

    def test_empty_input(self):
        projected, stats = normalize_entities_for_signal([])
        assert projected == []
        assert stats.input_entities == 0
        assert stats.emitted_concepts == 0


# ---------------------------------------------------------------------------
# End-to-end: normalization strengthens the R.2 signal
# ---------------------------------------------------------------------------
class TestSignalEffect:
    def test_case_split_now_counts_as_shared(self):
        """Without normalization the case-split entities are distinct rows so
        sources A and B share nothing; with normalization they share one
        concept and score > 0."""
        ents = [
            _ent("e1", "Energietransitie", "topic", ["s:a"]),
            _ent("e2", "energietransitie", "topic", ["s:b"]),
        ]
        # Raw R.2: A and B own DIFFERENT entity rows -> no shared entity.
        raw = score_related_sources("s:a", ents, n_sources=2)
        assert raw == []  # nothing shared

        # R.6-normalized: one concept in both -> they link.
        projected, _ = normalize_entities_for_signal(ents, drop_singletons=False)
        normd = score_related_sources("s:a", projected, n_sources=2)
        assert len(normd) == 1
        assert normd[0].source_id == "s:b"
        assert normd[0].score > 0.0

    def test_regio_cross_type_split_now_counts_as_shared(self):
        ents = [
            _ent("e1", "Regio", "programme", ["s:a"]),
            _ent("e2", "Regio", "administrative_area", ["s:b"]),
        ]
        raw = score_related_sources("s:a", ents, n_sources=2)
        assert raw == []
        projected, _ = normalize_entities_for_signal(ents, drop_singletons=False)
        normd = score_related_sources("s:a", projected, n_sources=2)
        assert len(normd) == 1 and normd[0].source_id == "s:b"


# ---------------------------------------------------------------------------
# Predicate canonicalization
# ---------------------------------------------------------------------------
class TestPredicateCanon:
    def test_typo_fixed(self):
        assert canonical_predicate("ACEPTS") == "ACCEPTS"

    def test_nl_variants_collapse(self):
        assert canonical_predicate("IS_PIJLER_VAN") == "IS_PILLAR_OF"
        assert canonical_predicate("is pijler van") == "IS_PILLAR_OF"
        assert canonical_predicate("LEIDT_TOT") == "LEADS_TO"

    def test_casing_spacing_hyphen_normalized(self):
        assert canonical_predicate("Leidt-Tot") == "LEADS_TO"
        assert canonical_predicate("  leidt   tot  ") == "LEADS_TO"

    def test_en_nl_synonyms_map_together(self):
        assert canonical_predicate("werkt_bij") == canonical_predicate("works_at")
        assert canonical_predicate("financiert") == canonical_predicate("funds")
        assert (
            canonical_predicate("onderdeel_van")
            == canonical_predicate("PART_OF")
        )

    def test_unmapped_passes_through_as_upper_snake(self):
        assert canonical_predicate("some_new_edge") == "SOME_NEW_EDGE"
        assert canonical_predicate("brand new edge") == "BRAND_NEW_EDGE"

    def test_empty(self):
        assert canonical_predicate("") == ""
        assert canonical_predicate(None) == ""

    def test_map_is_idempotent(self):
        """Every canonical target is itself a key mapping to itself — applying
        the canon twice is a fixpoint (reversible/stable)."""
        for target in set(PREDICATE_CANON.values()):
            assert canonical_predicate(target) == target

    def test_canonicalize_relations_rewrites_predicate_only(self):
        rels = [
            RelationRecord("entity:a", "entity:b", "ACEPTS"),
            RelationRecord("entity:c", "entity:d", "IS_PIJLER_VAN"),
        ]
        out = canonicalize_relations(rels)
        assert out[0].relation_type == "ACCEPTS"
        assert out[0].in_id == "entity:a" and out[0].out_id == "entity:b"
        assert out[1].relation_type == "IS_PILLAR_OF"
