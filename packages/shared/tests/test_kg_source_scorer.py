"""Unit tests for the pure KG-proximity source scorer (Track R.2).

No DB. These assert the scorer's mathematical contract directly: the
type-salience down-weight (generic ``topic``/``other`` ranks below named
types), the IDF rarity factor (rare entities outweigh corpus-wide ones),
shared-entity overlap accumulation, deterministic ordering, the per-pair
explanation lineage, and the gated 1-hop relation expansion.
"""

from __future__ import annotations

import math

from shared.retrieval.kg_source_scorer import (
    TYPE_SALIENCE,
    EntityRecord,
    RelationRecord,
    entity_weight,
    inverse_source_frequency,
    score_related_sources,
    type_salience,
)


def _ent(eid, name, etype, sources):
    return EntityRecord(
        entity_id=eid,
        canonical_name=name,
        entity_type=etype,
        source_documents=tuple(sources),
    )


# ---------------------------------------------------------------------------
# Salience + rarity primitives
# ---------------------------------------------------------------------------
class TestSalience:
    def test_named_types_outrank_generic_buckets(self):
        """AC2: named/typed entities weigh strictly more than generic buckets."""
        for named in (
            "person",
            "organization",
            "government_organization",
            "administrative_area",
            "programme",
            "technology",
        ):
            assert type_salience(named) > type_salience("topic")
            assert type_salience(named) > type_salience("concept")
            assert type_salience(named) > type_salience("other")

    def test_topic_and_other_are_low(self):
        assert type_salience("topic") < 0.5
        assert type_salience("other") < type_salience("topic")

    def test_unknown_type_falls_back_not_crash(self):
        # Schema drift / an untaught type must not raise and must be treated as
        # a generic bucket (not a high-salience driver).
        assert type_salience("frobnicator") == type_salience("concept")
        assert type_salience(None) > 0.0
        assert type_salience("") > 0.0

    def test_case_insensitive(self):
        assert type_salience("PROGRAMME") == type_salience("programme")

    def test_salience_table_covers_canonical_set(self):
        # The documented table must at least carry the noise buckets the
        # down-weight depends on.
        for t in ("topic", "concept", "other"):
            assert t in TYPE_SALIENCE


class TestRarity:
    def test_rarer_entity_scores_higher(self):
        n = 10
        rare = inverse_source_frequency(1, n)
        common = inverse_source_frequency(n, n)
        assert rare > common

    def test_strictly_positive_even_corpus_wide(self):
        # An entity in every source still counts a little (smoothing), never 0.
        assert inverse_source_frequency(4, 4) > 0.0

    def test_df_clamped(self):
        # df above n or below 1 is clamped, never produces NaN/negative.
        assert inverse_source_frequency(0, 4) == inverse_source_frequency(1, 4)
        assert inverse_source_frequency(99, 4) == inverse_source_frequency(4, 4)

    def test_matches_formula(self):
        assert math.isclose(
            inverse_source_frequency(2, 4), math.log(5 / 3) + 1.0
        )


class TestEntityWeight:
    def test_composes_salience_and_rarity(self):
        w = entity_weight("programme", document_frequency=2, n_sources=4)
        assert math.isclose(
            w, type_salience("programme") * inverse_source_frequency(2, 4)
        )

    def test_rare_named_beats_common_named(self):
        rare = entity_weight("programme", 1, 4)
        common = entity_weight("programme", 4, 4)
        assert rare > common


# ---------------------------------------------------------------------------
# The scorer — overlap, down-weight, ordering, explanation
# ---------------------------------------------------------------------------
class TestScoreRelatedSources:
    def test_shared_named_entity_links_two_sources(self):
        """AC1: a shared active named entity produces a ranked related source."""
        entities = [
            _ent("entity:p", "Regio Deal", "programme", ["source:a", "source:b"]),
        ]
        results = score_related_sources("source:a", entities)
        assert [r.source_id for r in results] == ["source:b"]
        assert results[0].score > 0
        # Explanation names the entity that drove it.
        names = [c.canonical_name for c in results[0].contributions]
        assert "Regio Deal" in names

    def test_named_pair_outranks_topic_only_pair(self):
        """AC2 (the core down-weight proof): two sources sharing only a generic
        ``topic`` rank BELOW two sharing a named org/programme."""
        entities = [
            # Q shares a named programme with B.
            _ent("entity:prog", "Regio Deal", "programme", ["source:q", "source:b"]),
            # Q shares only a generic topic with C.
            _ent("entity:top", "gezondheid", "topic", ["source:q", "source:c"]),
        ]
        results = score_related_sources("source:q", entities)
        order = [r.source_id for r in results]
        assert order.index("source:b") < order.index("source:c")
        score_b = next(r.score for r in results if r.source_id == "source:b")
        score_c = next(r.score for r in results if r.source_id == "source:c")
        assert score_b > score_c

    def test_other_bucket_is_weakest(self):
        """AC2: a shared ``other`` entity ranks below a shared ``topic``."""
        entities = [
            _ent("entity:t", "thing", "topic", ["source:q", "source:b"]),
            _ent("entity:o", "blob", "other", ["source:q", "source:c"]),
        ]
        results = score_related_sources("source:q", entities)
        order = [r.source_id for r in results]
        assert order.index("source:b") < order.index("source:c")

    def test_multiple_shared_entities_accumulate(self):
        """More shared entities ⇒ higher score; contributions list them all."""
        entities = [
            _ent("entity:1", "Org A", "organization", ["source:q", "source:b"]),
            _ent("entity:2", "Prog B", "programme", ["source:q", "source:b"]),
            _ent("entity:3", "Org C", "organization", ["source:q", "source:c"]),
        ]
        results = score_related_sources("source:q", entities)
        score_b = next(r.score for r in results if r.source_id == "source:b")
        score_c = next(r.score for r in results if r.source_id == "source:c")
        assert score_b > score_c
        contrib_b = next(
            r.contributions for r in results if r.source_id == "source:b"
        )
        assert len(contrib_b) == 2

    def test_query_source_excluded_from_results(self):
        entities = [
            _ent("entity:1", "Org A", "organization", ["source:q", "source:b"]),
        ]
        results = score_related_sources("source:q", entities)
        assert "source:q" not in [r.source_id for r in results]

    def test_unrelated_source_absent(self):
        """A source sharing no entities is not in the results at all."""
        entities = [
            _ent("entity:1", "Org A", "organization", ["source:q", "source:b"]),
            # source:z shares nothing with q.
            _ent("entity:2", "Org Z", "organization", ["source:z"]),
        ]
        results = score_related_sources("source:q", entities)
        assert [r.source_id for r in results] == ["source:b"]

    def test_deterministic_tie_break_by_source_id(self):
        """Equal-score sources come back in a stable id-ascending order."""
        entities = [
            _ent("entity:1", "Org A", "organization", ["source:q", "source:m"]),
            _ent("entity:2", "Org A2", "organization", ["source:q", "source:n"]),
        ]
        # Identical type + identical df ⇒ identical weight ⇒ tie.
        r1 = score_related_sources("source:q", entities)
        r2 = score_related_sources("source:q", entities)
        ids1 = [r.source_id for r in r1]
        ids2 = [r.source_id for r in r2]
        assert ids1 == ids2 == sorted(ids1)

    def test_contributions_sorted_by_weight_desc(self):
        entities = [
            _ent("entity:top", "topic ent", "topic", ["source:q", "source:b"]),
            _ent("entity:org", "org ent", "organization", ["source:q", "source:b"]),
        ]
        results = score_related_sources("source:q", entities)
        contribs = results[0].contributions
        assert contribs[0].entity_type == "organization"  # higher weight first
        weights = [c.weight for c in contribs]
        assert weights == sorted(weights, reverse=True)

    def test_shared_entity_not_double_counted(self):
        """The same entity shared on the query↔candidate pair counts once."""
        entities = [
            _ent("entity:1", "Org A", "organization",
                 ["source:q", "source:q", "source:b"]),  # dup query id
        ]
        results = score_related_sources("source:q", entities)
        contrib = results[0].contributions
        assert len(contrib) == 1

    def test_limit_caps_results(self):
        entities = [
            _ent("e1", "A", "organization", ["source:q", "source:b"]),
            _ent("e2", "B", "organization", ["source:q", "source:c"]),
            _ent("e3", "C", "organization", ["source:q", "source:d"]),
        ]
        results = score_related_sources("source:q", entities, limit=2)
        assert len(results) == 2

    def test_empty_when_no_entities(self):
        assert score_related_sources("source:q", []) == []

    def test_n_sources_override_changes_idf(self):
        """A larger corpus N makes a given df rarer ⇒ higher score."""
        entities = [
            _ent("e1", "A", "organization", ["source:q", "source:b"]),
        ]
        small = score_related_sources("source:q", entities, n_sources=2)
        big = score_related_sources("source:q", entities, n_sources=100)
        assert big[0].score > small[0].score


# ---------------------------------------------------------------------------
# Optional gated 1-hop relation expansion
# ---------------------------------------------------------------------------
class TestRelationExpansion:
    def test_off_by_default(self):
        """Relation expansion must be opt-in; default ignores relations."""
        entities = [
            _ent("entity:x", "X", "organization", ["source:q"]),
            _ent("entity:y", "Y", "organization", ["source:b"]),
        ]
        relations = [RelationRecord("entity:x", "entity:y", "RELATES_TO")]
        # Without expansion, X (q-only) and Y (b-only) share no source ⇒ empty.
        results = score_related_sources(
            "source:q", entities, relations=relations
        )
        assert results == []

    def test_one_hop_credits_connected_source(self):
        """AC1 (relation proximity): Q has X, X--rel-->Y, B has Y ⇒ B credited."""
        entities = [
            _ent("entity:x", "X", "organization", ["source:q"]),
            _ent("entity:y", "Y", "organization", ["source:b"]),
        ]
        relations = [RelationRecord("entity:x", "entity:y", "RELATES_TO")]
        results = score_related_sources(
            "source:q", entities, relations=relations, expand_relations=True
        )
        assert [r.source_id for r in results] == ["source:b"]
        # Explanation marks it as a relation path.
        assert results[0].contributions[0].via_relation is True

    def test_direct_share_beats_relation_path_for_same_entity(self):
        """A directly-shared entity is credited as direct, not via_relation,
        even if a relation path also reaches it."""
        entities = [
            _ent("entity:y", "Y", "organization", ["source:q", "source:b"]),
            _ent("entity:x", "X", "organization", ["source:q"]),
        ]
        relations = [RelationRecord("entity:x", "entity:y", "RELATES_TO")]
        results = score_related_sources(
            "source:q", entities, relations=relations, expand_relations=True
        )
        contrib = next(
            c
            for r in results
            if r.source_id == "source:b"
            for c in r.contributions
            if c.entity_id == "entity:y"
        )
        assert contrib.via_relation is False

    def test_self_loop_relation_ignored(self):
        entities = [
            _ent("entity:x", "X", "organization", ["source:q"]),
        ]
        relations = [RelationRecord("entity:x", "entity:x", "SELF")]
        results = score_related_sources(
            "source:q", entities, relations=relations, expand_relations=True
        )
        assert results == []
