"""Unit tests for the pure `mentions` edge projection (Track U.2).

No DB. These assert the projection's contract directly against the U.1 findings:

* edge weight == R.2 ``entity_weight`` on the UNIFIED concept (salience × rarity);
* df==1 singletons are dropped (the 466→67 legibility step) and generic
  ``topic``/``concept`` buckets are down-weighted, not torn out;
* case/type-duplicate rows unify to one concept (so they count as shared) yet the
  edge still anchors at a REAL representative entity id;
* a fully-shared concept across the four convenanten yields a K4-style edge set
  (every source pair connected through the shared entity);
* a source with no active entities is isolated (no edges) — the 2-papers fact;
* the ``min_weight`` cutoff and the 0.3 named-only preset behave;
* regeneration is order-stable (deterministic sort) — the projection half of the
  idempotency guarantee.
"""

from __future__ import annotations

import math

from shared.retrieval.kg_source_scorer import EntityRecord, entity_weight
from shared.retrieval.mentions_projection import (
    NAMED_ONLY_MIN_WEIGHT,
    project_mentions_edges,
)

# The four convenant source ids (the K4 cluster) + the two isolated papers.
C1, C2, C3, C4 = "source:c1", "source:c2", "source:c3", "source:c4"
P1, P2 = "source:p1", "source:p2"
ALL_CONVENANTEN = [C1, C2, C3, C4]


def _ent(eid, name, etype, sources):
    return EntityRecord(
        entity_id=eid,
        canonical_name=name,
        entity_type=etype,
        source_documents=tuple(sources),
    )


# ---------------------------------------------------------------------------
# Weight == R.2 entity_weight on the unified concept
# ---------------------------------------------------------------------------
class TestWeight:
    def test_edge_weight_is_r2_entity_weight(self):
        """AC2: each edge weight == type_salience × IDF rarity (R.2 unit)."""
        # One programme shared by 2 of 6 sources.
        ents = [_ent("entity:regiodeal", "Regio Deal", "programme", [C1, C2])]
        edges, stats = project_mentions_edges(ents, n_sources=6)

        expected = entity_weight("programme", document_frequency=2, n_sources=6)
        assert edges, "expected at least one edge"
        for e in edges:
            assert math.isclose(e.weight, expected, rel_tol=1e-9)
        assert stats.edges == 2  # one edge per (source, entity) membership

    def test_named_entity_outweighs_generic_topic(self):
        """A named programme weighs strictly more than a generic topic at same df."""
        named = _ent("entity:rd", "Regio Deal", "programme", [C1, C2])
        topic = _ent("entity:bw", "brede welvaart", "topic", [C1, C2])
        edges, _ = project_mentions_edges([named, topic], n_sources=6)

        by_name = {e.concept_name: e.weight for e in edges}
        assert by_name["Regio Deal"] > by_name["brede welvaart"]

    def test_rarity_rare_entity_outweighs_corpus_wide(self):
        """A df=2 entity outweighs a df=6 entity of the SAME type (IDF rarity)."""
        rare = _ent("entity:rare", "Rare Org", "government_organization", [C1, C2])
        common = _ent(
            "entity:common",
            "Common Org",
            "government_organization",
            [C1, C2, C3, C4, P1, P2],
        )
        edges, _ = project_mentions_edges([rare, common], n_sources=6)
        by_name = {e.concept_name: e.weight for e in edges}
        assert by_name["Rare Org"] > by_name["Common Org"]


# ---------------------------------------------------------------------------
# Singleton drop + generic handling (R.6 reuse)
# ---------------------------------------------------------------------------
class TestSingletonAndGeneric:
    def test_df1_singleton_dropped(self):
        """AC4: a concept in only one source is a leaf spoke → dropped by default."""
        shared = _ent("entity:rd", "Regio Deal", "programme", [C1, C2])
        spoke = _ent("entity:solo", "Only Here", "topic", [C1])
        edges, stats = project_mentions_edges([shared, spoke], n_sources=6)

        names = {e.concept_name for e in edges}
        assert "Only Here" not in names
        assert "Regio Deal" in names
        # The two Regio Deal edges survive; the singleton contributes none.
        assert stats.edges == 2

    def test_keep_singletons_when_disabled(self):
        """drop_singletons=False keeps the df==1 spoke (measurement mode)."""
        spoke = _ent("entity:solo", "Only Here", "topic", [C1])
        edges, _ = project_mentions_edges(
            [spoke], n_sources=6, drop_singletons=False
        )
        assert {e.concept_name for e in edges} == {"Only Here"}

    def test_generic_down_weighted_not_excluded(self):
        """Generic df≥2 topics survive (down-weighted), not torn out."""
        topic = _ent("entity:bw", "brede welvaart", "topic", [C1, C2, C3])
        edges, _ = project_mentions_edges([topic], n_sources=6)
        assert len(edges) == 3
        assert all(e.concept_type == "topic" for e in edges)


# ---------------------------------------------------------------------------
# Case/type unification — count as shared, but anchor a real entity
# ---------------------------------------------------------------------------
class TestConceptUnification:
    def test_case_duplicates_unify_to_one_concept(self):
        """``Energietransitie`` / ``energietransitie`` count as ONE df=2 concept."""
        a = _ent("entity:a", "Energietransitie", "topic", [C1])
        b = _ent("entity:b", "energietransitie", "topic", [C2])
        edges, stats = project_mentions_edges([a, b], n_sources=6)

        # Unified df=2 → both memberships survive the singleton drop.
        assert stats.edges == 2
        df_values = {e.document_frequency for e in edges}
        assert df_values == {2}
        # The endpoint is one of the real rows, never a synthetic concept id.
        assert {e.entity_id for e in edges} <= {"entity:a", "entity:b"}
        assert all(not e.entity_id.startswith("concept::") for e in edges)

    def test_named_cross_type_regio_unifies(self):
        """The programme/area "Regio" split unifies (named cross-type rule)."""
        prog = _ent("entity:rp", "Regio", "programme", [C1, C2])
        area = _ent("entity:ra", "Regio", "administrative_area", [C3, C4])
        edges, stats = project_mentions_edges([prog, area], n_sources=6)
        # One unified concept spanning all four → 4 edges, df=4.
        assert stats.edges == 4
        assert {e.document_frequency for e in edges} == {4}

    def test_no_unmapped_concepts(self):
        """Every emitted concept maps back to a representative entity (defensive)."""
        ents = [
            _ent("entity:rd", "Regio Deal", "programme", [C1, C2, C3, C4]),
            _ent("entity:bw", "brede welvaart", "topic", [C1, C2]),
        ]
        _, stats = project_mentions_edges(ents, n_sources=6)
        assert stats.unmapped_concepts == 0


# ---------------------------------------------------------------------------
# The K4 convenant cluster + isolated papers
# ---------------------------------------------------------------------------
class TestK4ClusterAndIsolation:
    def _corpus(self):
        """A shared programme across all 4 convenanten; papers carry nothing."""
        return [
            _ent("entity:rd", "Regio Deal", "programme", ALL_CONVENANTEN),
            _ent("entity:regio", "Regio", "administrative_area", ALL_CONVENANTEN),
            _ent("entity:bw", "brede welvaart", "topic", [C1, C2, C3]),
            # A singleton spoke that must NOT appear.
            _ent("entity:solo", "Local Thing", "topic", [C1]),
        ]

    def test_k4_every_convenant_pair_shares_an_entity(self):
        """AC6: a document→entity→document traversal returns the convenant K4.

        With "Regio Deal" in all four sources, every one of the 6 unordered
        convenant pairs is connected through that shared entity — the K4. We
        assert it at the bipartite layer: all four sources appear as edge
        endpoints of the shared concept, so the source-projection is complete.
        """
        edges, _ = project_mentions_edges(self._corpus(), n_sources=6)

        # Sources reachable through "Regio Deal".
        rd_sources = {e.source_id for e in edges if e.concept_name == "Regio Deal"}
        assert rd_sources == set(ALL_CONVENANTEN)

        # Bipartite → all 6 unordered source pairs co-occur on the shared entity.
        shared_pairs = set()
        for name in {e.concept_name for e in edges}:
            srcs = sorted(e.source_id for e in edges if e.concept_name == name)
            for i in range(len(srcs)):
                for j in range(i + 1, len(srcs)):
                    shared_pairs.add((srcs[i], srcs[j]))
        expected_k4 = {
            (a, b)
            for i, a in enumerate(ALL_CONVENANTEN)
            for b in ALL_CONVENANTEN[i + 1 :]
        }
        assert expected_k4 <= shared_pairs

    def test_isolated_papers_have_no_edges(self):
        """AC6: a source with no active entities is isolated (the 2-papers fact)."""
        edges, _ = project_mentions_edges(self._corpus(), n_sources=6)
        endpoints = {e.source_id for e in edges}
        assert P1 not in endpoints
        assert P2 not in endpoints

    def test_singleton_spoke_excluded_from_cluster(self):
        edges, _ = project_mentions_edges(self._corpus(), n_sources=6)
        assert "Local Thing" not in {e.concept_name for e in edges}


# ---------------------------------------------------------------------------
# min_weight cutoff + named-only preset
# ---------------------------------------------------------------------------
class TestThresholds:
    def _corpus(self):
        return [
            _ent("entity:rd", "Regio Deal", "programme", ALL_CONVENANTEN),
            _ent("entity:regio", "Regio", "administrative_area", ALL_CONVENANTEN),
            _ent("entity:bw", "brede welvaart", "topic", ALL_CONVENANTEN),
            _ent("entity:gz", "gezondheid", "topic", [C1, C2, C3]),
        ]

    def test_min_weight_zero_keeps_all_df2_edges(self):
        edges, stats = project_mentions_edges(
            self._corpus(), n_sources=6, min_weight=0.0
        )
        # 4+4+4+3 = 15 edges, none filtered.
        assert stats.edges == 15
        assert stats.edges_before_min_weight == 15

    def test_named_only_preset_collapses_to_anchors(self):
        """AC: the 0.3 named-only preset keeps only the named anchors."""
        edges, stats = project_mentions_edges(
            self._corpus(), n_sources=6, named_only=True
        )
        kept = {e.concept_name for e in edges}
        # Only the two high-salience named anchors survive 0.3.
        assert kept == {"Regio Deal", "Regio"}
        assert stats.min_weight == NAMED_ONLY_MIN_WEIGHT
        # The generic topics are filtered (below 0.3) but counted pre-cut.
        assert stats.edges_before_min_weight > stats.edges

    def test_explicit_min_weight_equivalent_to_named_only(self):
        a, _ = project_mentions_edges(self._corpus(), n_sources=6, min_weight=0.3)
        b, _ = project_mentions_edges(self._corpus(), n_sources=6, named_only=True)
        assert [e.concept_name for e in a] == [e.concept_name for e in b]


# ---------------------------------------------------------------------------
# Determinism (projection half of idempotency)
# ---------------------------------------------------------------------------
class TestDeterminism:
    def test_repeated_projection_is_identical(self):
        corpus = [
            _ent("entity:rd", "Regio Deal", "programme", ALL_CONVENANTEN),
            _ent("entity:bw", "brede welvaart", "topic", [C2, C1, C3]),
            _ent("entity:gz", "gezondheid", "topic", [C3, C1]),
        ]
        first, _ = project_mentions_edges(corpus, n_sources=6)
        second, _ = project_mentions_edges(corpus, n_sources=6)
        as_tuples = lambda es: [  # noqa: E731
            (e.source_id, e.entity_id, round(e.weight, 9)) for e in es
        ]
        assert as_tuples(first) == as_tuples(second)

    def test_input_order_does_not_change_edge_set(self):
        a = _ent("entity:a", "Energietransitie", "topic", [C1])
        b = _ent("entity:b", "energietransitie", "topic", [C2])
        fwd, _ = project_mentions_edges([a, b], n_sources=6)
        rev, _ = project_mentions_edges([b, a], n_sources=6)
        # Same (source, weight, df) edge set regardless of input order; the
        # representative entity id is order-stable via the max-salience tie-break.
        as_set = lambda es: {  # noqa: E731
            (e.source_id, round(e.weight, 9), e.document_frequency) for e in es
        }
        assert as_set(fwd) == as_set(rev)


# ---------------------------------------------------------------------------
# Empty / degenerate inputs
# ---------------------------------------------------------------------------
class TestEmpty:
    def test_empty_input(self):
        edges, stats = project_mentions_edges([], n_sources=6)
        assert edges == []
        assert stats.edges == 0

    def test_entities_without_sources_skipped(self):
        ents = [_ent("entity:x", "Nameless Source Bag", "programme", [])]
        edges, stats = project_mentions_edges(ents, n_sources=6)
        assert edges == []

    def test_n_sources_inferred_when_none(self):
        ents = [_ent("entity:rd", "Regio Deal", "programme", [C1, C2])]
        edges, _ = project_mentions_edges(ents, n_sources=None)
        assert len(edges) == 2
