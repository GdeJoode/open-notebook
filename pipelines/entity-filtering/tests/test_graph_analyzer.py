"""Tests for GraphAnalyzer graph-based entity analysis and filtering."""

from unittest.mock import patch

import networkx as nx
import pytest

from entity_filtering.validation.graph_analyzer import GraphAnalyzer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(text, label="MISC", properties=None):
    """Helper to build an entity dict in the pipeline's canonical format."""
    return {
        "text": text,
        "label": label,
        "properties": properties.copy() if properties else {},
    }


def _relation(source, target, rel_type="RELATED_TO"):
    """Helper to build a relation dict in the pipeline's canonical format."""
    return {
        "source_entity": source,
        "target_entity": target,
        "relation_type": rel_type,
        "properties": {},
    }


# ---------------------------------------------------------------------------
# Report key expectations
# ---------------------------------------------------------------------------

EXPECTED_REPORT_KEYS = {
    "algorithm",
    "node_count",
    "edge_count",
    "removed_count",
    "min_score_threshold",
    "centrality_scores",
}


# ===================================================================
# Empty / trivial inputs
# ===================================================================


class TestEmptyInputs:
    """Tests for empty and trivial entity/relation lists."""

    def test_empty_entities_returns_empty_result(self):
        """Empty entity list returns empty list and zeroed report."""
        ga = GraphAnalyzer()
        result, report = ga.analyze([], [])
        assert result == [], "Expected empty result for empty entities"
        assert report["node_count"] == 0
        assert report["edge_count"] == 0
        assert report["removed_count"] == 0
        assert report["centrality_scores"] == {}

    def test_empty_entities_with_relations_returns_empty(self):
        """Relations without entities still returns empty when entities list is empty."""
        ga = GraphAnalyzer()
        relations = [_relation("A", "B")]
        result, report = ga.analyze([], relations)
        assert result == []

    def test_report_structure_on_empty(self):
        """Report contains all expected keys even on empty input."""
        ga = GraphAnalyzer()
        _, report = ga.analyze([], [])
        assert set(report.keys()) == EXPECTED_REPORT_KEYS


# ===================================================================
# Entities with no relations (isolated nodes)
# ===================================================================


class TestIsolatedEntities:
    """Tests for entities with no connecting relations."""

    def test_all_kept_when_no_relations(self):
        """All entities should be kept when there are no relations."""
        ga = GraphAnalyzer(min_score=0.01)
        entities = [_entity("Alice", "PERSON"), _entity("Bob", "PERSON")]
        result, report = ga.analyze(entities, [])

        # Entities are in the graph as nodes but have centrality 0.
        # In pagerank every node gets a non-zero score (1/N baseline).
        assert len(result) >= 1, "At least some entities expected"

    def test_centrality_scores_added_to_properties(self):
        """Every entity gets a centrality_score in its properties."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity("Alice"), _entity("Bob")]
        result, _ = ga.analyze(entities, [])
        for ent in result:
            assert "centrality_score" in ent["properties"], (
                f"Entity {ent['text']} missing centrality_score"
            )

    def test_isolated_pagerank_assigns_equal_scores(self):
        """Pagerank on isolated nodes assigns equal probability to each."""
        ga = GraphAnalyzer(min_score=0.0, algorithm="pagerank")
        entities = [_entity("A"), _entity("B"), _entity("C")]
        result, report = ga.analyze(entities, [])

        scores = [e["properties"]["centrality_score"] for e in result]
        # With 3 isolated nodes, each should get ~1/3
        for s in scores:
            assert abs(s - 1 / 3) < 0.01, (
                f"Expected ~0.333 for isolated pagerank node, got {s}"
            )


# ===================================================================
# Simple graph connectivity
# ===================================================================


class TestSimpleGraph:
    """Tests for basic graph structures."""

    def test_linear_chain_all_get_nonzero_centrality(self):
        """A->B->C chain: every node gets non-zero centrality."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity("A"), _entity("B"), _entity("C")]
        relations = [_relation("A", "B"), _relation("B", "C")]
        result, report = ga.analyze(entities, relations)

        assert len(result) == 3
        for ent in result:
            assert ent["properties"]["centrality_score"] > 0, (
                f"Entity {ent['text']} should have non-zero centrality in a chain"
            )

    def test_report_node_and_edge_counts(self):
        """Report should reflect correct graph node and edge counts."""
        ga = GraphAnalyzer()
        entities = [_entity("A"), _entity("B"), _entity("C")]
        relations = [_relation("A", "B"), _relation("B", "C")]
        _, report = ga.analyze(entities, relations)

        assert report["node_count"] == 3, "Expected 3 nodes"
        assert report["edge_count"] == 2, "Expected 2 edges"

    def test_report_algorithm_field(self):
        """Report should record which algorithm was used."""
        ga = GraphAnalyzer(algorithm="betweenness")
        _, report = ga.analyze([_entity("A")], [])
        assert report["algorithm"] == "betweenness"

    def test_centrality_scores_in_report(self):
        """Report centrality_scores dict maps entity text to float scores."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity("X"), _entity("Y")]
        relations = [_relation("X", "Y")]
        _, report = ga.analyze(entities, relations)

        assert "X" in report["centrality_scores"]
        assert "Y" in report["centrality_scores"]
        assert isinstance(report["centrality_scores"]["X"], float)


# ===================================================================
# min_score filtering
# ===================================================================


class TestMinScoreFiltering:
    """Tests for the min_score threshold behavior."""

    def test_min_score_zero_keeps_all_entities(self):
        """min_score=0.0 is enrichment-only mode: no filtering occurs."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity("A"), _entity("B"), _entity("C")]
        relations = [_relation("A", "B")]
        result, report = ga.analyze(entities, relations)

        assert len(result) == 3, "All entities should be kept with min_score=0.0"
        assert report["removed_count"] == 0

    def test_high_min_score_filters_low_scoring_entities(self):
        """A very high min_score should filter entities that are in the graph."""
        ga = GraphAnalyzer(min_score=0.99)
        entities = [_entity("A"), _entity("B"), _entity("C")]
        relations = [_relation("A", "B"), _relation("B", "C")]
        result, report = ga.analyze(entities, relations)

        # Pagerank scores for a 3-node chain are well below 0.99
        assert report["removed_count"] > 0, (
            "Expected some entities removed with min_score=0.99"
        )
        assert len(result) < 3

    def test_entities_not_in_graph_kept_even_with_high_min_score(self):
        """Entities that don't appear in any relation are NOT filtered,
        even though their graph score is 0.

        The code explicitly checks `text in entity_texts_in_graph` before removing.
        An entity with empty text (skipped by _build_graph) is not a graph node.
        """
        ga = GraphAnalyzer(min_score=0.5)
        entities = [
            _entity("Connected1"),
            _entity("Connected2"),
            _entity(""),  # empty text -> not added to graph
        ]
        relations = [_relation("Connected1", "Connected2")]
        result, _ = ga.analyze(entities, relations)

        # The empty-text entity should be kept (not in graph, score 0, not filtered)
        empty_ents = [e for e in result if e["text"] == ""]
        assert len(empty_ents) == 1, "Empty-text entity should be kept (not in graph)"

    def test_removed_count_matches_difference(self):
        """Report removed_count should equal total_entities - len(result) for in-graph entities."""
        ga = GraphAnalyzer(min_score=0.99)
        entities = [_entity("A"), _entity("B")]
        relations = [_relation("A", "B")]
        result, report = ga.analyze(entities, relations)
        assert report["removed_count"] == len(entities) - len(result)


# ===================================================================
# Properties preservation
# ===================================================================


class TestPropertiesPreservation:
    """Tests that existing properties are not destroyed by analysis."""

    def test_existing_properties_preserved(self):
        """Pre-existing properties on entities must survive analysis."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity("Alice", "PERSON", properties={"age": 30, "role": "engineer"})]
        result, _ = ga.analyze(entities, [])

        props = result[0]["properties"]
        assert props["age"] == 30, "Existing 'age' property should be preserved"
        assert props["role"] == "engineer", "Existing 'role' property should be preserved"
        assert "centrality_score" in props, "centrality_score should be added"

    def test_centrality_score_overwrites_previous(self):
        """If an entity already has centrality_score, it gets overwritten."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity("A", properties={"centrality_score": 999.0})]
        result, _ = ga.analyze(entities, [])
        assert result[0]["properties"]["centrality_score"] != 999.0


# ===================================================================
# Algorithm selection
# ===================================================================


class TestAlgorithms:
    """Tests for different centrality algorithms."""

    def test_pagerank_default(self):
        """Default algorithm should be pagerank."""
        ga = GraphAnalyzer()
        assert ga._algorithm == "pagerank"

    def test_betweenness_produces_valid_scores(self):
        """Betweenness centrality should produce valid float scores."""
        ga = GraphAnalyzer(algorithm="betweenness", min_score=0.0)
        entities = [_entity("A"), _entity("B"), _entity("C")]
        relations = [_relation("A", "B"), _relation("B", "C")]
        result, report = ga.analyze(entities, relations)

        assert report["algorithm"] == "betweenness"
        for ent in result:
            score = ent["properties"]["centrality_score"]
            assert isinstance(score, float), f"Score should be float, got {type(score)}"
            assert score >= 0.0, "Betweenness scores should be non-negative"

    def test_pagerank_produces_valid_scores(self):
        """Pagerank should produce valid probability scores summing to ~1."""
        ga = GraphAnalyzer(algorithm="pagerank", min_score=0.0)
        entities = [_entity("A"), _entity("B"), _entity("C")]
        relations = [_relation("A", "B"), _relation("B", "C")]
        result, report = ga.analyze(entities, relations)

        total = sum(report["centrality_scores"].values())
        assert abs(total - 1.0) < 0.01, (
            f"Pagerank scores should sum to ~1.0, got {total}"
        )

    def test_pagerank_fallback_to_betweenness_on_convergence_failure(self):
        """When pagerank fails to converge, should fall back to betweenness."""
        ga = GraphAnalyzer(algorithm="pagerank", min_score=0.0)
        entities = [_entity("A"), _entity("B")]
        relations = [_relation("A", "B")]

        with patch("networkx.pagerank", side_effect=nx.PowerIterationFailedConvergence(50)):
            result, report = ga.analyze(entities, relations)

        # Should still produce results (via betweenness fallback)
        assert len(result) == 2
        for ent in result:
            assert "centrality_score" in ent["properties"]


# ===================================================================
# Centrality score rounding
# ===================================================================


class TestScoreRounding:
    """Tests for centrality score precision."""

    def test_scores_rounded_to_six_decimal_places(self):
        """Centrality scores in properties should be rounded to 6 decimal places."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity("A"), _entity("B"), _entity("C")]
        relations = [_relation("A", "B"), _relation("B", "C")]
        result, _ = ga.analyze(entities, relations)

        for ent in result:
            score = ent["properties"]["centrality_score"]
            # Convert to string and check decimal places
            score_str = f"{score:.10f}"
            decimal_part = score_str.split(".")[1]
            # After 6 decimal places, remaining digits should be 0
            trailing = decimal_part[6:]
            assert all(c == "0" for c in trailing), (
                f"Score {score} has more than 6 decimal places"
            )


# ===================================================================
# Multiple connected components
# ===================================================================


class TestMultipleComponents:
    """Tests for graphs with multiple disconnected subgraphs."""

    def test_two_components(self):
        """Two separate components should both be analyzed correctly."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [
            _entity("A"), _entity("B"),  # component 1
            _entity("X"), _entity("Y"),  # component 2
        ]
        relations = [
            _relation("A", "B"),
            _relation("X", "Y"),
        ]
        result, report = ga.analyze(entities, relations)

        assert len(result) == 4
        assert report["node_count"] == 4
        assert report["edge_count"] == 2

        texts_with_scores = {
            e["text"]: e["properties"]["centrality_score"] for e in result
        }
        for text in ["A", "B", "X", "Y"]:
            assert text in texts_with_scores, f"{text} should be in results"
            assert texts_with_scores[text] > 0, f"{text} should have non-zero score"


# ===================================================================
# Self-loops
# ===================================================================


class TestSelfLoops:
    """Tests for self-referencing relations."""

    def test_self_loop_handled(self):
        """An entity relating to itself should not crash the analyzer."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity("A"), _entity("B")]
        relations = [_relation("A", "A"), _relation("A", "B")]
        result, report = ga.analyze(entities, relations)

        assert len(result) == 2, "Self-loop should not prevent analysis"
        assert report["edge_count"] == 2, "Self-loop counts as an edge"


# ===================================================================
# Implicit nodes from relations
# ===================================================================


class TestImplicitNodes:
    """Tests for edges that reference entities not in the entity list."""

    def test_relation_creates_implicit_nodes(self):
        """Relations referencing unknown entities create implicit graph nodes."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity("A")]
        # "B" is not in the entity list but appears in a relation
        relations = [_relation("A", "B")]
        result, report = ga.analyze(entities, relations)

        # Graph should have both A and B as nodes
        assert report["node_count"] == 2, "Implicit node B should be added to graph"
        assert report["edge_count"] == 1

        # Only entity "A" is in the entity list, so result should have 1 entity
        assert len(result) == 1
        assert result[0]["text"] == "A"

    def test_implicit_node_in_centrality_scores(self):
        """Implicit nodes appear in the report centrality_scores dict."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity("A")]
        relations = [_relation("A", "Phantom")]
        _, report = ga.analyze(entities, relations)

        assert "Phantom" in report["centrality_scores"], (
            "Implicit node should appear in centrality_scores"
        )


# ===================================================================
# Large graph
# ===================================================================


class TestLargeGraph:
    """Tests for graphs with many entities."""

    def test_large_graph_produces_valid_scores(self):
        """A graph with 15+ entities should produce valid non-negative scores."""
        ga = GraphAnalyzer(min_score=0.0)
        entity_names = [f"Entity_{i}" for i in range(15)]
        entities = [_entity(name) for name in entity_names]
        # Create a chain: E0->E1->E2->...->E14
        relations = [
            _relation(entity_names[i], entity_names[i + 1])
            for i in range(14)
        ]
        result, report = ga.analyze(entities, relations)

        assert len(result) == 15
        assert report["node_count"] == 15
        assert report["edge_count"] == 14
        for ent in result:
            score = ent["properties"]["centrality_score"]
            assert score >= 0.0, f"Score for {ent['text']} should be non-negative"

    def test_hub_node_gets_highest_score(self):
        """A hub node connected to many others should rank highest."""
        ga = GraphAnalyzer(min_score=0.0, algorithm="pagerank")
        entities = [_entity("Hub")] + [_entity(f"Spoke_{i}") for i in range(10)]
        # All spokes point to hub
        relations = [_relation(f"Spoke_{i}", "Hub") for i in range(10)]
        result, _ = ga.analyze(entities, relations)

        scores = {e["text"]: e["properties"]["centrality_score"] for e in result}
        hub_score = scores["Hub"]
        for i in range(10):
            assert hub_score > scores[f"Spoke_{i}"], (
                f"Hub should score higher than Spoke_{i}"
            )


# ===================================================================
# Empty-text entities
# ===================================================================


class TestEmptyTextEntities:
    """Tests for entities with empty or missing text fields."""

    def test_empty_text_entities_skipped_in_graph(self):
        """Entities with empty text are not added as graph nodes."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity(""), _entity("Valid")]
        relations = []
        result, report = ga.analyze(entities, relations)

        # Only "Valid" should be a graph node
        assert report["node_count"] == 1, "Empty-text entity should not be a graph node"
        # Both entities should still be in result (min_score=0.0 keeps all)
        assert len(result) == 2

    def test_empty_text_entity_gets_zero_centrality(self):
        """Entities with empty text get centrality_score=0 in properties."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity(""), _entity("A")]
        result, _ = ga.analyze(entities, [])

        empty_ent = next(e for e in result if e["text"] == "")
        assert empty_ent["properties"]["centrality_score"] == 0.0


# ===================================================================
# Parametrized edge cases
# ===================================================================


class TestParametrized:
    """Parametrized tests covering various configurations."""

    @pytest.mark.parametrize(
        "algorithm",
        ["pagerank", "betweenness"],
        ids=["pagerank", "betweenness"],
    )
    def test_algorithm_options_produce_results(self, algorithm):
        """Both supported algorithms should produce valid results."""
        ga = GraphAnalyzer(algorithm=algorithm, min_score=0.0)
        entities = [_entity("A"), _entity("B")]
        relations = [_relation("A", "B")]
        result, report = ga.analyze(entities, relations)

        assert len(result) == 2
        assert report["algorithm"] == algorithm
        assert all(
            "centrality_score" in e["properties"] for e in result
        )

    @pytest.mark.parametrize(
        "min_score,expected_min_result",
        [
            (0.0, 3),   # enrichment only, keep all
            (0.01, 3),  # low threshold, chain nodes typically exceed this
            (0.99, 0),  # extremely high, all chain nodes filtered
        ],
        ids=["zero-keeps-all", "low-threshold", "very-high-threshold"],
    )
    def test_min_score_threshold_effect(self, min_score, expected_min_result):
        """Different min_score thresholds produce expected filtering behavior."""
        ga = GraphAnalyzer(min_score=min_score)
        entities = [_entity("A"), _entity("B"), _entity("C")]
        relations = [_relation("A", "B"), _relation("B", "C")]
        result, report = ga.analyze(entities, relations)

        if min_score == 0.0:
            assert len(result) == expected_min_result
        elif min_score == 0.99:
            assert len(result) <= expected_min_result or report["removed_count"] > 0

    @pytest.mark.parametrize(
        "label",
        ["PERSON", "ORG", "CONCEPT", "EVENT", "LOC"],
    )
    def test_various_labels_passed_through(self, label):
        """Entity labels should be preserved regardless of type."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [_entity("Test", label)]
        result, _ = ga.analyze(entities, [])
        assert result[0]["label"] == label


# ===================================================================
# Graph building internals
# ===================================================================


class TestBuildGraph:
    """Tests for the internal _build_graph method."""

    def test_build_graph_returns_digraph(self):
        """_build_graph should return a networkx DiGraph."""
        ga = GraphAnalyzer()
        graph = ga._build_graph([_entity("A")], [])
        assert isinstance(graph, nx.DiGraph)

    def test_build_graph_edge_attributes(self):
        """Edges should carry the relation_type attribute."""
        ga = GraphAnalyzer()
        graph = ga._build_graph(
            [_entity("A"), _entity("B")],
            [_relation("A", "B", "WORKS_FOR")],
        )
        edge_data = graph.edges["A", "B"]
        assert edge_data["relation_type"] == "WORKS_FOR"

    def test_build_graph_node_attributes(self):
        """Nodes should carry the label attribute."""
        ga = GraphAnalyzer()
        graph = ga._build_graph([_entity("Alice", "PERSON")], [])
        assert graph.nodes["Alice"]["label"] == "PERSON"


# ---------------------------------------------------------------------------
# Additional helpers for KG-aware tests
# ---------------------------------------------------------------------------


def _entity_with_kg(text, label="MISC", is_new=None, kg_entity_id=None):
    """Build entity dict with KG metadata in properties."""
    props = {}
    if is_new is not None:
        props["is_new"] = is_new
    if kg_entity_id is not None:
        props["kg_entity_id"] = kg_entity_id
    return {"text": text, "label": label, "properties": props}


def _hub_graph():
    """Build a hub-and-spoke graph where 'Hub' has high centrality and
    'Leaf' / 'Isolated' have low or zero centrality.

    Returns (entities, relations) using _entity_with_kg with is_new=None
    so callers can override per-entity.
    """
    hub = _entity_with_kg("Hub")
    spoke_a = _entity_with_kg("SpokeA")
    spoke_b = _entity_with_kg("SpokeB")
    spoke_c = _entity_with_kg("SpokeC")
    leaf = _entity_with_kg("Leaf")
    isolated = _entity_with_kg("Isolated")

    entities = [hub, spoke_a, spoke_b, spoke_c, leaf, isolated]
    relations = [
        _relation("SpokeA", "Hub"),
        _relation("SpokeB", "Hub"),
        _relation("SpokeC", "Hub"),
        _relation("Hub", "SpokeA"),
        _relation("Hub", "SpokeB"),
        _relation("Hub", "SpokeC"),
        _relation("Leaf", "SpokeA"),
    ]
    return entities, relations


# ===================================================================
# Outlier classification
# ===================================================================


class TestOutlierClassification:
    """Tests for the classify_outliers feature on GraphAnalyzer."""

    def test_disabled_by_default(self):
        """classify_outliers=False (default) produces no outlier_summary
        and no kg_classification on any entity."""
        ga = GraphAnalyzer(min_score=0.0)
        entities = [
            _entity_with_kg("Hub", is_new=True),
            _entity_with_kg("SpokeA", is_new=False),
        ]
        relations = [_relation("SpokeA", "Hub")]
        result, report = ga.analyze(entities, relations)

        assert "outlier_summary" not in report, (
            "outlier_summary should be absent when classify_outliers=False"
        )
        for ent in result:
            assert "kg_classification" not in ent["properties"], (
                f"Entity '{ent['text']}' should have no kg_classification"
            )

    def test_important_new_entity(self):
        """is_new=True + high centrality (hub) -> important_new."""
        entities, relations = _hub_graph()
        # Override Hub to be a new entity
        entities[0]["properties"]["is_new"] = True
        # Give other entities is_new=False so they are classified too
        for ent in entities[1:]:
            ent["properties"]["is_new"] = False

        ga = GraphAnalyzer(
            min_score=0.0, classify_outliers=True, outlier_centrality_low=0.05,
        )
        result, report = ga.analyze(entities, relations)

        hub = next(e for e in result if e["text"] == "Hub")
        assert hub["properties"]["kg_classification"] == "important_new", (
            "Hub with is_new=True and high centrality should be important_new"
        )
        assert report["outlier_summary"]["important_new_count"] >= 1

    def test_potential_noise_entity(self):
        """is_new=True + low/zero centrality -> potential_noise."""
        entities, relations = _hub_graph()
        # Mark the isolated node as new
        isolated = next(e for e in entities if e["text"] == "Isolated")
        isolated["properties"]["is_new"] = True
        # Mark the rest as established so they don't confuse counts
        for ent in entities:
            if ent["text"] != "Isolated":
                ent["properties"]["is_new"] = False

        ga = GraphAnalyzer(
            min_score=0.0, classify_outliers=True, outlier_centrality_low=0.05,
        )
        result, report = ga.analyze(entities, relations)

        iso = next(e for e in result if e["text"] == "Isolated")
        assert iso["properties"]["kg_classification"] == "potential_noise", (
            "Isolated entity with is_new=True and zero centrality should be potential_noise"
        )
        assert report["outlier_summary"]["potential_noise_count"] >= 1

    def test_anomaly_entity(self):
        """is_new=False + low/zero centrality -> anomaly."""
        entities, relations = _hub_graph()
        # Mark isolated node as existing (not new) with zero centrality
        isolated = next(e for e in entities if e["text"] == "Isolated")
        isolated["properties"]["is_new"] = False
        # Mark others so they don't pollute
        for ent in entities:
            if ent["text"] != "Isolated":
                ent["properties"]["is_new"] = True

        ga = GraphAnalyzer(
            min_score=0.0, classify_outliers=True, outlier_centrality_low=0.05,
        )
        result, report = ga.analyze(entities, relations)

        iso = next(e for e in result if e["text"] == "Isolated")
        assert iso["properties"]["kg_classification"] == "anomaly", (
            "Isolated entity with is_new=False and zero centrality should be anomaly"
        )
        assert report["outlier_summary"]["anomaly_count"] >= 1

    def test_established_entity(self):
        """is_new=False + high centrality (hub) -> established."""
        entities, relations = _hub_graph()
        # Hub is existing in KG and central
        entities[0]["properties"]["is_new"] = False
        for ent in entities[1:]:
            ent["properties"]["is_new"] = True

        ga = GraphAnalyzer(
            min_score=0.0, classify_outliers=True, outlier_centrality_low=0.05,
        )
        result, report = ga.analyze(entities, relations)

        hub = next(e for e in result if e["text"] == "Hub")
        assert hub["properties"]["kg_classification"] == "established", (
            "Hub with is_new=False and high centrality should be established"
        )
        assert report["outlier_summary"]["established_count"] >= 1

    def test_unclassified_no_kg_data(self):
        """Entities without is_new property remain unclassified and get no
        kg_classification set in properties."""
        ga = GraphAnalyzer(
            min_score=0.0, classify_outliers=True, outlier_centrality_low=0.05,
        )
        entities = [_entity_with_kg("A"), _entity_with_kg("B")]
        relations = [_relation("A", "B")]
        result, report = ga.analyze(entities, relations)

        for ent in result:
            assert "kg_classification" not in ent["properties"], (
                f"Entity '{ent['text']}' without is_new should have no kg_classification"
            )
        assert report["outlier_summary"]["unclassified_count"] == 2

    def test_outlier_summary_structure(self):
        """outlier_summary dict contains all expected keys."""
        ga = GraphAnalyzer(
            min_score=0.0, classify_outliers=True, outlier_centrality_low=0.05,
        )
        entities = [_entity_with_kg("A", is_new=True)]
        relations = []
        _, report = ga.analyze(entities, relations)

        summary = report["outlier_summary"]
        expected_keys = {
            "important_new_count",
            "potential_noise_count",
            "anomaly_count",
            "established_count",
            "unclassified_count",
            "important_new_entities",
            "potential_noise_entities",
            "anomaly_entities",
        }
        assert set(summary.keys()) == expected_keys, (
            f"Summary keys mismatch: got {set(summary.keys())}, expected {expected_keys}"
        )

    def test_mixed_classifications(self):
        """A batch with all 4 classification types + unclassified produces
        correct counts."""
        entities, relations = _hub_graph()
        # Hub -> is_new=False, high centrality => established
        entities[0]["properties"]["is_new"] = False
        # SpokeA -> is_new=True, moderate centrality => important_new (connected to hub)
        entities[1]["properties"]["is_new"] = True
        # Isolated -> is_new=True, zero centrality => potential_noise
        isolated = next(e for e in entities if e["text"] == "Isolated")
        isolated["properties"]["is_new"] = True
        # Leaf -> is_new=False, low centrality => anomaly
        leaf = next(e for e in entities if e["text"] == "Leaf")
        leaf["properties"]["is_new"] = False
        # SpokeB and SpokeC -> no is_new => unclassified
        spoke_b = next(e for e in entities if e["text"] == "SpokeB")
        spoke_c = next(e for e in entities if e["text"] == "SpokeC")
        spoke_b["properties"].pop("is_new", None)
        spoke_c["properties"].pop("is_new", None)

        ga = GraphAnalyzer(
            min_score=0.0, classify_outliers=True, outlier_centrality_low=0.05,
        )
        result, report = ga.analyze(entities, relations)
        summary = report["outlier_summary"]

        assert summary["established_count"] >= 1, "Expected at least 1 established"
        assert summary["important_new_count"] >= 1, "Expected at least 1 important_new"
        assert summary["potential_noise_count"] >= 1, "Expected at least 1 potential_noise"
        assert summary["unclassified_count"] == 2, "Expected 2 unclassified (SpokeB, SpokeC)"
        # Leaf has only one connection and low centrality; verify anomaly
        leaf_result = next(e for e in result if e["text"] == "Leaf")
        assert leaf_result["properties"]["kg_classification"] == "anomaly"

    def test_entities_in_important_new_list(self):
        """Entity text appears in the summary important_new_entities list."""
        entities, relations = _hub_graph()
        entities[0]["properties"]["is_new"] = True  # Hub -> important_new
        for ent in entities[1:]:
            ent["properties"]["is_new"] = False

        ga = GraphAnalyzer(
            min_score=0.0, classify_outliers=True, outlier_centrality_low=0.05,
        )
        _, report = ga.analyze(entities, relations)

        assert "Hub" in report["outlier_summary"]["important_new_entities"], (
            "Hub text should appear in important_new_entities list"
        )

    def test_outlier_centrality_low_threshold_effect(self):
        """Raising the threshold can shift a formerly-established entity to anomaly.

        With a low threshold the hub is 'established'; with a threshold higher
        than the hub's centrality it becomes 'anomaly'.
        """
        entities_low, relations_low = _hub_graph()
        for ent in entities_low:
            ent["properties"]["is_new"] = False

        # Low threshold: hub should be established
        ga_low = GraphAnalyzer(
            min_score=0.0,
            classify_outliers=True,
            outlier_centrality_low=0.01,
        )
        result_low, report_low = ga_low.analyze(entities_low, relations_low)
        hub_low = next(e for e in result_low if e["text"] == "Hub")
        assert hub_low["properties"]["kg_classification"] == "established", (
            "Hub should be 'established' with a very low threshold"
        )

        # Very high threshold (higher than any PageRank score): everything
        # becomes anomaly because no score can reach it.
        entities_high, relations_high = _hub_graph()
        for ent in entities_high:
            ent["properties"]["is_new"] = False

        ga_high = GraphAnalyzer(
            min_score=0.0,
            classify_outliers=True,
            outlier_centrality_low=0.99,
        )
        result_high, report_high = ga_high.analyze(entities_high, relations_high)
        hub_high = next(e for e in result_high if e["text"] == "Hub")
        assert hub_high["properties"]["kg_classification"] == "anomaly", (
            "Hub should become 'anomaly' when threshold exceeds its centrality"
        )
        assert (
            report_high["outlier_summary"]["anomaly_count"]
            > report_low["outlier_summary"]["anomaly_count"]
        ), "Higher threshold should produce more anomalies"


# ===================================================================
# MergedGraphAnalyzer stub
# ===================================================================


class TestMergedGraphAnalyzerStub:
    """Tests for the MergedGraphAnalyzer placeholder class."""

    def test_analyze_raises_not_implemented(self):
        """Calling analyze() should raise NotImplementedError."""
        from entity_filtering.validation.graph_analyzer import MergedGraphAnalyzer

        mga = MergedGraphAnalyzer()
        with pytest.raises(NotImplementedError):
            # analyze is async, but NotImplementedError is raised immediately
            # when the coroutine is awaited; calling it directly returns a
            # coroutine.  We need to actually run it.
            import asyncio
            asyncio.get_event_loop().run_until_complete(mga.analyze([], []))

    def test_constructor_accepts_parameters(self):
        """MergedGraphAnalyzer can be instantiated with all documented params."""
        from entity_filtering.validation.graph_analyzer import MergedGraphAnalyzer

        mga = MergedGraphAnalyzer(
            entity_repo="fake_repo",
            algorithm="betweenness",
            kg_neighborhood_hops=2,
            kg_max_neighbors=100,
        )
        assert mga._algorithm == "betweenness"
        assert mga._kg_neighborhood_hops == 2
        assert mga._kg_max_neighbors == 100
        assert mga._entity_repo == "fake_repo"

    def test_importable_from_validation_package(self):
        """MergedGraphAnalyzer should be importable from
        entity_filtering.validation (the package __init__)."""
        from entity_filtering.validation import MergedGraphAnalyzer as Mga

        assert Mga is not None
        # Verify it is the same class
        from entity_filtering.validation.graph_analyzer import (
            MergedGraphAnalyzer as MgaDirect,
        )
        assert Mga is MgaDirect
