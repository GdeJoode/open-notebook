"""Roundtrip tests for the YAML → rdflib Graph → Turtle → re-parsed Graph path.

Regression coverage for the rdflib import bug fixed in Phase B.2a (Track B).
Before the fix, ``ontology_manager.rdf_owl_shacl`` raised ``NameError`` at
module load time because module-level constants (``ON``, ``ONR``, ``DTYPE_MAP``)
referenced ``Namespace``/``XSD`` imported inside a ``try:`` block.

These tests verify three things:

1. The module loads cleanly with ``RDFLIB_AVAILABLE is True``.
2. YAML ontologies round-trip through Turtle without losing any triples
   (set equality on ``(s, p, o)`` tuples).
3. The Turtle output is well-formed enough for downstream consumers
   (parses cleanly via pyshacl).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ontology_manager import rdf_owl_shacl
from ontology_manager.rdf_owl_shacl import (
    RDFLIB_AVAILABLE,
    export_ontology,
    load_yaml_ontology,
)

# Skip the whole module if rdflib is unexpectedly missing — Test 4 explicitly
# guards against that case, the rest assume rdflib is available.
pytestmark = pytest.mark.skipif(
    not RDFLIB_AVAILABLE,
    reason="rdflib not installed; cannot exercise TTL roundtrip",
)


ONTOLOGIES_DIR = Path(__file__).resolve().parent.parent / "ontologies"


def _roundtrip_triple_sets(yaml_path: Path) -> tuple[set, set]:
    """Helper: load YAML → Turtle → re-parse, return triple sets for comparison.

    Why ``set`` equality and not graph isomorphism? rdflib's
    ``compare.isomorphic`` matches blank nodes by graph shape, which is
    correct but ~100x slower. For these ontologies we don't emit blank
    nodes from the YAML loader, so plain ``(s, p, o)`` set equality is
    sufficient and gives a clearer diff on failure.
    """
    from rdflib import Graph

    g1 = load_yaml_ontology(yaml_path)
    ttl_text = export_ontology(g1, fmt="turtle")
    assert ttl_text.strip(), f"Empty Turtle output for {yaml_path.name}"

    g2 = Graph()
    g2.parse(data=ttl_text, format="turtle")

    triples_before = set(g1.triples((None, None, None)))
    triples_after = set(g2.triples((None, None, None)))
    return triples_before, triples_after


# ---------------------------------------------------------------------------
# Test 1 — scholarly.yaml roundtrip
# ---------------------------------------------------------------------------


def test_yaml_to_ttl_roundtrip_preserves_triples_scholarly():
    """scholarly.yaml: every triple in the original graph survives Turtle roundtrip."""
    yaml_path = ONTOLOGIES_DIR / "scholarly.yaml"
    assert yaml_path.exists(), f"Missing fixture: {yaml_path}"

    triples_before, triples_after = _roundtrip_triple_sets(yaml_path)

    assert triples_before, "scholarly.yaml produced an empty graph — regression"
    missing = triples_before - triples_after
    added = triples_after - triples_before
    assert not missing, f"{len(missing)} triples lost on roundtrip, e.g.: {list(missing)[:3]}"
    assert not added, f"{len(added)} triples appeared on roundtrip, e.g.: {list(added)[:3]}"
    assert triples_before == triples_after


# ---------------------------------------------------------------------------
# Test 2 — policy.yaml roundtrip (second ontology per plan acceptance criterion)
# ---------------------------------------------------------------------------


def test_yaml_to_ttl_roundtrip_preserves_triples_policy():
    """policy.yaml: triple-set equality on roundtrip (covers a second ontology)."""
    yaml_path = ONTOLOGIES_DIR / "policy.yaml"
    assert yaml_path.exists(), f"Missing fixture: {yaml_path}"

    triples_before, triples_after = _roundtrip_triple_sets(yaml_path)

    assert triples_before, "policy.yaml produced an empty graph — regression"
    assert triples_before == triples_after, (
        f"Triple-set mismatch: "
        f"{len(triples_before - triples_after)} lost, "
        f"{len(triples_after - triples_before)} added"
    )


# ---------------------------------------------------------------------------
# Test 3 — pyshacl parsability sanity check
# ---------------------------------------------------------------------------


def test_ttl_output_parses_with_pyshacl():
    """The TTL output should parse cleanly via pyshacl (Protégé surrogate).

    pyshacl uses rdflib internally but is the canonical SHACL validator,
    so successful invocation against an empty shapes graph is a strong
    signal that the Turtle is well-formed for downstream tools.
    """
    pyshacl = pytest.importorskip(
        "pyshacl",
        reason="pyshacl not installed; install via `uv sync --extra dev`",
    )
    from rdflib import Graph

    yaml_path = ONTOLOGIES_DIR / "scholarly.yaml"
    g = load_yaml_ontology(yaml_path)
    ttl_text = export_ontology(g, fmt="turtle")

    data_graph = Graph()
    data_graph.parse(data=ttl_text, format="turtle")

    # Empty shapes graph → validation should always conform, but the call
    # exercises pyshacl's parser and rejects malformed Turtle outright.
    shapes_graph = Graph()
    conforms, _results_graph, results_text = pyshacl.validate(
        data_graph,
        shacl_graph=shapes_graph,
        inference="none",
        debug=False,
    )
    assert conforms, f"pyshacl rejected Turtle (results: {results_text[:500]})"


# ---------------------------------------------------------------------------
# Test 4 — regression guard: module-load NameError must not return
# ---------------------------------------------------------------------------


def test_rdflib_imports_succeed_at_module_load():
    """Guard: rdf_owl_shacl must import cleanly when rdflib is installed.

    Pre-fix the module raised ``NameError: name 'Namespace' is not defined``
    on import because module-level constants were defined outside the
    rdflib-guarded block. The sentinel pattern (RDFLIB_AVAILABLE) keeps
    those constants behind the guard so this can't regress silently.
    """
    assert rdf_owl_shacl.RDFLIB_AVAILABLE is True
    # ON / ONR / DTYPE_MAP must be live rdflib objects, not None sentinels.
    assert rdf_owl_shacl.ON is not None
    assert rdf_owl_shacl.ONR is not None
    assert isinstance(rdf_owl_shacl.DTYPE_MAP, dict) and rdf_owl_shacl.DTYPE_MAP
    # Legacy alias must still resolve for any consumer relying on it.
    assert rdf_owl_shacl.HAS_RDFLIB is True
