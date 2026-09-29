"""OE.6 review round 1: IRI-safe names and union endpoints in the OWL export
and its SHACL shapes (``ontology_manager.rdf_owl_shacl``).
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from urllib.parse import unquote

import pytest
from ontology_manager.rdf_owl_shacl import (
    ON,
    class_iri,
    generate_shacl_shapes,
    iri_fragment,
    ontology_graph,
    property_iri,
)
from ontology_manager.schema import Ontology
from rdflib import BNode, Graph, Literal, Namespace
from rdflib.collection import Collection
from rdflib.compare import isomorphic
from rdflib.namespace import OWL, RDF, RDFS

SH = Namespace("http://www.w3.org/ns/shacl#")
SHIPPED = sorted((Path(__file__).resolve().parents[1] / "ontologies").glob("*.yaml"))

#: Names that collapse under the pre-OE.5 CamelCase rule, or under a naive
#: encoding that leaves ``%`` alone.
TRICKY = ["Clinical Trial", "Clinical_Trial", "ClinicalTrial", "Clinical%20Trial",
          "Clinical-Trial", "Clinical.Trial", "Clinical/Trial", "Clinical Trial ",
          "clinical trial", "Überweisung", "Uberweisung", "Journal/ Publication",
          "PUBLISHED IN", "PUBLISHED_IN", "2024 cohort", "a<b>", 'quote"d']


def _shipped_names():
    for path in SHIPPED:
        ontology = Ontology.from_yaml(path)
        yield ontology.metadata.name
        for et in ontology.entity_types.values():
            yield et.name
            yield from (p.name for p in et.properties)
            if et.parent_type:
                yield et.parent_type
        for rt in ontology.relationship_types.values():
            yield rt.name
            yield from rt.domain
            yield from rt.range


def test_every_shipped_name_keeps_its_iri():
    """The shipped IRIs do not change: every shipped name is its own fragment."""
    changed = sorted({n for n in _shipped_names() if n != "*" and iri_fragment(n) != n})
    assert not changed


@pytest.mark.parametrize("name", TRICKY)
def test_a_fragment_decodes_back_to_its_name(name):
    """Decodable, hence no two names share a fragment."""
    assert unquote(iri_fragment(name)) == name


def test_no_two_tricky_names_share_a_fragment():
    for a, b in combinations(TRICKY, 2):
        assert iri_fragment(a) != iri_fragment(b), (a, b)


def _ontology(entity_types, relationship_types) -> Ontology:
    return Ontology(metadata={"name": "t", "version": "1"},
                    entity_types=entity_types, relationship_types=relationship_types)


def test_names_that_are_not_iri_safe_serialise_with_their_labels():
    types = [{"name": n, "properties": [{"name": f"{n} prop", "data_type": "string"}]}
             for n in TRICKY]
    types[1]["parent_type"] = TRICKY[0]
    graph = ontology_graph(_ontology(types, [
        {"name": "LINKS / TO", "domain": [TRICKY[0]], "range": [TRICKY[3]]}]))
    reparsed = Graph().parse(data=graph.serialize(format="turtle"), format="turtle")
    assert isomorphic(graph, reparsed)
    for name in TRICKY:
        assert (class_iri(name), RDFS.label, Literal(name)) in reparsed
        assert (property_iri(f"{name} prop"), RDFS.label, Literal(f"{name} prop")) in reparsed
    assert (class_iri(TRICKY[1]), RDFS.subClassOf, class_iri(TRICKY[0])) in reparsed
    assert (class_iri("LINKS / TO"), RDFS.label, Literal("LINKS / TO")) in reparsed


def test_a_property_and_a_relationship_of_one_name_are_two_iris():
    graph = ontology_graph(_ontology(
        [{"name": "A", "properties": [{"name": "author", "data_type": "string"}]}],
        [{"name": "author", "domain": ["A"], "range": ["A"]}]))
    assert (property_iri("author"), RDF.type, OWL.DatatypeProperty) in graph
    assert (class_iri("author"), RDF.type, OWL.ObjectProperty) in graph
    assert (class_iri("author"), RDF.type, OWL.DatatypeProperty) not in graph


def _union(graph, node):
    assert isinstance(node, BNode)
    return set(Collection(graph, graph.value(node, OWL.unionOf)))


def test_several_endpoint_types_are_one_union_and_one_is_itself():
    graph = ontology_graph(_ontology(
        [{"name": n} for n in "ABC"],
        [{"name": "R", "domain": ["A", "B"], "range": ["C"]},
         {"name": "ANY", "source_types": ["*"], "target_types": ["A", "*"]}]))
    [domain] = graph.objects(ON["R"], RDFS.domain)
    assert _union(graph, domain) == {ON["A"], ON["B"]}
    assert list(graph.objects(ON["R"], RDFS.range)) == [ON["C"]]
    assert list(graph.objects(ON["ANY"], RDFS.domain)) == [OWL.Thing]
    assert list(graph.objects(ON["ANY"], RDFS.range)) == [OWL.Thing]


def test_a_property_several_types_declare_has_their_union_as_domain():
    graph = ontology_graph(_ontology(
        [{"name": "A", "properties": [{"name": "p", "data_type": "reference",
                                       "reference_type": "B"}]},
         {"name": "B", "properties": [{"name": "p", "data_type": "reference",
                                       "reference_type": "C"}]},
         {"name": "C"}], []))
    [domain] = graph.objects(property_iri("p"), RDFS.domain)
    [range_] = graph.objects(property_iri("p"), RDFS.range)
    assert _union(graph, domain) == {ON["A"], ON["B"]}
    assert _union(graph, range_) == {ON["B"], ON["C"]}


def _property_shapes(shapes, cls, path):
    shape = ON[f"{cls}Shape"]
    return [ps for ps in shapes.objects(shape, SH.property)
            if (ps, SH.path, path) in shapes]


def test_shacl_allows_any_one_of_several_ranges():
    graph = ontology_graph(_ontology(
        [{"name": n} for n in "ABC"],
        [{"name": "R", "domain": ["A", "B"], "range": ["B", "C"]},
         {"name": "S", "domain": ["A"], "range": ["C"]}]))
    shapes = generate_shacl_shapes(graph)
    # One shape per named class; the union is no class of its own.
    assert set(shapes.subjects(RDF.type, SH.NodeShape)) == {ON["AShape"], ON["BShape"], ON["CShape"]}
    for cls in "AB":
        [ps] = _property_shapes(shapes, cls, ON["R"])
        assert list(shapes.objects(ps, SH["class"])) == []
        [options] = shapes.objects(ps, SH["or"])
        assert {shapes.value(o, SH["class"]) for o in Collection(shapes, options)} == {ON["B"], ON["C"]}
    [ps] = _property_shapes(shapes, "A", ON["S"])
    assert list(shapes.objects(ps, SH["class"])) == [ON["C"]]
    assert _property_shapes(shapes, "C", ON["R"]) == []
