"""
Ontology management module for the semantic intelligence layer.

Converts the existing YAML ontologies into RDF/OWL/SHACL and provides:
- Load/export ontologies in multiple formats (Turtle, OWL/XML, JSON-LD)
- Automatic SHACL shape generation from ontology definitions
- Data validation against SHACL shapes
- SKOS vocabulary management for thema-indelingen
- Bidirectional sync between RDF and SurrealDB entity types

Usage:
    from ontology_manager.rdf_owl_shacl import (
        load_yaml_ontology, export_ontology, validate_entities,
        generate_shacl_shapes, manage_skos_vocabulary,
    )

    # Load existing YAML ontology and convert to RDF
    graph = load_yaml_ontology("packages/ontology-manager/ontologies/regiodeal.yaml")

    # Export as Turtle
    export_ontology(graph, "turtle", "/data/ontologies/regiodeal.ttl")

    # Generate SHACL shapes
    shapes = generate_shacl_shapes(graph)

    # Validate SurrealDB entities against shapes
    report = await validate_entities(shapes, entity_type="Gemeente")

    python -m ontology_manager.rdf_owl_shacl
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ontology_manager.schema import Ontology

# ---------------------------------------------------------------------------
# Optional-dependency sentinels
# ---------------------------------------------------------------------------
# rdflib is a required dependency for TTL/RDFS/OWL operations. The legacy
# code wrapped the imports in a try/except so the module could still load
# without rdflib, but module-level constants below (ON, ONR, DTYPE_MAP)
# reference Namespace / XSD which raised NameError at import time when
# rdflib was missing. We now: (a) require rdflib in pyproject.toml,
# (b) keep the sentinel pattern for defensive graceful-failure, and
# (c) guard all module-level constants behind the RDFLIB_AVAILABLE flag.
try:
    import rdflib  # noqa: F401
    from rdflib import (
        BNode,
        Graph,
        Literal,
        Namespace,
        URIRef,
    )
    from rdflib.collection import Collection
    from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

    RDFLIB_AVAILABLE = True
except ImportError:  # pragma: no cover - rdflib is a hard dep now
    RDFLIB_AVAILABLE = False
    logger.warning(
        "rdflib not installed — TTL/RDFS/OWL features disabled. "
        "Install with: uv pip install rdflib"
    )

# Backwards-compat alias for code that imported HAS_RDFLIB.
HAS_RDFLIB = RDFLIB_AVAILABLE

try:
    from pyshacl import validate as shacl_validate

    HAS_PYSHACL = True
except ImportError:
    HAS_PYSHACL = False


_RDFLIB_REQUIRED_MSG = (
    "rdflib required for TTL operations; install with: uv pip install rdflib"
)


def _require_rdflib() -> None:
    """Raise a clear ImportError if rdflib is not installed."""
    if not RDFLIB_AVAILABLE:
        raise ImportError(_RDFLIB_REQUIRED_MSG)


# ---------------------------------------------------------------------------
# Module-level constants (only defined when rdflib is available)
# ---------------------------------------------------------------------------
if RDFLIB_AVAILABLE:
    # Namespace for our ontology
    ON = Namespace("https://open-notebook.dev/ontology/")
    ONR = Namespace("https://open-notebook.dev/resource/")
    # OE.6 review m7: entity-type properties get their own namespace, so a
    # property and a relationship type of the same name are two IRIs.
    ONP = Namespace("https://open-notebook.dev/ontology/property/")

    # Data type mapping from YAML to XSD
    DTYPE_MAP = {
        "string": XSD.string,
        "text": XSD.string,
        "int": XSD.integer,
        "integer": XSD.integer,
        "float": XSD.float,
        "bool": XSD.boolean,
        "boolean": XSD.boolean,
        "datetime": XSD.dateTime,
        "date": XSD.date,
        "url": XSD.anyURI,
        "reference": None,  # Handled as object property
        "array": None,
        "json": None,
    }
else:
    # Sentinels so the module can be imported (and tests can introspect
    # RDFLIB_AVAILABLE) even without rdflib installed.
    ON = None  # type: ignore[assignment]
    ONR = None  # type: ignore[assignment]
    ONP = None  # type: ignore[assignment]
    DTYPE_MAP = {}  # type: ignore[assignment]


# ============================================================================
# Ontology model → rdflib Graph
# ============================================================================

_IRI_SAFE = frozenset("-._~")


def iri_fragment(name: str) -> str:
    """The IRI fragment for a type, relationship or property name.

    A name is kept as it is when every character is a letter, a digit or one
    of ``-._~`` (every shipped name); any other character is percent-encoded
    as UTF-8. ``%`` itself is encoded, so two different names never get the
    same fragment. OE.6 review B1: a name published through the editor or an
    accepted Pass-1 proposal ("Clinical Trial Phase", "Journal/ Publication")
    made the Turtle serialiser fail. The name itself stays in ``rdfs:label``.
    """
    return "".join(
        c if c.isalnum() or c in _IRI_SAFE
        else "".join(f"%{b:02X}" for b in c.encode("utf-8"))
        for c in name
    )


def class_iri(name: str) -> URIRef:
    """The IRI of entity type (or relationship type) ``name``."""
    return ON[iri_fragment(name)]


def property_iri(name: str) -> URIRef:
    """The IRI of entity-type property ``name``."""
    return ONP[iri_fragment(name)]


def _class_expression(g: Graph, names: List[str]) -> Optional[Any]:
    """One class for one name; the ``owl:unionOf`` of several.

    OE.6 review M2: a relationship with two endpoint types written as two
    ``rdfs:domain`` triples says its subject is BOTH (the intersection). ``"*"``
    (any type) is ``owl:Thing``, and a union containing it is ``owl:Thing``.
    """
    nodes = list(dict.fromkeys(OWL.Thing if n == "*" else class_iri(n) for n in names))
    if not nodes:
        return None
    if len(nodes) == 1:
        return nodes[0]
    if OWL.Thing in nodes:
        return OWL.Thing
    return _union(g, nodes, OWL.Class)


def _union(g: Graph, nodes: List[Any], kind: Any) -> BNode:
    union, members = BNode(), BNode()
    g.add((union, RDF.type, kind))
    Collection(g, members, nodes)
    g.add((union, OWL.unionOf, members))
    return union


def union_members(g: Graph, node: Any) -> List[Any]:
    """The classes (or datatypes) ``node`` stands for: the members of an
    ``owl:unionOf``, else ``node`` itself."""
    members = g.value(node, OWL.unionOf) if isinstance(node, BNode) else None
    return list(Collection(g, members)) if members is not None else [node]


def ontology_graph(ontology: Ontology) -> Graph:
    """Convert an ontology model to an rdflib OWL graph.

    Produces:
    - owl:Class for each entity type, rdfs:subClassOf for its parent_type
    - owl:DatatypeProperty / owl:ObjectProperty for its properties
    - owl:ObjectProperty for each relationship type, with rdfs:domain /
      rdfs:range (either spelling: domain/range or source_types/target_types)
    - rdfs:comment for descriptions, skos:note for extraction_hints

    Track OE.6: every export builds from the MODEL, so it can take what the
    registry holds (the active version, ``extends`` / ``includes`` resolved)
    rather than a YAML file. Reading the raw YAML exported no relationship type
    (it looked for a key ``relation_types`` no ontology uses) and failed on the
    mapping form of ``entity_types`` (``base``, ``general``).
    """
    _require_rdflib()

    g = Graph()
    g.bind("on", ON)
    g.bind("onr", ONR)
    g.bind("owl", OWL)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)

    meta = ontology.metadata
    ont_uri = ON[iri_fragment(meta.name)]
    g.add((ont_uri, RDF.type, OWL.Ontology))
    g.add((ont_uri, RDFS.label, Literal(meta.name)))
    if meta.description is not None:
        g.add((ont_uri, RDFS.comment, Literal(meta.description)))
    g.add((ont_uri, OWL.versionInfo, Literal(meta.version)))

    # A property name used by several types is ONE property; its domain is
    # the union of those types (as for relationships), its range the union of
    # what they declare.
    properties: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for etype in ontology.entity_types.values():
        dumped = etype.model_dump(mode="json", exclude_none=True)
        _add_entity_type(g, dumped)
        for prop in dumped["properties"]:
            if prop.get("name"):
                properties.setdefault(prop["name"], []).append((etype.name, prop))

    for name, uses in properties.items():
        _add_property(g, name, uses)

    for rtype in ontology.relationship_types.values():
        _add_relation_type(g, rtype.model_dump(mode="json", exclude_none=True))

    logger.info(
        f"Built ontology graph '{meta.name}': "
        f"{len(list(g.subjects(RDF.type, OWL.Class)))} classes, "
        f"{len(g)} triples"
    )
    return g


def load_yaml_ontology(yaml_path: str | Path) -> Graph:
    """Load ONE YAML ontology file, as written (no ``extends`` / ``includes``),
    and convert it with :func:`ontology_graph`.

    For a named ontology as the app uses it, read it through the registry
    (:func:`registry_ontology_graph`): a published edit lives in the database,
    not in the file.
    """
    return ontology_graph(Ontology.from_yaml(yaml_path))


async def registry_ontology_graph(name: str, registry: Any = None) -> Optional[Graph]:
    """The graph of ontology ``name`` as the registry holds it: the ACTIVE
    version, ``extends`` / ``includes`` resolved. ``None`` when it is unknown.

    Track OE.6. Consumers: the notebook TTL export
    (``GET /api/notebooks/{id}/schema.ttl``) and :func:`load_all_ontologies`.
    """
    from ontology_manager.registry import get_ontology_registry

    ontology = await (registry or get_ontology_registry()).get(name)
    return None if ontology is None else ontology_graph(ontology)


def _add_entity_type(g: Graph, etype: Dict[str, Any]) -> None:
    """Add an entity type as an OWL class (its properties: :func:`_add_property`)."""
    name = etype.get("name", "")
    if not name:
        return

    cls_uri = class_iri(name)
    g.add((cls_uri, RDF.type, OWL.Class))
    g.add((cls_uri, RDFS.label, Literal(name)))

    if "description" in etype:
        g.add((cls_uri, RDFS.comment, Literal(etype["description"])))

    if "parent_type" in etype:
        g.add((cls_uri, RDFS.subClassOf, class_iri(etype["parent_type"])))

    # Extraction hints as SKOS notes
    for hint in etype.get("extraction_hints", []):
        g.add((cls_uri, SKOS.note, Literal(hint)))


def _add_property(g: Graph, name: str, uses: List[Tuple[str, Dict[str, Any]]]) -> None:
    """Add property ``name``, declared by the types in ``uses``."""
    prop_uri = property_iri(name)
    g.add((prop_uri, RDFS.label, Literal(name)))

    ranges: List[Any] = []
    for _type_name, prop in uses:
        dtype = prop.get("data_type", "string")
        if dtype == "reference":
            # Object property (links to another entity)
            g.add((prop_uri, RDF.type, OWL.ObjectProperty))
            ranges.append(class_iri(prop.get("reference_type", "Thing")))
        else:
            # Datatype property
            g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
            xsd_type = DTYPE_MAP.get(dtype, XSD.string)
            if xsd_type:
                ranges.append(xsd_type)
        if "description" in prop:
            g.add((prop_uri, RDFS.comment, Literal(prop["description"])))

    g.add((prop_uri, RDFS.domain, _class_expression(g, [t for t, _ in uses])))
    ranges = list(dict.fromkeys(ranges))
    if len(ranges) == 1:
        g.add((prop_uri, RDFS.range, ranges[0]))
    elif ranges:
        datatypes = all(str(r).startswith(str(XSD)) for r in ranges)
        g.add((prop_uri, RDFS.range,
               _union(g, ranges, RDFS.Datatype if datatypes else OWL.Class)))


def _add_relation_type(g: Graph, rtype: Dict[str, Any]) -> None:
    """Add a relation type as an OWL object property."""
    name = rtype.get("name", "")
    if not name:
        return

    prop_uri = class_iri(name)
    g.add((prop_uri, RDF.type, OWL.ObjectProperty))
    g.add((prop_uri, RDFS.label, Literal(name)))

    if "description" in rtype:
        g.add((prop_uri, RDFS.comment, Literal(rtype["description"])))
    # The model derives domain / range from the source_types / target_types
    # spelling most shipped relationships use.
    domain = _class_expression(g, rtype.get("domain", []))
    if domain is not None:
        g.add((prop_uri, RDFS.domain, domain))
    range_ = _class_expression(g, rtype.get("range", []))
    if range_ is not None:
        g.add((prop_uri, RDFS.range, range_))

    for hint in rtype.get("extraction_hints", []):
        g.add((prop_uri, SKOS.note, Literal(hint)))


# ============================================================================
# All ontologies, from the registry
# ============================================================================


async def load_all_ontologies(registry: Any = None) -> Graph:
    """Every ontology the registry lists, as it holds them, in one merged graph.

    Track OE.6: read through the registry (:func:`registry_ontology_graph`), not
    the YAML directory, so a published edit is in it.

    Returns:
        Merged rdflib Graph with all ontologies.
    """
    from ontology_manager.registry import get_ontology_registry

    _require_rdflib()
    registry = registry or get_ontology_registry()
    merged = Graph()
    merged.bind("on", ON)
    merged.bind("onr", ONR)
    merged.bind("owl", OWL)
    merged.bind("skos", SKOS)

    names = await registry.list_ontologies()
    for name in names:
        try:
            g = await registry_ontology_graph(name, registry)
            if g is not None:
                merged += g
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    logger.info(f"Merged ontology: {len(merged)} triples from {len(names)} ontologies")
    return merged


# ============================================================================
# Export ontology
# ============================================================================


def export_ontology(
    graph: Graph,
    fmt: str = "turtle",
    output_path: Optional[str | Path] = None,
) -> str:
    """Export an rdflib Graph to a file or string.

    Args:
        graph: The rdflib Graph to export.
        fmt: Output format — "turtle", "xml" (OWL/XML), "json-ld", "n3".
        output_path: File path to write to. If None, returns the string.

    Returns:
        The serialized ontology as string.
    """
    _require_rdflib()
    format_map = {
        "turtle": "turtle",
        "ttl": "turtle",
        "xml": "xml",
        "owl": "xml",
        "json-ld": "json-ld",
        "jsonld": "json-ld",
        "n3": "n3",
        "nt": "nt",
    }
    rdf_format = format_map.get(fmt.lower(), "turtle")

    serialized = graph.serialize(format=rdf_format)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(serialized, encoding="utf-8")
        logger.info(f"Exported ontology to {output_path} ({rdf_format}, {len(serialized)} bytes)")

    return serialized


# ============================================================================
# SHACL shape generation
# ============================================================================


def generate_shacl_shapes(ontology_graph: Graph) -> Graph:
    """Auto-generate SHACL shapes from an OWL ontology.

    For each owl:Class, creates a sh:NodeShape with:
    - sh:targetClass pointing to the OWL class
    - sh:property for each property with domain = this class
    - Appropriate sh:datatype or sh:class constraints
    - sh:minCount for required properties

    Args:
        ontology_graph: The OWL ontology graph.

    Returns:
        rdflib Graph containing SHACL shapes.
    """
    _require_rdflib()

    SH = Namespace("http://www.w3.org/ns/shacl#")

    shapes = Graph()
    shapes.bind("sh", SH)
    shapes.bind("on", ON)
    shapes.bind("xsd", XSD)

    # The properties of each class: a domain may be a union of classes.
    props_of: Dict[Any, List[Any]] = {}
    for prop_uri, domain in ontology_graph.subject_objects(RDFS.domain):
        for cls in union_members(ontology_graph, domain):
            props_of.setdefault(cls, []).append(prop_uri)

    # For each OWL class (not the anonymous unions), create a NodeShape
    for cls_uri in ontology_graph.subjects(RDF.type, OWL.Class):
        if isinstance(cls_uri, BNode):
            continue
        cls_name = str(cls_uri).split("/")[-1]
        shape_uri = ON[f"{cls_name}Shape"]

        shapes.add((shape_uri, RDF.type, SH.NodeShape))
        shapes.add((shape_uri, SH.targetClass, cls_uri))

        for prop_uri in props_of.get(cls_uri, []):
            prop_name = str(prop_uri).split("/")[-1]
            prop_shape = BNode()

            shapes.add((shape_uri, SH.property, prop_shape))
            shapes.add((prop_shape, SH.path, prop_uri))
            shapes.add((prop_shape, SH.name, Literal(prop_name)))

            # Object property -> sh:class; datatype property -> sh:datatype. A
            # union range allows any one of its members (sh:or), not all.
            kind = (SH["class"] if (prop_uri, RDF.type, OWL.ObjectProperty) in ontology_graph
                    else SH.datatype)
            for range_node in ontology_graph.objects(prop_uri, RDFS.range):
                members = union_members(ontology_graph, range_node)
                if len(members) == 1:
                    shapes.add((prop_shape, kind, members[0]))
                    continue
                alternatives = []
                for member in members:
                    alternative = BNode()
                    shapes.add((alternative, kind, member))
                    alternatives.append(alternative)
                options = BNode()
                Collection(shapes, options, alternatives)
                shapes.add((prop_shape, SH["or"], options))

    num_shapes = len(list(shapes.subjects(RDF.type, SH.NodeShape)))
    logger.info(f"Generated {num_shapes} SHACL shapes ({len(shapes)} triples)")
    return shapes


# ============================================================================
# Data validation
# ============================================================================


async def validate_entities(
    shacl_shapes: Graph,
    entity_type: Optional[str] = None,
    limit: int = 1000,
) -> Dict[str, Any]:
    """Validate SurrealDB entities against SHACL shapes.

    Exports entities from SurrealDB as RDF, then validates against
    the provided SHACL shapes.

    Args:
        shacl_shapes: SHACL shapes graph.
        entity_type: Optional filter by entity type.
        limit: Maximum entities to validate.

    Returns:
        Validation report dict with conformance status and violations.
    """
    if not HAS_PYSHACL:
        return {"error": "pyshacl not installed"}

    from surrealdb_service.connection import execute_query

    # Export entities as RDF
    query = "SELECT * FROM entity WHERE status = 'active'"
    if entity_type:
        query += f" AND entity_type = '{entity_type}'"
    query += f" LIMIT {limit}"

    entities = await execute_query(query)

    data_graph = Graph()
    data_graph.bind("on", ON)
    data_graph.bind("onr", ONR)

    for ent in entities:
        eid = str(ent.get("id", "")).replace(":", "_")
        subj = ONR[eid]
        etype = ent.get("entity_type", "Thing")
        data_graph.add((subj, RDF.type, class_iri(etype)))
        data_graph.add((subj, ON["canonical_name"], Literal(ent.get("canonical_name", ""))))

        for key, val in ent.get("properties", {}).items():
            if isinstance(val, str):
                data_graph.add((subj, property_iri(key), Literal(val)))
            elif isinstance(val, (int, float)):
                data_graph.add((subj, property_iri(key), Literal(val)))

    # Validate
    conforms, results_graph, results_text = shacl_validate(
        data_graph,
        shacl_graph=shacl_shapes,
    )

    # Parse violations
    SH = Namespace("http://www.w3.org/ns/shacl#")
    violations = []
    for result in results_graph.subjects(RDF.type, SH.ValidationResult):
        violation = {
            "focus": str(results_graph.value(result, SH.focusNode, default="")),
            "path": str(results_graph.value(result, SH.resultPath, default="")),
            "message": str(results_graph.value(result, SH.resultMessage, default="")),
            "severity": str(results_graph.value(result, SH.resultSeverity, default="")),
        }
        violations.append(violation)

    report = {
        "conforms": conforms,
        "entities_checked": len(entities),
        "violations": violations,
        "violation_count": len(violations),
    }

    logger.info(
        f"SHACL validation: {'PASS' if conforms else 'FAIL'} "
        f"({len(violations)} violations in {len(entities)} entities)"
    )
    return report


# ============================================================================
# SKOS vocabulary management
# ============================================================================


def create_skos_scheme(
    name: str,
    concepts: List[Dict[str, Any]],
    description: str = "",
) -> Graph:
    """Create a SKOS concept scheme.

    Useful for thema-indelingen (Brede Welvaart, IV3, etc.).

    Args:
        name: Scheme name (e.g. "BredeWelvaart").
        concepts: List of concept dicts with keys:
            - name: concept label
            - broader: parent concept name (optional)
            - description: definition (optional)
            - alt_labels: alternative labels (optional)
        description: Scheme description.

    Returns:
        rdflib Graph with the SKOS concept scheme.
    """
    _require_rdflib()
    g = Graph()
    g.bind("skos", SKOS)
    g.bind("on", ON)

    scheme_uri = ON[f"scheme/{name}"]
    g.add((scheme_uri, RDF.type, SKOS.ConceptScheme))
    g.add((scheme_uri, RDFS.label, Literal(name)))
    if description:
        g.add((scheme_uri, SKOS.definition, Literal(description)))

    # Add concepts
    concept_uris: Dict[str, URIRef] = {}
    for concept in concepts:
        cname = concept.get("name", "")
        if not cname:
            continue
        concept_uri = ON[f"concept/{cname}"]
        concept_uris[cname] = concept_uri

        g.add((concept_uri, RDF.type, SKOS.Concept))
        g.add((concept_uri, SKOS.inScheme, scheme_uri))
        g.add((concept_uri, SKOS.prefLabel, Literal(cname, lang="nl")))

        if "description" in concept:
            g.add((concept_uri, SKOS.definition, Literal(concept["description"], lang="nl")))

        for alt in concept.get("alt_labels", []):
            g.add((concept_uri, SKOS.altLabel, Literal(alt, lang="nl")))

    # Add broader/narrower relations
    for concept in concepts:
        cname = concept.get("name", "")
        broader = concept.get("broader")
        if broader and broader in concept_uris and cname in concept_uris:
            g.add((concept_uris[cname], SKOS.broader, concept_uris[broader]))
            g.add((concept_uris[broader], SKOS.narrower, concept_uris[cname]))

    # Add top concepts
    for concept in concepts:
        cname = concept.get("name", "")
        if not concept.get("broader") and cname in concept_uris:
            g.add((scheme_uri, SKOS.hasTopConcept, concept_uris[cname]))

    logger.info(f"Created SKOS scheme '{name}': {len(concept_uris)} concepts")
    return g


# ============================================================================
# Built-in: Brede Welvaart SKOS vocabulary
# ============================================================================


def create_brede_welvaart_skos() -> Graph:
    """Create a SKOS vocabulary for the Brede Welvaart framework.

    Based on the CBS/PBL Brede Welvaart model used in Dutch regional policy.
    """
    concepts = [
        {"name": "BredeWelvaart", "description": "Brede Welvaart — welzijn in brede zin"},
        # Pilaren
        {"name": "MaterieleWelvaart", "broader": "BredeWelvaart",
         "description": "Inkomen, werk, armoede", "alt_labels": ["Economische welvaart"]},
        {"name": "Gezondheid", "broader": "BredeWelvaart",
         "description": "Fysieke en mentale gezondheid, levensverwachting"},
        {"name": "Arbeid", "broader": "BredeWelvaart",
         "description": "Werkgelegenheid, arbeidsparticipatie, werkloosheid"},
        {"name": "WonenLeefomgeving", "broader": "BredeWelvaart",
         "description": "Woningmarkt, leefbaarheid, voorzieningen", "alt_labels": ["Wonen en leefomgeving"]},
        {"name": "Veiligheid", "broader": "BredeWelvaart",
         "description": "Criminaliteit, veiligheidsbeleving"},
        {"name": "Onderwijs", "broader": "BredeWelvaart",
         "description": "Opleidingsniveau, schooluitval, onderwijskwaliteit"},
        {"name": "SociaalKapitaal", "broader": "BredeWelvaart",
         "description": "Vertrouwen, sociale cohesie, participatie", "alt_labels": ["Sociaal kapitaal"]},
        {"name": "Milieu", "broader": "BredeWelvaart",
         "description": "Luchtkwaliteit, natuur, biodiversiteit, klimaat"},
        {"name": "SubjectiefWelzijn", "broader": "BredeWelvaart",
         "description": "Tevredenheid, geluk, welzijnservaring", "alt_labels": ["Subjectief welzijn"]},
        # Demografische thema's
        {"name": "Vergrijzing", "broader": "BredeWelvaart",
         "description": "Veroudering bevolking, druk op voorzieningen"},
        {"name": "Ontgroening", "broader": "BredeWelvaart",
         "description": "Afname jongere bevolking, braindrain"},
        {"name": "Bevolkingskrimp", "broader": "BredeWelvaart",
         "description": "Bevolkingsdaling in regio's", "alt_labels": ["BevolkingsDaling", "Krimp"]},
        # Subdimensies
        {"name": "Bereikbaarheid", "broader": "WonenLeefomgeving",
         "description": "OV, infrastructuur, digitale bereikbaarheid"},
        {"name": "Voorzieningen", "broader": "WonenLeefomgeving",
         "description": "Scholen, winkels, zorg, sport, cultuur"},
        {"name": "Armoede", "broader": "MaterieleWelvaart",
         "description": "Huishoudens onder armoedegrens"},
    ]

    return create_skos_scheme(
        name="BredeWelvaart",
        concepts=concepts,
        description="CBS/PBL Brede Welvaart model voor Nederlands regionaal beleid",
    )


# ============================================================================
# CLI entry point
# ============================================================================


async def _demo():
    """Demo: load ontologies, export, generate SHACL, create SKOS."""
    import os

    merged = await load_all_ontologies()
    logger.info(f"Merged ontology: {len(merged)} triples")

    # Export as Turtle
    out_dir = Path(os.getenv("SHARED_DATA_DIR", ".")) / "ontologies"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_ontology(merged, "turtle", out_dir / "merged_ontology.ttl")

    # Generate SHACL shapes
    shapes = generate_shacl_shapes(merged)
    export_ontology(shapes, "turtle", out_dir / "shapes.ttl")

    # Validate entities (if any exist)
    report = await validate_entities(shapes, limit=100)
    logger.info(f"Validation: {report}")

    # Create Brede Welvaart SKOS
    bw = create_brede_welvaart_skos()
    out_dir = Path(os.getenv("SHARED_DATA_DIR", ".")) / "ontologies"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_ontology(bw, "turtle", out_dir / "brede_welvaart.ttl")
    logger.info("Brede Welvaart SKOS created")


if __name__ == "__main__":
    asyncio.run(_demo())
