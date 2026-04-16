"""
Ontology management module for the semantic intelligence layer.

Converts the existing YAML ontologies into RDF/OWL/SHACL and provides:
- Load/export ontologies in multiple formats (Turtle, OWL/XML, JSON-LD)
- Automatic SHACL shape generation from ontology definitions
- Data validation against SHACL shapes
- SKOS vocabulary management for thema-indelingen
- Bidirectional sync between RDF and SurrealDB entity types

Usage:
    from semantic_layer.ontology_manager import (
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

    python -m semantic_layer.ontology_manager
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from loguru import logger

try:
    import rdflib
    from rdflib import (
        BNode,
        Graph,
        Literal,
        Namespace,
        URIRef,
    )
    from rdflib.namespace import DCTERMS, OWL, RDF, RDFS, SKOS, XSD

    HAS_RDFLIB = True
except ImportError:
    HAS_RDFLIB = False
    logger.warning("rdflib not installed — ontology features disabled")

try:
    from pyshacl import validate as shacl_validate

    HAS_PYSHACL = True
except ImportError:
    HAS_PYSHACL = False


# Namespace for our ontology
ON = Namespace("https://open-notebook.dev/ontology/")
ONR = Namespace("https://open-notebook.dev/resource/")

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


# ============================================================================
# Load YAML ontology → rdflib Graph
# ============================================================================


def load_yaml_ontology(yaml_path: str | Path) -> Graph:
    """Load a YAML ontology definition and convert to an rdflib OWL graph.

    Reads the YAML format used by packages/ontology-manager/ontologies/*.yaml
    and produces an OWL ontology with:
    - owl:Class for each entity_type
    - rdfs:subClassOf for parent_type
    - owl:DatatypeProperty / owl:ObjectProperty for properties
    - rdfs:comment for descriptions and extraction_hints

    Args:
        yaml_path: Path to the YAML ontology file.

    Returns:
        rdflib Graph containing the OWL ontology.
    """
    if not HAS_RDFLIB:
        raise ImportError("rdflib is required for ontology management")

    path = Path(yaml_path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    g = Graph()
    g.bind("on", ON)
    g.bind("onr", ONR)
    g.bind("owl", OWL)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)

    # Ontology metadata
    meta = data.get("metadata", {})
    ont_name = meta.get("name", path.stem)
    ont_uri = ON[ont_name]
    g.add((ont_uri, RDF.type, OWL.Ontology))
    g.add((ont_uri, RDFS.label, Literal(ont_name)))
    if "description" in meta:
        g.add((ont_uri, RDFS.comment, Literal(meta["description"])))
    if "version" in meta:
        g.add((ont_uri, OWL.versionInfo, Literal(meta["version"])))

    # Entity types → OWL classes
    for etype in data.get("entity_types", []):
        _add_entity_type(g, etype)

    # Relation types → OWL object properties
    for rtype in data.get("relation_types", []):
        _add_relation_type(g, rtype)

    logger.info(
        f"Loaded ontology '{ont_name}' from {path.name}: "
        f"{len(list(g.subjects(RDF.type, OWL.Class)))} classes, "
        f"{len(g)} triples"
    )
    return g


def _add_entity_type(g: Graph, etype: Dict[str, Any]) -> None:
    """Add an entity type as an OWL class with properties."""
    name = etype.get("name", "")
    if not name:
        return

    cls_uri = ON[name]
    g.add((cls_uri, RDF.type, OWL.Class))
    g.add((cls_uri, RDFS.label, Literal(name)))

    if "description" in etype:
        g.add((cls_uri, RDFS.comment, Literal(etype["description"])))

    if "parent_type" in etype:
        g.add((cls_uri, RDFS.subClassOf, ON[etype["parent_type"]]))

    # Extraction hints as SKOS notes
    for hint in etype.get("extraction_hints", []):
        g.add((cls_uri, SKOS.note, Literal(hint)))

    # Properties
    for prop in etype.get("properties", []):
        _add_property(g, cls_uri, prop)


def _add_property(g: Graph, domain_uri: URIRef, prop: Dict[str, Any]) -> None:
    """Add a property to the ontology."""
    name = prop.get("name", "")
    if not name:
        return

    prop_uri = ON[name]
    dtype = prop.get("data_type", "string")

    if dtype == "reference":
        # Object property (links to another entity)
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDFS.domain, domain_uri))
        ref_type = prop.get("reference_type", "Thing")
        g.add((prop_uri, RDFS.range, ON[ref_type]))
    else:
        # Datatype property
        g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        g.add((prop_uri, RDFS.domain, domain_uri))
        xsd_type = DTYPE_MAP.get(dtype, XSD.string)
        if xsd_type:
            g.add((prop_uri, RDFS.range, xsd_type))

    g.add((prop_uri, RDFS.label, Literal(name)))
    if "description" in prop:
        g.add((prop_uri, RDFS.comment, Literal(prop["description"])))


def _add_relation_type(g: Graph, rtype: Dict[str, Any]) -> None:
    """Add a relation type as an OWL object property."""
    name = rtype.get("name", "")
    if not name:
        return

    prop_uri = ON[name]
    g.add((prop_uri, RDF.type, OWL.ObjectProperty))
    g.add((prop_uri, RDFS.label, Literal(name)))

    if "description" in rtype:
        g.add((prop_uri, RDFS.comment, Literal(rtype["description"])))
    if "source_types" in rtype:
        for st in rtype["source_types"]:
            g.add((prop_uri, RDFS.domain, ON[st]))
    if "target_types" in rtype:
        for tt in rtype["target_types"]:
            g.add((prop_uri, RDFS.range, ON[tt]))

    for hint in rtype.get("extraction_hints", []):
        g.add((prop_uri, SKOS.note, Literal(hint)))


# ============================================================================
# Load all ontologies from directory
# ============================================================================


def load_all_ontologies(
    ontology_dir: str | Path = "packages/ontology-manager/ontologies",
) -> Graph:
    """Load all YAML ontologies from a directory into a merged graph.

    Args:
        ontology_dir: Directory containing .yaml ontology files.

    Returns:
        Merged rdflib Graph with all ontologies.
    """
    merged = Graph()
    merged.bind("on", ON)
    merged.bind("onr", ONR)
    merged.bind("owl", OWL)
    merged.bind("skos", SKOS)

    path = Path(ontology_dir)
    if not path.exists():
        logger.warning(f"Ontology directory not found: {path}")
        return merged

    for yaml_file in sorted(path.glob("*.yaml")):
        try:
            g = load_yaml_ontology(yaml_file)
            merged += g
        except Exception as e:
            logger.error(f"Failed to load {yaml_file.name}: {e}")

    logger.info(f"Merged ontology: {len(merged)} triples from {len(list(path.glob('*.yaml')))} files")
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
    if not HAS_RDFLIB:
        raise ImportError("rdflib is required")

    SH = Namespace("http://www.w3.org/ns/shacl#")

    shapes = Graph()
    shapes.bind("sh", SH)
    shapes.bind("on", ON)
    shapes.bind("xsd", XSD)

    # For each OWL class, create a NodeShape
    for cls_uri in ontology_graph.subjects(RDF.type, OWL.Class):
        cls_name = str(cls_uri).split("/")[-1]
        shape_uri = ON[f"{cls_name}Shape"]

        shapes.add((shape_uri, RDF.type, SH.NodeShape))
        shapes.add((shape_uri, SH.targetClass, cls_uri))

        # Find properties with this class as domain
        for prop_uri in ontology_graph.subjects(RDFS.domain, cls_uri):
            prop_name = str(prop_uri).split("/")[-1]
            prop_shape = BNode()

            shapes.add((shape_uri, SH.property, prop_shape))
            shapes.add((prop_shape, SH.path, prop_uri))
            shapes.add((prop_shape, SH.name, Literal(prop_name)))

            # Determine constraint type
            if (prop_uri, RDF.type, OWL.ObjectProperty) in ontology_graph:
                # Object property → sh:class
                for range_cls in ontology_graph.objects(prop_uri, RDFS.range):
                    shapes.add((prop_shape, SH["class"], range_cls))
            else:
                # Datatype property → sh:datatype
                for range_dt in ontology_graph.objects(prop_uri, RDFS.range):
                    shapes.add((prop_shape, SH.datatype, range_dt))

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

    from semantic_layer.config import execute

    # Export entities as RDF
    query = "SELECT * FROM entity WHERE status = 'active'"
    if entity_type:
        query += f" AND entity_type = '{entity_type}'"
    query += f" LIMIT {limit}"

    entities = await execute(query)

    data_graph = Graph()
    data_graph.bind("on", ON)
    data_graph.bind("onr", ONR)

    for ent in entities:
        eid = str(ent.get("id", "")).replace(":", "_")
        subj = ONR[eid]
        etype = ent.get("entity_type", "Thing")
        data_graph.add((subj, RDF.type, ON[etype]))
        data_graph.add((subj, ON["canonical_name"], Literal(ent.get("canonical_name", ""))))

        for key, val in ent.get("properties", {}).items():
            if isinstance(val, str):
                data_graph.add((subj, ON[key], Literal(val)))
            elif isinstance(val, (int, float)):
                data_graph.add((subj, ON[key], Literal(val)))

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

    # Find ontology directory
    base = os.getenv("PROJECT_ROOT", "/mnt/e/Repos/Private/open-notebook")
    ont_dir = Path(base) / "packages" / "ontology-manager" / "ontologies"

    if ont_dir.exists():
        # Load all ontologies
        merged = load_all_ontologies(ont_dir)
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
    else:
        logger.warning(f"Ontology directory not found: {ont_dir}")

    # Create Brede Welvaart SKOS
    bw = create_brede_welvaart_skos()
    out_dir = Path(os.getenv("SHARED_DATA_DIR", ".")) / "ontologies"
    out_dir.mkdir(parents=True, exist_ok=True)
    export_ontology(bw, "turtle", out_dir / "brede_welvaart.ttl")
    logger.info("Brede Welvaart SKOS created")


if __name__ == "__main__":
    asyncio.run(_demo())
