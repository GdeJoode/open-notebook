"""OE.6: the TTL export and the extraction service read the ontologies through
the registry, against a real SurrealDB (the unit-test conftest keeps the
registry on the YAML files; undone here, as in ``test_ontology_edit_service``).

- With no publish, over EVERY shipped ontology: the TTL export keeps every
  triple of the pre-OE.6 export and adds the ``extends`` / ``includes`` chain
  and the relationship types; the extraction prompt is the pre-OE.6 prompt.
  "Pre-OE.6" is the old readers, kept below verbatim as oracles over the YAML.
- After a publish (a new type, a new relationship type, a rename in
  ``scholarly``, which ``extends`` ``schema_core``; a new type in
  ``schema_core``), both show it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.connection as conn
import yaml
from app_main.api.routers import schemas as schemas_router
from app_main.api.routers import services_proxy
from app_main.dependencies import get_notebook_schema_repo, get_notebook_service
from app_main.services.ontology_edit_service import OntologyEditService
from fastapi import FastAPI
from ontology_manager.rdf_owl_shacl import (
    DCTERMS,
    DTYPE_MAP,
    ON,
    ONP,
    ONR,
    OWL,
    SKOS,
    XSD,
    class_iri,
    property_iri,
    union_members,
)
from ontology_manager.registry import OntologyRegistry, get_ontology_registry
from ontology_manager.schema import Ontology
from rdflib import BNode, Graph, Literal, URIRef
from rdflib.compare import isomorphic
from rdflib.namespace import RDF, RDFS
from shared.models.notebook_schema import NotebookSchema

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

_REPO = Path(__file__).resolve().parents[3]
ONTOLOGY_DIR = _REPO / "packages" / "ontology-manager" / "ontologies"
SHIPPED = sorted(p.stem for p in ONTOLOGY_DIR.glob("*.yaml"))
EXTRACTION_SERVICE = _REPO / "services" / "extraction"

_LOAD_FROM_DB = OntologyRegistry._load_from_db
_LIST_ONTOLOGIES = OntologyRegistry.list_ontologies


def _raw(name: str) -> Dict[str, Any]:
    return yaml.safe_load((ONTOLOGY_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The pre-OE.6 readers, verbatim (only the file location is a parameter)
# ---------------------------------------------------------------------------


def _todays_ttl(yaml_path: Path) -> Graph:
    """``rdf_owl_shacl.load_yaml_ontology`` before OE.6."""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    g = Graph()
    g.bind("on", ON)
    g.bind("onr", ONR)
    g.bind("owl", OWL)
    g.bind("skos", SKOS)
    g.bind("dcterms", DCTERMS)

    meta = data.get("metadata", {})
    ont_name = meta.get("name", yaml_path.stem)
    ont_uri = ON[ont_name]
    g.add((ont_uri, RDF.type, OWL.Ontology))
    g.add((ont_uri, RDFS.label, Literal(ont_name)))
    if "description" in meta:
        g.add((ont_uri, RDFS.comment, Literal(meta["description"])))
    if "version" in meta:
        g.add((ont_uri, OWL.versionInfo, Literal(meta["version"])))

    for etype in data.get("entity_types", []):
        _add_entity_type(g, etype)

    for rtype in data.get("relation_types", []):
        raise AssertionError("no shipped ontology has `relation_types`")
    return g


def _add_entity_type(g: Graph, etype: Dict[str, Any]) -> None:
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

    for hint in etype.get("extraction_hints", []):
        g.add((cls_uri, SKOS.note, Literal(hint)))

    for prop in etype.get("properties", []):
        _add_property(g, cls_uri, prop)


def _add_property(g: Graph, domain_uri: URIRef, prop: Dict[str, Any]) -> None:
    name = prop.get("name", "")
    if not name:
        return

    prop_uri = ON[name]
    dtype = prop.get("data_type", "string")

    if dtype == "reference":
        g.add((prop_uri, RDF.type, OWL.ObjectProperty))
        g.add((prop_uri, RDFS.domain, domain_uri))
        ref_type = prop.get("reference_type", "Thing")
        g.add((prop_uri, RDFS.range, ON[ref_type]))
    else:
        g.add((prop_uri, RDF.type, OWL.DatatypeProperty))
        g.add((prop_uri, RDFS.domain, domain_uri))
        xsd_type = DTYPE_MAP.get(dtype, XSD.string)
        if xsd_type:
            g.add((prop_uri, RDFS.range, xsd_type))

    g.add((prop_uri, RDFS.label, Literal(name)))
    if "description" in prop:
        g.add((prop_uri, RDFS.comment, Literal(prop["description"])))


def _missing_from(exported: Graph, today: Graph) -> list:
    """Today's triples the export does not carry, read the way OE.6 review
    round 1 changed them: a property is in its own namespace (``ONP``), and
    the types that declare it are one union domain, so a type of today's
    domain (or a range of today's) must be a MEMBER of the export's one."""
    properties = set(today.subjects(RDF.type, OWL.DatatypeProperty)) | set(
        today.subjects(RDF.type, OWL.ObjectProperty))
    moved = {p: ONP[str(p)[len(str(ON)):]] for p in properties}
    missing = []
    for s_, p_, o_ in today:
        s_ = moved.get(s_, s_)
        if p_ in (RDFS.domain, RDFS.range) and s_ in moved.values():
            if not any(o_ in union_members(exported, node) for node in exported.objects(s_, p_)):
                missing.append((s_, p_, o_))
        elif (s_, p_, o_) not in exported:
            missing.append((s_, p_, o_))
    return missing


def _todays_prompt(name: str, load=_raw) -> str:
    """``services/extraction/api.py`` ``_load_ontology_prompt`` before OE.6, in a
    fresh process (its caches start empty). ``load`` stands for reading
    ``ONTOLOGY_DIR / f"{name}.yaml"``; a missing file returned ``{}``."""
    _ontology_raw_cache: Dict[str, Dict] = {}

    def _load_ontology_yaml(name: str, _visited: Optional[set] = None) -> Dict:
        if name in _ontology_raw_cache:
            return _ontology_raw_cache[name]

        if _visited is None:
            _visited = set()
        if name in _visited:
            return {}
        _visited.add(name)

        if not (ONTOLOGY_DIR / f"{name}.yaml").exists():
            return {}
        ont = load(name) or {}

        merged_entity_types = list(ont.get("entity_types", []))
        merged_relation_types = list(ont.get("relationship_types", []))
        merged_extraction_rules = list(ont.get("extraction_rules", []))

        extends = ont.get("metadata", {}).get("extends")
        if extends and isinstance(extends, str):
            parent = _load_ontology_yaml(extends, _visited)
            merged_entity_types = list(parent.get("entity_types", [])) + merged_entity_types
            merged_relation_types = list(parent.get("relationship_types", [])) + merged_relation_types
            merged_extraction_rules = list(parent.get("extraction_rules", [])) + merged_extraction_rules

        includes = ont.get("metadata", {}).get("includes", [])
        if isinstance(includes, list):
            for inc_name in includes:
                inc = _load_ontology_yaml(inc_name, _visited)
                merged_entity_types.extend(inc.get("entity_types", []))
                merged_relation_types.extend(inc.get("relationship_types", []))
                merged_extraction_rules.extend(inc.get("extraction_rules", []))

        seen_entities = {}
        for et in merged_entity_types:
            seen_entities[et.get("name", "")] = et
        seen_relations = {}
        for rt in merged_relation_types:
            seen_relations[rt.get("name", "")] = rt

        result = {
            "metadata": ont.get("metadata", {}),
            "entity_types": list(seen_entities.values()),
            "relationship_types": list(seen_relations.values()),
            "extraction_rules": list(dict.fromkeys(merged_extraction_rules)),
            "concepts": ont.get("concepts", []),
            "claim_types": ont.get("claim_types", []),
        }

        _ontology_raw_cache[name] = result
        return result

    ont = _load_ontology_yaml(name)
    if not ont:
        return ""

    parts = ["ENTITY TYPES:"]
    for et in ont.get("entity_types", []):
        n = et.get("name", "")
        d = et.get("description", "")
        gr = et.get("graph_relevance", "active")
        hints = et.get("extraction_hints", [])
        if et.get("abstract"):
            continue
        line = f"- {n}: {d}"
        if gr == "contextual":
            line += " [contextual — extract but not for knowledge graph]"
        if hints:
            line += f" (look for: {'; '.join(hints[:2])})"
        parts.append(line)

    parts.append("\nRELATION TYPES:")
    for rt in ont.get("relationship_types", []):
        n = rt.get("name", "")
        d = rt.get("description", "")
        parts.append(f"- {n}: {d}")

    rules = ont.get("extraction_rules", [])
    if rules:
        parts.append("\nEXTRACTION RULES:")
        for rule in rules:
            parts.append(f"- {rule}")

    return "\n".join(parts)


def _as_lists(name: str) -> Dict[str, Any]:
    """The YAML with its mapping-form collections written as lists."""
    data = _raw(name)
    for key in ("entity_types", "relationship_types"):
        if isinstance(data.get(key), dict):
            data[key] = list(data[key].values())
    return data


# Derived, not listed: the old readers iterated `entity_types` as a list, so
# an ontology whose chain writes it as a mapping failed in both.
def _chain(name: str, seen: Optional[set] = None) -> list:
    seen = set() if seen is None else seen
    if name in seen:
        return []
    seen.add(name)
    meta = _raw(name).get("metadata", {})
    out = [name]
    for other in ([meta["extends"]] if meta.get("extends") else []) + (meta.get("includes") or []):
        out += _chain(other, seen)
    return out


def _node(endpoint: str):
    return OWL.Thing if endpoint == "*" else ON[endpoint]


MAPPING_FORM = {n for n in SHIPPED if isinstance(_raw(n).get("entity_types"), dict)}
TODAY_FAILS = {n for n in SHIPPED if any(c in MAPPING_FORM for c in _chain(n))}


# ---------------------------------------------------------------------------
# Fixtures and the two consumers, through their HTTP surfaces
# ---------------------------------------------------------------------------


@pytest.fixture
async def svc(live_surrealdb, monkeypatch):
    monkeypatch.setattr(OntologyRegistry, "_load_from_db", _LOAD_FROM_DB)
    monkeypatch.setattr(OntologyRegistry, "list_ontologies", _LIST_ONTOLOGIES)
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    registry = get_ontology_registry()
    registry.clear_cache()
    await conn.execute_query("DELETE ontology_draft; DELETE ontology_version;", {})
    yield OntologyEditService()
    await conn.execute_query("DELETE ontology_draft; DELETE ontology_version;", {})
    registry.clear_cache()
    await conn._pool.close_all()
    conn._pool = original


async def _ttl(name: str) -> Graph:
    """``GET /api/notebooks/{id}/schema.ttl`` for a notebook whose base is ``name``."""
    notebooks = AsyncMock()
    notebooks.get.return_value = object()
    schemas = AsyncMock()
    schemas.get_by_notebook.return_value = NotebookSchema(
        notebook="notebook:oe6", base_ontology=name)
    app = FastAPI()
    app.include_router(schemas_router.router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebooks
    app.dependency_overrides[get_notebook_schema_repo] = lambda: schemas
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        response = await client.get("/api/notebooks/notebook:oe6/schema.ttl")
    assert response.status_code == 200, response.text
    return Graph().parse(data=response.text, format="turtle")


def _extraction_service():
    """``services/extraction/api.py`` as its container runs it (its directory
    on the path for ``record_ids``)."""
    sys.path.insert(0, str(EXTRACTION_SERVICE))
    try:
        spec = importlib.util.spec_from_file_location("_oe6_extraction_api",
                                                      EXTRACTION_SERVICE / "api.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(EXTRACTION_SERVICE))


_SERVICE = _extraction_service()


async def _extraction_prompt(name: Optional[str], monkeypatch) -> str:
    """``POST /api/services/extract`` → the body the extraction service
    receives → the prompt it builds from it.

    The double stands in for the network hop only: what the proxy posts is
    parsed by the service's own ``ExtractRequest`` (a missing ``ontologies`` is
    a 422 there), and the prompt is built by the service's own code, with its
    own default for a request that names no ontology.
    """
    sent = {}

    async def _post(url, path, body=None, params=None):
        sent.update(path=path, body=body)
        return {}

    monkeypatch.setattr(services_proxy, "_proxy_post", _post)
    app = FastAPI()
    app.include_router(services_proxy.router, prefix="/api")
    request: Dict[str, Any] = {"file_path": "/data/doc.md"}
    if name:
        request["override"] = {"ontology": name}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        assert (await client.post("/api/services/extract", json=request)).status_code == 200
    assert sent["path"] == "/extract"
    received = _SERVICE.ExtractRequest.model_validate_json(httpx.Request(
        "POST", "http://x", json=sent["body"]).content)
    chosen = (received.override.ontology if received.override and received.override.ontology
              else _SERVICE.DEFAULT_ONTOLOGY)
    return _SERVICE._ontology_prompt(chosen, received.ontologies)


async def _store_every_seed():
    """What the first publish of each ontology does: store the YAML as the
    active version. The registry then reads the database, not the file."""
    for name in SHIPPED:
        assert await get_ontology_registry().register(
            Ontology.from_yaml(ONTOLOGY_DIR / f"{name}.yaml"), activate=True,
            note="YAML seed", created_by="test")


# ---------------------------------------------------------------------------
# No publish: today's output, over every shipped ontology
# ---------------------------------------------------------------------------


@pytest.mark.filterwarnings("ignore:The test <Function")
def test_the_space_is_every_shipped_ontology():
    assert len(SHIPPED) == 11
    assert MAPPING_FORM == {"base", "general"} and TODAY_FAILS == {"base", "general"}


@pytest.mark.parametrize("name", SHIPPED)
async def test_with_no_publish_the_ttl_export_keeps_todays_and_adds_the_chain(svc, name):
    exported = await _ttl(name)

    if name in TODAY_FAILS:
        with pytest.raises(AttributeError):
            _todays_ttl(ONTOLOGY_DIR / f"{name}.yaml")
    else:
        missing = _missing_from(exported, _todays_ttl(ONTOLOGY_DIR / f"{name}.yaml"))
        assert not missing, sorted(missing)[:5]

    # Every type and relationship type of the chain, from the raw YAML (both
    # collection forms, both endpoint spellings). A relationship is checked in
    # the definition that wins: the ontology's own, then its `extends` chain,
    # then its `includes` (the order `_chain` walks).
    winning: Dict[str, tuple] = {}
    for member in _chain(name):
        raw = _raw(member)
        types = raw.get("entity_types") or []
        for et in types.values() if isinstance(types, dict) else types:
            assert (ON[et["name"]], RDF.type, OWL.Class) in exported, (member, et["name"])
        rels = raw.get("relationship_types") or []
        for rt in rels.values() if isinstance(rels, dict) else rels:
            winning.setdefault(rt["name"], (member, rt))
    for rel_name, (member, rt) in winning.items():
        prop = ON[rel_name]
        assert (prop, RDF.type, OWL.ObjectProperty) in exported, (member, rel_name)
        for predicate, names in ((RDFS.domain, rt.get("domain") or rt.get("source_types")),
                                 (RDFS.range, rt.get("range") or rt.get("target_types"))):
            assert _endpoint(exported, prop, predicate) == _expected_endpoint(names), (
                member, rel_name, predicate)

    # Storing the seeds (what a first publish does) changes nothing.
    await _store_every_seed()
    assert isomorphic(await _ttl(name), exported)


def _expected_endpoint(names) -> tuple:
    """(is a union, the classes) a relationship's endpoint list means: one
    class, ``owl:Thing`` for any, or the UNION of several (OE.6 review M2)."""
    nodes = set(_node(n) for n in names or [])
    if OWL.Thing in nodes:
        return (False, {OWL.Thing})
    return (len(nodes) > 1, nodes)


def _endpoint(exported: Graph, prop, predicate) -> tuple:
    objects = list(exported.objects(prop, predicate))
    if not objects:
        return (False, set())
    assert len(objects) == 1, f"{prop} has {len(objects)} {predicate}: their intersection"
    node = objects[0]
    return (isinstance(node, BNode), set(union_members(exported, node)))


@pytest.mark.filterwarnings("ignore:The test <Function")
def test_many_shipped_relationships_have_several_endpoint_types():
    """Vacuity guard for the union form above: the space has them."""
    several = {rt["name"] for n in SHIPPED
               for rt in (lambda r: r.values() if isinstance(r, dict) else r)(
                   _raw(n).get("relationship_types") or [])
               if any(len(rt.get(k) or []) > 1
                      for k in ("domain", "range", "source_types", "target_types"))}
    assert len(several) > 30


@pytest.mark.parametrize("name", SHIPPED)
async def test_with_no_publish_the_extraction_prompt_is_todays(svc, monkeypatch, name):
    prompt = await _extraction_prompt(name, monkeypatch)
    if name in TODAY_FAILS:
        with pytest.raises(AttributeError):
            _todays_prompt(name)
        # What the old reader gives for the same content in list form.
        assert prompt == _todays_prompt(name, load=_as_lists)
    else:
        assert prompt == _todays_prompt(name)

    await _store_every_seed()
    assert await _extraction_prompt(name, monkeypatch) == prompt


async def test_a_request_naming_no_ontology_gets_the_service_default(svc, monkeypatch):
    assert _SERVICE.DEFAULT_ONTOLOGY in SHIPPED
    assert await _extraction_prompt(None, monkeypatch) == _todays_prompt(_SERVICE.DEFAULT_ONTOLOGY)


# ---------------------------------------------------------------------------
# After a publish
# ---------------------------------------------------------------------------


async def _publish_scholarly_edits(svc):
    """A new type, a new relationship type in each endpoint spelling, and a
    rename (whose old name the shipped relationships use as ``source_types``)."""
    await svc.put_entity_type("scholarly", "Preregistration", {
        "description": "A study protocol registered before the data is collected.",
        "parent_type": "ScholarlyArticle",
        "extraction_hints": ["preregistered at", "OSF registration"],
    })
    await svc.put_relationship_type("scholarly", "PREREGISTERED_AS", {
        "description": "Article whose study was preregistered as this protocol",
        "source_types": ["ScholarlyArticle"], "target_types": ["Preregistration"],
    })
    await svc.put_relationship_type("scholarly", "REGISTERS", {
        "description": "Protocol registering the study of an article",
        "domain": ["Preregistration"], "range": ["ScholarlyArticle"],
    })
    await svc.rename_entity_type("scholarly", "Thesis", "Dissertation")
    await svc.publish("scholarly", note="OE.6 test")


async def test_a_publish_reaches_the_ttl_export(svc):
    before = await _ttl("scholarly")
    await _publish_scholarly_edits(svc)
    after = await _ttl("scholarly")

    assert (ON["Preregistration"], RDF.type, OWL.Class) in after
    assert (ON["Preregistration"], RDFS.subClassOf, ON["ScholarlyArticle"]) in after
    for rel, domain, range_ in (("PREREGISTERED_AS", "ScholarlyArticle", "Preregistration"),
                                ("REGISTERS", "Preregistration", "ScholarlyArticle")):
        assert (ON[rel], RDF.type, OWL.ObjectProperty) in after
        assert (ON[rel], RDFS.domain, ON[domain]) in after
        assert (ON[rel], RDFS.range, ON[range_]) in after
    # The rename: the class and the relationships that named it.
    assert (ON["Dissertation"], RDF.type, OWL.Class) in after
    assert (ON["Thesis"], RDF.type, OWL.Class) in before
    assert (ON["Thesis"], RDF.type, OWL.Class) not in after
    assert (ON["SUPERVISED_BY"], RDFS.domain, ON["Dissertation"]) in after
    assert (ON["SUPERVISED_BY"], RDFS.domain, ON["Thesis"]) not in after
    # `extends` is still resolved: what `schema_core` defines is there.
    assert (ON["Person"], RDF.type, OWL.Class) in after


async def test_a_publish_reaches_the_extraction_service(svc, monkeypatch):
    await _publish_scholarly_edits(svc)
    prompt = await _extraction_prompt("scholarly", monkeypatch)
    assert ("- Preregistration: A study protocol registered before the data is collected."
            " (look for: preregistered at; OSF registration)") in prompt
    assert "- PREREGISTERED_AS: Article whose study was preregistered as this protocol" in prompt
    assert "- REGISTERS: Protocol registering the study of an article" in prompt
    assert "\n- Dissertation: " in prompt and "\n- Thesis: " not in prompt


async def test_names_that_are_not_iri_safe_are_exported_with_their_labels(svc):
    """OE.6 review B1: a name with a space or a slash (an accepted Pass-1
    proposal, an editor edit) made the serialiser fail the whole export."""
    await svc.put_entity_type("scholarly", "Clinical Trial Phase", {
        "description": "A phase of a clinical trial.", "parent_type": "ScholarlyArticle",
        "properties": [{"name": "trial phase", "data_type": "string"}]})
    await svc.put_entity_type("scholarly", "Journal/ Publication", {
        "description": "A journal publication.", "parent_type": "Clinical Trial Phase"})
    await svc.put_relationship_type("scholarly", "PUBLISHED IN", {
        "description": "Trial phase published in a journal",
        "domain": ["Clinical Trial Phase"], "range": ["Journal/ Publication", "Periodical"]})
    await svc.publish("scholarly", note="OE.6 review B1")

    exported = await _ttl("scholarly")
    for name in ("Clinical Trial Phase", "Journal/ Publication"):
        assert (class_iri(name), RDF.type, OWL.Class) in exported
        assert (class_iri(name), RDFS.label, Literal(name)) in exported
    assert (class_iri("Journal/ Publication"), RDFS.subClassOf,
            class_iri("Clinical Trial Phase")) in exported
    rel = class_iri("PUBLISHED IN")
    assert (rel, RDFS.label, Literal("PUBLISHED IN")) in exported
    assert _endpoint(exported, rel, RDFS.domain) == (False, {class_iri("Clinical Trial Phase")})
    assert _endpoint(exported, rel, RDFS.range) == (
        True, {class_iri("Journal/ Publication"), ON["Periodical"]})
    assert (property_iri("trial phase"), RDFS.label, Literal("trial phase")) in exported
    assert _endpoint(exported, property_iri("trial phase"), RDFS.domain) == (
        False, {class_iri("Clinical Trial Phase")})


async def test_an_unknown_base_is_a_500(svc):
    """A notebook whose base the registry does not know: a server-side
    misconfiguration, said as such (``schemas.py``)."""
    notebooks = AsyncMock()
    notebooks.get.return_value = object()
    schemas = AsyncMock()
    schemas.get_by_notebook.return_value = NotebookSchema(
        notebook="notebook:oe6", base_ontology="no_such_ontology")
    app = FastAPI()
    app.include_router(schemas_router.router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebooks
    app.dependency_overrides[get_notebook_schema_repo] = lambda: schemas
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        response = await client.get("/api/notebooks/notebook:oe6/schema.ttl")
    assert response.status_code == 500
    assert "no_such_ontology" in response.json()["detail"]


async def test_a_publish_in_the_parent_reaches_the_ontologies_that_extend_it(svc, monkeypatch):
    assert _raw("scholarly")["metadata"]["extends"] == "schema_core"
    await svc.put_entity_type("schema_core", "Streekarchief", {
        "description": "A regional archive.", "parent_type": "Organization"})
    await svc.publish("schema_core", note="OE.6 test")

    assert (ON["Streekarchief"], RDFS.subClassOf, ON["Organization"]) in await _ttl("scholarly")
    assert "- Streekarchief: A regional archive." in await _extraction_prompt(
        "scholarly", monkeypatch)


async def test_the_proxy_lists_the_registry_ontologies(svc, monkeypatch):
    async def _get(url, path, params=None):
        assert path == "/ontologies"
        return _SERVICE.get_ontologies()

    monkeypatch.setattr(services_proxy, "_proxy_get", _get)
    app = FastAPI()
    app.include_router(services_proxy.router, prefix="/api")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                 base_url="http://t") as client:
        body = (await client.get("/api/services/ontologies")).json()
    assert body == {"ontologies": SHIPPED, "default": _SERVICE.DEFAULT_ONTOLOGY}
