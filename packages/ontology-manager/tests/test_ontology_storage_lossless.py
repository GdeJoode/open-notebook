"""OE.0: an ontology can be stored as a version without losing anything.

The editor (Track OE) stores ontologies as versions through
``OntologyRegistry.register``. Before OE.0 every model ignored the keys it had
no field for, so storing a loaded ontology would have deleted the shipped
YAML's ``schema_org:`` URLs, ``graph_relevance``, property ``required`` /
``enum_values`` / ``validation_rules``, ``metadata.namespaces``,
``extraction_rules`` and ``abstract`` — and ``register`` wrote enums as Python
objects, so its YAML would not even load again.

The space is derived: every shipped YAML file. The round trip goes through the
serialisation ``register`` itself uses (``Ontology.to_yaml_string``) and, against
a real SurrealDB, through ``register`` and ``get`` themselves.
"""

from __future__ import annotations

import uuid
from pathlib import Path

import pytest
import surrealdb_service.connection as conn
import yaml
from ontology_manager.canonical_bridge import resolve_ontology_type
from ontology_manager.registry import OntologyRegistry
from ontology_manager.schema import (
    ConceptDefinition,
    EntityTypeDefinition,
    Ontology,
    PropertyDefinition,
)
from surrealdb_service.testing import live_surrealdb  # noqa: F401

ONTOLOGY_DIR = Path(__file__).resolve().parents[1] / "ontologies"
SHIPPED = sorted(ONTOLOGY_DIR.glob("*.yaml"))

_COLLECTIONS = ("entity_types", "relationship_types", "concepts", "claim_types",
                "extraction_patterns")


def _raw(name: str) -> dict:
    return yaml.safe_load((ONTOLOGY_DIR / f"{name}.yaml").read_text(encoding="utf-8"))


def _items(collection):
    """A YAML collection as {name: dict}, whichever form (list or dict) it uses."""
    if isinstance(collection, list):
        return {c["name"]: c for c in collection if isinstance(c, dict) and "name" in c}
    return dict(collection or {})


def _assert_same_content(stored: dict, raw: dict) -> None:
    assert set(stored) == set(raw)
    assert stored["metadata"] == raw["metadata"]
    for key in set(raw) - set(_COLLECTIONS) - {"metadata"}:
        assert stored[key] == raw[key], key
    for key in _COLLECTIONS:
        if key in raw:
            assert _items(stored[key]) == _items(raw[key]), key


def test_every_shipped_ontology_is_covered():
    assert len(SHIPPED) >= 11


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_nothing_is_lost_or_added(path):
    raw = _raw(path.stem)
    stored = yaml.safe_load(Ontology(**raw).to_yaml_string())
    _assert_same_content(stored, raw)
    assert Ontology(**stored) == Ontology(**raw)


# -- edits, then store: the editor's case ---------------------------------------


def _stored_after(edit, name="regiodeal") -> Ontology:
    ontology = Ontology.from_yaml(ONTOLOGY_DIR / f"{name}.yaml")
    edit(ontology)
    return Ontology.from_yaml_string(ontology.to_yaml_string())


def test_a_type_added_in_place_to_a_collection_the_yaml_never_set_is_stored():
    """Review M1: `regiodeal.yaml` has no `entity_types` key, so an
    `exclude_unset` dump dropped a type added to it."""
    assert "entity_types" not in _raw("regiodeal")
    back = _stored_after(lambda o: o.entity_types.__setitem__(
        "Streekfonds", EntityTypeDefinition(name="Streekfonds", parent_type="Grant")))
    assert back.entity_types["Streekfonds"].parent_type == "Grant"


def test_an_in_place_edit_below_the_top_is_stored():
    def edit(o):
        o.concepts["X"] = ConceptDefinition(name="X")
        o.entity_types["Project"].properties.append(PropertyDefinition(name="budget", data_type="string"))
        o.entity_types["Project"].description = "changed"

    back = _stored_after(edit, "deals")
    assert "X" in back.concepts
    assert back.entity_types["Project"].properties[-1].name == "budget"
    assert back.entity_types["Project"].description == "changed"


def test_a_derived_field_is_not_written_twice():
    """`domain`/`range` are derived from `source_types`/`target_types`."""
    raw = {"metadata": {"name": "t", "version": "1"},
           "relationship_types": [{"name": "r", "source_types": ["A"], "target_types": ["B"]}]}
    stored = yaml.safe_load(Ontology(**raw).to_yaml_string())
    assert stored["relationship_types"]["r"] == {"name": "r", "source_types": ["A"],
                                                 "target_types": ["B"]}


# -- inheritance --------------------------------------------------------------


def test_inheritance_combines_extra_keys_as_their_reader_does():
    """Review M4: `services/extraction/api.py` joins the parent's and the child's
    `extraction_rules`; the merge replaced the parent's."""
    parent = Ontology(metadata={"name": "p", "version": "1"},
                      extraction_rules=["a", "b"], settings={"x": 1, "y": 1}, abstract=True)
    child = Ontology(metadata={"name": "c", "version": "1", "extends": "p"},
                     extraction_rules=["b", "c"], settings={"y": 2}, abstract=False)
    merged = child.merge_with(parent)
    assert merged.model_extra == {"extraction_rules": ["a", "b", "c"],
                                  "settings": {"x": 1, "y": 2}, "abstract": False}


# -- the registry -------------------------------------------------------------


@pytest.fixture
def registry():
    reg = OntologyRegistry()
    reg.clear_cache()
    yield reg
    reg.clear_cache()


@pytest.mark.asyncio
async def test_own_and_merged_are_cached_apart(registry, monkeypatch):
    """The cache key ignored `resolve_inheritance`: after a merged get, an
    `own` get returned the merged ontology — which `register` would store."""

    async def _no_db(self, name, version=None):
        return None

    monkeypatch.setattr(OntologyRegistry, "_load_from_db", _no_db)
    merged = await registry.get("regiodeal")
    own = await registry.get("regiodeal", resolve_inheritance=False)
    assert "Person" in merged.entity_types
    assert "Person" not in own.entity_types


@pytest.fixture
async def db(live_surrealdb):  # noqa: F811
    """The registry queries through the GLOBAL pool: point it at the container,
    and remove every version a test stored (their versions carry `tag`)."""
    tag = f"oe0-{uuid.uuid4().hex[:8]}"
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    yield tag
    await conn.execute_query(
        "DELETE ontology_version WHERE string::contains(version, $tag);", {"tag": tag}
    )
    await conn._pool.close_all()
    conn._pool = original


def _versioned(name: str, tag: str) -> Ontology:
    ontology = Ontology.from_yaml(ONTOLOGY_DIR / f"{name}.yaml")
    ontology.metadata.version = f"{ontology.metadata.version}-{tag}"
    return ontology


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_every_shipped_ontology_round_trips_through_register(registry, db):
    """The plan's AC: load → register → load, for every shipped file."""
    for path in SHIPPED:
        ontology = _versioned(path.stem, db)
        assert await registry.register(ontology), path.stem
        [row] = await conn.execute_query(
            "SELECT yaml_content FROM ontology_version WHERE name = $n AND version = $v;",
            {"n": path.stem, "v": ontology.metadata.version},
        )
        stored = yaml.safe_load(row["yaml_content"])
        raw = _raw(path.stem)
        raw["metadata"]["version"] = ontology.metadata.version
        _assert_same_content(stored, raw)
        back = await registry.get(path.stem, version=ontology.metadata.version,
                                  resolve_inheritance=False)
        assert back.metadata.version == ontology.metadata.version  # from the DB, not the file
        assert back == ontology


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_storing_an_existing_version_again_is_refused_and_changes_nothing(registry, db):
    """Review B1: re-storing without activating deactivated the active version
    and overwrote the stored content."""
    first = _versioned("general", db)
    assert await registry.register(first, activate=True, note="first")
    second = _versioned("general", db)
    second.entity_types["Topic"].description = "overwritten"
    assert not await registry.register(second, activate=False, note="resave")

    [row] = [v for v in await registry.list_versions("general")
             if v["version"] == first.metadata.version]
    assert (row["is_active"], row["note"]) == (True, "first")
    active = await registry.get("general", resolve_inheritance=False)
    assert active.entity_types["Topic"].description != "overwritten"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_activating_a_parent_version_reaches_its_children(registry, db):
    """The OE.0 AC through `activate_version`: `regiodeal` extends `schema_core`
    three levels down, and must not keep the old parent in its cache."""
    before = await registry.get("regiodeal")
    assert resolve_ontology_type("Grant", [before]).canonical == "grant"

    edited = _versioned("schema_core", db)
    edited.entity_types["Grant"].schema_org_type = "schema:Event"
    assert await registry.register(edited)                  # stored, not active
    assert resolve_ontology_type("Grant", [await registry.get("regiodeal")]).canonical == "grant"

    assert await registry.activate_version("schema_core", edited.metadata.version)
    after = await registry.get("regiodeal")
    assert resolve_ontology_type("Grant", [after]).canonical == "event"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_activating_an_unknown_version_changes_nothing(registry, db):
    """It used to deactivate every version first and then activate none."""
    own = _versioned("base", db)
    assert await registry.register(own, activate=True)
    assert not await registry.activate_version("base", f"does-not-exist-{db}")
    versions = await registry.list_versions("base")
    assert [v["version"] for v in versions if v["is_active"]] == [own.metadata.version]
