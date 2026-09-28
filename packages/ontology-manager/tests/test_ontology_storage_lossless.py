"""OE.0: an ontology can be stored without losing anything.

The editor (Track OE) will store ontologies as versions through
``OntologyRegistry.register``. Before OE.0 every model ignored the keys it had
no field for, so storing a loaded ontology would have deleted the shipped
YAML's ``schema_org:`` URLs, ``graph_relevance``, property ``required`` /
``enum_values`` / ``validation_rules``, ``metadata.namespaces``,
``extraction_rules`` and ``abstract`` — and ``register`` wrote enums as Python
objects, so its YAML would not even load again.

The space is derived: every shipped YAML file, every entity type, relationship
type, property and concept in it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import surrealdb_service.connection as conn
import yaml
from ontology_manager.canonical_bridge import resolve_ontology_type
from ontology_manager.registry import OntologyRegistry
from ontology_manager.schema import Ontology
from surrealdb_service.testing import live_surrealdb  # noqa: F401

ONTOLOGY_DIR = Path(__file__).resolve().parents[1] / "ontologies"
SHIPPED = sorted(ONTOLOGY_DIR.glob("*.yaml"))

_COLLECTIONS = ("entity_types", "relationship_types", "concepts", "claim_types",
                "extraction_patterns")


def _items(collection):
    """A YAML collection as {name: dict}, whichever form (list or dict) it uses."""
    if isinstance(collection, list):
        return {c["name"]: c for c in collection if isinstance(c, dict) and "name" in c}
    return dict(collection or {})


def _stored(raw: dict) -> dict:
    """What `register` writes for this ontology, read back as YAML."""
    text = yaml.safe_dump(
        Ontology(**raw).model_dump(mode="json", exclude_unset=True), sort_keys=False
    )
    return yaml.safe_load(text)


def test_every_shipped_ontology_is_covered():
    assert len(SHIPPED) >= 11


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_nothing_is_lost_or_added(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    stored = _stored(raw)
    assert set(stored) == set(raw)
    assert stored["metadata"] == raw["metadata"]
    for key in set(raw) - set(_COLLECTIONS) - {"metadata"}:
        assert stored[key] == raw[key], key
    for key in _COLLECTIONS:
        if key in raw:
            assert _items(stored[key]) == _items(raw[key]), key


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_what_is_stored_loads_to_the_same_ontology(path):
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert Ontology(**_stored(raw)) == Ontology(**raw)


def test_inheritance_carries_the_keys_the_model_has_no_field_for():
    parent = Ontology(metadata={"name": "p", "version": "1"}, extraction_rules={"a": 1}, abstract=False)
    child = Ontology(metadata={"name": "c", "version": "1", "extends": "p"}, extraction_rules={"b": 2})
    merged = child.merge_with(parent)
    assert merged.model_extra == {"extraction_rules": {"b": 2}, "abstract": False}


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
def pool_on(live_surrealdb):  # noqa: F811
    """The registry queries through the GLOBAL pool; point it at the container."""
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    yield live_surrealdb
    conn._pool = original


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_a_stored_version_of_a_parent_reaches_its_children(registry, pool_on):
    """The OE.0 AC: activate a version of `schema_core`, and `regiodeal` (which
    extends it three levels down) resolves the new definition — a change to a
    parent must not stay behind in its children's cache."""
    before = await registry.get("regiodeal")
    assert resolve_ontology_type("Grant", [before]).canonical == "grant"

    own = Ontology.from_yaml(ONTOLOGY_DIR / "schema_core.yaml")
    edited = own.model_copy(deep=True)
    edited.metadata.version = "1.0.0-oe0-test"
    edited.entity_types["Grant"].schema_org_type = "schema:Event"
    assert await registry.register(edited, activate=True, note="OE.0 test")

    after = await registry.get("regiodeal")
    assert resolve_ontology_type("Grant", [after]).canonical == "event"

    versions = await registry.list_versions("schema_core")
    assert [v["version"] for v in versions if v["is_active"]] == ["1.0.0-oe0-test"]


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_what_the_database_holds_is_what_was_stored(registry, pool_on):
    raw = yaml.safe_load((ONTOLOGY_DIR / "deals.yaml").read_text(encoding="utf-8"))
    own = Ontology(**raw)
    own.metadata.version = "1.0.0-oe0-roundtrip"
    assert await registry.register(own)
    [row] = await conn.execute_query(
        "SELECT yaml_content FROM ontology_version WHERE name = 'deals' "
        "AND version = '1.0.0-oe0-roundtrip';", {}, pool_on,
    )
    back = yaml.safe_load(row["yaml_content"])
    assert _items(back["entity_types"]) == _items(raw["entity_types"])
    assert back["extraction_rules"] == raw["extraction_rules"]  # a key with no field
    assert await registry.get("deals", version="1.0.0-oe0-roundtrip") is not None


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_activating_an_unknown_version_changes_nothing(registry, pool_on):
    """It used to deactivate every version first and then activate none."""
    own = Ontology.from_yaml(ONTOLOGY_DIR / "base.yaml")
    own.metadata.version = "1.0.0-oe0-active"
    assert await registry.register(own, activate=True)
    assert not await registry.activate_version("base", "does-not-exist")
    versions = await registry.list_versions("base")
    assert [v["version"] for v in versions if v["is_active"]] == ["1.0.0-oe0-active"]
