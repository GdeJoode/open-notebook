"""Tests for :class:`NetworkxExportService` (Phase D.3).

Covers
======

* 7 happy-path round-trips (one per format) -- each format produces
  non-empty bytes that re-parse via the matching ``nx.read_*`` (or
  ``tree_graph`` / ``node_link_graph`` for JSON, ``pickle.loads`` for
  the binary).
* Attribute flattening: ``type_tags=["A", "B"]`` round-trips through
  GraphML as ``"A,B"`` -> ``["A", "B"]``.
* Empty notebook: every format produces a valid empty-graph output
  with 0 nodes / 0 edges.
* JSON-tree disconnected-graph fallback: 2 isolated nodes triggers the
  ``node_link_data`` fallback (envelope keys self-identify).
* Larger graph (5 entities / 3 relations) round-trips losslessly in
  GraphML for node/edge counts + canonical_name preservation.

Mocks
=====

The service depends only on ``list_entities_for_notebook`` /
``list_relations_for_notebook``; both are ``AsyncMock`` here. No DB,
no SurrealDB connection -- the tests run in-process.
"""

from __future__ import annotations

import io
import json
import pickle
from typing import List
from unittest.mock import AsyncMock

import networkx as nx
import pytest
from shared.models.entity import Entity, Relation
from shared.models.export import ExportFilter, NetworkxExportRequest

from app_main.services.networkx_export_service import NetworkxExportService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _entity(
    eid: str,
    name: str,
    type_tags: List[str] | None = None,
    primary_type: str | None = None,
    properties: dict | None = None,
    status: str = "active",
) -> Entity:
    """Build an ``Entity`` with sane export-ready defaults."""
    return Entity(
        id=eid,
        canonical_name=name,
        entity_type=primary_type or (type_tags[0] if type_tags else "Concept"),
        confidence=0.95,
        source_documents=["source:s1"],
        properties=properties or {},
        type_tags=type_tags or [],
        primary_type=primary_type or (type_tags[0] if type_tags else "Concept"),
        status=status,
    )


def _relation(
    src: str, dst: str, rel_type: str = "RELATED_TO", confidence: float = 0.9
) -> Relation:
    """Build a ``Relation`` between two entity ids."""
    return Relation(
        **{
            "in": src,
            "out": dst,
            "relation_type": rel_type,
            "confidence": confidence,
            "properties": {},
            "source_documents": ["source:s1"],
        }
    )


def _make_service(entities: List[Entity], relations: List[Relation]) -> NetworkxExportService:
    """Wire a service against AsyncMock repos returning the given rows."""
    repo = AsyncMock()
    repo.list_entities_for_notebook = AsyncMock(return_value=entities)
    repo.list_relations_for_notebook = AsyncMock(return_value=relations)
    return NetworkxExportService(entity_repository=repo, relation_repository=repo)


def _default_request(fmt: str) -> NetworkxExportRequest:
    """Build a request with relaxed filter so test fixtures pass through.

    The default ``ExportFilter`` is ``min_confidence=0.9`` which our
    fixtures already meet, but make ``min_connections=0`` so isolated
    nodes survive (the empty/disconnected tests need that).
    """
    return NetworkxExportRequest(
        format=fmt,
        filter=ExportFilter(min_connections=0),
    )


# ---------------------------------------------------------------------------
# 7 happy-path round-trips
# ---------------------------------------------------------------------------


@pytest.fixture
def small_graph_data() -> tuple[List[Entity], List[Relation]]:
    """A 3-node, 2-edge connected DAG (root -> a -> b)."""
    ents = [
        _entity("entity:root", "Alpha", type_tags=["Concept"]),
        _entity("entity:a", "Beta", type_tags=["Concept"]),
        _entity("entity:b", "Gamma", type_tags=["Concept"]),
    ]
    rels = [
        _relation("entity:root", "entity:a"),
        _relation("entity:a", "entity:b"),
    ]
    return ents, rels


async def test_export_graphml(small_graph_data):
    ents, rels = small_graph_data
    svc = _make_service(ents, rels)

    payload, report = await svc.export("notebook:test", _default_request("graphml"))

    assert payload, "graphml output must be non-empty"
    assert report.entities_written == 3
    assert report.relations_written == 2
    g = nx.read_graphml(io.BytesIO(payload))
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


async def test_export_gexf(small_graph_data):
    ents, rels = small_graph_data
    svc = _make_service(ents, rels)

    payload, _ = await svc.export("notebook:test", _default_request("gexf"))

    assert payload
    g = nx.read_gexf(io.BytesIO(payload))
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


async def test_export_gml(small_graph_data):
    ents, rels = small_graph_data
    svc = _make_service(ents, rels)

    payload, _ = await svc.export("notebook:test", _default_request("gml"))

    assert payload
    # ``nx.parse_gml`` accepts a string or iterable of strings -- we
    # decode the bytes once and split into lines for it.
    g = nx.parse_gml(payload.decode("utf-8").splitlines())
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


async def test_export_json_tree(small_graph_data):
    ents, rels = small_graph_data
    svc = _make_service(ents, rels)

    payload, _ = await svc.export("notebook:test", _default_request("json-tree"))

    assert payload
    parsed = json.loads(payload)
    # A connected DAG with 3 nodes + 2 edges is a tree -- tree_graph
    # is the matching reader. The root selection picks the
    # alphabetically-first canonical_name ("Alpha" -> entity:root).
    g = nx.tree_graph(parsed)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


async def test_export_edge_list(small_graph_data):
    ents, rels = small_graph_data
    svc = _make_service(ents, rels)

    payload, _ = await svc.export("notebook:test", _default_request("edge-list"))

    assert payload
    g = nx.read_edgelist(io.BytesIO(payload), create_using=nx.DiGraph)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


async def test_export_adjacency_list(small_graph_data):
    ents, rels = small_graph_data
    svc = _make_service(ents, rels)

    payload, _ = await svc.export("notebook:test", _default_request("adjacency-list"))

    assert payload
    g = nx.read_adjlist(io.BytesIO(payload), create_using=nx.DiGraph)
    assert g.number_of_nodes() == 3
    # The adjacency-list format encodes "src tgt1 tgt2 ..." -- it
    # preserves all edges in a DAG. ``read_adjlist`` reconstructs them
    # to a DiGraph correctly.
    assert g.number_of_edges() == 2


async def test_export_pickle(small_graph_data):
    ents, rels = small_graph_data
    svc = _make_service(ents, rels)

    payload, _ = await svc.export("notebook:test", _default_request("pickle"))

    assert payload
    g = pickle.loads(payload)
    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


# ---------------------------------------------------------------------------
# Larger round-trip integrity (5 entities / 3 relations)
# ---------------------------------------------------------------------------


async def test_graphml_round_trip_preserves_canonical_name_and_counts():
    """The acceptance criterion: counts + names survive GraphML."""
    ents = [
        _entity("entity:n1", "Alice", type_tags=["Person"]),
        _entity("entity:n2", "Acme Corp", type_tags=["Organization"]),
        _entity("entity:n3", "Berlin", type_tags=["Location"]),
        _entity("entity:n4", "Project X", type_tags=["Project"]),
        _entity("entity:n5", "Bob", type_tags=["Person"]),
    ]
    rels = [
        _relation("entity:n1", "entity:n2", "WORKS_AT"),
        _relation("entity:n2", "entity:n3", "LOCATED_IN"),
        _relation("entity:n1", "entity:n4", "CONTRIBUTES_TO"),
    ]
    svc = _make_service(ents, rels)

    payload, _ = await svc.export("notebook:test", _default_request("graphml"))
    g = nx.read_graphml(io.BytesIO(payload))

    assert g.number_of_nodes() == 5
    assert g.number_of_edges() == 3

    # canonical_name preserved on every node.
    names = {data.get("canonical_name") for _, data in g.nodes(data=True)}
    assert names == {"Alice", "Acme Corp", "Berlin", "Project X", "Bob"}


# ---------------------------------------------------------------------------
# Attribute flattening
# ---------------------------------------------------------------------------


async def test_type_tags_flatten_and_unflatten_via_graphml():
    """``type_tags=["Person", "Researcher"]`` -> ``"Person,Researcher"`` -> back."""
    ent = _entity(
        "entity:multi",
        "Marie Curie",
        type_tags=["Person", "Researcher"],
        primary_type="Person",
        properties={"nobel_count": 2, "fields": ["physics", "chemistry"]},
    )
    svc = _make_service([ent], [])

    payload, _ = await svc.export("notebook:test", _default_request("graphml"))
    g = nx.read_graphml(io.BytesIO(payload))

    node_data = list(g.nodes(data=True))[0][1]

    # Flat representation in the wire format.
    assert node_data["type_tags"] == "Person,Researcher"

    # Consumer re-splits as documented.
    assert node_data["type_tags"].split(",") == ["Person", "Researcher"]

    # Properties round-trip as JSON-encoded string.
    rehydrated = json.loads(node_data["properties"])
    assert rehydrated == {"nobel_count": 2, "fields": ["physics", "chemistry"]}


# ---------------------------------------------------------------------------
# Empty notebook -- every format must produce a valid empty graph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fmt",
    ["graphml", "gexf", "gml", "json-tree", "edge-list", "adjacency-list", "pickle"],
)
async def test_empty_notebook_produces_valid_empty_graph(fmt):
    """An empty notebook must serialise + re-parse with 0 nodes/edges.

    Note on ``edge-list``: ``nx.write_edgelist`` of an empty graph
    legitimately emits 0 bytes -- there are no edges to encode. The
    reader still returns an empty DiGraph from empty bytes, so the
    "valid empty graph" semantics are preserved even though the byte
    string itself is empty. We assert bytes-non-empty for every format
    EXCEPT edge-list to capture this corner of the writer surface.
    """
    svc = _make_service([], [])

    payload, report = await svc.export("notebook:test", _default_request(fmt))

    if fmt != "edge-list":
        assert payload, f"{fmt}: expected non-empty bytes for empty graph"
    assert report.entities_written == 0
    assert report.relations_written == 0

    if fmt == "graphml":
        g = nx.read_graphml(io.BytesIO(payload))
    elif fmt == "gexf":
        g = nx.read_gexf(io.BytesIO(payload))
    elif fmt == "gml":
        g = nx.parse_gml(payload.decode("utf-8").splitlines())
    elif fmt == "json-tree":
        parsed = json.loads(payload)
        # Empty graph falls through to node_link_data fast-path; reader
        # is ``node_link_graph``. ``edges="links"`` mirrors the writer.
        g = nx.node_link_graph(parsed, edges="links")
    elif fmt == "edge-list":
        g = nx.read_edgelist(io.BytesIO(payload), create_using=nx.DiGraph)
    elif fmt == "adjacency-list":
        g = nx.read_adjlist(io.BytesIO(payload), create_using=nx.DiGraph)
    elif fmt == "pickle":
        g = pickle.loads(payload)
    else:  # pragma: no cover
        raise AssertionError(f"unhandled fmt {fmt}")

    assert g.number_of_nodes() == 0
    assert g.number_of_edges() == 0


# ---------------------------------------------------------------------------
# JSON-tree fallback for disconnected graphs
# ---------------------------------------------------------------------------


async def test_json_tree_disconnected_graph_uses_node_link_fallback():
    """Two disconnected nodes -> ``node_link_data`` envelope, not ``tree_data``."""
    ents = [
        _entity("entity:iso1", "Alpha"),
        _entity("entity:iso2", "Beta"),
    ]
    svc = _make_service(ents, [])

    payload, _ = await svc.export("notebook:test", _default_request("json-tree"))

    envelope = json.loads(payload)

    # node_link_data envelope self-identifies: it has "nodes" + "links"
    # at the top level, while tree_data emits "id" + "children" at the root.
    assert "nodes" in envelope
    assert "links" in envelope
    assert envelope.get("directed") is True
    # Re-parse via the matching reader -- 2 nodes, 0 edges.
    g = nx.node_link_graph(envelope, edges="links")
    assert g.number_of_nodes() == 2
    assert g.number_of_edges() == 0


async def test_json_tree_multi_rooted_dag_falls_back_to_node_link():
    """REGRESSION PIN: multi-rooted DAG must NOT silently truncate (D.3 M1).

    ``nx.tree_data`` does not raise on a multi-rooted DAG; it walks
    from the chosen root and quietly drops unreachable nodes/edges.
    The previous try/except shape would let those silent losses
    through. This test pins the documented behaviour: the multi-rooted
    DAG ``a->c, b->c`` (Alpha + Beta share a child Gamma) must
    round-trip via the ``node_link_data`` fallback with ALL 3 nodes
    and BOTH edges intact.
    """
    ents = [
        _entity("entity:a", "Alpha"),
        _entity("entity:b", "Beta"),
        _entity("entity:c", "Gamma"),
    ]
    rels = [
        _relation("entity:a", "entity:c"),
        _relation("entity:b", "entity:c"),
    ]
    svc = _make_service(ents, rels)

    payload, report = await svc.export(
        "notebook:test", _default_request("json-tree")
    )

    envelope = json.loads(payload)

    # Self-identifying node-link envelope (NOT tree_data's "id"/"children").
    assert "nodes" in envelope
    assert "links" in envelope
    assert envelope.get("directed") is True

    # Round-trip via the matching reader -- ALL nodes/edges preserved.
    g = nx.node_link_graph(envelope, edges="links")
    assert g.number_of_nodes() == 3, "multi-rooted DAG truncated by tree_data"
    assert g.number_of_edges() == 2, "multi-rooted DAG lost an edge"

    # The build summary reports the same shape -- nothing dropped on
    # the graph-construction side either.
    assert report.entities_written == 3
    assert report.relations_written == 2


async def test_json_tree_rooted_arborescence_uses_tree_data():
    """PIN: a true rooted arborescence still goes through ``tree_data``.

    Counterpart to the multi-rooted-DAG pin: a single-rooted tree
    ``root -> a -> b`` must serialise via ``tree_data`` (not the
    fallback). We detect the chosen branch by inspecting envelope
    keys -- ``tree_data`` emits ``id`` + ``children`` at the top
    level, ``node_link_data`` emits ``nodes`` + ``links``.
    """
    ents = [
        _entity("entity:root", "Alpha", type_tags=["Concept"]),
        _entity("entity:a", "Beta", type_tags=["Concept"]),
        _entity("entity:b", "Gamma", type_tags=["Concept"]),
    ]
    rels = [
        _relation("entity:root", "entity:a"),
        _relation("entity:a", "entity:b"),
    ]
    svc = _make_service(ents, rels)

    payload, _ = await svc.export("notebook:test", _default_request("json-tree"))

    envelope = json.loads(payload)

    # tree_data shape: top-level dict has an "id" + nested "children".
    assert "id" in envelope, "rooted tree should serialise via tree_data"
    assert "nodes" not in envelope, "tree_data envelope should not have node-link keys"
    # Round-trip via tree_graph preserves all 3 nodes + 2 edges.
    g = nx.tree_graph(envelope)
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 2


async def test_edge_list_documented_behavior_drops_isolated_nodes():
    """REGRESSION PIN: edge-list discards isolated nodes (D.3 M2).

    ``nx.write_edgelist`` only emits edges, so an entity with zero
    in- and out-relations does not appear in the wire format. We
    pin this NetworkX-library behaviour explicitly so any future
    change to either side (writer swap, isolate-as-comment workaround)
    surfaces in CI rather than silently mutating the export shape.

    The service's module docstring documents this limitation under
    "Format limitations"; the UI dropdown carries an aria-label and
    a "(no isolates)" suffix to surface it to users at the click site.

    Fixture: 3 entities ``a, b, c`` with a single edge ``a->b``. The
    isolated ``c`` must NOT appear in the round-tripped graph; exactly
    2 nodes + 1 edge must survive.
    """
    ents = [
        _entity("entity:a", "Alpha"),
        _entity("entity:b", "Beta"),
        _entity("entity:c", "Gamma"),  # isolated -- no in/out relations
    ]
    rels = [
        _relation("entity:a", "entity:b"),
    ]
    svc = _make_service(ents, rels)

    payload, report = await svc.export(
        "notebook:test", _default_request("edge-list")
    )

    # The service still *counts* all 3 entities -- it writes the graph
    # with isolates intact; only the wire format drops them.
    assert report.entities_written == 3
    assert report.relations_written == 1

    # Round-trip via the matching reader: only 2 nodes + 1 edge survive.
    # The isolated ``entity:c`` (Gamma) has vanished from the wire format.
    g = nx.read_edgelist(io.BytesIO(payload), create_using=nx.DiGraph)
    assert g.number_of_nodes() == 2, (
        "edge-list should drop isolated nodes per NetworkX writer behaviour"
    )
    assert g.number_of_edges() == 1
    assert "entity:c" not in g.nodes, "isolated node leaked into edge-list output"
    # Both endpoints of the single edge survive.
    assert "entity:a" in g.nodes
    assert "entity:b" in g.nodes


# ---------------------------------------------------------------------------
# Status post-filter (archived / merged dropped)
# ---------------------------------------------------------------------------


async def test_archived_and_merged_entities_are_filtered_out():
    """Belt-and-braces filter on Entity.status (see service docstring)."""
    ents = [
        _entity("entity:active", "Active", status="active"),
        _entity("entity:archived", "Archived", status="archived"),
        _entity("entity:merged", "Merged", status="merged"),
    ]
    rels = [
        # An edge into a filtered entity must NOT survive (silent drop).
        _relation("entity:active", "entity:archived"),
    ]
    svc = _make_service(ents, rels)

    payload, report = await svc.export(
        "notebook:test", _default_request("graphml")
    )

    g = nx.read_graphml(io.BytesIO(payload))
    assert g.number_of_nodes() == 1
    assert report.entities_written == 1
    # The dangling edge was dropped because its endpoint vanished.
    assert g.number_of_edges() == 0
    assert report.relations_written == 0


# ---------------------------------------------------------------------------
# Telemetry payload shape (Q-D-8: counts only, no IDs)
# ---------------------------------------------------------------------------


async def test_telemetry_payload_has_counts_only(monkeypatch):
    """``record_metric`` payload must include format + counts -- no entity ids."""
    seen: list[dict] = []

    async def _spy(event_type, payload, source=None, notebook=None):
        seen.append({
            "event_type": event_type,
            "payload": payload,
            "source": source,
            "notebook": notebook,
        })

    # The autouse fixture sets OPEN_NOTEBOOK_DISABLE_METRICS=1 which
    # short-circuits record_metric. We unset it AND patch the symbol the
    # service has already imported so the spy fires.
    monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS", raising=False)
    monkeypatch.setattr(
        "app_main.services.networkx_export_service.record_metric", _spy
    )

    ents = [_entity("entity:a", "Alpha"), _entity("entity:b", "Beta")]
    rels = [_relation("entity:a", "entity:b")]
    svc = _make_service(ents, rels)

    await svc.export("notebook:tel", _default_request("graphml"))

    assert len(seen) == 1
    event = seen[0]
    assert event["event_type"] == "export.networkx"
    assert event["notebook"] == "notebook:tel"
    payload = event["payload"]
    assert payload["format"] == "graphml"
    assert payload["entities_written"] == 2
    assert payload["relations_written"] == 1
    # No ID leakage anywhere in the payload (case-insensitive search
    # across the JSON repr; matches both "entity:a" and "entity_id").
    payload_blob = json.dumps(payload)
    assert "entity:" not in payload_blob
    assert "entity_id" not in payload_blob
