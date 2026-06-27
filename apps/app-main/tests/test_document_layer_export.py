"""Tests for the Track U.5 document-graph export layer.

Three concerns, all in-process (AsyncMock repos, no DB):

1. The shared :func:`fetch_document_layer` fetcher — notebook scoping, the
   no-dangling-endpoint prune, source-node anchoring, the 0-cites no-op.
2. The NetworkX exporter document layer — ``source`` nodes + ``mentions`` /
   ``cites`` edges land in the graph, tagged with ``node_kind`` / ``edge_kind``,
   and the entity-only output is unchanged when the flag is off.
3. The JSONL exporter document layer — sources/mentions/cites .jsonl members
   appear only when the flag is on; the cites member exists even when empty.
"""

from __future__ import annotations

import io
import zipfile
from typing import Any, Dict, List
from unittest.mock import AsyncMock

import networkx as nx
import pytest
from shared.models.entity import Entity, Relation
from shared.models.export import (
    ExportFilter,
    JsonlExportRequest,
    NetworkxExportRequest,
)

from app_main.services.document_layer_export import fetch_document_layer
from app_main.services.jsonl_export_service import JsonlExportService
from app_main.services.networkx_export_service import NetworkxExportService


# ---------------------------------------------------------------------------
# Fixtures / builders
# ---------------------------------------------------------------------------


def _entity(eid: str, name: str, source_documents: List[str] | None = None) -> Entity:
    return Entity(
        id=eid,
        canonical_name=name,
        entity_type="Concept",
        confidence=0.95,
        source_documents=source_documents or ["source:s1"],
        properties={},
        type_tags=["Concept"],
        primary_type="Concept",
        status="active",
    )


def _relation(src: str, dst: str) -> Relation:
    return Relation(
        **{
            "in": src,
            "out": dst,
            "relation_type": "RELATED_TO",
            "confidence": 0.95,
            "properties": {},
            "source_documents": ["source:s1"],
        }
    )


def _mention_row(
    source: str, target: str, weight: float = 0.5, name: str = "Regio Deal"
) -> Dict[str, Any]:
    return {
        "id": f"mentions:{source}-{target}",
        "source": source,
        "target": target,
        "weight": weight,
        "concept_name": name,
        "concept_type": "Programme",
        "document_frequency": 4,
    }


def _cites_row(
    source: str, target: str, confidence: float = 0.9
) -> Dict[str, Any]:
    return {
        "id": f"cites:{source}-{target}",
        "source": source,
        "target": target,
        "confidence": confidence,
        "reference_text": "Some, A. (2024). Title.",
        "match_method": "doi",
    }


def _doc_repo(
    *,
    notebook_source_ids: List[str],
    mentions: List[Dict[str, Any]] | None = None,
    cites: List[Dict[str, Any]] | None = None,
    titles: Dict[str, str] | None = None,
) -> AsyncMock:
    """A combined entity+source repo mock with the U.2/U.3/U.5 seams."""
    repo = AsyncMock()
    repo.load_notebook_source_ids = AsyncMock(return_value=notebook_source_ids)
    repo.load_mentions_edges = AsyncMock(return_value=mentions or [])
    repo.load_cites_edges = AsyncMock(return_value=cites or [])
    repo.get_titles_by_ids = AsyncMock(
        return_value={k: v for k, v in (titles or {}).items()}
    )
    return repo


# ---------------------------------------------------------------------------
# 1. fetch_document_layer
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_layer_scopes_mentions_to_notebook_sources():
    """An out-of-notebook source endpoint is dropped from the layer."""
    repo = _doc_repo(
        notebook_source_ids=["source:s1"],
        mentions=[
            _mention_row("source:s1", "entity:a"),
            _mention_row("source:other", "entity:a"),  # out of notebook
        ],
        titles={"source:s1": "Doc One"},
    )
    layer = await fetch_document_layer(
        "notebook:n1",
        surviving_entity_ids={"entity:a"},
        entity_repo=repo,
        source_repo=repo,
    )
    assert {m.source_id for m in layer.mentions} == {"source:s1"}
    assert [n.id for n in layer.source_nodes] == ["source:s1"]
    assert layer.source_nodes[0].title == "Doc One"


@pytest.mark.asyncio
async def test_layer_prunes_mention_to_filtered_entity():
    """A mention to an entity that didn't survive the filter is dropped."""
    repo = _doc_repo(
        notebook_source_ids=["source:s1"],
        mentions=[
            _mention_row("source:s1", "entity:kept"),
            _mention_row("source:s1", "entity:filtered"),  # not surviving
        ],
        titles={"source:s1": "Doc One"},
    )
    layer = await fetch_document_layer(
        "notebook:n1",
        surviving_entity_ids={"entity:kept"},
        entity_repo=repo,
        source_repo=repo,
    )
    assert {m.entity_id for m in layer.mentions} == {"entity:kept"}


@pytest.mark.asyncio
async def test_layer_cites_intra_notebook_only():
    """A cites edge with an endpoint outside the notebook is dropped."""
    repo = _doc_repo(
        notebook_source_ids=["source:s1", "source:s2"],
        cites=[
            _cites_row("source:s1", "source:s2"),  # both in notebook
            _cites_row("source:s1", "source:ext"),  # target out of notebook
        ],
        titles={"source:s1": "A", "source:s2": "B"},
    )
    layer = await fetch_document_layer(
        "notebook:n1",
        surviving_entity_ids=set(),
        entity_repo=repo,
        source_repo=repo,
    )
    assert len(layer.cites) == 1
    assert layer.cites[0].from_source_id == "source:s1"
    assert layer.cites[0].to_source_id == "source:s2"
    # Both endpoints became source nodes (they anchor the kept cites edge).
    assert {n.id for n in layer.source_nodes} == {"source:s1", "source:s2"}


@pytest.mark.asyncio
async def test_layer_empty_when_no_notebook_sources():
    repo = _doc_repo(notebook_source_ids=[])
    layer = await fetch_document_layer(
        "notebook:n1",
        surviving_entity_ids={"entity:a"},
        entity_repo=repo,
        source_repo=repo,
    )
    assert layer.source_nodes == [] and layer.mentions == [] and layer.cites == []
    # No point loading edges when the notebook has no sources.
    repo.load_mentions_edges.assert_not_called()


@pytest.mark.asyncio
async def test_layer_cites_zero_no_op():
    """The 0-citation corpus yields an empty cites list, not an error."""
    repo = _doc_repo(
        notebook_source_ids=["source:s1"],
        mentions=[_mention_row("source:s1", "entity:a")],
        cites=[],
        titles={"source:s1": "Doc"},
    )
    layer = await fetch_document_layer(
        "notebook:n1",
        surviving_entity_ids={"entity:a"},
        entity_repo=repo,
        source_repo=repo,
    )
    assert layer.cites == []
    assert len(layer.mentions) == 1


@pytest.mark.asyncio
async def test_layer_isolated_source_not_injected():
    """A notebook source with no kept edge does NOT become a node (export)."""
    repo = _doc_repo(
        notebook_source_ids=["source:s1", "source:lonely"],
        mentions=[_mention_row("source:s1", "entity:a")],
        titles={"source:s1": "Doc", "source:lonely": "Lonely"},
    )
    layer = await fetch_document_layer(
        "notebook:n1",
        surviving_entity_ids={"entity:a"},
        entity_repo=repo,
        source_repo=repo,
    )
    assert [n.id for n in layer.source_nodes] == ["source:s1"]


# ---------------------------------------------------------------------------
# 2. NetworkX exporter document layer
# ---------------------------------------------------------------------------


def _nx_service(
    entities: List[Entity],
    relations: List[Relation],
    *,
    source_repo: AsyncMock | None = None,
) -> NetworkxExportService:
    repo = AsyncMock()
    repo.list_entities_for_notebook = AsyncMock(return_value=entities)
    repo.list_relations_for_notebook = AsyncMock(return_value=relations)
    if source_repo is not None:
        # the entity repo also answers load_mentions_edges
        repo.load_mentions_edges = source_repo.load_mentions_edges
    return NetworkxExportService(
        entity_repository=repo,
        relation_repository=repo,
        source_repository=source_repo,
    )


@pytest.mark.asyncio
async def test_networkx_document_layer_off_by_default():
    """Flag off → no source nodes, no mentions edges; entity graph intact."""
    ents = [_entity("entity:a", "Alpha"), _entity("entity:b", "Beta")]
    rels = [_relation("entity:a", "entity:b")]
    repo = _doc_repo(
        notebook_source_ids=["source:s1"],
        mentions=[_mention_row("source:s1", "entity:a")],
        titles={"source:s1": "Doc"},
    )
    svc = _nx_service(ents, rels, source_repo=repo)
    req = NetworkxExportRequest(
        format="graphml", filter=ExportFilter(min_connections=0)
    )
    _payload, report = await svc.export("notebook:n1", req)
    # No document layer requested → source_repo never consulted.
    repo.load_notebook_source_ids.assert_not_called()
    assert report.entities_written == 2
    assert report.metadata is None


@pytest.mark.asyncio
async def test_networkx_document_layer_adds_source_nodes_and_mentions():
    ents = [_entity("entity:a", "Alpha"), _entity("entity:b", "Beta")]
    rels = [_relation("entity:a", "entity:b")]
    repo = _doc_repo(
        notebook_source_ids=["source:s1"],
        mentions=[
            _mention_row("source:s1", "entity:a", weight=0.7),
            _mention_row("source:s1", "entity:b", weight=0.3),
        ],
        titles={"source:s1": "Doc One"},
    )
    svc = _nx_service(ents, rels, source_repo=repo)
    req = NetworkxExportRequest(
        format="pickle",
        filter=ExportFilter(min_connections=0, include_document_layer=True),
    )
    payload, report = await svc.export("notebook:n1", req)

    import pickle

    g: nx.DiGraph = pickle.loads(payload)
    # 2 entity nodes + 1 source node.
    assert g.number_of_nodes() == 3
    assert g.nodes["source:s1"]["node_kind"] == "source"
    assert g.nodes["source:s1"]["title"] == "Doc One"
    assert g.nodes["entity:a"]["node_kind"] == "entity"
    # 1 relation edge + 2 mentions edges.
    mentions = [
        (u, v) for u, v, d in g.edges(data=True) if d.get("edge_kind") == "mentions"
    ]
    assert set(mentions) == {("source:s1", "entity:a"), ("source:s1", "entity:b")}
    assert g["source:s1"]["entity:a"]["weight"] == 0.7
    assert report.metadata["document_layer"] == {
        "source_nodes": 1,
        "mentions_edges": 2,
        "cites_edges": 0,
    }


@pytest.mark.asyncio
async def test_networkx_document_layer_cites_edges_when_present():
    ents = [_entity("entity:a", "Alpha")]
    repo = _doc_repo(
        notebook_source_ids=["source:s1", "source:s2"],
        mentions=[_mention_row("source:s1", "entity:a")],
        cites=[_cites_row("source:s1", "source:s2", confidence=0.88)],
        titles={"source:s1": "A", "source:s2": "B"},
    )
    svc = _nx_service(ents, [], source_repo=repo)
    req = NetworkxExportRequest(
        format="pickle",
        filter=ExportFilter(min_connections=0, include_document_layer=True),
    )
    payload, report = await svc.export("notebook:n1", req)

    import pickle

    g: nx.DiGraph = pickle.loads(payload)
    cites = [
        (u, v, d) for u, v, d in g.edges(data=True) if d.get("edge_kind") == "cites"
    ]
    assert len(cites) == 1
    assert cites[0][0] == "source:s1" and cites[0][1] == "source:s2"
    assert cites[0][2]["confidence"] == 0.88
    assert report.metadata["document_layer"]["cites_edges"] == 1


@pytest.mark.asyncio
async def test_networkx_document_layer_graphml_roundtrips():
    """The mixed entity+document graph survives a GraphML round-trip."""
    ents = [_entity("entity:a", "Alpha")]
    repo = _doc_repo(
        notebook_source_ids=["source:s1"],
        mentions=[_mention_row("source:s1", "entity:a")],
        titles={"source:s1": "Doc"},
    )
    svc = _nx_service(ents, [], source_repo=repo)
    req = NetworkxExportRequest(
        format="graphml",
        filter=ExportFilter(min_connections=0, include_document_layer=True),
    )
    payload, _report = await svc.export("notebook:n1", req)
    g = nx.read_graphml(io.BytesIO(payload))
    assert "source:s1" in g.nodes
    assert g.nodes["source:s1"]["node_kind"] == "source"


# ---------------------------------------------------------------------------
# 3. JSONL exporter document layer
# ---------------------------------------------------------------------------


def _jsonl_service(
    entities: List[Entity],
    relations: List[Relation],
    *,
    source_repo: AsyncMock | None = None,
) -> JsonlExportService:
    repo = AsyncMock()
    repo.list_entities_for_notebook = AsyncMock(return_value=entities)
    repo.list_relations_for_notebook = AsyncMock(return_value=relations)
    if source_repo is not None:
        repo.load_mentions_edges = source_repo.load_mentions_edges
    return JsonlExportService(
        entity_repository=repo,
        relation_repository=repo,
        source_repository=source_repo,
    )


async def _drain(gen) -> bytes:
    chunks = []
    async for c in gen:
        chunks.append(c)
    return b"".join(chunks)


@pytest.mark.asyncio
async def test_jsonl_document_layer_off_keeps_two_files():
    ents = [_entity("entity:a", "Alpha")]
    repo = _doc_repo(
        notebook_source_ids=["source:s1"],
        mentions=[_mention_row("source:s1", "entity:a")],
        titles={"source:s1": "Doc"},
    )
    svc = _jsonl_service(ents, [], source_repo=repo)
    req = JsonlExportRequest(filter=ExportFilter(min_connections=0, min_confidence=0.0))
    payload = await _drain(svc.stream_jsonl("notebook:n1", req))
    zf = zipfile.ZipFile(io.BytesIO(payload))
    assert set(zf.namelist()) == {"entities.jsonl", "relations.jsonl"}
    repo.load_notebook_source_ids.assert_not_called()


@pytest.mark.asyncio
async def test_jsonl_document_layer_adds_three_files():
    ents = [_entity("entity:a", "Alpha")]
    repo = _doc_repo(
        notebook_source_ids=["source:s1"],
        mentions=[_mention_row("source:s1", "entity:a", weight=0.6)],
        cites=[],  # 0-cites corpus
        titles={"source:s1": "Doc One"},
    )
    svc = _jsonl_service(ents, [], source_repo=repo)
    req = JsonlExportRequest(
        filter=ExportFilter(
            min_connections=0, min_confidence=0.0, include_document_layer=True
        )
    )
    payload = await _drain(svc.stream_jsonl("notebook:n1", req))
    zf = zipfile.ZipFile(io.BytesIO(payload))
    # cites.jsonl present even though empty (consumer can rely on it).
    assert set(zf.namelist()) == {
        "entities.jsonl",
        "relations.jsonl",
        "sources.jsonl",
        "mentions.jsonl",
        "cites.jsonl",
    }
    import json

    sources = [
        json.loads(line)
        for line in zf.read("sources.jsonl").decode().splitlines()
    ]
    assert sources == [{"id": "source:s1", "title": "Doc One", "kind": "source"}]
    mentions = [
        json.loads(line)
        for line in zf.read("mentions.jsonl").decode().splitlines()
    ]
    assert mentions[0]["source"] == "source:s1"
    assert mentions[0]["target"] == "entity:a"
    assert mentions[0]["weight"] == 0.6
    assert zf.read("cites.jsonl").decode() == ""
