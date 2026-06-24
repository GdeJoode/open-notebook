"""Q.4 backbone check: weak-edge warnings for well-connected affected nodes.

The backbone check (B6) flags nodes that LOOK well-connected (effective degree
>= the config ``well_connected_threshold``, default 3) but whose connectivity
rests on WEAK (``RELATED``) edges. It is a pure read-time computation over the
config predicate-strength + the persisted edges, so these tests mock the repo's
``relations_for_entities`` and exercise ONLY the partition/flag logic. No DB.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest
from app_main.services.triage.backbone_check import BackboneCheckService
from app_main.services.triage.triage_config import load_triage_config

pytestmark = pytest.mark.asyncio


def _service(edges: List[Dict[str, Any]]):
    """A backbone service whose repo returns ``edges`` for any id set."""
    repo = AsyncMock()
    repo.relations_for_entities = AsyncMock(return_value=edges)
    return BackboneCheckService(repo, config=load_triage_config()), repo


async def test_below_threshold_nodes_are_skipped():
    """A node under the well-connected threshold yields no warning + no query."""
    svc, repo = _service([])
    warnings = await svc.warnings_for(["entity:a"], {"entity:a": 2})
    assert warnings == []
    repo.relations_for_entities.assert_not_awaited()


async def test_well_connected_node_with_weak_edges_is_flagged():
    """A degree-3 node resting on RELATED edges is flagged with those edges."""
    edges = [
        {"in": "entity:a", "out": "entity:b", "relation_type": "RELATED",
         "source_documents": ["s:1", "s:2"]},
        {"in": "entity:a", "out": "entity:c", "relation_type": "BIJDRAGT_AAN",
         "source_documents": ["s:1"]},
        {"in": "entity:d", "out": "entity:a", "relation_type": "RELATED",
         "source_documents": ["s:3"]},
    ]
    svc, _ = _service(edges)
    warnings = await svc.warnings_for(["entity:a"], {"entity:a": 3})
    assert len(warnings) == 1
    w = warnings[0]
    assert w.entity_id == "entity:a"
    assert w.structural_degree == 3
    # Only the two RELATED edges are weak; the structural one is excluded.
    assert len(w.weak_edges) == 2
    others = {e.other for e in w.weak_edges}
    assert others == {"entity:b", "entity:d"}
    doc_counts = {e.other: e.doc_count for e in w.weak_edges}
    assert doc_counts["entity:b"] == 2
    assert doc_counts["entity:d"] == 1


async def test_well_connected_node_with_no_weak_edges_yields_no_warning():
    """A well-connected node resting purely on structural edges → no warning."""
    edges = [
        {"in": "entity:a", "out": "entity:b", "relation_type": "BIJDRAGT_AAN",
         "source_documents": ["s:1"]},
        {"in": "entity:a", "out": "entity:c", "relation_type": "FINANCIERT",
         "source_documents": ["s:1"]},
    ]
    svc, _ = _service(edges)
    warnings = await svc.warnings_for(["entity:a"], {"entity:a": 5})
    assert warnings == []


async def test_weak_edge_between_two_well_connected_nodes_warns_both():
    """A weak edge spanning two flagged nodes is a warning for each endpoint."""
    edges = [
        {"in": "entity:a", "out": "entity:b", "relation_type": "RELATED",
         "source_documents": ["s:1", "s:2"]},
    ]
    svc, _ = _service(edges)
    warnings = await svc.warnings_for(
        ["entity:a", "entity:b"], {"entity:a": 3, "entity:b": 4}
    )
    flagged = {w.entity_id for w in warnings}
    assert flagged == {"entity:a", "entity:b"}
    for w in warnings:
        assert len(w.weak_edges) == 1


async def test_unmapped_predicate_defaults_to_weak():
    """A predicate absent from config is treated as weak (config default)."""
    edges = [
        {"in": "entity:a", "out": "entity:b", "relation_type": "SOME_UNKNOWN",
         "source_documents": ["s:1"]},
    ]
    svc, _ = _service(edges)
    warnings = await svc.warnings_for(["entity:a"], {"entity:a": 3})
    assert len(warnings) == 1
    assert warnings[0].weak_edges[0].relation_type == "SOME_UNKNOWN"
