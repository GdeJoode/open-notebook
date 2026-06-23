"""Unit tests for the P.1 entity-embedding backfill script (DB-free).

A fake repo (in-memory missing-set + update) and a fake embedding model stand in
for the live DB / model layer, so these run without Docker or a configured model.
They assert the loop's idempotency (only ever embeds the still-missing set),
batching, dimension reporting, and the dry-run no-write path.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

# Load the script module by path (scripts/ is not an importable package).
_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "backfill_entity_embeddings.py"
)
_spec = importlib.util.spec_from_file_location("backfill_entity_embeddings", _SCRIPT)
backfill = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(backfill)


class FakeRepo:
    """In-memory entity store: id -> {canonical_name, embedding}."""

    def __init__(self, entities: Dict[str, str]) -> None:
        # entities: id -> canonical_name (all start with an empty embedding)
        self._rows = {
            eid: {"id": eid, "canonical_name": name, "embedding": []}
            for eid, name in entities.items()
        }
        self.update_calls = 0

    async def list_active_entities_missing_embedding(
        self, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        missing = [
            {"id": r["id"], "canonical_name": r["canonical_name"]}
            for r in self._rows.values()
            if not r["embedding"]
        ]
        return missing if limit is None else missing[:limit]

    async def update_entity_embedding(
        self, entity_id: str, embedding: List[float]
    ) -> bool:
        if not entity_id or not embedding:
            return False
        self._rows[entity_id]["embedding"] = list(embedding)
        self.update_calls += 1
        return True


class FakeModel:
    """Returns a fixed-dimension vector per text (dim=4 stands in for 768)."""

    def __init__(self, dim: int = 4) -> None:
        self.dim = dim
        self.batches: List[int] = []

    async def aembed(self, texts: List[str]) -> List[List[float]]:
        self.batches.append(len(texts))
        return [[float(len(t))] * self.dim for t in texts]


class FakeEmbedSvc:
    def __init__(self, model: FakeModel) -> None:
        self.embedding_model = model


@pytest.mark.asyncio
async def test_backfills_all_missing_entities() -> None:
    repo = FakeRepo({"entity:a": "Alpha", "entity:b": "Beta", "entity:c": "Gamma"})
    model = FakeModel(dim=4)

    async def fake_factory():
        return FakeEmbedSvc(model)

    with patch.object(backfill, "EntityRepository", return_value=repo), patch(
        "app_main.dependencies.get_embedding_service", new=fake_factory
    ):
        summary = await backfill._backfill(batch_size=2, limit=None, dry_run=False)

    assert summary == {"backfilled": 3, "failed": 0, "remaining": 0}
    # Every row now carries the 4-dim vector.
    for r in repo._rows.values():
        assert len(r["embedding"]) == 4
    # Batched in chunks of 2 (2 + 1).
    assert model.batches == [2, 1]


@pytest.mark.asyncio
async def test_idempotent_skips_already_embedded() -> None:
    repo = FakeRepo({"entity:a": "Alpha", "entity:b": "Beta"})
    # Pre-embed one entity — the backfill must skip it.
    repo._rows["entity:a"]["embedding"] = [0.1, 0.2, 0.3, 0.4]
    model = FakeModel(dim=4)

    async def fake_factory():
        return FakeEmbedSvc(model)

    with patch.object(backfill, "EntityRepository", return_value=repo), patch(
        "app_main.dependencies.get_embedding_service", new=fake_factory
    ):
        summary = await backfill._backfill(batch_size=10, limit=None, dry_run=False)

    assert summary["backfilled"] == 1  # only entity:b
    assert summary["remaining"] == 0
    # The pre-embedded vector is untouched.
    assert repo._rows["entity:a"]["embedding"] == [0.1, 0.2, 0.3, 0.4]


@pytest.mark.asyncio
async def test_limit_caps_one_run() -> None:
    repo = FakeRepo({f"entity:{i}": f"E{i}" for i in range(5)})
    model = FakeModel(dim=4)

    async def fake_factory():
        return FakeEmbedSvc(model)

    with patch.object(backfill, "EntityRepository", return_value=repo), patch(
        "app_main.dependencies.get_embedding_service", new=fake_factory
    ):
        summary = await backfill._backfill(batch_size=10, limit=2, dry_run=False)

    assert summary["backfilled"] == 2
    # 3 still missing — a follow-up run would finish them (resumable).
    assert summary["remaining"] == 3


@pytest.mark.asyncio
async def test_dry_run_writes_nothing() -> None:
    repo = FakeRepo({"entity:a": "Alpha", "entity:b": "Beta"})
    model = FakeModel(dim=4)

    async def fake_factory():
        raise AssertionError("dry-run must not resolve the embedding model")

    with patch.object(backfill, "EntityRepository", return_value=repo), patch(
        "app_main.dependencies.get_embedding_service", new=fake_factory
    ):
        summary = await backfill._backfill(
            batch_size=10, limit=None, dry_run=True
        )

    assert summary == {"backfilled": 0, "failed": 0, "remaining": 2}
    assert repo.update_calls == 0
    assert model.batches == []  # model never invoked
