"""Track M.1 — per-model context params: seed backfill.

  * ``backfill_chain_model_params`` fills ``context_window`` /
    ``max_output_tokens`` only when null (idempotent, operator-safe).

(F2-08: the factory-threading tests went with ``ModelFactory``; every model
call goes via the model gate to llama-swap.)
"""

from typing import Any, Dict, List, Optional

from surrealdb import RecordID


class _FakeDB:
    """Minimal execute_query fake: one model row, records UPDATE statements."""

    def __init__(self, row: Optional[Dict[str, Any]]):
        self._row = row
        self.updates: List[Dict[str, Any]] = []

    async def execute_query(self, query: str, params: Dict[str, Any] | None = None):
        params = params or {}
        if query.strip().upper().startswith("SELECT"):
            return [self._row] if self._row is not None else []
        if query.strip().upper().startswith("UPDATE"):
            # Like SurrealDB: `UPDATE $id` on a plain string is rejected
            # ("Can not execute UPDATE statement using value: 'model:…'").
            if not isinstance(params.get("id"), RecordID):
                raise RuntimeError(
                    f"Can not execute UPDATE statement using value: {params.get('id')!r}"
                )
            self.updates.append(params)
            # Apply the update to the in-memory row so a re-run sees it filled.
            for k, v in params.items():
                if k != "id":
                    self._row[k] = v
            return [self._row]
        return []


class TestSeedBackfill:
    async def test_fills_nulls(self, monkeypatch):
        row = {
            "id": "model:llama",
            "name": "llama3.1:8b",
            "provider": "ollama",
            "context_window": None,
            "max_output_tokens": None,
        }
        db = _FakeDB(row)
        import app_main.services.model_routing.seed_chain_models as scm

        monkeypatch.setattr(
            "surrealdb_service.connection.execute_query", db.execute_query
        )
        await scm._backfill_model_row(
            name="llama3.1:8b",
            provider="ollama",
            context_window=8192,
            max_output_tokens=4096,
        )
        assert row["context_window"] == 8192
        assert row["max_output_tokens"] == 4096
        assert len(db.updates) == 1

    async def test_does_not_overwrite_operator_values(self, monkeypatch):
        row = {
            "id": "model:llama",
            "name": "llama3.1:8b",
            "provider": "ollama",
            "context_window": 32_000,  # operator-tuned
            "max_output_tokens": 2048,
        }
        db = _FakeDB(row)
        import app_main.services.model_routing.seed_chain_models as scm

        monkeypatch.setattr(
            "surrealdb_service.connection.execute_query", db.execute_query
        )
        await scm._backfill_model_row(
            name="llama3.1:8b",
            provider="ollama",
            context_window=8192,
            max_output_tokens=4096,
        )
        # No UPDATE — operator values preserved.
        assert row["context_window"] == 32_000
        assert row["max_output_tokens"] == 2048
        assert db.updates == []

    async def test_missing_row_is_noop(self, monkeypatch):
        db = _FakeDB(None)
        import app_main.services.model_routing.seed_chain_models as scm

        monkeypatch.setattr(
            "surrealdb_service.connection.execute_query", db.execute_query
        )
        result = await scm._backfill_model_row(
            name="nope",
            provider="ollama",
            context_window=8192,
            max_output_tokens=4096,
        )
        assert result is None
        assert db.updates == []

    async def test_idempotent_second_run(self, monkeypatch):
        row = {
            "id": "model:llama",
            "name": "llama3.1:8b",
            "provider": "ollama",
            "context_window": None,
            "max_output_tokens": None,
        }
        db = _FakeDB(row)
        import app_main.services.model_routing.seed_chain_models as scm

        monkeypatch.setattr(
            "surrealdb_service.connection.execute_query", db.execute_query
        )
        for _ in range(2):
            await scm._backfill_model_row(
                name="llama3.1:8b",
                provider="ollama",
                context_window=8192,
                max_output_tokens=4096,
            )
        # Only the first run wrote; the second saw populated fields.
        assert len(db.updates) == 1


class TestLLMTaskGuardrail:
    """The I.G 768-dim embedding pin canary — M.1 touches model params, not
    routing tasks, so the 3-member LLMTask enum must stay exactly three."""

    def test_exactly_three_tasks(self):
        from app_main.services.model_routing.route_resolver import LLMTask

        assert len(LLMTask) == 3
        assert {t.value for t in LLMTask} == {
            "entity_extraction",
            "summarization",
            "chat",
        }
