"""Track M.1 — per-model context params: factory threading + seed backfill.

Two surfaces:

  * ``ModelFactory.create_model`` threads ``model.max_output_tokens`` into the
    esperanto ``config["max_tokens"]`` for EVERY chain provider (google /
    nvidia / ollama) — regression-pinning the existing factory behaviour so the
    NIM-truncation class of bug cannot recur, and a caller-supplied max_tokens
    still wins.
  * ``backfill_chain_model_params`` fills ``context_window`` /
    ``max_output_tokens`` only when null (idempotent, operator-safe).
"""

from typing import Any, Dict, List, Optional

import pytest
from llm_manager.providers.factory import ModelFactory
from shared.models.llm import Model


def _capture_max_tokens(monkeypatch) -> Dict[str, Any]:
    """Patch ``create_language_model`` to capture the config it receives.

    Returns a mutable dict the test reads after ``create_model``; avoids any
    real esperanto / network construction.
    """
    captured: Dict[str, Any] = {}

    def _fake_create_language_model(model_name, provider, config=None):
        captured["model_name"] = model_name
        captured["provider"] = provider
        captured["config"] = dict(config or {})
        return object()  # stand-in LanguageModel; the test never calls it

    monkeypatch.setattr(
        ModelFactory, "create_language_model", staticmethod(_fake_create_language_model)
    )
    return captured


class TestFactoryThreading:
    @pytest.mark.parametrize(
        "provider,max_output",
        [("google", 8192), ("nvidia", 16384), ("ollama", 4096)],
    )
    def test_max_output_tokens_threaded_per_provider(
        self, monkeypatch, provider, max_output
    ):
        captured = _capture_max_tokens(monkeypatch)
        model = Model(
            name="m",
            provider=provider,
            type="language",
            context_window=128_000,
            max_output_tokens=max_output,
        )
        ModelFactory.create_model(model)
        assert captured["config"]["max_tokens"] == max_output

    def test_caller_supplied_max_tokens_wins(self, monkeypatch):
        captured = _capture_max_tokens(monkeypatch)
        model = Model(
            name="m", provider="nvidia", type="language", max_output_tokens=16384
        )
        ModelFactory.create_model(model, config={"max_tokens": 999})
        # The existing ``"max_tokens" not in config`` guard preserves the
        # caller's value over the model's ceiling.
        assert captured["config"]["max_tokens"] == 999

    def test_null_max_output_injects_no_key(self, monkeypatch):
        captured = _capture_max_tokens(monkeypatch)
        model = Model(name="m", provider="ollama", type="language")
        ModelFactory.create_model(model)
        # Null ceiling -> no max_tokens key -> esperanto's own default applies.
        assert "max_tokens" not in captured["config"]


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
