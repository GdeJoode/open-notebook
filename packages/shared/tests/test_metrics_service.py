"""Unit tests for ``shared.services.metrics.record_metric``.

These tests intentionally avoid any real DB connectivity — the function's
contract is:

1. Skip the write when ``OPEN_NOTEBOOK_DISABLE_METRICS=1``.
2. Forward ``event_type``/``payload``/``source``/``notebook`` to the
   persistence layer with the documented param shape.
3. Swallow exceptions raised by the persistence layer.

The persistence layer is patched via ``sys.modules`` so the lazy import
inside ``record_metric`` resolves to a stub. This mirrors the pattern
used by ``surrealdb_service.testing`` and keeps these tests fast.
"""

from __future__ import annotations

import sys
import types
from typing import Any, Dict, List, Optional

import pytest

from shared.services.metrics import record_metric


# ---------------------------------------------------------------------------
# Fake persistence layer
# ---------------------------------------------------------------------------


class _FakeExecuteQuery:
    """Capturing stub for ``surrealdb_service.connection.execute_query``.

    Records each invocation so individual tests can assert on the SQL
    string + params. Optionally raises to exercise the swallow-exception
    contract.
    """

    def __init__(self, raise_with: Optional[BaseException] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._raise_with = raise_with

    async def __call__(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        config: Any = None,
    ) -> List[Dict[str, Any]]:
        self.calls.append(
            {"query": query, "params": params or {}, "config": config}
        )
        if self._raise_with is not None:
            raise self._raise_with
        return []


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no test leaks ``OPEN_NOTEBOOK_DISABLE_METRICS`` into the next."""
    monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS", raising=False)


@pytest.fixture
def stub_execute_query(monkeypatch: pytest.MonkeyPatch) -> _FakeExecuteQuery:
    """Inject a fake ``surrealdb_service.connection`` module.

    ``record_metric`` lazy-imports ``execute_query`` from
    ``surrealdb_service.connection``. We replace that path in
    ``sys.modules`` so the import resolves to a controllable stub
    without requiring the real ``surrealdb-service`` package to be
    installed for these unit tests.
    """
    fake = _FakeExecuteQuery()
    # Build a minimal package shape: surrealdb_service.connection.execute_query
    pkg = types.ModuleType("surrealdb_service")
    conn = types.ModuleType("surrealdb_service.connection")
    conn.execute_query = fake  # type: ignore[attr-defined]
    pkg.connection = conn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "surrealdb_service", pkg)
    monkeypatch.setitem(sys.modules, "surrealdb_service.connection", conn)
    return fake


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_metric_writes_full_payload(
    stub_execute_query: _FakeExecuteQuery,
) -> None:
    """All four caller-supplied fields make it into params verbatim."""
    await record_metric(
        "extraction.complete",
        {"entity_count": 12, "relation_count": 5, "avg_confidence": 0.87},
        source="source:abc",
        notebook="notebook:xyz",
    )

    assert len(stub_execute_query.calls) == 1
    call = stub_execute_query.calls[0]
    assert "CREATE metrics SET" in call["query"]
    assert call["params"] == {
        "event_type": "extraction.complete",
        "payload": {
            "entity_count": 12,
            "relation_count": 5,
            "avg_confidence": 0.87,
        },
        "source": "source:abc",
        "notebook": "notebook:xyz",
    }


@pytest.mark.asyncio
async def test_record_metric_defaults_optional_keys_to_none(
    stub_execute_query: _FakeExecuteQuery,
) -> None:
    """Omitting ``source``/``notebook`` writes ``None`` (matching schema option<…>)."""
    await record_metric("extraction.auto_fallback", {"confidence": 0.4})

    assert len(stub_execute_query.calls) == 1
    params = stub_execute_query.calls[0]["params"]
    assert params["source"] is None
    assert params["notebook"] is None
    assert params["payload"] == {"confidence": 0.4}


@pytest.mark.asyncio
async def test_record_metric_coerces_non_dict_payload_to_empty(
    stub_execute_query: _FakeExecuteQuery,
) -> None:
    """A defensive caller may hand us ``None`` — we write ``{}`` to honour the schema default."""
    # type-ignore: callers shouldn't do this, but we want to verify the guard.
    await record_metric("extraction.complete", None)  # type: ignore[arg-type]

    assert len(stub_execute_query.calls) == 1
    assert stub_execute_query.calls[0]["params"]["payload"] == {}


# ---------------------------------------------------------------------------
# Env-flag opt-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_metric_skips_when_env_disabled(
    stub_execute_query: _FakeExecuteQuery,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPEN_NOTEBOOK_DISABLE_METRICS=1 → no write, no error."""
    monkeypatch.setenv("OPEN_NOTEBOOK_DISABLE_METRICS", "1")

    await record_metric("extraction.complete", {"entity_count": 1})

    assert stub_execute_query.calls == []


@pytest.mark.asyncio
async def test_record_metric_writes_when_env_unset(
    stub_execute_query: _FakeExecuteQuery,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty / unset env var → writes normally (regression guard)."""
    # Explicitly empty (not "1") — should still write.
    monkeypatch.setenv("OPEN_NOTEBOOK_DISABLE_METRICS", "")

    await record_metric("extraction.complete", {"entity_count": 1})

    assert len(stub_execute_query.calls) == 1


@pytest.mark.asyncio
async def test_record_metric_skip_is_evaluated_per_call(
    stub_execute_query: _FakeExecuteQuery,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Re-checking the env on every call lets tests toggle mid-run."""
    monkeypatch.setenv("OPEN_NOTEBOOK_DISABLE_METRICS", "1")
    await record_metric("extraction.complete", {})

    monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS")
    await record_metric("extraction.complete", {})

    assert len(stub_execute_query.calls) == 1


# ---------------------------------------------------------------------------
# Exception swallowing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_metric_swallows_persistence_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A raising backend must not propagate — extraction pipelines never crash on telemetry."""
    raising = _FakeExecuteQuery(raise_with=RuntimeError("simulated DB down"))
    pkg = types.ModuleType("surrealdb_service")
    conn = types.ModuleType("surrealdb_service.connection")
    conn.execute_query = raising  # type: ignore[attr-defined]
    pkg.connection = conn  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "surrealdb_service", pkg)
    monkeypatch.setitem(sys.modules, "surrealdb_service.connection", conn)

    # Must not raise.
    await record_metric("extraction.complete", {"entity_count": 1})

    # The backend was invoked (we got past the lazy import and the env check)
    # but the exception was caught.
    assert len(raising.calls) == 1


@pytest.mark.asyncio
async def test_record_metric_handles_missing_persistence_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``surrealdb_service`` isn't installed, log + swallow — never crash."""
    # Force the lazy import to fail with ImportError.
    monkeypatch.setitem(sys.modules, "surrealdb_service.connection", None)

    # Must not raise.
    await record_metric("extraction.complete", {"entity_count": 1})


# ---------------------------------------------------------------------------
# Defensive input handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_record_metric_rejects_empty_event_type(
    stub_execute_query: _FakeExecuteQuery,
) -> None:
    """Empty / non-string event_type is logged + skipped — preserves schema invariant."""
    await record_metric("", {"x": 1})
    await record_metric(None, {"x": 1})  # type: ignore[arg-type]

    assert stub_execute_query.calls == []
