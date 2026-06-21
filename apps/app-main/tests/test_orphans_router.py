"""Tests for the orphans dashboard router (Phase B.5b).

Covers the two endpoints exposed by ``app_main.api.routers.orphans``:

* ``GET /api/notebooks/{id}/orphans`` -- dashboard list endpoint
  (pending + archived counts, normalised row shape).

* ``POST /api/notebooks/{id}/orphans/{entity_id}/reconnect`` -- per-row
  manual retry. The attempt-1 review (M4) flagged that this endpoint
  was fanning out across every pending orphan in the notebook; the
  attempt-2 fix wires the new ``entity_id_filter`` parameter on
  :func:`retry_pending_reconnects` so the retry is scoped to the
  clicked orphan. These tests pin the fan-out / single-entity contract.

The tests use FastAPI's ``dependency_overrides`` plus ``AsyncMock``s
for the notebook service, entity repo, and source repo so no live DB
or LLM is required.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.orphans import router
from app_main.dependencies import (
    get_entity_repo,
    get_notebook_service,
    get_source_repo,
)
from app_main.services.notebook_service import NotebookService
from shared.models import Notebook
from surrealdb_service.repositories import EntityRepository, SourceRepository


_NOW = datetime(2026, 6, 1, 12, 0, 0)


def _make_notebook(notebook_id: str = "notebook:test1") -> Notebook:
    return Notebook(
        id=notebook_id,
        name="Test Notebook",
        description="",
        archived=False,
        created=_NOW,
        updated=_NOW,
    )


def _make_app(
    notebook_svc: AsyncMock,
    entity_repo: AsyncMock,
    source_repo: AsyncMock,
) -> TestClient:
    """Wire a FastAPI test client with the three deps overridden."""
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_entity_repo] = lambda: entity_repo
    app.dependency_overrides[get_source_repo] = lambda: source_repo
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET /orphans
# ---------------------------------------------------------------------------


class TestGetOrphans:
    """Dashboard list endpoint."""

    def test_returns_counts_and_normalised_rows(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        entity_repo = AsyncMock(spec=EntityRepository)
        # First call (pending) returns 1 row, second (archived) returns 1.
        entity_repo.list_orphans_with_status.side_effect = [
            [
                {
                    "id": "entity:pending-1",
                    "canonical_name": "Alice",
                    "entity_type": "PERSON",
                    "orphan_status": "pending_reconnect",
                    "reconnect_attempts": 1,
                    "first_orphaned_at": _NOW,
                    "last_reconnect_attempt_at": _NOW,
                }
            ],
            [
                {
                    "id": "entity:archived-1",
                    "canonical_name": "Bob",
                    "entity_type": "ORG",
                    "orphan_status": "archived",
                    "reconnect_attempts": 3,
                    "first_orphaned_at": _NOW,
                    "last_reconnect_attempt_at": _NOW,
                }
            ],
        ]

        source_repo = AsyncMock(spec=SourceRepository)

        client = _make_app(notebook_svc, entity_repo, source_repo)
        resp = client.get("/api/notebooks/notebook:test1/orphans")

        assert resp.status_code == 200
        body = resp.json()
        assert body["notebook_id"] == "notebook:test1"
        assert body["pending_count"] == 1
        assert body["archived_count"] == 1
        ids = [row["id"] for row in body["items"]]
        assert ids == ["entity:pending-1", "entity:archived-1"]

    def test_404_for_unknown_notebook(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = None

        entity_repo = AsyncMock(spec=EntityRepository)
        source_repo = AsyncMock(spec=SourceRepository)

        client = _make_app(notebook_svc, entity_repo, source_repo)
        resp = client.get("/api/notebooks/notebook:missing/orphans")
        assert resp.status_code == 404
        # The entity repo was never touched.
        assert not entity_repo.list_orphans_with_status.called


# ---------------------------------------------------------------------------
# POST /orphans/{eid}/reconnect -- M4 (attempt 2) single-orphan contract
# ---------------------------------------------------------------------------


class TestManualReconnect:
    """Per-row reconnect endpoint. Attempt-2 review M4: single-orphan only."""

    def _setup_entity_repo(
        self,
        entity_id: str = "entity:o1",
        status: str = "pending_reconnect",
        source_id: str = "source:s1",
    ) -> AsyncMock:
        entity_repo = AsyncMock(spec=EntityRepository)
        entity_repo.get_entity_detail.side_effect = [
            # First call: pre-retry shape (still pending).
            {
                "id": entity_id,
                "orphan_status": status,
                "source_documents": [source_id],
                "canonical_name": "Alice",
            },
            # Second call: post-retry shape (cleared, simulating success).
            {
                "id": entity_id,
                "orphan_status": None,
                "source_documents": [source_id],
                "canonical_name": "Alice",
            },
        ]
        return entity_repo

    def test_forwards_entity_id_filter_to_retry(self):
        """The router MUST pass ``entity_id_filter`` so the retry is scoped.

        Attempt-1 review M4 blocker: ``retry_pending_reconnects`` was
        invoked with notebook scope only, iterating every pending row.
        Attempt 2 introduces ``entity_id_filter`` -- the router must
        forward it.
        """
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        entity_id = "entity:o1"
        entity_repo = self._setup_entity_repo(entity_id=entity_id)

        # The source repo returns no chunks; the retry will still run.
        source_repo = AsyncMock(spec=SourceRepository)
        source_repo.get_chunks.return_value = []

        # Patch at the router-module level (where the symbol is bound),
        # not at the source module (FastAPI imports it by name).
        with patch(
            "app_main.api.routers.orphans.retry_pending_reconnects",
            new_callable=AsyncMock,
        ) as retry_mock, patch(
            "app_main.services.entity_extraction_service."
            "make_default_llm_caller",
            new_callable=AsyncMock,
            return_value=None,
        ):

            # Default outcome -- attempted=1 so the response reads as
            # "we tried but nothing came back" (this test doesn't care
            # about success/fail, only that the call was scoped).
            class _Outcome:
                attempted = 1
                reconnected = 0
                still_pending = 1

            retry_mock.return_value = _Outcome()

            client = _make_app(notebook_svc, entity_repo, source_repo)
            resp = client.post(
                f"/api/notebooks/notebook:test1/orphans/{entity_id}/reconnect"
            )

        assert resp.status_code == 200, resp.text
        # The retry function was called with entity_id_filter set to
        # this orphan's id -- the core M4 contract.
        assert retry_mock.await_count == 1
        kwargs = retry_mock.await_args.kwargs
        assert kwargs.get("entity_id_filter") == entity_id

    def test_private_source_reconnect_builds_only_local_model(self):
        """PRIVACY (J.4 rev2): a reconnect on a ``source.private`` orphan must

        build a LOCAL-only caller — the orphans router previously called
        ``make_default_llm_caller`` with no ``privacy_mode`` (-> CLOUD), so a
        private document could run NIM calls. This asserts:

          1. the factory receives ``privacy_mode=PrivacyMode.PRIVATE`` (the
             resolved sticky mode for a private source), and
          2. mirroring the AC6 factory-spy pattern, the resulting route only
             ever constructs a LOCAL ``LanguageModel`` (no cloud provider is
             built even though cloud behaviour is registered).
        """
        from datetime import datetime as _dt
        from unittest.mock import MagicMock

        import esperanto
        from app_main import dependencies
        from app_main.services.entity_extraction_service import (
            make_default_llm_caller,
        )
        from app_main.services.model_routing.route_resolver import (
            LLMTask,
            ModelCandidate,
            PrivacyMode,
            ResolvedRoute,
        )
        from llm_manager.manager import ModelManager
        from shared.models import Source
        from shared.models.llm import Model

        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        entity_id = "entity:o1"
        entity_repo = self._setup_entity_repo(
            entity_id=entity_id, source_id="source:priv1"
        )

        source_repo = AsyncMock(spec=SourceRepository)
        source_repo.get_chunks.return_value = []
        # The orphan's source is flagged private -> resolver is sticky-PRIVATE.
        source_repo.get.return_value = Source(
            id="source:priv1",
            title="Secret",
            private=True,
            created=_dt(2026, 6, 1, 12, 0, 0),
            updated=_dt(2026, 6, 1, 12, 0, 0),
        )

        # Fresh breaker/limiter registries so the real caller runs cleanly.
        dependencies.get_circuit_breaker_registry.cache_clear()
        dependencies.get_rate_limiter_registry.cache_clear()

        # A PRIVATE route is local-only; the factory spy must never see cloud.
        def _cand(provider, model_id, is_local):
            return ModelCandidate(
                provider=provider,
                model_id=model_id,
                is_local=is_local,
                model=Model(
                    id=model_id,
                    name=model_id,
                    provider=provider,
                    type="language",
                    is_local=is_local,
                ),
            )

        route = ResolvedRoute(
            task=LLMTask.ENTITY_EXTRACTION,
            mode=PrivacyMode.PRIVATE,
            ordered_candidates=[_cand("ollama", "model:local", True)],
        )
        resolver = MagicMock()
        resolver.resolve = AsyncMock(return_value=route)

        built_providers: list[str] = []

        class _FakeLM:
            def __init__(self, behavior):
                self._behavior = behavior

            async def achat_complete(self, messages):
                resp = MagicMock()
                resp.choices = [
                    MagicMock(message=MagicMock(content=self._behavior))
                ]
                return resp

        def _fake_get_model_from_config(self, model, **kwargs):
            built_providers.append(model.provider)
            return _FakeLM("PRIVATE_RECONNECT")

        captured_kwargs: dict = {}

        async def _spy_make_caller(*args, **kwargs):
            captured_kwargs.update(kwargs)
            return await make_default_llm_caller(*args, **kwargs)

        async def _fake_retry(*args, **kwargs):
            # Drive the caller exactly like the real retry would so the
            # factory-build spy observes which providers get constructed.
            caller = kwargs.get("llm_caller")
            if caller is not None:
                await caller("sys", "usr", "default")

            class _Outcome:
                attempted = 1
                reconnected = 0
                still_pending = 1

            return _Outcome()

        with patch(
            "app_main.api.routers.orphans.retry_pending_reconnects",
            new=_fake_retry,
        ), patch(
            "app_main.services.entity_extraction_service."
            "make_default_llm_caller",
            new=_spy_make_caller,
        ), patch.object(
            dependencies, "get_route_resolver", lambda: resolver
        ), patch.object(
            ModelManager, "get_model_from_config", _fake_get_model_from_config
        ), patch.object(
            esperanto, "LanguageModel", _FakeLM
        ):
            client = _make_app(notebook_svc, entity_repo, source_repo)
            resp = client.post(
                f"/api/notebooks/notebook:test1/orphans/{entity_id}/reconnect"
            )

        dependencies.get_circuit_breaker_registry.cache_clear()
        dependencies.get_rate_limiter_registry.cache_clear()

        assert resp.status_code == 200, resp.text
        # 1. The factory was told to route PRIVATE for the private source.
        assert captured_kwargs.get("privacy_mode") == PrivacyMode.PRIVATE
        # 2. Only a local model was ever constructed — no cloud leak.
        assert built_providers == ["ollama"]
        assert "nvidia" not in built_providers
        assert "openai" not in built_providers

    def test_409_when_entity_already_archived(self):
        """Archived entities can't be retried via this endpoint."""
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        entity_repo = AsyncMock(spec=EntityRepository)
        entity_repo.get_entity_detail.return_value = {
            "id": "entity:o1",
            "orphan_status": "archived",
            "source_documents": ["source:s1"],
        }

        source_repo = AsyncMock(spec=SourceRepository)

        client = _make_app(notebook_svc, entity_repo, source_repo)
        resp = client.post(
            "/api/notebooks/notebook:test1/orphans/entity:o1/reconnect"
        )

        assert resp.status_code == 409
        assert "archived" in resp.json()["detail"].lower()

    def test_404_when_entity_unknown(self):
        notebook_svc = AsyncMock(spec=NotebookService)
        notebook_svc.get.return_value = _make_notebook()

        entity_repo = AsyncMock(spec=EntityRepository)
        entity_repo.get_entity_detail.return_value = None

        source_repo = AsyncMock(spec=SourceRepository)

        client = _make_app(notebook_svc, entity_repo, source_repo)
        resp = client.post(
            "/api/notebooks/notebook:test1/orphans/entity:missing/reconnect"
        )
        assert resp.status_code == 404
