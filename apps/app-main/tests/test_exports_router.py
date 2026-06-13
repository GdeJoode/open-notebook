"""Tests for the Track-D exports router (Phase D.3).

Covers
======

* Happy path: ``POST /api/notebooks/{id}/export-networkx`` with
  ``format=graphml`` returns 200 + the right ``Content-Type`` +
  ``Content-Disposition`` filename.
* 404 path: unknown notebook id -> 404 from the notebook lookup.
* 422 path: invalid ``format`` literal triggers Pydantic validation.
* Telemetry: the service-level ``record_metric`` call fires once with
  ``event_type='export.networkx'`` and ``format`` in the payload.

These are router-level smoke tests -- the service-level round-trip and
flattening tests live in ``test_networkx_export_service.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.models.export import ExportFilter, ExportReport, NetworkxExportRequest

from app_main.api.routers.exports import router
from app_main.dependencies import (
    get_networkx_export_service,
    get_notebook_service,
)

from tests.conftest import make_notebook


def _make_app(notebook_svc, export_svc):
    """Build a FastAPI test app with the dependencies stubbed."""
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_networkx_export_service] = lambda: export_svc
    return TestClient(app)


def _report(payload_size: int = 32) -> ExportReport:
    """Build a representative ``ExportReport`` for the service mock."""
    return ExportReport(
        entities_written=3,
        relations_written=2,
        files_written=1,
        bytes_written=payload_size,
        duration_ms=5,
        filter_applied=ExportFilter(),
    )


class TestExportNetworkxRouter:

    def test_graphml_happy_path(self):
        """Returns 200 + application/xml + Content-Disposition header."""
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        export_svc = AsyncMock()
        body = b'<?xml version="1.0"?><graphml/>'
        export_svc.export.return_value = (body, _report(len(body)))

        client = _make_app(notebook_svc, export_svc)

        resp = client.post(
            "/api/notebooks/notebook:abc/export-networkx",
            json={"format": "graphml", "filter": {"min_connections": 0}},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/xml")
        assert resp.content == body
        # Filename uses the sanitised notebook id (colon -> underscore)
        # + the format-specific extension. See ``_safe_filename``.
        assert (
            resp.headers["content-disposition"]
            == 'attachment; filename="notebook_abc.graphml"'
        )

        # Service called with the parsed request.
        export_svc.export.assert_awaited_once()
        call_args = export_svc.export.await_args
        assert call_args.args[0] == "notebook:abc"
        assert isinstance(call_args.args[1], NetworkxExportRequest)
        assert call_args.args[1].format == "graphml"

    def test_json_tree_has_application_json_content_type(self):
        """Different format -> different Content-Type."""
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        export_svc = AsyncMock()
        body = b'{"nodes": [], "links": []}'
        export_svc.export.return_value = (body, _report(len(body)))

        client = _make_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:abc/export-networkx",
            json={"format": "json-tree"},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.headers["content-disposition"].endswith('.json"')

    def test_pickle_has_octet_stream_content_type(self):
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        export_svc = AsyncMock()
        body = b"\x80\x05\x95"
        export_svc.export.return_value = (body, _report(len(body)))

        client = _make_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:abc/export-networkx",
            json={"format": "pickle"},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/octet-stream")
        assert resp.headers["content-disposition"].endswith('.pkl"')

    def test_404_unknown_notebook(self):
        """Unknown notebook id -> 404 from the upstream lookup."""
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = None  # not found
        export_svc = AsyncMock()

        client = _make_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:nope/export-networkx",
            json={"format": "graphml"},
        )

        assert resp.status_code == 404
        assert resp.json() == {"detail": "Notebook not found"}
        # Export service is never invoked when the notebook lookup fails.
        export_svc.export.assert_not_awaited()

    def test_422_invalid_format(self):
        """Pydantic rejects unknown ``format`` literal before the handler runs."""
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")
        export_svc = AsyncMock()

        client = _make_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:abc/export-networkx",
            json={"format": "not-a-real-format"},
        )

        assert resp.status_code == 422
        # The notebook-service lookup happens after body validation, so
        # we expect zero calls to either dependency.
        notebook_svc.get.assert_not_called()
        export_svc.export.assert_not_awaited()


class TestExportTelemetry:

    @pytest.mark.asyncio
    async def test_telemetry_fires_with_format_in_payload(self, monkeypatch):
        """End-to-end: service emits ``export.networkx`` metric with format."""
        from app_main.services.networkx_export_service import NetworkxExportService

        events: list[dict] = []

        async def _spy(event_type, payload, source=None, notebook=None):
            events.append({
                "event_type": event_type,
                "payload": payload,
                "notebook": notebook,
            })

        monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS", raising=False)
        monkeypatch.setattr(
            "app_main.services.networkx_export_service.record_metric", _spy
        )

        repo = AsyncMock()
        repo.list_entities_for_notebook.return_value = []
        repo.list_relations_for_notebook.return_value = []

        svc = NetworkxExportService(entity_repository=repo, relation_repository=repo)
        await svc.export(
            "notebook:tel-test",
            NetworkxExportRequest(format="gexf", filter=ExportFilter(min_connections=0)),
        )

        assert len(events) == 1
        assert events[0]["event_type"] == "export.networkx"
        assert events[0]["notebook"] == "notebook:tel-test"
        assert events[0]["payload"]["format"] == "gexf"
