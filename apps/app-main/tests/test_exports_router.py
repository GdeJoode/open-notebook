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

import io
import zipfile
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.models.export import (
    ExportFilter,
    ExportReport,
    JsonlExportRequest,
    NetworkxExportRequest,
    ObsidianExportRequest,
)

from app_main.api.routers.exports import router
from app_main.dependencies import (
    get_jsonl_export_service,
    get_networkx_export_service,
    get_notebook_service,
    get_obsidian_export_service,
)
from app_main.services.obsidian_export_service import (
    ExportArtifact,
    VaultPathNotConfigured,
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


# ---------------------------------------------------------------------------
# D.1a -- Obsidian zip export router tests
# ---------------------------------------------------------------------------


def _make_obsidian_app(notebook_svc, export_svc):
    """Build a FastAPI test app with Obsidian dependencies stubbed."""
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_obsidian_export_service] = lambda: export_svc
    return TestClient(app)


def _build_test_zip() -> bytes:
    """Build a minimal valid zip with a README + one .md for the router test."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("README.md", "# Test export\n")
        archive.writestr("alice.md", "---\nid: entity:alice\n---\n\n# Alice\n")
    return buf.getvalue()


def _obsidian_report(payload_size: int) -> ExportReport:
    return ExportReport(
        entities_written=1,
        relations_written=0,
        files_written=2,  # README + alice.md
        bytes_written=payload_size,
        duration_ms=8,
        filter_applied=ExportFilter(),
    )


class TestExportObsidianRouter:

    def test_obsidian_zip_happy_path(self):
        """POST /export-obsidian -> 200 + application/zip + parseable zip."""
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        zip_bytes = _build_test_zip()
        export_svc = AsyncMock()
        export_svc.export.return_value = ExportArtifact(
            mode="zip",
            report=_obsidian_report(len(zip_bytes)),
            zip_bytes=zip_bytes,
        )

        client = _make_obsidian_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:abc/export-obsidian",
            json={"mode": "zip", "filter": {"min_connections": 0}},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/zip")
        # The router rebuilds a sanitised filename: colon -> underscore,
        # ".zip" extension.
        assert (
            resp.headers["content-disposition"]
            == 'attachment; filename="notebook_abc.zip"'
        )

        # Zip content parses cleanly and has the expected files.
        archive = zipfile.ZipFile(io.BytesIO(resp.content))
        names = archive.namelist()
        assert "README.md" in names
        assert "alice.md" in names

        # Service called with the parsed request.
        export_svc.export.assert_awaited_once()
        call_args = export_svc.export.await_args
        assert call_args.args[0] == "notebook:abc"
        assert isinstance(call_args.args[1], ObsidianExportRequest)
        assert call_args.args[1].mode == "zip"

    def test_obsidian_404_unknown_notebook(self):
        """Unknown notebook id -> 404 from the upstream lookup."""
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = None
        export_svc = AsyncMock()

        client = _make_obsidian_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:nope/export-obsidian",
            json={"mode": "zip"},
        )

        assert resp.status_code == 404
        assert resp.json() == {"detail": "Notebook not found"}
        export_svc.export.assert_not_awaited()

    def test_obsidian_filter_validation(self):
        """Invalid filter shape -> 422 from Pydantic before the handler runs."""
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")
        export_svc = AsyncMock()

        client = _make_obsidian_app(notebook_svc, export_svc)
        # ``min_confidence`` is bounded [0, 1]; 5.0 must trigger 422.
        resp = client.post(
            "/api/notebooks/notebook:abc/export-obsidian",
            json={"mode": "zip", "filter": {"min_confidence": 5.0}},
        )

        assert resp.status_code == 422
        notebook_svc.get.assert_not_called()
        export_svc.export.assert_not_awaited()

    def test_obsidian_vault_path_happy_path(self):
        """``mode=vault_path`` -> 200 + application/json + ExportReport body.

        D.1b: the router returns the ExportReport as JSON (no file body)
        when the service writes to disk successfully. The artifact
        carries the resolved target dir for debugging but the router
        does not echo it back (the path is server-side; the client
        doesn't need to know where on disk it landed).
        """
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        report = ExportReport(
            entities_written=3,
            relations_written=2,
            files_written=4,  # README + 3 entity files
            bytes_written=512,
            duration_ms=12,
            # Match the filter the request body carries so the
            # round-trip assertion below makes sense (the service
            # would echo the same filter into the report).
            filter_applied=ExportFilter(min_connections=0),
        )
        export_svc = AsyncMock()
        export_svc.export.return_value = ExportArtifact(
            mode="vault_path",
            report=report,
            zip_bytes=None,
            vault_dir="/var/vaults/notebook-abc/Entities",
        )

        client = _make_obsidian_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:abc/export-obsidian",
            json={"mode": "vault_path", "filter": {"min_connections": 0}},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/json")
        # No Content-Disposition -- this is not a download.
        assert "content-disposition" not in {k.lower() for k in resp.headers}

        body = resp.json()
        # Body is the ExportReport (counts + filter snapshot + duration).
        assert body["entities_written"] == 3
        assert body["relations_written"] == 2
        assert body["files_written"] == 4
        assert body["bytes_written"] == 512
        assert body["duration_ms"] == 12
        # Filter snapshot round-trips.
        assert body["filter_applied"]["min_connections"] == 0

        # Service called with the parsed request.
        export_svc.export.assert_awaited_once()
        call_args = export_svc.export.await_args
        assert call_args.args[0] == "notebook:abc"
        assert isinstance(call_args.args[1], ObsidianExportRequest)
        assert call_args.args[1].mode == "vault_path"

    def test_obsidian_vault_path_not_configured_400(self):
        """Settings.vault_path missing -> 400 with a clear error message.

        The service raises ``VaultPathNotConfigured``; the router maps
        it to 400 (user/operator can fix it via the Settings UI).
        """
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        export_svc = AsyncMock()
        export_svc.export.side_effect = VaultPathNotConfigured(
            "Settings.vault_path is not configured. Configure it in "
            "Settings before exporting in vault_path mode."
        )

        client = _make_obsidian_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:abc/export-obsidian",
            json={"mode": "vault_path"},
        )

        assert resp.status_code == 400
        body = resp.json()
        assert "vault_path" in body["detail"].lower()
        assert "configure" in body["detail"].lower()

    def test_obsidian_vault_path_filesystem_failure_500_with_partial(self):
        """Filesystem failure mid-write -> 500 with partial entities_written.

        Verifies the per-file atomicity contract surfaces correctly at
        the HTTP boundary: when ``_write_to_vault`` propagates an
        ``OSError`` with ``{"entities_written": N}`` in its args, the
        router echoes that count in the 500 body so the client knows
        where the batch stopped.
        """
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        export_svc = AsyncMock()
        # Mimic the _write_to_vault failure mode: OSError with a
        # partial-state dict in args.
        export_svc.export.side_effect = OSError(
            "simulated disk full",
            {"entities_written": 2, "failed_file": "carol.md"},
        )

        client = _make_obsidian_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:abc/export-obsidian",
            json={"mode": "vault_path"},
        )

        assert resp.status_code == 500
        body = resp.json()
        # detail carries the structured failure body.
        detail = body["detail"]
        assert isinstance(detail, dict)
        assert "error" in detail
        assert detail["entities_written"] == 2
        assert detail.get("failed_file") == "carol.md"


# ---------------------------------------------------------------------------
# D.2 -- JSONL streaming export router tests
# ---------------------------------------------------------------------------


def _make_jsonl_app(notebook_svc, export_svc):
    """Build a FastAPI test app with the JSONL dependency stubbed."""
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_jsonl_export_service] = lambda: export_svc
    return TestClient(app)


class TestExportJsonlRouter:

    def test_jsonl_happy_path(self):
        """POST /export-jsonl -> 200 + application/zip + parseable zip body.

        Verifies the StreamingResponse passes the service's chunks
        through unchanged and the Content-Disposition uses the
        ``.jsonl.zip`` extension so the OS save dialog disambiguates
        from the Obsidian export's plain ``.zip``.
        """
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        # Hand-build a zip with two .jsonl members so the test asserts
        # against a known structure. The stream_jsonl method on the mock
        # is an *async generator*, so we wire it via a sync function
        # returning the generator directly (AsyncMock would return a
        # coroutine, breaking StreamingResponse iteration).
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("entities.jsonl", '{"id":"entity:a"}\n')
            archive.writestr("relations.jsonl", "")
        payload = buf.getvalue()

        async def _fake_stream(_nb, _req):
            # Split arbitrarily so the integration test exercises >1
            # chunk path through StreamingResponse.
            yield payload[: len(payload) // 2]
            yield payload[len(payload) // 2 :]

        export_svc = AsyncMock()
        export_svc.stream_jsonl = _fake_stream  # plain callable, not AsyncMock

        client = _make_jsonl_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:abc/export-jsonl",
            json={"filter": {"min_connections": 0}},
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("application/zip")
        assert (
            resp.headers["content-disposition"]
            == 'attachment; filename="notebook_abc.jsonl.zip"'
        )

        # Parseable zip with the two expected members.
        archive = zipfile.ZipFile(io.BytesIO(resp.content))
        names = set(archive.namelist())
        assert names == {"entities.jsonl", "relations.jsonl"}

    def test_jsonl_404_unknown_notebook(self):
        """Unknown notebook id -> 404 from the upstream lookup."""
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = None
        export_svc = AsyncMock()

        client = _make_jsonl_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:nope/export-jsonl",
            json={"filter": {}},
        )

        assert resp.status_code == 404
        assert resp.json() == {"detail": "Notebook not found"}

    def test_jsonl_filter_validation(self):
        """Invalid filter shape -> 422 from Pydantic before the handler runs.

        ``min_confidence`` is bounded [0, 1]; 5.0 must trigger 422 so the
        client knows to fix the slider, not retry blindly.
        """
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")
        export_svc = AsyncMock()

        client = _make_jsonl_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:abc/export-jsonl",
            json={"filter": {"min_confidence": 5.0}},
        )

        assert resp.status_code == 422
        notebook_svc.get.assert_not_called()

    def test_jsonl_request_forwards_filter_to_service(self):
        """The router builds a JsonlExportRequest from the body and forwards it.

        Pins that the filter knobs from the body land in the
        service-side request object -- a regression where the router
        instantiated ``JsonlExportRequest()`` with no args would still
        pass the happy-path test but would silently drop the user's
        filter.
        """
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        captured: list[tuple] = []

        async def _capture_stream(notebook_id, request):
            captured.append((notebook_id, request))
            yield b""

        export_svc = AsyncMock()
        export_svc.stream_jsonl = _capture_stream

        client = _make_jsonl_app(notebook_svc, export_svc)
        resp = client.post(
            "/api/notebooks/notebook:abc/export-jsonl",
            json={
                "filter": {
                    "min_connections": 3,
                    "min_confidence": 0.85,
                    "include_orphans": True,
                }
            },
        )

        assert resp.status_code == 200
        assert len(captured) == 1
        nb_id, request = captured[0]
        assert nb_id == "notebook:abc"
        assert isinstance(request, JsonlExportRequest)
        assert request.filter.min_connections == 3
        assert request.filter.min_confidence == 0.85
        assert request.filter.include_orphans is True
