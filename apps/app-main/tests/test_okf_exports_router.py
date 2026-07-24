"""Tests for the OKF export router endpoint (Phase OKF.2).

Router-level smoke tests with the services stubbed (mirrors the existing
``test_exports_router`` pattern — TestClient + dependency overrides). The pure
bundle mapper is covered in ``test_okf_export_service``.

Covered
=======
* Inline happy path: small notebook -> 200 + application/zip + the ExportReport
  (incl. omitted-field ledger) surfaced in the ``X-OKF-Export-Report`` header.
* Large notebook: entity count over the threshold -> 202 + job enqueued at the
  ``CommandService.submit_command_job`` seam (job-queue-singleton pattern).
* 404 unknown notebook.
* 422 invalid filter shape.
"""

from __future__ import annotations

import io
import json
import zipfile
from unittest.mock import AsyncMock, patch

from app_main.api.routers import exports as exports_router
from app_main.api.routers.exports import router
from app_main.dependencies import get_notebook_service, get_okf_export_service
from app_main.services.okf_export_service import OKF_OMITTED_FIELDS, OkfExportArtifact
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.models.export import ExportFilter, ExportReport

from tests.conftest import make_notebook


def _make_app(notebook_svc, export_svc) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_okf_export_service] = lambda: export_svc
    return TestClient(app)


def _build_test_zip() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("index.md", "---\ntype: Index\nokf_version: '0.1'\n---\n")
        z.writestr("entities/alice.md", "---\ntype: Person\n---\n# Alice\n")
    return buf.getvalue()


def _report(bytes_written: int) -> ExportReport:
    return ExportReport(
        entities_written=2,
        relations_written=1,
        files_written=2,
        bytes_written=bytes_written,
        duration_ms=4,
        filter_applied=ExportFilter(),
        metadata={
            "notes_written": 0,
            "sources_written": 0,
            "dropped_relations": 0,
            "omitted_fields": OKF_OMITTED_FIELDS,
            "okf_version": "0.1",
        },
    )


def test_okf_inline_zip_happy_path():
    notebook_svc = AsyncMock()
    notebook_svc.get.return_value = make_notebook(id="notebook:abc")

    zip_bytes = _build_test_zip()
    export_svc = AsyncMock()
    export_svc.count_entities = AsyncMock(return_value=12)
    export_svc.export = AsyncMock(
        return_value=OkfExportArtifact(report=_report(len(zip_bytes)), zip_bytes=zip_bytes)
    )

    client = _make_app(notebook_svc, export_svc)
    resp = client.post(
        "/api/notebooks/notebook:abc/export/okf",
        json={"filter": {"min_connections": 0}},
    )

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/zip")
    assert (
        resp.headers["content-disposition"]
        == 'attachment; filename="notebook_abc.okf.zip"'
    )

    # Bundle parses and carries the reserved index.
    archive = zipfile.ZipFile(io.BytesIO(resp.content))
    assert "index.md" in archive.namelist()

    # ExportReport (incl. the omitted-field ledger) surfaced in the header.
    report = json.loads(resp.headers["X-OKF-Export-Report"])
    assert report["entities_written"] == 2
    assert "entity.embedding" in report["metadata"]["omitted_fields"]

    export_svc.export.assert_awaited_once()


def test_okf_large_notebook_defers_to_job_queue():
    notebook_svc = AsyncMock()
    notebook_svc.get.return_value = make_notebook(id="notebook:big")

    export_svc = AsyncMock()
    # Over the threshold -> the endpoint must NOT build inline.
    export_svc.count_entities = AsyncMock(
        return_value=exports_router.OKF_ASYNC_ENTITY_THRESHOLD + 1
    )
    export_svc.export = AsyncMock()

    client = _make_app(notebook_svc, export_svc)
    with patch.object(
        exports_router.CommandService,
        "submit_command_job",
        new=AsyncMock(return_value="job:okf-1"),
    ) as submit:
        resp = client.post(
            "/api/notebooks/notebook:big/export/okf",
            json={"filter": {"min_connections": 0}},
        )

    assert resp.status_code == 202
    body = resp.json()
    assert body["job_id"] == "job:okf-1"
    assert body["status"] == "queued"
    assert body["entity_count"] == exports_router.OKF_ASYNC_ENTITY_THRESHOLD + 1

    # Enqueue asserted at the seam; inline export never ran.
    submit.assert_awaited_once()
    kwargs = submit.await_args.kwargs
    assert kwargs["command_name"] == "export_okf"
    assert kwargs["command_args"]["notebook_id"] == "notebook:big"
    export_svc.export.assert_not_awaited()


def test_okf_404_unknown_notebook():
    notebook_svc = AsyncMock()
    notebook_svc.get.return_value = None
    export_svc = AsyncMock()

    client = _make_app(notebook_svc, export_svc)
    resp = client.post(
        "/api/notebooks/notebook:nope/export/okf", json={"filter": {}}
    )

    assert resp.status_code == 404
    export_svc.count_entities.assert_not_awaited()


def test_okf_invalid_filter_422():
    notebook_svc = AsyncMock()
    notebook_svc.get.return_value = make_notebook(id="notebook:abc")
    export_svc = AsyncMock()

    client = _make_app(notebook_svc, export_svc)
    resp = client.post(
        "/api/notebooks/notebook:abc/export/okf",
        json={"filter": {"min_confidence": 5.0}},
    )

    assert resp.status_code == 422
    export_svc.count_entities.assert_not_awaited()


# ---------------------------------------------------------------------------
# Deferred-export download endpoint (OKF.2 async artifact store)
# ---------------------------------------------------------------------------

from pathlib import Path  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from app_main.dependencies import get_settings_service  # noqa: E402
from app_main.services.okf_export_service import okf_artifact_path  # noqa: E402


def _make_download_app(notebook_svc, settings_svc) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_settings_service] = lambda: settings_svc
    return TestClient(app)


def _settings_svc(vault_path):
    svc = AsyncMock()
    svc.get = AsyncMock(return_value=SimpleNamespace(vault_path=vault_path))
    return svc


def _notebook_svc(notebook):
    svc = AsyncMock()
    svc.get = AsyncMock(return_value=notebook)
    return svc


def test_okf_download_serves_persisted_artifact(tmp_path: Path):
    # Persist an artifact where the async handler would write it.
    artifact = okf_artifact_path(str(tmp_path), "notebook:abc")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(_build_test_zip())

    client = _make_download_app(
        _notebook_svc(make_notebook(id="notebook:abc")), _settings_svc(str(tmp_path))
    )
    resp = client.get("/api/notebooks/notebook:abc/export/okf/download")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert "attachment" in resp.headers["content-disposition"]
    # Body is the persisted zip, intact.
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        assert "index.md" in z.namelist()


def test_okf_download_404_when_not_ready(tmp_path: Path):
    # vault configured, but the job has not written the artifact yet.
    client = _make_download_app(
        _notebook_svc(make_notebook(id="notebook:abc")), _settings_svc(str(tmp_path))
    )
    resp = client.get("/api/notebooks/notebook:abc/export/okf/download")
    assert resp.status_code == 404


def test_okf_download_404_when_no_vault():
    client = _make_download_app(
        _notebook_svc(make_notebook(id="notebook:abc")), _settings_svc(None)
    )
    resp = client.get("/api/notebooks/notebook:abc/export/okf/download")
    assert resp.status_code == 404


def test_okf_download_404_unknown_notebook(tmp_path: Path):
    client = _make_download_app(_notebook_svc(None), _settings_svc(str(tmp_path)))
    resp = client.get("/api/notebooks/notebook:nope/export/okf/download")
    assert resp.status_code == 404


def test_okf_202_body_carries_download_url():
    notebook_svc = AsyncMock()
    notebook_svc.get = AsyncMock(return_value=make_notebook(id="notebook:big"))
    export_svc = AsyncMock()
    export_svc.count_entities = AsyncMock(return_value=10_000_000)
    client = _make_app(notebook_svc, export_svc)

    with patch.object(
        exports_router.CommandService,
        "submit_command_job",
        new=AsyncMock(return_value="job:xyz"),
    ):
        resp = client.post(
            "/api/notebooks/notebook:big/export/okf", json={"filter": {}}
        )
    assert resp.status_code == 202
    body = resp.json()
    assert body["download_url"] == (
        "/api/notebooks/notebook:big/export/okf/download"
    )
