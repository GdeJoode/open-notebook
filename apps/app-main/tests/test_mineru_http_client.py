"""
Tests for app_main.services.mineru_http_client.

We mock the network with ``httpx.MockTransport`` (mirrors the docling
test pattern) and we write fixture JSON + markdown to disk so the
layout parser has something real to read. No MinerU install required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from app_main.services import mineru_http_client as mineru_client_module
from app_main.services.mineru_http_client import (
    MineruHttpClient,
    MineruServiceError,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_source_pdf(tmp_path: Path, name: str = "report.pdf") -> Path:
    src = tmp_path / name
    src.write_bytes(b"%PDF-1.4 fake")
    return src


def _make_service_output(
    output_root: Path,
    *,
    stem: str = "report",
    method: str = "auto",
    records: list[dict] | None = None,
    backend: str = "pipeline",
    include_markdown: bool = True,
    include_middle: bool = True,
) -> Path:
    """Write a fake MinerU output tree and return the per-document dir."""
    out_dir = output_root / stem / method
    out_dir.mkdir(parents=True, exist_ok=True)

    if records is None:
        records = [
            {"type": "text", "text": "Report title", "text_level": 1, "bbox": [0, 0, 1000, 100], "page_idx": 0},
            {"type": "text", "text": "Body text.", "bbox": [0, 110, 1000, 200], "page_idx": 0},
        ]

    (out_dir / f"{stem}_content_list.json").write_text(json.dumps(records), encoding="utf-8")

    if include_markdown:
        (out_dir / f"{stem}.md").write_text("# Report title\n\nBody text.\n", encoding="utf-8")
    if include_middle:
        (out_dir / f"{stem}_middle.json").write_text(
            json.dumps({"_backend": backend, "_version_name": "2.0.0", "pdf_info": []}),
            encoding="utf-8",
        )

    # Optional images dir to exercise the image_paths population branch.
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "images" / "img1.jpg").write_bytes(b"fakejpeg")

    return out_dir


def _make_success_response(out_dir: Path, stem: str = "report") -> dict[str, Any]:
    return {
        "total_files": 1,
        "succeeded": 1,
        "failed": 0,
        "total_seconds": 1.0,
        "results": [
            {
                "source": "/data/input/xxx__report.pdf",
                "success": True,
                "output_dir": str(out_dir),
                "markdown_path": str(out_dir / f"{stem}.md"),
                "content_list_path": str(out_dir / f"{stem}_content_list.json"),
                "middle_json_path": str(out_dir / f"{stem}_middle.json"),
                "page_count": 1,
                "table_count": 0,
                "image_count": 0,
                "processing_seconds": 0.5,
                "backend": "pipeline",
                "method": "auto",
            }
        ],
    }


def _make_client(
    tmp_path: Path,
    *,
    transport: httpx.MockTransport | None = None,
) -> MineruHttpClient:
    """Build a client wired to the shared tmp dirs."""
    client = MineruHttpClient(
        service_url="http://mineru.test:8104",
        input_dir=tmp_path / "input",
        output_dir=tmp_path / "mineru_output",
    )
    if transport is not None:
        # Patch httpx.AsyncClient so the client uses the mock transport.
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = transport
            return original(*args, **kwargs)

        client._async_client_factory = _factory  # type: ignore[attr-defined]
    return client


# ---------------------------------------------------------------------------
# httpx.AsyncClient patching helper
# ---------------------------------------------------------------------------


class _PatchedAsyncClient:
    """Context-manager wrapper that injects MockTransport into httpx.AsyncClient.

    Patching at the module level (``mineru_client_module.httpx.AsyncClient``)
    is the cleanest way to intercept the client's network calls without
    refactoring the production code to accept an injected transport.
    """

    def __init__(self, transport: httpx.MockTransport) -> None:
        self.transport = transport
        self._patch = None

    def __enter__(self) -> "_PatchedAsyncClient":
        original = httpx.AsyncClient

        def _factory(*args, **kwargs):
            kwargs["transport"] = self.transport
            # Drop any explicit transport= the production code passes; the
            # current implementation doesn't pass one but defensive.
            return original(*args, **kwargs)

        self._patch = patch.object(mineru_client_module.httpx, "AsyncClient", _factory)
        self._patch.start()
        return self

    def __exit__(self, *exc_info) -> None:
        if self._patch is not None:
            self._patch.stop()


def _mock_transport(responder) -> httpx.MockTransport:
    return httpx.MockTransport(responder)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_process_success_returns_ingestion_result_with_document(tmp_path: Path) -> None:
    src = _make_source_pdf(tmp_path)
    out_dir = _make_service_output(tmp_path / "mineru_output")

    def responder(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/process"
        body = json.loads(request.content)
        assert body["input_path"].endswith("__report.pdf")
        assert body["output_path"].endswith("mineru_output")
        return httpx.Response(200, json=_make_success_response(out_dir))

    with _PatchedAsyncClient(_mock_transport(responder)):
        client = _make_client(tmp_path)
        result = await client.process(src)

    assert result.success is True
    assert result.error_message is None
    assert result.document is not None
    assert result.document.title == "Report title"
    assert result.document.page_count == 1
    assert result.document.full_text  # populated
    assert result.markdown_path is not None and result.markdown_path.exists()
    assert result.json_path is not None and result.json_path.exists()
    assert result.source_metadata.page_count == 1
    # image_paths populated from the on-disk images/ dir.
    assert any(p.name == "img1.jpg" for p in result.image_paths)


@pytest.mark.asyncio
async def test_process_cleans_up_staged_input_on_success(tmp_path: Path) -> None:
    src = _make_source_pdf(tmp_path)
    out_dir = _make_service_output(tmp_path / "mineru_output")

    def responder(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_make_success_response(out_dir))

    with _PatchedAsyncClient(_mock_transport(responder)):
        client = _make_client(tmp_path)
        await client.process(src)

    # The staged copy lives under input_dir with a uuid prefix; nothing should remain.
    staged_remaining = list((tmp_path / "input").glob("*report.pdf"))
    assert staged_remaining == []


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_service_reports_failure_yields_failed_result(tmp_path: Path) -> None:
    src = _make_source_pdf(tmp_path)

    def responder(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "total_files": 1,
                "succeeded": 0,
                "failed": 1,
                "total_seconds": 0.1,
                "results": [
                    {
                        "source": "/data/input/xxx__report.pdf",
                        "success": False,
                        "error": "mineru CLI exit 1: model download failed",
                        "processing_seconds": 0.1,
                    }
                ],
            },
        )

    with _PatchedAsyncClient(_mock_transport(responder)):
        client = _make_client(tmp_path)
        result = await client.process(src)

    assert result.success is False
    assert "model download failed" in (result.error_message or "")


@pytest.mark.asyncio
async def test_service_empty_results_yields_failed_result(tmp_path: Path) -> None:
    src = _make_source_pdf(tmp_path)

    def responder(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"total_files": 0, "succeeded": 0, "failed": 0, "results": [], "total_seconds": 0})

    with _PatchedAsyncClient(_mock_transport(responder)):
        client = _make_client(tmp_path)
        result = await client.process(src)

    assert result.success is False
    assert "no results" in (result.error_message or "").lower()


@pytest.mark.asyncio
async def test_service_missing_content_list_path_yields_failed_result(tmp_path: Path) -> None:
    src = _make_source_pdf(tmp_path)
    out_dir = tmp_path / "mineru_output" / "report" / "auto"
    out_dir.mkdir(parents=True)

    def responder(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "total_files": 1,
                "succeeded": 1,
                "failed": 0,
                "total_seconds": 0.1,
                "results": [
                    {
                        "source": "/data/input/xxx__report.pdf",
                        "success": True,
                        "output_dir": str(out_dir),
                        "markdown_path": str(out_dir / "report.md"),
                        # content_list_path deliberately absent
                        "processing_seconds": 0.1,
                    }
                ],
            },
        )

    with _PatchedAsyncClient(_mock_transport(responder)):
        client = _make_client(tmp_path)
        result = await client.process(src)

    assert result.success is False
    assert "content_list_path" in (result.error_message or "")


@pytest.mark.asyncio
async def test_http_5xx_response_yields_failed_result(tmp_path: Path) -> None:
    src = _make_source_pdf(tmp_path)

    def responder(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal error")

    with _PatchedAsyncClient(_mock_transport(responder)):
        client = _make_client(tmp_path)
        result = await client.process(src)

    # process() catches the raise_for_status() and returns a failed result
    # rather than propagating the exception.
    assert result.success is False
    assert result.error_message


@pytest.mark.asyncio
async def test_missing_source_file_raises_file_not_found(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    with pytest.raises(FileNotFoundError):
        await client.process(tmp_path / "does_not_exist.pdf")


@pytest.mark.asyncio
async def test_layout_parse_failure_surfaces_as_failed_result(tmp_path: Path) -> None:
    src = _make_source_pdf(tmp_path)
    # Build an output dir where the content_list.json is malformed so
    # the layout parser raises MineruLayoutParseError.
    out_dir = tmp_path / "mineru_output" / "report" / "auto"
    out_dir.mkdir(parents=True)
    (out_dir / "report.md").write_text("# x", encoding="utf-8")
    (out_dir / "report_content_list.json").write_text("garbage", encoding="utf-8")

    def responder(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "total_files": 1,
                "succeeded": 1,
                "failed": 0,
                "total_seconds": 0.1,
                "results": [
                    {
                        "source": "/data/input/xxx__report.pdf",
                        "success": True,
                        "output_dir": str(out_dir),
                        "markdown_path": str(out_dir / "report.md"),
                        "content_list_path": str(out_dir / "report_content_list.json"),
                        "processing_seconds": 0.1,
                    }
                ],
            },
        )

    with _PatchedAsyncClient(_mock_transport(responder)):
        client = _make_client(tmp_path)
        result = await client.process(src)
    assert result.success is False
    assert "not valid JSON" in (result.error_message or "")


# ---------------------------------------------------------------------------
# Config / env handling
# ---------------------------------------------------------------------------


def test_service_url_resolves_env_then_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("MINERU_SERVICE_URL", raising=False)
    client = MineruHttpClient(input_dir=tmp_path / "in", output_dir=tmp_path / "out")
    assert client.service_url == "http://mineru:8104"

    monkeypatch.setenv("MINERU_SERVICE_URL", "http://other-host:9000/")
    client = MineruHttpClient(input_dir=tmp_path / "in", output_dir=tmp_path / "out")
    assert client.service_url == "http://other-host:9000"


def test_explicit_service_url_takes_precedence_over_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MINERU_SERVICE_URL", "http://env-host:1234")
    client = MineruHttpClient(
        service_url="http://explicit:5678",
        input_dir=tmp_path / "in",
        output_dir=tmp_path / "out",
    )
    assert client.service_url == "http://explicit:5678"


def test_is_supported_extension_classmethod() -> None:
    assert MineruHttpClient.is_supported_extension(Path("foo.pdf")) is True
    assert MineruHttpClient.is_supported_extension(Path("foo.PDF")) is True
    assert MineruHttpClient.is_supported_extension(Path("foo.docx")) is True
    assert MineruHttpClient.is_supported_extension(Path("foo.html")) is False
    assert MineruHttpClient.is_supported_extension(Path("foo.txt")) is False
    assert MineruHttpClient.is_supported_extension(Path("foo.png")) is True


def test_mineru_service_error_is_runtime_error() -> None:
    assert issubclass(MineruServiceError, RuntimeError)
