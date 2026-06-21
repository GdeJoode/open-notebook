"""Track J.3 AC5 — the per-document ``private`` flag plumbs from the upload
form through ``SourceCreate`` to the persisted source + command args.

Two layers, both fast (no DB):

1. ``parse_source_form_data`` maps the ``private`` form string to a
   ``SourceCreate.private`` bool (the string -> bool seam).
2. An integration-style POST through the upload router asserts the resolved
   bool reaches ``source_svc.create`` and the ``process_source`` command args.

The full end-to-end "re-fetch the source and read private==True off the DB"
path is exercised in the J.6 integration suite (it needs a live worker + DB);
here we stop at the service boundary, which is where the plumbing lives.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from app_main.api.rate_limit import limiter
from app_main.api.routers.sources_upload import router
from app_main.dependencies import (
    get_notebook_service,
    get_source_service,
    get_transformation_service,
)
from app_main.services.notebook_service import NotebookService
from app_main.services.source_service import SourceService
from app_main.services.transformation_service import TransformationService
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Integration: POST private=<str> -> SourceCreate.private bool -> create()
# + command args. Goes through the real FastAPI Form binding (and thus the
# real ``parse_source_form_data`` / ``str_to_bool``), which is the faithful way
# to exercise the string -> bool seam — a direct call leaks unfilled Form(...)
# sentinels for every other form param.
# ---------------------------------------------------------------------------


def _make_client(source_svc: AsyncMock) -> TestClient:
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router, prefix="/sources")

    notebook_svc = AsyncMock(spec=NotebookService)
    transformation_svc = AsyncMock(spec=TransformationService)

    app.dependency_overrides[get_source_service] = lambda: source_svc
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_transformation_service] = (
        lambda: transformation_svc
    )
    return TestClient(app)


@pytest.fixture(autouse=True)
def _generous_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_RPM", "100000")


def _make_created_stub() -> AsyncMock:
    created = AsyncMock()
    created.id = "source:abc"
    created.title = "Processing..."
    created.topics = []
    created.created = "2026-06-21T00:00:00Z"
    created.updated = "2026-06-21T00:00:00Z"
    return created


def _post_text(client: TestClient, data: dict) -> "object":
    """POST a text source through the async path with the worker mocked out."""
    payload = {"type": "text", "content": "hello world", "async_processing": "true"}
    payload.update(data)
    with patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(return_value="command:xyz"),
    ), patch(
        "surrealdb_service.connection.execute_query",
        new=AsyncMock(return_value=None),
    ):
        return client.post("/sources/", data=payload)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("true", True),
        ("True", True),
        ("1", True),
        ("yes", True),
        ("on", True),
        ("false", False),
        ("0", False),
        ("no", False),
    ],
)
def test_post_private_string_to_bool(raw: str, expected: bool):
    """The ``private`` form string parses to a bool on the persisted source.

    Exercises the real FastAPI Form binding -> ``parse_source_form_data`` ->
    ``str_to_bool`` -> ``SourceCreate.private`` -> ``source_svc.create`` payload.
    """
    source_svc = AsyncMock(spec=SourceService)
    source_svc.create.return_value = _make_created_stub()
    client = _make_client(source_svc)

    resp = _post_text(client, {"private": raw})

    assert resp.status_code == 200, resp.text
    create_payload = source_svc.create.await_args.args[0]
    assert create_payload["private"] is expected


def test_post_private_true_threads_to_create_and_command(
    monkeypatch: pytest.MonkeyPatch,
):
    """POST /sources with private=true persists private on create + command.

    The async path is used (``async_processing=true``) so the request returns
    right after submitting the command — we patch ``submit_command_job`` to a
    controlled value and inspect the create payload + the command args it was
    handed.
    """
    source_svc = AsyncMock(spec=SourceService)
    created = AsyncMock()
    created.id = "source:abc"
    created.title = "Processing..."
    created.topics = []
    created.created = "2026-06-21T00:00:00Z"
    created.updated = "2026-06-21T00:00:00Z"
    source_svc.create.return_value = created
    client = _make_client(source_svc)

    submit = AsyncMock(return_value="command:xyz")
    with patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=submit,
    ), patch(
        "surrealdb_service.connection.execute_query",
        new=AsyncMock(return_value=None),
    ):
        resp = client.post(
            "/sources/",
            data={
                "type": "text",
                "content": "hello world",
                "private": "true",
                "async_processing": "true",
            },
        )

    assert resp.status_code == 200, resp.text

    # create() received the private flag.
    source_svc.create.assert_awaited_once()
    create_payload = source_svc.create.await_args.args[0]
    assert create_payload["private"] is True

    # The process_source command args carry private + content_state.private.
    submit.assert_awaited_once()
    command_args = submit.await_args.args[2]
    assert command_args["private"] is True
    assert command_args["content_state"]["private"] is True


def test_post_private_default_false(monkeypatch: pytest.MonkeyPatch):
    """Without the flag, the source is created with private=False."""
    source_svc = AsyncMock(spec=SourceService)
    created = AsyncMock()
    created.id = "source:def"
    created.title = "Processing..."
    created.topics = []
    created.created = "2026-06-21T00:00:00Z"
    created.updated = "2026-06-21T00:00:00Z"
    source_svc.create.return_value = created
    client = _make_client(source_svc)

    with patch(
        "app_main.services.command_service.CommandService.submit_command_job",
        new=AsyncMock(return_value="command:xyz"),
    ), patch(
        "surrealdb_service.connection.execute_query",
        new=AsyncMock(return_value=None),
    ):
        resp = client.post(
            "/sources/",
            data={
                "type": "text",
                "content": "hello world",
                "async_processing": "true",
            },
        )

    assert resp.status_code == 200, resp.text
    create_payload = source_svc.create.await_args.args[0]
    assert create_payload["private"] is False
