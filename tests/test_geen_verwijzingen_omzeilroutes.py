"""Geen routes of diensten meer die het classificatiefilter omzeilen (F2-02).

ADR 0002 D en E, ADR 0012. Weg zijn: de router ``services_proxy``
(``/api/services/*``), de drie MCP-servers in de fork (``surrealdb_service.mcp``,
``llm_manager.mcp``, ``file_manager.mcp``), de losse API
``surrealdb_service.api`` met de script-ingang ``surrealdb-service``, de losse
API ``file_manager.api`` (0.0.0.0:8002) met de script-ingang ``file-manager``, en
de extractiedienst ``services/extraction``. De querylogica achter de MCP-tools
blijft in de repositories (``SearchRepository``, ``GraphRepository``).

Deze test faalt zodra een van die namen terugkomt in de code, de interface of
de bouw- en Compose-bestanden, en toetst dat ``/api/services/...`` HTTP 404
geeft.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEZE_TEST = Path(__file__).resolve()

#: De mappen uit de verwijdercontrole V (plan §1), plus de bestanden buiten de
#: code waarin een dienst, ingang of afhankelijkheid kan blijven hangen.
MAPPEN = ("apps", "packages", "pipelines", "services", "frontend/src", ".github")
LOSSE_BESTANDEN = (
    "docker-compose.yml",
    "setup_guide/docker-compose.yml",
    "Dockerfile",
    ".env.example",
    "Makefile",
    "pyproject.toml",
    ".mcp.json",
)
SOORTEN = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".json", ".toml", ".yml",
    ".yaml", ".sh", ".cfg", ".ini", ".conf", ".txt", ".env", ".example",
}
OVERSLAAN = ("node_modules", "__pycache__", ".venv", ".next", "/fixtures/")

VERBODEN = {
    "services_proxy": re.compile(r"\bservices_proxy\b"),
    "/api/services": re.compile(r"/api/services\b"),
    "interface /services/": re.compile(r"[\"'`]/services/"),
    "surrealdb_service.mcp": re.compile(r"\bsurrealdb_service[./]mcp\b"),
    "surrealdb_service.api": re.compile(r"\bsurrealdb_service[./]api\b"),
    "llm_manager.mcp": re.compile(r"\bllm_manager[./]mcp\b"),
    "file_manager.mcp": re.compile(r"\bfile_manager[./]mcp\b"),
    "file_manager.api": re.compile(r"\bfile_manager[./]api\b"),
    "from <pakket> import mcp/api": re.compile(
        r"\bfrom\s+(?:surrealdb_service|llm_manager|file_manager)\s+import\s+\(?\s*(?:mcp|api)\b"
    ),
    "script-ingang file-manager": re.compile(r"^\s*file-manager\s*=\s*[\"']", re.M),
    "script-ingang surrealdb-service": re.compile(r"^\s*surrealdb-service\s*=\s*[\"']", re.M),
    "script-ingang *-mcp": re.compile(r"\b(?:surrealdb|llm-manager|file-manager)-mcp\b"),
    "services/extraction": re.compile(r"\bservices/extraction(?![\w-])"),
    "EXTRACTION_SERVICE_URL": re.compile(r"\bEXTRACTION_SERVICE_URL\b"),
    "poort 8103": re.compile(r"\b8103\b"),
}


def _bestanden():
    for map_ in MAPPEN:
        for pad in (ROOT / map_).rglob("*"):
            rel = pad.relative_to(ROOT).as_posix()
            if not pad.is_file() or any(o in f"/{rel}" for o in OVERSLAAN):
                continue
            if pad.suffix in SOORTEN or pad.name.startswith("Dockerfile"):
                yield rel, pad
    for rel in LOSSE_BESTANDEN:
        pad = ROOT / rel
        if pad.is_file():
            yield rel, pad


def test_geen_verwijzingen_naar_omzeilroutes() -> None:
    treffers = []
    for rel, pad in _bestanden():
        if pad.resolve() == DEZE_TEST:
            continue
        tekst = pad.read_text(encoding="utf-8", errors="ignore")
        for naam, patroon in VERBODEN.items():
            for m in patroon.finditer(tekst):
                regel = tekst.count("\n", 0, m.start()) + 1
                treffers.append(f"{rel}:{regel}: {naam}")
    assert not treffers, "verwijderde route of dienst nog genoemd:\n" + "\n".join(treffers)


def test_verwijderde_paden_bestaan_niet() -> None:
    for rel in (
        "apps/app-main/src/app_main/api/routers/services_proxy.py",
        "frontend/src/lib/api/services.ts",
        "packages/surrealdb-service/src/surrealdb_service/mcp",
        "packages/surrealdb-service/src/surrealdb_service/api",
        "packages/llm-manager/src/llm_manager/mcp",
        "packages/file-manager/src/file_manager/mcp",
        "packages/file-manager/src/file_manager/api",
        "services/extraction",
    ):
        assert not (ROOT / rel).exists(), rel


def test_de_zoektest_ziet_de_mappen() -> None:
    """Controle: een zoektest die niets leest, slaagt voor niets."""
    gezien = {rel for rel, _ in _bestanden()}
    for verwacht in (
        "apps/app-main/src/app_main/api/app.py",
        "packages/surrealdb-service/pyproject.toml",
        "frontend/src/lib/api/client.ts",
        "docker-compose.yml",
    ):
        assert verwacht in gezien, verwacht
    for naam, voorbeeld in (
        ("services_proxy", "from app_main.api.routers import services_proxy"),
        ("/api/services", 'client.get("/api/services/health")'),
        ("interface /services/", "apiClient.get('/services/health')"),
        ("surrealdb_service.mcp", "surrealdb-mcp = \"surrealdb_service.mcp.server:main\""),
        ("surrealdb_service.api", "from surrealdb_service.api.app import create_app"),
        ("script-ingang surrealdb-service", 'surrealdb-service = "x:main"'),
        ("file_manager.api", "from file_manager.api.routers import files"),
        ("from <pakket> import mcp/api", "from surrealdb_service import mcp"),
        ("from <pakket> import mcp/api", "from llm_manager import api"),
        ("script-ingang file-manager", 'file-manager = "file_manager.api.app:main"'),
        ("services/extraction", "dockerfile: services/extraction/Dockerfile"),
    ):
        assert VERBODEN[naam].search(voorbeeld), naam
    # Een werkruimte-bron is geen script-ingang.
    for naam, bron in (
        ("script-ingang surrealdb-service", "surrealdb-service = { workspace = true }"),
        ("script-ingang file-manager", "file-manager = { workspace = true }"),
    ):
        assert not VERBODEN[naam].search(bron), naam
    # Een ander pakket met hetzelfde voorvoegsel telt niet.
    assert not VERBODEN["services/extraction"].search("app_main/services/extraction_service/")


@pytest.mark.parametrize(
    "methode, pad",
    [
        ("GET", "/api/services/health"),
        ("GET", "/api/services/ontologies"),
        ("POST", "/api/services/extract"),
        ("POST", "/api/services/vocabularies/refresh"),
    ],
)
def test_api_services_geeft_404(monkeypatch, methode, pad) -> None:
    """Ook met een sleutel ``beheerder`` bestaat ``/api/services/...`` niet meer."""
    from app_main.api import agent_auth
    from app_main.api.app import create_app
    from app_main.services.agents.audit_service import AgentAuditService
    from fastapi.testclient import TestClient

    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(
            return_value={
                "id": "agent_keys:proef",
                "agent_id": "proef",
                "permission": "beheerder",
                "person": "Proefpersoon",
                "label": "proef",
            }
        ),
    )
    client = TestClient(create_app(), raise_server_exceptions=False)
    resp = client.request(methode, pad, headers={"Authorization": "Bearer ak_proef"}, json={})
    assert resp.status_code == 404
