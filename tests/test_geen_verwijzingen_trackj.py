"""Geen Track J-routering, ``model_route``, modelrijen of oude privacyvelden meer (F2-09).

ADR 0006, 0007 en 0001. Weg zijn:

- de Track J-routering in ``app_main/services/model_routing/`` (resolver,
  failover, circuit breaker, rate limiter, foutvertaling, privacyresolver en de
  seeds), de router ``model_routes`` en de tabel ``model_route``;
- ``shared.model_routing`` met ``model_routing.yaml``;
- de modelrijen en standaardmodellen in de database (``ModelService``,
  ``ModelRepository``, ``DefaultModels``, ``ModelManager``), het providerregister
  met cloudsleutels in ``llm_manager/providers/registry.py`` en de routes
  ``/api/models/defaults``, ``/status`` en ``/providers`` met ``ModelResponse``;
- Esperanto;
- de velden ``private``, ``privacy_mode`` en ``default_privacy_mode`` en het
  formulierveld ``private`` (migratie 103);
- de privacykeuze in de interface (Public — Cloud API) met de parameter
  ``privacy``.

Deze test faalt zodra een van die namen terugkomt in de code, de interface of
de bouwbestanden, en toetst dat de vervallen routes HTTP 404 geven. Migraties
staan buiten de zoekmappen; de test van migratie 103 noemt de velden die hij
verwijdert.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEZE_TEST = Path(__file__).resolve()

MAPPEN = ("apps", "packages", "pipelines", "services", "frontend/src")
LOSSE_BESTANDEN = (
    "pyproject.toml", "uv.lock", "Dockerfile", "docker-compose.yml", ".env.example",
)
SOORTEN = {
    ".py", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".json", ".toml", ".yml",
    ".yaml", ".sh", ".cfg", ".ini", ".conf", ".txt", ".env", ".example", ".lock",
}
OVERSLAAN = ("node_modules", "__pycache__", ".venv", ".next", ".pytest_cache")

MIGRATIE_103 = "packages/surrealdb-service/tests/test_migration_103_trackj_privacyvelden.py"
#: De tests van migraties 64, 65 en 89 toetsen hun eigen migratie op ``private``.
MIGRATIETESTS_PRIVATE = (
    MIGRATIE_103,
    "packages/surrealdb-service/tests/test_migrations_roundtrip.py",
    "packages/surrealdb-service/tests/test_migration_65_strict_drift_guard.py",
    "packages/surrealdb-service/tests/test_migration_89_classification.py",
)

#: Uitzonderingen per patroon, nooit per bestand:
#: - de migratietests noemen het veld ``private`` (en 103 ook de velden en de
#:   tabel die hij verwijdert);
#: - ``services/summarization`` gebruikt Esperanto nog; die dienst vervalt in
#:   F2-23 (ADR 0015, correctie in voortgang F2-08).
UITGEZONDERD_PER_NAAM = {
    "veld private": MIGRATIETESTS_PRIVATE,
    "privacy_mode": (MIGRATIE_103,),
    "default_privacy_mode": (MIGRATIE_103,),
    "model_route": (MIGRATIE_103,),
    "esperanto": ("services/summarization/", "docker-compose.yml"),
}

VERBODEN = {
    # Uit de stopconditie van F2-09.
    "PrivacyMode": re.compile(r"\bPrivacyMode\b"),
    "privacy_mode": re.compile(r"\bprivacy_mode\b"),
    "default_privacy_mode": re.compile(r"\bdefault_privacy_mode\b"),
    "model_route": re.compile(r"\bmodel_routes?\b|\bModelRoute\w*"),
    "shared.model_routing": re.compile(r"\bshared[./]model_routing\b|model_routing\.yaml"),
    # De modules van de Track J-routering.
    "Track J-module": re.compile(
        r"\b(?:route_resolver|failover_executor|circuit_breaker|rate_limiter|error_mapping"
        r"|privacy_resolver|seed_chain_models)\b|model_routing[./]seed\b"
    ),
    "Track J-namen": re.compile(
        r"\b(?:resolve_route|RouteResolver|FailoverExecutor|resolve_privacy_mode"
        r"|record_routing_event|CircuitBreakerRegistry|ProviderRateLimiter)\b"
    ),
    "route /model-routes": re.compile(r"/model-routes\b"),
    # Modelrijen en standaardmodellen in de database (ADR 0007).
    "modelrijen": re.compile(
        r"\b(?:ModelService|ModelRepository|DefaultModels\w*|ModelManager|get_model_manager"
        r"|get_model_service|get_default_models_repo|ModelResponse)\b"
    ),
    "route /models/...": re.compile(r"/models/(?:defaults|status|providers)\b"),
    "providerregister": re.compile(
        r"\b(?:PROVIDER_CONFIGS|get_provider_config|api_key_env_var)\b|providers[./]registry\b"
    ),
    "esperanto": re.compile(r"\besperanto\b", re.I),
    # Oude privacyvelden en het formulierveld ``private``.
    "veld private": re.compile(
        r"\bsource\.private\b|\bisPrivate\b|append\(\s*['\"]private['\"]"
        r"|['\"]private['\"]\s*:|^\s*private\s*[:=]\s*(?:bool|boolean|str\s*=\s*Form|z\.boolean)\b"
        r"|^\s*private\?\s*:|getattr\([^)]*['\"]private['\"]|\.private\b",
        re.M,
    ),
    # De privacykeuze in de interface en de parameter ``privacy`` (Public — Cloud
    # API / NVIDIA / Ollama).
    "privacykeuze": re.compile(
        r"\bPrivacyLevel\b|\bprivacyLevel\b|\bprivacy\??\s*[:=]|\{\s*privacy\s*\}"
        r"|Cloud API|['\"]public['\"]\s*\|\s*['\"]internal['\"]"
    ),
    "privacy-interface": re.compile(r"\bPrivacyModeToggle\b|privacy-mode\b|\bPrivacyValue\b"),
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


def test_geen_verwijzingen_naar_trackj() -> None:
    treffers = []
    for rel, pad in _bestanden():
        if pad.resolve() == DEZE_TEST:
            continue
        tekst = pad.read_text(encoding="utf-8", errors="ignore")
        for naam, patroon in VERBODEN.items():
            if rel.startswith(UITGEZONDERD_PER_NAAM.get(naam, ())):
                continue
            for m in patroon.finditer(tekst):
                regel = tekst.count("\n", 0, m.start()) + 1
                treffers.append(f"{rel}:{regel}: {naam}: {m.group(0).strip()}")
    assert not treffers, "verwijderde naam nog genoemd:\n" + "\n".join(treffers)


def test_verwijderde_paden_bestaan_niet() -> None:
    routing = "apps/app-main/src/app_main/services/model_routing"
    for rel in (
        *(f"{routing}/{naam}.py" for naam in (
            "route_resolver", "failover_executor", "circuit_breaker", "rate_limiter",
            "error_mapping", "privacy_resolver", "seed", "seed_chain_models")),
        "apps/app-main/src/app_main/api/routers/model_routes.py",
        "apps/app-main/src/app_main/services/model_service.py",
        "packages/shared/src/shared/model_routing.py",
        "packages/shared/src/shared/model_routing.yaml",
        "packages/surrealdb-service/src/surrealdb_service/repositories/model_route.py",
        "packages/surrealdb-service/src/surrealdb_service/repositories/model.py",
        "packages/llm-manager/src/llm_manager/providers/registry.py",
        "packages/llm-manager/src/llm_manager/manager.py",
        "frontend/src/app/(dashboard)/settings/model-routing",
        "frontend/src/components/routing",
    ):
        assert not (ROOT / rel).exists(), rel


def test_de_zoektest_ziet_de_mappen() -> None:
    """Controle: een zoektest die niets leest, slaagt voor niets."""
    gezien = {rel for rel, _ in _bestanden()}
    for verwacht in (
        "apps/app-main/src/app_main/api/app.py",
        "packages/llm-manager/pyproject.toml",
        "frontend/src/lib/api/models.ts",
        "pyproject.toml",
    ):
        assert verwacht in gezien, verwacht
    for naam, voorbeeld in (
        ("PrivacyMode", "from x import PrivacyMode"),
        ("model_route", "DEFINE TABLE model_route"),
        ("model_route", "from app_main.api.routers import model_routes"),
        ("shared.model_routing", "from shared.model_routing import call_llm"),
        ("Track J-module", "from app_main.services.model_routing.route_resolver import LLMTask"),
        ("Track J-module", "from app_main.services.model_routing.seed import seed"),
        ("route /model-routes", "apiClient.get('/model-routes')"),
        ("modelrijen", "from shared.models import DefaultModels"),
        ("route /models/...", "apiClient.get('/models/defaults')"),
        ("providerregister", 'api_key_env_var="OPENAI_API_KEY"'),
        ("esperanto", '"esperanto>=2.4.1",'),
        ("veld private", "formData.append('private', 'true')"),
        ("veld private", "    private: bool = False"),
        ("veld private", '    private: str = Form("false"),'),
        ("veld private", "  private?: boolean"),
        ("veld private", '{"title": "x", "private": False}'),
        ("veld private", "{source.private && <Badge />}"),
        ("privacy-interface", "import { PrivacyModeToggle } from '@/components/routing'"),
        ("esperanto", "De client gebruikt httpx en niet Esperanto."),
        ("veld private", 'bool(getattr(source, "private", False))'),
        ("veld private", "if (doc.private) {"),
        ("privacykeuze", "export type PrivacyLevel = 'public' | 'internal' | 'confidential'"),
        ("privacykeuze", "  privacy?: PrivacyLevel"),
        ("privacykeuze", "      ...(privacy ? { privacy } : {}),"),
        ("privacykeuze", "<span>— Cloud API (sneller, beter)</span>"),
    ):
        assert VERBODEN[naam].search(voorbeeld), (naam, voorbeeld)
    # Wat blijft: de dunne aanroeper, het IP-filter en de modelpoort.
    for blijft in (
        "from app_main.services.model_routing.llm_call import get_aanroeper",
        "or ip.is_private",
        "GET /api/models",
    ):
        assert not any(p.search(blijft) for p in VERBODEN.values()), blijft


@pytest.mark.parametrize(
    "methode, pad",
    [
        ("GET", "/api/model-routes"),
        ("PUT", "/api/model-routes/chat"),
        ("GET", "/api/models/defaults"),
        ("PUT", "/api/models/defaults"),
        ("GET", "/api/models/status"),
        ("GET", "/api/models/providers"),
    ],
)
def test_vervallen_routes_geven_404(monkeypatch, methode, pad) -> None:
    """Ook met een sleutel ``beheerder`` bestaan de oude routes niet meer."""
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
    assert resp.status_code in (404, 405), (pad, resp.status_code)
    if methode == "GET":
        assert resp.status_code == 404, pad


def test_preprocessing_kent_geen_privacykeuze(monkeypatch) -> None:
    """De parameter ``privacy`` (Public — Cloud API) bestaat niet meer: 422."""
    from app_main.api import agent_auth
    from app_main.api.app import create_app
    from app_main.services.agents.audit_service import AgentAuditService
    from fastapi.testclient import TestClient

    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    monkeypatch.setattr(
        agent_auth.AgentKeyService,
        "authenticate",
        AsyncMock(return_value={"id": "agent_keys:proef", "agent_id": "proef",
                                "permission": "beheerder", "person": "Proefpersoon",
                                "label": "proef"}),
    )
    client = TestClient(create_app(), raise_server_exceptions=False)
    resp = client.post("/api/preprocessing/run", headers={"Authorization": "Bearer ak_proef"},
                       json={"source_id": "source:proef", "privacy": "public"})
    assert resp.status_code == 422
