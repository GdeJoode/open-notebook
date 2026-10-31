"""Eén weg naar een model: via ``ModelGate`` naar llama-swap (F2-08, ADR 0006).

Stopconditie F2-08. De test faalt:

1. bij een Ollama-eindpunt (``/api/generate``, ``/api/chat``, ``/api/embed``,
   ``/api/embeddings``), de Ollama-poort ``11434`` of ``host.docker.internal`` in
   ``apps/``, ``packages/`` of ``pipelines/``;
2. bij code buiten de poort die een model zelf opbouwt of aanroept: een import
   van Esperanto of van een modelbibliotheek (OpenAI, Anthropic, Ollama,
   LangChain-aanbieders, LiteLLM), of van de llama-swap-client in
   ``llm_manager/providers/factory.py`` buiten ``llm_manager/aanroep.py``;
3. bij een aanroep van ``shared.model_routing`` (``call_llm``, ``embed_text``)
   buiten die module zelf.

Tijdelijke uitzondering (plan F2-08): ``packages/shared/src/shared/model_routing.py``,
dat met F2-09 vervalt.

Wat geen modelaanroep is en dus mag:

- een routeprefix van de fork zelf (``prefix="/api/embeddings"``) en een subpad
  zoals ``/api/chat/context``: Ollama kent geen subpaden;
- ``host.docker.internal`` in de blokkadelijst van de URL-controle van de
  bronkaart (``source_map_service.py``) en in de test daarvan: die lijst
  weigert zo'n adres juist;
- de import van ``openai`` in ``error_mapping.py`` (Track J, vervalt in F2-09):
  alleen uitzonderingsklassen, geen client;
- de import van ``langextract`` in de uitgeschakelde LangExtract-extractor: de
  constructor weigert altijd.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MAPPEN = ("apps", "packages", "pipelines")
SOORTEN = {
    ".py", ".ts", ".tsx", ".js", ".json", ".toml", ".yml", ".yaml", ".sh",
    ".cfg", ".ini", ".conf", ".env", ".example", ".txt",
}
OVERSLAAN = ("node_modules", "__pycache__", ".venv", "/app-main/data/")

#: De enige tijdelijke uitzondering (plan F2-08); vervalt in F2-09.
TIJDELIJK = {"packages/shared/src/shared/model_routing.py"}

VERBODEN = {
    "Ollama-eindpunt": re.compile(r"/api/(?:generate|chat|embed|embeddings)(?![\w/-])"),
    "poort 11434": re.compile(r"11434"),
    "host.docker.internal": re.compile(r"host\.docker\.internal"),
}

#: Een routeprefix van de fork is geen Ollama-eindpunt.
ROUTEPREFIX = re.compile(r"prefix\s*=\s*[\"']/api/")

#: (bestand, patroon): een blokkade, geen aanroep.
BLOKKADE = {
    ("apps/app-main/src/app_main/services/source_map_service.py", "host.docker.internal"),
    ("apps/app-main/tests/test_source_map_service.py", "host.docker.internal"),
}

#: Modules waarmee code zelf een model opbouwt of aanroept.
MODELBIBLIOTHEKEN = (
    "esperanto", "openai", "anthropic", "ollama", "litellm", "langextract",
    "langchain_openai", "langchain_ollama", "langchain_anthropic",
    "langchain_google_genai", "langchain_groq", "langchain_mistralai",
)

#: Track J, vervalt in F2-09: importeert alleen uitzonderingsklassen van de
#: OpenAI-SDK om fouten in te delen; er wordt geen client opgebouwd.
ALLEEN_UITZONDERINGEN = {"apps/app-main/src/app_main/services/model_routing/error_mapping.py"}

#: LangExtract is uitgeschakeld: de constructor weigert altijd
#: (``LangExtractUitgeschakeld``). Voorstel: verwijderen met F2-09.
LANGEXTRACT_UITGESCHAKELD = {
    "pipelines/ontology-extraction/src/ontology_extraction/extractors/langextract_extractor.py",
    "pipelines/ontology-extraction/src/ontology_extraction/extractors/example_builder.py",
}

#: De llama-swap-client mag alleen de dunne aanroeper van de poort gebruiken.
CLIENT_MODULE = "llm_manager.providers.factory"
CLIENT_NAMEN = {"LlamaSwapClient"}
CLIENT_TOEGESTAAN = {
    "packages/llm-manager/src/llm_manager/aanroep.py",
    "packages/llm-manager/src/llm_manager/providers/factory.py",
}


def _bestanden():
    for map_ in MAPPEN:
        for pad in sorted((ROOT / map_).rglob("*")):
            if not pad.is_file() or pad.suffix not in SOORTEN:
                continue
            posix = pad.as_posix()
            if any(o in posix for o in OVERSLAAN):
                continue
            yield pad, pad.relative_to(ROOT).as_posix()


def _is_test(rel: str) -> bool:
    naam = rel.rsplit("/", 1)[-1]
    return "/tests/" in rel or naam.startswith("test_") or naam == "conftest.py"


def test_geen_ollama_eindpunten():
    gevonden = []
    for pad, rel in _bestanden():
        if rel in TIJDELIJK:
            continue
        for nr, regel in enumerate(pad.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
            for naam, patroon in VERBODEN.items():
                if not patroon.search(regel):
                    continue
                if naam == "Ollama-eindpunt" and ROUTEPREFIX.search(regel):
                    continue
                if (rel, naam) in BLOKKADE:
                    continue
                gevonden.append(f"{rel}:{nr}: {naam}: {regel.strip()[:120]}")
    assert not gevonden, "Modelaanroep buiten llama-swap:\n" + "\n".join(gevonden)


def _imports(boom: ast.AST):
    for knoop in ast.walk(boom):
        if isinstance(knoop, ast.Import):
            for alias in knoop.names:
                yield knoop.lineno, alias.name, None
        elif isinstance(knoop, ast.ImportFrom) and knoop.module and knoop.level == 0:
            for alias in knoop.names:
                yield knoop.lineno, knoop.module, alias.name


def test_geen_model_buiten_de_poort():
    gevonden = []
    for pad, rel in _bestanden():
        if pad.suffix != ".py" or rel in TIJDELIJK:
            continue
        boom = ast.parse(pad.read_text(encoding="utf-8"), filename=rel)
        for nr, module, naam in _imports(boom):
            wortel = module.split(".")[0]
            if (wortel in MODELBIBLIOTHEKEN and not _is_test(rel)
                    and not (rel in ALLEEN_UITZONDERINGEN and wortel == "openai")
                    and not (rel in LANGEXTRACT_UITGESCHAKELD and wortel == "langextract")):
                gevonden.append(f"{rel}:{nr}: import {module}")
            raakt_client = (
                module == CLIENT_MODULE
                or (module == "llm_manager.providers" and naam in CLIENT_NAMEN)
                or (module == "llm_manager.providers" and naam == "factory")
            )
            if raakt_client and rel not in CLIENT_TOEGESTAAN and not _is_test(rel):
                gevonden.append(f"{rel}:{nr}: llama-swap-client buiten de poort")
            if module == "shared.model_routing" or (module == "shared" and naam == "model_routing"):
                if naam in {"call_llm", "embed_text", "model_routing"} or (
                    naam and naam.startswith("_call")
                ):
                    gevonden.append(f"{rel}:{nr}: shared.model_routing.{naam}")
    assert not gevonden, "Modelaanroep om de poort heen:\n" + "\n".join(gevonden)


def test_de_test_vindt_wat_hij_zoekt():
    """De patronen raken de echte vormen, en sparen de routes van de fork."""
    eindpunt = VERBODEN["Ollama-eindpunt"]
    for regel in ('f"{url}/api/generate"', "'/api/chat'", '"/api/embed"',
                  'f"{base}/api/embeddings",'):
        assert eindpunt.search(regel), regel
    assert not eindpunt.search('c.post("/api/chat/context")')
    assert ROUTEPREFIX.search('router, prefix="/api/embeddings", tags=["x"]')
    assert VERBODEN["poort 11434"].search("http://localhost:11434")
    boom = ast.parse(
        "import esperanto\nfrom llm_manager.providers.factory import LlamaSwapClient\n"
        "from shared.model_routing import call_llm\n"
    )
    modules = {m for _, m, _ in _imports(boom)}
    assert {"esperanto", CLIENT_MODULE, "shared.model_routing"} <= modules
