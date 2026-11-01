"""Eén weg naar een model: via ``ModelGate`` naar llama-swap (F2-08, ADR 0006).

Stopconditie F2-08. De test faalt:

1. bij een Ollama-eindpunt (``/api/generate``, ``/api/chat``, ``/api/embed``,
   ``/api/embeddings``), de Ollama-poort ``11434`` of ``host.docker.internal`` in
   ``apps/``, ``packages/`` of ``pipelines/``;
2. bij een OpenAI-compatibel eindpunt (``/v1/chat``, ``/v1/embeddings``,
   ``/v1/completions``) of het adres ``llama-swap:8080`` in code buiten
   ``llm_manager/providers/factory.py`` (in Python: tekstconstanten, geen
   docstrings; tests tellen niet mee, die bootsen llama-swap na);
3. bij code buiten de poort die een model zelf opbouwt of aanroept: een import
   van Esperanto of van een modelbibliotheek (OpenAI, Anthropic, Ollama,
   LangChain-aanbieders, LiteLLM, LangExtract), of van de llama-swap-client in
   ``llm_manager/providers/factory.py`` buiten ``llm_manager/aanroep.py``. Ook een
   relatieve import en ``importlib.import_module``/``__import__`` met zo'n module
   als tekst tellen;
4. bij ``import shared.model_routing`` of een aanroep daaruit (``call_llm``,
   ``embed_text``, ``_call_*``); die module is in F2-09 verwijderd.

Er is geen uitzondering meer: ``shared/model_routing.py``, de Track
J-foutvertaling (``error_mapping.py``) en LangExtract zijn in F2-09 verwijderd.

Wat geen modelaanroep is en dus mag:

- een routeprefix van de fork zelf (``prefix="/api/embeddings"``) en een subpad
  zoals ``/api/chat/context``: Ollama kent geen subpaden;
- ``host.docker.internal`` in de blokkadelijst van de URL-controle van de
  bronkaart (``source_map_service.py``) en in de test daarvan: die lijst
  weigert zo'n adres juist.
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

VERBODEN = {
    "Ollama-eindpunt": re.compile(r"/api/(?:generate|chat|embed|embeddings)(?![\w/-])"),
    "poort 11434": re.compile(r"11434"),
    "host.docker.internal": re.compile(r"host\.docker\.internal"),
}

#: Alleen in code buiten de client (en niet in tests): het OpenAI-pad en het adres.
VERBODEN_BUITEN_CLIENT = {
    "OpenAI-eindpunt": re.compile(r"/v1/(?:chat|embeddings|completions)\b"),
    "adres llama-swap": re.compile(r"llama-swap:8080"),
}
CLIENT_BESTAND = "packages/llm-manager/src/llm_manager/providers/factory.py"

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

#: De llama-swap-client mag alleen de dunne aanroeper van de poort gebruiken.
CLIENT_MODULE = "llm_manager.providers.factory"
CLIENT_NAMEN = {"LlamaSwapClient"}
CLIENT_TOEGESTAAN = {
    "packages/llm-manager/src/llm_manager/aanroep.py",
    CLIENT_BESTAND,
}

MODEL_ROUTING = "shared.model_routing"


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


def _module_van(rel: str) -> list[str]:
    """De modulenaam van een bestand onder ``src/`` als delen (voor relatieve imports)."""
    delen = rel[:-3].split("/") if rel.endswith(".py") else rel.split("/")
    if "src" in delen:
        delen = delen[delen.index("src") + 1:]
    return delen


def _imports(boom: ast.AST, rel: str = ""):
    """(regel, module, naam) per import, relatieve imports opgelost.

    Ook ``importlib.import_module("x")`` en ``__import__("x")`` met een tekst
    als argument tellen als import van ``x``.
    """
    for knoop in ast.walk(boom):
        if isinstance(knoop, ast.Import):
            for alias in knoop.names:
                yield knoop.lineno, alias.name, None
        elif isinstance(knoop, ast.ImportFrom):
            if knoop.level == 0:
                module = knoop.module or ""
            else:
                basis = _module_van(rel)
                if not rel.endswith("__init__.py"):
                    basis = basis[:-1]
                basis = basis[: len(basis) - (knoop.level - 1)] if knoop.level > 1 else basis
                module = ".".join([*basis, *([knoop.module] if knoop.module else [])])
            for alias in knoop.names:
                yield knoop.lineno, module, alias.name
                if knoop.level and not knoop.module:
                    # ``from . import factory``: de naam is zelf een module.
                    yield knoop.lineno, f"{module}.{alias.name}", None
        elif isinstance(knoop, ast.Call) and knoop.args:
            functie = knoop.func
            naam = getattr(functie, "attr", None) or getattr(functie, "id", None)
            eerste = knoop.args[0]
            if naam in ("import_module", "__import__") and isinstance(eerste, ast.Constant) \
                    and isinstance(eerste.value, str):
                yield knoop.lineno, eerste.value, None


def _docstrings(boom: ast.AST) -> set[int]:
    ids = set()
    for knoop in ast.walk(boom):
        if isinstance(knoop, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            romp = getattr(knoop, "body", [])
            if romp and isinstance(romp[0], ast.Expr) and isinstance(romp[0].value, ast.Constant):
                ids.add(id(romp[0].value))
    return ids


def _bevindingen_tekst(rel: str, inhoud: str) -> list[str]:
    gevonden = []
    for nr, regel in enumerate(inhoud.splitlines(), 1):
        for naam, patroon in VERBODEN.items():
            if not patroon.search(regel):
                continue
            if naam == "Ollama-eindpunt" and ROUTEPREFIX.search(regel):
                continue
            if (rel, naam) in BLOKKADE:
                continue
            gevonden.append(f"{rel}:{nr}: {naam}: {regel.strip()[:120]}")
    if _is_test(rel) or rel == CLIENT_BESTAND:
        return gevonden
    if rel.endswith(".py"):
        boom = ast.parse(inhoud)
        docs = _docstrings(boom)
        teksten = [(k.lineno, k.value) for k in ast.walk(boom)
                   if isinstance(k, ast.Constant) and isinstance(k.value, str)
                   and id(k) not in docs]
    else:
        teksten = list(enumerate(inhoud.splitlines(), 1))
    for nr, tekst in teksten:
        for naam, patroon in VERBODEN_BUITEN_CLIENT.items():
            if patroon.search(tekst):
                gevonden.append(f"{rel}:{nr}: {naam}: {tekst.strip()[:120]}")
    return gevonden


def _bevindingen_imports(rel: str, inhoud: str) -> list[str]:
    gevonden = []
    boom = ast.parse(inhoud, filename=rel)
    test = _is_test(rel)
    for nr, module, naam in _imports(boom, rel):
        wortel = module.split(".")[0]
        if wortel in MODELBIBLIOTHEKEN and not test:
            gevonden.append(f"{rel}:{nr}: import {module}")
        raakt_client = (
            module == CLIENT_MODULE
            or (module == "llm_manager.providers" and naam in CLIENT_NAMEN | {"factory"})
        )
        if raakt_client and rel not in CLIENT_TOEGESTAAN and not test:
            gevonden.append(f"{rel}:{nr}: llama-swap-client buiten de poort")
        if not test and (module == MODEL_ROUTING or (module == "shared" and naam == "model_routing")):
            if naam is None or naam in {"call_llm", "embed_text", "model_routing"} or (
                naam.startswith("_call")
            ):
                gevonden.append(f"{rel}:{nr}: {MODEL_ROUTING}.{naam or '*'}")
    return gevonden


def test_geen_ollama_eindpunten():
    gevonden = []
    for pad, rel in _bestanden():
        gevonden += _bevindingen_tekst(rel, pad.read_text(encoding="utf-8", errors="ignore"))
    assert not gevonden, "Modelaanroep buiten llama-swap:\n" + "\n".join(gevonden)


#: Review F2-08: de stoptest geldt ook voor de modelserver en de Compose-opzet (F2-11).
CONFIGURATIE = ("services/llama-swap", "docker-compose.signaalsysteem.yml")


def test_geen_ollama_in_llama_swap_en_compose():
    gevonden = []
    for naam in CONFIGURATIE:
        pad = ROOT / naam
        for bestand in sorted(pad.rglob("*")) if pad.is_dir() else [pad]:
            if not bestand.is_file():
                continue
            tekst = bestand.read_text(encoding="utf-8", errors="ignore")
            rel = bestand.relative_to(ROOT).as_posix()
            for nr, regel in enumerate(tekst.splitlines(), 1):
                for soort, patroon in {**VERBODEN, "ollama": re.compile(r"(?i)ollama")}.items():
                    if patroon.search(regel):
                        gevonden.append(f"{rel}:{nr}: {soort}")
    assert not gevonden, "Ollama in de configuratie:\n" + "\n".join(gevonden)


def test_geen_model_buiten_de_poort():
    gevonden = []
    for pad, rel in _bestanden():
        if pad.suffix != ".py":
            continue
        gevonden += _bevindingen_imports(rel, pad.read_text(encoding="utf-8"))
    assert not gevonden, "Modelaanroep om de poort heen:\n" + "\n".join(gevonden)


BRON = "apps/app-main/src/app_main/services/iets.py"


def test_de_test_vindt_wat_hij_zoekt():
    """De patronen raken de echte vormen, en sparen de routes van de fork."""
    eindpunt = VERBODEN["Ollama-eindpunt"]
    for regel in ('f"{url}/api/generate"', "'/api/chat'", '"/api/embed"',
                  'f"{base}/api/embeddings",'):
        assert eindpunt.search(regel), regel
    assert not eindpunt.search('c.post("/api/chat/context")')
    assert ROUTEPREFIX.search('router, prefix="/api/embeddings", tags=["x"]')
    assert VERBODEN["poort 11434"].search("http://localhost:11434")


def test_de_omzeilvormen_uit_de_review_worden_gevonden():
    """De vormen uit de probe van de reviewer (F2-08)."""
    omzeilvormen = {
        "import shared.model_routing\n": "shared.model_routing",
        "import importlib\nm = importlib.import_module('esperanto')\n": "import esperanto",
        "m = __import__('openai')\n": "import openai",
        "from ..providers.factory import LlamaSwapClient\n": "client buiten de poort",
    }
    for bron, verwacht in omzeilvormen.items():
        rel = "packages/llm-manager/src/llm_manager/api/iets.py" if "providers" in bron else BRON
        assert any(verwacht in b for b in _bevindingen_imports(rel, bron)), bron
    # ``from . import factory`` in het pakket ``providers``.
    rel = "packages/llm-manager/src/llm_manager/providers/iets.py"
    assert _bevindingen_imports(rel, "from . import factory\n")
    # Het OpenAI-pad en het adres van llama-swap in code buiten de client.
    for bron in ('client.post("https://api.openai.com/v1/chat/completions")\n',
                 'url = "http://llama-swap:8080/v1"\n',
                 'pad = "/v1/embeddings"\n'):
        assert _bevindingen_tekst(BRON, bron), bron
    # Maar niet in een docstring, en niet in de client zelf.
    assert not _bevindingen_tekst(BRON, '"""Gaat naar /v1/embeddings van llama-swap:8080."""\n')
    assert not _bevindingen_tekst(CLIENT_BESTAND, 'LLAMA_SWAP_URL = "http://llama-swap:8080/v1"\n')
    # Wat mag, blijft mogen.
    assert not _bevindingen_imports(BRON, "from shared.model_routing import _load_config\n")
    # Een import van ``openai`` telt overal, ook voor alleen een uitzonderingsklasse.
    assert _bevindingen_imports(BRON, "from openai import RateLimitError\n")
