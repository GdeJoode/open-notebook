"""Stopconditie F2-12: het startscript van langflow weigert verboden variabelen.

ADR 0022 F: met een variabele die begint met een voorvoegsel van een tracer van
Langflow 1.13.0 start de container niet; één test per voorvoegsel. Daarnaast
ANTHROPIC_ en CLAUDE_ (geen Claude-sleutel in Langflow) en de vaste instellingen
(ADR 0017, 0022 D). Het script draait met ``--controle``: het stopt na de
controles, vóór het modelregister en het starten van Langflow.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

START = Path(__file__).resolve().parents[1] / "scripts" / "start.sh"

VOORVOEGSELS = ("LANGCHAIN_", "LANGSMITH_", "LANGFUSE_", "LANGWATCH_", "ARIZE_", "OPIK_",
                "PHOENIX_", "TRACELOOP_", "OPENLAYER_", "OTEL_EXPORTER_OTLP_", "LANGFLOW_FEATURE_")

VAST = {
    "LANGFLOW_AUTO_LOGIN": "false",
    "LANGFLOW_ALLOW_CUSTOM_COMPONENTS": "false",
    "LANGFLOW_SUBSTITUTE_OUTDATED_COMPONENT_CODE": "false",
    "LANGFLOW_BLOCK_CODE_INTERPRETER_COMPONENTS": "true",
    "LANGFLOW_AGENTIC_EXPERIENCE": "false",
    "LANGFLOW_ALLOW_COMPONENTS_PATHS_OVERRIDE": "true",
    "LANGFLOW_COMPONENTS_PATH": "/app/flows-git/services/langflow/components",
    "LANGFLOW_CREATE_STARTER_PROJECTS": "false",
    "LANGFLOW_STORE": "false",
    "LANGFLOW_CORS_ORIGINS": "http://127.0.0.1:8530",
    "LANGFLOW_DEACTIVATE_TRACING": "true",
    "DO_NOT_TRACK": "true",
    "LANGFLOW_MCP_SERVER_ENABLED": "false",
    "LANGFLOW_ADD_PROJECTS_TO_MCP_SERVERS": "false",
    "LANGFLOW_MCP_COMPOSER_ENABLED": "false",
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "LANGFLOW_MODELS_DEV_REFRESH": "false",
}


def _start(extra: dict[str, str] | None = None, zonder: str | None = None):
    sh = shutil.which("sh")
    if sh is None:
        pytest.skip("geen sh")
    env = {"PATH": "/usr/bin:/bin", **VAST, **(extra or {})}
    env.pop(zonder, None)
    return subprocess.run([sh, str(START), "--controle"], env=env, capture_output=True, text=True)


def test_zonder_verboden_variabele_slaagt_de_controle() -> None:
    uit = _start()
    assert uit.returncode == 0, uit.stderr
    assert "controle geslaagd" in uit.stdout


@pytest.mark.parametrize("voorvoegsel", VOORVOEGSELS)
def test_startscript_weigert_voorvoegsel(voorvoegsel: str) -> None:
    naam = f"{voorvoegsel}ENDPOINT"
    uit = _start({naam: "http://ergens"})
    assert uit.returncode == 1
    assert "weigert te starten" in uit.stderr and naam in uit.stderr


@pytest.mark.parametrize("naam", ["ANTHROPIC_API_KEY", "CLAUDE_CODE_OAUTH_TOKEN"])
def test_startscript_weigert_claude_sleutel(naam: str) -> None:
    uit = _start({naam: "x"})
    assert uit.returncode == 1 and naam in uit.stderr


def test_kleine_letters_tellen_ook() -> None:
    uit = _start({"langsmith_api_key": "x"})
    assert uit.returncode == 1 and "langsmith_api_key" in uit.stderr


def test_lege_waarde_telt_ook() -> None:
    uit = _start({"OTEL_EXPORTER_OTLP_ENDPOINT": ""})
    assert uit.returncode == 1


@pytest.mark.parametrize("naam", sorted(VAST))
def test_startscript_weigert_zonder_vaste_instelling(naam: str) -> None:
    uit = _start(zonder=naam)
    assert uit.returncode == 1 and naam in uit.stderr


@pytest.mark.parametrize("naam,waarde", [
    ("LANGFLOW_AUTO_LOGIN", "true"),
    ("LANGFLOW_ALLOW_CUSTOM_COMPONENTS", "true"),
    ("LANGFLOW_DEACTIVATE_TRACING", "false"),
    ("LANGFLOW_MCP_SERVER_ENABLED", "true"),
    ("DO_NOT_TRACK", "false"),
])
def test_startscript_weigert_afwijkende_instelling(naam: str, waarde: str) -> None:
    uit = _start({naam: waarde})
    assert uit.returncode == 1 and naam in uit.stderr


@pytest.mark.parametrize("naam", sorted(VAST))
def test_startscript_weigert_naam_in_andere_hoofdletters(naam: str) -> None:
    """Review F2-12, punt 1: pydantic leest ook kleine letters; een tweede variabele
    met dezelfde naam in andere hoofdletters wordt geweigerd, ook met de vaste waarde."""
    anders = naam.lower() if naam != naam.lower() else naam.upper()
    uit = _start({anders: VAST[naam]})
    assert uit.returncode == 1, uit.stderr
    assert anders in uit.stderr and "andere hoofdletters" in uit.stderr


def test_gemengde_hoofdletters_worden_ook_geweigerd() -> None:
    uit = _start({"Langflow_Auto_Login": "false"})
    assert uit.returncode == 1 and "Langflow_Auto_Login" in uit.stderr
