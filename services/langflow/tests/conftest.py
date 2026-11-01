"""Hulp voor de tests van services/langflow (F2-12).

De scripts staan los in ``services/langflow/scripts`` (zo komen ze in het image);
de tests laden ze van daar. De live-tests draaien tegen de container ``langflow``
van project ``signaal`` als die draait en healthy is, anders worden ze
overgeslagen.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

LANGFLOW = Path(__file__).resolve().parents[1]
SCRIPTS = LANGFLOW / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ROOT = LANGFLOW.parents[1]

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


@pytest.fixture(scope="session")
def flows_laden():
    return importlib.import_module("flows_laden")


@pytest.fixture(scope="session")
def flows_exporteren():
    return importlib.import_module("flows_exporteren")


class Live:
    """``docker exec`` in de draaiende container langflow."""

    def __init__(self, container: str) -> None:
        self.container = container

    def exec(self, *args: str, invoer: str | None = None) -> subprocess.CompletedProcess:
        return subprocess.run(["docker", "exec", "-i", self.container, *args], input=invoer,
                              capture_output=True, text=True, timeout=300)

    def zet_map(self, map_: str, bestanden: dict[str, dict]) -> None:
        """Een lege map in /tmp van de container met deze flowbestanden."""
        uit = self.exec("rm", "-rf", map_)
        assert uit.returncode == 0, uit.stderr
        assert self.exec("mkdir", "-p", map_).returncode == 0
        for naam, flow in bestanden.items():
            uit = self.exec("sh", "-c", f"cat > {map_}/{naam}", invoer=json.dumps(flow))
            assert uit.returncode == 0, uit.stderr

    def python(self, code: str) -> subprocess.CompletedProcess:
        return self.exec("python", "-c", f"import sys; sys.path.insert(0, '/app/signaal')\n{code}")


@pytest.fixture(scope="session")
def live() -> Live:
    if shutil.which("docker") is None:
        pytest.skip("docker niet beschikbaar")
    uit = subprocess.run(["docker", "ps", "-q", "--filter", "label=com.docker.compose.project=signaal",
                          "--filter", "label=com.docker.compose.service=langflow"],
                         capture_output=True, text=True)
    container = uit.stdout.strip().splitlines()[0] if uit.stdout.strip() else ""
    if not container:
        pytest.skip("container langflow van project signaal draait niet")
    gezond = subprocess.run(["docker", "inspect", "-f", "{{.State.Health.Status}}", container],
                            capture_output=True, text=True).stdout.strip()
    if gezond != "healthy":
        pytest.skip(f"langflow is niet healthy ({gezond})")
    return Live(container)
