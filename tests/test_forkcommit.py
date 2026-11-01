"""``scripts/signaalsysteem/controle/forkcommit.sh`` (F2-10, herreview punt 2).

De commit van de fork in het image volgt de code: HEAD, of HEAD-dirty als de
werkmap niet schoon is; zonder ``--build`` waarschuwt start.sh als het label van
het image afwijkt. Toetst het script tegen een tijdelijke Git-repository.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "signaalsysteem" / "controle" / "forkcommit.sh"

pytestmark = pytest.mark.skipif(shutil.which("git") is None or shutil.which("bash") is None,
                                reason="git of bash ontbreekt")


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _script(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), str(repo), *args], capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path) -> Path:
    _git(tmp_path, "init", "-q")
    (tmp_path / "app.py").write_text("print('fork')\n", encoding="utf-8")
    (tmp_path / ".gitignore").write_text("genegeerd/\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "-c", "user.name=Toets", "-c", "user.email=toets@example.org",
         "commit", "-q", "-m", "begin")
    return tmp_path


def test_schone_werkmap_geeft_head(repo) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    (repo / "genegeerd").mkdir()
    (repo / "genegeerd" / "x.txt").write_text("x", encoding="utf-8")
    (repo / "docs").mkdir()
    (repo / "docs" / "notitie.md").write_text("x", encoding="utf-8")
    assert _script(repo).stdout.strip() == head


@pytest.mark.parametrize("wijziging", ["gewijzigd", "nieuw"])
def test_vuile_werkmap_geeft_dirty(repo, wijziging) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    if wijziging == "gewijzigd":
        (repo / "app.py").write_text("print('anders')\n", encoding="utf-8")
    else:
        (repo / "nieuw.py").write_text("x = 1\n", encoding="utf-8")
    assert _script(repo).stdout.strip() == f"{head}-dirty"


def test_label_dat_afwijkt_geeft_een_waarschuwing(repo) -> None:
    head = _git(repo, "rev-parse", "HEAD")
    assert _script(repo, "--label", head).returncode == 0
    for label in ("0" * 40, f"{head}-dirty", ""):
        uit = _script(repo, "--label", label)
        assert uit.returncode == 1 and "start.sh --build" in uit.stderr, label
