"""Register van de flows in Git (F2-04, ADR 0022 E, ADR 0017 B).

De flows uit ``services/langflow/flows/`` zijn alleen-lezen gekoppeld in de
container ``open-notebook``, met de Git-gegevens van de repository. Een run legt
de Git-commit en de SHA-256 van de flow-JSON vast (``pipeline_version``). De
flow is *vastgelegd* als:

* de opgegeven commit gelijk is aan HEAD van de repository;
* een opgegeven ``SIGNAAL_FLOWS_COMMIT`` ook gelijk is aan HEAD;
* de map met flows schoon is (``git status --porcelain`` op die map is leeg);
* de flow in HEAD (``git show HEAD:<pad>``, niet de werkmap) precies die hash
  heeft, het bestand in de werkmap gelijk is aan die blob en geen symbolische
  link is, en de naam niet bij twee flows hoort.

In alle andere gevallen, ook zonder Git, is niets vastgelegd; wat dan volgt,
bepaalt ``ProcessingRunService``. Een flow heet naar haar bestandsnaam zonder
``.json``, of naar het veld ``id`` in de JSON. De hashes worden per HEAD
bewaard en opnieuw gelezen zodra HEAD verandert.

Instelling: ``SIGNAAL_FLOWS_DIR`` (standaard ``services/langflow/flows`` in de
root van de repository) en ``SIGNAAL_FLOWS_COMMIT`` (optioneel).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

#: ``apps/app-main/src/app_main/services/flow_register.py`` → root van de repository.
_ROOT = Path(__file__).resolve().parents[5]
STANDAARD_MAP = _ROOT / "services" / "langflow" / "flows"


class FlowRegister:
    """De hashes van de flows in HEAD, met controle op HEAD en een schone map."""

    def __init__(self, map_: Path, verwachte_commit: Optional[str] = None) -> None:
        self.map = Path(map_)
        self.verwachte_commit = (verwachte_commit or "").strip().lower() or None
        self._head: Optional[str] = None
        self._hashes: Dict[str, str] = {}
        self._bestanden: Dict[str, str] = {}

    @classmethod
    def from_env(cls) -> "FlowRegister":
        map_ = Path(os.environ.get("SIGNAAL_FLOWS_DIR") or STANDAARD_MAP)
        return cls(map_, os.environ.get("SIGNAAL_FLOWS_COMMIT"))

    @staticmethod
    @lru_cache(maxsize=1)
    def standaard() -> "FlowRegister":
        """Eén register per proces, zodat de hashes bewaard blijven."""
        return FlowRegister.from_env()

    def _git(self, *args: str, binair: bool = False):
        if not self.map.is_dir():
            return None
        try:
            uit = subprocess.run(
                ["git", "-C", str(self.map), *args],
                capture_output=True, timeout=10, check=True,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return uit.stdout if binair else uit.stdout.decode("utf-8", "replace")

    def head(self) -> Optional[str]:
        uit = self._git("rev-parse", "HEAD")
        return uit.strip().lower() if uit else None

    def schoon(self) -> bool:
        """Geen wijziging in de map, ook niet verborgen voor ``git status``.

        ``git status`` ziet een bestand met ``skip-worktree`` of
        ``assume-unchanged`` niet. Daarom vergelijkt het register ook elk
        bestand in de werkmap met zijn blob in HEAD, en weigert het een
        symbolische link in de werkmap.
        """
        uit = self._git("status", "--porcelain", "--untracked-files=all", "--", ".")
        if uit is None or uit.strip() != "":
            return False
        self.hashes()
        for naam, h in self._bestanden.items():
            pad = self.map / naam
            if pad.is_symlink() or not pad.is_file():
                return False
            if hashlib.sha256(pad.read_bytes()).hexdigest() != h:
                return False
        return True

    def _laad(self, head: str) -> None:
        """Lees de flows in HEAD: per bestand de hash, per naam de hash.

        Een symbolische link (modus 120000) telt niet als flow. Een naam die
        bij twee verschillende flows hoort (een bestandsnaam en een ``id`` in
        een andere flow), telt ook niet: die botsing wordt gelogd.
        """
        self._bestanden = {}
        self._hashes = {}
        botsingen: set[str] = set()
        regels = self._git("ls-tree", "-r", head, "--", ".") or ""
        for regel in regels.splitlines():
            kop, _, naam = regel.partition("\t")
            modus = kop.split(" ")[0]
            if not naam.endswith(".json"):
                continue
            if modus == "120000":
                logger.warning(f"Flow {naam} is een symbolische link; niet vastgelegd")
                continue
            inhoud = self._git("show", f"{head}:./{naam}", binair=True)
            if inhoud is None:
                continue
            h = hashlib.sha256(inhoud).hexdigest()
            self._bestanden[naam] = h
            try:
                flow_id = json.loads(inhoud).get("id")
            except (ValueError, AttributeError):
                flow_id = None
            namen = {Path(naam).stem}
            if isinstance(flow_id, str) and flow_id:
                namen.add(flow_id)
            for n in namen:
                if n in self._hashes and self._hashes[n] != h:
                    botsingen.add(n)
                self._hashes[n] = h
        for n in botsingen:
            logger.warning(f"Flownaam {n!r} hoort bij meer dan één flow; niet vastgelegd")
            del self._hashes[n]

    def hashes(self) -> Dict[str, str]:
        """De hashes in HEAD; leeg zonder Git."""
        head = self.head()
        if head is None:
            return {}
        if head != self._head:
            self._laad(head)
            self._head = head
        return self._hashes

    def is_vastgelegd(
        self, pipeline_id: str, git_commit: Optional[str], flow_hash: Optional[str]
    ) -> bool:
        if not (git_commit and flow_hash):
            return False
        head = self.head()
        if head is None:
            logger.warning(f"Geen Git in {self.map}: geen enkele flow is vastgelegd")
            return False
        if self.verwachte_commit and self.verwachte_commit != head:
            return False
        if git_commit.strip().lower() != head or not self.schoon():
            return False
        return self.hashes().get(pipeline_id) == flow_hash.strip().lower()


@lru_cache(maxsize=1)
def fork_commit() -> Optional[str]:
    """De Git-commit van de fork.

    In de container de commit waaruit het image is gebouwd (build-argument
    ``SIGNAAL_FORK_COMMIT``, F2-10); daarbuiten HEAD van de repository. De
    commit van de flows (``SIGNAAL_FLOWS_COMMIT``) telt hier niet.
    """
    gezet = (os.environ.get("SIGNAAL_FORK_COMMIT") or "").strip().lower()
    if gezet and gezet != "onbekend":
        return gezet
    return FlowRegister(_ROOT).head()
