"""Register van de flows in Git (F2-04, ADR 0022 E, ADR 0017 B).

De flows uit ``services/langflow/flows/`` zijn alleen-lezen gekoppeld in de
container ``open-notebook``, met de Git-gegevens van de repository. Een run legt
de Git-commit en de SHA-256 van de flow-JSON vast (``pipeline_version``). De
flow is *vastgelegd* als:

* de opgegeven commit gelijk is aan HEAD van de repository;
* een opgegeven ``SIGNAAL_FLOWS_COMMIT`` ook gelijk is aan HEAD;
* de map met flows schoon is (``git status --porcelain`` op die map is leeg);
* de flow in HEAD (``git show HEAD:<pad>``, niet de werkmap) precies die hash
  heeft.

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
        uit = self._git("status", "--porcelain", "--untracked-files=all", "--", ".")
        return uit is not None and uit.strip() == ""

    def _laad(self, head: str) -> Dict[str, str]:
        hashes: Dict[str, str] = {}
        namen = self._git("ls-tree", "-r", "--name-only", head, "--", ".") or ""
        for naam in namen.splitlines():
            if not naam.endswith(".json"):
                continue
            inhoud = self._git("show", f"{head}:./{naam}", binair=True)
            if inhoud is None:
                continue
            h = hashlib.sha256(inhoud).hexdigest()
            hashes[Path(naam).stem] = h
            try:
                flow_id = json.loads(inhoud).get("id")
            except (ValueError, AttributeError):
                flow_id = None
            if isinstance(flow_id, str) and flow_id:
                hashes[flow_id] = h
        return hashes

    def hashes(self) -> Dict[str, str]:
        """De hashes in HEAD; leeg zonder Git."""
        head = self.head()
        if head is None:
            return {}
        if head != self._head:
            self._hashes = self._laad(head)
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
