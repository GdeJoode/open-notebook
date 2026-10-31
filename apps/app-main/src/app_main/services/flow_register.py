"""Register van de flows in Git (F2-04, ADR 0022 E, ADR 0017 B).

De flows uit ``services/langflow/flows/`` zijn alleen-lezen gekoppeld in de
container ``open-notebook``. Een run legt de Git-commit en de SHA-256 van de
flow-JSON vast (``pipeline_version``). De flow is *vastgelegd* als de commit
gelijk is aan die van de gekoppelde flows en het bestand van de flow precies
die hash heeft. Anders niet; wat dan volgt, bepaalt ``ProcessingRunService``.

Instelling:

* ``SIGNAAL_FLOWS_DIR``: de map met flows (standaard ``services/langflow/flows``);
* ``SIGNAAL_FLOWS_COMMIT``: de Git-commit van die map. Zonder deze waarde
  probeert het register ``git rev-parse HEAD`` in de map; lukt dat niet, dan is
  geen enkele flow vastgelegd.

Een flow heet naar haar bestandsnaam zonder ``.json``, of naar het veld ``id``
in de JSON.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

STANDAARD_MAP = "services/langflow/flows"


def _git_commit(map_: Path) -> Optional[str]:
    try:
        uit = subprocess.run(
            ["git", "-C", str(map_), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return uit.stdout.strip() or None


@dataclass
class FlowRegister:
    """De hashes van de flows in de gekoppelde map, met de commit van die map."""

    git_commit: Optional[str]
    hashes: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dir(cls, map_: Path, git_commit: Optional[str]) -> "FlowRegister":
        hashes: Dict[str, str] = {}
        if map_.is_dir():
            for pad in sorted(map_.rglob("*.json")):
                inhoud = pad.read_bytes()
                h = hashlib.sha256(inhoud).hexdigest()
                hashes[pad.stem] = h
                try:
                    flow_id = json.loads(inhoud).get("id")
                except (ValueError, AttributeError):
                    flow_id = None
                if isinstance(flow_id, str) and flow_id:
                    hashes[flow_id] = h
        else:
            logger.warning(f"Flowmap {map_} ontbreekt: geen enkele flow is vastgelegd")
        return cls(git_commit=git_commit, hashes=hashes)

    @classmethod
    def from_env(cls) -> "FlowRegister":
        map_ = Path(os.environ.get("SIGNAAL_FLOWS_DIR", STANDAARD_MAP))
        commit = os.environ.get("SIGNAAL_FLOWS_COMMIT") or _git_commit(map_)
        return cls.from_dir(map_, commit)

    def is_vastgelegd(
        self, pipeline_id: str, git_commit: Optional[str], flow_hash: Optional[str]
    ) -> bool:
        if not (self.git_commit and git_commit and flow_hash):
            return False
        if git_commit.strip().lower() != self.git_commit.strip().lower():
            return False
        return self.hashes.get(pipeline_id) == flow_hash.strip().lower()
