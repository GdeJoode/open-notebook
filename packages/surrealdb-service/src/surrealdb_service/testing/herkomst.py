"""Herkomst voor testfragmenten (F2-04, migratie 90).

Sinds migratie 90 zijn ``source_version`` en ``run`` verplicht op ``chunk`` en
``doc_node``. Tests die rechtstreeks fragmenten schrijven, halen die hier op:
de huidige versie van de bron met haar run, of anders een nieuwe run en
versie.
"""

from __future__ import annotations

import hashlib
from typing import Dict, Optional

from surrealdb import RecordID

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.processing_run import ProcessingRunRepository
from surrealdb_service.repositories.source_version import SourceVersionRepository


async def herkomst(bron: str, config: Optional[SurrealDBConfig] = None) -> Dict[str, RecordID]:
    """``{"source_version": ..., "run": ...}`` voor een fragment van ``bron``."""
    rows = await execute_query(
        "SELECT current_version, current_version.current_run AS run FROM $bron",
        {"bron": ensure_record_id(bron)},
        config,
    )
    rij = rows[0] if rows else {}
    if rij.get("current_version") and rij.get("run"):
        return {
            "source_version": ensure_record_id(str(rij["current_version"])),
            "run": ensure_record_id(str(rij["run"])),
        }
    run = await ProcessingRunRepository(config).create(trigger="handmatig", pipeline_id="test")
    versie, _ = await SourceVersionRepository(config).register(
        bron, hashlib.sha256(str(bron).encode()).hexdigest(), str(run.id)
    )
    return {
        "source_version": ensure_record_id(str(versie.id)),
        "run": ensure_record_id(str(versie.current_run or run.id)),
    }
