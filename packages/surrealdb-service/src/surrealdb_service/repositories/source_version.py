"""Bronversies (F2-04, ADR 0003).

Een versie is uniek op bron en inhoudshash. :meth:`SourceVersionRepository.register`
geeft bij een bekende hash de bestaande versie terug en maakt anders een nieuwe,
met de vorige als ``predecessor``; de bron wijst daarna naar de nieuwe versie.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from shared.models.source_version import SourceVersion
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    ensure_record_id,
    execute_query,
    execute_transaction,
)

# Nieuwe versie en `source.current_version` in één transactie. Het nummer is
# het aantal versies plus één; versies worden niet verwijderd. Een gelijktijdige
# tweede registratie faalt op de unieke index en leest dan de bestaande.
_REGISTREER = """
BEGIN TRANSACTION;
LET $data = object::from_entries(array::concat(object::entries($velden), [
    ['version_no', array::len((SELECT VALUE id FROM source_version WHERE source = $bron)) + 1],
    ['predecessor', (SELECT VALUE current_version FROM ONLY $bron)]
]));
LET $versie = (CREATE ONLY source_version CONTENT $data RETURN id);
UPDATE $bron SET current_version = $versie.id RETURN NONE;
COMMIT TRANSACTION;
"""


class SourceVersionRepository:
    """Lezen en schrijven van ``source_version``."""

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    async def find(self, source_id: str, content_hash: str) -> Optional[SourceVersion]:
        rows = await execute_query(
            "SELECT * FROM source_version WHERE source = $bron AND content_hash = $hash LIMIT 1",
            {"bron": ensure_record_id(source_id), "hash": content_hash},
            self.config,
        )
        return SourceVersion(**rows[0]) if rows else None

    async def get(self, version_id: str) -> Optional[SourceVersion]:
        rows = await execute_query(
            "SELECT * FROM $id", {"id": ensure_record_id(version_id)}, self.config
        )
        return SourceVersion(**rows[0]) if rows else None

    async def list_by_source(self, source_id: str) -> List[SourceVersion]:
        rows = await execute_query(
            "SELECT * FROM source_version WHERE source = $bron ORDER BY version_no",
            {"bron": ensure_record_id(source_id)},
            self.config,
        )
        return [SourceVersion(**r) for r in rows or []]

    async def register(
        self,
        source_id: str,
        content_hash: str,
        run_id: str,
        *,
        original: Optional[str] = None,
        classification: Optional[str] = None,
    ) -> Tuple[SourceVersion, bool]:
        """De versie voor deze inhoud, en of die nieuw is.

        Bij een bekende hash verandert er niets (``ongewijzigd``). Een nieuwe
        versie krijgt de classificatie van de bron, of strenger (``intern``);
        een interne versie maakt de bron intern (migratie 90).
        """
        bestaand = await self.find(source_id, content_hash)
        if bestaand is not None:
            return bestaand, False
        velden: Dict[str, Any] = {
            "source": ensure_record_id(source_id),
            "content_hash": content_hash,
            "created_by_run": ensure_record_id(run_id),
            "current_run": ensure_record_id(run_id),
        }
        if original is not None:
            velden["original"] = original
        if classification is not None:
            velden["classification"] = classification
        try:
            await execute_transaction(
                _REGISTREER,
                {"bron": ensure_record_id(source_id), "velden": velden},
                self.config,
            )
        except RuntimeError as e:
            bestaand = await self.find(source_id, content_hash)
            if bestaand is not None and "idx_source_version_hash" in str(e):
                return bestaand, False
            raise
        nieuw = await self.find(source_id, content_hash)
        if nieuw is None:  # pragma: no cover - de transactie slaagde
            raise RuntimeError("bronversie niet teruggevonden")
        return nieuw, True

    async def set_current_run(self, version_id: str, run_id: str) -> None:
        await execute_query(
            "UPDATE $id SET current_run = $run RETURN NONE",
            {"id": ensure_record_id(version_id), "run": ensure_record_id(run_id)},
            self.config,
        )
