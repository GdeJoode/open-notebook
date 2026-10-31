"""Bronversies (F2-04, ADR 0003).

Een versie is uniek op bron en inhoudshash. :meth:`SourceVersionRepository.register`
geeft bij een bekende hash de bestaande versie terug en maakt anders een nieuwe,
met de vorige als ``predecessor``; de bron wijst daarna naar de nieuwe versie.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from shared.models.source_version import SourceVersion

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    ensure_record_id,
    execute_query,
    execute_transaction,
)
from surrealdb_service.repositories.processing_run import ProcessingRunRepository

#: Een nieuwe versie ``$versie`` (vooraf gekozen id) van ``$bron`` met de velden
#: ``$versie_data``, binnen een transactie (F2-14, migratie 96). Het nummer komt
#: uit de teller ``source.version_seq``: de teller en ``current_version`` gaan
#: alleen mee als de teller nog de gelezen waarde heeft, anders ``THROW`` en
#: rolt alles terug. Zo krijgen gelijktijdige versies unieke nummers en een
#: lineaire keten (``predecessor`` is de versie met het vorige nummer).
NIEUWE_VERSIE = """
LET $oud = (SELECT VALUE version_seq FROM ONLY $bron) ?? 0;
LET $vorige = (SELECT VALUE id FROM source_version WHERE source = $bron AND version_no = $oud LIMIT 1)[0];
LET $data = object::from_entries(array::concat(object::entries($versie_data), [
    ['source', $bron], ['version_no', $oud + 1], ['predecessor', $vorige]
]));
CREATE $versie CONTENT $data RETURN NONE;
LET $gezet = (UPDATE $bron SET version_seq = $oud + 1, current_version = $versie
    WHERE (version_seq ?? 0) = $oud RETURN id);
IF array::len($gezet) = 0 { THROW 'bronversie:gelijktijdig' };
"""

#: Melding van :data:`NIEUWE_VERSIE` bij een gelijktijdige nieuwe versie.
GELIJKTIJDIG = "bronversie:gelijktijdig"

# Nieuwe versie en `source.current_version` in één transactie.
_REGISTREER = "BEGIN TRANSACTION;\n" + NIEUWE_VERSIE + "COMMIT TRANSACTION;\n"


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
                {"bron": ensure_record_id(source_id), "versie_data": velden,
                 "versie": ensure_record_id(f"source_version:{uuid.uuid4().hex}")},
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

    async def set_current_run(self, version_id: str, run_id: str) -> bool:
        """Maak ``run_id`` de huidige run van de versie; ``False`` als dat niet mag.

        Een handmatige run is beschermd (ADR 0003, migratie 92): een run met een
        andere trigger verzet ``current_run`` niet zolang die naar een
        handmatige run wijst. De run krijgt dan een regel in zijn logboek en de
        beheerder een melding in ``beheer_melding``; alleen de beheerder kiest
        welke run de huidige wordt.
        """
        versie = ensure_record_id(version_id)
        run = ensure_record_id(run_id)
        rows = await execute_query(
            "UPDATE $id SET current_run = $run WHERE current_run = $run"
            " OR current_run.trigger != 'handmatig' OR $run.trigger = 'handmatig'"
            " RETURN id",
            {"id": versie, "run": run},
            self.config,
        )
        if rows:
            return True
        huidig = await execute_query(
            "SELECT source, current_run FROM $id", {"id": versie}, self.config
        )
        if not huidig:
            raise ValueError(f"Bronversie niet gevonden: {version_id}")
        melding = "De huidige run is handmatig; current_run is niet verzet"
        await ProcessingRunRepository(self.config).append_log(run_id, [{
            "at": datetime.now(timezone.utc), "step": "huidige_run",
            "status": "beschermd", "message": melding,
        }])
        await execute_query(
            "CREATE beheer_melding CONTENT $d RETURN NONE",
            {"d": {
                "soort": "handmatige_run_beschermd",
                "melding": melding,
                "source": ensure_record_id(str(huidig[0]["source"])),
                "source_version": versie,
                "run": run,
                "current_run": ensure_record_id(str(huidig[0]["current_run"])),
            }},
            self.config,
        )
        logger.warning(
            "Run {run} verzet de handmatige run van {versie} niet; melding voor de beheerder",
            run=run_id, versie=version_id,
        )
        return False
