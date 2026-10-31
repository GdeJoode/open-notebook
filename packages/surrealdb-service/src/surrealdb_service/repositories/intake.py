"""Innamebatches en de innamewachtrij (F2-13, ADR 0019, 0020, migratie 93).

Een batch van een bulkupload heeft één classificatie en één route. Een
wachtrij-item (watchfolder, losse upload, bron zonder regel) wacht op het
besluit van de beheerder. Wijzigingen gaan via :func:`wijzig_met_rev`; een
regel in ``classification_log`` gaat mee in dezelfde transactie.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from shared.models.source_map import IntakeBatch, IntakeItem

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.base import (
    RecordNotFoundError,
    RevWijziging,
    eigen_record_id,
    wijzig_met_rev,
)
from surrealdb_service.repositories.source_map import maak_met_log


async def _wijzig(
    tabel: str,
    record_id: str,
    *,
    rev: int,
    data: Dict[str, Any],
    door: str,
    log: Optional[Dict[str, Any]],
    config: Optional[SurrealDBConfig],
) -> Dict[str, Any]:
    rid = eigen_record_id(record_id, tabel)
    if rid is None:
        raise RecordNotFoundError([str(record_id)])
    aanmaken = [("classification_log", {**log, tabel: rid})] if log else []
    [rij] = await wijzig_met_rev(
        [RevWijziging(rid, rev, data)], door=door, aanmaken=aanmaken, config=config
    )
    return rij


async def _get(tabel: str, record_id: str, config: Optional[SurrealDBConfig]):
    rid = eigen_record_id(record_id, tabel)
    if rid is None:
        return None
    rows = await execute_query("SELECT * FROM $id", {"id": rid}, config)
    return rows[0] if rows else None


class IntakeBatchRepository:
    TABEL = "intake_batch"

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    async def create(
        self, data: Dict[str, Any], *, log: Optional[Dict[str, Any]] = None
    ) -> IntakeBatch:
        return IntakeBatch(**await maak_met_log(self.TABEL, data, log, self.config))

    async def get(self, batch_id: str) -> Optional[IntakeBatch]:
        rij = await _get(self.TABEL, batch_id, self.config)
        return IntakeBatch(**rij) if rij else None

    async def list(self) -> List[IntakeBatch]:
        rows = await execute_query(
            "SELECT * FROM intake_batch ORDER BY created_at DESC", {}, self.config
        )
        return [IntakeBatch(**r) for r in rows or []]

    async def update(
        self, batch_id: str, *, rev: int, data: Dict[str, Any], door: str,
        log: Optional[Dict[str, Any]] = None,
    ) -> IntakeBatch:
        return IntakeBatch(**await _wijzig(
            self.TABEL, batch_id, rev=rev, data=data, door=door, log=log, config=self.config
        ))


class IntakeItemRepository:
    TABEL = "intake_item"

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    async def register(self, data: Dict[str, Any]) -> tuple[IntakeItem, bool]:
        """Meld een item aan; hetzelfde item (herkomst, pad, hash) nog eens
        geeft het bestaande terug. Geeft (item, nieuw)."""
        sleutel = {k: data[k] for k in ("origin", "path", "content_hash")}
        bestaand = await execute_query(
            "SELECT * FROM intake_item WHERE origin = $origin AND path = $path"
            " AND content_hash = $content_hash LIMIT 1",
            sleutel,
            self.config,
        )
        if bestaand:
            return IntakeItem(**bestaand[0]), False
        try:
            rij = await maak_met_log(self.TABEL, data, None, self.config)
        except RuntimeError as e:
            # Een gelijktijdige aanmelding: de unieke index weigert de tweede.
            if "idx_intake_item_uniek" not in str(e):
                raise
            rij = (await execute_query(
                "SELECT * FROM intake_item WHERE origin = $origin AND path = $path"
                " AND content_hash = $content_hash LIMIT 1",
                sleutel,
                self.config,
            ))[0]
            return IntakeItem(**rij), False
        return IntakeItem(**rij), True

    async def get(self, item_id: str) -> Optional[IntakeItem]:
        rij = await _get(self.TABEL, item_id, self.config)
        return IntakeItem(**rij) if rij else None

    async def list(self, *, status: Optional[str] = None) -> List[IntakeItem]:
        if status is None:
            rows = await execute_query(
                "SELECT * FROM intake_item ORDER BY created_at", {}, self.config
            )
        else:
            rows = await execute_query(
                "SELECT * FROM intake_item WHERE status = $s ORDER BY created_at",
                {"s": status},
                self.config,
            )
        return [IntakeItem(**r) for r in rows or []]

    async def update(
        self, item_id: str, *, rev: int, data: Dict[str, Any], door: str,
        log: Optional[Dict[str, Any]] = None,
    ) -> IntakeItem:
        return IntakeItem(**await _wijzig(
            self.TABEL, item_id, rev=rev, data=data, door=door, log=log, config=self.config
        ))
