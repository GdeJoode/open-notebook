"""Bronkaartregels (F2-13, ADR 0020, migratie 93).

Lezen en schrijven van ``source_map_rule``. Een wijziging gaat altijd via
:func:`wijzig_met_rev` (optimistische vergrendeling, ADR 0003); een regel in
``classification_log`` gaat mee in dezelfde transactie. Rechten en de
locatiecontrole doet de dienst (``SourceMapService``) in app-main.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shared.models.source_map import SourceMapRule
from surrealdb import RecordID

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query, execute_transaction
from surrealdb_service.repositories.base import (
    RecordNotFoundError,
    RevWijziging,
    eigen_record_id,
    wijzig_met_rev,
)

TABEL = "source_map_rule"


def _zonder_none(data: Dict[str, Any]) -> Dict[str, Any]:
    # Weglaten geeft NONE en dus de standaard van het schema.
    return {k: v for k, v in data.items() if v is not None}


async def maak_met_log(
    tabel: str,
    data: Dict[str, Any],
    log: Optional[Dict[str, Any]],
    config: Optional[SurrealDBConfig],
) -> Dict[str, Any]:
    """Maak een record van ``tabel``; met ``log`` ook een regel in
    ``classification_log`` met dat record als onderwerp, in één transactie."""
    rid = RecordID(tabel, uuid.uuid4().hex)
    sql = ["BEGIN TRANSACTION;", "CREATE $id CONTENT $d RETURN NONE;"]
    params: Dict[str, Any] = {"id": rid, "d": _zonder_none(data)}
    if log:
        params["log"] = {**log, tabel: rid}
        sql.append("CREATE classification_log CONTENT $log RETURN NONE;")
    sql.append("COMMIT TRANSACTION;")
    await execute_transaction("\n".join(sql), params, config)
    rows = await execute_query("SELECT * FROM $id", {"id": rid}, config)
    return rows[0]


class SourceMapRuleRepository:
    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    async def create(
        self, data: Dict[str, Any], *, log: Optional[Dict[str, Any]] = None
    ) -> SourceMapRule:
        """Maak een regel; ``log`` komt in dezelfde transactie in ``classification_log``."""
        return SourceMapRule(**await maak_met_log(TABEL, data, log, self.config))

    async def get(self, rule_id: str) -> Optional[SourceMapRule]:
        rid = eigen_record_id(rule_id, TABEL)
        if rid is None:
            return None
        rows = await execute_query("SELECT * FROM $id", {"id": rid}, self.config)
        return SourceMapRule(**rows[0]) if rows else None

    async def list(self, *, active: Optional[bool] = None) -> List[SourceMapRule]:
        if active is None:
            rows = await execute_query(
                "SELECT * FROM source_map_rule ORDER BY name", {}, self.config
            )
        else:
            rows = await execute_query(
                "SELECT * FROM source_map_rule WHERE active = $a ORDER BY name",
                {"a": active},
                self.config,
            )
        return [SourceMapRule(**r) for r in rows or []]

    async def update(
        self,
        rule_id: str,
        *,
        rev: int,
        data: Dict[str, Any],
        door: str,
        log: Optional[Dict[str, Any]] = None,
    ) -> SourceMapRule:
        """Wijzig met ``rev``; een verouderde ``rev`` geeft ``RevConflictError``.

        Een waarde ``None`` in ``data`` wist het veld (NONE).
        """
        rid = eigen_record_id(rule_id, TABEL)
        if rid is None:
            raise RecordNotFoundError([str(rule_id)])
        aanmaken: Sequence[Tuple[str, Dict[str, Any]]] = (
            [("classification_log", {**log, "source_map_rule": rid})] if log else []
        )
        [rij] = await wijzig_met_rev(
            [RevWijziging(rid, rev, data)], door=door, aanmaken=aanmaken, config=self.config
        )
        return SourceMapRule(**rij)
