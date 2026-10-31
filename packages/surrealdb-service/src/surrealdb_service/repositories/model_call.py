"""Auditlog van modelaanroepen, ``model_call`` (F2-07, ADR 0007, migratie 101).

Alleen aanvullen: de database weigert UPDATE en DELETE. Een regel bevat nooit
prompt- of antwoordtekst.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query


class ModelCallRepository:
    """Schrijven en lezen van ``model_call``."""

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    async def create(self, regel: Dict[str, Any]) -> Dict[str, Any]:
        data = {k: v for k, v in regel.items() if v is not None}
        if "run" in data:
            data["run"] = ensure_record_id(data["run"])
        rows = await execute_query(
            "CREATE ONLY model_call CONTENT $data", {"data": data}, self.config
        )
        return rows[0] if isinstance(rows, list) else rows

    async def meld_cloudaanroep_intern(self, model_call_id: str, run_id: Optional[str]) -> None:
        """Een ``beheer_melding``: Langflow meldt een cloudaanroep met interne of
        onbekende invoer (harde eis 1, migratie 102)."""
        data: Dict[str, Any] = {
            "soort": "cloudaanroep_intern",
            "melding": "Cloudaanroep met interne of onbekende invoer gemeld",
            "model_call": ensure_record_id(model_call_id),
        }
        if run_id:
            data["run"] = ensure_record_id(run_id)
        await execute_query("CREATE beheer_melding CONTENT $d RETURN NONE", {"d": data},
                            self.config)

    async def for_run(self, run_id: str) -> List[Dict[str, Any]]:
        rows = await execute_query(
            "SELECT * FROM model_call WHERE run = $run ORDER BY at",
            {"run": ensure_record_id(run_id)},
            self.config,
        )
        return [r for r in rows or [] if isinstance(r, dict)]
