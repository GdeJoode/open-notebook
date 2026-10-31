"""Verwerkingsruns (F2-04, ADR 0003).

Een run wordt aangemaakt met status ``bezig`` en één keer afgesloten. Het
schema (migratie 90) laat een afgesloten run niet terug naar een andere status;
:meth:`ProcessingRunRepository.close` sluit alleen een run die nog ``bezig`` is.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from shared.models.processing_run import ProcessingRun
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query


def _nu() -> datetime:
    return datetime.now(timezone.utc)


def _rij(rows: Any) -> Optional[Dict[str, Any]]:
    if isinstance(rows, list):
        return rows[0] if rows else None
    return rows if isinstance(rows, dict) else None


class ProcessingRunRepository:
    """Lezen en schrijven van ``processing_run``."""

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    async def create(
        self,
        *,
        trigger: str,
        pipeline_id: str,
        pipeline_version: Optional[Dict[str, Any]] = None,
        input_classification: str = "onbekend",
        source_map_rule_id: Optional[str] = None,
        source_map_rule_rev: Optional[int] = None,
        label_schema: Optional[Dict[str, Any]] = None,
        flow_vastgelegd: bool = False,
        status: str = "bezig",
        error: Optional[str] = None,
        log: Optional[List[Dict[str, Any]]] = None,
        actor: Optional[str] = None,
        created_by_key: Optional[str] = None,
        intake_batch: Optional[str] = None,
        intake_item: Optional[str] = None,
    ) -> ProcessingRun:
        data: Dict[str, Any] = {
            "trigger": trigger,
            "actor": actor,
            "pipeline_id": pipeline_id,
            "pipeline_version": pipeline_version,
            "input_classification": input_classification,
            "source_map_rule_id": source_map_rule_id,
            "source_map_rule_rev": source_map_rule_rev,
            "label_schema": label_schema,
            "flow_vastgelegd": flow_vastgelegd,
            "status": status,
            "error": error,
            "log": log or [],
            # Binding van de run (F2-14, migratie 94).
            "created_by_key": created_by_key,
            "intake_batch": ensure_record_id(intake_batch) if intake_batch else None,
            "intake_item": ensure_record_id(intake_item) if intake_item else None,
        }
        if status != "bezig":
            data["ended_at"] = _nu()
        # Een expliciete None wordt in de database NULL; weglaten geeft NONE.
        data = {k: v for k, v in data.items() if v is not None}
        rows = await execute_query(
            "CREATE ONLY processing_run CONTENT $data", {"data": data}, self.config
        )
        return ProcessingRun(**_rij(rows))

    async def get(self, run_id: str) -> Optional[ProcessingRun]:
        rid = _run_id(run_id)
        if rid is None:
            return None
        rows = await execute_query("SELECT * FROM ONLY $id", {"id": rid}, self.config)
        rij = _rij(rows)
        return ProcessingRun(**rij) if rij else None

    async def append_log(self, run_id: str, entries: List[Dict[str, Any]]) -> None:
        """Vul het logboek aan; ook na het afsluiten (bijvoorbeeld een waarschuwing)."""
        if not entries:
            return
        await execute_query(
            "UPDATE $id SET log = array::concat(log, $regels) RETURN NONE",
            {"id": _run_id_of_fout(run_id), "regels": entries},
            self.config,
        )

    async def set_models(self, run_id: str, models: List[Dict[str, Any]]) -> None:
        await execute_query(
            "UPDATE $id SET models = $models RETURN NONE",
            {"id": _run_id_of_fout(run_id), "models": models},
            self.config,
        )

    async def close(
        self,
        run_id: str,
        *,
        status: str,
        ended_at: Optional[datetime] = None,
        error: Optional[str] = None,
        log: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[ProcessingRun]:
        """Sluit een run die nog ``bezig`` is; ``None`` als die al was afgesloten."""
        params: Dict[str, Any] = {
            "id": _run_id_of_fout(run_id),
            "status": status,
            "einde": ended_at or _nu(),
            "regels": log or [],
        }
        fout = ""
        if error is not None:
            params["fout"] = error
            fout = ", error = $fout"
        rows = await execute_query(
            "UPDATE $id SET status = $status, ended_at = $einde, "
            f"log = array::concat(log, $regels){fout} "
            "WHERE status = 'bezig' RETURN AFTER",
            params,
            self.config,
        )
        rij = _rij(rows)
        return ProcessingRun(**rij) if rij else None

    async def sluit_verlopen(self, grens: datetime, melding: str) -> int:
        """Sluit elke run die sinds vóór ``grens`` ``bezig`` staat als ``mislukt``.

        Een run die door een afgebroken proces blijft lopen, kan anders blijven
        schrijven (ADR 0003, restrisico F2-04). Geeft het aantal gesloten runs.
        """
        rows = await execute_query(
            "UPDATE processing_run SET status = 'mislukt', ended_at = time::now(),"
            " error = $melding, log = array::concat(log, [$regel])"
            " WHERE status = 'bezig' AND started_at < $grens RETURN id",
            {"grens": grens, "melding": melding,
             "regel": {"at": _nu(), "step": "bewaking", "status": "mislukt",
                       "message": melding}},
            self.config,
        )
        return len(rows or [])

    async def has_new_content(self, run_id: str) -> bool:
        """Maakte de run een bronversie, fragment of ``doc_node``?"""
        rows = await execute_query(
            "RETURN array::len((SELECT VALUE id FROM source_version WHERE created_by_run = $id LIMIT 1))"
            " + array::len((SELECT VALUE id FROM chunk WHERE run = $id LIMIT 1))"
            " + array::len((SELECT VALUE id FROM doc_node WHERE run = $id LIMIT 1))",
            {"id": _run_id_of_fout(run_id)},
            self.config,
        )
        waarde = rows[0] if isinstance(rows, list) and rows else rows
        return bool(waarde)

    async def models_from_model_calls(self, run_id: str) -> List[Dict[str, Any]]:
        """De modellen uit ``model_call`` van deze run (ADR 0003, 0007).

        De tabel ``model_call`` komt met ADR 0007; tot dan is de lijst leeg.
        """
        rows = await execute_query(
            "SELECT model, model_version, invocation FROM model_call WHERE run = $id "
            "GROUP BY model, model_version, invocation",
            {"id": _run_id_of_fout(run_id)},
            self.config,
        )
        return [r for r in rows or [] if isinstance(r, dict)]


def _run_id_of_fout(run_id: str):
    """Als :func:`_run_id`, maar een ongeldig id is een ``ValueError``."""
    rid = _run_id(run_id)
    if rid is None:
        raise ValueError(f"Geen id van een verwerkingsrun: {run_id!r}")
    return rid


def _run_id(run_id: str):
    """Een geldig id van ``processing_run``, anders ``None`` (de router geeft 404)."""
    try:
        rid = ensure_record_id(run_id if ":" in run_id else f"processing_run:{run_id}")
    except Exception:
        return None
    return rid if rid.table_name == "processing_run" else None


async def source_map_rule_snapshot(
    rule_id: str, config: Optional[SurrealDBConfig] = None
) -> Optional[Dict[str, Any]]:
    """Classificatie, bevestiging, status, route en ``rev`` van een bronkaartregel.

    De tabel ``source_map_rule`` staat in migratie 93 (F2-13, ADR 0020). Geen
    regel gevonden: ``None``, en de invoer van een run is onbekend.
    """
    try:
        rid = ensure_record_id(rule_id if ":" in rule_id else f"source_map_rule:{rule_id}")
    except Exception:
        return None
    if rid.table_name != "source_map_rule":
        return None
    rows = await execute_query(
        "SELECT classification, classification_confirmed_by, active, route_type, rev FROM $id",
        {"id": rid},
        config,
    )
    return _rij(rows)
