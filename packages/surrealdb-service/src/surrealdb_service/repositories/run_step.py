"""Stappen van een run (F2-15, ADR 0011, migratie 104).

Een stap is uniek op run, stap en bronversie. Hij wordt alleen gemaakt in een
lopende run, en alleen voor een bronversie waarvan de run de inhoud schreef:
de versie zelf, of fragmenten van die versie.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from shared.models import Chunk
from shared.models.processing_run import RunStep
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query


def _rij(rows: Any) -> Optional[Dict[str, Any]]:
    if isinstance(rows, list):
        return rows[0] if rows else None
    return rows if isinstance(rows, dict) else None


def _id(waarde: str, tabel: str):
    """Een record-id van ``tabel``; ``None`` als het geen geldig id is."""
    try:
        rid = ensure_record_id(waarde if ":" in waarde else f"{tabel}:{waarde}")
    except Exception:
        return None
    return rid if rid.table_name == tabel else None


class RunStepRepository:
    """Lezen en schrijven van ``run_step``."""

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    async def bron_van_versie(self, run_id: str, source_version_id: str) -> Optional[str]:
        """De bron van de versie als de run haar inhoud schreef, anders ``None``.

        Een run schreef de inhoud als hij de versie maakte, of fragmenten van
        die versie. Een onbekende of ongeldige versie geeft ook ``None``.
        """
        run, versie = _id(run_id, "processing_run"), _id(source_version_id, "source_version")
        if run is None or versie is None:
            return None
        rij = _rij(await execute_query(
            "SELECT source FROM $versie WHERE created_by_run = $run"
            " OR array::len((SELECT VALUE id FROM chunk WHERE source_version = $versie"
            " AND run = $run LIMIT 1)) > 0",
            {"run": run, "versie": versie},
            self.config,
        ))
        return str(rij["source"]) if rij and rij.get("source") else None

    async def create(
        self, *, run_id: str, stap: str, source_version_id: str, source_id: str
    ) -> RunStep:
        rows = await execute_query(
            "CREATE ONLY run_step CONTENT $d",
            {"d": {
                "run": ensure_record_id(run_id),
                "stap": stap,
                "source_version": ensure_record_id(source_version_id),
                "source": ensure_record_id(source_id),
            }},
            self.config,
        )
        return RunStep(**_rij(rows))

    async def get(self, step_id: str) -> Optional[RunStep]:
        rid = _id(step_id, "run_step")
        if rid is None:
            return None
        rij = _rij(await execute_query("SELECT * FROM ONLY $id", {"id": rid}, self.config))
        return RunStep(**rij) if rij else None

    async def vind(self, run_id: str, stap: str, source_version_id: Optional[str] = None
                   ) -> Optional[RunStep]:
        """De stap van deze run; zonder versie de laatst gemaakte met die naam."""
        run = _id(run_id, "processing_run")
        if run is None:
            return None
        params: Dict[str, Any] = {"run": run, "stap": stap}
        filter_ = ""
        if source_version_id is not None:
            versie = _id(source_version_id, "source_version")
            if versie is None:
                return None
            params["versie"] = versie
            filter_ = " AND source_version = $versie"
        rows = await execute_query(
            f"SELECT * FROM run_step WHERE run = $run AND stap = $stap{filter_}"
            " ORDER BY created_at DESC LIMIT 1",
            params, self.config,
        )
        rij = _rij(rows)
        return RunStep(**rij) if rij else None

    async def zet_job(self, step_id: str, job_id: str) -> bool:
        """Koppel de job, alleen als de stap er nog geen heeft."""
        rows = await execute_query(
            "UPDATE $id SET job = $job WHERE job = NONE RETURN id",
            {"id": ensure_record_id(step_id), "job": ensure_record_id(job_id)},
            self.config,
        )
        return bool(rows)

    async def fragmenten_met_embedding(self, run_id: str) -> set[str]:
        """De fragmenten die in deze run al een embedding hebben."""
        rows = await execute_query(
            "SELECT VALUE chunk FROM source_embedding WHERE run = $run AND chunk != NONE",
            {"run": ensure_record_id(run_id)},
            self.config,
        )
        return {str(r) for r in rows or [] if r}

    async def start_poging(self, step_id: str) -> Optional[RunStep]:
        """Zet een stap die nog niet klaar is op ``bezig`` en tel de poging."""
        rows = await execute_query(
            "UPDATE $id SET status = 'bezig', attempts += 1, error = NONE"
            " WHERE status NOT IN ['geslaagd', 'mislukt'] RETURN AFTER",
            {"id": ensure_record_id(step_id)},
            self.config,
        )
        rij = _rij(rows)
        return RunStep(**rij) if rij else None

    async def zet_status(
        self, step_id: str, status: str, *, error: Optional[str] = None,
        review_item: Optional[str] = None,
    ) -> Optional[RunStep]:
        """Zet de status van een stap die nog niet klaar is; ``None`` als hij klaar was."""
        params: Dict[str, Any] = {"id": ensure_record_id(step_id), "status": status,
                                  "fout": error}
        extra = ""
        if review_item is not None:
            params["review"] = ensure_record_id(review_item)
            extra = ", review_item = $review"
        rows = await execute_query(
            f"UPDATE $id SET status = $status, error = $fout{extra}"
            " WHERE status NOT IN ['geslaagd', 'mislukt'] RETURN AFTER",
            params, self.config,
        )
        rij = _rij(rows)
        return RunStep(**rij) if rij else None

    async def fragmenten(self, source_version_id: str, run_id: str) -> List[Chunk]:
        """De fragmenten van deze versie die deze run schreef, op volgorde."""
        rows = await execute_query(
            "SELECT * FROM chunk WHERE source_version = $versie AND run = $run"
            " ORDER BY order ASC",
            {"versie": ensure_record_id(source_version_id), "run": ensure_record_id(run_id)},
            self.config,
        )
        return [Chunk(**r) for r in rows or []]

    async def melding_mislukt(self, stap: RunStep, melding: str) -> None:
        """Een ``beheer_melding`` ``stap_mislukt``: alleen verwijzingen en een vaste melding."""
        await execute_query("CREATE beheer_melding CONTENT $d RETURN NONE", {"d": {
            "soort": "stap_mislukt",
            "melding": melding,
            "run": ensure_record_id(stap.run),
            "source": ensure_record_id(stap.source),
            "source_version": ensure_record_id(stap.source_version),
            "run_step": ensure_record_id(str(stap.id)),
        }}, self.config)
