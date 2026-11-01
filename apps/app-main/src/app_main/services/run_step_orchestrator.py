"""Stappen van een run starten, uitvoeren en volgen (F2-15, ADR 0011, plan §5.4).

Langflow start elke stap na de import via ``POST /api/v1/runs/{run_id}/steps/{stap}``
en volgt hem met ``GET``. De fork zet per stap één job (met ``job.run``) in de
wachtrij van ``packages/job-queue``; de worker voert hem uit en doet de
herhaalpogingen. Na het maximum gaan stap en run op ``mislukt``, met een vaste
melding zonder inhoud in de stap, de run en ``beheer_melding``.

Regels:

* Recht ``pijplijn`` (router), en alleen de sleutel die de run aanmaakte (F2-14).
* Alleen in een lopende run, en alleen op een bronversie waarvan deze run de
  inhoud schreef. Een stap verwerkt alleen de fragmenten van die versie en run,
  met hun eigen classificatie; elke modelaanroep gaat via ``ModelGate`` met
  ``run_id`` (ADR 0003, 0007).
* Dezelfde stap op dezelfde versie in dezelfde run geeft dezelfde stap.
* Elke aanroep sluit eerst runs die te lang ``bezig`` staan (F2-14); een run met
  een stap in ``wacht_op_review`` valt daarbuiten (B5).
* ``processing_stage`` van de bron blijft een statusveld; de stap zet het.

``entity_dedup`` draait tot F3-07 binnen ``entity_extraction`` (de
``FilteringWorkflow``); de stap legt dat vast in het runlogboek.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, Optional

from job_queue import JobNotRetryableError, JobPausedForReviewError
from loguru import logger
from shared.models.jobs import JobSubmitRequest
from shared.models.processing_run import RunStep
from shared.types.enums import JobType
from surrealdb_service.repositories.run_step import RunStepRepository

from app_main.api.fouten import log_fout
from app_main.exceptions import NotFoundError
from app_main.services.processing_run_service import (
    NietDeAanmakerError,
    ProcessingRunService,
    RunClosedError,
)
from app_main.services.source_pipeline import set_failed_stage, set_processing_stage

#: Nieuwe pogingen na de eerste; daarna is de stap ``mislukt``.
MAX_HERHALINGEN = 2

#: De ``processing_stage`` na een geslaagde stap (statusveld, ADR 0011).
STADIUM = {"embed": ["embedded"], "entity_extraction": ["extracted"],
           "graph": ["graphed", "complete"]}

Uitvoerder = Callable[[RunStep], Awaitable[str]]


class StapNietGebondenError(Exception):
    """De bronversie hoort niet bij de run (HTTP 409 ``niet_gebonden``)."""


class StapFout(Exception):
    """Een mislukte poging, zonder inhoud: alleen het soort fout.

    De volledige fout staat alleen in het lokale log, onder het fout-id.
    """

    def __init__(self, stap: str, soort: str, fout_id: str) -> None:
        super().__init__(f"Stap {stap} mislukt ({soort}) fout-id {fout_id}")
        self.soort = soort


def _soort(fout: BaseException) -> str:
    """Het soort fout als korte code (patroon van `run_step.error`, migratie 104)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", type(fout).__name__)[:80]


def _nu() -> datetime:
    return datetime.now(timezone.utc)


def _regel(stap: str, status: str, melding: str = "") -> Dict[str, Any]:
    return {"at": _nu(), "step": stap, "status": status, "message": melding}


class RunStepOrchestrator:
    def __init__(
        self,
        run_service: Optional[ProcessingRunService] = None,
        steps: Optional[RunStepRepository] = None,
        jobs: Any = None,
        uitvoerders: Optional[Dict[str, Uitvoerder]] = None,
    ) -> None:
        self.runs = run_service or ProcessingRunService()
        self.steps = steps or RunStepRepository(self.runs.run_repo.config)
        self._jobs = jobs
        self._uitvoerders: Dict[str, Uitvoerder] = {
            "embed": self._embed,
            "entity_extraction": self._entity_extraction,
            "entity_dedup": self._entity_dedup,
            "graph": self._graph,
            **(uitvoerders or {}),
        }

    @property
    def jobs(self):
        if self._jobs is None:
            from app_main.services.command_service import get_service

            self._jobs = get_service()
        return self._jobs

    # ------------------------------------------------------------------
    # Eindpunten
    # ------------------------------------------------------------------

    async def _run_van_aanmaker(self, run_id: str, key_id: Optional[str]):
        await self.runs.sluit_verlopen_runs()
        run = await self.runs.get(run_id)
        if not run.created_by_key or run.created_by_key != key_id:
            raise NietDeAanmakerError("Alleen de aanmaker van de run")
        return run

    async def start(
        self, run_id: str, stap: str, source_version_id: str, *, key_id: Optional[str]
    ) -> RunStep:
        """Start een stap, of geef de bestaande stap met dezelfde versie."""
        run = await self._run_van_aanmaker(run_id, key_id)
        if run.afgesloten:
            raise RunClosedError("Run is al afgesloten")
        bron = await self.steps.bron_van_versie(str(run.id), source_version_id)
        if bron is None:
            raise StapNietGebondenError("Bronversie hoort niet bij deze run")
        versie = source_version_id if ":" in source_version_id \
            else f"source_version:{source_version_id}"

        stap_rec = await self.steps.vind(str(run.id), stap, versie)
        if stap_rec is None:
            try:
                stap_rec = await self.steps.create(
                    run_id=str(run.id), stap=stap, source_version_id=versie, source_id=bron)
            except RuntimeError:
                # Gelijktijdig dezelfde stap (unieke index): neem die.
                stap_rec = await self.steps.vind(str(run.id), stap, versie)
                if stap_rec is None:
                    raise
        if stap_rec.job is None:
            job = await self.jobs.submit(JobSubmitRequest(
                job_type=JobType.RUN_STEP,
                payload={"step_id": str(stap_rec.id), "run_id": str(run.id), "stap": stap},
                max_retries=MAX_HERHALINGEN,
                source_id=bron,
                run=str(run.id),
            ))
            if await self.steps.zet_job(str(stap_rec.id), str(job.id)):
                stap_rec.job = str(job.id)
            else:
                # Een gelijktijdige aanroep zette al een job: deze vervalt.
                await self.jobs.cancel(str(job.id))
                stap_rec = await self.steps.get(str(stap_rec.id))
        return stap_rec

    async def status(
        self, run_id: str, stap: str, *, key_id: Optional[str],
        source_version_id: Optional[str] = None,
    ) -> RunStep:
        """De stap van deze run; zonder versie de laatst gestarte met die naam."""
        run = await self._run_van_aanmaker(run_id, key_id)
        stap_rec = await self.steps.vind(str(run.id), stap, source_version_id)
        if stap_rec is None:
            raise NotFoundError("Stap niet gevonden")
        return stap_rec

    # ------------------------------------------------------------------
    # Uitvoeren (handler van de job) en definitief mislukken
    # ------------------------------------------------------------------

    async def voer_uit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Eén poging van een stap; de worker doet de herhaalpogingen."""
        stap_rec = await self.steps.get(str(payload.get("step_id") or ""))
        if stap_rec is None:
            raise JobNotRetryableError("Stap onbekend")
        run = await self.runs.run_repo.get(stap_rec.run)
        if run is None or run.afgesloten:
            raise JobNotRetryableError("Run is al afgesloten")
        stap_rec = await self.steps.start_poging(str(stap_rec.id))
        if stap_rec is None:
            return {"step_id": payload.get("step_id"), "status": "klaar"}

        stap = stap_rec.stap
        try:
            melding = await self._uitvoerders[stap](stap_rec)
        except JobPausedForReviewError:
            await self.steps.zet_status(str(stap_rec.id), "wacht_op_review")
            await self.runs.run_repo.append_log(
                stap_rec.run, [_regel(stap, "wacht_op_review", "Wacht op de beheerder")])
            await set_processing_stage(stap_rec.source, "awaiting_schema_review")
            raise
        except JobNotRetryableError:
            raise
        except Exception as e:  # noqa: BLE001 — de worker probeert opnieuw
            soort = _soort(e)
            fout_id = log_fout(e, f"Stap {stap} van {stap_rec.run} mislukt")
            await self.steps.zet_status(
                str(stap_rec.id), "wachtrij", error=f"Poging {stap_rec.attempts} mislukt ({soort})")
            raise StapFout(stap, soort, fout_id) from None

        await self.steps.zet_status(str(stap_rec.id), "geslaagd")
        await self.runs.run_repo.append_log(stap_rec.run, [_regel(stap, "geslaagd", melding)])
        for waarde in STADIUM.get(stap, []):
            await set_processing_stage(stap_rec.source, waarde)
        return {"step_id": str(stap_rec.id), "status": "geslaagd"}

    async def bij_definitief_mislukt(self, job: Any, fout: BaseException) -> None:
        """Na de laatste poging: stap en run ``mislukt``, met een melding zonder inhoud."""
        stap_rec = await self.steps.get(str((job.payload or {}).get("step_id") or ""))
        if stap_rec is None:
            return
        soort = getattr(fout, "soort", None) or _soort(fout)
        melding = f"Stap {stap_rec.stap} mislukt na {stap_rec.attempts} pogingen ({soort})"
        await self.steps.zet_status(str(stap_rec.id), "mislukt", error=melding)
        regel = [_regel(stap_rec.stap, "mislukt", melding)]
        if await self.runs.run_repo.close(stap_rec.run, status="mislukt", error=melding,
                                          log=regel) is None:
            await self.runs.run_repo.append_log(stap_rec.run, regel)
        await self.steps.melding_mislukt(stap_rec, melding)
        await set_failed_stage(stap_rec.source, stap_rec.stap)
        logger.warning(melding)

    # ------------------------------------------------------------------
    # De stappen
    # ------------------------------------------------------------------

    async def _embed(self, stap_rec: RunStep) -> str:
        from app_main.dependencies import get_embedding_service

        fragmenten = await self.steps.fragmenten(stap_rec.source_version, stap_rec.run)
        gedaan = await self.steps.fragmenten_met_embedding(stap_rec.run)
        service = await get_embedding_service()
        uitkomst = await service.embed_fragmenten(
            stap_rec.source, fragmenten, run_id=stap_rec.run, overslaan=gedaan)
        if uitkomst.afgekapt:
            await self.runs.run_repo.append_log(stap_rec.run, [_regel(
                "embed", "afgekapt",
                f"{uitkomst.afgekapt} fragmenten afgekapt voor de embedding (ADR 0008)")])
        return f"{uitkomst.embeddings_created} embeddings, {uitkomst.afgekapt} afgekapt"

    async def _entity_extraction(self, stap_rec: RunStep) -> str:
        from app_main.dependencies import get_entity_extraction_service, get_source_repo

        fragmenten = await self.steps.fragmenten(stap_rec.source_version, stap_rec.run)
        notebook_id = None
        try:
            notebook_id = await get_source_repo().get_notebook_id(stap_rec.source)
        except Exception as e:  # noqa: BLE001 — zonder notebook: enkel schema
            logger.warning(f"Notebook van {stap_rec.source} niet gevonden: {type(e).__name__}")
        uitkomst = await get_entity_extraction_service().run_extraction(
            source_id=stap_rec.source, notebook_id=notebook_id,
            run_id=stap_rec.run, fragmenten=fragmenten)
        return (f"{uitkomst.get('entity_count', 0)} entiteiten, "
                f"{uitkomst.get('relation_count', 0)} relaties")

    async def _entity_dedup(self, stap_rec: RunStep) -> str:
        return "Deduplicatie draait binnen entity_extraction tot F3-07"

    async def _graph(self, stap_rec: RunStep) -> str:
        from app_main.dependencies import get_mentions_projection_service

        uitkomst = await get_mentions_projection_service().refresh_source(stap_rec.source)
        return f"{uitkomst.created} vermeldingsrelaties"
