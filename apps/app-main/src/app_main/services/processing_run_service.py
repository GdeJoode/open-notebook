"""Verwerkingsruns aanmaken en afsluiten (F2-04, ADR 0003, 0022 E, plan §5.4).

``POST /api/v1/runs`` maakt een run met status ``bezig``. Eerst controleert de
fork de flow (ADR 0022 E, ADR 0017 B): staat de flow niet in Git (commit of
hash wijkt af van de gekoppelde flows) en is de invoer intern of onbekend, dan
wordt de run vastgelegd als ``mislukt`` en volgt HTTP 409. Bij externe invoer
blijft het een waarschuwing ``niet_vastgelegd`` in het runlogboek.

De invoer is alleen extern als de bronkaartregel van de run extern is, door een
beheerder bevestigd en actief, op een openbare route (ADR 0001, 0019, 0020). Zonder regel is de invoer
onbekend. De run legt id en ``rev`` van zijn regel vast (F2-13).

``PATCH /api/v1/runs/{id}`` sluit een run af. Een afgesloten run geeft 409. Een
run die ``geslaagd`` meldt maar geen bronversie, fragment of ``doc_node`` maakte,
eindigt als ``niets_nieuws``. ``models`` neemt de fork niet over van Langflow:
de lijst komt uit ``model_call`` van de run (ADR 0007); een verschil komt als
waarschuwing in het logboek.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from shared.models.processing_run import AFGESLOTEN, ProcessingRun
from shared.models.source_map import regel_classificatie
from surrealdb_service.repositories.processing_run import (
    ProcessingRunRepository,
    source_map_rule_snapshot,
)

from app_main.exceptions import InvalidInputError, NotFoundError
from app_main.services.flow_register import FlowRegister


class FlowNotRegisteredError(Exception):
    """De flow staat niet in Git en de invoer is intern of onbekend (HTTP 409)."""

    def __init__(self, run_id: str) -> None:
        super().__init__("Flow niet vastgelegd in Git; invoer intern of onbekend")
        self.run_id = run_id


class RunClosedError(Exception):
    """De run is al afgesloten (HTTP 409)."""


def _nu() -> datetime:
    return datetime.now(timezone.utc)


def _regel(step: str, status: str, message: str = "") -> Dict[str, Any]:
    return {"at": _nu(), "step": step, "status": status, "message": message}


def _model_sleutel(m: Dict[str, Any]) -> tuple:
    return (
        m.get("model") or m.get("name"),
        m.get("model_version") or m.get("version"),
        m.get("invocation") or m.get("aanroepwijze"),
    )


class ProcessingRunService:
    def __init__(
        self,
        run_repo: Optional[ProcessingRunRepository] = None,
        flow_register: Optional[FlowRegister] = None,
    ) -> None:
        self.run_repo = run_repo or ProcessingRunRepository()
        self._flow_register = flow_register

    @property
    def flow_register(self) -> FlowRegister:
        if self._flow_register is None:
            self._flow_register = FlowRegister.standaard()
        return self._flow_register

    async def _invoer(self, source_map_rule_id: Optional[str]) -> tuple[str, Optional[int]]:
        """De classificatie van de invoer en de ``rev`` van de regel."""
        if not source_map_rule_id:
            return "onbekend", None
        regel = await source_map_rule_snapshot(source_map_rule_id, self.run_repo.config)
        if not regel:
            return "onbekend", None
        rev = regel.get("rev") if isinstance(regel.get("rev"), int) else None
        return regel_classificatie(
            regel.get("classification"),
            regel.get("classification_confirmed_by"),
            active=regel.get("active"),
            route_type=regel.get("route_type"),
        ), rev

    async def start(
        self,
        *,
        trigger: str,
        pipeline_id: str,
        git_commit: Optional[str],
        flow_hash: Optional[str],
        source_map_rule_id: Optional[str] = None,
        label_schema: Optional[Dict[str, Any]] = None,
        actor: Optional[str] = None,
    ) -> ProcessingRun:
        """Maak een run; ``FlowNotRegisteredError`` als de flow niet mag draaien.

        ``actor`` is verplicht bij trigger ``handmatig`` (migratie 92).
        """
        invoer, rev = await self._invoer(source_map_rule_id)
        vastgelegd = self.flow_register.is_vastgelegd(pipeline_id, git_commit, flow_hash)
        velden: Dict[str, Any] = dict(
            trigger=trigger,
            pipeline_id=pipeline_id,
            pipeline_version={"git_commit": git_commit, "flow_hash": flow_hash},
            input_classification=invoer,
            source_map_rule_id=source_map_rule_id,
            source_map_rule_rev=rev,
            label_schema=label_schema,
            flow_vastgelegd=vastgelegd,
            actor=actor,
        )
        if vastgelegd:
            return await self.run_repo.create(**velden, log=[_regel("flow", "vastgelegd")])
        if invoer == "extern":
            return await self.run_repo.create(
                **velden,
                log=[_regel("flow", "niet_vastgelegd",
                            "Flow staat niet in Git; toegestaan bij externe invoer")],
            )
        melding = "Flow staat niet in Git en de invoer is intern of onbekend"
        run = await self.run_repo.create(
            **velden,
            status="mislukt",
            error=melding,
            log=[_regel("flow", "niet_vastgelegd", melding)],
        )
        raise FlowNotRegisteredError(str(run.id))

    async def get(self, run_id: str) -> ProcessingRun:
        run = await self.run_repo.get(run_id)
        if run is None:
            raise NotFoundError("Run niet gevonden")
        return run

    async def update(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        ended_at: Optional[datetime] = None,
        log: Optional[List[Dict[str, Any]]] = None,
        models: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
    ) -> ProcessingRun:
        """Werk een lopende run bij of sluit hem af."""
        run = await self.get(run_id)
        if run.afgesloten:
            raise RunClosedError("Run is al afgesloten")
        if status is not None and status not in AFGESLOTEN | {"bezig"}:
            raise InvalidInputError("Onbekende status")
        regels = [
            {"at": r.get("at") or _nu(), "step": r.get("step", ""),
             "status": r.get("status", ""), "message": r.get("message", "")}
            for r in (log or [])
        ]
        regels += await self._vergelijk_modellen(run_id, models)

        if status in AFGESLOTEN:
            if status == "geslaagd" and not await self.run_repo.has_new_content(run_id):
                status = "niets_nieuws"
                regels.append(_regel("afsluiten", "niets_nieuws", "Geen nieuwe inhoud"))
            gesloten = await self.run_repo.close(
                run_id, status=status, ended_at=ended_at, error=error, log=regels
            )
            if gesloten is None:
                raise RunClosedError("Run is al afgesloten")
            return gesloten

        await self.run_repo.append_log(run_id, regels)
        return await self.get(run_id)

    async def _vergelijk_modellen(
        self, run_id: str, gemeld: Optional[List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Leid ``models`` af uit ``model_call``; een verschil wordt een waarschuwing."""
        afgeleid = await self.run_repo.models_from_model_calls(run_id)
        await self.run_repo.set_models(run_id, afgeleid)
        if gemeld is None:
            return []
        verschil = {_model_sleutel(m) for m in gemeld} ^ {_model_sleutel(m) for m in afgeleid}
        if not verschil:
            return []
        return [_regel("models", "waarschuwing",
                       "Opgave van de flow wijkt af van model_call; niet overgenomen")]
