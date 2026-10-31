"""Telemetrie van modelaanroepen.

:class:`ModelCallTelemetry` schrijft het auditlog ``model_call`` (F2-07, ADR
0007): voor :class:`llm_manager.gate.ModelGate` in de fork en voor
``POST /api/v1/model-calls`` van Langflow. Na elke regel met een run leidt de
fork ``processing_run.models`` opnieuw af uit ``model_call`` (ADR 0003).

Track J (vervalt in F2-09): routing telemetry write (Track J.4, J-Q7).

After a routed LLM stage completes, :func:`record_routing_event` writes one
``routing.served`` metric row carrying the served provider/model plus the
failover trail (``was_failover`` + ``fallback_from``). It reuses the always-on,
never-blocking :func:`shared.services.metrics.record_metric` sink so a telemetry
hiccup never fails the LLM stage.

This is the single served-provider write per document stage that J.5's UI reads
to show the cloud-vs-local + fallback summary, and the provenance source the
extraction path stamps onto persisted entities.
"""

from __future__ import annotations

from typing import List, Optional

from llm_manager.gate import ModelCallRegel
from llm_manager.register import ModelRegister, get_register
from loguru import logger
from surrealdb_service.repositories import ModelCallRepository, ProcessingRunRepository

from app_main.services.model_routing.route_resolver import LLMTask

#: Pogingen om ``processing_run.models`` bij te werken bij een schrijfconflict.
_MODELS_POGINGEN = 3


class ModelNietInRegisterError(ValueError):
    """Een uitgevoerde aanroep met een model dat niet (zo) in het register staat."""


class NietDeAanmakerVanDeRunError(PermissionError):
    """Alleen de sleutel die de run aanmaakte, schrijft regels bij die run (F2-14)."""


class ModelCallTelemetry:
    """Schrijver van ``model_call``; nooit prompt- of antwoordtekst (ADR 0007)."""

    def __init__(
        self,
        repo: Optional[ModelCallRepository] = None,
        runs: Optional[ProcessingRunRepository] = None,
        register: Optional[ModelRegister] = None,
    ) -> None:
        self.repo = repo or ModelCallRepository()
        self.runs = runs or ProcessingRunRepository()
        self._register = register

    @property
    def register(self) -> ModelRegister:
        return self._register or get_register()

    async def schrijf(self, regel: ModelCallRegel) -> None:
        """De :class:`llm_manager.gate.AuditSchrijver` van de fork."""
        await self.repo.create(regel.model_dump())
        if regel.run:
            await self._ververs_models(regel.run)

    async def schrijf_van_langflow(self, regel: ModelCallRegel, *, key_id: Optional[str]) -> str:
        """Een regel van ``POST /api/v1/model-calls``; geeft het id.

        Een uitgevoerde aanroep (niet ``geweigerd``) moet een model uit het
        register noemen, met dezelfde aanroepwijze en locatie. Een regel bij een
        run mag alleen de sleutel die de run aanmaakte schrijven.
        """
        regel = regel.model_copy(update={"origin": "langflow"})
        if regel.status != "geweigerd":
            model = self.register.zoek(regel.model)
            if (model is None or model.aanroepwijze != regel.invocation
                    or model.locatie != regel.location):
                raise ModelNietInRegisterError("Model niet zo in het register")
            if model.locatie == "cloud" and regel.classification != "extern":
                # Harde eis 1 is dan al geschonden; de regel wordt wel vastgelegd.
                logger.error("Cloudaanroep met invoer {} gemeld door Langflow (model {})",
                             regel.classification, regel.model)
        if regel.run:
            run = await self.runs.get(regel.run)
            if run is None:
                raise LookupError("Run onbekend")
            if not run.created_by_key or run.created_by_key != key_id:
                raise NietDeAanmakerVanDeRunError("Alleen de aanmaker van de run")
            regel = regel.model_copy(update={"run": str(run.id)})
        rij = await self.repo.create(regel.model_dump())
        if regel.run:
            await self._ververs_models(regel.run)
        return str(rij["id"])

    async def _ververs_models(self, run_id: str) -> None:
        """Werk ``models`` bij; een fout laat de auditregel staan.

        Bij het afsluiten van de run leidt de fork ``models`` opnieuw af.
        """
        for poging in range(1, _MODELS_POGINGEN + 1):
            try:
                await self.runs.ververs_models(run_id)
                return
            except Exception as e:  # noqa: BLE001
                if poging == _MODELS_POGINGEN:
                    logger.warning("models van run {} niet bijgewerkt: {}",
                                   run_id, type(e).__name__)


async def record_routing_event(
    *,
    task: LLMTask,
    served_provider: str,
    served_model_id: Optional[str],
    was_failover: bool,
    fallback_from: List[str],
    source_id: Optional[str] = None,
    notebook_id: Optional[str] = None,
) -> None:
    """Record one ``routing.served`` telemetry event for a routed LLM stage.

    Best-effort: never raises (``record_metric`` swallows its own failures, and
    we guard the call too) so the LLM stage is never failed by telemetry.
    """
    try:
        from shared.services.metrics import record_metric

        await record_metric(
            "routing.served",
            {
                "task": task.value,
                "served_provider": served_provider,
                "served_model_id": served_model_id,
                "was_failover": was_failover,
                "fallback_from": fallback_from,
            },
            source=source_id,
            notebook=notebook_id,
        )
    except Exception as e:  # noqa: BLE001 — telemetry never fails the caller
        logger.warning(f"routing telemetry write failed (ignored): {e}")
