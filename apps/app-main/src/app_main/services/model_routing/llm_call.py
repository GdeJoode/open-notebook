"""De dunne aanroeper van ``ModelGate`` in de fork (F2-08, ADR 0006).

Elke modelaanroep uit app-main loopt hierlangs: via
:class:`llm_manager.aanroep.PoortAanroeper` en :class:`llm_manager.gate.ModelGate`
naar llama-swap. De poort schrijft per aanroep één regel in ``model_call`` met
:class:`~app_main.services.model_routing.telemetry.ModelCallTelemetry`.

- De aanroeper geeft altijd de classificatie van de invoer mee; ``None`` of
  ``onbekend`` telt als intern.
- Het model komt uit het register, per taak. Er is geen terugval naar een ander
  model, naar Ollama of naar een cloudaanbieder.
- ``LLMTask`` (Track J) blijft alleen de taaknaam; de routering van Track J
  (failover, providerketens) bereikt geen model meer en vervalt in F2-09.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm_manager import aanroep
from llm_manager.aanroep import ChatAntwoord, PoortAanroeper

#: Track J-taak (``LLMTask.name``) naar de taak in het register.
TAAK_VAN_LLMTASK = {
    "ENTITY_EXTRACTION": "entity_extraction",
    "SUMMARIZATION": "summarization",
    "CHAT": "chat",
}


def taak_van(task: Any) -> str:
    """De registertaak bij een ``LLMTask`` of een taaknaam."""
    naam = getattr(task, "name", task)
    return TAAK_VAN_LLMTASK.get(str(naam).upper(), str(naam).lower())


def registermodel(taak: str) -> Optional[str]:
    """De naam van het model voor een taak uit het register, of ``None``.

    Voor de routers: welk model er is, staat in het register, niet in
    ``default_models`` of in modelrijen (F2-08, ADR 0007).
    """
    from llm_manager.register import get_register

    model = get_register().voor_taak(taak)
    return model.naam if model else None


def get_aanroeper() -> PoortAanroeper:
    """De poort van dit proces, met ``model_call`` als auditlog."""
    if not aanroep.is_ingesteld():
        from app_main.services.model_routing.telemetry import ModelCallTelemetry

        aanroep.configureer(ModelCallTelemetry())
    return aanroep.get_aanroeper()


async def roep_taalmodel_aan(
    system: str,
    user: str,
    *,
    taak: str,
    classificatie: Optional[str],
    run_id: Optional[str] = None,
    json_mode: bool = False,
    max_tokens: Optional[int] = None,
) -> ChatAntwoord:
    """Eén chataanroep met een systeem- en een gebruikersbericht."""
    return await get_aanroeper().chat(
        taak=taak, classificatie=classificatie, run_id=run_id, json_mode=json_mode,
        max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )


@dataclass
class _Antwoord:
    """Wat de samenvattingsstrategieën en de voorbewerking lezen: ``.content``."""

    content: str


class PoortTaalmodel:
    """Een taalmodel met ``achat_complete(messages)`` dat via de poort loopt.

    Voor code die een object met ``achat_complete`` verwacht (samenvatten,
    voorbewerking). Classificatie, taak en run liggen vast bij het maken.
    """

    def __init__(self, *, taak: str, classificatie: Optional[str],
                 run_id: Optional[str] = None, json_mode: bool = False,
                 max_tokens: Optional[int] = None) -> None:
        self.taak = taak
        self.classificatie = classificatie
        self.run_id = run_id
        self.json_mode = json_mode
        self.max_tokens = max_tokens
        #: Het model dat het laatst antwoordde (herkomst).
        self.model: Optional[str] = None

    async def achat_complete(self, messages: List[Dict[str, Any]], **_opties: Any) -> _Antwoord:
        antwoord = await get_aanroeper().chat(
            taak=self.taak, classificatie=self.classificatie, run_id=self.run_id,
            messages=list(messages), json_mode=self.json_mode, max_tokens=self.max_tokens)
        self.model = antwoord.model
        return _Antwoord(content=antwoord.tekst)
