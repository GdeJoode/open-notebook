"""De enige weg naar een model in de fork: ``ModelGate`` en dan llama-swap (F2-08).

:class:`PoortAanroeper` biedt chat en embeddings. Elke aanroep:

- kiest het model uit het register: het opgegeven model of het eerste model met
  de taak (:meth:`llm_manager.register.ModelRegister.voor_taak`);
- gaat door :meth:`llm_manager.gate.ModelGate.roep_aan`, met de classificatie
  van de invoer (verplicht; alles behalve ``extern`` telt als intern) en de run
  als die er is. De poort toetst, probeert bij 502 opnieuw en schrijft precies
  één regel in ``model_call``;
- gaat naar ``http://llama-swap:8080/v1`` en nergens anders
  (:mod:`llm_manager.providers.factory`).

De fork stelt bij het starten de auditschrijver in (:func:`configureer`).
Zonder auditschrijver geen aanroep: :func:`get_aanroeper` weigert dan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm_manager.gate import AuditSchrijver, ModelGate
from llm_manager.providers.factory import LlamaSwapClient
from llm_manager.register import ModelRegister, RegisterModel, get_register

#: Taak voor embeddings in het register.
EMBEDDING = "embedding"


class GeenPoort(RuntimeError):
    """De poort is niet ingesteld; er is geen weg naar een model."""


@dataclass(frozen=True)
class ChatAntwoord:
    """Het antwoord van een chataanroep: tekst en het model dat antwoordde."""

    tekst: str
    model: str


def _tekst(data: Dict[str, Any]) -> str:
    try:
        return data["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError, AttributeError):
        return ""


class PoortAanroeper:
    """Chat en embeddings via :class:`ModelGate` naar llama-swap."""

    def __init__(self, gate: ModelGate, client: Optional[LlamaSwapClient] = None) -> None:
        self.gate = gate
        self.client = client or LlamaSwapClient()

    def model_voor(self, taak: str) -> str:
        """De modelnaam voor een taak; zonder model weigert de poort de aanroep."""
        gevonden = self.gate.register.voor_taak(taak)
        return gevonden.naam if gevonden else f"geen-model-voor-{taak}"

    async def chat(
        self,
        *,
        taak: str,
        classificatie: Optional[str],
        messages: List[Dict[str, Any]],
        run_id: Optional[str] = None,
        model: Optional[str] = None,
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> ChatAntwoord:
        opties: Dict[str, Any] = {}
        if json_mode:
            opties["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            opties["max_tokens"] = max_tokens
        if temperature is not None:
            opties["temperature"] = temperature

        async def uitvoer(gekozen: RegisterModel) -> Dict[str, Any]:
            return await self.client.chat(gekozen.naam, messages, **opties)

        naam = model or self.model_voor(taak)
        data = await self.gate.roep_aan(model=naam, classificatie=classificatie, taak=taak,
                                        uitvoer=uitvoer, run_id=run_id)
        return ChatAntwoord(tekst=_tekst(data), model=naam)

    async def embed(
        self,
        *,
        teksten: List[str],
        classificatie: Optional[str],
        run_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        async def uitvoer(gekozen: RegisterModel) -> Dict[str, Any]:
            return await self.client.embed(gekozen.naam, teksten)

        data = await self.gate.roep_aan(model=model or self.model_voor(EMBEDDING),
                                        classificatie=classificatie, taak=EMBEDDING,
                                        uitvoer=uitvoer, run_id=run_id)
        rijen = sorted(data.get("data", []), key=lambda r: r.get("index", 0))
        vectoren = [[float(v) for v in r["embedding"]] for r in rijen]
        if len(vectoren) != len(teksten):
            raise ValueError("llama-swap gaf niet één vector per tekst")
        return vectoren


_aanroeper: Optional[PoortAanroeper] = None


def configureer(audit: AuditSchrijver, *, register: Optional[ModelRegister] = None,
                client: Optional[LlamaSwapClient] = None) -> PoortAanroeper:
    """Stel de poort van dit proces in, met de auditschrijver van de fork."""
    global _aanroeper
    _aanroeper = PoortAanroeper(ModelGate(register or get_register(), audit), client)
    return _aanroeper


def get_aanroeper() -> PoortAanroeper:
    """De poort van dit proces; weigert als er geen auditschrijver is ingesteld."""
    if _aanroeper is None:
        raise GeenPoort("Modelpoort niet ingesteld: geen auditschrijver, geen modelaanroep")
    return _aanroeper


def is_ingesteld() -> bool:
    return _aanroeper is not None


def reset() -> None:
    """Voor tests."""
    global _aanroeper
    _aanroeper = None
