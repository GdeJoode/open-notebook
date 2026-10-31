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


class OngeldigAntwoord(RuntimeError):
    """llama-swap gaf een antwoord met een onverwachte vorm.

    Wordt binnen de poort opgeworpen, zodat de auditregel ``mislukt`` wordt
    (foutcode ``ongeldigantwoord``) en nooit ``geslaagd`` met daarna een fout.
    """


@dataclass(frozen=True)
class ChatAntwoord:
    """Het antwoord van een chataanroep: tekst en het model dat antwoordde."""

    tekst: str
    model: str


def _chat_vorm(data: Any) -> Dict[str, Any]:
    """Controleer een chatantwoord: ``choices[0].message.content`` is tekst."""
    try:
        inhoud = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise OngeldigAntwoord("chatantwoord zonder choices[0].message.content") from None
    if not isinstance(inhoud, str):
        raise OngeldigAntwoord("chatantwoord zonder tekst")
    return {"tekst": inhoud, "usage": data.get("usage")}


def _embedding_vorm(data: Any, aantal: int) -> Dict[str, Any]:
    """Controleer een embeddingantwoord: precies één getallenvector per tekst."""
    try:
        rijen = sorted(data["data"], key=lambda r: r["index"])
        vectoren = [[float(v) for v in r["embedding"]] for r in rijen]
    except (KeyError, TypeError, ValueError):
        raise OngeldigAntwoord("embeddingantwoord met onverwachte vorm") from None
    if len(vectoren) != aantal or [r["index"] for r in rijen] != list(range(aantal)):
        raise OngeldigAntwoord("llama-swap gaf niet één vector per tekst")
    if not all(vectoren) or len({len(v) for v in vectoren}) != 1:
        raise OngeldigAntwoord("vectoren leeg of van ongelijke lengte")
    return {"vectoren": vectoren, "usage": data.get("usage")}


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
            # De vorm wordt binnen de poort gecontroleerd: een afwijking is een
            # mislukte aanroep in de auditregel.
            return _chat_vorm(await self.client.chat(gekozen.naam, messages, **opties))

        naam = model or self.model_voor(taak)
        data = await self.gate.roep_aan(model=naam, classificatie=classificatie, taak=taak,
                                        uitvoer=uitvoer, run_id=run_id)
        return ChatAntwoord(tekst=data["tekst"], model=naam)

    async def embed(
        self,
        *,
        teksten: List[str],
        classificatie: Optional[str],
        run_id: Optional[str] = None,
        model: Optional[str] = None,
    ) -> List[List[float]]:
        async def uitvoer(gekozen: RegisterModel) -> Dict[str, Any]:
            return _embedding_vorm(await self.client.embed(gekozen.naam, teksten), len(teksten))

        data = await self.gate.roep_aan(model=model or self.model_voor(EMBEDDING),
                                        classificatie=classificatie, taak=EMBEDDING,
                                        uitvoer=uitvoer, run_id=run_id)
        return data["vectoren"]


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
