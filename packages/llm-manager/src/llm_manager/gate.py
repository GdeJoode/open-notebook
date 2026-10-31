"""De modelpoort: elke modelaanroep loopt hierdoorheen (ADR 0006, 0007, 0009).

Per aanroep doet :class:`ModelGate`:

1. het model opzoeken in het register (:mod:`llm_manager.register`); een onbekend
   model wordt geweigerd;
2. de invoer toetsen: is die intern of onbekend en staat het register het model
   niet toe voor intern, dan weigert de poort. Er is geen terugval naar een
   ander model of een ander adres: de aanroep krijgt alleen het gevraagde model;
3. vanuit de fork alleen aanroepwijze ``llama-swap`` toelaten (ADR 0006);
4. bij HTTP 502 van llama-swap opnieuw proberen, na een pauze en tot een
   maximum (ADR 0009: na een stop van buiten llama-swap);
5. precies één auditregel schrijven, ook bij een weigering of een fout. De
   regel bevat nooit prompt- of antwoordtekst, alleen kenmerken en een
   foutcode (ADR 0007).

De aanroep zelf is een functie van de aanroeper die het registermodel krijgt;
zo blijft de poort los van Esperanto of HTTP (F2-08).
"""

from __future__ import annotations

import asyncio
import re
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Literal, Optional, Protocol, TypeVar

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from llm_manager.register import ModelRegister, RegisterModel

T = TypeVar("T")

Classificatie = Literal["intern", "extern", "onbekend"]
Herkomst = Literal["fork", "langflow"]
CallStatus = Literal["geslaagd", "mislukt", "geweigerd"]

#: Foutcodes en taaknamen zijn korte codes, geen vrije tekst.
CODE_PATROON = r"^[a-z0-9_.:-]{1,64}$"
MODEL_PATROON = r"^[A-Za-z0-9_.:/@-]{1,128}$"


class ModelCallRegel(BaseModel):
    """Eén auditregel in ``model_call`` (migratie 101). Nooit prompt- of antwoordtekst.

    Ook het verzoek van ``POST /api/v1/model-calls``: een onbekend veld, zoals
    ``prompt``, wordt geweigerd.
    """

    model_config = ConfigDict(extra="forbid")

    at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    run: Optional[str] = None
    model: str = Field(pattern=MODEL_PATROON)
    model_version: Optional[str] = Field(default=None, max_length=256)
    invocation: Literal["llama-swap", "api", "cli"]
    location: Literal["lokaal", "cloud"]
    task: str = Field(pattern=CODE_PATROON)
    classification: Classificatie
    input_tokens: Optional[int] = Field(default=None, ge=0)
    output_tokens: Optional[int] = Field(default=None, ge=0)
    duration_ms: Optional[int] = Field(default=None, ge=0)
    status: CallStatus
    attempts: int = Field(default=1, ge=0)
    error_code: Optional[str] = Field(default=None, pattern=CODE_PATROON)
    origin: Herkomst


class AuditSchrijver(Protocol):
    """Schrijft een auditregel: in de fork naar ``model_call``, in Langflow via de API."""

    async def schrijf(self, regel: ModelCallRegel) -> None: ...


class ModelGeweigerd(Exception):
    """De poort weigert de aanroep; het model is niet aangeroepen."""

    def __init__(self, code: str, model: str) -> None:
        super().__init__(f"Model {model!r} geweigerd: {code}")
        self.code = code
        self.model = model


class AuditNietGeschreven(RuntimeError):
    """De aanroep slaagde, maar de auditregel kon niet worden geschreven.

    Het resultaat wordt dan niet teruggegeven: zonder auditregel is het model
    in de herkomst niet herleidbaar (harde eis 2).
    """


def _classificatie(waarde: Optional[str]) -> Classificatie:
    return waarde if waarde in ("intern", "extern") else "onbekend"


def _http_status(fout: BaseException) -> Optional[int]:
    status = getattr(fout, "status_code", None)
    if isinstance(status, int):
        return status
    status = getattr(getattr(fout, "response", None), "status_code", None)
    return status if isinstance(status, int) else None


def _foutcode(fout: BaseException) -> str:
    status = _http_status(fout)
    if status is not None:
        return f"http_{status}"
    return re.sub(r"[^a-z0-9_.:-]", "_", type(fout).__name__.lower())[:64] or "fout"


def _tokens(resultaat: Any) -> tuple[Optional[int], Optional[int]]:
    usage = getattr(resultaat, "usage", None)
    if usage is None and isinstance(resultaat, dict):
        usage = resultaat.get("usage")
    if usage is None:
        return None, None
    lees = usage.get if isinstance(usage, dict) else lambda k: getattr(usage, k, None)
    invoer, uitvoer = lees("prompt_tokens"), lees("completion_tokens")
    return (invoer if isinstance(invoer, int) else None,
            uitvoer if isinstance(uitvoer, int) else None)


class ModelGate:
    """Eén modelpoort voor de fork en de Langflow-componenten (ADR 0006)."""

    def __init__(
        self,
        register: ModelRegister,
        audit: AuditSchrijver,
        *,
        herkomst: Herkomst = "fork",
        max_pogingen: int = 6,
        pauze: float = 5.0,
        slaap: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        if max_pogingen < 1:
            raise ValueError("max_pogingen is ten minste 1")
        self.register = register
        self.audit = audit
        self.herkomst = herkomst
        self.max_pogingen = max_pogingen
        self.pauze = pauze
        self._slaap = slaap

    def toets(self, model: str, classificatie: Optional[str]) -> RegisterModel:
        """Zoek het model op en toets de invoer; zonder aanroep en zonder auditregel.

        Voor de validatie bij het verbinden in Langflow (FO 11). Weigert met
        :class:`ModelGeweigerd`.
        """
        gevonden = self.register.zoek(model)
        if gevonden is None:
            raise ModelGeweigerd("model_onbekend", model)
        if self.herkomst == "fork" and gevonden.aanroepwijze != "llama-swap":
            raise ModelGeweigerd("aanroepwijze_niet_in_fork", model)
        if _classificatie(classificatie) != "extern" and not gevonden.intern_toegestaan:
            raise ModelGeweigerd("niet_toegestaan_intern", model)
        return gevonden

    async def roep_aan(
        self,
        *,
        model: str,
        classificatie: Optional[str],
        taak: str,
        uitvoer: Callable[[RegisterModel], Awaitable[T]],
        run_id: Optional[str] = None,
    ) -> T:
        """Roep ``uitvoer`` aan met het registermodel, als de poort het toelaat.

        ``classificatie`` is die van de invoer; alles behalve ``extern`` telt als
        intern. ``run_id`` hoort bij de aanroep als die bij een run hoort.
        """
        if not re.fullmatch(CODE_PATROON, taak):
            raise ValueError("taak is een korte code")
        klasse = _classificatie(classificatie)
        gevonden = self.register.zoek(model)
        # Een onbekende naam kan vrije tekst zijn; die gaat niet in de auditregel.
        naam = model if re.fullmatch(MODEL_PATROON, model) else "ongeldige_naam"

        def regel(status: CallStatus, **velden: Any) -> ModelCallRegel:
            return ModelCallRegel(
                run=run_id, model=naam,
                model_version=gevonden.versie if gevonden else None,
                invocation=gevonden.aanroepwijze if gevonden else "llama-swap",
                location=gevonden.locatie if gevonden else "lokaal",
                task=taak, classification=klasse, status=status,
                origin=self.herkomst, **velden,
            )

        try:
            gevonden = self.toets(model, klasse)
        except ModelGeweigerd as weigering:
            await self._schrijf_zonder_te_falen(
                regel("geweigerd", attempts=0, error_code=weigering.code))
            raise

        begin = time.monotonic()
        poging = 0
        while True:
            poging += 1
            try:
                resultaat = await uitvoer(gevonden)
                break
            except Exception as fout:
                if _http_status(fout) == 502 and poging < self.max_pogingen:
                    logger.warning("Model {} gaf 502 (poging {}); opnieuw na {} s",
                                   model, poging, self.pauze)
                    await self._slaap(self.pauze)
                    continue
                await self._schrijf_zonder_te_falen(regel(
                    "mislukt", attempts=poging, error_code=_foutcode(fout),
                    duration_ms=_ms(begin)))
                raise

        invoer, uitvoer_tokens = _tokens(resultaat)
        try:
            await self.audit.schrijf(regel(
                "geslaagd", attempts=poging, duration_ms=_ms(begin),
                input_tokens=invoer, output_tokens=uitvoer_tokens))
        except Exception as e:
            raise AuditNietGeschreven("Auditregel niet geschreven") from e
        return resultaat

    async def _schrijf_zonder_te_falen(self, regel: ModelCallRegel) -> None:
        """Bij een weigering of fout gaat de oorspronkelijke uitzondering voor."""
        try:
            await self.audit.schrijf(regel)
        except Exception as e:  # noqa: BLE001
            logger.error("Auditregel voor {} niet geschreven ({}): {}",
                         regel.model, regel.status, type(e).__name__)


def _ms(begin: float) -> int:
    return int((time.monotonic() - begin) * 1000)
