"""Modelregister: een YAML-bestand in Git (ADR 0007, FO 9).

Het register legt per model vast: naam, versie, aanroepwijze (``llama-swap``,
``api`` of ``cli``), locatie (``lokaal`` of ``cloud``), of het model interne
invoer mag krijgen (``toegestaan_intern``, standaard ``nee``) en de taken.
:class:`llm_manager.gate.ModelGate` leest het register; de Langflow-componenten
lezen hetzelfde bestand.

Het schema is streng: een onbekende sleutel, een dubbele naam of een cloudmodel
met ``toegestaan_intern: ja`` maakt het register ongeldig (harde eis 1).

Plaats: ``SIGNAAL_MODELREGISTER``, anders ``config/modelregister.yaml`` in de
root van de repository.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Aanroepwijze = Literal["llama-swap", "api", "cli"]
Locatie = Literal["lokaal", "cloud"]

_ROOT = Path(__file__).resolve().parents[4]
STANDAARD_PAD = _ROOT / "config" / "modelregister.yaml"


class RegisterFout(ValueError):
    """Het register ontbreekt of voldoet niet aan het schema."""


class RegisterModel(BaseModel):
    """Eén model of verwerkingsdienst in het register."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    naam: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.:/@-]+$")
    versie: str = Field(min_length=1)
    aanroepwijze: Aanroepwijze
    locatie: Locatie
    #: ``ja`` of ``nee``; standaard ``nee`` (ADR 0007).
    toegestaan_intern: Literal["ja", "nee"] = "nee"
    taken: List[str] = Field(min_length=1)

    @field_validator("toegestaan_intern", mode="before")
    @classmethod
    def _geen_yaml_bool(cls, waarde: object) -> object:
        # YAML 1.1 maakt van `yes`/`no` een bool; alleen `ja` of `nee` telt.
        if isinstance(waarde, bool):
            raise ValueError("toegestaan_intern is 'ja' of 'nee'")
        return waarde

    @model_validator(mode="after")
    def _cloud_nooit_intern(self) -> "RegisterModel":
        if self.aanroepwijze in ("api", "cli") and self.locatie != "cloud":
            raise ValueError(f"{self.naam}: aanroepwijze {self.aanroepwijze} is altijd cloud")
        if self.locatie == "cloud" and self.toegestaan_intern == "ja":
            raise ValueError(f"{self.naam}: een cloudmodel is nooit toegestaan voor intern")
        return self

    @property
    def intern_toegestaan(self) -> bool:
        return self.toegestaan_intern == "ja"


class ModelRegister(BaseModel):
    """Het hele register: een versie en de modellen."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    versie: int
    modellen: List[RegisterModel]

    @model_validator(mode="after")
    def _unieke_namen(self) -> "ModelRegister":
        namen = [m.naam for m in self.modellen]
        dubbel = sorted({n for n in namen if namen.count(n) > 1})
        if dubbel:
            raise ValueError(f"dubbele modelnamen: {', '.join(dubbel)}")
        return self

    def zoek(self, naam: str) -> Optional[RegisterModel]:
        return self._index().get(naam)

    def _index(self) -> Dict[str, RegisterModel]:
        return {m.naam: m for m in self.modellen}


def lees_register(pad: Optional[Path | str] = None) -> ModelRegister:
    """Lees en valideer het register; een fout geeft :class:`RegisterFout`."""
    pad = Path(pad or os.environ.get("SIGNAAL_MODELREGISTER") or STANDAARD_PAD)
    try:
        inhoud = yaml.safe_load(pad.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        raise RegisterFout(f"Modelregister niet te lezen: {pad}") from e
    try:
        return ModelRegister.model_validate(inhoud)
    except ValueError as e:
        raise RegisterFout(f"Modelregister ongeldig: {pad}: {e}") from e


@lru_cache(maxsize=1)
def get_register() -> ModelRegister:
    """Het register van deze omgeving; wijzigen vraagt een herstart (ADR 0007)."""
    return lees_register()
