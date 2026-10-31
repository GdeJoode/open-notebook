"""Bronkaartregels beheren (F2-13, ADR 0019, 0020, 0004; plan §5.2).

* Standaardclassificatie per innameroute (ADR 0019): ``api``, ``mcp``,
  ``feed`` en ``website`` stellen ``extern`` voor; de rest is ``intern``.
  ``extern`` geldt pas na een bevestiging door een beheerder
  (``classification_confirmed_by``, ``_at``); zonder bevestiging is de regel
  intern.
* Aanmaken, wijzigen en bevestigen alleen door een sleutel ``beheerder`` met
  persoon. Elke keer dat een beheerder een classificatie vastlegt (bevestigen
  of wijzigen), komt er één regel in ``classification_log`` met de persoon,
  in dezelfde transactie als de wijziging.
* Een wijziging van de classificatie, locatie, route of het brontype trekt een
  eerdere bevestiging in: de bevestiging gold de oude regel.
* Een wijziging gaat met ``rev`` (``wijzig_met_rev``); verouderd geeft 409.
* **Geen locatie in de vault of de exportmap** (ADR 0004, 0020): de dienst
  vergelijkt genormaliseerde paden, ook na het oplossen van symbolische
  koppelingen; een ``obsidian://``-adres is altijd verboden.
"""

from __future__ import annotations

import os
import posixpath
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import unquote, urlsplit

from shared.models.agents import Right
from shared.models.source_map import (
    SourceMapRule,
    effectieve_classificatie,
    pipeline_start_voor,
    standaard_classificatie,
    standaard_fetch_content,
)
from surrealdb_service.repositories.source_map import SourceMapRuleRepository

from app_main import config
from app_main.exceptions import InvalidInputError, NotFoundError


class VerbodenLocatieError(Exception):
    """De locatie ligt in de vault of de exportmap (HTTP 422, ADR 0004)."""


# ---------------------------------------------------------------------------
# Locatiecontrole
# ---------------------------------------------------------------------------

_URL = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")
_WINDOWS = re.compile(r"^([A-Za-z]):(?:/|$)(.*)$")
_WSL = re.compile(r"^/mnt/([A-Za-z])(?:/|$)(.*)$")


def _windows(letter: str, rest: str) -> str:
    # Een Windows-pad is hoofdletterongevoelig.
    return f"{letter.lower()}:" + posixpath.normpath("/" + rest.lstrip("/")).lower()


def canoniek_pad(waarde: str) -> Optional[str]:
    """``waarde`` als genormaliseerd pad, of ``None`` als het geen pad is (een URL).

    Een Windows-pad (``D:\\...``) en het WSL-koppelpunt (``/mnt/d/...``)
    krijgen dezelfde vorm (``d:/...``, kleine letters). Een POSIX-pad gaat door
    ``os.path.realpath``: ``..`` en symbolische koppelingen worden opgelost.
    """
    tekst = (waarde or "").strip()
    url = _URL.match(tekst)
    if url:
        if url.group(1).lower() != "file":
            return None
        tekst = unquote(urlsplit(tekst).path)
        if re.match(r"^/[A-Za-z]:", tekst):
            tekst = tekst[1:]
    tekst = tekst.replace("\\", "/")
    win = _WINDOWS.match(tekst)
    if win:
        return _windows(win.group(1), win.group(2))
    echt = os.path.realpath(tekst)
    wsl = _WSL.match(echt)
    if wsl:
        return _windows(wsl.group(1), wsl.group(2))
    return echt


def _ligt_in(kind: str, ouder: str) -> bool:
    ouder = ouder.rstrip("/") or "/"
    return kind == ouder or kind.startswith(ouder if ouder.endswith("/") else ouder + "/")


def ligt_in_verboden_map(locatie: str, verboden: Iterable[str]) -> bool:
    """Ligt ``locatie`` in (of op) een van de ``verboden`` paden?"""
    url = _URL.match((locatie or "").strip())
    if url and url.group(1).lower() == "obsidian":
        return True
    kind = canoniek_pad(locatie)
    if kind is None:
        return False
    for pad in verboden:
        ouder = canoniek_pad(pad)
        if ouder and _ligt_in(kind, ouder):
            return True
    return False


class Locatiecontrole:
    """De verboden paden komen uit de instellingen van de fork (``config``)."""

    def __init__(
        self,
        vault_paden: Optional[List[str]] = None,
        exportmap_paden: Optional[List[str]] = None,
    ) -> None:
        self._vault = vault_paden
        self._export = exportmap_paden

    @property
    def verboden(self) -> List[str]:
        vault = self._vault if self._vault is not None else config.get_vault_paden()
        export = self._export if self._export is not None else config.get_exportmap_paden()
        return [*vault, *export]

    def controleer(self, locatie: str) -> None:
        if ligt_in_verboden_map(locatie, self.verboden):
            raise VerbodenLocatieError("Locatie ligt in de vault of de exportmap")


# ---------------------------------------------------------------------------
# Dienst
# ---------------------------------------------------------------------------

#: Velden die een eerdere bevestiging intrekken als ze veranderen.
_TREKT_BEVESTIGING_IN = ("classification", "location", "route_type", "source_type")
#: Velden die niet leeg mogen worden.
_VERPLICHT = ("name", "route_type", "source_type", "location", "classification",
              "route", "owner", "fetch_content", "active")


def _nu() -> datetime:
    return datetime.now(timezone.utc)


def beheerder(permission: Right, person: Optional[str]) -> str:
    """De persoon van een sleutel ``beheerder``; anders ``PermissionError``."""
    person = (person or "").strip()
    if permission != Right.BEHEERDER or not person:
        raise PermissionError("Vraagt een sleutel beheerder met persoon")
    return person


def logregel(*, van: str, naar: str, reden: str, key_id: str, person: str) -> Dict[str, Any]:
    """Een regel voor ``classification_log`` (zonder onderwerp)."""
    return {"from_classification": van, "to_classification": naar,
            "reason": reden, "agent_key": key_id, "person": person}


class SourceMapService:
    def __init__(
        self,
        repo: Optional[SourceMapRuleRepository] = None,
        locatiecontrole: Optional[Locatiecontrole] = None,
    ) -> None:
        self.repo = repo or SourceMapRuleRepository()
        self.locatiecontrole = locatiecontrole or Locatiecontrole()

    async def list(self, *, active: Optional[bool] = None) -> List[SourceMapRule]:
        return await self.repo.list(active=active)

    async def get(self, rule_id: str) -> SourceMapRule:
        regel = await self.repo.get(rule_id)
        if regel is None:
            raise NotFoundError("Bronkaartregel niet gevonden")
        return regel

    async def create(
        self,
        data: Dict[str, Any],
        *,
        bevestig: bool,
        permission: Right,
        person: Optional[str],
        key_id: str,
        reason: Optional[str] = None,
    ) -> SourceMapRule:
        """Maak een regel; met ``bevestig`` legt de beheerder ``extern`` meteen vast."""
        person = beheerder(permission, person)
        data = {k: v for k, v in data.items() if v is not None}
        route_type = data.get("route_type")
        self.locatiecontrole.controleer(data.get("location", ""))
        data.setdefault("classification", standaard_classificatie(route_type))
        data.setdefault("fetch_content", standaard_fetch_content(data.get("source_type", "")))
        data["pipeline_start"] = pipeline_start_voor(route_type)
        data.update(modified_by=person, modified_at=_nu())
        log = None
        if bevestig:
            if data["classification"] != "extern":
                raise InvalidInputError("Alleen extern wordt bevestigd")
            data.update(classification_confirmed_by=person, classification_confirmed_at=_nu())
            log = logregel(van="intern", naar=effectieve_classificatie(
                "extern", person, active=data.get("active", True)),
                reden=reason or "Bronkaartregel aangemaakt en extern bevestigd",
                key_id=key_id, person=person)
        return await self.repo.create(data, log=log)

    async def update(
        self,
        rule_id: str,
        changes: Dict[str, Any],
        *,
        rev: int,
        bevestig: bool,
        permission: Right,
        person: Optional[str],
        key_id: str,
        reason: Optional[str] = None,
    ) -> SourceMapRule:
        """Wijzig een regel met ``rev``; ``None`` wist een optioneel veld."""
        person = beheerder(permission, person)
        huidig = await self.get(rule_id)
        leeg = [k for k in _VERPLICHT if k in changes and changes[k] is None]
        if leeg:
            raise InvalidInputError(f"Verplicht veld leeg: {', '.join(leeg)}")
        if "location" in changes:
            self.locatiecontrole.controleer(changes["location"])
        data = dict(changes)
        if "route_type" in data:
            data["pipeline_start"] = pipeline_start_voor(data["route_type"])
        na = huidig.model_copy(update=data)
        if any(getattr(na, k) != getattr(huidig, k) for k in _TREKT_BEVESTIGING_IN):
            data.update(classification_confirmed_by=None, classification_confirmed_at=None)
            na = na.model_copy(update={"classification_confirmed_by": None})
        if bevestig:
            if na.classification != "extern":
                raise InvalidInputError("Alleen extern wordt bevestigd")
            data.update(classification_confirmed_by=person, classification_confirmed_at=_nu())
            na = na.model_copy(update={"classification_confirmed_by": person})
        log = None
        if bevestig or na.classification != huidig.classification:
            log = logregel(
                van=huidig.effectieve_classificatie, naar=na.effectieve_classificatie,
                reden=reason or ("Bronkaartregel extern bevestigd" if bevestig
                                 else "Classificatie bronkaartregel gewijzigd"),
                key_id=key_id, person=person)
        return await self.repo.update(str(huidig.id), rev=rev, data=data, door=person, log=log)

    async def confirm(
        self,
        rule_id: str,
        *,
        rev: int,
        permission: Right,
        person: Optional[str],
        key_id: str,
        reason: Optional[str] = None,
    ) -> SourceMapRule:
        """Bevestig ``extern`` (de beheerder legt de classificatie vast)."""
        return await self.update(rule_id, {}, rev=rev, bevestig=True, permission=permission,
                                 person=person, key_id=key_id, reason=reason)
