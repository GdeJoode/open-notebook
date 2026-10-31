"""Bronkaartregels beheren (F2-13, ADR 0019, 0020, 0004; plan §5.2).

* Standaardclassificatie per innameroute (ADR 0019): ``api``, ``mcp``,
  ``feed`` en ``website`` stellen ``extern`` voor. Alleen die routes kunnen op
  regelniveau extern zijn; ``watchfolder``, ``upload`` en ``bulkupload``
  weigeren ``extern`` (422): daar kiest de beheerder per document of batch.
* ``extern`` geldt pas na een bevestiging door een beheerder
  (``classification_confirmed_by``, ``_at``) bij een actieve regel. Bevestigen
  van een inactieve regel wordt geweigerd (422).
* Elke inhoudelijke wijziging trekt een bevestiging in; alleen
  :data:`ZONDER_INTREKKING` (``name``, ``owner``, ``document_type``: puur
  beschrijvend) niet. Ook ``active`` van false naar true trekt haar in.
* Verandert de **effectieve** classificatie (voor en na), dan komt er één
  regel in ``classification_log`` met de persoon, in dezelfde transactie als
  de wijziging: bij bevestigen (intern -> extern) en bij elke intrekking of
  wijziging die een externe regel intern maakt (extern -> intern).
* Een wijziging gaat met ``rev`` (``wijzig_met_rev``); verouderd geeft 409.
* **Locatie** (ADR 0004, 0020): een padlocatie moet onder een toegestane
  hoofdmap liggen (``SIGNAAL_INNAME_PADEN``, allowlist; leeg weigert elk pad)
  en niet in de vault of de exportmap. Omwegen via Windows-padvormen worden
  geweigerd (zie :func:`weigerreden`).
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
    OPENBARE_ROUTES,
    SourceMapRule,
    pipeline_start_voor,
    standaard_classificatie,
    standaard_fetch_content,
)
from surrealdb_service.repositories.source_map import SourceMapRuleRepository

from app_main import config
from app_main.exceptions import InvalidInputError, NotFoundError


class VerbodenLocatieError(Exception):
    """De locatie is niet toegestaan (HTTP 422, ADR 0004)."""


# ---------------------------------------------------------------------------
# Locatiecontrole
# ---------------------------------------------------------------------------

_URL = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*)://")
_SCHEMA = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*):")
_WINDOWS = re.compile(r"^([A-Za-z]):/(.*)$")
_WSL = re.compile(r"^/mnt/([A-Za-z])(?:/|$)(.*)$")
_KORTE_NAAM = re.compile(r"~\d")
_LOKAAL = {"localhost", "127.0.0.1", "::1", "[::1]"}


def _windows(letter: str, rest: str) -> str:
    # Een Windows-pad is hoofdletterongevoelig.
    return f"{letter.lower()}:" + posixpath.normpath("/" + rest.lstrip("/")).lower()


def weigerreden(pad: str) -> Optional[str]:
    """Waarom een pad (geen URL) wordt geweigerd, of ``None``.

    Geweigerd: ``\\\\`` of ``//`` vooraan (UNC, ``\\\\?\\``, ``\\\\.\\``), een
    ``:`` na de schijfletter (``::$DATA``) of ergens anders, een schijfletter
    zonder scheidingsteken (``D:#``), een segment dat eindigt op een punt of
    spatie, en een korte 8.3-naam (``~1``).
    """
    tekst = pad.replace("\\", "/")
    if tekst.startswith("//"):
        return "UNC of apparaatpad"
    if re.match(r"^[A-Za-z]:", tekst):
        if not re.match(r"^[A-Za-z]:/", tekst):
            return "schijfletter zonder scheidingsteken"
        rest = tekst[2:]
    else:
        rest = tekst
    if ":" in rest:
        return "dubbele punt in het pad"
    for segment in rest.split("/"):
        if segment in ("", ".", ".."):
            continue
        if segment[-1] in ". ":
            return "segment eindigt op punt of spatie"
        if _KORTE_NAAM.search(segment):
            return "korte 8.3-naam"
    return None


def canoniek_pad(pad: str) -> str:
    """Een pad dat :func:`weigerreden` doorliet, genormaliseerd.

    Een Windows-pad (``D:\\...``) en het WSL-koppelpunt (``/mnt/d/...``)
    krijgen dezelfde vorm (``d:/...``, kleine letters). Een POSIX-pad gaat door
    ``os.path.realpath``: ``..`` en symbolische koppelingen worden opgelost.
    """
    tekst = pad.strip().replace("\\", "/")
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


def _in_een_van(kind: str, mappen: Iterable[str]) -> bool:
    for pad in mappen:
        if weigerreden(pad.strip()) is None and _ligt_in(kind, canoniek_pad(pad)):
            return True
    return False


class Locatiecontrole:
    """Allowlist van innamemappen, en nooit de vault of de exportmap.

    De mappen komen uit de instellingen van de fork (``config``), tenzij ze
    hier worden meegegeven (tests).
    """

    def __init__(
        self,
        vault_paden: Optional[List[str]] = None,
        exportmap_paden: Optional[List[str]] = None,
        inname_paden: Optional[List[str]] = None,
    ) -> None:
        self._vault = vault_paden
        self._export = exportmap_paden
        self._inname = inname_paden

    @property
    def verboden(self) -> List[str]:
        vault = self._vault if self._vault is not None else config.get_vault_paden()
        export = self._export if self._export is not None else config.get_exportmap_paden()
        return [*vault, *export]

    @property
    def toegestaan(self) -> List[str]:
        return self._inname if self._inname is not None else config.get_inname_paden()

    def _in_vault(self, pad: str) -> bool:
        return weigerreden(pad) is None and _in_een_van(canoniek_pad(pad), self.verboden)

    def controleer(self, locatie: str) -> None:
        """``VerbodenLocatieError`` als de locatie niet is toegestaan."""
        tekst = (locatie or "").strip()
        url = _URL.match(tekst)
        if url:
            schema = url.group(1).lower()
            if schema == "file":
                raise VerbodenLocatieError("file-adres")
            if schema == "obsidian":
                raise VerbodenLocatieError("obsidian-adres")
            delen = urlsplit(tekst)
            if (delen.hostname or "").lower() in _LOKAAL:
                pad = unquote(delen.path)
                if self._in_vault(pad) or self._in_vault(pad.lstrip("/")):
                    raise VerbodenLocatieError("lokaal adres naar de vault")
            return
        schema = _SCHEMA.match(tekst)
        if schema and len(schema.group(1)) > 1:
            # Een schema zonder `//`, zoals `file:/D:/...`.
            raise VerbodenLocatieError("adres zonder //")
        if weigerreden(tekst) is not None:
            raise VerbodenLocatieError("ongeldige padvorm")
        kind = canoniek_pad(tekst)
        if _in_een_van(kind, self.verboden):
            raise VerbodenLocatieError("pad in de vault of de exportmap")
        if not _in_een_van(kind, self.toegestaan):
            raise VerbodenLocatieError("pad buiten de innamemappen")


# ---------------------------------------------------------------------------
# Dienst
# ---------------------------------------------------------------------------

#: Velden die een bevestiging niet intrekken als ze veranderen: puur
#: beschrijvend, zonder invloed op wat er wordt opgehaald of hoe.
ZONDER_INTREKKING = frozenset({"name", "owner", "document_type"})
#: Velden die niet leeg mogen worden.
_VERPLICHT = ("name", "route_type", "source_type", "location", "classification",
              "route", "owner", "fetch_content", "active")
#: Velden die niet tot de inhoud van de regel horen.
_BEHEER = frozenset({"id", "rev", "modified_by", "modified_at", "created", "updated",
                     "created_at", "classification_confirmed_by",
                     "classification_confirmed_at"})


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


def _controleer_extern(route_type: str, classification: str) -> None:
    if classification == "extern" and route_type not in OPENBARE_ROUTES:
        raise InvalidInputError("Extern kan alleen op een openbare route")


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
        _controleer_extern(route_type, data["classification"])
        data.setdefault("fetch_content", standaard_fetch_content(data.get("source_type", "")))
        data["pipeline_start"] = pipeline_start_voor(route_type)
        data.update(modified_by=person, modified_at=_nu())
        log = None
        if bevestig:
            self._kan_bevestigen(data["classification"], data.get("active", True))
            data.update(classification_confirmed_by=person, classification_confirmed_at=_nu())
            log = logregel(van="intern", naar="extern",
                           reden=reason or "Bronkaartregel aangemaakt en extern bevestigd",
                           key_id=key_id, person=person)
        return await self.repo.create(data, log=log)

    @staticmethod
    def _kan_bevestigen(classification: str, active: Optional[bool]) -> None:
        if classification != "extern":
            raise InvalidInputError("Alleen extern wordt bevestigd")
        if active is False:
            raise InvalidInputError("Een inactieve regel wordt niet bevestigd")

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
        data = {k: v for k, v in changes.items() if k not in _BEHEER}
        if "route_type" in data:
            data["pipeline_start"] = pipeline_start_voor(data["route_type"])
        na = huidig.model_copy(update=data)
        _controleer_extern(na.route_type, na.classification)

        gewijzigd = {k for k, v in data.items() if getattr(huidig, k, None) != v}
        if gewijzigd - ZONDER_INTREKKING and huidig.classification_confirmed_by:
            data.update(classification_confirmed_by=None, classification_confirmed_at=None)
            na = na.model_copy(update={"classification_confirmed_by": None,
                                       "classification_confirmed_at": None})
        if bevestig:
            self._kan_bevestigen(na.classification, na.active)
            data.update(classification_confirmed_by=person, classification_confirmed_at=_nu())
            na = na.model_copy(update={"classification_confirmed_by": person})

        log = None
        voor, daarna = huidig.effectieve_classificatie, na.effectieve_classificatie
        if voor != daarna:
            standaard = ("Bronkaartregel extern bevestigd" if daarna == "extern"
                         else "Bevestiging bronkaartregel ingetrokken")
            log = logregel(van=voor, naar=daarna, reden=reason or standaard,
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
