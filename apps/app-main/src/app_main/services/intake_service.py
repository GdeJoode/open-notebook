"""Innamebatches en de innamewachtrij (F2-13, ADR 0019, 0020; plan §5.2, §5.4).

* **Batch** (bulkupload): één classificatie en één route voor de hele batch;
  zonder keuze ``intern``. ``extern`` geldt pas na bevestiging door een
  beheerder. Aanmaken, wijzigen en bevestigen alleen door een ``beheerder``.
* **Wachtrij-item**: Langflow (``pijplijn``) meldt documenten uit een
  watchfolder, losse uploads en bronnen zonder regel aan, haalt besloten items
  op en zet een item op ``verwerkt``. Een watchfolder-item heeft geen
  standaardclassificatie en wacht; een upload krijgt het voorstel van de
  poortwachter. Tot het besluit van de beheerder geldt ``intern`` (FO 9).
  Het besluit (route en classificatie) neemt alleen een ``beheerder``, met
  één regel in ``classification_log``; daarna ligt het vast (409).
* Paden volgen de locatiecontrole van een regel: onder een innamemap, niet in
  de vault of de exportmap (ADR 0004).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from shared.models.agents import Right
from shared.models.source_map import IntakeBatch, IntakeItem
from surrealdb_service.repositories.base import eigen_record_id
from surrealdb_service.repositories.intake import (
    IntakeBatchRepository,
    IntakeItemRepository,
)
from surrealdb_service.repositories.source_map import SourceMapRuleRepository

from app_main.exceptions import InvalidInputError, NotFoundError
from app_main.services.source_map_service import Locatiecontrole, beheerder, logregel

#: Velden van een batch die een eerdere bevestiging intrekken als ze veranderen.
_BATCH_TREKT_BEVESTIGING_IN = ("classification", "route", "files")


class IntakeConflictError(Exception):
    """Het item staat niet in de status die de actie vraagt (HTTP 409)."""


def _nu() -> datetime:
    return datetime.now(timezone.utc)


class IntakeService:
    def __init__(
        self,
        batch_repo: Optional[IntakeBatchRepository] = None,
        item_repo: Optional[IntakeItemRepository] = None,
        rule_repo: Optional[SourceMapRuleRepository] = None,
        locatiecontrole: Optional[Locatiecontrole] = None,
    ) -> None:
        self.batch_repo = batch_repo or IntakeBatchRepository()
        self.item_repo = item_repo or IntakeItemRepository()
        self.rule_repo = rule_repo or SourceMapRuleRepository()
        self.locatiecontrole = locatiecontrole or Locatiecontrole()

    # -- Batches ------------------------------------------------------------

    async def list_batches(self) -> List[IntakeBatch]:
        return await self.batch_repo.list()

    async def get_batch(self, batch_id: str) -> IntakeBatch:
        batch = await self.batch_repo.get(batch_id)
        if batch is None:
            raise NotFoundError("Batch niet gevonden")
        return batch

    async def create_batch(
        self,
        *,
        route: str,
        files: List[str],
        classification: Optional[str] = None,
        source_map_rule: Optional[str] = None,
        bevestig: bool = False,
        permission: Right,
        person: Optional[str],
        key_id: str,
        reason: Optional[str] = None,
    ) -> IntakeBatch:
        """Maak een batch; zonder keuze ``intern`` (ADR 0019)."""
        person = beheerder(permission, person)
        for pad in files:
            self.locatiecontrole.controleer(pad)
        data: Dict[str, Any] = {
            "route": route,
            "files": files,
            "classification": classification or "intern",
            "created_by": person,
            "modified_by": person,
            "modified_at": _nu(),
        }
        if source_map_rule:
            regel = await self.rule_repo.get(source_map_rule)
            if regel is None:
                raise InvalidInputError("Onbekende bronkaartregel")
            if regel.route_type != "bulkupload":
                raise InvalidInputError("Een batch hoort bij een regel voor bulkupload")
            data["source_map_rule"] = eigen_record_id(regel.id, "source_map_rule")
        log = None
        if bevestig:
            if data["classification"] != "extern":
                raise InvalidInputError("Alleen extern wordt bevestigd")
            data.update(classification_confirmed_by=person, classification_confirmed_at=_nu())
            log = logregel(van="intern", naar="extern",
                           reden=reason or "Batch aangemaakt en extern bevestigd",
                           key_id=key_id, person=person)
        return await self.batch_repo.create(data, log=log)

    async def update_batch(
        self,
        batch_id: str,
        changes: Dict[str, Any],
        *,
        rev: int,
        bevestig: bool = False,
        permission: Right,
        person: Optional[str],
        key_id: str,
        reason: Optional[str] = None,
    ) -> IntakeBatch:
        person = beheerder(permission, person)
        huidig = await self.get_batch(batch_id)
        leeg = [k for k in ("route", "classification", "files") if k in changes and changes[k] is None]
        if leeg:
            raise InvalidInputError(f"Verplicht veld leeg: {', '.join(leeg)}")
        for pad in changes.get("files") or []:
            self.locatiecontrole.controleer(pad)
        data = dict(changes)
        na = huidig.model_copy(update=data)
        if any(getattr(na, k) != getattr(huidig, k) for k in _BATCH_TREKT_BEVESTIGING_IN):
            data.update(classification_confirmed_by=None, classification_confirmed_at=None)
            na = na.model_copy(update={"classification_confirmed_by": None})
        if bevestig:
            if na.classification != "extern":
                raise InvalidInputError("Alleen extern wordt bevestigd")
            data.update(classification_confirmed_by=person, classification_confirmed_at=_nu())
            na = na.model_copy(update={"classification_confirmed_by": person})
        # Eén logregel als de effectieve classificatie verandert (bevestiging of
        # intrekking), zoals bij een bronkaartregel.
        log = None
        voor, daarna = huidig.effectieve_classificatie, na.effectieve_classificatie
        if voor != daarna:
            log = logregel(
                van=voor, naar=daarna,
                reden=reason or ("Batch extern bevestigd" if daarna == "extern"
                                 else "Bevestiging batch ingetrokken"),
                key_id=key_id, person=person)
        return await self.batch_repo.update(str(huidig.id), rev=rev, data=data, door=person, log=log)

    async def confirm_batch(self, batch_id: str, *, rev: int, permission: Right,
                            person: Optional[str], key_id: str,
                            reason: Optional[str] = None) -> IntakeBatch:
        return await self.update_batch(batch_id, {}, rev=rev, bevestig=True,
                                       permission=permission, person=person,
                                       key_id=key_id, reason=reason)

    # -- Wachtrij -----------------------------------------------------------

    async def register_item(
        self,
        *,
        origin: str,
        path: str,
        content_hash: str,
        gatekeeper: Optional[Dict[str, Any]],
        registered_by: str,
    ) -> tuple[IntakeItem, bool]:
        """Meld een item aan; geeft (item, nieuw). Hetzelfde item nog eens geeft het bestaande."""
        self.locatiecontrole.controleer(path)
        voorstel = None
        if origin == "upload":
            # Losse upload: voorstel van de poortwachter, zonder voorstel intern.
            voorstel = (gatekeeper or {}).get("decision") or "intern"
        # Watchfolder en bron zonder regel: geen standaard (ADR 0019).
        return await self.item_repo.register({
            "origin": origin,
            "path": path,
            "content_hash": content_hash,
            "gatekeeper": gatekeeper,
            "proposed_classification": voorstel,
            "registered_by": registered_by,
        })

    async def list_items(self, *, status: Optional[str], permission: Right) -> List[IntakeItem]:
        """``pijplijn`` haalt alleen besloten items op; de beheerder alles."""
        if permission != Right.BEHEERDER and status != "besloten":
            raise PermissionError("Alleen besloten items")
        return await self.item_repo.list(status=status)

    async def get_item(self, item_id: str) -> IntakeItem:
        item = await self.item_repo.get(item_id)
        if item is None:
            raise NotFoundError("Wachtrij-item niet gevonden")
        return item

    async def decide_item(
        self,
        item_id: str,
        *,
        rev: int,
        route: str,
        classification: str,
        permission: Right,
        person: Optional[str],
        key_id: str,
        reason: Optional[str] = None,
    ) -> IntakeItem:
        """Het besluit van de beheerder: route en classificatie (status ``besloten``)."""
        person = beheerder(permission, person)
        huidig = await self.get_item(item_id)
        if huidig.status != "wacht":
            # Het besluit ligt vast; een ander besluit vraagt een nieuw item.
            raise IntakeConflictError("Over dit item is al besloten")
        besluit = {"route": route, "classification": classification,
                   "decided_by": person, "decided_at": _nu()}
        log = logregel(van=huidig.effectieve_classificatie, naar=classification,
                       reden=reason or "Besluit over wachtrij-item",
                       key_id=key_id, person=person)
        return await self.item_repo.update(
            str(huidig.id), rev=rev, data={"status": "besloten", "decision": besluit},
            door=person, log=log)

    async def mark_processed(
        self, item_id: str, *, rev: int, run_id: Optional[str], door: str
    ) -> IntakeItem:
        """Zet een besloten item op ``verwerkt`` (``pijplijn``)."""
        huidig = await self.get_item(item_id)
        if huidig.status != "besloten":
            raise IntakeConflictError("Alleen een besloten item wordt verwerkt")
        data: Dict[str, Any] = {"status": "verwerkt"}
        if run_id:
            run = eigen_record_id(run_id, "processing_run")
            if run is None:
                raise InvalidInputError("Geen id van een verwerkingsrun")
            data["run"] = run
        return await self.item_repo.update(str(huidig.id), rev=rev, data=data, door=door)
