"""Innamebatches en de innamewachtrij (F2-13, ADR 0019, 0020; plan §5.2, §5.4).

* **Batch** (bulkupload): één classificatie en één route voor de hele batch;
  zonder keuze ``intern``. ``extern`` geldt pas na bevestiging door een
  beheerder. Aanmaken, wijzigen en bevestigen alleen door een ``beheerder``.
* **Wachtrij-item**: Langflow (``pijplijn``) meldt documenten uit een
  watchfolder, losse uploads en bronnen zonder regel aan, haalt besloten items
  op en zet een item op ``verwerkt``, alleen met een run die aan dat item
  gebonden is (F2-14). De hash van een watchfolder-item berekent de fork
  zelf (B29). Een watchfolder-item heeft geen
  standaardclassificatie en wacht; een upload krijgt het voorstel van de
  poortwachter. Tot het besluit van de beheerder geldt ``intern`` (FO 9).
  Het besluit (route en classificatie) neemt alleen een ``beheerder``, met
  één regel in ``classification_log``; daarna ligt het vast (409).
* **Bestandshashes** (F2-14): bij het aanmaken en wijzigen van een batch
  berekent de fork per bestand zelf de SHA-256 (``file_hashes``); bij een
  bevestiging naar ``extern`` moet elk bestand leesbaar zijn (anders 422). De
  import toetst het origineel aan die hash.
* Paden volgen de locatiecontrole van een regel: onder een innamemap, niet in
  de vault of de exportmap (ADR 0004).
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from shared.models.agents import Right
from shared.models.source_map import IntakeBatch, IntakeItem
from surrealdb_service.repositories.base import eigen_record_id
from surrealdb_service.repositories.intake import (
    IntakeBatchRepository,
    IntakeItemRepository,
)
from surrealdb_service.repositories.processing_run import ProcessingRunRepository
from surrealdb_service.repositories.source_map import SourceMapRuleRepository

from app_main.exceptions import InvalidInputError, NotFoundError
from app_main.services.inname_binding import BindingError
from app_main.services.source_map_service import (
    Locatiecontrole,
    beheerder,
    classificatielogs,
    controleer_rev,
    logregel,
)

#: Velden van een batch die een eerdere bevestiging intrekken als ze veranderen.
_BATCH_TREKT_BEVESTIGING_IN = ("classification", "route", "files")


class IntakeConflictError(Exception):
    """Het item staat niet in de status die de actie vraagt (HTTP 409)."""


def _nu() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(pad: str) -> str:
    h = hashlib.sha256()
    with open(pad, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    return "sha256:" + h.hexdigest()


async def bestandshashes(files: List[str], *, streng: bool) -> List[Dict[str, str]]:
    """Per bestand de SHA-256, door de fork zelf berekend (F2-14).

    ``streng``: elk bestand moet leesbaar zijn (bevestiging naar extern),
    anders 422; zonder ``streng`` ontbreekt een onleesbaar bestand in de lijst.
    """
    uit = []
    for pad in files:
        try:
            uit.append({"path": pad, "content_hash": await asyncio.to_thread(_sha256, pad)})
        except OSError:
            if streng:
                raise InvalidInputError("Batchbestand niet leesbaar") from None
    return uit


class IntakeService:
    def __init__(
        self,
        batch_repo: Optional[IntakeBatchRepository] = None,
        item_repo: Optional[IntakeItemRepository] = None,
        rule_repo: Optional[SourceMapRuleRepository] = None,
        locatiecontrole: Optional[Locatiecontrole] = None,
        run_repo: Optional[ProcessingRunRepository] = None,
    ) -> None:
        self.batch_repo = batch_repo or IntakeBatchRepository()
        self.item_repo = item_repo or IntakeItemRepository()
        self.rule_repo = rule_repo or SourceMapRuleRepository()
        self.locatiecontrole = locatiecontrole or Locatiecontrole()
        self.run_repo = run_repo or ProcessingRunRepository()

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
            "file_hashes": await bestandshashes(files, streng=bevestig),
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
        controleer_rev(rev, huidig.rev, huidig.id)
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
        tussen = na.effectieve_classificatie
        if bevestig or "files" in changes:
            data["file_hashes"] = await bestandshashes(na.files, streng=bevestig)
        if bevestig:
            if na.classification != "extern":
                raise InvalidInputError("Alleen extern wordt bevestigd")
            data.update(classification_confirmed_by=person, classification_confirmed_at=_nu())
        logs = classificatielogs(
            voor=huidig.effectieve_classificatie, tussen=tussen, bevestig=bevestig,
            reden=reason, onderwerp="Batch", key_id=key_id, person=person)
        return await self.batch_repo.update(
            str(huidig.id), rev=huidig.rev, data=data, door=person, logs=logs)

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
        """Meld een item aan; geeft (item, nieuw). Hetzelfde item nog eens geeft het bestaande.

        Een watchfolder-item hasht de fork zelf (B29): de watchfolder is
        alleen-lezen gekoppeld; een onleesbaar bestand of een andere hash dan
        opgegeven geeft 422.
        """
        self.locatiecontrole.controleer(path)
        if origin == "watchfolder":
            try:
                berekend = await asyncio.to_thread(_sha256, path)
            except OSError:
                raise InvalidInputError("Wachtrij-item niet leesbaar") from None
            if berekend != content_hash:
                raise InvalidInputError("Hash van het wachtrij-item wijkt af")
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
        controleer_rev(rev, huidig.rev, huidig.id)
        if huidig.status != "wacht":
            # Het besluit ligt vast; een ander besluit vraagt een nieuw item.
            raise IntakeConflictError("Over dit item is al besloten")
        besluit = {"route": route, "classification": classification,
                   "decided_by": person, "decided_at": _nu()}
        log = logregel(van=huidig.effectieve_classificatie, naar=classification,
                       reden=reason or "Besluit over wachtrij-item",
                       key_id=key_id, person=person)
        return await self.item_repo.update(
            str(huidig.id), rev=huidig.rev, data={"status": "besloten", "decision": besluit},
            door=person, logs=[log])

    async def mark_processed(
        self, item_id: str, *, rev: int, run_id: Optional[str], door: str,
        key_id: Optional[str] = None,
    ) -> IntakeItem:
        """Zet een besloten item op ``verwerkt`` (``pijplijn``); een run moet aan
        het item gebonden zijn en door dezelfde sleutel zijn aangemaakt (F2-14)."""
        huidig = await self.get_item(item_id)
        controleer_rev(rev, huidig.rev, huidig.id)
        if huidig.status != "besloten":
            raise IntakeConflictError("Alleen een besloten item wordt verwerkt")
        data: Dict[str, Any] = {"status": "verwerkt"}
        if run_id:
            run = eigen_record_id(run_id, "processing_run")
            if run is None:
                raise InvalidInputError("Geen id van een verwerkingsrun")
            # Alleen een run die aan dit item gebonden is (F2-14).
            gelezen = await self.run_repo.get(str(run))
            if (gelezen is None or gelezen.intake_item != str(huidig.id)
                    or not gelezen.created_by_key or gelezen.created_by_key != key_id):
                raise BindingError("run hoort niet bij het item")
            data["run"] = run
        return await self.item_repo.update(str(huidig.id), rev=huidig.rev, data=data, door=door)
