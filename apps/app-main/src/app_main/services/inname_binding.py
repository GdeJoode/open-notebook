"""Binding van een run en een import aan regel, batch en wachtrij-item (F2-14).

ADR 0019 en 0020 ("Uitwerking in F2-13"), ADR 0003 ("Uitwerking in F2-04"):
de classificatie van de invoer komt uit de database, nooit uit de aanroeper.

* **Openbare route** (API, MCP, feed, website): de bronkaartregel, alleen
  extern als die bevestigd, actief en extern is (:func:`regel_classificatie`).
  Een batch of item hoort hier niet bij.
* **Bulkupload**: de batch, die bij de regel moet horen
  (``intake_batch.source_map_rule``).
* **Watchfolder en losse upload**: het besluit over het wachtrij-item. Het item
  moet ``besloten`` zijn, van dezelfde herkomst als de route van de regel, en
  het pad moet onder de locatie van de regel liggen. Een item met herkomst
  ``zonder_regel`` hoort bij een run zonder regel.
* Een inactieve regel geeft ``intern``.

:func:`bepaal_invoer` leest regel, batch en item telkens opnieuw, zodat de
import het actuele besluit toetst en niet de stand van het moment waarop de
run begon.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from shared.models.source_map import (
    OPENBARE_ROUTES,
    Classificatie,
    IntakeBatch,
    IntakeItem,
    SourceMapRule,
    regel_classificatie,
)
from surrealdb_service.repositories.base import eigen_record_id
from surrealdb_service.repositories.intake import (
    IntakeBatchRepository,
    IntakeItemRepository,
)
from surrealdb_service.repositories.source_map import SourceMapRuleRepository

from app_main.services.source_map_service import canoniek_pad, ligt_onder, weigerreden

#: Herkomst van een wachtrij-item per route van de regel.
_HERKOMST_PER_ROUTE = {"watchfolder": "watchfolder", "upload": "upload"}


class BindingError(Exception):
    """Run, regel, batch of item horen niet bij elkaar (HTTP 409, ``niet_gebonden``)."""


class ItemStatusError(Exception):
    """Het wachtrij-item is niet (meer) besloten (HTTP 409, ``verkeerde_status``)."""


@dataclass(frozen=True)
class Invoer:
    """De classificatie van de invoer en waar die vandaan komt."""

    classification: Classificatie
    basis: str
    regel: Optional[SourceMapRule] = None
    batch: Optional[IntakeBatch] = None
    item: Optional[IntakeItem] = None


def record_tekst(waarde: Optional[str], tabel: str) -> Optional[str]:
    """Een id als ``tabel:sleutel``; ``None`` als het leeg of ongeldig is."""
    if not waarde:
        return None
    rid = eigen_record_id(waarde, tabel)
    return str(rid) if rid is not None else None


class InnameBinding:
    """Leest regel, batch en item en toetst of ze bij elkaar horen."""

    def __init__(
        self,
        rule_repo: Optional[SourceMapRuleRepository] = None,
        batch_repo: Optional[IntakeBatchRepository] = None,
        item_repo: Optional[IntakeItemRepository] = None,
    ) -> None:
        self.rule_repo = rule_repo or SourceMapRuleRepository()
        self.batch_repo = batch_repo or IntakeBatchRepository()
        self.item_repo = item_repo or IntakeItemRepository()

    async def bepaal_invoer(
        self,
        *,
        rule_id: Optional[str],
        batch_id: Optional[str],
        item_id: Optional[str],
        run_id: Optional[str] = None,
    ) -> Invoer:
        """De invoer van een run of import; ``BindingError`` als iets niet klopt.

        ``run_id`` is gezet bij een import: een item dat al ``verwerkt`` is,
        telt dan alleen als het door die run is verwerkt.
        """
        regel = None
        if rule_id:
            regel = await self.rule_repo.get(rule_id)
            if regel is None:
                raise BindingError("onbekende bronkaartregel")
        batch = item = None
        if batch_id and item_id:
            raise BindingError("een run hoort bij een batch of een item, niet bij beide")
        if batch_id:
            batch = await self.batch_repo.get(batch_id)
            if batch is None:
                raise BindingError("onbekende batch")
            if regel is None or regel.route_type != "bulkupload" or (
                record_tekst(batch.source_map_rule, "source_map_rule") != str(regel.id)
            ):
                raise BindingError("batch hoort niet bij de regel")
            klasse = batch.effectieve_classificatie
            basis = "batch"
        elif item_id:
            item = await self.item_repo.get(item_id)
            if item is None:
                raise BindingError("onbekend wachtrij-item")
            self._item_hoort_bij_regel(item, regel)
            if item.status == "wacht" or (
                item.status == "verwerkt"
                and record_tekst(item.run, "processing_run") != record_tekst(run_id, "processing_run")
            ):
                raise ItemStatusError("item is niet besloten")
            klasse = item.effectieve_classificatie
            basis = "besluit"
        elif regel is not None:
            if regel.route_type not in OPENBARE_ROUTES:
                # Bulkupload vraagt een batch, watchfolder en upload een item.
                raise BindingError("deze route vraagt een batch of wachtrij-item")
            klasse = regel_classificatie(
                regel.classification, regel.classification_confirmed_by,
                active=regel.active, route_type=regel.route_type,
            )
            basis = "bronkaartregel"
        else:
            return Invoer(classification="intern", basis="standaard")
        if regel is not None and not regel.active:
            klasse = "intern"
        return Invoer(classification=klasse, basis=basis, regel=regel, batch=batch, item=item)

    @staticmethod
    def _item_hoort_bij_regel(item: IntakeItem, regel: Optional[SourceMapRule]) -> None:
        if item.origin == "zonder_regel":
            if regel is not None:
                raise BindingError("item zonder regel hoort niet bij een regel")
            return
        if regel is None or _HERKOMST_PER_ROUTE.get(regel.route_type) != item.origin:
            raise BindingError("item hoort niet bij de regel")
        if weigerreden(regel.location) is not None or weigerreden(item.path) is not None:
            raise BindingError("item hoort niet bij de regel")
        if not ligt_onder(canoniek_pad(item.path), canoniek_pad(regel.location)):
            raise BindingError("item ligt niet onder de locatie van de regel")
