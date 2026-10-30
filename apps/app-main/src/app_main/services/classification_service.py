"""Herclassificatie van een bron (F2-03, ADR 0001, 0002 B; FO 6 en 14).

Alleen een sleutel met recht ``beheerder`` en een persoon kan een bron
herclassificeren, en alleen met een reden. De bron, haar fragmenten,
embeddings en ``doc_node``-records krijgen in één transactie de nieuwe waarde,
de afgeleide records (entiteit, alias, relatie, triage-item) worden opnieuw
berekend, en er komt precies één regel in ``classification_log``.

De regels zelf (standaard intern, nooit losser dan de bron, intern naar extern
alleen hier) staan in de database (``migrations/89.surrealql``), zodat geen
ander schrijfpad ze kan omzeilen.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from shared.models.agents import Right
from surrealdb_service.repositories.source import SourceRepository

from app_main.exceptions import InvalidInputError, NotFoundError

CLASSIFICATIES = ("intern", "extern")


@dataclass(frozen=True)
class Herclassificatie:
    source_id: str
    from_classification: str
    to_classification: str
    changed: bool
    log_id: Optional[str] = None


class ClassificationService:
    def __init__(self, source_repo: Optional[SourceRepository] = None) -> None:
        self.source_repo = source_repo or SourceRepository()

    async def reclassify(
        self,
        source_id: str,
        to: str,
        reason: str,
        *,
        permission: Right,
        person: Optional[str],
        key_id: str,
    ) -> Herclassificatie:
        """Zet de bron op ``to``. Geen wijziging: geen logregel.

        Raises:
            PermissionError: de sleutel is geen ``beheerder`` of heeft geen persoon.
            InvalidInputError: onbekende classificatie, lege reden of geen bron-id.
            NotFoundError: de bron bestaat niet.
        """
        person = (person or "").strip()
        if permission != Right.BEHEERDER or not person:
            raise PermissionError("Herclassificeren vraagt een beheerder met persoon")
        if to not in CLASSIFICATIES:
            raise InvalidInputError(f"Onbekende classificatie: {to!r}")
        reason = (reason or "").strip()
        if not reason:
            raise InvalidInputError("Een reden is verplicht")
        source_id = source_id if ":" in source_id else f"source:{source_id}"
        if not source_id.startswith("source:"):
            raise InvalidInputError(f"Geen bron: {source_id!r}")

        current = await self.source_repo.get_classification(source_id)
        if current is None:
            raise NotFoundError(f"Bron niet gevonden: {source_id}")
        if current == to:
            return Herclassificatie(source_id, current, to, changed=False)

        log_id = await self.source_repo.reclassify(
            source_id,
            from_classification=current,
            to_classification=to,
            reason=reason,
            agent_key=key_id,
            person=person,
        )
        return Herclassificatie(source_id, current, to, changed=True, log_id=log_id)
