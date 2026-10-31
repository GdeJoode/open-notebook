"""Bronkaart, innamebatch en innamewachtrij (F2-13, ADR 0019, 0020; plan §5.2).

De regels over classificatie staan hier op één plek:

* :func:`standaard_classificatie`: de standaard per innameroute (ADR 0019);
* :func:`effectieve_classificatie`: ``extern`` geldt alleen met een
  bevestiging door een beheerder en bij een actieve regel; anders ``intern``;
* :func:`regel_classificatie`: daarbovenop voor een bronkaartregel: alleen een
  openbare route (API, MCP, feed, website) kan extern zijn;
* :func:`pipeline_start_voor`: de pipelinestart volgt uit de innameroute.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import Field

from shared.models.base import ObjectModel

Classificatie = Literal["intern", "extern"]
RouteType = Literal["api", "mcp", "feed", "website", "bulkupload", "watchfolder", "upload"]
SourceType = Literal["pdf", "map", "website", "api", "video", "podcast", "mcp"]
PipelineStart = Literal["automatisch", "na_bevestiging", "wachtrij"]
FetchContent = Literal["metadata", "transcript", "volledige_media", "volledig_document"]
ItemOrigin = Literal["watchfolder", "upload", "zonder_regel"]
ItemStatus = Literal["wacht", "besloten", "verwerkt"]

#: Routes waarvan de bronnen in de regel openbaar zijn: voorstel ``extern``,
#: de beheerder bevestigt (ADR 0019).
OPENBARE_ROUTES: frozenset[str] = frozenset({"api", "mcp", "feed", "website"})


def standaard_classificatie(route_type: str) -> Classificatie:
    """De standaard van een nieuwe regel: ``extern`` als voorstel voor een
    openbare route, anders ``intern`` (bulkupload: keuze per batch; watchfolder:
    besluit per document; upload: voorstel van de poortwachter per document)."""
    return "extern" if route_type in OPENBARE_ROUTES else "intern"


def pipeline_start_voor(route_type: str) -> PipelineStart:
    """De pipelinestart van een route (plan §5.2; migratie 93 dwingt dit af)."""
    if route_type in OPENBARE_ROUTES:
        return "automatisch"
    if route_type == "watchfolder":
        return "wachtrij"
    return "na_bevestiging"


def standaard_fetch_content(source_type: str) -> FetchContent:
    """Wat er standaard wordt opgehaald; video (YouTube) alleen metadata (FO 5)."""
    return "metadata" if source_type == "video" else "volledig_document"


def effectieve_classificatie(
    classification: Optional[str],
    confirmed_by: Optional[str],
    *,
    active: Optional[bool] = True,
) -> Classificatie:
    """``extern`` alleen als bevestigd door een persoon en actief; anders ``intern``."""
    if (
        classification == "extern"
        and bool(str(confirmed_by or "").strip())
        and active is not False
    ):
        return "extern"
    return "intern"


def regel_classificatie(
    classification: Optional[str],
    confirmed_by: Optional[str],
    *,
    active: Optional[bool],
    route_type: Optional[str],
) -> Classificatie:
    """De effectieve classificatie van een bronkaartregel.

    Alleen een regel op een openbare route kan extern zijn (ADR 0019): een
    watchfolder, upload of bulkupload kiest per document of per batch.
    """
    if route_type not in OPENBARE_ROUTES:
        return "intern"
    return effectieve_classificatie(classification, confirmed_by, active=active)


class SourceMapRule(ObjectModel):
    """Een record van ``source_map_rule`` (migratie 93)."""

    table_name: ClassVar[str] = "source_map_rule"

    name: str
    route_type: RouteType
    source_type: SourceType
    location: str
    schedule: Optional[Dict[str, Any]] = None
    classification: Classificatie = "intern"
    classification_confirmed_by: Optional[str] = None
    classification_confirmed_at: Optional[datetime] = None
    pipeline_start: PipelineStart
    document_type: Optional[List[str]] = None
    route: str
    selection: Optional[List[str]] = None
    label_schema: Optional[Dict[str, Any]] = None
    owner: str = "beheerder"
    fetch_content: FetchContent
    active: bool = True
    rev: int = 0
    modified_by: Optional[str] = None
    modified_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    @property
    def effectieve_classificatie(self) -> Classificatie:
        return regel_classificatie(
            self.classification, self.classification_confirmed_by,
            active=self.active, route_type=self.route_type,
        )


class IntakeBatch(ObjectModel):
    """Een record van ``intake_batch``: één classificatie en route per batch."""

    table_name: ClassVar[str] = "intake_batch"

    classification: Classificatie = "intern"
    classification_confirmed_by: Optional[str] = None
    classification_confirmed_at: Optional[datetime] = None
    route: str
    source_map_rule: Optional[str] = None
    files: List[str] = Field(default_factory=list)
    #: Per bestand de SHA-256 die de fork zelf berekende (F2-14, migratie 95).
    file_hashes: List[Dict[str, Any]] = Field(default_factory=list)
    created_by: str
    rev: int = 0
    modified_by: Optional[str] = None
    modified_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    @property
    def effectieve_classificatie(self) -> Classificatie:
        return effectieve_classificatie(self.classification, self.classification_confirmed_by)


class IntakeItem(ObjectModel):
    """Een record van ``intake_item``, de innamewachtrij."""

    table_name: ClassVar[str] = "intake_item"

    origin: ItemOrigin
    path: str
    content_hash: str
    proposed_classification: Optional[Classificatie] = None
    gatekeeper: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None
    status: ItemStatus = "wacht"
    run: Optional[str] = None
    registered_by: str
    rev: int = 0
    modified_by: Optional[str] = None
    modified_at: Optional[datetime] = None
    created_at: Optional[datetime] = None

    @property
    def effectieve_classificatie(self) -> Classificatie:
        """Het besluit van de beheerder; zonder besluit ``intern`` (FO 9)."""
        if self.status != "wacht" and self.decision:
            return effectieve_classificatie(
                self.decision.get("classification"), self.decision.get("decided_by")
            )
        return "intern"


__all__ = [
    "IntakeBatch",
    "IntakeItem",
    "OPENBARE_ROUTES",
    "SourceMapRule",
    "effectieve_classificatie",
    "pipeline_start_voor",
    "regel_classificatie",
    "standaard_classificatie",
    "standaard_fetch_content",
]
