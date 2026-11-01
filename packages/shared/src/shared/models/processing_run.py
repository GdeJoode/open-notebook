"""Verwerkingsrun (F2-04, ADR 0003; PROV-O activiteit).

Een run legt vast welke pipeline (flow uit Git, met commit en hash) een
bronversie verwerkte, met status, logboek en modellen. Langflow maakt de run
aan met ``POST /api/v1/runs`` en sluit hem af met ``PATCH`` (plan §5.4).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from shared.models.base import ObjectModel

RunStatus = Literal["bezig", "geslaagd", "mislukt", "niets_nieuws"]
#: ``inname``: de eerste verwerking door de fork zelf; ``handmatig`` alleen voor
#: correcties door de beheerder (ADR 0005).
RunTrigger = Literal["schema", "mapcontrole", "feed", "inname", "handmatig", "herverwerking"]
#: De classificatie van de invoer van een run; 'onbekend' telt als intern.
InputClassification = Literal["intern", "extern", "onbekend"]

#: Een run met een van deze statussen is afgesloten.
AFGESLOTEN: frozenset[str] = frozenset({"geslaagd", "mislukt", "niets_nieuws"})


class PipelineVersion(BaseModel):
    """De flow van een run: Git-commit en SHA-256 van de flow-JSON."""

    git_commit: Optional[str] = None
    flow_hash: Optional[str] = None


class RunLogEntry(BaseModel):
    """Eén regel in het runlogboek: tijdstip, stap, status en melding."""

    at: Optional[datetime] = None
    step: str
    status: str
    message: str = ""


class ProcessingRun(ObjectModel):
    """Een record van ``processing_run`` (migratie 90)."""

    table_name: ClassVar[str] = "processing_run"

    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    status: RunStatus = "bezig"
    trigger: RunTrigger
    pipeline_id: str
    pipeline_version: Optional[PipelineVersion] = None
    input_classification: InputClassification = "onbekend"
    flow_vastgelegd: bool = False
    label_schema: Optional[Dict[str, Any]] = None
    models: List[Dict[str, Any]] = Field(default_factory=list)
    log: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    attempts: int = 0
    notified_at: Optional[datetime] = None
    source_map_rule_id: Optional[str] = None
    source_map_rule_rev: Optional[int] = None
    #: Wie de run startte; verplicht bij ``handmatig`` (migratie 92, ADR 0005).
    actor: Optional[str] = None
    #: De sleutel die de run aanmaakte via ``POST /api/v1/runs`` (F2-14,
    #: migratie 94). Alleen die importeert in de run en sluit hem af.
    created_by_key: Optional[str] = None
    #: De batch of het wachtrij-item waaraan de run gebonden is (F2-14).
    intake_batch: Optional[str] = None
    intake_item: Optional[str] = None

    @property
    def afgesloten(self) -> bool:
        return self.status in AFGESLOTEN


#: De stappen na de import die Langflow start (plan §5.4, ADR 0011).
StapNaam = Literal["embed", "entity_extraction", "entity_dedup", "graph"]
StapStatus = Literal["wachtrij", "bezig", "geslaagd", "mislukt", "wacht_op_review"]
#: Een stap met een van deze statussen is klaar; hij verandert niet meer.
STAP_KLAAR: frozenset[str] = frozenset({"geslaagd", "mislukt"})


class RunStep(ObjectModel):
    """Een record van ``run_step`` (migratie 104, F2-15): één stap van een run
    op één bronversie, uitgevoerd door een job in de wachtrij van de fork."""

    table_name: ClassVar[str] = "run_step"

    run: str
    stap: StapNaam
    source_version: str
    source: str
    job: Optional[str] = None
    status: StapStatus = "wachtrij"
    attempts: int = 0
    #: Vaste melding zonder inhoud (patroon in migratie 104).
    error: Optional[str] = None
    review_item: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
