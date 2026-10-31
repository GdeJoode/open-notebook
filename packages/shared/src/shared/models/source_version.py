"""Bronversie (F2-04, ADR 0003; PROV-O entiteit).

Een gewijzigde bron levert een nieuwe versie; de vorige blijft bewaard. Een
versie is uniek op bron en inhoudshash (SHA-256 van de ruwe inhoud).
"""

from __future__ import annotations

from datetime import datetime
from typing import ClassVar, Optional

from shared.models.base import ObjectModel
from shared.models.source import Classification


class SourceVersion(ObjectModel):
    """Een record van ``source_version`` (migratie 90)."""

    table_name: ClassVar[str] = "source_version"

    source: str
    version_no: int
    content_hash: str
    fetched_at: Optional[datetime] = None
    original: Optional[str] = None
    predecessor: Optional[str] = None
    classification: Optional[Classification] = None
    created_by_run: str
    current_run: Optional[str] = None
