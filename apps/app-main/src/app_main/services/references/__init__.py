"""Reference-extraction orchestration (Track V.5 / G.3).

Bridges the GROBID reference producer (``GrobidReferenceService``) to Track U.3's
``cites`` materialization: extract a source's references from its original PDF
(``source.asset.file_path``), then feed the corpus' references into U.3's
whole-corpus matcher.
"""

from app_main.services.references.reference_extraction_service import (
    ReferenceExtractionService,
    ReferenceMaterializationSummary,
)

__all__ = [
    "ReferenceExtractionService",
    "ReferenceMaterializationSummary",
]
