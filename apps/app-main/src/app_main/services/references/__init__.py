"""Reference-extraction orchestration (Track V.5).

Bridges the pure V.1-V.3 reference producer (``shared.references``) to Track
U.3's ``cites`` materialization: extract a source's references from its persisted
chunks, then feed the corpus' references into U.3's whole-corpus matcher.
"""

from app_main.services.references.reference_extraction_service import (
    ReferenceExtractionService,
    ReferenceMaterializationSummary,
)

__all__ = [
    "ReferenceExtractionService",
    "ReferenceMaterializationSummary",
]
