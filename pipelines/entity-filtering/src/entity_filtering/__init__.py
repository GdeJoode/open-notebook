"""
Entity filtering pipeline.

Generic entity/relation filtering, normalization, deduplication,
and scoring. Takes ExtractionResult from ontology-extraction and
produces FilteredResult.
"""

from entity_filtering.config import FilteringConfig
from entity_filtering.filters.base import FilterBase
from entity_filtering.workflow import FilteringWorkflow

__all__ = [
    "FilteringConfig",
    "FilteringWorkflow",
    "FilterBase",
]
