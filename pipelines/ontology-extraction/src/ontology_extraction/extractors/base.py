"""Abstract base class for ontology-guided extractors."""

from abc import ABC, abstractmethod

from ontology_manager.schema import Ontology
from shared.models.extraction import ExtractionResult


class ExtractorBase(ABC):
    """Base class for all ontology-guided extractors."""

    @abstractmethod
    async def extract(self, text: str, ontology: Ontology, **kwargs) -> ExtractionResult:
        """Extract entities and relations from text using the given ontology."""
        ...
