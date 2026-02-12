"""
Abstract base class for entity/relation filters.

All concrete filters must implement filter_entities and filter_relations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List


class FilterBase(ABC):
    """Base class for entity/relation filters.

    Subclasses implement the two abstract methods to provide
    specific filtering logic. Filters are composable and applied
    sequentially by the workflow orchestrator.
    """

    @abstractmethod
    def filter_entities(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter entities, returning only those that pass.

        Args:
            entities: List of entity dicts (model_dump output).

        Returns:
            Filtered list of entity dicts.
        """
        ...

    @abstractmethod
    def filter_relations(
        self,
        relations: List[Dict[str, Any]],
        valid_entities: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter relations, ensuring they reference valid entities.

        Args:
            relations: List of relation dicts (model_dump output).
            valid_entities: Entities that survived filtering.

        Returns:
            Filtered list of relation dicts.
        """
        ...
