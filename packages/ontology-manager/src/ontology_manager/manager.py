"""
Ontology Manager - facade for ontology operations.

Provides a singleton manager that coordinates registry, validator,
evolution agent, and document mapper.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from ontology_manager.config import OntologyManagerConfig, get_config
from ontology_manager.document_mapper import get_ontology_for_document_type
from ontology_manager.evolution import OntologyEvolutionAgent, OntologyGap, SchemaProposal
from ontology_manager.prompts import OntologyPromptGenerator
from ontology_manager.registry import OntologyRegistry
from ontology_manager.schema import Ontology
from ontology_manager.validator import OntologyValidator, ValidationReport


class OntologyManager:
    """
    Singleton manager for all ontology operations.

    Provides a unified API over:
    - OntologyRegistry: loading, caching, versioning
    - OntologyValidator: SHACL-like validation
    - OntologyEvolutionAgent: gap tracking and schema proposals
    - OntologyPromptGenerator: LLM prompt generation
    - DocumentMapper: document type to ontology mapping
    """

    _instance: Optional["OntologyManager"] = None
    _initialized: bool = False

    def __new__(cls, config: Optional[OntologyManagerConfig] = None) -> "OntologyManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[OntologyManagerConfig] = None) -> None:
        if OntologyManager._initialized:
            return
        OntologyManager._initialized = True

        self._config = config or get_config()
        self._registry = OntologyRegistry()
        self._evolution = OntologyEvolutionAgent(
            frequency_threshold=self._config.gap_auto_propose_threshold,
            auto_propose=self._config.gap_tracking_enabled,
        )
        logger.info("OntologyManager initialized")

    @property
    def config(self) -> OntologyManagerConfig:
        return self._config

    @property
    def registry(self) -> OntologyRegistry:
        return self._registry

    @property
    def evolution(self) -> OntologyEvolutionAgent:
        return self._evolution

    # --- Ontology Access ---

    async def get_ontology(
        self,
        name: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Optional[Ontology]:
        """Get an ontology by name (defaults to configured default)."""
        name = name or self._config.default_ontology
        return await self._registry.get(name, version)

    async def get_ontology_for_document(
        self,
        document_type: Optional[str] = None,
    ) -> Optional[Ontology]:
        """Get the appropriate ontology for a document type."""
        ontology_name = get_ontology_for_document_type(document_type)
        return await self._registry.get(ontology_name)

    async def list_ontologies(self) -> List[str]:
        """List all available ontology names."""
        return await self._registry.list_ontologies()

    # --- Validation ---

    def create_validator(
        self, ontology: Ontology, strict: bool = False
    ) -> OntologyValidator:
        """Create a validator for the given ontology."""
        return OntologyValidator(ontology, strict=strict)

    async def validate(
        self,
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        ontology_name: Optional[str] = None,
        strict: bool = False,
    ) -> ValidationReport:
        """Validate extracted entities and relationships against an ontology."""
        ontology = await self.get_ontology(ontology_name)
        if not ontology:
            return ValidationReport(
                is_valid=False,
                entity_count=len(entities),
                relationship_count=len(relationships),
            )
        validator = self.create_validator(ontology, strict=strict)
        return validator.validate_extraction_result(entities, relationships)

    # --- Prompt Generation ---

    def create_prompt_generator(
        self, ontology: Ontology
    ) -> OntologyPromptGenerator:
        """Create a prompt generator for the given ontology."""
        return OntologyPromptGenerator(ontology)

    async def generate_extraction_prompt(
        self,
        ontology_name: Optional[str] = None,
        include_concepts: bool = True,
        include_claims: bool = True,
    ) -> Optional[str]:
        """Generate a combined extraction prompt for the given ontology."""
        ontology = await self.get_ontology(ontology_name)
        if not ontology:
            return None
        generator = self.create_prompt_generator(ontology)
        return generator.generate_combined_extraction_prompt(
            include_concepts=include_concepts,
            include_claims=include_claims,
        )

    # --- Gap Tracking ---

    async def track_gap(
        self,
        entity_text: str,
        entity_type_guess: Optional[str] = None,
        context: Optional[str] = None,
        source_id: Optional[str] = None,
        ontology_name: Optional[str] = None,
    ) -> OntologyGap:
        """Record a gap in the ontology (entity that didn't match)."""
        ontology_name = ontology_name or self._config.default_ontology
        return await self._evolution.record_gap(
            entity_text=entity_text,
            entity_type_guess=entity_type_guess,
            context=context,
            source_id=source_id,
            ontology_name=ontology_name,
        )

    async def get_proposals(
        self, ontology_name: Optional[str] = None
    ) -> List[SchemaProposal]:
        """List schema evolution proposals."""
        ontology_name = ontology_name or self._config.default_ontology
        return await self._evolution.list_proposals(ontology_name)

    # --- Reset (for testing) ---

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing only)."""
        cls._instance = None
        cls._initialized = False
        OntologyRegistry._instance = None
        OntologyRegistry._initialized = False


# Module-level accessor
_manager: Optional[OntologyManager] = None


def get_ontology_manager(
    config: Optional[OntologyManagerConfig] = None,
) -> OntologyManager:
    """Get the singleton ontology manager instance."""
    global _manager
    if _manager is None:
        _manager = OntologyManager(config)
    return _manager
