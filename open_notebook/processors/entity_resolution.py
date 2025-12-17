"""
Entity Resolution Pipeline

Three-tier entity resolution for document processing:
1. EntityRuler: Fast O(1) pattern matching for known entity names
2. EntityLinker: Context-based disambiguation using KnowledgeBase
3. Semantic Fallback: Embedding KNN search for unknown entities

This module integrates with the spaCy-Layout pipeline to provide
comprehensive entity extraction and resolution.

See docs/INTEGRATED_IMPLEMENTATION_PLAN.md 1.7.4 for documentation.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc, Span
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    Language = Any
    Doc = Any
    Span = Any

from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.knowledge_graph import Entity, EntityType, compute_entity_hash
from open_notebook.processors.embeddings import (
    EmbeddingConfig,
    KnowledgeGraphEmbeddings,
    get_kg_embeddings,
)
from open_notebook.processors.entity_linking import (
    EntityLinker,
    EntityLinkingConfig,
)
from open_notebook.processors.kb_builder import (
    KnowledgeBaseConfig,
    SpacyKnowledgeBaseBuilder,
    get_kb_builder,
)
from open_notebook.processors.openie import ExtractedEntity, map_entity_type


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class EntityResolutionConfig:
    """Configuration for the entity resolution pipeline."""

    # Tier 1: EntityRuler settings
    enable_entity_ruler: bool = True
    pattern_priority: bool = True  # Give patterns priority over NER

    # Tier 2: EntityLinker settings
    enable_entity_linker: bool = True
    linker_threshold: float = 0.8  # Minimum confidence for EntityLinker

    # Tier 3: Semantic fallback settings
    enable_semantic_fallback: bool = True
    semantic_threshold: float = 0.85  # Minimum similarity for semantic match

    # New entity creation
    create_new_entities: bool = True
    min_entity_confidence: float = 0.6

    # KB builder configuration
    kb_config: Optional[KnowledgeBaseConfig] = None

    # Linking configuration
    linking_config: Optional[EntityLinkingConfig] = None

    # Entity types to process
    include_entity_types: Optional[List[str]] = None


@dataclass
class ResolvedEntity:
    """Result of entity resolution."""

    # Original span information
    text: str
    start_char: int
    end_char: int
    label: str  # spaCy label (PERSON, ORG, etc.)

    # Resolution result
    entity_id: Optional[str] = None
    entity_name: Optional[str] = None
    entity_type: Optional[str] = None

    # Resolution method
    resolution_method: str = "none"  # "pattern", "linker", "semantic", "new"
    confidence: float = 0.0

    # Additional info
    context: Optional[str] = None  # Surrounding text context
    is_new: bool = False


@dataclass
class EntityResolutionResult:
    """Complete result of entity resolution for a document."""

    # Resolved entities
    entities: List[ResolvedEntity] = field(default_factory=list)

    # Statistics
    total_spans: int = 0
    resolved_by_pattern: int = 0
    resolved_by_linker: int = 0
    resolved_by_semantic: int = 0
    new_entities_created: int = 0
    unresolved: int = 0

    # Entity IDs for creating mentions
    entity_ids: List[str] = field(default_factory=list)


# =============================================================================
# ENTITY RESOLUTION PIPELINE
# =============================================================================


class EntityResolutionPipeline:
    """
    Three-tier entity resolution pipeline.

    Integrates:
    1. spaCy EntityRuler for pattern-based matching
    2. spaCy EntityLinker for context-based disambiguation
    3. Semantic fallback using embedding similarity

    Usage:
        pipeline = EntityResolutionPipeline()
        await pipeline.initialize(nlp)

        result = await pipeline.resolve_entities(doc, source_id)
    """

    def __init__(self, config: Optional[EntityResolutionConfig] = None):
        self.config = config or EntityResolutionConfig()
        self._initialized = False
        self._nlp: Optional[Language] = None
        self._kb_builder: Optional[SpacyKnowledgeBaseBuilder] = None
        self._entity_linker: Optional[EntityLinker] = None
        self._embeddings: Optional[KnowledgeGraphEmbeddings] = None

    async def initialize(
        self,
        nlp: Language,
        force_rebuild_kb: bool = False,
    ) -> bool:
        """
        Initialize the entity resolution pipeline.

        Sets up:
        - EntityRuler with patterns from database
        - EntityLinker with KnowledgeBase
        - Semantic fallback with embedding service

        Args:
            nlp: spaCy Language pipeline to enhance
            force_rebuild_kb: Force rebuild of KB from database

        Returns:
            True if initialization successful
        """
        if not SPACY_AVAILABLE:
            logger.error("spaCy not available. Entity resolution disabled.")
            return False

        try:
            self._nlp = nlp

            # Initialize KB builder
            self._kb_builder = get_kb_builder(self.config.kb_config)

            # Initialize entity linker for semantic fallback
            self._entity_linker = EntityLinker(
                config=self.config.linking_config,
            )
            self._embeddings = get_kg_embeddings()

            # Setup EntityRuler if enabled
            if self.config.enable_entity_ruler:
                await self._setup_entity_ruler(force_rebuild_kb)

            # Setup EntityLinker if enabled
            if self.config.enable_entity_linker:
                await self._setup_entity_linker(force_rebuild_kb)

            self._initialized = True
            logger.info("Entity resolution pipeline initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize entity resolution pipeline: {e}")
            return False

    async def _setup_entity_ruler(self, force_rebuild: bool = False):
        """Setup EntityRuler with patterns from database."""
        if "entity_ruler" in self._nlp.pipe_names:
            self._nlp.remove_pipe("entity_ruler")

        # Get patterns from KB builder
        patterns = await self._kb_builder.build_entity_ruler_patterns(
            force_rebuild=force_rebuild
        )

        if not patterns:
            logger.warning("No patterns available for EntityRuler")
            return

        # Add EntityRuler to pipeline
        # Place before NER if pattern_priority is True
        if self.config.pattern_priority and "ner" in self._nlp.pipe_names:
            ruler = self._nlp.add_pipe(
                "entity_ruler",
                before="ner",
                config={"overwrite_ents": True}
            )
        else:
            ruler = self._nlp.add_pipe(
                "entity_ruler",
                config={"overwrite_ents": False}
            )

        ruler.add_patterns(patterns)
        logger.info(f"Added EntityRuler with {len(patterns)} patterns")

    async def _setup_entity_linker(self, force_rebuild: bool = False):
        """Setup EntityLinker with KnowledgeBase from database."""
        if "entity_linker" in self._nlp.pipe_names:
            self._nlp.remove_pipe("entity_linker")

        # Build KnowledgeBase
        kb = await self._kb_builder.build_knowledge_base(
            self._nlp.vocab,
            force_rebuild=force_rebuild
        )

        if kb is None or kb.get_size_entities() == 0:
            logger.warning("No entities in KnowledgeBase - EntityLinker disabled")
            return

        # Add EntityLinker to pipeline
        try:
            entity_linker = self._nlp.add_pipe("entity_linker", last=True)
            entity_linker.set_kb(kb)
            logger.info(
                f"Added EntityLinker with {kb.get_size_entities()} entities, "
                f"{kb.get_size_aliases()} aliases"
            )
        except Exception as e:
            logger.warning(f"Failed to add EntityLinker: {e}")

    async def resolve_entities(
        self,
        doc: Doc,
        source_id: Optional[str] = None,
        context_window: int = 100,
    ) -> EntityResolutionResult:
        """
        Resolve entities in a spaCy Doc using the three-tier approach.

        Args:
            doc: spaCy Doc with NER annotations
            source_id: Optional source document ID for provenance
            context_window: Characters of context to extract around entities

        Returns:
            EntityResolutionResult with all resolved entities
        """
        if not self._initialized:
            logger.warning("Pipeline not initialized")
            return EntityResolutionResult()

        result = EntityResolutionResult()
        result.total_spans = len(doc.ents)

        seen_entity_ids: Set[str] = set()

        for ent in doc.ents:
            resolved = await self._resolve_single_entity(
                ent, doc.text, context_window
            )

            result.entities.append(resolved)

            # Update statistics
            if resolved.resolution_method == "pattern":
                result.resolved_by_pattern += 1
            elif resolved.resolution_method == "linker":
                result.resolved_by_linker += 1
            elif resolved.resolution_method == "semantic":
                result.resolved_by_semantic += 1
            elif resolved.resolution_method == "new":
                result.new_entities_created += 1
            else:
                result.unresolved += 1

            # Track unique entity IDs
            if resolved.entity_id and resolved.entity_id not in seen_entity_ids:
                result.entity_ids.append(resolved.entity_id)
                seen_entity_ids.add(resolved.entity_id)

        logger.info(
            f"Resolved {result.total_spans} entities: "
            f"{result.resolved_by_pattern} by pattern, "
            f"{result.resolved_by_linker} by linker, "
            f"{result.resolved_by_semantic} by semantic, "
            f"{result.new_entities_created} new, "
            f"{result.unresolved} unresolved"
        )

        return result

    async def _resolve_single_entity(
        self,
        ent: Span,
        full_text: str,
        context_window: int,
    ) -> ResolvedEntity:
        """Resolve a single entity span."""
        # Extract context
        start_ctx = max(0, ent.start_char - context_window)
        end_ctx = min(len(full_text), ent.end_char + context_window)
        context = full_text[start_ctx:end_ctx]

        resolved = ResolvedEntity(
            text=ent.text,
            start_char=ent.start_char,
            end_char=ent.end_char,
            label=ent.label_,
            context=context,
        )

        # Tier 1: Check EntityRuler result (if entity has kb_id)
        if hasattr(ent, "kb_id") and ent.kb_id:
            entity = await Entity.get(ent.kb_id)
            if entity:
                resolved.entity_id = entity.id
                resolved.entity_name = entity.name
                resolved.entity_type = entity.entity_type.value if isinstance(
                    entity.entity_type, EntityType) else entity.entity_type
                resolved.resolution_method = "pattern"
                resolved.confidence = 1.0
                return resolved

        # Tier 1b: Check exact hash match
        hash_id = compute_entity_hash(ent.text)
        existing = await Entity.get_by_hash(hash_id)
        if existing:
            resolved.entity_id = existing.id
            resolved.entity_name = existing.name
            resolved.entity_type = existing.entity_type.value if isinstance(
                existing.entity_type, EntityType) else existing.entity_type
            resolved.resolution_method = "pattern"
            resolved.confidence = 1.0
            return resolved

        # Tier 2: EntityLinker (already applied if in pipeline)
        # The EntityLinker would have set ent.kb_id_ if it found a match
        if hasattr(ent, "kb_id_") and ent.kb_id_:
            entity = await Entity.get(ent.kb_id_)
            if entity:
                resolved.entity_id = entity.id
                resolved.entity_name = entity.name
                resolved.entity_type = entity.entity_type.value if isinstance(
                    entity.entity_type, EntityType) else entity.entity_type
                resolved.resolution_method = "linker"
                resolved.confidence = 0.9  # EntityLinker confidence
                return resolved

        # Tier 3: Semantic fallback
        if self.config.enable_semantic_fallback:
            semantic_match = await self._semantic_entity_search(
                ent.text, ent.label_
            )
            if semantic_match:
                entity, similarity = semantic_match
                resolved.entity_id = entity.id
                resolved.entity_name = entity.name
                resolved.entity_type = entity.entity_type.value if isinstance(
                    entity.entity_type, EntityType) else entity.entity_type
                resolved.resolution_method = "semantic"
                resolved.confidence = similarity
                return resolved

        # No match found - create new entity if configured
        if self.config.create_new_entities:
            new_entity = await self._create_new_entity(ent)
            if new_entity:
                resolved.entity_id = new_entity.id
                resolved.entity_name = new_entity.name
                resolved.entity_type = new_entity.entity_type.value if isinstance(
                    new_entity.entity_type, EntityType) else new_entity.entity_type
                resolved.resolution_method = "new"
                resolved.confidence = 0.7  # New entity confidence
                resolved.is_new = True

                # Add to KB cache for future lookups
                if new_entity.embedding:
                    await self._kb_builder.add_entity_to_cache(
                        entity_id=new_entity.id,
                        name=new_entity.name,
                        entity_type=resolved.entity_type,
                        embedding=new_entity.embedding,
                    )

                return resolved

        return resolved

    async def _semantic_entity_search(
        self,
        text: str,
        label: str,
    ) -> Optional[Tuple[Entity, float]]:
        """Search for semantically similar entities."""
        if not self._entity_linker:
            return None

        # Map spaCy label to entity type
        label_to_type = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.PRODUCT,
            "WORK_OF_ART": EntityType.CONCEPT,
        }
        entity_type = label_to_type.get(label)

        try:
            similar = await self._entity_linker.find_similar_entities(
                name=text,
                entity_type=entity_type,
                limit=5,
            )

            for entity, similarity in similar:
                if similarity >= self.config.semantic_threshold:
                    return entity, similarity

        except Exception as e:
            logger.debug(f"Semantic search failed: {e}")

        return None

    async def _create_new_entity(self, ent: Span) -> Optional[Entity]:
        """Create a new entity from an unmatched span."""
        # Map spaCy label to entity type
        label_to_type = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "EVENT": EntityType.EVENT,
            "PRODUCT": EntityType.PRODUCT,
            "WORK_OF_ART": EntityType.CONCEPT,
            "FAC": EntityType.LOCATION,
            "NORP": EntityType.ORGANIZATION,
            "LAW": EntityType.CONCEPT,
        }
        entity_type = label_to_type.get(ent.label_, EntityType.OTHER)

        # Generate embedding
        embedding = None
        if self._embeddings:
            embedding = await self._embeddings.embed_entity(ent.text)

        try:
            # Use find_or_create to handle potential duplicates
            entity = await Entity.find_or_create(
                name=ent.text,
                entity_type=entity_type,
                embedding=embedding,
            )
            return entity

        except Exception as e:
            logger.error(f"Failed to create new entity '{ent.text}': {e}")
            return None

    async def create_mentions(
        self,
        source_id: str,
        result: EntityResolutionResult,
    ) -> int:
        """
        Create mentions edges between source and resolved entities.

        Args:
            source_id: Source document ID
            result: Entity resolution result

        Returns:
            Number of mentions created
        """
        mentions_created = 0
        source_record_id = ensure_record_id(source_id)

        for resolved in result.entities:
            if not resolved.entity_id:
                continue

            entity_record_id = ensure_record_id(resolved.entity_id)

            try:
                # Check if mention already exists
                existing = await repo_query(
                    """
                    SELECT id FROM mentions
                    WHERE in = $source AND out = $entity
                    LIMIT 1
                    """,
                    {"source": source_record_id, "entity": entity_record_id}
                )

                if existing:
                    continue

                # Create mention edge
                await repo_query(
                    """
                    RELATE $source->mentions->$entity SET
                        context = $context,
                        confidence = $confidence,
                        resolution_method = $method,
                        start_char = $start_char,
                        end_char = $end_char
                    """,
                    {
                        "source": source_record_id,
                        "entity": entity_record_id,
                        "context": resolved.context,
                        "confidence": resolved.confidence,
                        "method": resolved.resolution_method,
                        "start_char": resolved.start_char,
                        "end_char": resolved.end_char,
                    }
                )

                mentions_created += 1

            except Exception as e:
                logger.debug(f"Failed to create mention for {resolved.text}: {e}")

        logger.info(f"Created {mentions_created} mention edges for source {source_id}")
        return mentions_created

    async def refresh_knowledge_base(self):
        """Refresh KnowledgeBase and patterns from database."""
        if not self._initialized or not self._nlp:
            return

        await self._setup_entity_ruler(force_rebuild=True)
        await self._setup_entity_linker(force_rebuild=True)
        logger.info("KnowledgeBase and patterns refreshed")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the pipeline."""
        stats = {
            "initialized": self._initialized,
            "entity_ruler_enabled": self.config.enable_entity_ruler,
            "entity_linker_enabled": self.config.enable_entity_linker,
            "semantic_fallback_enabled": self.config.enable_semantic_fallback,
        }

        if self._kb_builder:
            stats["kb_stats"] = self._kb_builder.get_stats()

        if self._nlp:
            stats["pipeline_components"] = self._nlp.pipe_names

        return stats


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


# Global pipeline instance
_resolution_pipeline: Optional[EntityResolutionPipeline] = None


def get_resolution_pipeline(
    config: Optional[EntityResolutionConfig] = None
) -> EntityResolutionPipeline:
    """Get or create a singleton EntityResolutionPipeline instance."""
    global _resolution_pipeline
    if _resolution_pipeline is None:
        _resolution_pipeline = EntityResolutionPipeline(config)
    return _resolution_pipeline


async def resolve_entities_in_doc(
    doc,
    source_id: Optional[str] = None,
    nlp=None,
    config: Optional[EntityResolutionConfig] = None,
) -> EntityResolutionResult:
    """
    Convenience function to resolve entities in a document.

    Args:
        doc: spaCy Doc or text string
        source_id: Optional source ID for mentions
        nlp: spaCy pipeline (required if doc is string)
        config: Optional resolution config

    Returns:
        EntityResolutionResult
    """
    pipeline = get_resolution_pipeline(config)

    # Process text if needed
    if isinstance(doc, str) and nlp:
        doc = nlp(doc)

    if not pipeline._initialized and nlp:
        await pipeline.initialize(nlp)

    result = await pipeline.resolve_entities(doc, source_id)

    # Create mentions if source_id provided
    if source_id and result.entity_ids:
        await pipeline.create_mentions(source_id, result)

    return result


async def process_entities_for_source(
    source_id: str,
    text: str,
    nlp,
    config: Optional[EntityResolutionConfig] = None,
) -> EntityResolutionResult:
    """
    Process and resolve all entities in a source document.

    This is the main entry point for entity extraction and resolution
    during document processing.

    Args:
        source_id: Source document ID
        text: Full text of the document
        nlp: Configured spaCy pipeline
        config: Optional resolution configuration

    Returns:
        EntityResolutionResult with all entities and mentions created
    """
    pipeline = get_resolution_pipeline(config)

    if not pipeline._initialized:
        await pipeline.initialize(nlp)

    # Process document
    doc = nlp(text)

    # Resolve entities
    result = await pipeline.resolve_entities(doc, source_id)

    # Create mentions edges
    if result.entity_ids:
        await pipeline.create_mentions(source_id, result)

    return result
