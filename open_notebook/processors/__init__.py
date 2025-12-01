"""
Custom document processors for Open Notebook.

Note: GPU and VLM processing is now handled by content-core.
See open_notebook/utils/content_core_config.py for configuration bridge.
"""

from open_notebook.processors.chunk_extractor import extract_chunks_from_docling

# OpenIE exports
from open_notebook.processors.openie import (
    OpenIEConfig,
    OpenIEExtractor,
    ExtractionResult,
    ExtractedEntity,
    ExtractedTriple,
    ExtractedClaim,
    extract_knowledge_from_text,
    extract_entities_from_text,
    map_entity_type,
    map_claim_type,
)

# Embedding exports
from open_notebook.processors.embeddings import (
    EmbeddingConfig,
    EmbeddingService,
    KnowledgeGraphEmbeddings,
    generate_embedding,
    generate_embeddings_batch,
    get_kg_embeddings,
)

# Entity linking exports
from open_notebook.processors.entity_linking import (
    EntityLinkingConfig,
    EntityLinker,
    link_entity,
    find_duplicate_entities,
)

__all__ = [
    # Chunk extraction
    "extract_chunks_from_docling",
    # OpenIE
    "OpenIEConfig",
    "OpenIEExtractor",
    "ExtractionResult",
    "ExtractedEntity",
    "ExtractedTriple",
    "ExtractedClaim",
    "extract_knowledge_from_text",
    "extract_entities_from_text",
    "map_entity_type",
    "map_claim_type",
    # Embeddings
    "EmbeddingConfig",
    "EmbeddingService",
    "KnowledgeGraphEmbeddings",
    "generate_embedding",
    "generate_embeddings_batch",
    "get_kg_embeddings",
    # Entity linking
    "EntityLinkingConfig",
    "EntityLinker",
    "link_entity",
    "find_duplicate_entities",
]
