"""
Custom document processors for Open Notebook.

Document processing is handled by the spaCy-Layout pipeline,
which integrates Docling for multi-format parsing.
"""

from open_notebook.processors.chunk_extractor import extract_chunks_from_docling

# GPU Detection
from open_notebook.processors.gpu_detection import (
    GPUConfig,
    GPUDevice,
    detect_gpu,
    get_gpu_config,
    get_optimal_config,
    setup_spacy_gpu,
)

# spaCy-Layout Pipeline
from open_notebook.processors.spacy_pipeline import (
    ChunkData,
    ProcessingInput,
    ProcessingOutput,
    SpacyLayoutPipeline,
    process_document,
)

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

# KnowledgeBase Builder exports
from open_notebook.processors.kb_builder import (
    KnowledgeBaseConfig,
    SpacyKnowledgeBaseBuilder,
    get_kb_builder,
    build_knowledge_base,
    get_entity_ruler_patterns,
    add_entity_to_kb,
)

# Entity Resolution Pipeline exports
from open_notebook.processors.entity_resolution import (
    EntityResolutionConfig,
    EntityResolutionPipeline,
    EntityResolutionResult,
    ResolvedEntity,
    get_resolution_pipeline,
    resolve_entities_in_doc,
    process_entities_for_source,
)

# Metadata enrichment exports
from open_notebook.processors.metadata_enrichment import (
    EnrichmentConfig,
    MetadataEnrichmentService,
    AcademicMetadata,
    Author,
    CrossRefClient,
    OpenAlexClient,
    enrich_academic_paper,
)

# Ontology validation exports
from open_notebook.processors.validation import (
    ValidationSeverity,
    ValidationIssue,
    ValidationResult,
    OntologyValidator,
    validate_entity,
    validate_entities_batch,
    get_validation_report,
)

__all__ = [
    # Chunk extraction
    "extract_chunks_from_docling",
    # GPU Detection
    "GPUConfig",
    "GPUDevice",
    "detect_gpu",
    "get_gpu_config",
    "get_optimal_config",
    "setup_spacy_gpu",
    # spaCy-Layout Pipeline
    "ChunkData",
    "ProcessingInput",
    "ProcessingOutput",
    "SpacyLayoutPipeline",
    "process_document",
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
    # KnowledgeBase Builder
    "KnowledgeBaseConfig",
    "SpacyKnowledgeBaseBuilder",
    "get_kb_builder",
    "build_knowledge_base",
    "get_entity_ruler_patterns",
    "add_entity_to_kb",
    # Entity Resolution Pipeline
    "EntityResolutionConfig",
    "EntityResolutionPipeline",
    "EntityResolutionResult",
    "ResolvedEntity",
    "get_resolution_pipeline",
    "resolve_entities_in_doc",
    "process_entities_for_source",
    # Metadata enrichment
    "EnrichmentConfig",
    "MetadataEnrichmentService",
    "AcademicMetadata",
    "Author",
    "CrossRefClient",
    "OpenAlexClient",
    "enrich_academic_paper",
    # Ontology validation
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationResult",
    "OntologyValidator",
    "validate_entity",
    "validate_entities_batch",
    "get_validation_report",
]
