"""
Source Processing Graph using spaCy-Layout Pipeline.

This module defines the LangGraph workflow for processing source documents,
replacing the previous content-core based implementation.

Includes integration with:
- Entity Resolution Pipeline (3-tier entity extraction and linking)
- RAPTOR Hierarchical Summarization
- Knowledge Graph Extraction
"""

import operator
from typing import Any, Dict, List, Optional

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger
from typing_extensions import Annotated, TypedDict

from open_notebook.domain.content_settings import ContentSettings
from open_notebook.domain.notebook import Asset, Chunk, Source
from open_notebook.domain.transformation import Transformation
from open_notebook.graphs.transformation import graph as transform_graph
from open_notebook.processors.spacy_pipeline import (
    ProcessingInput,
    ProcessingOutput,
    process_document,
)


class SourceState(TypedDict):
    """State for source processing workflow."""
    # Input data
    file_path: Optional[str]
    url: Optional[str]
    content: Optional[str]
    title: Optional[str]
    # Processing state
    processing_output: Optional[ProcessingOutput]
    apply_transformations: List[Transformation]
    source_id: str
    notebook_ids: List[str]
    source: Source
    transformation: Annotated[list, operator.add]
    embed: bool
    chunks: Optional[List[Dict[str, Any]]]
    # Knowledge Graph extraction
    extract_knowledge: bool
    kg_extraction_result: Optional[Dict[str, Any]]
    # RAPTOR processing
    raptor_enabled: bool
    raptor_result: Optional[Dict[str, Any]]


class TransformationState(TypedDict):
    """State for transformation subprocess."""
    source: Source
    transformation: Transformation


async def content_process(state: SourceState) -> dict:
    """
    Process document using spaCy-Layout pipeline.

    Replaces the previous content-core based extraction.
    """
    # Load settings from database
    content_settings = await ContentSettings.get_instance()

    # Get input data from state
    file_path = state.get("file_path")
    url = state.get("url")
    content = state.get("content")
    title = state.get("title")

    # Log processing configuration
    gpu_enabled = content_settings.spacy_gpu_enabled if hasattr(content_settings, 'spacy_gpu_enabled') else True
    gpu_device = content_settings.spacy_gpu_device if hasattr(content_settings, 'spacy_gpu_device') else "auto"

    logger.info(
        f"📄 Processing with spaCy-Layout pipeline, "
        f"gpu_enabled={gpu_enabled}, device={gpu_device}"
    )

    # Process using new pipeline
    try:
        processing_output, chunks = await process_document(
            file_path=file_path,
            url=url,
            content=content,
            title=title,
            gpu_enabled=gpu_enabled,
            gpu_device=gpu_device,
        )
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise

    # File Management System Integration
    if file_path:
        try:
            from open_notebook.utils.file_manager import organize_file

            # Get file management settings
            file_operation = content_settings.file_operation or "copy"
            naming_scheme = content_settings.output_naming_scheme or "date_prefix"
            input_dir = content_settings.input_directory_path or "./data/input"
            output_dir = content_settings.output_directory_path or "./data/output"

            logger.info(f"📁 Starting file management workflow for {file_path}")

            # Organize file (copy/move to INPUT, copy to OUTPUT)
            input_path, output_path = organize_file(
                file_path=file_path,
                input_dir=input_dir,
                output_dir=output_dir,
                file_operation=file_operation,
                naming_scheme=naming_scheme,
            )

            # Store organized paths in metadata
            if processing_output.metadata is None:
                processing_output.metadata = {}
            processing_output.metadata["organized_input_path"] = str(input_path) if input_path else None
            processing_output.metadata["organized_output_path"] = str(output_path)

            logger.info("✅ File management workflow completed successfully")

        except Exception as e:
            logger.error(f"File management workflow failed: {e}")
            # Don't fail the entire processing if file management fails

    return {
        "processing_output": processing_output,
        "chunks": chunks
    }


async def save_source(state: SourceState) -> dict:
    """Save processed source and chunks to database."""
    from open_notebook.database.repository import repo_query, ensure_record_id

    processing_output = state["processing_output"]

    # Get existing source using the provided source_id
    source = await Source.get(state["source_id"])
    if not source:
        raise ValueError(f"Source with ID {state['source_id']} not found")

    # Update source with processed content
    source.asset = Asset(
        url=processing_output.url,
        file_path=processing_output.file_path
    )
    source.full_text = processing_output.content

    # Preserve existing title if none provided in processed content
    if processing_output.title:
        source.title = processing_output.title

    await source.save()

    # Save chunks if extracted
    chunks = state.get("chunks")
    if chunks:
        try:
            logger.info(f"Saving {len(chunks)} chunks for source {source.id}")

            # Delete existing chunks for this source first (for idempotency)
            delete_result = await repo_query(
                "DELETE chunk WHERE source = $source_id",
                {"source_id": ensure_record_id(source.id)}
            )
            deleted_count = len(delete_result) if delete_result else 0
            if deleted_count > 0:
                logger.info(f"Deleted {deleted_count} existing chunks for source {source.id}")

            # Create new chunks
            for chunk_data in chunks:
                chunk = Chunk(
                    source=source.id,
                    text=chunk_data["text"],
                    chunk_order=chunk_data["chunk_order"],
                    physical_page=chunk_data["physical_page"],
                    printed_page=chunk_data.get("printed_page"),
                    chapter=chunk_data.get("chapter"),
                    paragraph_number=chunk_data.get("paragraph_number"),
                    element_type=chunk_data["element_type"],
                    positions=chunk_data.get("positions", []),
                    metadata=chunk_data.get("metadata", {})
                )
                await chunk.save()

            logger.info(f"Successfully saved {len(chunks)} chunks for source {source.id}")

        except Exception as e:
            logger.error(f"Error saving chunks for source {source.id}: {str(e)}")
            logger.exception(e)
            # Don't fail the whole process if chunk saving fails
            logger.warning("Continuing despite chunk save failure")

    # Embed content for vector search if requested
    if state["embed"]:
        logger.debug("Embedding content for vector search")
        await source.vectorize()

    return {"source": source}


async def extract_knowledge_graph(state: SourceState) -> dict:
    """
    Extract entities and claims for the knowledge graph.

    Uses ontology-driven extraction with document type detection:
    1. Detect document type for specialized extraction
    2. Select appropriate ontology based on document type
    3. Extract entities and claims with EntityLinker deduplication
    """
    if not state.get("extract_knowledge", False):
        return {"kg_extraction_result": None}

    source = state["source"]
    if not source or not source.full_text:
        logger.warning("No source text for KG extraction")
        return {"kg_extraction_result": None}

    try:
        from open_notebook.processors.openie import OpenIEExtractor, map_claim_type
        from open_notebook.processors.entity_linking import EntityLinker
        from open_notebook.domain.knowledge_graph import Claim
        from open_notebook.database.repository import repo_query, ensure_record_id

        # Get content settings for extraction config
        content_settings = await ContentSettings.get_instance()
        kg_enabled = getattr(content_settings, 'kg_extraction_enabled', True)
        linking_enabled = getattr(content_settings, 'kg_entity_linking_enabled', True)

        if not kg_enabled:
            logger.info("Knowledge graph extraction is disabled in settings")
            return {"kg_extraction_result": None}

        logger.info(f"🧠 Starting knowledge graph extraction for source {source.id}")

        extractor = OpenIEExtractor()
        entities_extracted = 0
        claims_extracted = 0
        detected_type = None

        # Import ontology mapper for document-type-specific extraction
        try:
            from open_notebook.ontology.document_ontology_mapper import get_ontology_for_document_type

            # Phase 1: Detect document type
            logger.info("Phase 1: Detecting document type...")
            doc_type_result = await extractor._detect_document_type(source.full_text)
            detected_type = doc_type_result.get("document_type")
            type_confidence = doc_type_result.get("confidence", 0.0)

            # Update source with detected document type if confident
            if detected_type and type_confidence >= 0.7:
                source.source_type = detected_type
                await source.save()
                logger.info(f"Detected document type: {detected_type} (confidence: {type_confidence:.2f})")

            # Phase 2: Select appropriate ontology based on document type
            ontology_name = get_ontology_for_document_type(detected_type)
            logger.info(f"Phase 2: Using ontology '{ontology_name}' for extraction (document type: {detected_type})")

            # Phase 3: Extract with specialized ontology
            extraction_result = await extractor.extract_with_ontology(
                source.full_text,
                ontology_name=ontology_name,
                detect_document_type=False  # Already detected
            )
            extraction_result.document_type = detected_type
            extraction_result.document_type_confidence = type_confidence

        except Exception as ontology_err:
            logger.warning(f"Ontology-driven extraction failed, using generic: {ontology_err}")
            extraction_result = await extractor.extract_all(source.full_text)

        if not extraction_result:
            logger.warning("No extraction result from OpenIE")
            return {"kg_extraction_result": None}

        # Process entities with deduplication through EntityLinker
        if linking_enabled and extraction_result.entities:
            linker = EntityLinker()
            link_results = await linker.process_extracted_entities(
                extraction_result.entities,
                source_id=str(source.id)
            )
            entities_extracted = link_results["total"]

            # Create mentions relationships
            source_record_id = ensure_record_id(str(source.id))
            for entity_info in link_results["entities"]:
                try:
                    confidence = next(
                        (e.confidence for e in extraction_result.entities
                         if e.name == entity_info["extracted_name"]),
                        0.8
                    )
                    # Check if mention already exists
                    existing = await repo_query(
                        "SELECT * FROM mentions WHERE in = $source_id AND out = $entity_id LIMIT 1",
                        {"source_id": source_record_id, "entity_id": ensure_record_id(entity_info["entity_id"])}
                    )
                    if not existing:
                        await repo_query(
                            "RELATE $source_id->mentions->$entity_id SET confidence = $confidence, created_at = time::now()",
                            {
                                "source_id": source_record_id,
                                "entity_id": ensure_record_id(entity_info["entity_id"]),
                                "confidence": confidence
                            }
                        )
                except Exception as me:
                    logger.debug(f"Error creating mention: {me}")

            logger.info(
                f"Entity linking: {link_results['new']} new, "
                f"{link_results['existing']} existing, "
                f"{link_results['linked']} linked"
            )

        # Process claims
        if extraction_result.claims:
            source_record_id = ensure_record_id(str(source.id))
            for extracted_claim in extraction_result.claims:
                try:
                    claim_type = map_claim_type(extracted_claim.claim_type)
                    claim = Claim(
                        statement=extracted_claim.statement,
                        claim_type=claim_type,
                        confidence=extracted_claim.confidence
                    )
                    await claim.save()
                    claims_extracted += 1

                    # Create supports relationship
                    await repo_query(
                        "RELATE $source_id->supports->$claim_id SET strength = $strength, quote = $quote, created_at = time::now()",
                        {
                            "source_id": source_record_id,
                            "claim_id": ensure_record_id(claim.id),
                            "strength": extracted_claim.confidence,
                            "quote": getattr(extracted_claim, 'supporting_quote', None) or getattr(extracted_claim, 'context', None)
                        }
                    )
                except Exception as ce:
                    logger.debug(f"Error processing claim: {ce}")

            logger.info(f"Created {claims_extracted} claims for source {source.id}")

        result = {
            "entities_extracted": entities_extracted,
            "claims_extracted": claims_extracted,
            "triples_extracted": len(extraction_result.triples) if extraction_result.triples else 0,
            "document_type": detected_type,
        }

        logger.info(f"✅ KG extraction complete: {result}")
        return {"kg_extraction_result": result}

    except Exception as e:
        logger.error(f"Knowledge graph extraction failed: {e}")
        logger.exception(e)
        return {"kg_extraction_result": {"error": str(e)}}


async def process_raptor(state: SourceState) -> dict:
    """
    Process source with RAPTOR hierarchical summarization.

    Creates multi-layer summary chunks for improved retrieval.
    """
    if not state.get("raptor_enabled", False):
        return {"raptor_result": None}

    source = state["source"]
    if not source:
        return {"raptor_result": None}

    try:
        # Check if RAPTOR is enabled in settings
        content_settings = await ContentSettings.get_instance()
        raptor_enabled = getattr(content_settings, 'raptor_enabled', False)

        if not raptor_enabled:
            logger.debug("RAPTOR is disabled in settings")
            return {"raptor_result": None}

        # Check minimum chunk requirement
        min_chunks = getattr(content_settings, 'raptor_min_chunks', 5)
        chunks = state.get("chunks", [])

        if not chunks or len(chunks) < min_chunks:
            logger.info(
                f"Source has {len(chunks) if chunks else 0} chunks, "
                f"below minimum {min_chunks} for RAPTOR"
            )
            return {"raptor_result": {"skipped": "insufficient_chunks"}}

        logger.info(f"🌲 Starting RAPTOR processing for source {source.id}")

        # Import RAPTOR components
        from open_notebook.processors.raptor import (
            RaptorProcessor,
            RaptorConfig,
        )

        # Build RAPTOR config from settings
        max_layers = getattr(content_settings, 'raptor_max_layers', 5)
        summarization_model = getattr(content_settings, 'raptor_summarization_model', None)

        config = RaptorConfig(
            max_layers=max_layers,
            summarization_model=summarization_model,
        )

        # Process with RAPTOR
        processor = RaptorProcessor(config)
        tree = await processor.process_source(source.id)

        result = {
            "layers": tree.num_layers if tree else 0,
            "nodes_created": len(tree.nodes) if tree and tree.nodes else 0,
        }

        logger.info(f"✅ RAPTOR complete: {result['layers']} layers, {result['nodes_created']} nodes")
        return {"raptor_result": result}

    except Exception as e:
        logger.error(f"RAPTOR processing failed: {e}")
        logger.exception(e)
        return {"raptor_result": {"error": str(e)}}


def trigger_transformations(state: SourceState, config: RunnableConfig) -> List[Send]:
    """Trigger transformation subprocesses if any are requested."""
    if len(state["apply_transformations"]) == 0:
        return []

    to_apply = state["apply_transformations"]
    logger.debug(f"Applying transformations {to_apply}")

    return [
        Send(
            "transform_content",
            {
                "source": state["source"],
                "transformation": t,
            },
        )
        for t in to_apply
    ]


async def transform_content(state: TransformationState) -> Optional[dict]:
    """Apply a transformation to source content."""
    source = state["source"]
    content = source.full_text
    if not content:
        return None
    transformation: Transformation = state["transformation"]

    logger.debug(f"Applying transformation {transformation.name}")
    result = await transform_graph.ainvoke(
        dict(input_text=content, transformation=transformation)
    )
    await source.add_insight(transformation.title, result["output"])
    return {
        "transformation": [
            {
                "output": result["output"],
                "transformation_name": transformation.name,
            }
        ]
    }


def should_extract_knowledge(state: SourceState) -> str:
    """Determine if we should extract knowledge graph data."""
    if state.get("extract_knowledge", False):
        return "extract_knowledge_graph"
    return "check_raptor"


def should_process_raptor(state: SourceState) -> str:
    """Determine if we should process with RAPTOR."""
    if state.get("raptor_enabled", False):
        return "process_raptor"
    return "trigger_transformations"


def trigger_transformations_or_end(state: SourceState) -> str:
    """Determine if we should trigger transformations or end."""
    if len(state.get("apply_transformations", [])) > 0:
        return "transform_content"
    return END


# Create and compile the workflow
workflow = StateGraph(SourceState)

# Add nodes
workflow.add_node("content_process", content_process)
workflow.add_node("save_source", save_source)
workflow.add_node("extract_knowledge_graph", extract_knowledge_graph)
workflow.add_node("process_raptor", process_raptor)
workflow.add_node("transform_content", transform_content)

# Define the graph edges
# Main flow: content_process -> save_source -> optional steps -> transformations
workflow.add_edge(START, "content_process")
workflow.add_edge("content_process", "save_source")

# After saving, conditionally extract knowledge graph
workflow.add_conditional_edges(
    "save_source",
    should_extract_knowledge,
    {
        "extract_knowledge_graph": "extract_knowledge_graph",
        "check_raptor": "process_raptor",  # Skip KG extraction
    }
)

# After KG extraction, check RAPTOR
workflow.add_conditional_edges(
    "extract_knowledge_graph",
    should_process_raptor,
    {
        "process_raptor": "process_raptor",
        "trigger_transformations": "transform_content",
    }
)

# After RAPTOR, trigger transformations
workflow.add_conditional_edges(
    "process_raptor",
    trigger_transformations_or_end,
    {
        "transform_content": "transform_content",
        END: END,
    }
)

# Transformations end the workflow
workflow.add_edge("transform_content", END)

# Compile the graph
source_graph = workflow.compile()
