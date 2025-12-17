import time
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel
from surreal_commands import CommandInput, CommandOutput, command

from open_notebook.database.repository import ensure_record_id
from open_notebook.domain.notebook import Source
from open_notebook.domain.transformation import Transformation

try:
    from open_notebook.graphs.source import source_graph
except ImportError as e:
    logger.error(f"Failed to import source_graph: {e}")
    raise ValueError("source_graph not available")


def full_model_dump(model):
    if isinstance(model, BaseModel):
        return model.model_dump()
    elif isinstance(model, dict):
        return {k: full_model_dump(v) for k, v in model.items()}
    elif isinstance(model, list):
        return [full_model_dump(item) for item in model]
    else:
        return model


class SourceProcessingInput(CommandInput):
    source_id: str
    content_state: Dict[str, Any]
    notebook_ids: List[str]
    transformations: List[str]
    embed: bool
    extract_knowledge: bool = False
    document_type: Optional[str] = None


class SourceProcessingOutput(CommandOutput):
    success: bool
    source_id: str
    embedded_chunks: int = 0
    insights_created: int = 0
    entities_extracted: int = 0
    claims_extracted: int = 0
    raptor_layers: int = 0
    raptor_nodes: int = 0
    processing_time: float
    error_message: Optional[str] = None


@command("process_source", app="open_notebook")
async def process_source_command(
    input_data: SourceProcessingInput,
) -> SourceProcessingOutput:
    """
    Process source content using the source_graph workflow
    """
    start_time = time.time()

    try:
        logger.info(f"Starting source processing for source: {input_data.source_id}")
        logger.info(f"Notebook IDs: {input_data.notebook_ids}")
        logger.info(f"Transformations: {input_data.transformations}")
        logger.info(f"Embed: {input_data.embed}")
        logger.info(f"Extract Knowledge: {input_data.extract_knowledge}")

        # 1. Load transformation objects from IDs
        transformations = []
        for trans_id in input_data.transformations:
            logger.info(f"Loading transformation: {trans_id}")
            transformation = await Transformation.get(trans_id)
            if not transformation:
                raise ValueError(f"Transformation '{trans_id}' not found")
            transformations.append(transformation)

        logger.info(f"Loaded {len(transformations)} transformations")

        # 2. Get existing source record to update its command field
        source = await Source.get(input_data.source_id)
        if not source:
            raise ValueError(f"Source '{input_data.source_id}' not found")

        # Update source with command reference
        source.command = (
            ensure_record_id(input_data.execution_context.command_id)
            if input_data.execution_context
            else None
        )
        await source.save()

        logger.info(f"Updated source {source.id} with command reference")

        # 3. Process source with all notebooks
        logger.info(f"Processing source with {len(input_data.notebook_ids)} notebooks")

        # Check RAPTOR settings
        from open_notebook.domain.content_settings import ContentSettings
        content_settings = await ContentSettings.get_instance()
        raptor_enabled = getattr(content_settings, 'raptor_enabled', False)

        # Extract content state fields for graph
        content_state = input_data.content_state
        file_path = content_state.get("file_path")
        url = content_state.get("url")
        content = content_state.get("content")
        title = content_state.get("title")

        # Execute source_graph with all notebooks and KG/RAPTOR flags
        result = await source_graph.ainvoke(
            {  # type: ignore[arg-type]
                # Content input fields
                "file_path": file_path,
                "url": url,
                "content": content,
                "title": title,
                # Processing configuration
                "notebook_ids": input_data.notebook_ids,
                "apply_transformations": transformations,
                "embed": input_data.embed,
                "source_id": input_data.source_id,
                # Knowledge Graph and RAPTOR flags
                "extract_knowledge": input_data.extract_knowledge,
                "raptor_enabled": raptor_enabled,
            }
        )

        processed_source = result["source"]

        # 4. Gather processing results (notebook associations handled by source_graph)
        embedded_chunks = (
            await processed_source.get_embedded_chunks() if input_data.embed else 0
        )
        insights_list = await processed_source.get_insights()
        insights_created = len(insights_list)

        # 5. Get KG extraction results from graph (extraction now handled in graph)
        entities_extracted = 0
        claims_extracted = 0
        kg_result = result.get("kg_extraction_result")
        if kg_result and not kg_result.get("error"):
            entities_extracted = kg_result.get("entities_extracted", 0)
            claims_extracted = kg_result.get("claims_extracted", 0)
            logger.info(f"Knowledge extraction complete: {entities_extracted} entities, {claims_extracted} claims")
        elif kg_result and kg_result.get("error"):
            logger.warning(f"Knowledge extraction failed: {kg_result.get('error')}")

        # 6. Log RAPTOR results if processed
        raptor_result = result.get("raptor_result")
        if raptor_result and not raptor_result.get("error"):
            raptor_layers = raptor_result.get("layers", 0)
            raptor_nodes = raptor_result.get("nodes_created", 0)
            logger.info(f"RAPTOR processing complete: {raptor_layers} layers, {raptor_nodes} nodes")
        elif raptor_result and raptor_result.get("error"):
            logger.warning(f"RAPTOR processing failed: {raptor_result.get('error')}")

        # 7. Run metadata enrichment for academic papers
        if input_data.document_type == "academic_paper":
            try:
                logger.info(f"Starting metadata enrichment for academic paper: {processed_source.id}")
                from open_notebook.processors.metadata_enrichment import MetadataEnrichmentService

                enrichment_service = MetadataEnrichmentService()
                enrichment_result = await enrichment_service.enrich(
                    title=processed_source.title or "",
                    text=processed_source.full_text or "",
                    existing_metadata=processed_source.type_metadata or {},
                )

                if enrichment_result:
                    processed_source.source_type = enrichment_result.get("document_type", "academic_paper")
                    processed_source.type_metadata = enrichment_result.get("metadata", {})
                    processed_source.external_ids = enrichment_result.get("external_ids", {})
                    await processed_source.save()
                    logger.info(f"Metadata enrichment complete for source: {processed_source.id}")
                else:
                    logger.info(f"No enrichment data found for source: {processed_source.id}")
            except Exception as me:
                logger.warning(f"Metadata enrichment failed (non-fatal): {me}")
                # Don't fail the entire processing if enrichment fails

        processing_time = time.time() - start_time
        logger.info(
            f"Successfully processed source: {processed_source.id} in {processing_time:.2f}s"
        )
        logger.info(
            f"Created {insights_created} insights and {embedded_chunks} embedded chunks"
        )
        if input_data.extract_knowledge:
            logger.info(f"Extracted {entities_extracted} entities and {claims_extracted} claims")

        # Calculate RAPTOR results for output
        raptor_layers_out = 0
        raptor_nodes_out = 0
        if raptor_result and not raptor_result.get("error"):
            raptor_layers_out = raptor_result.get("layers", 0)
            raptor_nodes_out = raptor_result.get("nodes_created", 0)

        return SourceProcessingOutput(
            success=True,
            source_id=str(processed_source.id),
            embedded_chunks=embedded_chunks,
            insights_created=insights_created,
            entities_extracted=entities_extracted,
            claims_extracted=claims_extracted,
            raptor_layers=raptor_layers_out,
            raptor_nodes=raptor_nodes_out,
            processing_time=processing_time,
        )

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Source processing failed: {e}")
        logger.exception(e)

        return SourceProcessingOutput(
            success=False,
            source_id=input_data.source_id,
            processing_time=processing_time,
            error_message=str(e),
        )
