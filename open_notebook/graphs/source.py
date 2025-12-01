import operator
from typing import Any, Dict, List, Optional

from content_core import extract_content
from content_core.common import ProcessSourceInput, ProcessSourceOutput
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger
from typing_extensions import Annotated, TypedDict

from open_notebook.domain.content_settings import ContentSettings
from open_notebook.domain.notebook import Asset, Chunk, Source
from open_notebook.domain.transformation import Transformation
from open_notebook.graphs.transformation import graph as transform_graph
from open_notebook.processors import extract_chunks_from_docling
from open_notebook.utils.content_core_config import apply_content_core_settings, serialize_docling_document


class SourceState(TypedDict):
    content_state: ProcessSourceInput | ProcessSourceOutput  # Input initially, Output after processing
    apply_transformations: List[Transformation]
    source_id: str
    notebook_ids: List[str]
    source: Source
    transformation: Annotated[list, operator.add]
    embed: bool
    chunks: Optional[List[Dict[str, Any]]]  # Extracted document chunks with positions


class TransformationState(TypedDict):
    source: Source
    transformation: Transformation


async def content_process(state: SourceState) -> dict:
    # Load settings from database instead of hardcoding
    content_settings = await ContentSettings.get_instance()
    content_state = state["content_state"]

    # Convert dict to ProcessSourceInput if needed (LangGraph serializes to dict)
    if isinstance(content_state, dict):
        content_input = ProcessSourceInput(**content_state)
    else:
        content_input = content_state

    # Apply content-core configuration from settings (GPU, pipeline, VLM, etc.)
    await apply_content_core_settings(content_settings)

    # Configure extraction settings
    content_input.url_engine = (
        content_settings.default_content_processing_engine_url or "auto"
    )
    content_input.document_engine = (
        content_settings.default_content_processing_engine_doc or "docling"
    )
    content_input.output_format = "markdown"

    # Log the configuration being used
    logger.info(
        f"📄 Processing with engine={content_input.document_engine}, "
        f"pipeline={content_settings.docling_pipeline}, "
        f"gpu={content_settings.docling_gpu_enabled}"
    )

    # Use content-core for all extraction (GPU/VLM configured via apply_content_core_settings)
    processed_state = await extract_content(content_input)

    # Extract chunks BEFORE serializing DoclingDocument (need original object for chunking)
    chunks = None
    extraction_engine = (processed_state.metadata or {}).get("extraction_engine", "")
    file_path = processed_state.file_path
    is_pdf = file_path and file_path.lower().endswith('.pdf')
    has_docling_doc = processed_state.metadata and processed_state.metadata.get("docling_document")

    # Use content-core's chunking if enabled and we have a docling document
    if content_settings.docling_chunking_enabled and has_docling_doc:
        try:
            from content_core.content.chunking import chunk_content

            logger.info(
                f"📦 Chunking with content-core: method={content_settings.docling_chunking_method}, "
                f"max_tokens={content_settings.docling_chunking_max_tokens}, merge_peers=True"
            )

            # Use content-core's HybridChunker with merge_peers for proper chunk merging
            chunk_outputs = await chunk_content(
                content=processed_state,
                chunk_size=content_settings.docling_chunking_max_tokens,
                method=content_settings.docling_chunking_method or "hybrid",
                merge_peers=True,  # Merge small consecutive chunks
                preserve_metadata=True,
                include_bboxes=True,
            )

            # Convert ChunkOutput to open-notebook's chunk format
            chunks = []
            for chunk_out in chunk_outputs:
                # Extract page numbers from metadata
                page_numbers = chunk_out.metadata.page_numbers or []
                physical_page = page_numbers[0] if page_numbers else 0

                # Convert bounding boxes to normalized positions format
                # Docling uses BOTTOMLEFT origin with absolute coordinates in points
                # Frontend expects TOPLEFT origin with normalized (0-1) coordinates
                # Format: [page_number, x_left, x_right, y_top, y_bottom] normalized 0-1
                positions = []

                # Helper to normalize and convert coordinates
                def normalize_bbox(bbox_dict: dict, page_no: int, chunk_idx: int = 0) -> list:
                    """Convert absolute BOTTOMLEFT coords to normalized TOPLEFT coords."""
                    left = float(bbox_dict['left'])
                    right = float(bbox_dict['right'])
                    top = float(bbox_dict['top'])
                    bottom = float(bbox_dict['bottom'])
                    page_width = bbox_dict.get('page_width')
                    page_height = bbox_dict.get('page_height')
                    coord_origin = str(bbox_dict.get('coord_origin', 'BOTTOMLEFT')).upper()

                    # Default page size if not provided (A4 in points)
                    if not page_width or page_width <= 0:
                        page_width = 595.0  # A4 width
                    if not page_height or page_height <= 0:
                        page_height = 842.0  # A4 height

                    # Normalize to 0-1 range
                    x_left = left / page_width
                    x_right = right / page_width

                    # Convert Y coordinates based on origin
                    # Content-core's bbox uses 'top' and 'bottom' which are y-coordinates
                    # In BOTTOMLEFT: higher y = higher on page visually
                    # 'top' edge of bbox has higher y, 'bottom' edge has lower y
                    if 'BOTTOMLEFT' in coord_origin:
                        # In BOTTOMLEFT: y=0 is bottom, y increases upward
                        # Convert to TOPLEFT: y=0 is top, y increases downward
                        # Visual top of box (high y in BOTTOMLEFT) -> small y in TOPLEFT
                        y_top_converted = 1.0 - (top / page_height)
                        y_bottom_converted = 1.0 - (bottom / page_height)
                    else:
                        # Already TOPLEFT origin
                        y_top_converted = top / page_height
                        y_bottom_converted = bottom / page_height

                    # ALWAYS ensure y_top < y_bottom for TOPLEFT display
                    y_top = min(y_top_converted, y_bottom_converted)
                    y_bottom = max(y_top_converted, y_bottom_converted)

                    # Clamp values to 0-1 range
                    x_left = max(0.0, min(1.0, x_left))
                    x_right = max(0.0, min(1.0, x_right))
                    y_top = max(0.0, min(1.0, y_top))
                    y_bottom = max(0.0, min(1.0, y_bottom))

                    # Debug: log coordinate conversion for first few chunks
                    if chunk_idx < 5 or page_no == 22:
                        logger.info(
                            f"normalize_bbox chunk={chunk_idx}: page={page_no}, origin={coord_origin}, "
                            f"raw(l={left:.1f}, r={right:.1f}, t={top:.1f}, b={bottom:.1f}), "
                            f"pageSize=({page_width:.1f}x{page_height:.1f}), "
                            f"→ y_top={y_top:.3f} ({y_top*100:.1f}%), y_bottom={y_bottom:.3f} ({y_bottom*100:.1f}%)"
                        )

                    return [page_no, x_left, x_right, y_top, y_bottom]

                # doc_items contain page-specific bounding box information
                chunk_idx = chunk_out.index
                for doc_item in chunk_out.metadata.doc_items:
                    if 'bbox' in doc_item:
                        bbox = doc_item['bbox']
                        page_no = doc_item.get('page_no', page_numbers[0] if page_numbers else 0)
                        positions.append(normalize_bbox(bbox, page_no, chunk_idx))

                # Fallback: If doc_items don't have bbox but bounding_boxes does
                if not positions and chunk_out.metadata.bounding_boxes:
                    for i, bbox in enumerate(chunk_out.metadata.bounding_boxes):
                        page_no = page_numbers[i] if i < len(page_numbers) else (page_numbers[0] if page_numbers else 0)
                        bbox_dict = {
                            'left': bbox.left,
                            'right': bbox.right,
                            'top': bbox.top,
                            'bottom': bbox.bottom,
                            'page_width': bbox.page_width,
                            'page_height': bbox.page_height,
                            'coord_origin': bbox.coord_origin,
                        }
                        positions.append(normalize_bbox(bbox_dict, page_no, chunk_idx))

                chunk = {
                    'text': chunk_out.text,
                    'order': chunk_out.index,
                    'physical_page': physical_page,
                    'printed_page': physical_page + 1,
                    'chapter': chunk_out.metadata.headings[0] if chunk_out.metadata.headings else None,
                    'paragraph_number': None,
                    'element_type': 'chunk',  # Merged chunk
                    'positions': positions,
                    'metadata': {
                        'has_spatial_data': len(positions) > 0,
                        'num_locations': len(positions),
                        'headings': chunk_out.metadata.headings,
                        'page_numbers': page_numbers,
                    }
                }
                chunks.append(chunk)

            logger.info(f"✅ Extracted {len(chunks)} merged chunks (content-core HybridChunker)")

        except ImportError as e:
            logger.warning(f"Content-core chunking not available: {e}. Falling back to element-based extraction.")
            chunks = None
        except Exception as e:
            logger.warning(f"Content-core chunking failed: {e}. Falling back to element-based extraction.")
            chunks = None

    # Fallback: Extract raw document elements if chunking disabled or failed
    if chunks is None and ("docling" in extraction_engine.lower() or is_pdf) and file_path:
        try:
            logger.info(f"Extracting document elements (no merging) from {file_path}")

            from open_notebook.utils.docling_utils import reconstruct_docling_document
            from open_notebook.processors.chunk_extractor import (
                extract_chunks_from_existing_document,
                extract_chunks_from_docling
            )

            # Try to get DoclingDocument from metadata before serialization
            doc = processed_state.metadata.get("docling_document") if processed_state.metadata else None
            if doc:
                logger.debug("Using DoclingDocument from metadata for element extraction")
                chunks = extract_chunks_from_existing_document(doc)
            else:
                # Fallback: Re-process with docling
                logger.debug("Re-converting document for element extraction")
                _, chunks, _ = extract_chunks_from_docling(file_path, output_format="markdown")

            logger.info(f"Extracted {len(chunks)} document elements (unmerged)")
        except Exception as e:
            logger.warning(f"Failed to extract document elements: {e}")
            chunks = None

    # Serialize DoclingDocument for LangGraph state persistence (AFTER chunking)
    if processed_state.metadata:
        processed_state.metadata = serialize_docling_document(processed_state.metadata)

    # File Management System Integration
    # Organize files and export markdown with assets if file_path exists
    if file_path:
        try:
            from open_notebook.utils.file_manager import organize_file

            # Get file management settings
            file_operation = content_settings.file_operation or "copy"
            naming_scheme = content_settings.output_naming_scheme or "date_prefix"
            input_dir = content_settings.input_directory_path or "./data/input"
            markdown_dir = content_settings.markdown_directory_path or "./data/markdown"
            output_dir = content_settings.output_directory_path or "./data/output"

            logger.info(f"📁 Starting file management workflow for {file_path}")

            # Step 1: Organize file (copy/move to INPUT, copy to OUTPUT)
            input_path, output_path = organize_file(
                file_path=file_path,
                input_dir=input_dir,
                output_dir=output_dir,
                file_operation=file_operation,
                naming_scheme=naming_scheme,
            )

            # Store organized paths in metadata
            if processed_state.metadata is None:
                processed_state.metadata = {}
            processed_state.metadata["organized_input_path"] = str(input_path) if input_path else None
            processed_state.metadata["organized_output_path"] = str(output_path)

            # NOTE: Markdown export with assets requires reconstructing DoclingDocument from JSON
            # Content-core stores the document in metadata, which we serialize for LangGraph
            markdown_export_path = processed_state.metadata.get("markdown_export_path")

            if not markdown_export_path:
                # Fallback: Try to reconstruct DoclingDocument and export
                from open_notebook.utils.docling_utils import reconstruct_docling_document
                from open_notebook.utils.file_manager import (
                    create_document_subdirectory,
                    generate_unique_document_name,
                )
                from open_notebook.utils.markdown_exporter import (
                    export_document_to_markdown_with_assets,
                )

                doc = reconstruct_docling_document(processed_state.metadata)
                if doc:
                    try:
                        from pathlib import Path
                        doc_base_name = Path(file_path).stem
                        doc_name = generate_unique_document_name(doc_base_name, timestamp=True)
                        doc_dir = create_document_subdirectory(markdown_dir, doc_name)

                        markdown_path, num_images, num_tables = export_document_to_markdown_with_assets(
                            doc=doc,
                            output_dir=str(doc_dir),
                            document_name="document",
                        )

                        processed_state.metadata["markdown_export_path"] = markdown_path
                        processed_state.metadata["markdown_images_count"] = num_images
                        processed_state.metadata["markdown_tables_count"] = num_tables

                        logger.success(
                            f"📄 Exported markdown with {num_images} images and {num_tables} tables to {markdown_path}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to export markdown in fallback: {e}")

            logger.info(f"✅ File management workflow completed successfully")

        except Exception as e:
            logger.error(f"File management workflow failed: {e}")
            # Don't fail the entire processing if file management fails
            # Continue with normal flow

    return {
        "content_state": processed_state,
        "chunks": chunks
    }


async def save_source(state: SourceState) -> dict:
    from open_notebook.database.repository import repo_query, ensure_record_id

    content_state = state["content_state"]

    # Get existing source using the provided source_id
    source = await Source.get(state["source_id"])
    if not source:
        raise ValueError(f"Source with ID {state['source_id']} not found")

    # content_state is a ProcessSourceOutput Pydantic object from content_process
    source.asset = Asset(
        url=content_state.url,
        file_path=content_state.file_path
    )
    source.full_text = content_state.content

    # Preserve existing title if none provided in processed content
    if content_state.title:
        source.title = content_state.title

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
                    order=chunk_data["order"],
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

    # NOTE: Notebook associations are created by the API immediately for UI responsiveness
    # No need to create them here to avoid duplicate edges

    if state["embed"]:
        logger.debug("Embedding content for vector search")
        await source.vectorize()

    return {"source": source}


def trigger_transformations(state: SourceState, config: RunnableConfig) -> List[Send]:
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
    source = state["source"]
    content = source.full_text
    if not content:
        return None
    transformation: Transformation = state["transformation"]

    logger.debug(f"Applying transformation {transformation.name}")
    result = await transform_graph.ainvoke(
        dict(input_text=content, transformation=transformation)  # type: ignore[arg-type]
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


# Create and compile the workflow
workflow = StateGraph(SourceState)

# Add nodes
workflow.add_node("content_process", content_process)
workflow.add_node("save_source", save_source)
workflow.add_node("transform_content", transform_content)
# Define the graph edges
workflow.add_edge(START, "content_process")
workflow.add_edge("content_process", "save_source")
workflow.add_conditional_edges(
    "save_source", trigger_transformations, ["transform_content"]
)
workflow.add_edge("transform_content", END)

# Compile the graph
source_graph = workflow.compile()
