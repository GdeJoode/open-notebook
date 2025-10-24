import operator
from typing import Any, Dict, List, Optional

from content_core import extract_content
from content_core.common import ProcessSourceState
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


class SourceState(TypedDict):
    content_state: ProcessSourceState
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
    content_settings = ContentSettings(
        default_content_processing_engine_doc="auto",
        default_content_processing_engine_url="auto",
        default_embedding_option="ask",
        auto_delete_files="yes",
        youtube_preferred_languages=["en", "pt", "es", "de", "nl", "en-GB", "fr", "hi", "ja"]
    )
    content_state: Dict[str, Any] = state["content_state"]  # type: ignore[assignment]

    content_state["url_engine"] = (
        content_settings.default_content_processing_engine_url or "auto"
    )
    content_state["document_engine"] = (
        content_settings.default_content_processing_engine_doc or "auto"
    )
    content_state["output_format"] = "markdown"

    processed_state = await extract_content(content_state)

    # Extract chunks with bounding boxes if using docling and processing a file
    chunks = None
    extraction_engine = processed_state.metadata.get("extraction_engine", "")

    # Check if docling was used and we have a file path
    if "docling" in extraction_engine.lower() and processed_state.file_path:
        try:
            logger.info(f"Extracting chunks with spatial data from {processed_state.file_path}")
            # Note: We already have the content from extract_content, so we just need chunks
            # Re-process with docling to get chunks (slightly inefficient but clean separation)
            _, chunks, _ = extract_chunks_from_docling(
                processed_state.file_path,
                output_format="markdown"
            )
            logger.info(f"Extracted {len(chunks)} chunks with bounding boxes")
        except Exception as e:
            logger.warning(f"Failed to extract chunks from document: {e}")
            chunks = None

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

    # Update the source with processed content
    source.asset = Asset(url=content_state.url, file_path=content_state.file_path)
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
