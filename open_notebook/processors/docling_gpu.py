"""
GPU-accelerated Docling processor for Open Notebook.

This module provides a GPU-accelerated document processing engine that can be selected
as an alternative to the standard 'docling' engine.
"""

from content_core.common.state import ProcessSourceState
from content_core.config import CONFIG
from loguru import logger

from open_notebook.utils.docling_gpu import create_gpu_document_converter

# Create a single global instance of the GPU-enabled converter
# This avoids reinitializing the model for each document
_GPU_CONVERTER = None


def get_gpu_converter():
    """Get or create the global GPU-enabled DocumentConverter instance."""
    global _GPU_CONVERTER
    if _GPU_CONVERTER is None:
        logger.info("Initializing GPU-accelerated Docling converter...")
        _GPU_CONVERTER = create_gpu_document_converter()
    return _GPU_CONVERTER


async def extract_with_docling_gpu(state: ProcessSourceState) -> ProcessSourceState:
    """
    Use GPU-accelerated Docling to parse files, URLs, or content into the desired format.

    This processor provides GPU acceleration for document parsing, offering significant
    speedups compared to CPU-only processing, especially for large documents and PDFs.

    Args:
        state: ProcessSourceState containing the document to process

    Returns:
        ProcessSourceState with extracted content

    Raises:
        ValueError: If no input source is provided
    """
    # Get the GPU-enabled converter
    converter = get_gpu_converter()

    # Determine source: file path, URL, or direct content
    source = state.file_path or state.url or state.content
    if not source:
        raise ValueError("No input provided for Docling GPU extraction.")

    logger.info(f"🚀 Processing document with GPU-accelerated Docling: {source[:100]}...")

    # Convert document
    result = converter.convert(source)
    doc = result.document

    # Determine output format (per execution override, metadata, then config)
    cfg_fmt = (
        CONFIG.get("extraction", {}).get("docling", {}).get("output_format", "markdown")
    )
    fmt = state.output_format or state.metadata.get("docling_format") or cfg_fmt

    # Record the format used and GPU acceleration status
    state.metadata["docling_format"] = fmt
    state.metadata["docling_gpu_enabled"] = True
    state.metadata["extraction_engine"] = "docling_gpu"

    if fmt == "html":
        output = doc.export_to_html()
    elif fmt == "json":
        output = doc.export_to_json()
    else:
        output = doc.export_to_markdown()

    logger.success(f"✅ Document processed with GPU acceleration. Output: {len(output)} chars")

    # Update state
    state.content = output
    return state
