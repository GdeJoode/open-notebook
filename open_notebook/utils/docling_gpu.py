"""
GPU-accelerated Docling document processor for Open Notebook.

This module provides GPU-accelerated document processing using Docling with CUDA support.
Automatically detects and uses available GPU acceleration for faster PDF and document parsing.
"""

from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice,
)
from docling.document_converter import DocumentConverter
from loguru import logger


def create_gpu_document_converter() -> DocumentConverter:
    """
    Create a DocumentConverter configured for optimal GPU performance.

    Returns:
        DocumentConverter: Configured converter with GPU acceleration enabled
    """
    try:
        # Configure accelerator for CUDA GPU
        accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.CUDA
        )

        # Configure PDF pipeline with accelerator options
        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options

        # Create converter with GPU-enabled pipeline
        converter = DocumentConverter(
            pipeline_options=pipeline_options
        )

        logger.info("✅ Docling DocumentConverter initialized with CUDA GPU acceleration")
        logger.info(f"   Device: {AcceleratorDevice.CUDA}")
        logger.info(f"   Threads: 8")

        return converter

    except Exception as e:
        logger.warning(f"Failed to initialize GPU-accelerated Docling: {e}")
        logger.info("Falling back to AUTO device selection (CPU/GPU auto-detect)")

        # Fallback to AUTO device (will use GPU if available, CPU otherwise)
        accelerator_options = AcceleratorOptions(
            num_threads=8,
            device=AcceleratorDevice.AUTO
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.accelerator_options = accelerator_options

        converter = DocumentConverter(
            pipeline_options=pipeline_options
        )

        logger.info("✅ Docling DocumentConverter initialized with AUTO device selection")

        return converter
