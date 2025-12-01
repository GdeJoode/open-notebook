"""
Bridge between open-notebook settings and content-core configuration.

This module applies open-notebook's ContentSettings to content-core's
runtime configuration before document processing.
"""
from loguru import logger

from content_core.config import (
    set_docling_gpu_enabled,
    set_docling_gpu_device,
    set_docling_pipeline,
    set_docling_vlm_model,
    set_docling_vlm_framework,
    set_docling_table_structure_mode,
    set_docling_image_scale,
    set_docling_auto_export_images,
    set_docling_chunking_enabled,
    set_docling_chunking_config,
    get_docling_ocr_engine,
)


async def apply_content_core_settings(content_settings) -> None:
    """
    Apply open-notebook ContentSettings to content-core configuration.

    This should be called before any content-core extraction operations.
    It configures content-core's runtime settings based on user preferences
    stored in open-notebook's database.

    Args:
        content_settings: ContentSettings instance from open-notebook
    """
    logger.debug("Applying content-core configuration from settings")

    # GPU Settings
    if content_settings.docling_gpu_enabled is not None:
        set_docling_gpu_enabled(content_settings.docling_gpu_enabled)
        if content_settings.docling_gpu_enabled:
            logger.info("🚀 GPU acceleration enabled for content-core")

    if content_settings.docling_gpu_device:
        set_docling_gpu_device(content_settings.docling_gpu_device)

    # Pipeline Settings
    if content_settings.docling_pipeline:
        set_docling_pipeline(content_settings.docling_pipeline)
        logger.info(f"📄 Using {content_settings.docling_pipeline} pipeline")

    # VLM Settings (only relevant when pipeline=vlm)
    if content_settings.docling_vlm_model:
        set_docling_vlm_model(content_settings.docling_vlm_model)

    if content_settings.docling_vlm_framework:
        set_docling_vlm_framework(content_settings.docling_vlm_framework)

    # OCR Settings are configured via environment variables or cc_config.yaml
    # Log the current OCR engine for visibility
    current_ocr = get_docling_ocr_engine()
    logger.debug(f"OCR engine: {current_ocr}")

    # Table Settings
    if content_settings.docling_table_mode:
        set_docling_table_structure_mode(content_settings.docling_table_mode)

    # Image Settings (not yet functional, but configure anyway)
    if content_settings.docling_auto_export_images is not None:
        set_docling_auto_export_images(content_settings.docling_auto_export_images)

    if content_settings.docling_image_scale:
        set_docling_image_scale(content_settings.docling_image_scale)

    # Chunking Settings
    if content_settings.docling_chunking_enabled is not None:
        set_docling_chunking_enabled(content_settings.docling_chunking_enabled)

    if content_settings.docling_chunking_method or content_settings.docling_chunking_max_tokens:
        set_docling_chunking_config(
            method=content_settings.docling_chunking_method,
            max_tokens=content_settings.docling_chunking_max_tokens,
        )

    logger.debug("✅ Content-core configuration applied")


def serialize_docling_document(metadata: dict) -> dict:
    """
    Serialize DoclingDocument in metadata for LangGraph state persistence.

    Content-core stores the DoclingDocument object in metadata["docling_document"].
    This object is not JSON-serializable, so we need to convert it to JSON
    and store it separately for LangGraph state persistence.

    Args:
        metadata: The metadata dict from ProcessSourceOutput

    Returns:
        Updated metadata dict with serialized document
    """
    if not metadata:
        return metadata

    doc = metadata.get("docling_document")
    if doc is None:
        return metadata

    try:
        # Serialize to JSON (use exclude_none=True to reduce size)
        metadata["docling_document_json"] = doc.model_dump_json(exclude_none=True)
        # Remove the non-serializable object
        del metadata["docling_document"]
        logger.debug("✅ Serialized DoclingDocument to JSON for state persistence")
    except Exception as e:
        logger.warning(f"Failed to serialize DoclingDocument: {e}")
        # Remove the non-serializable object anyway to prevent LangGraph errors
        if "docling_document" in metadata:
            del metadata["docling_document"]

    return metadata
