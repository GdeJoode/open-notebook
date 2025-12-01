from fastapi import APIRouter, HTTPException
from loguru import logger

from api.models import SettingsResponse, SettingsUpdate
from open_notebook.domain.content_settings import ContentSettings
from open_notebook.exceptions import InvalidInputError

router = APIRouter()


def _build_settings_response(settings: ContentSettings) -> SettingsResponse:
    """Build a SettingsResponse from ContentSettings."""
    return SettingsResponse(
        # Basic Settings
        default_content_processing_engine_doc=settings.default_content_processing_engine_doc,
        default_content_processing_engine_url=settings.default_content_processing_engine_url,
        default_embedding_option=settings.default_embedding_option,
        auto_delete_files=settings.auto_delete_files,
        youtube_preferred_languages=settings.youtube_preferred_languages,

        # GPU Acceleration Settings
        docling_gpu_enabled=settings.docling_gpu_enabled,
        docling_gpu_device=settings.docling_gpu_device,

        # Pipeline Settings
        docling_pipeline=settings.docling_pipeline,

        # VLM Settings
        docling_vlm_model=settings.docling_vlm_model,
        docling_vlm_framework=settings.docling_vlm_framework,

        # OCR Settings
        docling_ocr_engine=settings.docling_ocr_engine,
        docling_ocr_languages=settings.docling_ocr_languages,
        docling_ocr_use_gpu=settings.docling_ocr_use_gpu,

        # Table Processing Settings
        docling_table_mode=settings.docling_table_mode,

        # Image Export Settings
        docling_auto_export_images=settings.docling_auto_export_images,
        docling_image_scale=settings.docling_image_scale,

        # Chunking Settings
        docling_chunking_enabled=settings.docling_chunking_enabled,
        docling_chunking_method=settings.docling_chunking_method,
        docling_chunking_max_tokens=settings.docling_chunking_max_tokens,

        # File Management Settings
        input_directory_path=settings.input_directory_path,
        markdown_directory_path=settings.markdown_directory_path,
        output_directory_path=settings.output_directory_path,
        file_operation=settings.file_operation,
        output_naming_scheme=settings.output_naming_scheme,
    )


@router.get("/settings", response_model=SettingsResponse)
async def get_settings():
    """Get all application settings."""
    try:
        settings: ContentSettings = await ContentSettings.get_instance()  # type: ignore[assignment]
        return _build_settings_response(settings)
    except Exception as e:
        logger.error(f"Error fetching settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching settings: {str(e)}")


@router.put("/settings", response_model=SettingsResponse)
async def update_settings(settings_update: SettingsUpdate):
    """Update application settings."""
    try:
        settings: ContentSettings = await ContentSettings.get_instance()  # type: ignore[assignment]

        # Update only provided fields - Basic Settings
        if settings_update.default_content_processing_engine_doc is not None:
            from typing import Literal, cast
            settings.default_content_processing_engine_doc = cast(
                Literal["auto", "docling", "simple"],
                settings_update.default_content_processing_engine_doc
            )
        if settings_update.default_content_processing_engine_url is not None:
            from typing import Literal, cast
            settings.default_content_processing_engine_url = cast(
                Literal["auto", "firecrawl", "jina", "simple"],
                settings_update.default_content_processing_engine_url
            )
        if settings_update.default_embedding_option is not None:
            from typing import Literal, cast
            settings.default_embedding_option = cast(
                Literal["ask", "always", "never"],
                settings_update.default_embedding_option
            )
        if settings_update.auto_delete_files is not None:
            from typing import Literal, cast
            settings.auto_delete_files = cast(
                Literal["yes", "no"],
                settings_update.auto_delete_files
            )
        if settings_update.youtube_preferred_languages is not None:
            settings.youtube_preferred_languages = settings_update.youtube_preferred_languages

        # GPU Acceleration Settings
        if settings_update.docling_gpu_enabled is not None:
            settings.docling_gpu_enabled = settings_update.docling_gpu_enabled
        if settings_update.docling_gpu_device is not None:
            from typing import Literal, cast
            settings.docling_gpu_device = cast(
                Literal["auto", "cuda", "cpu"],
                settings_update.docling_gpu_device
            )

        # Pipeline Settings
        if settings_update.docling_pipeline is not None:
            from typing import Literal, cast
            settings.docling_pipeline = cast(
                Literal["auto", "standard", "vlm"],
                settings_update.docling_pipeline
            )

        # VLM Settings
        if settings_update.docling_vlm_model is not None:
            from typing import Literal, cast
            settings.docling_vlm_model = cast(
                Literal["granite-docling-258m", "smoldocling-256m"],
                settings_update.docling_vlm_model
            )
        if settings_update.docling_vlm_framework is not None:
            from typing import Literal, cast
            settings.docling_vlm_framework = cast(
                Literal["auto", "transformers", "mlx"],
                settings_update.docling_vlm_framework
            )

        # OCR Settings
        if settings_update.docling_ocr_engine is not None:
            from typing import Literal, cast
            settings.docling_ocr_engine = cast(
                Literal["auto", "easyocr", "rapidocr", "tesseract"],
                settings_update.docling_ocr_engine
            )
        if settings_update.docling_ocr_languages is not None:
            settings.docling_ocr_languages = settings_update.docling_ocr_languages
        if settings_update.docling_ocr_use_gpu is not None:
            settings.docling_ocr_use_gpu = settings_update.docling_ocr_use_gpu

        # Table Processing Settings
        if settings_update.docling_table_mode is not None:
            from typing import Literal, cast
            settings.docling_table_mode = cast(
                Literal["accurate", "fast"],
                settings_update.docling_table_mode
            )

        # Image Export Settings
        if settings_update.docling_auto_export_images is not None:
            settings.docling_auto_export_images = settings_update.docling_auto_export_images
        if settings_update.docling_image_scale is not None:
            settings.docling_image_scale = settings_update.docling_image_scale

        # Chunking Settings
        if settings_update.docling_chunking_enabled is not None:
            settings.docling_chunking_enabled = settings_update.docling_chunking_enabled
        if settings_update.docling_chunking_method is not None:
            from typing import Literal, cast
            settings.docling_chunking_method = cast(
                Literal["hybrid", "hierarchical"],
                settings_update.docling_chunking_method
            )
        if settings_update.docling_chunking_max_tokens is not None:
            settings.docling_chunking_max_tokens = settings_update.docling_chunking_max_tokens

        # File Management Settings
        if settings_update.input_directory_path is not None:
            settings.input_directory_path = settings_update.input_directory_path
        if settings_update.markdown_directory_path is not None:
            settings.markdown_directory_path = settings_update.markdown_directory_path
        if settings_update.output_directory_path is not None:
            settings.output_directory_path = settings_update.output_directory_path
        if settings_update.file_operation is not None:
            from typing import Literal, cast
            settings.file_operation = cast(
                Literal["copy", "move", "none"],
                settings_update.file_operation
            )
        if settings_update.output_naming_scheme is not None:
            from typing import Literal, cast
            settings.output_naming_scheme = cast(
                Literal["timestamp_prefix", "date_prefix", "datetime_suffix", "original"],
                settings_update.output_naming_scheme
            )

        await settings.update()
        return _build_settings_response(settings)

    except HTTPException:
        raise
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating settings: {str(e)}")
