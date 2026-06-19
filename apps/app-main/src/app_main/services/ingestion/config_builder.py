"""
Bridge ContentSettings (DB-backed) → IngestionConfig (pipeline-runtime).

Was ``SourceProcessingService._build_ingestion_config`` before Phase 2.
Pulled out as a pure function so it can be tested without instantiating
the service.
"""

from __future__ import annotations

from shared.models.settings import ContentSettings


def build_ingestion_config(content_settings: ContentSettings):
    """Translate persisted ContentSettings into a runtime IngestionConfig."""
    from ingestion.config import (
        AcceleratorDevice,
        DoclingConfig,
        IngestionConfig,
        OcrEngine,
        TableMode,
    )

    # Map GPU device
    device_map = {
        "auto": AcceleratorDevice.AUTO,
        "cuda": AcceleratorDevice.CUDA,
        "cpu": AcceleratorDevice.CPU,
    }
    device = device_map.get(
        content_settings.docling_gpu_device or "auto",
        AcceleratorDevice.AUTO,
    )
    gpu_enabled = content_settings.docling_gpu_enabled
    if gpu_enabled is not None and not gpu_enabled:
        device = AcceleratorDevice.CPU

    # Map OCR engine
    ocr_map = {
        "auto": OcrEngine.EASYOCR,
        "easyocr": OcrEngine.EASYOCR,
        "rapidocr": OcrEngine.RAPIDOCR,
        "tesseract": OcrEngine.TESSERACT,
    }
    ocr_engine = ocr_map.get(
        content_settings.docling_ocr_engine or "easyocr",
        OcrEngine.EASYOCR,
    )

    # Map table mode
    table_mode_map = {
        "accurate": TableMode.ACCURATE,
        "fast": TableMode.FAST,
    }
    table_mode = table_mode_map.get(
        content_settings.docling_table_mode or "accurate",
        TableMode.ACCURATE,
    )

    # Map VLM model short names to HuggingFace repo IDs
    vlm_model_map = {
        "granite-docling-258m": "ibm-granite/granite-docling-258M",
        "smoldocling-256m": "ds4sd/SmolDocling-256M-preview",
    }
    vlm_model = vlm_model_map.get(
        content_settings.docling_vlm_model or "granite-docling-258m",
        "ibm-granite/granite-docling-258M",
    )

    # Pipeline selection: "vlm" enables VLM features, "standard" disables them
    pipeline = content_settings.docling_pipeline or "vlm"
    use_vlm = pipeline == "vlm"

    # Picture classification is independently controllable (Phase I.D-2).
    # When the setting is None it falls back to the VLM toggle, preserving the
    # pre-I.D-2 coupling so existing users see no behaviour change.
    do_picture_classification = getattr(
        content_settings, "docling_do_picture_classification", None
    )
    if do_picture_classification is None:
        do_picture_classification = use_vlm

    # Enrichment toggles (Phase I.D-2). getattr with the DoclingConfig default
    # keeps callers that build ContentSettings without these fields working.
    do_code_enrichment = getattr(
        content_settings, "docling_do_code_enrichment", None
    )
    do_formula_enrichment = getattr(
        content_settings, "docling_do_formula_enrichment", None
    )
    generate_page_images = getattr(
        content_settings, "docling_generate_page_images", None
    )

    docling_cfg = DoclingConfig(
        device=device,
        do_ocr=True,
        ocr_engine=ocr_engine,
        ocr_lang=content_settings.docling_ocr_languages or ["en"],
        table_mode=table_mode,
        images_scale=content_settings.docling_image_scale or 2.0,
        generate_picture_images=bool(
            content_settings.docling_auto_export_images
        ),
        do_picture_description=use_vlm,
        do_picture_classification=do_picture_classification,
        vlm_model=vlm_model,
    )

    if do_code_enrichment is not None:
        docling_cfg.do_code_enrichment = do_code_enrichment
    if do_formula_enrichment is not None:
        docling_cfg.do_formula_extraction = do_formula_enrichment
    if generate_page_images is not None:
        docling_cfg.generate_page_images = generate_page_images

    return IngestionConfig(docling=docling_cfg)
