"""
VLM-based handwriting reader.

Takes handwriting regions detected by handwriting_detector and sends them
to a vision-language model for text recognition. Supports local models via
Ollama and cloud APIs (OpenAI, Anthropic) via override.

Results are injected back into the docling document as new elements or
attached to existing annotations.
"""

import base64
import os
from typing import Optional

import httpx
from loguru import logger

from handwriting_detector import HandwritingRegion
from ingestion.models.document import (
    AnnotationInfo,
    BoundingBox,
    ElementType,
    ExtractedDocument,
    ExtractedElement,
    PageContent,
)

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

VLM_PROVIDER = os.getenv("VLM_PROVIDER", "ollama")
VLM_MODEL = os.getenv("VLM_MODEL", "granite3.2-vision")
VLM_BASE_URL = os.getenv("VLM_BASE_URL", "http://host.docker.internal:11434")
VLM_TIMEOUT = int(os.getenv("VLM_TIMEOUT", "120"))

# Prompts
FULL_PAGE_PROMPT = (
    "Read all handwritten text on this page. Transcribe every word exactly "
    "as written, preserving the original language. If there are drawings or "
    "diagrams with labels, include those labels. Maintain the reading order "
    "from top to bottom, left to right. Output only the transcribed text."
)

REGION_PROMPT = (
    "Read the handwritten text in this image. Transcribe every word exactly "
    "as written, preserving the original language. Output only the transcribed "
    "text, nothing else."
)

ANNOTATION_PROMPT = (
    "This is a handwritten annotation or note in the margin of a document. "
    "Read and transcribe the handwritten text exactly as written, preserving "
    "the original language. Output only the transcribed text."
)


# ---------------------------------------------------------------------------
# VLM call
# ---------------------------------------------------------------------------


def _call_vlm_ollama(
    image_bytes: bytes,
    prompt: str,
    model: str = "",
    base_url: str = "",
) -> str:
    """Call Ollama vision model with an image."""
    model = model or VLM_MODEL
    base_url = base_url or VLM_BASE_URL

    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = httpx.post(
        f"{base_url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "images": [b64_image],
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 4096},
        },
        timeout=VLM_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("response", "").strip()


def _call_vlm_openai(
    image_bytes: bytes,
    prompt: str,
    model: str = "gpt-4o",
    base_url: str = "https://api.openai.com/v1",
    api_key: str = "",
) -> str:
    """Call OpenAI-compatible vision model with an image."""
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{b64_image}",
                    }},
                ],
            }],
            "max_tokens": 2000,
            "temperature": 0.1,
        },
        timeout=VLM_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def call_vlm(
    image_bytes: bytes,
    prompt: str,
    provider: str = "",
    model: str = "",
    base_url: str = "",
    api_key: str = "",
) -> str:
    """Call a VLM to read text from an image.

    Dispatches to the appropriate provider based on configuration.
    """
    provider = provider or VLM_PROVIDER

    if provider == "ollama":
        return _call_vlm_ollama(image_bytes, prompt, model, base_url)
    elif provider in ("openai", "anthropic"):
        return _call_vlm_openai(
            image_bytes, prompt, model, base_url,
            api_key or os.getenv("VLM_API_KEY", ""),
        )
    else:
        return _call_vlm_ollama(image_bytes, prompt, model, base_url)


# ---------------------------------------------------------------------------
# Process handwriting regions
# ---------------------------------------------------------------------------


def read_handwriting_regions(
    regions: list[HandwritingRegion],
    document: ExtractedDocument,
    provider: str = "",
    model: str = "",
    base_url: str = "",
    api_key: str = "",
) -> ExtractedDocument:
    """Process handwriting regions through VLM and inject results into the document.

    For full-page handwriting: creates new elements with source="vlm_handwriting".
    For ink annotations: attaches handwritten_text to the annotation.
    For empty chunks: replaces the empty content with VLM-recognized text.

    Args:
        regions: Handwriting regions from detect_handwriting().
        document: The docling document to enrich.
        provider/model/base_url/api_key: Optional VLM endpoint overrides.

    Returns:
        The enriched document with handwritten text filled in.
    """
    if not regions:
        return document

    success_count = 0
    fail_count = 0

    for region in regions:
        if not region.image_bytes:
            logger.warning(f"Skipping region on page {region.page}: no image data")
            fail_count += 1
            continue

        # Select prompt based on region type
        if region.region_type == "full_page":
            prompt = FULL_PAGE_PROMPT
        elif region.region_type == "ink_annotation":
            prompt = ANNOTATION_PROMPT
        else:
            prompt = REGION_PROMPT

        try:
            recognized_text = call_vlm(
                region.image_bytes, prompt,
                provider=provider, model=model,
                base_url=base_url, api_key=api_key,
            )
        except Exception as e:
            logger.error(f"VLM failed for page {region.page} ({region.region_type}): {e}")
            fail_count += 1
            continue

        if not recognized_text:
            logger.warning(f"VLM returned empty for page {region.page}")
            fail_count += 1
            continue

        logger.info(
            f"VLM recognized {len(recognized_text)} chars on page {region.page} "
            f"({region.region_type})"
        )
        success_count += 1

        # Inject into document
        if region.region_type == "full_page":
            _inject_full_page(document, region, recognized_text)
        elif region.region_type == "ink_annotation":
            _inject_ink_annotation(document, region, recognized_text)
        elif region.region_type == "empty_chunk":
            _inject_empty_chunk(document, region, recognized_text)

    logger.info(
        f"VLM handwriting: {success_count} regions recognized, {fail_count} failed"
    )
    return document


def _inject_full_page(
    document: ExtractedDocument,
    region: HandwritingRegion,
    text: str,
) -> None:
    """Add VLM-recognized text as a new element on a handwritten page."""
    page = document.get_page(region.page)
    if not page:
        # Create new page
        page = PageContent(page_number=region.page)
        document.pages.append(page)
        document.pages.sort(key=lambda p: p.page_number)

    element = ExtractedElement(
        content=text,
        element_type=ElementType.TEXT,
        page=region.page,
        bbox=region.bbox,
        confidence=0.7,  # Lower confidence for VLM-recognized text
        metadata={"vlm_region_type": "full_page"},
        source="vlm_handwriting",
    )
    page.elements.append(element)
    page.text_content = text


def _inject_ink_annotation(
    document: ExtractedDocument,
    region: HandwritingRegion,
    text: str,
) -> None:
    """Attach VLM-recognized text to the nearest element as a handwritten annotation."""
    page = document.get_page(region.page)
    if not page:
        return

    annotation = AnnotationInfo(
        type="ink",
        handwritten_text=text,
    )

    # Find nearest element by bbox proximity
    if region.bbox and page.elements:
        best_elem = None
        best_dist = float("inf")
        for elem in page.elements:
            if not elem.bbox:
                continue
            # Distance between centers
            cx1, cy1 = region.bbox.center
            cx2, cy2 = elem.bbox.center
            dist = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
            if dist < best_dist:
                best_dist = dist
                best_elem = elem

        if best_elem:
            best_elem.annotations.append(annotation)
            return

    # No nearby element — create a standalone element
    element = ExtractedElement(
        content=text,
        element_type=ElementType.TEXT,
        page=region.page,
        bbox=region.bbox,
        confidence=0.7,
        metadata={"vlm_region_type": "ink_annotation"},
        annotations=[annotation],
        source="vlm_handwriting",
    )
    page.elements.append(element)


def _inject_empty_chunk(
    document: ExtractedDocument,
    region: HandwritingRegion,
    text: str,
) -> None:
    """Replace empty chunk content with VLM-recognized handwritten text."""
    page = document.get_page(region.page)
    if not page:
        return

    # Find the empty element by matching bbox
    for elem in page.elements:
        if not elem.bbox or not region.bbox:
            continue
        if elem.bbox.overlap_ratio(region.bbox) > 0.8 and len(elem.content.strip()) < 10:
            elem.content = text
            elem.source = "vlm_handwriting"
            elem.confidence = 0.7
            return
