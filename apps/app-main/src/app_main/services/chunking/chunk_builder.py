"""
Pure functions that turn extraction results into chunk dicts.

These were ``SourceProcessingService._document_to_chunks`` /
``_transcription_to_chunks`` / ``_prepare_chunks_for_db`` before the
Phase 2 refactor. Pulled out so they can be tested without spinning up
the service + all its repository mocks.

No state, no I/O, no repository access — every function is input → output.
"""

from __future__ import annotations

from typing import Any, Dict, List

from shared.utils.text import strip_null_bytes


def from_document(document: Any) -> List[Dict[str, Any]]:
    """Convert ExtractedDocument pages/elements into chunk dicts.

    Iterates each page's text elements, tables, and images in source
    order; emits one chunk per element with positional metadata.

    Emitted ``positions`` are ``[page, x1, x2, y1, y2]`` with x/y already
    normalized to 0–1 (TOPLEFT origin) by ``BoundingBox.from_*`` — no
    rescaling happens here, so every parser's output is uniform.
    """
    chunks: List[Dict[str, Any]] = []
    order = 0
    # Global image counter matching the exporter's enumerate(all_images, 1)
    image_idx = 0

    for page in document.pages:
        # Text elements
        for elem in page.elements:
            bbox = elem.bbox
            positions = []
            if bbox and (bbox.width > 0 or bbox.height > 0):
                positions = [
                    [bbox.page, bbox.x, bbox.x2, bbox.y, bbox.y2]
                ]

            chunks.append(
                {
                    "text": elem.content,
                    "order": order,
                    "physical_page": page.page_number - 1,
                    "printed_page": page.page_number,
                    "chapter": (
                        elem.section_path[-1] if elem.section_path else None
                    ),
                    "paragraph_number": None,
                    "element_type": elem.element_type.value,
                    "positions": positions,
                    "metadata": {
                        "has_spatial_data": len(positions) > 0,
                        "section_path": elem.section_path,
                        "section_level": elem.section_level,
                    },
                }
            )
            order += 1

        # Tables
        for table in page.tables:
            bbox = table.bbox
            positions = []
            if bbox and (bbox.width > 0 or bbox.height > 0):
                positions = [
                    [bbox.page, bbox.x, bbox.x2, bbox.y, bbox.y2]
                ]

            chunks.append(
                {
                    "text": table.markdown or table.content,
                    "order": order,
                    "physical_page": page.page_number - 1,
                    "printed_page": page.page_number,
                    "chapter": None,
                    "paragraph_number": None,
                    "element_type": "table",
                    "positions": positions,
                    "metadata": {
                        "has_spatial_data": len(positions) > 0,
                        "num_rows": table.num_rows,
                        "num_cols": table.num_cols,
                    },
                }
            )
            order += 1

        # Images — store filename reference, not base64 data
        for image in page.images:
            image_idx += 1
            bbox = image.bbox
            positions = []
            if bbox and (bbox.width > 0 or bbox.height > 0):
                positions = [
                    [bbox.page, bbox.x, bbox.x2, bbox.y, bbox.y2]
                ]

            # Filename matches the exporter convention:
            # image_{idx:03d}_page{page}.{format}
            img_format = image.format or "png"
            image_file = (
                f"image_{image_idx:03d}_page{image.page}.{img_format}"
            )

            chunks.append(
                {
                    "text": image.description or image.content or f"[{image.classification}]",
                    "order": order,
                    "physical_page": page.page_number - 1,
                    "printed_page": page.page_number,
                    "chapter": None,
                    "paragraph_number": None,
                    "element_type": "image",
                    "positions": positions,
                    "metadata": {
                        "has_spatial_data": len(positions) > 0,
                        "classification": image.classification,
                        "width": image.width,
                        "height": image.height,
                        "format": img_format,
                        "image_file": image_file,
                    },
                }
            )
            order += 1

    return chunks


def from_transcription(transcription: Any) -> List[Dict[str, Any]]:
    """Convert TranscriptionResult segments into chunk dicts.

    One chunk per segment, preserving ``start``/``end``/``speaker`` in
    metadata for downstream time-aligned retrieval.
    """
    chunks: List[Dict[str, Any]] = []

    for i, segment in enumerate(transcription.segments):
        chunks.append(
            {
                "text": segment.text,
                "order": i,
                "physical_page": 0,
                "printed_page": None,
                "chapter": None,
                "paragraph_number": None,
                "element_type": "transcription_segment",
                "positions": [],
                "metadata": {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": getattr(segment, "speaker", None),
                },
            }
        )

    return chunks


def prepare_for_db(
    chunks: List[Dict[str, Any]],
    source_id: str,
) -> List[Dict[str, Any]]:
    """Map extracted chunk dicts to the shape expected by ChunkRepository.

    Attaches ``source`` foreign key and strips null bytes from text fields
    so SurrealDB's serializer doesn't reject the row on read-back.
    """
    prepared = []
    for chunk_data in chunks:
        prepared.append(
            {
                "source": source_id,
                "text": strip_null_bytes(chunk_data["text"]),
                "order": chunk_data["order"],
                "physical_page": chunk_data["physical_page"],
                "printed_page": chunk_data.get("printed_page"),
                "chapter": strip_null_bytes(chunk_data.get("chapter")),
                "paragraph_number": chunk_data.get("paragraph_number"),
                "element_type": chunk_data.get("element_type", "paragraph"),
                "positions": chunk_data.get("positions", []),
                "metadata": strip_null_bytes(chunk_data.get("metadata", {})),
            }
        )
    return prepared
