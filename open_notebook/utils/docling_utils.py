"""
Utility functions for working with DoclingDocument objects in the processing pipeline.
"""

from typing import Optional

from docling_core.types.doc import DoclingDocument
from loguru import logger
from pydantic import ValidationError


def reconstruct_docling_document(metadata: dict) -> Optional[DoclingDocument]:
    """
    Reconstruct a DoclingDocument from serialized JSON in metadata.

    Args:
        metadata: State metadata dictionary potentially containing docling_document_json

    Returns:
        DoclingDocument instance if reconstruction succeeds, None otherwise

    Example:
        ```python
        from open_notebook.utils.docling_utils import reconstruct_docling_document

        # In a LangGraph node
        doc = reconstruct_docling_document(state["content_state"].metadata)
        if doc:
            # Use the document for embeddings, further processing, etc.
            markdown = doc.export_to_markdown()
        ```
    """
    docling_json = metadata.get("docling_document_json")
    if not docling_json:
        logger.debug("No docling_document_json found in metadata")
        return None

    try:
        # Reconstruct the Pydantic model from JSON
        doc = DoclingDocument.model_validate_json(docling_json)
        logger.debug("✅ Successfully reconstructed DoclingDocument from JSON")
        return doc
    except ValidationError as e:
        logger.error(f"Validation failed while reconstructing DoclingDocument: {e}")
        for error in e.errors():
            logger.error(f"  - {error['loc']}: {error['msg']}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reconstructing DoclingDocument: {e}")
        return None


def has_docling_document(metadata: dict) -> bool:
    """
    Check if metadata contains a serialized DoclingDocument.

    Args:
        metadata: State metadata dictionary

    Returns:
        True if docling_document_json exists in metadata
    """
    return "docling_document_json" in metadata and metadata["docling_document_json"] is not None
