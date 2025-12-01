"""
Custom document processors for Open Notebook.

Note: GPU and VLM processing is now handled by content-core.
See open_notebook/utils/content_core_config.py for configuration bridge.
"""

from open_notebook.processors.chunk_extractor import extract_chunks_from_docling

__all__ = ["extract_chunks_from_docling"]
