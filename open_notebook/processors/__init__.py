"""
Custom document processors for Open Notebook.
"""

from open_notebook.processors.chunk_extractor import extract_chunks_from_docling
from open_notebook.processors.docling_gpu import extract_with_docling_gpu

__all__ = ["extract_chunks_from_docling", "extract_with_docling_gpu"]
