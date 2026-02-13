"""
Document chunking for the ingestion pipeline.

Provides:
- Chunk: Data model for semantic chunks
- ChunkExtractor: Extract chunks with bounding boxes from documents
- ChunkRegrouper: Merge 1:1 element chunks into semantically meaningful units
"""

from .extractor import Chunk, ChunkExtractor, ChunkPosition, extract_chunks
from .regrouper import ChunkRegrouper

__all__ = [
    "Chunk",
    "ChunkExtractor",
    "ChunkPosition",
    "ChunkRegrouper",
    "extract_chunks",
]
