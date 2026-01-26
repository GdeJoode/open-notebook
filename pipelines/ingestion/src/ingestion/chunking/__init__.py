"""
Document chunking for the ingestion pipeline.

Provides:
- Chunk: Data model for semantic chunks
- ChunkExtractor: Extract chunks with bounding boxes from documents
"""

from .extractor import Chunk, ChunkExtractor, ChunkPosition, extract_chunks

__all__ = [
    "Chunk",
    "ChunkExtractor",
    "ChunkPosition",
    "extract_chunks",
]
