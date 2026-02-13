"""
Document parsers for the ingestion pipeline.

Provides:
- DoclingParser: Full-featured document parsing with Docling
- parse_document: Convenience function for document parsing
"""

from .docling_parser import DoclingParser, parse_document

__all__ = [
    "DoclingParser",
    "parse_document",
]
