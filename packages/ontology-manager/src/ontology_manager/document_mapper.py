"""
Document Type to Ontology Mapper

Maps detected document types to the most appropriate ontology for extraction.
Ontologies form an inheritance hierarchy:
  - base (root)
  - general (extends base)
  - scholarly (extends general) - for academic papers
  - policy (extends general) - for policy documents
  - social_profiles (extends general) - for social media content
"""

from typing import Optional
from loguru import logger


# Document type to ontology mapping
# Maps SourceType values to ontology names
DOCUMENT_TYPE_ONTOLOGY_MAP = {
    # Academic/Research documents → scholarly ontology
    "academic_paper": "scholarly",
    "research_paper": "scholarly",
    "journal_article": "scholarly",
    "conference_paper": "scholarly",
    "thesis": "scholarly",
    "dissertation": "scholarly",
    "preprint": "scholarly",
    "working_paper": "scholarly",

    # Policy documents → policy ontology
    "policy_document": "policy",
    "policy_advice": "policy",
    "policy_brief": "policy",
    "legislation": "policy",
    "regulation": "policy",
    "government_report": "policy",
    "white_paper": "policy",

    # Social media → social_profiles ontology
    "social_media": "social_profiles",
    "social_media_post": "social_profiles",
    "tweet": "social_profiles",
    "linkedin_post": "social_profiles",
    "blog_post": "social_profiles",

    # News and general content → general ontology
    "news_article": "general",
    "press_release": "general",
    "report": "general",
    "presentation": "general",
    "legal_document": "general",
    "book": "general",
    "book_chapter": "general",
    "manual": "general",
    "documentation": "general",

    # Default/unknown → general ontology
    "other": "general",
    "unknown": "general",
}

# Default ontology when no mapping found
DEFAULT_ONTOLOGY = "general"


def get_ontology_for_document_type(document_type: Optional[str]) -> str:
    """
    Get the appropriate ontology name for a document type.

    Args:
        document_type: The detected document type (e.g., "academic_paper", "policy_document")

    Returns:
        The ontology name to use for extraction (e.g., "scholarly", "policy", "general")

    Examples:
        >>> get_ontology_for_document_type("academic_paper")
        'scholarly'
        >>> get_ontology_for_document_type("policy_document")
        'policy'
        >>> get_ontology_for_document_type("news_article")
        'general'
        >>> get_ontology_for_document_type(None)
        'general'
    """
    if not document_type:
        return DEFAULT_ONTOLOGY

    # Normalize the document type (lowercase, strip whitespace)
    normalized = document_type.lower().strip()

    # Look up in mapping
    ontology = DOCUMENT_TYPE_ONTOLOGY_MAP.get(normalized, DEFAULT_ONTOLOGY)

    if ontology != DEFAULT_ONTOLOGY:
        logger.debug(f"Document type '{document_type}' mapped to ontology '{ontology}'")

    return ontology


def get_supported_document_types() -> list[str]:
    """
    Get list of all supported document types.

    Returns:
        List of document type strings that have ontology mappings
    """
    return list(DOCUMENT_TYPE_ONTOLOGY_MAP.keys())


def get_document_types_for_ontology(ontology_name: str) -> list[str]:
    """
    Get all document types that map to a specific ontology.

    Args:
        ontology_name: The ontology name (e.g., "scholarly", "policy")

    Returns:
        List of document types that use this ontology
    """
    return [
        doc_type
        for doc_type, ont in DOCUMENT_TYPE_ONTOLOGY_MAP.items()
        if ont == ontology_name
    ]
