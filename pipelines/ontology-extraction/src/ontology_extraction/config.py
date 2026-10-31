"""Configuration for the ontology-guided extraction pipeline."""

from dataclasses import dataclass


@dataclass
class ExtractionConfig:
    """Configuration for the ontology-guided extraction pipeline."""

    ontology_name: str = "general"
    llm_model: str = "default"
    max_entities_per_chunk: int = 50
    max_relations_per_chunk: int = 30
    batch_size: int = 10
    include_concepts: bool = True
    include_claims: bool = True
    confidence_threshold: float = 0.5

    # Per-chunk extraction timeout in seconds (0 = no timeout)
    extraction_timeout: int = 300

    # Extractor selection: alleen "llm" (LangExtract is weg, F2-09)
    extractor_type: str = "llm"

