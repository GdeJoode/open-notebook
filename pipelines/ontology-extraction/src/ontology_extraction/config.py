"""Configuration for the ontology-guided extraction pipeline."""

from dataclasses import dataclass, field


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
