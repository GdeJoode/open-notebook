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

    # Per-chunk extraction timeout in seconds (0 = no timeout)
    extraction_timeout: int = 300

    # Extractor selection: "llm" (default) or "langextract"
    extractor_type: str = "llm"

    # LangExtract-specific settings
    langextract_model_id: str = "qwen2.5:latest"
    langextract_model_url: str | None = "http://localhost:11434"
    langextract_extraction_passes: int = 1
    langextract_max_workers: int = 4
    langextract_max_char_buffer: int = 5000
    langextract_examples_dir: str | None = None

    # Performance tuning
    langextract_batch_length: int | None = None

    # Model behavior
    langextract_temperature: float | None = None
    langextract_max_output_tokens: int | None = None
    langextract_top_p: float | None = None
    langextract_top_k: int | None = None

    # Schema/output control
    langextract_use_schema_constraints: bool = True
    langextract_fence_output: bool = True

    # Provider configuration
    langextract_api_key: str | None = None
    langextract_provider: str | None = None
    langextract_provider_kwargs: dict | None = None
    langextract_language_model_params: dict | None = None

    # I/O features
    langextract_save_jsonl: bool = False
    langextract_jsonl_output_dir: str | None = None
    langextract_visualize: bool = False
    langextract_visualize_output_dir: str | None = None
