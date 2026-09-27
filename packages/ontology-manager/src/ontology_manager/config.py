"""
Configuration for ontology manager service.
"""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class OntologyManagerConfig(BaseSettings):
    """Ontology manager configuration."""

    model_config = {
        "env_prefix": "ONTOLOGY_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # Ontology directory
    ontologies_dir: str = Field(
        default="ontologies",
        description="Directory containing YAML ontology definitions",
    )

    # Default ontology
    default_ontology: str = Field(
        default="general",
        description="Default ontology name to use when none specified",
    )

    # Cache settings
    cache_enabled: bool = Field(
        default=True,
        description="Cache loaded ontologies in memory",
    )

    # Evolution/gap tracking
    gap_tracking_enabled: bool = Field(
        default=True,
        description="Track extraction gaps for ontology evolution",
    )
    gap_auto_propose_threshold: int = Field(
        default=5,
        description="Frequency threshold before auto-proposing schema changes",
    )


# Global config instance
_config: Optional[OntologyManagerConfig] = None


def get_config() -> OntologyManagerConfig:
    """Get the global ontology manager configuration."""
    global _config
    if _config is None:
        _config = OntologyManagerConfig()
    return _config


def set_config(config: OntologyManagerConfig) -> None:
    """Set the global ontology manager configuration."""
    global _config
    _config = config
