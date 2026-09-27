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
    gap_auto_propose_enabled: bool = Field(
        default=False,
        description=(
            "Let the evolution agent turn a recurring gap into a schema_proposal "
            "row. Off (PC.5a, owner 2026-09-27): a gap is a NEW INSTANCE of a "
            "type that resolved (`GAP_LICENSING_CODES`), so the proposal named a "
            "new type after one entity (`Gemeente Opmeer` -> `GemeenteOpmeer`), "
            "and nothing read the rows. Until PC.5a this followed "
            "gap_tracking_enabled, so tracking gaps silently meant proposing types."
        ),
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
