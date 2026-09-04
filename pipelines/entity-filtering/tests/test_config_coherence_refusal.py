"""The workflow refuses a config in which an enabled stage cannot run (PC.6).

`packages/shared/tests/test_config_coherence.py` proves the checks are correct.
This proves they are REACHED from the one place that sees the whole config — the
gap review found in round 1, where the ontology finding existed and its only
production caller never passed the arguments that could produce it.
"""

from __future__ import annotations

import pytest
from entity_filtering.config import (
    FilteringConfig,
    OntologyValidationConfig,
    SemanticConfig,
)
from entity_filtering.workflow import FilteringWorkflow
from shared.config_coherence import ConfigurationError


def _codes(excinfo) -> set:
    return {f.code for f in excinfo.value.findings}


def test_validation_without_an_ontology_is_refused_at_construction() -> None:
    cfg = FilteringConfig(ontology_validation=OntologyValidationConfig(enabled=True))
    with pytest.raises(ConfigurationError) as excinfo:
        FilteringWorkflow(config=cfg)
    assert "validation-without-ontology" in _codes(excinfo)


def test_linking_without_a_linker_is_refused() -> None:
    cfg = FilteringConfig(semantic=SemanticConfig(entity_linking_enabled=True))
    with pytest.raises(ConfigurationError) as excinfo:
        FilteringWorkflow(config=cfg)
    assert "linking-without-a-linker" in _codes(excinfo)


def test_an_injected_linker_satisfies_the_dependency() -> None:
    """The escape hatch must work, or the check forbids dependency injection."""
    cfg = FilteringConfig(semantic=SemanticConfig(entity_linking_enabled=True))
    FilteringWorkflow(config=cfg, entity_linker=object())


def test_outliers_without_centrality_are_refused() -> None:
    cfg = FilteringConfig(
        ontology_validation=OntologyValidationConfig(
            outlier_detection_enabled=True, graph_centrality_enabled=False
        )
    )
    with pytest.raises(ConfigurationError) as excinfo:
        FilteringWorkflow(config=cfg)
    assert "outliers-without-centrality" in _codes(excinfo)


def test_the_shipped_defaults_build() -> None:
    """The counterweight, and the one that catches an over-eager refusal.

    A check that refuses `FilteringConfig()` refuses every run, and would be
    disabled within the hour.
    """
    FilteringWorkflow(config=FilteringConfig())
