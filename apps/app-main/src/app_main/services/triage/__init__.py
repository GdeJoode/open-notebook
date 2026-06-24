"""Post-extraction triage services (Track Q).

Tiering, predicate-strength, and status-rule configuration is sourced
exclusively from ``config/triage_config.json`` via :mod:`triage_config`.
"""

from app_main.services.triage.signals_service import (
    EntitySignals,
    SignalsReport,
    TriageSignalsService,
)
from app_main.services.triage.triage_config import (
    TriageConfig,
    TriageConfigError,
    load_triage_config,
)

__all__ = [
    "TriageConfig",
    "TriageConfigError",
    "load_triage_config",
    "TriageSignalsService",
    "SignalsReport",
    "EntitySignals",
]
