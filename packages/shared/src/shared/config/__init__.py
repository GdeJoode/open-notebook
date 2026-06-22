"""Shared configuration: cross-package threshold constants + runtime config.

Single source of truth for thresholds that must agree across the backend
pipelines (``ontology-extraction``) and the API/UI layer. A change in policy is a
one-file edit and the import graph forces every consumer to read the same value.

This is a **package** (the ``alias_overrides`` submodule, Track K.2, holds the
runtime org-alias overlay). The B.1e soft-nudge constants below were previously a
top-level ``config.py`` module; when K.2 introduced this package the module was
shadowed (``from shared.config import MIN_APPLICABLE_CONFIDENCE`` started failing
and broke ``create_app()``). They now live here so both the constants and the
submodule resolve under ``shared.config``.

Phase B.1e (multi-schema orchestrator) soft-nudge policy
========================================================
- ``SOFT_NUDGE_COVERAGE_HIGH``: above this, Pass-1 stays silent (schema fits).
- ``SOFT_NUDGE_COVERAGE_LOW``: between LOW and HIGH -> ``extension_suggested``;
  below LOW -> ``schema_mismatch``.
- ``MIN_APPLICABLE_CONFIDENCE``: applicability-score floor for
  ``detect_applicable_schemas`` candidate ranking.
"""

from __future__ import annotations

# Coverage threshold above which Pass-1 stays silent (no soft-nudge).
SOFT_NUDGE_COVERAGE_HIGH: float = 0.95

# Coverage threshold below which the orchestrator surfaces a stronger
# "schema mismatch" hint. Between LOW and HIGH it only suggests extensions.
SOFT_NUDGE_COVERAGE_LOW: float = 0.80

# Floor on schema applicability score returned by ``detect_applicable_schemas``.
MIN_APPLICABLE_CONFIDENCE: float = 0.3


__all__ = [
    "MIN_APPLICABLE_CONFIDENCE",
    "SOFT_NUDGE_COVERAGE_HIGH",
    "SOFT_NUDGE_COVERAGE_LOW",
]
