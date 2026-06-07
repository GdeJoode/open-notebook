"""
Cross-package configuration constants.

Single source of truth for thresholds that must agree across the
backend pipelines (``ontology-extraction``) and the frontend / UI
layer (``apps/app-main`` API + future B.3c notebook UI). Putting them
here means a change in policy is a one-file edit, and the import
graph forces every consumer to read the same value.

Phase B.1e (multi-schema orchestrator)
======================================
The three constants below back the *soft-nudge* policy described in
``docs/tracks/B-kg-quality/plan.md`` §Phase B.1e:

- ``SOFT_NUDGE_COVERAGE_HIGH``: above this, the orchestrator stays
  silent (the schema covers the source well enough).
- ``SOFT_NUDGE_COVERAGE_LOW``: between LOW and HIGH the orchestrator
  emits an ``extension_suggested`` decision; below LOW it emits
  ``schema_mismatch``.
- ``MIN_APPLICABLE_CONFIDENCE``: applicability score floor when
  ``detect_applicable_schemas`` ranks candidates.

These were lifted out of the orchestrator module per RETRO lesson #2
(single-source-of-truth for ops-tunable thresholds) so the eventual
B.3c slider reads the same values the backend enforces.
"""

from __future__ import annotations

# Coverage threshold above which Pass-1 stays silent (no soft-nudge).
# Set high enough that a well-fitted schema doesn't pester the user.
SOFT_NUDGE_COVERAGE_HIGH: float = 0.95

# Coverage threshold below which the orchestrator surfaces a stronger
# "schema mismatch" hint. Between LOW and HIGH the orchestrator only
# suggests extensions.
SOFT_NUDGE_COVERAGE_LOW: float = 0.80

# Floor on schema applicability score returned by
# ``detect_applicable_schemas``. Candidates scoring below this never
# enter the top-K shortlist regardless of their relative ranking.
MIN_APPLICABLE_CONFIDENCE: float = 0.3


__all__ = [
    "MIN_APPLICABLE_CONFIDENCE",
    "SOFT_NUDGE_COVERAGE_HIGH",
    "SOFT_NUDGE_COVERAGE_LOW",
]
