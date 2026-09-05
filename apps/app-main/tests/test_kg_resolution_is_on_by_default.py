"""Cross-document resolution is in the DEFAULT path (PC.3 step 2).

`KGResolutionConfig.enabled` defaults to False, and before this the
`FilteringConfig` the app builds never set it — so stage 10, the stage that
matches a new mention against entities the graph already holds, never ran in a
real extraction. Every document wrote its entities fresh.

The guard is DERIVED from the constructor call rather than asserted against a
built object, because building one needs the extraction service's whole
dependency graph. It reads the source by AST, finds the `FilteringConfig(...)`
the service constructs, and checks the keyword — so a later edit that drops it is
caught, which an assertion on `KGResolutionConfig()` alone would not be: that
would still pass while the app stopped passing it.
"""

from __future__ import annotations

import ast
import inspect
from typing import List

from app_main.services import entity_extraction_service


def _filtering_config_calls() -> List[ast.Call]:
    tree = ast.parse(inspect.getsource(entity_extraction_service))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "FilteringConfig"
        and node.keywords  # the bare `FilteringConfig()` fallback is a different site
    ]


def test_the_app_builds_its_config_with_kg_resolution_enabled() -> None:
    calls = _filtering_config_calls()
    assert calls, "walker control: no keyword FilteringConfig(...) call found"

    for call in calls:
        keywords = {k.arg for k in call.keywords}
        assert "kg_resolution" in keywords, (
            f"the FilteringConfig at line {call.lineno} does not set "
            f"`kg_resolution`. Stage 10 is off by default, so omitting it means "
            f"every document writes its entities fresh — the defect PC.3 exists "
            f"to fix, reintroduced silently."
        )
        kg = next(k.value for k in call.keywords if k.arg == "kg_resolution")
        enabled = [k for k in getattr(kg, "keywords", []) if k.arg == "enabled"]
        assert enabled, f"line {call.lineno}: kg_resolution passed without `enabled`"
        assert getattr(enabled[0].value, "value", None) is True, (
            f"line {call.lineno}: kg_resolution is not enabled"
        )


def test_the_alias_policy_is_not_restated_here() -> None:
    """`register_aliases` must come from PC.2's default, not be set again.

    PC.2 settled that a fuzzy match may not register an alias by itself, and put
    that in `KGResolutionConfig`. Restating it at a call site is how one decision
    grows two answers — the exact defect PC.6 spent five rounds on. Enabling
    stage 10 is the moment that risk becomes live, because tier 1 is the alias
    tier.
    """
    from entity_filtering.config import KGResolutionConfig

    assert KGResolutionConfig().register_aliases is False

    for call in _filtering_config_calls():
        kg = next(
            (k.value for k in call.keywords if k.arg == "kg_resolution"), None
        )
        if kg is None:
            continue
        restated = {k.arg for k in getattr(kg, "keywords", [])} & {"register_aliases"}
        assert not restated, (
            f"line {call.lineno} restates {restated}; leave it to "
            f"KGResolutionConfig so the policy lives in one place"
        )
