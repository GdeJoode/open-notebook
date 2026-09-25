"""Stage 13 (edge prediction) is off in every path the app builds itself (D6).

D6 measured what the stage wrote: 538 of 1,892 relations, persisted as `RELATED`
with `extraction_method = 'llm'`, indistinguishable from extracted edges except
by a property nothing reads. About one in six is a relation the source text
states (16.8%, 95% CI 8.5-26.8%, against 90% for extracted edges); 528 of 538
join two entities that already share an extracted neighbour. Report:
`docs/tracks/PC-pipeline-coherence/phase-D6-edge-prediction.md`.

What the guard pins:

* the two production sites that build a `FilteringConfig` for a real run (the
  default extraction path and the `run-filtering` route) STATE
  `edge_prediction_enabled` as the literal False. Silence is refused there: it
  would fall through to a default three files away;
* no other keyword `FilteringConfig(...)` in production code enables it, by
  literal or by any computed value;
* the dataclass default those silent sites fall through to is False.

Re-enabling is meant to fail here. The condition for the stage's return is that
its output is stamped with its own provenance and routed to the curator queue as
a proposal; at that point this guard is rewritten to check that destination.

DECLARED LIMITS. Call sites are derived by
`test_kg_resolution_default._filtering_config_calls`: calls spelled
`FilteringConfig(...)` with named keywords in tracked `/src/` code. It does not
see `FilteringConfig(**data)` (the CLI's operator-supplied `--config` file),
attribute-qualified calls, `dataclasses.replace(...)`, attribute stores after
construction, or `scripts/`. None of those enable the stage today.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Tuple

from entity_filtering.config import FilteringConfig
from tests.test_kg_resolution_default import _filtering_config_calls

#: The production sites that must state the choice, not inherit it.
MUST_STATE = (
    "app_main/services/entity_extraction_service.py",
    "app_main/api/routers/sources_processing.py",
)


def _violations(calls: List[Tuple[str, ast.Call]]) -> List[str]:
    bad: List[str] = []
    stated = set()
    for rel, call in calls:
        kw = next((k for k in call.keywords if k.arg == "edge_prediction_enabled"), None)
        if kw is None:
            continue
        if not (isinstance(kw.value, ast.Constant) and kw.value.value is False):
            bad.append(f"{rel}:{call.lineno} sets edge_prediction_enabled to "
                       f"{ast.unparse(kw.value)}")
            continue
        stated.update(s for s in MUST_STATE if rel.endswith(s))
    bad += [f"{s} does not state edge_prediction_enabled=False"
            for s in MUST_STATE if s not in stated]
    return bad


def test_no_path_the_app_builds_enables_edge_prediction() -> None:
    calls = _filtering_config_calls()
    assert calls, "walker control: no keyword FilteringConfig(...) call found"
    bad = _violations(calls)
    assert not bad, (
        f"{bad}. D6 measured stage 13 writing two-hop shortcuts as unmarked "
        f"relations; it returns only as a proposer with its own provenance. "
        f"Rewrite this guard for that destination rather than flipping it."
    )


def test_the_default_silent_sites_fall_through_to_is_off() -> None:
    assert FilteringConfig().edge_prediction_enabled is False


def _real_site(suffix: str) -> Tuple[str, str]:
    rel = next(r for r, _ in _filtering_config_calls() if r.endswith(suffix))
    src = (Path(__file__).resolve().parents[3] / rel).read_text(encoding="utf-8")
    assert src.count("edge_prediction_enabled=False,") == 1
    return rel, src


def _calls_with(rel: str, src: str) -> List[Tuple[str, ast.Call]]:
    """The real tree, with one file's source replaced by `src`."""
    others = [(r, c) for r, c in _filtering_config_calls() if r != rel]
    mutated = [(rel, n) for n in ast.walk(ast.parse(src))
               if isinstance(n, ast.Call) and getattr(n.func, "id", "") == "FilteringConfig"
               and n.keywords and all(k.arg for k in n.keywords)]
    return others + mutated


def test_the_guard_fails_on_the_states_it_prevents() -> None:
    """THE states, not ones of their shape: both real sites, each mutated three
    ways, and each caught by the check meant for it."""
    for suffix in MUST_STATE:
        rel, src = _real_site(suffix)
        assert not _violations(_calls_with(rel, src)), "control: the unmutated tree passes"
        for mutation, expected in (
            ("edge_prediction_enabled=True,", "sets edge_prediction_enabled to True"),
            ("edge_prediction_enabled=body.edge_prediction_enabled,",
             "sets edge_prediction_enabled to body.edge_prediction_enabled"),
            ("", f"{suffix} does not state edge_prediction_enabled=False"),
        ):
            found = _violations(_calls_with(
                rel, src.replace("edge_prediction_enabled=False,", mutation)))
            assert any(expected in v for v in found), (
                f"{suffix} with {mutation or 'the keyword deleted'!r}: expected "
                f"{expected!r}, got {found}"
            )
