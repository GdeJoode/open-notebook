"""Track N.5d — the extraction regression gate.

The original scope was "assert the pre-LLM layer and abstention cut LLM calls
and/or over-generation WITHOUT dropping golden entities". Two things about that
sentence decide the design here.

**It is heterogeneous.** A run may legitimately get cheaper OR cleaner; it is not
required to do both. So the cost dimensions are checked as "did not get worse",
and at least one of them must have been CHECKABLE for a pass to mean anything.

**"Without dropping entities" is the binding half.** Recall is the floor: a
change that halves the LLM calls by extracting nothing is a regression, and it is
the one a cost-only gate would wave through.

Why a comparison module rather than a test that runs the pipeline
================================================================
The measurement needs Ollama, a database and minutes per document; the COMPARISON
is arithmetic. Splitting them means the rule that decides pass or fail is unit
tested and deterministic, while the expensive part stays an opt-in script. A gate
whose logic only executes during a twenty-minute live run is a gate nobody runs.

The missing-baseline trap
=========================
Some dimensions did not exist when the baseline was measured. Over-generation and
abstention are the concrete case: `_merge_results` discarded the per-pass counters
on the multi-schema path, so every merged run in the corpus baseline reports 0.0
for both — not because nothing was culled but because nobody carried the numbers
(fixed in N.5a). A gate that treated "no baseline value" as "not worse" would
report as a guard while being unable to fail on exactly the metrics this track
added. So an absent baseline yields :data:`GateOutcome.SKIPPED`, the summary names
which dimensions were skipped, and :func:`compare_against_baseline` refuses a
verdict when EVERY dimension skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Mapping, Optional, Sequence


class GateOutcome(str, Enum):
    """What a single dimension concluded."""

    PASSED = "passed"
    FAILED = "failed"
    #: No baseline value existed for this dimension. NOT a pass — a gate that
    #: silently treats an absent baseline as a pass cannot fail on a new metric.
    SKIPPED = "skipped"


@dataclass(frozen=True)
class GateDimension:
    """One checked property, with the numbers that decided it."""

    name: str
    outcome: GateOutcome
    baseline: Optional[float]
    current: Optional[float]
    detail: str

    @property
    def failed(self) -> bool:
        return self.outcome is GateOutcome.FAILED


@dataclass
class GateResult:
    """The verdict, and every dimension that contributed to it."""

    passed: bool
    dimensions: List[GateDimension] = field(default_factory=list)
    #: Set when no dimension could be evaluated at all. The gate is then
    #: inconclusive rather than green, because there is nothing behind a pass.
    inconclusive: bool = False

    @property
    def failures(self) -> List[GateDimension]:
        return [d for d in self.dimensions if d.failed]

    @property
    def skipped(self) -> List[GateDimension]:
        return [d for d in self.dimensions if d.outcome is GateOutcome.SKIPPED]

    #: ``{metric: count}`` for metrics some documents could not supply, carried
    #: from the run summary so a skip or failure can say WHICH documents fell
    #: short rather than only that something did.
    missing_counters: Dict[str, int] = field(default_factory=dict)

    def report(self) -> str:
        """A human-readable summary; the script prints this and CI logs it."""
        lines: List[str] = []
        for d in self.dimensions:
            mark = {
                GateOutcome.PASSED: "ok  ",
                GateOutcome.FAILED: "FAIL",
                GateOutcome.SKIPPED: "skip",
            }[d.outcome]
            lines.append(f"  [{mark}] {d.name}: {d.detail}")
        for metric, count in sorted(self.missing_counters.items()):
            lines.append(
                f"         ({count} document(s) in this run carried no inputs for "
                f"{metric})"
            )
        if self.inconclusive:
            lines.append("  INCONCLUSIVE — no dimension had a baseline to compare against")
        else:
            lines.append(f"  => {'PASS' if self.passed else 'FAIL'}")
        return "\n".join(lines)


# How much a metric may drift before it counts. Entity recall gets the tighter
# band because it is the floor the whole gate exists to protect; the cost
# dimensions get more room because they are ratios over small counts, where one
# document swings them.
RECALL_TOLERANCE = 0.10
COST_TOLERANCE = 0.15


# Which counters each rate needs, and the rule that a rate is reported ONLY when
# every document could supply all of them.
#
# A review found the alternative the hard way. The first version set a single
# `saw_counters` flag from whole-dict truthiness and then guarded each rate on
# one input: `abstain_rate` on `chunk_count > 0`, which is its DENOMINATOR. The
# legacy single-schema path emits `chunk_count` and none of the abstention
# counters — it uses a pluggable extractor rather than `run_pass2`, so it
# genuinely cannot count abstention — and the result was `abstain_rate: 0.0` for
# a run in which abstention was never measured. That is this track's own thesis,
# committed by the code written to enforce it: an absent measurement reading as a
# measurement of zero.
#
# Requiring EVERY document to carry a metric's inputs is stricter than requiring
# one to, and deliberately so. A mixed corpus where half the documents took the
# legacy path would otherwise sum legacy `chunk_count` into the denominator while
# only multi-schema documents could contribute to the numerator, understating the
# rate by exactly the legacy share. `documents_missing_counters` records the count
# so a skip can say why rather than merely saying it.
_RATE_INPUTS: Dict[str, tuple] = {
    "over_generation_rate": ("entities_extracted", "entities_kept"),
    "abstain_rate": ("abstained_chunks", "chunk_count"),
}
_COUNTER_INPUTS = tuple(
    dict.fromkeys(key for inputs in _RATE_INPUTS.values() for key in inputs)
)


def summarise_run(documents: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Reduce a harness run to the numbers the gate compares.

    ``documents`` are the per-document records `n_pipeline_review_run.py` writes:
    a ``result`` block with the counts, and optionally the N.5a metadata counters.
    Documents that produced no chunks are counted but contribute no entities,
    which is what a parse failure looks like and is worth carrying rather than
    silently dropping.

    Metrics whose inputs are absent come back as ``None`` rather than 0.0, so a
    missing measurement can be told apart from a measured zero. That distinction
    is the whole of the missing-baseline handling above.
    """
    total_entities = 0
    total_relations = 0
    documents_with_entities = 0
    totals: Dict[str, int] = {key: 0 for key in _COUNTER_INPUTS}
    # Per METRIC, how many documents could not contribute to it. A metric is
    # reported only when that count is zero — see `_RATE_INPUTS` below.
    missing: Dict[str, int] = {metric: 0 for metric in _RATE_INPUTS}

    for doc in documents:
        result = doc.get("result") or {}
        entities = int(result.get("entity_count", 0) or 0)
        total_entities += entities
        total_relations += int(result.get("relation_count", 0) or 0)
        if entities > 0:
            documents_with_entities += 1

        counters = result.get("counters") or doc.get("counters") or {}
        for metric, inputs in _RATE_INPUTS.items():
            if all(key in counters for key in inputs):
                for key in inputs:
                    try:
                        totals[key] += int(counters[key] or 0)
                    except (TypeError, ValueError):
                        missing[metric] += 1
                        break
            else:
                missing[metric] += 1

    summary: Dict[str, Any] = {
        "documents": len(documents),
        "documents_with_entities": documents_with_entities,
        "total_entities": total_entities,
        "total_relations": total_relations,
        # None, not 0.0: "nobody measured" is not "nothing was culled". Before
        # N.5a the merge discarded these counters, so a corpus measured then has
        # no value here and must not be compared as though it had one.
        "over_generation_rate": None,
        "abstain_rate": None,
        "documents_missing_counters": {
            metric: count for metric, count in missing.items() if count
        },
    }

    extracted = totals["entities_extracted"]
    if not missing["over_generation_rate"] and extracted > 0:
        summary["over_generation_rate"] = max(
            0.0, (extracted - totals["entities_kept"]) / extracted
        )
    chunk_passes = totals["chunk_count"]
    if not missing["abstain_rate"] and chunk_passes > 0:
        summary["abstain_rate"] = min(1.0, totals["abstained_chunks"] / chunk_passes)
    return summary


def _unmeasurable(
    name: str, baseline: Optional[float], current: Optional[float]
) -> Optional[GateDimension]:
    """Decide whether a dimension can be compared at all, and how it fails if not.

    Three cases, and a review found that collapsing them was wrong twice over.

    * **No baseline, no current** — nobody has ever measured this. SKIPPED.
    * **No baseline, current present** — the legitimate case this rule was written
      for: a metric that did not exist when the baseline was taken. SKIPPED, and
      never PASSED, because a gate that reads "no baseline" as "not worse" cannot
      fail on anything newly added.
    * **Baseline present, current absent** — the MEASUREMENT DISAPPEARED. That is
      not a baseline problem, it is this track's own defect recurring: N.5a
      existed because the merge stopped carrying these counters. Reporting it as
      "no baseline value" and passing was the first version's behaviour and it
      was exactly backwards, so it FAILS.
    """
    if baseline is None and current is None:
        return GateDimension(
            name, GateOutcome.SKIPPED, baseline, current,
            "never measured on either side — not checked, and not a pass",
        )
    if baseline is None:
        return GateDimension(
            name, GateOutcome.SKIPPED, baseline, current,
            f"no baseline value (this run measured {current:g}) — not checked, "
            "and not counted as a pass",
        )
    if current is None:
        return GateDimension(
            name, GateOutcome.FAILED, baseline, current,
            f"the baseline measured {baseline:g} and this run measured nothing. "
            "Either the counters stopped being carried (a regression in the "
            "extraction path) or this run took a route that cannot count them "
            "(the legacy single-schema path emits no abstention counters) — "
            "both make the two runs incomparable, so neither is a pass",
        )
    return None


def _compare_floor(
    name: str,
    baseline: Optional[float],
    current: Optional[float],
    tolerance: float,
) -> GateDimension:
    """A dimension that must not FALL: recall and its kin."""
    unmeasurable = _unmeasurable(name, baseline, current)
    if unmeasurable is not None:
        return unmeasurable
    if baseline <= 0:
        # A floor derived from zero holds nothing up: `0 * (1 - tolerance)` is
        # 0.0 and every possible run clears it. A review reached this by
        # recording a baseline from a corpus where every document yielded
        # nothing — a plausible accident with Ollama down — and the gate was
        # then permanently green while reporting `inconclusive=False`.
        return GateDimension(
            name, GateOutcome.SKIPPED, baseline, current,
            "baseline is zero — a floor derived from it would pass anything",
        )
    floor = baseline * (1.0 - tolerance)
    ok = current >= floor
    return GateDimension(
        name,
        GateOutcome.PASSED if ok else GateOutcome.FAILED,
        baseline,
        current,
        f"{current:g} vs baseline {baseline:g} (floor {floor:.2f}, "
        f"tolerance {tolerance:.0%})",
    )


def _compare_ceiling(
    name: str,
    baseline: Optional[float],
    current: Optional[float],
    tolerance: float,
) -> GateDimension:
    """A dimension that must not RISE: the cost metrics."""
    unmeasurable = _unmeasurable(name, baseline, current)
    if unmeasurable is not None:
        return unmeasurable
    ceiling = baseline + tolerance
    ok = current <= ceiling
    return GateDimension(
        name,
        GateOutcome.PASSED if ok else GateOutcome.FAILED,
        baseline,
        current,
        f"{current:.3f} vs baseline {baseline:.3f} (ceiling {ceiling:.3f})",
    )


def compare_against_baseline(
    baseline: Mapping[str, Any],
    current: Mapping[str, Any],
    *,
    recall_tolerance: float = RECALL_TOLERANCE,
    cost_tolerance: float = COST_TOLERANCE,
) -> GateResult:
    """Compare a run summary against a baseline summary.

    Both arguments are :func:`summarise_run` output. The dimensions:

    * **total_entities** — the recall floor. The binding half of the gate.
    * **documents_with_entities** — a document that used to yield entities and
      now yields none is a regression even when the totals hold, because the
      totals can be carried by one verbose document.
    * **over_generation_rate** / **abstain_rate** — the cost half. Must not rise.

    Returns ``inconclusive`` when every dimension skipped, which is what a
    baseline predating the measurement looks like. A caller must treat that as
    "nothing was verified", never as a pass.
    """
    dimensions = [
        _compare_floor(
            "total_entities",
            _num(baseline.get("total_entities")),
            _num(current.get("total_entities")),
            recall_tolerance,
        ),
        _compare_floor(
            "documents_with_entities",
            _num(baseline.get("documents_with_entities")),
            _num(current.get("documents_with_entities")),
            recall_tolerance,
        ),
        _compare_ceiling(
            "over_generation_rate",
            _num(baseline.get("over_generation_rate")),
            _num(current.get("over_generation_rate")),
            cost_tolerance,
        ),
        _compare_ceiling(
            "abstain_rate",
            _num(baseline.get("abstain_rate")),
            _num(current.get("abstain_rate")),
            cost_tolerance,
        ),
    ]
    missing = dict(current.get("documents_missing_counters") or {})
    evaluated = [d for d in dimensions if d.outcome is not GateOutcome.SKIPPED]
    if not evaluated:
        return GateResult(
            passed=False,
            dimensions=dimensions,
            inconclusive=True,
            missing_counters=missing,
        )
    return GateResult(
        passed=not any(d.failed for d in dimensions),
        dimensions=dimensions,
        missing_counters=missing,
    )


def _num(value: Any) -> Optional[float]:
    """Coerce to float, or ``None`` for anything that is not a number.

    A baseline is a JSON file somebody may hand-edit; a string where a number
    belongs must degrade to "not measured" rather than raise inside a gate.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
