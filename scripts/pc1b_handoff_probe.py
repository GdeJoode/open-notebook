"""Track PC.1b — measure the handoffs against a live database.

Answers three questions, none of which depends on a particular corpus:

1. **Which derived-state fields still have no reader?** The same AST scan the
   invariant test uses, printed rather than asserted, so a human can see the list
   move rather than only learn that it did not grow.
2. **Is the soft-nudge banner reachable?** Counts `notebook_event` rows by type
   and, for every source with Pass-1 rows, recomputes what `_decide_soft_nudge`
   would say. Before W1 landed this printed "0 events, 5 verdicts" — a producer
   and a consumer that had never met.
3. **How far apart are extracted and persisted?** The gap the pipeline review
   measured (124 entities against 117 rows) and that the system had no way to
   report until W3.

Run inside the container, or with SURREAL_DATABASE pointing at the target::

    SURREAL_DATABASE=staging uv run --project apps/app-main \\
        python scripts/pc1b_handoff_probe.py

**On an empty graph it prints zeros and says so.** That is honest and it is not a
pass — the same discipline N.5d's gate uses for a baseline it cannot compare.
Read-only: it writes nothing.
"""

from __future__ import annotations

import ast
import asyncio
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("SURREAL_DATABASE", "staging")

sys.path.insert(0, str(REPO_ROOT / "packages/shared/src"))


# ---------------------------------------------------------------------------
# 1. Derived state without a reader — the same scan the invariant test performs
# ---------------------------------------------------------------------------

_PAYLOAD_FIELDS = {
    "entities",
    "relations",
    "metadata",
    "removed_entities",
    "merged_entity_groups",
    "match_candidates",
    "predicted_edges",
}


def _derived_fields() -> Set[str]:
    model = REPO_ROOT / "packages/shared/src/shared/models/extraction.py"
    tree = ast.parse(model.read_text(encoding="utf-8"))
    found: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name not in {"ExtractionResult", "FilteredResult"}:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.target.id not in _PAYLOAD_FIELDS:
                    found.add(stmt.target.id)
    return found


def _readers(field: str) -> List[str]:
    result = subprocess.run(
        ["git", "grep", "-l", "--", f"\\.{field}\\b"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return [
        p
        for p in result.stdout.splitlines()
        if p.endswith(".py")
        and "/tests/" not in p
        and not Path(p).name.startswith("test_")
        and not p.endswith("shared/models/extraction.py")
    ]


def report_derived_state() -> None:
    print("== derived state and its readers")
    fields = sorted(_derived_fields())
    if not fields:
        print("   no derived-state fields declared — check the scanner, not the code")
        return
    for field in fields:
        readers = _readers(field)
        mark = "orphan" if not readers else f"{len(readers)} reader(s)"
        print(f"   {field:28s} {mark}")


# ---------------------------------------------------------------------------
# 2 & 3. The live measurements
# ---------------------------------------------------------------------------


async def report_soft_nudge() -> None:
    from ontology_extraction.multi_schema_orchestrator import _decide_soft_nudge
    from surrealdb_service.connection import execute_query

    print("\n== the soft-nudge banner")
    events = await execute_query("SELECT event_type FROM notebook_event;", {})
    by_type = Counter(str(r.get("event_type")) for r in events)
    print(f"   notebook_event rows: {len(events)} {dict(by_type) or ''}")

    rows = await execute_query("SELECT source, coverage_pct FROM pass1_results;", {})
    if not rows:
        print("   no pass1_results rows — nothing to compare, and that is not a pass")
        return

    best: Dict[str, float] = {}
    for row in rows:
        source = str(row.get("source"))
        best[source] = max(best.get(source, 0.0), float(row.get("coverage_pct") or 0))

    verdicts = Counter(
        getattr(_decide_soft_nudge(cov), "value", "?") for cov in best.values()
    )
    would_fire = sum(count for name, count in verdicts.items() if name != "none")
    print(f"   sources with Pass-1 coverage: {len(best)}")
    print(f"   verdicts a run would compute: {dict(verdicts)}")
    print(f"   of which would raise a banner: {would_fire}")
    if would_fire and not events:
        print("   -> a producer and a consumer that have never met (the PC.1b case)")


async def report_extracted_vs_persisted() -> None:
    from surrealdb_service.connection import execute_query

    print("\n== extracted against persisted")
    rows = await execute_query(
        "SELECT source_id, entity_count, metadata FROM extraction_result;", {}
    )
    if not rows:
        print("   no extraction_result rows — nothing measured, and that is not a pass")
        return

    with_persisted = [
        r
        for r in rows
        if isinstance((r.get("metadata") or {}).get("persisted"), dict)
    ]
    print(f"   extraction_result rows: {len(rows)}")
    print(
        f"   runs carrying a `persisted` block: {len(with_persisted)} of {len(rows)}"
    )

    # Deliberately NOT compared against `SELECT count() FROM entity`. That table
    # accumulates across every run ever made, including archived and reference
    # rows from other tracks, so the difference between it and one run's
    # `entity_count` is not the extracted-versus-persisted gap — it is mostly
    # history. The whole point of W3 is that the gap must be measured PER RUN, by
    # the run itself, which is what the block below reports once runs carry it.
    if not with_persisted:
        print(
            "   -> no run has recorded what it WROTE; only what the LLM produced. "
            "W3 fills this from the next extraction onward, and until one has run "
            "the gap is not measurable — which is not the same as being zero."
        )
        return

    for row in with_persisted:
        persisted = row["metadata"]["persisted"]
        claimed = int(row.get("entity_count") or 0)
        upserted = int(persisted.get("entities_upserted") or 0)
        failed = int(persisted.get("entities_failed") or 0)
        print(
            f"   {str(row.get('source_id'))[-12:]}: extracted {claimed}, "
            f"upserted {upserted}, failed {failed}, gap {claimed - upserted}"
        )


async def main() -> int:
    report_derived_state()
    await report_soft_nudge()
    await report_extracted_vs_persisted()
    print("\nRead-only: nothing was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
