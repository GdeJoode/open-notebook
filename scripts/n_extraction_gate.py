"""Track N.5d — run the extraction regression gate against a measured run.

Splits into two steps on purpose. Measuring needs Ollama, a database and minutes
per document; deciding is arithmetic and lives in
``shared.regression.extraction_gate``, where the suite exercises it on every run.
This script is only the glue.

Typical use, after re-measuring with the harness::

    SURREAL_DATABASE=staging uv run --project apps/app-main \\
        python scripts/n_pipeline_review_run.py --pdf <doc> --json out.json
    uv run python scripts/n_extraction_gate.py --run out.json

To record a new baseline once a run is known-good::

    uv run python scripts/n_extraction_gate.py --run out.json --write-baseline

Exit codes: 0 pass, 1 regression, 2 inconclusive (nothing could be compared),
3 refused (the input is not usable as what it was offered as). An inconclusive
result is deliberately NOT 0 — a gate that exits green having verified nothing is
the failure this whole design is built against — and a refusal is deliberately not
1, so a CI wrapper can tell "you regressed" from "I would not act on that file".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "packages/shared/src"))

from shared.regression import compare_against_baseline, summarise_run  # noqa: E402

DEFAULT_BASELINE = REPO_ROOT / "tests/regression/n_extraction_baseline.json"

#: Exit code for "I will not act on this input". Distinct from 1 (a regression)
#: because a CI wrapper reacts differently to the two, and a review found that
#: `raise SystemExit(<str>)` had quietly merged them.
REFUSED = 3


class GateRefusal(Exception):
    """The input is not usable as the thing it was offered as."""


def _load_documents(path: Path) -> list:
    """Accept either a multi-document corpus file or a single-document run.

    Refuses a file in which NOTHING carries a ``result`` block. Handing this the
    BASELINE file by mistake otherwise reads as a run of one document that
    extracted nothing, and the gate reports a catastrophic regression instead of
    "wrong file" — a confusing failure that looks exactly like a real one. A
    document whose parse failed still has a ``result`` with zero counts, so this
    discriminates the two.
    """
    payload = json.loads(path.read_text())
    documents = payload.get("documents")
    documents = documents if isinstance(documents, list) else [payload]
    if not any(isinstance(d, dict) and "result" in d for d in documents):
        raise GateRefusal(
            f"{path} carries no document with a `result` block — this does not "
            "look like harness output. (Did you pass the baseline file?)"
        )
    return documents


def main(argv: Optional[list] = None) -> int:
    """Parse ``argv`` (default ``sys.argv[1:]``) and run one comparison.

    Takes its arguments rather than reading them so the tests can drive it
    directly. A review pointed out that this glue had no tests at all while two of
    the review's own fixes lived only here — not a guard that cannot fail, but a
    behaviour a mutation cannot be applied to, which is the same gap seen from
    further away.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", required=True, type=Path, help="JSON written by the harness"
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--write-baseline",
        action="store_true",
        help="record this run as the new baseline instead of comparing",
    )
    parser.add_argument(
        "--allow-empty-baseline",
        action="store_true",
        help="record a baseline even though the run produced no entities",
    )
    args = parser.parse_args(argv)

    documents = _load_documents(args.run)
    current = summarise_run(documents)

    if args.write_baseline:
        # Refuse a run that measured nothing. A review reached this by accident:
        # with Ollama down every document still produces a `result` block, so the
        # file looks like harness output, and a baseline of zero entities makes
        # every later floor `0 * (1 - tolerance)` — permanently green while
        # reporting `inconclusive=False`. The gate now skips a zero floor too, but
        # refusing to RECORD one is the cheaper place to stop it.
        if not current.get("total_entities") and not args.allow_empty_baseline:
            raise GateRefusal(
                f"{args.run} measured {current.get('total_entities', 0)} entities "
                f"across {current.get('documents', 0)} documents. Refusing to "
                "record that as a baseline — a zero floor holds nothing up. Fix "
                "the run first, or pass --allow-empty-baseline if this is "
                "genuinely what the corpus produces."
            )

        previous = (
            json.loads(args.baseline.read_text()) if args.baseline.exists() else {}
        )
        # Only the fields that survive a re-record are carried forward. A review
        # caught the rest: re-recording from a different corpus kept
        # `_provenance.documents` listing the PREVIOUS seven filenames, plus the
        # old `measured` date and `sources` list, beside a note that had been
        # correctly superseded. The note was honest and the metadata around it
        # described the previous measurement — the same class of defect one field
        # over. So describe THIS run, and keep only the prose.
        previous_provenance = dict(previous.get("_provenance") or {})
        provenance: Dict[str, Any] = {
            "harness": previous_provenance.get(
                "harness", "scripts/n_pipeline_review_run.py"
            ),
            "measured_from": str(args.run),
            "measured": _today(),
            "documents": _document_names(documents),
        }
        # Carry the previous note forward, but do NOT carry a note that has become
        # false. The checked-in one explains why two dimensions are null; once a
        # run measures them, that explanation describes a state that no longer
        # exists, and a stale explanation is worse than none.
        measured_now = [
            k
            for k in ("over_generation_rate", "abstain_rate")
            if current.get(k) is not None
        ]
        note = previous_provenance.get("note")
        if note:
            provenance["note"] = note
        if measured_now and note:
            provenance["note"] = (
                "Superseded: this baseline measures "
                + ", ".join(measured_now)
                + ". The previous note explained why they were null; it no longer "
                "applies. Previous note: " + note
            )
        current["_provenance"] = provenance
        args.baseline.write_text(json.dumps(current, indent=2) + "\n")
        print(f"baseline written to {args.baseline}")
        return 0

    baseline = json.loads(args.baseline.read_text())
    result = compare_against_baseline(baseline, current)
    print(f"extraction regression gate — {args.run.name} vs {args.baseline.name}")
    print(result.report())

    if result.inconclusive:
        return 2
    return 0 if result.passed else 1


def _today() -> str:
    from datetime import date

    return date.today().isoformat()


def _document_names(documents: list) -> list:
    """The filenames THIS run measured, for the provenance block."""
    names = []
    for doc in documents:
        pdf = str((doc or {}).get("pdf") or "")
        names.append(pdf.split("/")[-1] if pdf else "?")
    return names


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except GateRefusal as refusal:
        print(refusal)
        raise SystemExit(REFUSED) from None
