"""Track N.5d — the gate's glue, which a review pointed out had no tests at all.

The arithmetic half of this gate is unit-tested on every suite run because a rule
that executes once a fortnight is a rule nobody trusts. The script around it was
not, and two fixes from the review live only there: the refusal to record a
baseline from a run that measured nothing, and the refusal to accept a file that
is not harness output. Neither is a guard that cannot fail — they are behaviours a
mutation cannot be applied to, which is the same gap seen from further away.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "n_extraction_gate.py"
_spec = importlib.util.spec_from_file_location("n_extraction_gate", _SCRIPT)
gate_script = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(gate_script)


def _write(tmp_path: Path, name: str, payload) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return path


def _run_file(tmp_path: Path, entities, pdf="a.pdf", counters=None):
    result = {"entity_count": entities, "relation_count": 0}
    if counters:
        result["counters"] = counters
    return _write(
        tmp_path, "run.json", {"documents": [{"pdf": pdf, "result": result}]}
    )


class TestItRefusesInputItCannotUse:
    def test_a_file_with_no_result_block_is_refused(self, tmp_path):
        """Handing it the BASELINE by mistake otherwise reads as a run of one
        document that extracted nothing, and the gate reports a catastrophic
        regression instead of "wrong file" — a confusing failure that looks
        exactly like a real one.
        """
        baseline_shaped = _write(
            tmp_path, "baseline.json", {"total_entities": 124, "documents": 7}
        )
        with pytest.raises(gate_script.GateRefusal) as exc:
            gate_script._load_documents(baseline_shaped)
        assert "does not look like harness output" in str(exc.value)

    def test_a_document_whose_parse_failed_is_still_accepted(self, tmp_path):
        """The vacuity guard for the test above. A failed parse still carries a
        `result` with zero counts, so the refusal must discriminate "wrong file"
        from "document that produced nothing" — otherwise it would reject the
        `Bennett_test.pdf` shape the baseline deliberately includes.
        """
        run = _run_file(tmp_path, 0)
        assert len(gate_script._load_documents(run)) == 1

    def test_recording_a_baseline_that_measured_nothing_is_refused(self, tmp_path):
        run = _run_file(tmp_path, 0)
        baseline = tmp_path / "baseline.json"
        with pytest.raises(gate_script.GateRefusal) as exc:
            gate_script.main(["--run", str(run), "--baseline", str(baseline), "--write-baseline"]
            )
        assert "a zero floor holds nothing up" in str(exc.value)
        assert not baseline.exists()

    def test_the_refusal_can_be_overridden_deliberately(self, tmp_path):
        run = _run_file(tmp_path, 0)
        baseline = tmp_path / "baseline.json"
        assert (
            gate_script.main([
                    "--run", str(run), "--baseline", str(baseline),
                    "--write-baseline", "--allow-empty-baseline",
                ]
            )
            == 0
        )
        assert json.loads(baseline.read_text())["total_entities"] == 0


class TestTheExitCodes:
    """A review found that `raise SystemExit(<str>)` yields 1, which this script
    reserves for "regression" — so a CI wrapper could not tell "you regressed"
    from "I would not act on that file".
    """

    def test_a_passing_comparison_exits_zero(self, tmp_path):
        run = _run_file(tmp_path, 100)
        baseline = _write(
            tmp_path, "baseline.json",
            {"total_entities": 100, "documents_with_entities": 1},
        )
        assert (
            gate_script.main(["--run", str(run), "--baseline", str(baseline)]
            )
            == 0
        )

    def test_a_regression_exits_one(self, tmp_path):
        run = _run_file(tmp_path, 10)
        baseline = _write(
            tmp_path, "baseline.json",
            {"total_entities": 100, "documents_with_entities": 1},
        )
        assert (
            gate_script.main(["--run", str(run), "--baseline", str(baseline)]
            )
            == 1
        )

    def test_an_inconclusive_comparison_exits_two(self, tmp_path):
        run = _run_file(tmp_path, 10)
        baseline = _write(
            tmp_path, "baseline.json",
            {"total_entities": None, "documents_with_entities": None},
        )
        assert (
            gate_script.main(["--run", str(run), "--baseline", str(baseline)]
            )
            == 2
        )

    def test_the_refusal_code_is_not_the_regression_code(self):
        assert gate_script.REFUSED == 3
        assert gate_script.REFUSED != 1


class TestTheProvenanceDescribesThisRun:
    """A review executed a re-record and found the note correctly superseded
    while the metadata beside it still described the PREVIOUS measurement — the
    same class of defect one field over.
    """

    STALE = {
        "harness": "scripts/n_pipeline_review_run.py",
        "measured": "2026-09-02",
        "documents": ["old_one.pdf", "old_two.pdf"],
        "sources": ["claudedocs/pipeline-review-corpus.json"],
        "note": "the two rates are null because the merge discarded the counters",
    }

    def _rerecord(self, tmp_path, counters=None):
        run = _run_file(tmp_path, 42, pdf="dir/new_document.pdf", counters=counters)
        baseline = _write(
            tmp_path, "baseline.json", {"total_entities": 1, "_provenance": self.STALE}
        )
        gate_script.main(["--run", str(run), "--baseline", str(baseline), "--write-baseline"]
        )
        return json.loads(baseline.read_text())["_provenance"]

    def test_the_document_list_is_this_run_s(self, tmp_path):
        provenance = self._rerecord(tmp_path)
        assert provenance["documents"] == ["new_document.pdf"]
        assert "old_one.pdf" not in provenance["documents"]

    def test_stale_fields_do_not_survive(self, tmp_path):
        provenance = self._rerecord(tmp_path)
        assert provenance["measured"] != "2026-09-02"
        assert "sources" not in provenance

    def test_the_prose_note_does_survive_when_it_is_still_true(self, tmp_path):
        """It is the one field a re-record cannot regenerate, and it explains
        something a reader cannot infer from the numbers.
        """
        provenance = self._rerecord(tmp_path)
        assert self.STALE["note"] in provenance["note"]

    def test_a_note_that_has_become_false_is_marked_superseded(self, tmp_path):
        provenance = self._rerecord(
            tmp_path,
            counters={
                "entities_extracted": 10, "entities_kept": 4,
                "abstained_chunks": 2, "chunk_count": 10,
            },
        )
        assert provenance["note"].startswith("Superseded:")
        assert "over_generation_rate" in provenance["note"]
        # ...and the old text is kept behind it rather than discarded.
        assert self.STALE["note"] in provenance["note"]
