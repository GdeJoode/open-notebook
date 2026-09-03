"""PC.1b — every piece of derived state names a reader, or an owner phase.

**What this guards.** The extraction chain repeatedly computes something — a
measurement, a judgement, an attempt — carries it faithfully, and hands it to a
place nothing reads. Track N found six such boundaries, each one phase after the
phase that introduced it; the PC.1b inventory found roughly twenty.

**Why this shape rather than a run object.** `FilteredResult` is already a typed
run-state carrier, and on that one object every payload field is read in at least
one production file while every derived-state field is read in none. A carrier was not the missing thing, so
this is not a third carrier: it is the rule, made able to fail. A producer names its
consumer or it goes — the generalisation of what N.5b decided about the Hearst miner.

**The three ways a guard like this fails**, in the order this project has been
burned by them, and what each control below does about it:

1. *It runs nowhere.* It lives in the root ``tests/`` (the `testpaths` in
   ``pyproject.toml``) and `.github/workflows/unit-guards.yml` runs it on every PR.
   Before PC.1b, no CI job ran any Python unit suite at all.
2. *It scans nothing and passes vacuously.* `test_the_scanner_reaches_the_model`
   and `test_the_scanner_finds_a_field_it_should` fail if resolution or detection
   breaks, rather than the sweep quietly finding zero.
3. *It matches a literal that a rename evades.* Detection is by AST over the model
   definition, and `test_a_planted_dead_field_is_caught` runs the same detector over
   an in-test source string containing a fresh dead field. That is what makes "a
   future dead field would be caught" a claim rather than a hope.

**What it does NOT cover**, so a reader does not assume otherwise: a *consumer with
no producer* — `notebook_event{extension_suggested, schema_mismatch}` is polled by
the frontend and has never been written. That direction needs a different scan and
is recorded in the inventory as a follow-up.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_MODEL = (
    REPO_ROOT / "packages/shared/src/shared/models/extraction.py"
)

#: Fields on the extraction result models that carry the PAYLOAD rather than
#: derived state. They are out of scope: nobody has ever lost an entity list at a
#: handoff, and the reader counts confirm it: each is read somewhere, each
#: derived-state field was read nowhere.
_PAYLOAD_FIELDS = {
    "entities",
    "relations",
    "metadata",
    "removed_entities",
    "merged_entity_groups",
    "match_candidates",
    "predicted_edges",
}

#: Derived-state fields allowed to have no reader TODAY, each with the phase that
#: owns deciding its fate. An entry here is a promise recorded in
#: `docs/tracks/PC-pipeline-coherence/handoff-inventory.md`, not an exemption:
#: the owning phase either gives the field a reader or deletes it.
#:
#: Adding a name here is the deliberate act this guard exists to force. Adding one
#: without a row in the inventory is the thing to catch in review.
_OWNED_WITHOUT_READER: Dict[str, str] = {
    # PC.1b found four derived-state fields with zero readers and split them by
    # whether a named phase actually wants the measurement:
    #
    #   deleted — `linked_entities` (a second copy of URIs that already travel in
    #   each entity's `properties`, which IS persisted) and
    #   `llm_verification_results` (never written by anything at all).
    #
    #   kept, with an owner — these two. Each is a measurement a named phase
    #   needs, and deleting it one phase early would be churn.
    "kg_resolution_report": "PC.3 — its AC needs a measured figure for how many "
    "rows cross-document resolution collapses",
    "validation_report": "PC.6 — stage 11 is inert because no production call "
    "site passes an ontology; PC.6 owns making that visible",
}


#: Attribute names common enough elsewhere in the codebase that a read of them
#: cannot be attributed to these models by name alone. A derived-state field must
#: not use one, because the guard below could not tell its readers from anybody
#: else's — measured today: `confidence` reads in 27 files, `status` 26,
#: `metrics` 9. A review planted `metrics` and `confidence` as dead fields and
#: the first version of this guard stayed green.
#:
#: This converts an undetectable case into a naming rule, which is the honest
#: trade: the guard cannot do type inference, so it refuses names it would have
#: to guess about.
_AMBIGUOUS_FIELD_NAMES = frozenset(
    {
        "confidence",
        "status",
        "metrics",
        "report",
        "errors",
        "summary",
        "result",
        "results",
        "data",
        "config",
        "stats",
        "count",
        "name",
        "text",
        "value",
        "items",
    }
)


def _result_model_names(tree: ast.AST) -> Set[str]:
    """`ExtractionResult` and everything that inherits from it, transitively.

    A review planted `class ResolvedResult(FilteredResult)` with a dead field and
    the first version — which filtered on a hard-coded pair of class names — was
    green. Any subclass carries the same handoff risk, so the set is derived.
    """
    by_name: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            by_name[node.name] = [
                b.id for b in node.bases if isinstance(b, ast.Name)
            ]
    models = {"ExtractionResult"}
    changed = True
    while changed:
        changed = False
        for name, bases in by_name.items():
            if name not in models and models & set(bases):
                models.add(name)
                changed = True
    return models


def _derived_state_fields(source: str) -> Dict[str, str]:
    """``{field: declaring class}`` for every non-payload field on the models."""
    tree = ast.parse(source)
    models = _result_model_names(tree)
    fields: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in models:
            continue
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.target.id not in _PAYLOAD_FIELDS:
                    fields[stmt.target.id] = node.name
    return fields


def _production_sources() -> List[Path]:
    """Every production .py file git tracks, tests and the model excluded."""
    result = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    return [
        REPO_ROOT / p
        for p in result.stdout.splitlines()
        if p.endswith(".py")
        # `startswith` as well as `in`: the repo's own root suite is `tests/…`,
        # which contains no leading slash and slipped through the first version.
        # Control 2 below caught it, which is what that control is for.
        and not p.startswith("tests/")
        and "/tests/" not in p
        and not Path(p).name.startswith("test_")
        and not p.endswith("shared/models/extraction.py")
    ]


def _count_reads(field: str, sources: Optional[List[Path]] = None) -> int:
    """How many production files READ ``.field`` — not write it, not name it.

    The rule this guard states is "a producer names its CONSUMER". The first
    version counted `git grep '\.field'`, which matches a write as readily as a
    read: a review planted a single line, `result.zzz_orphan_writeonly_report =
    {"n": 1}`, and the guard went green. The four orphans it did find were found
    only incidentally — all four were written through constructor kwargs, which
    carry no leading dot.

    So: parse, and count `ast.Attribute` nodes in a **Load** context only. A
    file that merely assigns the field is a second producer, not a consumer.
    """
    seen = 0
    for path in sources if sources is not None else _production_sources():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            continue
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Attribute)
                and node.attr == field
                and isinstance(node.ctx, ast.Load)
            ):
                seen += 1
                break
    return seen


@pytest.fixture(scope="module")
def derived_fields() -> Dict[str, str]:
    return _derived_state_fields(EXTRACTION_MODEL.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def production_sources() -> List[Path]:
    """Parsed once. Re-walking the tree per field made the suite minutes long."""
    return _production_sources()


# ---------------------------------------------------------------------------
# Controls — each fails on its own if the machinery breaks
# ---------------------------------------------------------------------------


def test_the_scanner_reaches_the_model():
    """Control 1: path resolution. Without it, a moved model file makes the sweep
    find nothing and pass — the vacuity failure that made PC.1's first attempt
    fully green and completely inert.
    """
    assert EXTRACTION_MODEL.exists(), f"{EXTRACTION_MODEL} not found"
    assert "class FilteredResult" in EXTRACTION_MODEL.read_text(encoding="utf-8")


def test_the_scanner_walks_the_production_tree(production_sources: List[Path]):
    """Control 2: the file list. An empty or tiny list would make every field
    look like an orphan, or — with the `or` the other way — like a reader.
    """
    assert len(production_sources) > 200, len(production_sources)
    assert any(p.name == "workflow.py" for p in production_sources)
    assert not any("/tests/" in str(p) for p in production_sources)


def test_the_scanner_finds_a_field_it_should(derived_fields: Dict[str, str]):
    """Control 3: detection. `concept_alignment_report` is derived state and is
    still declared, so the AST walk must see it and attribute it to its class.
    """
    assert derived_fields, "no derived-state fields detected at all"
    assert derived_fields.get("concept_alignment_report") == "FilteredResult"


def test_payload_fields_are_excluded_deliberately(derived_fields: Dict[str, str]):
    """Control 4: the exclusion list does work rather than swallowing the set."""
    assert "entities" not in derived_fields
    assert "merged_entity_groups" not in derived_fields
    assert _PAYLOAD_FIELDS & set(derived_fields) == set()


def test_a_planted_dead_field_is_caught():
    """Control 5: the detector is not a hard-coded list of today's names."""
    planted = '''
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    entities: list = []
    metadata: Dict[str, Any] = {}


class FilteredResult(ExtractionResult):
    predicted_edges: list = []
    a_freshly_added_report_nobody_reads: Optional[Dict[str, Any]] = None
'''
    detected = _derived_state_fields(planted)
    assert "a_freshly_added_report_nobody_reads" in detected
    assert "predicted_edges" not in detected  # payload, correctly excluded
    assert _count_reads("a_freshly_added_report_nobody_reads") == 0


def test_a_dead_field_on_a_SUBCLASS_is_caught():
    """Control 6, from review finding 2b. The first version filtered on a
    hard-coded `{"ExtractionResult", "FilteredResult"}`, so a new subclass was
    invisible — a planted `class ResolvedResult(FilteredResult)` with a dead
    field left the guard green. Inheritance is now resolved transitively.
    """
    planted = '''
from typing import Any, Dict, Optional
from pydantic import BaseModel


class ExtractionResult(BaseModel):
    entities: list = []


class FilteredResult(ExtractionResult):
    predicted_edges: list = []


class ResolvedResult(FilteredResult):
    cross_document_resolution_report: Optional[Dict[str, Any]] = None
'''
    detected = _derived_state_fields(planted)
    assert detected.get("cross_document_resolution_report") == "ResolvedResult"


def test_a_write_is_not_mistaken_for_a_read(tmp_path: Path):
    """Control 7, from review finding 2c — the decisive one.

    The rule says a producer names its CONSUMER. The first version counted
    `git grep '\.field'`, which matches an assignment as readily as a read, so a
    single planted line — `result.zzz_orphan_writeonly_report = {"n": 1}` — made
    the guard green. The four orphans it did find were found only incidentally:
    all four were written through constructor kwargs, which carry no leading dot.
    """
    writer = tmp_path / "writer.py"
    writer.write_text("def f(result):\n    result.zzz_only_written = {'n': 1}\n")
    reader = tmp_path / "reader.py"
    reader.write_text("def g(result):\n    return result.zzz_also_read\n")

    assert _count_reads("zzz_only_written", [writer, reader]) == 0
    assert _count_reads("zzz_also_read", [writer, reader]) == 1


def test_the_counter_can_tell_read_from_unread(production_sources: List[Path]):
    """Control 8: the counter itself. All-zero would make the guard pass
    trivially; all-nonzero would make it fail loudly and get deleted.
    """
    assert _count_reads("concept_alignment_report", production_sources) > 0
    assert _count_reads("zzz_certainly_not_an_attribute_anywhere", production_sources) == 0


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_no_derived_state_field_uses_an_ambiguous_name(derived_fields: Dict[str, str]):
    """A field whose name is common elsewhere cannot be checked by this guard.

    From review finding 2a: `confidence` is read in 27 production files and
    `metrics` in 9, so a dead field with either name would look well-read. The
    guard cannot do type inference, so it refuses the names it would have to
    guess about — and says which, rather than silently passing them.
    """
    ambiguous = sorted(set(derived_fields) & _AMBIGUOUS_FIELD_NAMES)
    assert not ambiguous, (
        "derived-state fields with names too common to attribute: "
        + ", ".join(ambiguous)
        + ". Rename them (e.g. `metrics` -> `extraction_metrics_report`) so a "
        "read of the name can only mean this model."
    )


def test_every_derived_state_field_has_a_reader_or_an_owner(
    derived_fields: Dict[str, str], production_sources: List[Path]
):
    """The invariant. A field that measures something must be READ by something.

    When this fails, the fix is one of three — and "add it to the allow-list" is
    only the third:

    1. Give it a reader. It was probably measured for a purpose.
    2. Delete it. N.5b ruled out shipping a producer that survives by accident.
    3. Add it to `_OWNED_WITHOUT_READER` **and** a row to
       `docs/tracks/PC-pipeline-coherence/handoff-inventory.md` naming the phase
       that will do (1) or (2). That is a deliberate, reviewable act.
    """
    orphans = sorted(
        field
        for field in derived_fields
        if field not in _OWNED_WITHOUT_READER
        and _count_reads(field, production_sources) == 0
    )
    assert not orphans, (
        "derived state with no reader and no owner phase: "
        + ", ".join(orphans)
        + ". Give it a reader, delete it, or record an owner in "
        "docs/tracks/PC-pipeline-coherence/handoff-inventory.md and "
        "_OWNED_WITHOUT_READER."
    )


def test_the_allow_list_does_not_outlive_its_entries(derived_fields: Dict[str, str]):
    """An allow-list entry for a field that no longer exists is stale permission:
    it would silently cover a future field that reuses the name.
    """
    stale = sorted(set(_OWNED_WITHOUT_READER) - set(derived_fields))
    assert not stale, f"_OWNED_WITHOUT_READER names fields that no longer exist: {stale}"
