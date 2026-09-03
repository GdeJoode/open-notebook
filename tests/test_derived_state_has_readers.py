"""PC.1b — every piece of derived state names a reader, or an owner phase.

**What this guards.** The extraction chain repeatedly computes something — a
measurement, a judgement, an attempt — carries it faithfully, and hands it to a
place nothing reads. Track N found six such boundaries, each one phase after the
phase that introduced it; the PC.1b inventory found roughly twenty.

**Why this shape rather than a run object.** `FilteredResult` is already a typed
run-state carrier, and on that one object the payload fields have 4/4/2/2 readers
while the derived-state fields have 0/0/0/0. A carrier was not the missing thing, so
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
from typing import Dict, List, Set

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_MODEL = (
    REPO_ROOT / "packages/shared/src/shared/models/extraction.py"
)

#: Fields on the extraction result models that carry the PAYLOAD rather than
#: derived state. They are out of scope: nobody has ever lost an entity list at a
#: handoff, and the reader counts confirm it (4/4/2/2 against 0/0/0/0).
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


def _model_classes(tree: ast.AST) -> Dict[str, List[str]]:
    """Map class name -> annotated field names, for every class in a module."""
    found: Dict[str, List[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        fields = [
            stmt.target.id
            for stmt in node.body
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
        ]
        if fields:
            found[node.name] = fields
    return found


def _derived_state_fields(source: str) -> Set[str]:
    """Every annotated field on the extraction models that is not payload."""
    classes = _model_classes(ast.parse(source))
    fields: Set[str] = set()
    for name, declared in classes.items():
        if name not in {"ExtractionResult", "FilteredResult"}:
            continue
        fields.update(f for f in declared if f not in _PAYLOAD_FIELDS)
    return fields


def _reader_count(field: str) -> int:
    """Production readers of ``.field``, excluding tests and the model itself.

    Uses git grep so the search respects .gitignore and never walks .venv or
    node_modules — a plain filesystem walk here was measured taking minutes.
    """
    result = subprocess.run(
        ["git", "grep", "-l", "--", f"\\.{field}\\b"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    paths = [p for p in result.stdout.splitlines() if p.strip()]
    return len(
        [
            p
            for p in paths
            if p.endswith(".py")
            and "/tests/" not in p
            and not Path(p).name.startswith("test_")
            and not p.endswith("shared/models/extraction.py")
        ]
    )


@pytest.fixture(scope="module")
def derived_fields() -> Set[str]:
    return _derived_state_fields(EXTRACTION_MODEL.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Controls — each fails on its own if the machinery breaks
# ---------------------------------------------------------------------------


def test_the_scanner_reaches_the_model():
    """Control 1: path resolution. Without this, a moved or renamed model file
    makes the sweep below find nothing and pass — the vacuity failure that made
    PC.1's first attempt fully green and completely inert.
    """
    assert EXTRACTION_MODEL.exists(), f"{EXTRACTION_MODEL} not found"
    assert "class FilteredResult" in EXTRACTION_MODEL.read_text(encoding="utf-8")


def test_the_scanner_finds_a_field_it_should(derived_fields: Set[str]):
    """Control 2: detection. `concept_alignment_report` is derived state and is
    still declared, so the AST walk must see it. If the walk breaks, this fails
    rather than the sweep reporting an empty set.
    """
    assert derived_fields, "no derived-state fields detected at all"
    assert "concept_alignment_report" in derived_fields


def test_payload_fields_are_excluded_deliberately(derived_fields: Set[str]):
    """Control 3: the exclusion list is doing work rather than swallowing the set.
    `entities` must be excluded, `concept_alignment_report` must not be.
    """
    assert "entities" not in derived_fields
    assert "merged_entity_groups" not in derived_fields
    assert _PAYLOAD_FIELDS & derived_fields == set()


def test_a_planted_dead_field_is_caught():
    """Control 4, the one that makes the promise real.

    Runs the SAME detector over a source string containing a field that does not
    exist anywhere in the repo. No file is written. If the detector is ever
    narrowed to a hard-coded list of today's names, this fails — which is the
    failure mode a guard like this dies of.
    """
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
    assert _reader_count("a_freshly_added_report_nobody_reads") == 0


def test_the_reader_count_can_tell_read_from_unread():
    """Control 5: the counter itself. A counter that returns 0 for everything
    would make the guard below pass trivially; one that returns >0 for everything
    would make it fail loudly and get deleted. Pin both ends against fields whose
    status PC.1b measured.
    """
    assert _reader_count("merged_entity_groups") > 0
    assert _reader_count("a_field_that_certainly_does_not_exist_anywhere") == 0


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_every_derived_state_field_has_a_reader_or_an_owner(derived_fields: Set[str]):
    """The invariant. A field that measures something must be read by something.

    When this fails, the fix is one of three — and "add it to the allow-list" is
    only the third:

    1. Give it a reader. It was probably measured for a purpose.
    2. Delete it. N.5b ruled out shipping a producer that survives by accident.
    3. Add it to `_OWNED_WITHOUT_READER` **and** add a row to
       `docs/tracks/PC-pipeline-coherence/handoff-inventory.md` naming the phase
       that will do (1) or (2). That is a deliberate, reviewable act.
    """
    orphans = sorted(
        field
        for field in derived_fields
        if field not in _OWNED_WITHOUT_READER and _reader_count(field) == 0
    )
    assert not orphans, (
        "derived state with no reader and no owner phase: "
        + ", ".join(orphans)
        + ". Give it a reader, delete it, or record an owner in "
        "docs/tracks/PC-pipeline-coherence/handoff-inventory.md and "
        "_OWNED_WITHOUT_READER."
    )


def test_the_allow_list_does_not_outlive_its_entries(derived_fields: Set[str]):
    """An allow-list entry for a field that no longer exists is stale permission.
    It would silently cover a future field that happens to reuse the name.
    """
    stale = sorted(set(_OWNED_WITHOUT_READER) - derived_fields)
    assert not stale, f"_OWNED_WITHOUT_READER names fields that no longer exist: {stale}"
