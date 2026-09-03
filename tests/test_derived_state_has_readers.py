"""PC.1b — every piece of derived state declares who reads it.

**What this guards.** The extraction chain repeatedly computes something — a
measurement, a judgement, an attempt — carries it faithfully, and hands it to a
place nothing reads. Track N found six such boundaries, each one phase after the
phase that introduced it; the PC.1b inventory found roughly twenty.

**Why this shape rather than a run object.** `FilteredResult` is already a typed
run-state carrier. Every payload field on it is read somewhere; of its
derived-state fields only `concept_alignment_report` is, and that reader drops 5
of its 11 keys at the boundary. A carrier was not the missing thing, so this is
not a third carrier: it is the rule, made able to fail.

What this checks, exactly
=========================
It does **not** search the repository for someone who might read a field. Two
review rounds established that such a search cannot be made honest: ``x.foo``
says nothing about the type of ``x``, so a name-based sweep either misses a dead
field whose name is common — ``error`` is Load-read in 88 production files,
``info`` in 141 — or blocks a legitimate one. Denying names one at a time closed
each planted field and never the property that let it through.

So the requirement is inverted. **Every derived-state field must declare its
consumer here**, and the guard verifies the declaration:

* :class:`Reads` — the named file must Load that attribute, and must not be its
  only writer. A producer that logs its own output is not a consumer; the
  inventory records exactly that defect elsewhere (``aliases_registered`` is
  initialised and logged, never incremented).
* :class:`Owned` — no reader yet, and a named phase owns giving it one or
  deleting it. The claim must also appear in the track's ``handoff-inventory.md``.

A field with no entry fails. That is the whole mechanism: adding derived state to
a result model becomes a deliberate act that forces a sentence about who consumes
it.

**What it cannot do, stated so nobody assumes otherwise.** It cannot prove the
Load it finds in the declared file refers to *this* model rather than a
same-named attribute on something else — that needs type inference. What it does
is narrow the collision surface from 478 files to one, and make the claim
reviewable. That is as far as name-based checking honestly goes, and saying so is
cheaper than a fourth round of denylist. Reads through ``getattr``,
``model_dump()[...]`` or ``model_extra[...]`` are not attribute Loads and will
not satisfy a ``Reads`` entry — that fails closed, which is the safe direction.

**Three ways a guard like this fails**, in the order this project has been burned:
it runs nowhere (``.github/workflows/unit-guards.yml`` runs it on every PR and
push to main — before PC.1b no CI job ran any Python unit suite); it scans nothing
and passes vacuously (the controls below); it verifies a shape rather than a
property (which is why a declaration, not a sweep, is what is checked).
"""

from __future__ import annotations

import ast
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTRACTION_MODEL = REPO_ROOT / "packages/shared/src/shared/models/extraction.py"
INVENTORY = REPO_ROOT / "docs/tracks/PC-pipeline-coherence/handoff-inventory.md"

#: The result models' PAYLOAD. Out of scope: nobody has ever lost an entity list
#: at a handoff, and the reader counts confirm it — each is read somewhere.
_PAYLOAD_FIELDS = {
    "entities",
    "relations",
    "metadata",
    "removed_entities",
    "merged_entity_groups",
    "match_candidates",
    "predicted_edges",
}


@dataclass(frozen=True)
class Reads:
    """A production file that consumes this field. Verified, not trusted."""

    path: str


#: A phase label must look like ``PC.<n>`` or ``N.<n>`` — this track's own naming.
#: A review passed ``Owned("")`` and ``Owned("PC.9 — a phase that does not exist")``,
#: so the check is both that a phase is named and that a row about the field
#: names it too.
_PHASE_RE = re.compile(r"\b(PC\.\d[a-z]?|N\.\d[a-z]?)\b")


@dataclass(frozen=True)
class Owned:
    """No consumer yet; a named phase owns giving it one or deleting it."""

    phase: str


#: The declaration. Every derived-state field on every result model must appear.
#:
#: PC.1b found four fields with no reader and split them by whether a named phase
#: actually wants the measurement: ``linked_entities`` (a second copy of URIs that
#: already travel in each entity's ``properties``) and ``llm_verification_results``
#: (never written at all) were deleted; these two were kept, because deleting a
#: measurement one phase before it is wanted is churn.
#: The value type is deliberately a UNION of the two verified shapes, not
#: ``object``. A review got a genuinely dead field past the round-3 guard with
#: ``"zzz_dead": "PC.3 will sort it out"``: the membership check accepted it and
#: both verifying tests skipped it on an ``isinstance``, so a bare string was a
#: third, entirely unchecked declaration state. Worse, it is the exact shape this
#: table had one commit earlier (``Dict[str, str]``), so anyone copying from git
#: history lands in it. ``test_every_declaration_has_a_verified_shape`` closes it.
_DECLARED: Dict[str, "Reads | Owned"] = {
    "concept_alignment_report": Reads(
        "apps/app-main/src/app_main/services/entity_extraction_service.py"
    ),
    "kg_resolution_report": Owned(
        "PC.3 — its AC needs a measured figure for how many rows cross-document "
        "resolution collapses"
    ),
    "validation_report": Owned(
        "PC.6 — stage 11 is inert because no production call site passes an "
        "ontology; PC.6 owns making that visible"
    ),
}


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------


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
        and not p.startswith("tests/")
        and "/tests/" not in p
        and not Path(p).name.startswith("test_")
        and not p.endswith("shared/models/extraction.py")
    ]


def _parse(path: Path) -> ast.AST:
    """Parse, or raise.

    Deliberately not tolerant. The first version swallowed ``SyntaxError`` and
    moved on, and a review lost time to it: a broken file was silently skipped and
    the sweep reported a plausible orphan for entirely the wrong reason. It
    matters more than usual because CI pins Python 3.11 while development runs
    3.12, so a 3.12-only construct would parse locally and vanish in CI only — in
    the direction that hides readers.
    """
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _bases(node: ast.ClassDef) -> Set[str]:
    """Base names, bare or dotted (``extraction.FilteredResult``)."""
    names: Set[str] = set()
    for base in node.bases:
        if isinstance(base, ast.Name):
            names.add(base.id)
        elif isinstance(base, ast.Attribute):
            names.add(base.attr)
    return names


def _result_model_fields(
    sources: List[Path], root_file: Optional[Path] = None
) -> Dict[str, str]:
    """``{field: declaring class}`` for every non-payload field on any result model.

    Walks the WHOLE production tree, not only ``shared/models/extraction.py``. A
    review declared ``class ResolvedResult(FilteredResult)`` in
    ``entity_filtering/workflow.py`` and the single-file version was green — and a
    PC.3 result model would most plausibly be declared in the package producing
    it. Inheritance resolves transitively across files, by class name.
    """
    # (file, class) -> (base names, nodes). Keyed by FILE as well as name because
    # the repo has two unrelated classes called `ExtractionResult`: the pydantic
    # model in `shared/models/extraction.py` and a dataclass in
    # `app_main/services/source_extractor.py` carrying `chunks`, `title`, `url`.
    # Keying by name alone dragged the second one in, and this scan found it —
    # which is the reason the ROOT is anchored to its file below while
    # descendants are still matched by name.
    root_file = root_file or EXTRACTION_MODEL
    classes: Dict[Tuple[Path, str], Tuple[Set[str], List[ast.ClassDef]]] = {}
    for path in sources:
        for node in ast.walk(_parse(path)):
            if isinstance(node, ast.ClassDef):
                bases, nodes = classes.setdefault(
                    (path, node.name), (set(), [])
                )
                bases |= _bases(node)
                nodes.append(node)

    # The root is THE `ExtractionResult` declared in the model file. Everything
    # inheriting from it, or from anything that does, transitively — matched by
    # base NAME, which is the residual imprecision: a third unrelated class named
    # `FilteredResult` would be pulled in. Stated rather than hidden; the
    # alternative is import resolution, which is a different tool.
    models: Set[str] = {"ExtractionResult"}
    roots = {
        key
        for key in classes
        if key[1] == "ExtractionResult" and key[0] == root_file
    }
    assert roots, "the root ExtractionResult was not found in the model file"

    changed = True
    while changed:
        changed = False
        for (_path, name), (bases, _nodes) in classes.items():
            if name not in models and models & bases:
                models.add(name)
                changed = True

    fields: Dict[str, str] = {}
    for (path, name), (_bases_, nodes) in classes.items():
        if name not in models:
            continue
        if name == "ExtractionResult" and path != root_file:
            continue  # the unrelated dataclass; see the comment above
        for node in nodes:
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(
                    stmt.target, ast.Name
                ):
                    if stmt.target.id not in _PAYLOAD_FIELDS:
                        fields[stmt.target.id] = name
    return fields


def _attribute_use(path: Path, field: str) -> Tuple[bool, bool]:
    """``(reads, writes)`` — does this file Load the attribute, Store it, or both?"""
    reads = writes = False
    for node in ast.walk(_parse(path)):
        if isinstance(node, ast.Attribute) and node.attr == field:
            if isinstance(node.ctx, ast.Load):
                reads = True
            else:
                writes = True
        elif isinstance(node, ast.keyword) and node.arg == field:
            writes = True  # constructor kwarg: FilteredResult(field=...)
    return reads, writes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def sources() -> List[Path]:
    return _production_sources()


@pytest.fixture(scope="module")
def derived_fields(sources: List[Path]) -> Dict[str, str]:
    return _result_model_fields([EXTRACTION_MODEL, *sources])


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


def test_the_scanner_walks_the_production_tree(sources: List[Path]):
    """Control 2: the file list. It caught a real hole on its first run — the
    exclusion filter tested ``"/tests/" in path`` while the repo's own root suite
    is ``tests/…``, with no leading slash, so root test files were scanned as
    production code.
    """
    assert len(sources) > 200, len(sources)
    assert any(p.name == "workflow.py" for p in sources)
    root_tests = str(REPO_ROOT / "tests")
    assert not any(
        "/tests/" in str(p) or str(p).startswith(root_tests) for p in sources
    )


def test_every_production_file_parses(sources: List[Path]):
    """Control 3: parsing is not tolerant, and this proves it.

    A review's broken plant was silently skipped by the old tolerant version and
    the sweep reported a plausible orphan for the wrong reason. If a file cannot
    be parsed the guard says which, rather than quietly seeing fewer readers.
    """
    unparseable: List[str] = []
    for path in sources:
        try:
            _parse(path)
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            unparseable.append(f"{path.relative_to(REPO_ROOT)}: {exc}")
    assert not unparseable, "\n".join(unparseable)


def test_the_scanner_finds_a_field_it_should(derived_fields: Dict[str, str]):
    """Control 4: detection, attributed to the declaring class."""
    assert derived_fields, "no derived-state fields detected at all"
    assert derived_fields.get("concept_alignment_report") == "FilteredResult"


def test_payload_fields_are_excluded_deliberately(derived_fields: Dict[str, str]):
    """Control 5: the exclusion list does work rather than swallowing the set."""
    assert "entities" not in derived_fields
    assert _PAYLOAD_FIELDS & set(derived_fields) == set()


def test_a_subclass_in_another_file_is_found(tmp_path: Path):
    """Control 6, from review round 2. The previous version parsed exactly one
    file, so a result model declared in the package that produces it — the most
    likely place for one — was invisible.
    """
    base = tmp_path / "base.py"
    base.write_text(
        "class ExtractionResult:\n    entities: list = []\n"
        "class FilteredResult(ExtractionResult):\n    predicted_edges: list = []\n"
    )
    elsewhere = tmp_path / "elsewhere.py"
    elsewhere.write_text(
        "from typing import Optional\n"
        "class ResolvedResult(FilteredResult):\n"
        "    cross_document_resolution_report: Optional[dict] = None\n"
    )
    found = _result_model_fields([base, elsewhere], root_file=base)
    assert found.get("cross_document_resolution_report") == "ResolvedResult"


def test_a_dotted_base_is_resolved(tmp_path: Path):
    """Control 7: ``class X(extraction.FilteredResult)`` is an ``ast.Attribute``,
    not an ``ast.Name``. The round-2 version handled only the bare form.
    """
    base = tmp_path / "base.py"
    base.write_text("class ExtractionResult:\n    entities: list = []\n")
    dotted = tmp_path / "dotted.py"
    dotted.write_text(
        "from typing import Optional\n"
        "class DottedResult(extraction.ExtractionResult):\n"
        "    a_dotted_orphan_report: Optional[dict] = None\n"
    )
    assert "a_dotted_orphan_report" in _result_model_fields(
        [base, dotted], root_file=base
    )


def test_reads_and_writes_are_told_apart(tmp_path: Path):
    """Control 8, from review round 1's decisive finding. The rule says a producer
    names its CONSUMER; a counter matching ``.field`` anywhere counts an
    assignment as readily as a read.
    """
    writer = tmp_path / "writer.py"
    writer.write_text("def f(r):\n    r.zzz_written = 1\n")
    reader = tmp_path / "reader.py"
    reader.write_text("def g(r):\n    return r.zzz_read\n")
    kwarg = tmp_path / "kwarg.py"
    kwarg.write_text("def h():\n    return Model(zzz_kwarg=1)\n")

    assert _attribute_use(writer, "zzz_written") == (False, True)
    assert _attribute_use(reader, "zzz_read") == (True, False)
    assert _attribute_use(kwarg, "zzz_kwarg") == (False, True)


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_every_derived_state_field_is_declared(derived_fields: Dict[str, str]):
    """The invariant. Adding derived state to a result model forces a sentence
    about who consumes it.

    Inverted deliberately: two review rounds showed the other direction cannot be
    made honest. Searching the repo for a reader by bare attribute name misses a
    dead field called ``error`` (Load-read in 88 files), and denying names one at
    a time closes the planted field and never the property.

    When this fails, three fixes, and "declare it owned" is the third:

    1. Give it a reader and declare ``Reads(path)``.
    2. Delete it — N.5b ruled out shipping a producer that survives by accident.
    3. ``Owned(phase)`` plus a row in ``handoff-inventory.md``.
    """
    undeclared = sorted(set(derived_fields) - set(_DECLARED))
    assert not undeclared, (
        "derived state with no declared consumer: "
        + ", ".join(f"{f} (on {derived_fields[f]})" for f in undeclared)
        + ". Declare Reads(path), or Owned(phase) plus a row in "
        "docs/tracks/PC-pipeline-coherence/handoff-inventory.md — or delete it."
    )


def test_every_reads_declaration_is_true(sources: List[Path]):
    """A declared consumer that does not consume is worse than none: it reports as
    a reader. The named file must Load the attribute, and must not be its only
    writer — a producer logging its own output is not a consumer, which is exactly
    the defect the inventory records for ``aliases_registered``.
    """
    problems: List[str] = []
    for field, claim in _DECLARED.items():
        if not isinstance(claim, Reads):
            continue
        path = REPO_ROOT / claim.path
        if not path.exists():
            problems.append(f"{field}: declared consumer {claim.path} does not exist")
            continue
        reads, writes = _attribute_use(path, field)
        if not reads:
            problems.append(f"{field}: {claim.path} does not read it")
        elif writes and not any(
            _attribute_use(other, field)[0] for other in sources if other != path
        ):
            problems.append(
                f"{field}: {claim.path} both writes and reads it, and nothing else "
                "reads it — a producer logging its own output is not a consumer"
            )
    assert not problems, "\n".join(problems)


def test_every_owned_declaration_appears_in_the_inventory():
    """An ``Owned`` entry is a promise made in two places or it is not a promise.
    The inventory is where a human looks; this file is where CI looks.
    """
    assert INVENTORY.exists(), f"{INVENTORY} not found"
    lines = INVENTORY.read_text(encoding="utf-8").splitlines()
    problems: List[str] = []
    for field, claim in _DECLARED.items():
        if not isinstance(claim, Owned):
            continue
        phase = _PHASE_RE.search(claim.phase)
        if not phase:
            problems.append(
                f"{field}: Owned({claim.phase!r}) names no phase — a review passed "
                'Owned("") and Owned("PC.9 - a phase that does not exist")'
            )
            continue
        # A ROW about this field, not a substring anywhere in the document. The
        # inventory is a document about this subject area, so its vocabulary
        # overlaps plausible field names: a review satisfied the previous check
        # with `metrics` (a table name in it) and `repair` (inside another row's
        # prose). Requiring the field in backticks AND the owning phase on the
        # same line is what makes it evidence rather than coincidence.
        if not any(
            f"`{field}`" in line and phase.group(0) in line for line in lines
        ):
            problems.append(
                f"{field}: no row in handoff-inventory.md names both `{field}` "
                f"and {phase.group(0)}"
            )
    assert not problems, "\n".join(problems)


def test_every_declaration_has_a_verified_shape():
    """Closes the round-3 blocker: a bare value is a third, unchecked state.

    Both verifying tests branch on ``isinstance``, so anything that is neither a
    ``Reads`` nor an ``Owned`` satisfies the membership check and is then skipped
    by both. A review got a dead field through with
    ``"zzz_dead": "PC.3 will sort it out"``. It is the exact shape this table had
    one commit earlier, so it is what anyone copying from history would write.
    """
    wrong = sorted(
        f"{field}: {type(claim).__name__}"
        for field, claim in _DECLARED.items()
        if not isinstance(claim, (Reads, Owned))
    )
    assert not wrong, (
        "declarations that are neither Reads nor Owned, and are therefore "
        f"verified by nothing: {wrong}"
    )


def test_the_declaration_does_not_outlive_its_fields(derived_fields: Dict[str, str]):
    """An entry for a field that no longer exists is stale permission: it would
    silently cover a future field that reuses the name.
    """
    stale = sorted(set(_DECLARED) - set(derived_fields))
    assert not stale, f"_DECLARED names fields that no longer exist: {stale}"
