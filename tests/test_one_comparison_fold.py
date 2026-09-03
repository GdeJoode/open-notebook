"""PC.2 — the comparison fold exists once.

Four stages carried a byte-identical private copy of the same four-line transform
(NFKC → lower → strip → collapse), and a fifth call site reached one of them
through a private attribute across a package boundary. They are now shims over
``shared.utils.text_folding.fold_for_comparison``.

This guard exists because the copies were identical *today* and there was nothing
to stop the sixth diverging. A drifted duplicate is worse than a duplicate: two
stages would disagree about whether two names are the same string, and the
disagreement would show up as an entity that deduplicates in one stage and not the
next.

**What it checks and what it cannot.** It detects the SHAPE — a function whose body
combines an NFKC normalise, a ``.lower()`` and a ``re.sub`` on ``\\s+`` — by AST,
not by matching source text, so renaming the function or its locals does not evade
it. It cannot detect a fold spelled a different way (``str.casefold``, a manual
character loop, ``' '.join(text.split())``). That limit is stated rather than
implied, and the mutant control below is what keeps the detector honest about the
shape it does claim.

PC.1b's guard took four review rounds to become one that could fail. The controls
here are the ones that phase ended up needing: the walker must prove it reached
something, the detector must prove it finds a known instance, parsing must fail
loudly rather than skip, and a planted copy must be caught by the same detector
that guards the tree.
"""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SHARED_FOLD = REPO_ROOT / "packages/shared/src/shared/utils/text_folding.py"

#: Functions that trip the shape and are deliberately NOT the comparison fold.
#: Each needs a reason, because an entry here is permission to keep a duplicate.
#: Keyed by ``path::name``, not by name. Two unrelated functions in this repo are
#: both called ``_normalize_text`` — one in `cites_matching.py` (a citation-title
#: matcher, which DOES trip the shape) and one in `entity_filtering/filters/
#: normalizer.py` (which does not, because it never lowercases). A name-keyed
#: list conflated them and would have granted the second one permission it was
#: never asked for. PC.1b learned the same lesson about `ExtractionResult`.
_ALLOWED: Dict[str, str] = {
    "packages/shared/src/shared/utils/text_folding.py::fold_for_comparison": (
        "the implementation"
    ),
    "packages/shared/src/shared/retrieval/cites_matching.py::_normalize_text": (
        "a citation-title matcher in a different domain; it also strips "
        "punctuation, which the comparison fold must not do"
    ),
}

#: Not detected, and deliberately so — recorded here so a reader does not assume
#: coverage. `EntityNormalizer._normalize_text` is a configurable merging
#: transform inside a pipeline stage rather than a comparison key: it does not
#: lowercase, and it strips English articles. Folding it in would change what the
#: Normalizer stage merges, on by default. `_fold_host` in `routers/agents.py`
#: folds a hostname and does no whitespace collapse.
_KNOWN_NON_MATCHES = (
    "pipelines/entity-filtering/src/entity_filtering/filters/normalizer.py::_normalize_text",
    "apps/app-main/src/app_main/api/routers/agents.py::_fold_host",
)


def _production_sources() -> List[Path]:
    # Tracked AND untracked-but-not-ignored: a sixth copy sitting in someone's
    # working tree should fail before it is committed, not after. The control
    # below caught this — the shared implementation itself was untracked when the
    # guard first ran, so the sweep could not see the one file it must.
    tracked = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout.splitlines()
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "--", "*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout.splitlines()
    return [
        REPO_ROOT / p
        for p in dict.fromkeys([*tracked, *untracked])
        if p.endswith(".py")
        and not p.startswith("tests/")
        and "/tests/" not in p
        and not Path(p).name.startswith("test_")
    ]


def _is_nfkc_call(node: ast.AST) -> bool:
    """``unicodedata.normalize("NFKC", …)``, however the module is imported."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name != "normalize" or not node.args:
        return False
    first = node.args[0]
    return isinstance(first, ast.Constant) and first.value == "NFKC"


def _is_lower_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "lower"
    )


#: Module-level names bound to `re.compile(r"\s+")`, per parsed module. A
#: precompiled pattern is how anybody writing this for performance would spell it
#: — the shared implementation itself does — and the first version of the detector
#: missed exactly that, which its own control caught.
def _whitespace_pattern_names(tree: ast.AST) -> Set[str]:
    names: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        func = call.func
        attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if attr != "compile" or not call.args:
            continue
        first = call.args[0]
        if isinstance(first, ast.Constant) and first.value in (r"\s+", "\\s+"):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _is_whitespace_sub(node: ast.AST, compiled: Set[str]) -> bool:
    r"""``re.sub(r"\s+", …)``, or ``<name>.sub(…)`` for a name compiled from it."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "sub":
        return False
    if isinstance(func.value, ast.Name) and func.value.id in compiled:
        return True
    if not node.args:
        return False
    first = node.args[0]
    return isinstance(first, ast.Constant) and first.value in (r"\s+", "\\s+")


def detect_folds(source: str) -> Set[str]:
    """Names of functions whose body combines all three operations.

    All three, not any: an NFKC call alone is a unicode nicety and a ``.lower()``
    alone is everywhere. It is the combination that makes a comparison fold.
    """
    tree = ast.parse(source)
    compiled = _whitespace_pattern_names(tree)
    found: Set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        nfkc = lower = ws = False
        for inner in ast.walk(node):
            nfkc = nfkc or _is_nfkc_call(inner)
            lower = lower or _is_lower_call(inner)
            ws = ws or _is_whitespace_sub(inner, compiled)
        if nfkc and lower and ws:
            found.add(node.name)
    return found


def _scanned_keys() -> Set[str]:
    """``path::name`` for every fold the detector finds in the tree."""
    return {
        f"{path.relative_to(REPO_ROOT)}::{name}" for path, name in _scan_tree()
    }


def _scan_tree() -> List[Tuple[Path, str]]:
    hits: List[Tuple[Path, str]] = []
    for path in _production_sources():
        for name in detect_folds(path.read_text(encoding="utf-8")):
            hits.append((path, name))
    return hits


# ---------------------------------------------------------------------------
# Controls
# ---------------------------------------------------------------------------


def test_the_walker_reaches_the_tree():
    """Without this, a broken file list makes the sweep find nothing and pass."""
    sources = _production_sources()
    assert len(sources) > 200, len(sources)
    assert SHARED_FOLD in sources
    assert not any("/tests/" in str(p) for p in sources)


def test_every_production_file_parses():
    """Parsing is not tolerant. In PC.1b a silently-skipped unparseable file made
    a sweep report a plausible result for entirely the wrong reason.
    """
    broken: List[str] = []
    for path in _production_sources():
        try:
            ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError) as exc:
            broken.append(f"{path.relative_to(REPO_ROOT)}: {exc}")
    assert not broken, "\n".join(broken)


def test_the_detector_finds_the_implementation():
    """Control: if the detector stops matching, the sweep below reports zero
    duplicates and passes — which is the vacuity failure, not a clean tree.
    """
    assert "fold_for_comparison" in detect_folds(
        SHARED_FOLD.read_text(encoding="utf-8")
    )


def test_the_detector_requires_all_three_operations():
    """Control: any-of instead of all-of would flag half the codebase, and the
    author would delete the guard rather than the duplicates.
    """
    assert detect_folds("def f(t):\n    return t.lower()\n") == set()
    assert detect_folds(
        "import unicodedata\ndef f(t):\n    return unicodedata.normalize('NFKC', t)\n"
    ) == set()
    assert detect_folds("import re\ndef f(t):\n    return re.sub(r'\\s+', ' ', t)\n") == set()


def test_the_detector_sees_a_precompiled_pattern():
    """Control, and it caught a real gap: the shared implementation itself uses a
    module-level `_WHITESPACE_RE`, which the first detector missed entirely. A
    performance-minded sixth copy would be written exactly that way.
    """
    planted = """
import re
import unicodedata

_WS = re.compile(r"\\s+")


def squash(raw):
    return _WS.sub(" ", unicodedata.normalize("NFKC", raw).lower().strip())
"""
    assert "squash" in detect_folds(planted)


def test_a_planted_sixth_copy_is_caught():
    """The assertion that makes 'a sixth copy would be caught' a claim rather than
    a hope. Different function name, different locals, different statement order —
    the detector must see the shape, not the text.
    """
    planted = """
import re as regexlib
import unicodedata as ucd


def squash_a_name(raw):
    cleaned = ucd.normalize("NFKC", raw)
    cleaned = cleaned.strip().lower()
    return regexlib.sub(r"\\s+", " ", cleaned)
"""
    assert "squash_a_name" in detect_folds(planted)


def test_the_shims_do_not_hide_the_implementation():
    """The four stages now call the shared function rather than inlining it, so
    the sweep should find the implementation and NOT them. If a shim were reverted
    to an inline copy, the sweep catches it — which is the whole point.
    """
    from entity_filtering.deduplication.entity_deduplicator import EntityDeduplicator
    from entity_filtering.deduplication.fuzzy_resolver import FuzzyResolver
    from entity_filtering.resolution.kg_resolver import KGResolver

    probe = "  Brede   Welvaart  "
    from shared.utils.text_folding import fold_for_comparison

    assert (
        EntityDeduplicator._normalize_key(probe)
        == FuzzyResolver._normalize(probe)
        == KGResolver._normalize(probe)
        == fold_for_comparison(probe)
        == "brede welvaart"
    )


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def test_the_comparison_fold_exists_once():
    """One implementation, plus an allow-list whose entries carry a reason.

    When this fails, the fix is to call `fold_for_comparison` — not to add a name
    here. An allow-list entry is permission to keep a duplicate, and a duplicate
    that drifts makes two stages disagree about whether two names are equal.
    """
    unexpected = sorted(
        key for key in _scanned_keys() if key not in _ALLOWED
    )
    assert not unexpected, (
        "functions that fold text for comparison outside the shared one: "
        + ", ".join(unexpected)
        + ". Call shared.utils.text_folding.fold_for_comparison instead, or add "
        "the name to _ALLOWED with the reason it must differ."
    )


def test_the_allow_list_does_not_outlive_its_entries():
    """An entry for a function that no longer exists is stale permission: it would
    silently cover a future duplicate that reuses the name.
    """
    stale = sorted(set(_ALLOWED) - _scanned_keys())
    assert not stale, f"_ALLOWED names functions that no longer fold: {stale}"


@pytest.mark.parametrize("name", _KNOWN_NON_MATCHES, ids=lambda k: k.split("::")[-1])
def test_the_documented_non_matches_really_are_not_matched(name: str):
    """The docstring says these are deliberately out of scope. If one of them ever
    starts tripping the detector, the docstring becomes wrong — and a reader who
    trusts it would assume coverage that does not exist.
    """
    assert name not in _scanned_keys()
