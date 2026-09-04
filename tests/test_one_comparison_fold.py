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

**What it checks.** It detects the SHAPE — a callable whose body combines an NFKC
normalise, a ``.lower()`` and a ``re.sub`` on ``\\s+`` — by AST rather than by
matching source text, so renaming the function or its locals does not evade it.
Review round 1 documented three limits and implied the rest was covered; a
reviewer then planted 18 variants and six got through. The parametrised cases at
the foot of this file are those six plus two more, so the claim below is a
measured one rather than a description of intent.

**Handled**: plain function, method, ``staticmethod``, ``classmethod``; a
precompiled pattern bound at module level, in a class body (used via ``self.``,
``cls.`` or any attribute), inside a function, or onto an instance attribute in
``__init__``; annotated assignment; the ``"NFKC"`` form or the ``\\s+`` literal held
in a constant; the keyword forms ``re.sub(pattern=…)`` and ``normalize(form=…)``;
a lambda, reported under the name it is bound to; ``str.lower(t)`` and aliased
imports.

**Not handled, and each of these would evade it today**:

* a fold spelled a different way — ``str.casefold``, a manual character loop,
  ``' '.join(text.split())``, or the pattern ``r"\\s\\s*"``;
* a pattern reached through a subscript, a tuple unpack, or another module's
  namespace, which would need import resolution.

The list is exhaustive as far as it has been attacked, which is not the same as
exhaustive. It is here so a reader does not infer coverage from the cases that ARE
handled — the mistake round 1 invited.

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
from typing import Dict, List, Sequence, Set, Tuple

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


def _is_nfkc_call(node: ast.AST, nfkc_names: Set[str]) -> bool:
    """``unicodedata.normalize("NFKC", …)``, however the module is imported.

    ``nfkc_names`` carries the module's names bound to the literal ``"NFKC"``, so
    holding the form in a constant does not hide the call.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
    if name != "normalize":
        return False
    first = _arg_or_keyword(node, 0, "form")
    if first is None:
        return False
    return (isinstance(first, ast.Constant) and first.value == "NFKC") or (
        isinstance(first, ast.Name) and first.id in nfkc_names
    )


def _is_lower_call(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "lower"
    )


_WS_LITERALS = (r"\s+", "\\s+")


def _arg_or_keyword(call: ast.Call, index: int, name: str) -> ast.expr | None:
    """The positional argument at ``index``, or the keyword called ``name``.

    `re.sub(pattern=…)` and `normalize(form=…)` are legal and were both evasions
    while the detector read positional arguments only.
    """
    if len(call.args) > index:
        return call.args[index]
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _bound_targets(node: ast.AST) -> List[str]:
    """Names an assignment binds: `x = …`, `x: T = …`, `a = b = …`, `self.x = …`.

    Attribute targets are reported by their attribute name, because that is how
    the value is later read (`self._ws.sub(...)`) and `_is_whitespace_sub` matches
    an attribute access by name. `self._ws = re.compile(r"\s+")` in `__init__` is
    a plausible spelling and was an evasion until review named it.
    """
    targets: List[ast.expr]
    if isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        return []
    names: List[str] = []
    for target in targets:
        if isinstance(target, ast.Name):
            names.append(target.id)
        elif isinstance(target, ast.Attribute):
            names.append(target.attr)
    return names


def _names_bound_to_constant(tree: ast.AST, values: Sequence[object]) -> Set[str]:
    r"""Names bound anywhere to one of ``values`` as a plain literal.

    Catches the indirection of holding the literal in a constant —
    ``_WS = r"\s+"`` then ``re.sub(_WS, …)``, or ``_FORM = "NFKC"`` then
    ``normalize(_FORM, …)``. Both were live evasions found by adversarial review.
    """
    names: Set[str] = set()
    for node in ast.walk(tree):
        value = getattr(node, "value", None)
        if isinstance(value, ast.Constant) and value.value in values:
            names.update(_bound_targets(node))
    return names


#: Names bound to `re.compile(r"\s+")`, per parsed module — at module level, in a
#: class body, or inside a function. A precompiled pattern is how anybody writing
#: this for performance would spell it — the shared implementation itself does —
#: and the first version of the detector missed exactly that, which its own
#: control caught. A later review found it still missed the CLASS-attribute form
#: (`self._WS.sub`), which is this repo's own idiom (`CandidateDedupService._ROMAN_RE`).
def _whitespace_pattern_names(tree: ast.AST) -> Set[str]:
    literal_names = _names_bound_to_constant(tree, _WS_LITERALS)
    names: Set[str] = set()
    for node in ast.walk(tree):
        call = getattr(node, "value", None)
        if not isinstance(call, ast.Call):
            continue
        func = call.func
        attr = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if attr != "compile":
            continue
        first = _arg_or_keyword(call, 0, "pattern")
        if first is None:
            continue
        pattern_is_ws = (
            isinstance(first, ast.Constant) and first.value in _WS_LITERALS
        ) or (isinstance(first, ast.Name) and first.id in literal_names)
        if pattern_is_ws:
            names.update(_bound_targets(node))
    return names


def _is_whitespace_sub(
    node: ast.AST, compiled: Set[str], literal_names: Set[str]
) -> bool:
    r"""``re.sub(r"\s+", …)``, or ``<name>.sub(…)`` for a name compiled from it."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "sub":
        return False
    target = func.value
    # `_WS.sub(…)`, and equally `self._WS.sub(…)` / `cls._WS.sub(…)` / any
    # `<obj>._WS.sub(…)`. The attribute form is this repo's own idiom, and the
    # detector missed it until a review planted it.
    if isinstance(target, ast.Name) and target.id in compiled:
        return True
    if isinstance(target, ast.Attribute) and target.attr in compiled:
        return True
    first = _arg_or_keyword(node, 0, "pattern")
    if first is None:
        return False
    return (isinstance(first, ast.Constant) and first.value in _WS_LITERALS) or (
        isinstance(first, ast.Name) and first.id in literal_names
    )


def detect_folds(source: str) -> Set[str]:
    """Names of functions whose body combines all three operations.

    All three, not any: an NFKC call alone is a unicode nicety and a ``.lower()``
    alone is everywhere. It is the combination that makes a comparison fold.
    """
    tree = ast.parse(source)
    compiled = _whitespace_pattern_names(tree)
    literal_names = _names_bound_to_constant(tree, _WS_LITERALS)
    nfkc_names = _names_bound_to_constant(tree, ("NFKC",))
    found: Set[str] = set()
    for node in ast.walk(tree):
        # Lambdas count: `fold = lambda t: ...` is a fifth copy that happens to
        # have no `def`. It is reported under the name it is bound to.
        if isinstance(node, ast.Lambda):
            name = next(
                (n for parent in ast.walk(tree)
                 for n in _bound_targets(parent)
                 if getattr(parent, "value", None) is node),
                "<lambda>",
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name
        else:
            continue
        nfkc = lower = ws = False
        for inner in ast.walk(node):
            nfkc = nfkc or _is_nfkc_call(inner, nfkc_names)
            lower = lower or _is_lower_call(inner)
            ws = ws or _is_whitespace_sub(inner, compiled, literal_names)
        if nfkc and lower and ws:
            found.add(name)
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
    from entity_filtering.resolution import concept_alignment
    from entity_filtering.resolution.kg_resolver import KGResolver
    from shared.utils.text_folding import fold_for_comparison

    # All FOUR shims, and over inputs where a partial revert would show: interior
    # runs, an NFKC-only difference, and a case-only difference. A single probe
    # like "  Brede   Welvaart  " is satisfied by a shim that dropped the NFKC
    # step, which is exactly the silent partial revert this test is for.
    shims = (
        EntityDeduplicator._normalize_key,
        FuzzyResolver._normalize,
        KGResolver._normalize,
        concept_alignment._normalize,
    )
    for probe, expected in (
        ("  Brede   Welvaart  ", "brede welvaart"),
        ("Ｒｅｇｉｏ\tDeal", "regio deal"),   # full-width + tab: NFKC + collapse
        ("REGIO DEAL", "regio deal"),         # case only
        ("ﬁnanciering", "financiering"),      # ligature: NFKC only
        ("", ""),
    ):
        got = {shim(probe) for shim in shims}
        assert got == {expected}, f"{probe!r}: shims disagree or drifted: {got}"
        assert fold_for_comparison(probe) == expected


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


# ---------------------------------------------------------------------------
# Evasions found by adversarial review
#
# Round 1 of this guard documented three limits and implied the rest was covered.
# A reviewer planted 18 variants and six got through. Each is now a case, because
# "the detector handles precompiled patterns" was true of exactly one spelling of
# that idea.
# ---------------------------------------------------------------------------

_PRELUDE = "import re, unicodedata\n"
_TAIL = 'unicodedata.normalize("NFKC", t).lower().strip()'

_EVASIONS = {
    "class attribute via self": _PRELUDE + (
        "class C:\n"
        '    _WS = re.compile(r"\\s+")\n'
        "    def fold(self, t):\n"
        f'        return self._WS.sub(" ", {_TAIL})\n'
    ),
    "class attribute via cls": _PRELUDE + (
        "class C:\n"
        '    _WS = re.compile(r"\\s+")\n'
        "    @classmethod\n"
        "    def fold(cls, t):\n"
        f'        return cls._WS.sub(" ", {_TAIL})\n'
    ),
    "annotated assignment": _PRELUDE + (
        '_WS: re.Pattern = re.compile(r"\\s+")\n'
        "def fold(t):\n"
        f'    return _WS.sub(" ", {_TAIL})\n'
    ),
    "NFKC held in a constant": _PRELUDE + (
        '_FORM = "NFKC"\n'
        "def fold(t):\n"
        '    return re.sub(r"\\s+", " ", '
        "unicodedata.normalize(_FORM, t).lower().strip())\n"
    ),
    "pattern literal held in a constant": _PRELUDE + (
        '_WS_PAT = r"\\s+"\n'
        "def fold(t):\n"
        f'    return re.sub(_WS_PAT, " ", {_TAIL})\n'
    ),
    "constant compiled, then used": _PRELUDE + (
        '_WS_PAT = r"\\s+"\n'
        "_WS = re.compile(_WS_PAT)\n"
        "def fold(t):\n"
        f'    return _WS.sub(" ", {_TAIL})\n'
    ),
    "lambda": _PRELUDE + (
        f'fold = lambda t: re.sub(r"\\s+", " ", {_TAIL})\n'
    ),
    "instance attribute compiled in __init__": _PRELUDE + (
        "class C:\n"
        "    def __init__(self):\n"
        '        self._ws = re.compile(r"\\s+")\n'
        "    def fold(self, t):\n"
        f'        return self._ws.sub(" ", {_TAIL})\n'
    ),
    "keyword arguments": _PRELUDE + (
        "def fold(t):\n"
        '    return re.sub(pattern=r"\\s+", repl=" ", '
        'string=unicodedata.normalize(form="NFKC", unistr=t).lower().strip())\n'
    ),
    "function-local compile": _PRELUDE + (
        "def fold(t):\n"
        '    ws = re.compile(r"\\s+")\n'
        f'    return ws.sub(" ", {_TAIL})\n'
    ),
}


@pytest.mark.parametrize("shape", sorted(_EVASIONS), ids=lambda k: k.replace(" ", "-"))
def test_a_planted_copy_is_caught_in_every_shape(shape: str) -> None:
    """Each of these was a green pass against an earlier version of the detector.

    The class-attribute forms matter most: `CandidateDedupService._ROMAN_RE` is
    exactly that idiom in this repository, so a sixth copy written in the local
    style would have slipped through.
    """
    assert detect_folds(_EVASIONS[shape]), f"evasion not caught: {shape}"


def test_the_detector_still_needs_all_three_operations() -> None:
    """Widening the detector must not make it fire on two of the three.

    The counterweight to the cases above: an NFKC call alone is a unicode nicety
    and `.lower()` alone is everywhere. Without this, closing the evasions could
    be "fixed" by loosening the conjunction, and the guard would start failing on
    unrelated code — a guard that cries wolf gets an allow-list entry, which is
    how it stops guarding.
    """
    two_of_three = _PRELUDE + (
        '_WS = re.compile(r"\\s+")\n'
        "class C:\n"
        "    def not_a_fold(self, t):\n"
        '        return self._WS.sub(" ", unicodedata.normalize("NFKC", t).strip())\n'
    )
    assert detect_folds(two_of_three) == set()
