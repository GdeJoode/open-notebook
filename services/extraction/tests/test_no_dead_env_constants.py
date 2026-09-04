"""No module constant reads the environment and is then never used (PC.6).

Two of them shipped here: `OLLAMA_MODEL` (from `EXTRACTION_MODEL`) and
`OLLAMA_NUM_CTX` (from `EXTRACTION_NUM_CTX`). Each was assigned once and used
nowhere, because `_extract_from_chunk` resolves the model through
`model_routing.call_llm(step="extraction")` instead.

That is worse than dead code. A knob set in `docker-compose.yml` and read into a
constant reads as a working control, so the compose file documents a configuration
the service does not have. It cost a fully reverted branch: `EXTRACTION_MODEL` was
pointed at a different model, the container was restarted, and the resolver went
on returning `llama3.1:8b-instruct-q4_0`.

This guard is the class, not the two instances — the next one is caught by
existing. When it fires the fix is to use the value or delete the constant, never
to add a name to the allow-list without a reason beside it.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict, Set

import pytest

SERVICE = Path(__file__).resolve().parents[1]

#: Every module in the service, not just `api.py`. Review found a live instance
#: one file over — `entity_validator.py` bound `OLLAMA_URL` and never read it —
#: while this guard claimed to cover "the class".
SOURCES = sorted(p for p in SERVICE.glob("*.py") if p.name != "__init__.py")

#: `file.py::NAME` entries that are legitimately assigned and not referenced
#: again in their own file, with the reason. Empty today, and it should stay hard
#: to add to.
_ALLOWED: Dict[str, str] = {}


def _reads_environment(node: ast.AST) -> bool:
    """True when the expression reads the process environment, in any spelling.

    Review probed six shapes against the first version and all six slipped
    through. `os.environ.get(...)` is the most likely next occurrence and was the
    one the docstring implied was covered: its `func.attr` is `get`, not
    `environ`, and `os.environ[...]` is a Subscript rather than a Call at all.
    """
    for inner in ast.walk(node):
        # os.getenv(...) / getenv(...) after `from os import getenv`
        if isinstance(inner, ast.Call):
            func = inner.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name == "getenv":
                return True
            # os.environ.get(...)
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "get"
                and isinstance(func.value, ast.Attribute)
                and func.value.attr == "environ"
            ):
                return True
        # os.environ["X"] / environ["X"]
        if isinstance(inner, ast.Subscript):
            value = inner.value
            if isinstance(value, ast.Attribute) and value.attr == "environ":
                return True
            if isinstance(value, ast.Name) and value.id == "environ":
                return True
    return False


def _env_constants(tree: ast.AST) -> Dict[str, ast.AST]:
    """Module-level names bound to a value that reads the environment.

    Handles `X = …`, `X: T = …` (AnnAssign), `X, Y = …, …` (tuple unpack) and the
    walrus — all four were evasions in the first version. The wrapper around the
    read (`int(...)`, `Path(...)`) is why a name-only match is not enough.
    """
    found: Dict[str, ast.AST] = {}

    def _bind(targets, statement) -> None:
        for target in targets:
            if isinstance(target, ast.Name):
                found[target.id] = statement
            elif isinstance(target, (ast.Tuple, ast.List)):
                _bind(target.elts, statement)

    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            if node.value is not None and _reads_environment(node.value):
                _bind([node.target], node)
        elif isinstance(node, ast.Assign):
            # `X, Y = os.getenv('A'), 1` — bind each target to the element that
            # actually reads the environment, not the whole tuple.
            targets = node.targets
            if (
                len(targets) == 1
                and isinstance(targets[0], (ast.Tuple, ast.List))
                and isinstance(node.value, (ast.Tuple, ast.List))
                and len(targets[0].elts) == len(node.value.elts)
            ):
                for name_node, value in zip(targets[0].elts, node.value.elts):
                    if isinstance(name_node, ast.Name) and _reads_environment(value):
                        found[name_node.id] = node
                continue
            if _reads_environment(node.value):
                _bind(targets, node)
        else:
            # `if (X := os.getenv("A")):` at module level.
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.NamedExpr)
                    and isinstance(inner.target, ast.Name)
                    and _reads_environment(inner.value)
                ):
                    found[inner.target.id] = inner
    return found


def _loaded_names(tree: ast.AST, skip: Set[int]) -> Set[str]:
    """Every name READ anywhere, excluding the assignment statements themselves.

    Load context only: counting a `Store` would make each constant its own reader,
    which is the trap PC.1b's invariant took four rounds to escape.
    """
    used: Set[str] = set()
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    return used


def test_no_env_constant_is_assigned_and_never_read() -> None:
    assert SOURCES, "walker control: no source files were scanned"
    scanned = 0
    dead: list = []
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        constants = _env_constants(tree)
        scanned += len(constants)
        skip = {id(n) for stmt in constants.values() for n in ast.walk(stmt)}
        used = _loaded_names(tree, skip)
        dead += [
            f"{path.name}::{name}"
            for name in sorted(constants)
            if name not in used and f"{path.name}::{name}" not in _ALLOWED
        ]

    assert scanned, "detector control: no env constants found in any file"
    assert not dead, (
        f"assigned from the environment and never read: {dead}. Use the value, or "
        f"delete the constant AND the compose/Dockerfile entries that set it — a "
        f"knob that configures nothing is worse than no knob."
    )


def test_the_detector_finds_a_planted_dead_constant() -> None:
    """Mutant control: prove the scan can fail.

    Without this, the test above passes just as well against a scan that finds
    nothing — which is how a guard ends up reporting as a guard while unable to.
    """
    planted = ast.parse(
        "import os\n"
        "ALIVE = os.getenv('A', '1')\n"
        "DEAD = int(os.getenv('B', '2'))\n"
        "print(ALIVE)\n"
    )
    constants = _env_constants(planted)
    assert set(constants) == {"ALIVE", "DEAD"}
    skip = {id(n) for stmt in constants.values() for n in ast.walk(stmt)}
    used = _loaded_names(planted, skip)
    assert "DEAD" not in used and "ALIVE" in used


_EVASIONS = {
    "tuple unpack": "import os\nX, Y = os.getenv('A'), 1\n",
    "annotated assignment": "import os\nX: str = os.getenv('A')\n",
    "environ subscript": "import os\nX = os.environ['A']\n",
    "environ.get": "import os\nX = os.environ.get('A')\n",
    "bare getenv import": "from os import getenv\nX = getenv('A')\n",
    "walrus at module level": "import os\nif (X := os.getenv('A')):\n    pass\n",
    "wrapped in a cast": "import os\nX = int(os.getenv('A', '1'))\n",
}


@pytest.mark.parametrize("shape", sorted(_EVASIONS), ids=lambda k: k.replace(" ", "-"))
def test_the_detector_sees_every_spelling(shape: str) -> None:
    """Each of these bound an environment value invisibly to the first version.

    Review probed six and all six slipped through. `os.environ.get` is the one
    that mattered most: it is the most likely next occurrence, and the old
    docstring implied it was covered — its `func.attr` is `get`, not `environ`,
    and `os.environ[...]` is not a Call at all.
    """
    assert "X" in _env_constants(ast.parse(_EVASIONS[shape])), shape


def test_a_value_that_is_used_is_not_reported() -> None:
    """The counterweight: widening the detector must not make it fire on live code.

    Without this, closing the evasions could be "fixed" by reporting every
    environment read, and the guard would be disabled the first time it cried
    wolf.
    """
    src = "import os\nX = os.environ.get('A')\nprint(X)\n"
    tree = ast.parse(src)
    constants = _env_constants(tree)
    skip = {id(n) for stmt in constants.values() for n in ast.walk(stmt)}
    assert "X" in constants
    assert "X" in _loaded_names(tree, skip)
