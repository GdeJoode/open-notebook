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

API = Path(__file__).resolve().parents[1] / "api.py"

#: Constants that are legitimately assigned and not referenced again IN THIS FILE,
#: with the reason. Empty today, and it should stay hard to add to.
_ALLOWED: Dict[str, str] = {}


def _env_constants(tree: ast.AST) -> Dict[str, ast.AST]:
    """Module-level names assigned from `os.getenv`, however it is wrapped.

    Catches `X = os.getenv(...)`, `X = int(os.getenv(...))`, `X = Path(os.getenv(...))`
    — the wrapper is why a name-only match would miss two of the three shapes in
    this file.
    """
    found: Dict[str, ast.AST] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        for inner in ast.walk(node.value):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and inner.func.attr in ("getenv", "environ")
            ):
                found[target.id] = node
                break
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
    tree = ast.parse(API.read_text(encoding="utf-8"))
    constants = _env_constants(tree)
    assert constants, "walker control: the scan found no env constants at all"

    skip = {id(n) for stmt in constants.values() for n in ast.walk(stmt)}
    used = _loaded_names(tree, skip)

    dead = sorted(name for name in constants if name not in used and name not in _ALLOWED)
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
