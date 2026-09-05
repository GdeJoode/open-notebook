"""Every conditionally-built stage is reachable once built (PC.6, round 4).

**Why this exists, in one sentence**: three review rounds each found another flag
that was on and did nothing, because the check enumerated the pairs someone had
thought of. Review named the habit precisely — *the generalisation was asserted in
prose and sampled in a test* — and the fix that worked every time was the same
move: stop exhibiting members of the space and start deriving it.

So this derives the space. It reads `FilteringWorkflow.__init__` by AST, finds
every `self._X: Optional[...] = None` that is then assigned inside an `if`, and
asserts each one is READ somewhere in the class. A stage that is built under a
condition and never consulted is exactly the shape of all four findings:

    entity_linker      built when linking is on, but only used if a provider resolves
    graph_analyzer     built only when centrality is on — outlier detection reads it
    orphan connector   gated on config, needs three arguments nobody passed
    semantic_blocker   built when its flag is on, used only while matching runs

Each of those cost a review round. This is the enumeration that would have found
all four in one pass, and it is here to find the next one before a reviewer does.

It deliberately does NOT assert the *dependency pairs* — that is
`check_feature_dependencies`' job, and duplicating it here would be a second
sampling. What this asserts is weaker and structural: nothing is constructed that
the class cannot reach.
"""

from __future__ import annotations

import ast
import inspect
from typing import Dict, Set

from entity_filtering.workflow import FilteringWorkflow

#: `self._X` attributes built conditionally and legitimately not read by this
#: class, with the reason. Empty, and an entry must name where the value goes.
_ALLOWED: Dict[str, str] = {}


def _init_node(tree: ast.AST) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == "__init__":
                return node
    raise AssertionError("walker control: __init__ not found in the class source")


def _conditionally_built(init: ast.AST) -> Set[str]:
    """`self._X` attributes assigned inside an `if` within `__init__`.

    The declaration `self._X: Optional[T] = None` alone is not enough — that is
    how an unconditional attribute is spelled too. What marks a STAGE is that a
    real value is assigned under a condition.
    """
    built: Set[str] = set()
    for node in ast.walk(init):
        if not isinstance(node, ast.If):
            continue
        for inner in ast.walk(node):
            targets = []
            if isinstance(inner, ast.Assign):
                targets = inner.targets
            elif isinstance(inner, ast.AnnAssign):
                targets = [inner.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and target.attr.startswith("_")
                ):
                    # `= None` is the declaration, not the construction.
                    value = getattr(inner, "value", None)
                    if isinstance(value, ast.Constant) and value.value is None:
                        continue
                    built.add(target.attr)
    return built


def _read_attributes(tree: ast.AST, init: ast.AST) -> Set[str]:
    """`self._X` read anywhere in the class EXCEPT inside `__init__`.

    Excluding `__init__` is the load-bearing part: a stage's own construction
    reads the attribute it just set (`if self._x is not None`), so counting those
    would make every stage its own consumer — the trap PC.1b's invariant took four
    rounds to escape, in a new costume.
    """
    init_nodes = {id(n) for n in ast.walk(init)}
    read: Set[str] = set()
    for node in ast.walk(tree):
        if id(node) in init_nodes:
            continue
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.ctx, ast.Load)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            read.add(node.attr)
    return read


def test_every_conditionally_built_stage_is_read() -> None:
    tree = ast.parse(inspect.getsource(FilteringWorkflow))
    init = _init_node(tree)

    built = _conditionally_built(init)
    assert len(built) >= 8, (
        f"detector control: found only {len(built)} conditionally-built stages "
        f"({sorted(built)}) — the scan is not seeing the constructor"
    )

    read = _read_attributes(tree, init)
    assert "_kg_resolver" in read, "detector control: a known-read stage is missing"

    unused = sorted(a for a in built if a not in read and a not in _ALLOWED)
    assert not unused, (
        f"built under a condition and never read by the class: {unused}. Either "
        f"use it, delete it, or — if the value leaves the class — add it to "
        f"_ALLOWED with the destination named. A stage constructed and never "
        f"consulted is a flag that is on and does nothing."
    )


def test_the_detector_catches_a_planted_orphan_stage() -> None:
    """Mutant control: the scan must be able to fail.

    Without it, the test above passes equally well against a detector that finds
    nothing conditionally built — which is how a guard reports as a guard while
    being unable to.
    """
    planted = ast.parse(
        "class W:\n"
        "    def __init__(self, cfg):\n"
        "        self._used = None\n"
        "        self._orphan = None\n"
        "        if cfg.a:\n"
        "            self._used = 1\n"
        "        if cfg.b:\n"
        "            self._orphan = 2\n"
        "    def run(self):\n"
        "        return self._used\n"
    )
    init = _init_node(planted)
    built = _conditionally_built(init)
    read = _read_attributes(planted, init)
    assert built == {"_used", "_orphan"}
    assert "_used" in read and "_orphan" not in read


def test_a_declaration_alone_is_not_a_construction() -> None:
    """`self._x: Optional[T] = None` under an `if` is not a built stage.

    Counting it would report every declaration as an orphan and the guard would be
    disabled the first time it ran.
    """
    planted = ast.parse(
        "class W:\n"
        "    def __init__(self, cfg):\n"
        "        if cfg.a:\n"
        "            self._declared = None\n"
    )
    assert _conditionally_built(_init_node(planted)) == set()
