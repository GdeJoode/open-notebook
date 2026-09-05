"""Stage 10 is off in the default path, and the config SAYS which way (PC.3).

PC.3 turned it on, measured it against the whole active graph, and turned it back
off. The reason is not that the stage is bad — it is that its answer has nowhere
to go. `kg_entity_id`, `kg_match_type` and `kg_similarity_score` have no consumer
anywhere in the production tree, and persistence identifies the entity by
`name_key` exactly as it does with the stage off. So the verdict is written into
the properties bag and never read.

Measured across all five active entity types: 18 merges over 531 entities, ~5
defensible. Seven of the eight clear errors are correct RELATIONS recorded as
identity — `coöperatief wonen` is narrower than `wonen`, the Staatssecretaris is
an organ of IenW, VROM is the predecessor of VRO. The signal is real and the
destination is missing.

THIS GUARD IS DIRECTION-AGNOSTIC ON PURPOSE. It does not assert "off" as a value
someone can flip back by editing one word; it asserts that the app STATES the
choice explicitly, and that flipping it back is a deliberate edit to a keyword
carrying the measurement's reason. Pinning `False` alone would make the next
person's re-enable look like a test failure rather than a decision; pinning
nothing would let the keyword vanish and reintroduce the silent default PC.3 was
opened to fix.

So: the keyword must be present, it must carry an explicit `enabled`, and when it
is True the guard demands the destination that is missing today. That is the
condition under which the stage should come back.
"""

from __future__ import annotations

import ast
import inspect
from typing import List

from app_main.services import entity_extraction_service


def _filtering_config_calls() -> List[ast.Call]:
    tree = ast.parse(inspect.getsource(entity_extraction_service))
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "FilteringConfig"
        and node.keywords  # the bare `FilteringConfig()` fallback is a different site
    ]


def test_the_app_states_the_kg_resolution_choice_explicitly() -> None:
    """The keyword must be there and must say which way — silence is the defect."""
    calls = _filtering_config_calls()
    assert calls, "walker control: no keyword FilteringConfig(...) call found"

    for call in calls:
        keywords = {k.arg for k in call.keywords}
        assert "kg_resolution" in keywords, (
            f"the FilteringConfig at line {call.lineno} does not set "
            f"`kg_resolution` at all. Omitting it is how stage 10 sat at a "
            f"config default nothing set — the silent state PC.3 was opened to "
            f"end. State the choice, either way."
        )
        kg = next(k.value for k in call.keywords if k.arg == "kg_resolution")
        enabled = [k for k in getattr(kg, "keywords", []) if k.arg == "enabled"]
        assert enabled, (
            f"line {call.lineno}: kg_resolution passed without an explicit "
            f"`enabled`, so the behaviour comes from a default three files away"
        )
        assert isinstance(getattr(enabled[0].value, "value", None), bool), (
            f"line {call.lineno}: `enabled` must be a literal True or False"
        )


def _reads_the_key(path: "Path", key: str) -> bool:
    """Does this module READ `key`, in code?

    A substring search does not answer that, and the first version of this guard
    proved it: it reported two "readers", and both were prose — the comment in
    `entity_extraction_service` explaining why the stage is off, and the
    measurement script that names the key in its own docstring. Matching a string
    that also appears in commentary is the same defect this session already found
    in an `"ORDER BY" in getsource(...)` check.

    AST does not see comments at all, which removes that class by construction.
    Docstrings survive as `ast.Constant`, so they are dropped explicitly.
    """
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return False

    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                docstrings.add(id(body[0].value))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value == key
            and id(node) not in docstrings
        ):
            return True
    return False


def test_re_enabling_requires_the_destination_that_is_missing() -> None:
    """If someone turns stage 10 back on, its output must reach something.

    This is a condition, not a veto. Today `kg_entity_id` is read by nothing:
    `entity_persistence_service` contains zero occurrences of `kg_`, so the
    verdict is written into the properties bag and never used. Turning the stage
    on without a destination re-creates exactly what PC.3 measured — a stage that
    runs, costs a candidate fetch plus Levenshtein and cosine per entity, and
    changes nothing.

    Derived, not sampled: it searches the production tree for a reader rather
    than naming the file that ought to become one. `scripts/` is excluded — a
    measurement that prints the key is not a destination for it.
    """
    import subprocess
    from pathlib import Path

    repo = Path(__file__).resolve().parents[3]
    enabled_anywhere = any(
        getattr(kw.value, "value", None) is True
        for call in _filtering_config_calls()
        for kg in [next((k.value for k in call.keywords if k.arg == "kg_resolution"), None)]
        if kg is not None
        for kw in getattr(kg, "keywords", [])
        if kw.arg == "enabled"
    )
    if not enabled_anywhere:
        return  # off, and the measurement says that is correct today

    listed = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=repo, capture_output=True, text=True,
    ).stdout.splitlines()
    readers = [
        rel
        for rel in listed
        if "/tests/" not in rel
        and not Path(rel).name.startswith("test_")
        and "kg_resolver.py" not in rel          # the producer is not a reader
        and not rel.startswith("scripts/")       # a measurement is not a destination
        and _reads_the_key(repo / rel, "kg_entity_id")
    ]

    assert readers, (
        "stage 10 is enabled but `kg_entity_id` is read by nothing outside the "
        "resolver, so its verdict is written to the graph and never used. Give "
        "it a destination first — a relation proposal, or the curator queue a "
        "human already reads. See "
        "docs/tracks/PC-pipeline-coherence/phase-PC.3-measurement.md"
    )


def test_the_reader_check_is_not_satisfied_by_prose() -> None:
    """The guard above must not be satisfied by a comment or a docstring.

    Both false readers the first version found were exactly that. This pins the
    fix rather than trusting it.
    """
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        prose = Path(tmp) / "prose.py"
        prose.write_text(
            '"""A module docstring naming kg_entity_id."""\n'
            "# a comment naming kg_entity_id\n"
            "def f():\n"
            '    """kg_entity_id in a function docstring."""\n'
            "    return 1\n"
        )
        assert not _reads_the_key(prose, "kg_entity_id"), (
            "prose satisfied the reader check"
        )

        real = Path(tmp) / "real.py"
        real.write_text(
            "def f(props):\n"
            '    return props.get("kg_entity_id")\n'
        )
        assert _reads_the_key(real, "kg_entity_id"), (
            "a genuine read was not recognised"
        )


def test_the_alias_policy_is_not_restated_here() -> None:
    """`register_aliases` must come from PC.2's default, not be set again.

    PC.2 settled that a fuzzy match may not register an alias by itself, and put
    that in `KGResolutionConfig`. Restating it at a call site is how one decision
    grows two answers — the exact defect PC.6 spent five rounds on. Enabling
    stage 10 is the moment that risk becomes live, because tier 1 is the alias
    tier.
    """
    from entity_filtering.config import KGResolutionConfig

    assert KGResolutionConfig().register_aliases is False

    for call in _filtering_config_calls():
        kg = next(
            (k.value for k in call.keywords if k.arg == "kg_resolution"), None
        )
        if kg is None:
            continue
        restated = {k.arg for k in getattr(kg, "keywords", [])} & {"register_aliases"}
        assert not restated, (
            f"line {call.lineno} restates {restated}; leave it to "
            f"KGResolutionConfig so the policy lives in one place"
        )
