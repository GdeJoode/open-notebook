"""Every production writer of `entity` derives `name_key` the same way (PC.3).

Three code paths create entity rows: `EntityRepository.upsert_entity`,
`vault_sync_service`, and the extraction service's opt-in `write_to_db` (which
defaults to True and is exposed that way by the app proxy). Migration 79 puts a
UNIQUE index on `(name_key, entity_type)`, so if two writers normalise one name
differently they produce two identities for it — the exact duplication this phase
exists to remove, reintroduced by the fix.

The guard is DERIVED, not sampled: it finds every `CREATE entity` in production
source by AST/regex and asserts each supplies `name_key` from
`normalize_entity_name`. A writer added later is covered by existing.

DECLARED LIMITS, found by review against the structural rewrite and left open
because no writer in the tree has these shapes:

* two `CREATE … SET` statements in one string — the first vouches for the second,
  because `_set_body` stops at the first `;`;
* `name_key` assigned only inside a nested subquery in the SET body;
* a lookup written as `UPDATE entity … WHERE canonical_name = $n` (the lookup
  guard treats only `SELECT` as a lookup);
* `WHERE canonical_name = $n OR name_key = $k` — an identity term anywhere in the
  clause satisfies it, even in an OR branch;
* `FROM type::table('entity')` rather than `FROM entity`;
* SQL assembled into a variable before the call, which is invisible to both
  guards.

Test fixtures are deliberately out of scope. They use a SurrealQL-side
`string::lowercase(string::trim(...))`, which equals `normalize_entity_name` for
names without a leading article or a curated org alias — true of invented fixture
names, and false for `De Gemeente` (→ `gemeente`) or `BZK` (→ the expanded form).
That approximation is safe in a fixture and would not be in production, which is
why this scan covers production only and says so.
"""

from __future__ import annotations

import ast
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Production files that create an `entity` row without calling
#: `normalize_entity_name`, with the reason. Empty, and it should stay hard to
#: add to: an entry here is a second identity rule.
_ALLOWED: Dict[str, str] = {}

#: File-level filter: does this file mention creating an entity at all?
_CREATE = re.compile(r"CREATE entity\b(?!_)")

#: Statement-level. A call whose literals merely CONTAIN the words is not a
#: write — `RuntimeError("CREATE entity returned no rows")` sits three lines
#: below the real one in `entity.py` and was flagged by the first version of
#: this detector. SurrealQL creates a row with `SET` or `CONTENT`, or with
#: `INSERT INTO`; nothing else writes.
_CREATE_STMT = re.compile(
    r"\b(?:CREATE\s+entity\b(?!_)\s*(?:SET|CONTENT)\b"
    r"|INSERT\s+INTO\s+entity\b(?!_))",
    re.I,
)


def _production_sources() -> List[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout.splitlines()
    return [
        REPO_ROOT / rel
        for rel in listed
        # `scripts/` counts. `semantic-intelligence/scripts/test_pipeline.py` is
        # a documented entry point (named in the package's own __init__) that
        # writes real entity rows, and it was invisible to this guard for two
        # independent reasons: it is not under `/src/`, and its filename starts
        # with `test_`. It was found by the entity_alias vocabulary guard, which
        # walks a wider tree — a scope difference between two guards over the
        # same question is itself a defect.
        if ("/src/" in rel or rel.startswith("services/") or "scripts/" in rel)
        # Exclude by PATH, not by filename: a file is a test because of where it
        # lives, not because of what it is called.
        and "/tests/" not in rel
        and not rel.startswith("tests/")
    ]


def _sql_text(node: ast.Call) -> str:
    """The QUERY literal of a call — its first positional argument, only.

    Split out from the rest because the first version of this guard joined every
    string constant in the call, INCLUDING the keys of the params dict. So
    `execute_query("CREATE entity SET name = $name, …", {"name_key": …})` reported
    `name_key` as present when the SQL no longer set it. Deleting
    `name_key = $name_key` from `vault_sync_service`'s statement left this guard
    green; the params key stood in for the column. Verified by doing it.
    """
    if not node.args:
        return ""
    return " ".join(
        sub.value
        for sub in ast.walk(node.args[0])
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    )


def _payload_keys(node: ast.Call) -> Set[str]:
    """Dict-literal KEYS in everything after the query.

    `CREATE entity CONTENT $data` names its columns in the payload rather than in
    the statement, so for that form the keys ARE the write. For the `SET` form
    they are parameter bindings and prove nothing — which is the distinction the
    first version collapsed.
    """
    keys: Set[str] = set()
    for arg in list(node.args[1:]) + [kw.value for kw in node.keywords]:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Dict):
                for key in sub.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
    return keys


def _literal_text(node: ast.Call) -> str:
    """Every string literal in the call — used only to FIND write statements."""
    return " ".join(
        sub.value
        for sub in ast.walk(node)
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str)
    )


#: Clause extractors. Both guards previously asked whether a token APPEARED in
#: the query; the questions are structural — "is the column ASSIGNED" and "is the
#: lookup KEYED on it" — and a token can appear in a projection, a sibling dict or
#: an unrelated clause without answering either. Review demonstrated all three.
_SET_CLAUSE = re.compile(r"\bSET\b(?P<body>.*?)(?:\bRETURN\b|;|$)", re.I | re.S)
_WHERE_CLAUSE = re.compile(
    r"\bWHERE\b(?P<body>.*?)(?:\bLIMIT\b|\bORDER\b|\bGROUP\b|\bFETCH\b|;|$)",
    re.I | re.S,
)


def _assigns(clause: str, column: str) -> bool:
    """Is `column` given a value here — not merely mentioned, not compared?

    `=` and not `==`: review pointed out that `SET flag = (name_key == $k)`
    satisfied the previous predicate, which both guards rest on. SurrealQL
    compares with `==` (and `!=`, `<=`, `>=`), so a lookahead excluding a second
    `=` is the difference between an assignment and a test.
    """
    return bool(
        re.search(rf"(?<![_a-zA-Z!<>=]){re.escape(column)}\s*=(?!=)", clause)
    )


def _set_body(sql: str) -> str:
    m = _SET_CLAUSE.search(sql)
    return m.group("body") if m else ""


def _where_body(sql: str) -> str:
    m = _WHERE_CLAUSE.search(sql)
    return m.group("body") if m else ""


def _content_payload_keys(node: ast.Call, sql: str) -> Set[str]:
    """Keys of the dict bound to the param the CONTENT clause names.

    `CONTENT $data` means the payload is whatever `$data` holds, so the guard
    resolves THAT key rather than walking every dict in the call: review showed a
    sibling dict (`{"data": {...}, "audit": {"name_key": k}}`) vouching for a
    payload that did not set the column.
    """
    m = re.search(r"\bCONTENT\s+\$(?P<param>\w+)", sql, re.I)
    if not m:
        return set()
    param = m.group("param")
    keys: Set[str] = set()
    for arg in list(node.args[1:]) + [kw.value for kw in node.keywords]:
        for sub in ast.walk(arg):
            if not isinstance(sub, ast.Dict):
                continue
            for key, value in zip(sub.keys, sub.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == param
                    and isinstance(value, ast.Dict)
                ):
                    for inner in value.keys:
                        if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                            keys.add(inner.value)
    return keys


def _enclosing_functions(tree: ast.AST) -> List[Tuple[int, int, ast.AST]]:
    spans: List[Tuple[int, int, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno)
            spans.append((node.lineno, end, node))
    # Innermost first, so a nested function wins over its parent.
    spans.sort(key=lambda s: s[1] - s[0])
    return spans


def _innermost(spans: List[Tuple[int, int, ast.AST]], line: int) -> Optional[ast.AST]:
    for lo, hi, node in spans:
        if lo <= line <= hi:
            return node
    return None


def _calls_the_one_rule(scope: ast.AST) -> bool:
    for sub in ast.walk(scope):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if isinstance(fn, ast.Name) and fn.id == "normalize_entity_name":
            return True
        if isinstance(fn, ast.Attribute) and fn.attr == "normalize_entity_name":
            return True
    return False


def find_identity_violations(path: str, source: str) -> List[str]:
    """One message per `CREATE entity` site that does not state its identity.

    Statement-level on purpose. The first version of this guard asked whether the
    FILE mentioned `normalize_entity_name` anywhere, which is a proxy for the
    question and not the question: deleting `name_key = $name_key` from the real
    production CREATE left the import in place and the suite green. Verified by
    doing exactly that, against `entity.py` and the semantic-intelligence script.
    """
    problems: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{path}: unparseable ({exc})"]

    spans = _enclosing_functions(tree)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        text = _literal_text(node)
        if not _CREATE_STMT.search(text):
            continue
        line = node.lineno

        sql = _sql_text(node)
        # SurrealQL names the columns in the statement for `SET`, and in the
        # bound payload for `CONTENT $data`. Ask the right one of the two, and
        # ask whether the column is ASSIGNED rather than whether the word
        # appears: `CREATE entity SET canonical_name = $n RETURN id, name_key`
        # mentions it in a projection and sets nothing.
        # `CONTENT` as a CLAUSE, not as a column name. `\bCONTENT\b` matched
        # `CREATE entity SET content = $c, …, name_key = $k` and routed a correct
        # writer to the payload branch, refusing it with a message about a
        # payload it does not have. The clause is always `CONTENT $param`.
        if re.search(r"\bCONTENT\s+\$\w+", sql, re.I):
            states_identity = "name_key" in _content_payload_keys(node, sql)
            where_it_should_be = "the CONTENT payload"
        else:
            states_identity = _assigns(_set_body(sql), "name_key")
            where_it_should_be = "the SET statement"

        if not states_identity:
            problems.append(
                f"{path}:{line}: `CREATE entity` does not set `name_key` in "
                f"{where_it_should_be}. Migration 79 makes it the identity, "
                f"TYPE string with no default, so this write is rejected "
                f"outright."
            )
            continue

        fn = _innermost(spans, line)
        scope = fn if fn is not None else tree
        if not _calls_the_one_rule(scope):
            where = getattr(fn, "name", "<module>")
            problems.append(
                f"{path}:{line}: `CREATE entity` sets `name_key`, but "
                f"`{where}` never calls `normalize_entity_name` — the value "
                f"comes from somewhere else, which is a second identity rule."
            )
    return problems


def test_every_entity_writer_uses_the_one_identity_rule() -> None:
    sources = _production_sources()
    assert sources, "walker control: no production sources found"

    writers = [p for p in sources if _CREATE.search(p.read_text(encoding="utf-8"))]
    assert len(writers) >= 4, (
        f"detector control: found only {len(writers)} entity writers "
        f"({[p.name for p in writers]}) — the scan is not seeing them"
    )

    offenders: List[str] = []
    for path in writers:
        rel = str(path.relative_to(REPO_ROOT))
        if rel in _ALLOWED:
            continue
        offenders += find_identity_violations(rel, path.read_text(encoding="utf-8"))

    assert not offenders, (
        "these create `entity` rows without deriving `name_key` from "
        "`normalize_entity_name`:\n" + "\n".join(offenders)
    )


#: Three states a file-level guard passed. Each is a real writer's shape put in
#: the state this guard claims to forbid — not a state of that shape.
_FORBIDDEN_STATES = {
    "name_key deleted from the CREATE, import left in place (entity.py)": (
        "from shared.utils.name_normalizer import normalize_entity_name\n"
        "class EntityRepository:\n"
        "    async def upsert_entity(self, entity):\n"
        "        name_key = normalize_entity_name(entity.canonical_name)\n"
        "        result = await execute_query(\n"
        "            'CREATE entity SET canonical_name = $canonical_name, '\n"
        "            'entity_type = $entity_type, confidence = $confidence',\n"
        "            params,\n"
        "        )\n"
    ),
    "name_key deleted from the CONTENT payload (semantic-intelligence script)": (
        "from shared.utils.name_normalizer import normalize_entity_name\n"
        "async def ingest(name, etype, desc):\n"
        "    result = await execute_query(\n"
        "        'CREATE entity CONTENT $data RETURN id',\n"
        "        {'data': {'canonical_name': name, 'entity_type': etype}},\n"
        "    )\n"
    ),
    "name_key set from something that is not the one rule": (
        "async def ingest(name, etype):\n"
        "    result = await execute_query(\n"
        "        'CREATE entity SET canonical_name = $n, name_key = $k',\n"
        "        {'n': name, 'k': name.lower().strip()},\n"
        "    )\n"
    ),
}


def test_guard_fails_in_each_state_it_forbids() -> None:
    """The half a green suite does not give you.

    The first version of this guard passed all three of these. It was verified
    by running the suite after a mutation, which is the same mistake one level
    up: the suite was green because the guard could not see the statement.
    """
    for label, source in _FORBIDDEN_STATES.items():
        assert find_identity_violations(label, source), f"guard did NOT flag: {label}"


def test_guard_accepts_each_real_writer_as_it_stands() -> None:
    """The mirror: a guard that flags everything would satisfy the test above."""
    for rel in (
        "packages/surrealdb-service/src/surrealdb_service/repositories/entity.py",
        "apps/app-main/src/app_main/services/vault_sync_service.py",
        "services/extraction/api.py",
        "packages/semantic-intelligence/scripts/test_pipeline.py",
    ):
        path = REPO_ROOT / rel
        assert path.exists(), f"writer moved: {rel}"
        assert not find_identity_violations(rel, path.read_text(encoding="utf-8")), (
            f"guard wrongly flagged the corrected writer: {rel}"
        )


def test_the_writers_agree_on_adversarial_names() -> None:
    """One function, so the answers must be identical — including where it bites.

    These are the shapes where `normalize_entity_name` does more than lowercase,
    and where a writer using a naive fold would silently disagree.
    """
    from shared.utils.name_normalizer import normalize_entity_name

    for name in (
        "De Gemeente Leudal",
        "BZK",
        "Ministerie van BZK",
        "  Brede   Welvaart  ",
        "Regio Deal Groningen",
    ):
        key = normalize_entity_name(name)
        assert key == normalize_entity_name(name), "not deterministic"
        assert key == key.strip() and "  " not in key, f"unclean key: {key!r}"
        assert key == normalize_entity_name(key), (
            f"not idempotent: {name!r} -> {key!r} -> "
            f"{normalize_entity_name(key)!r}; a re-normalised key must be itself, "
            f"or a second write of the same row lands on a different identity"
        )


def test_guard_ignores_prose_that_merely_names_the_statement() -> None:
    """A message is not a write.

    `entity.py` raises `RuntimeError(f"CREATE entity returned no rows for ...")`
    three lines below the real CREATE. The first version of this detector flagged
    it, which would have made the guard unpassable and therefore useless.
    """
    prose = (
        "def f():\n"
        "    if not result:\n"
        "        raise RuntimeError('CREATE entity returned no rows for ' + name)\n"
    )
    assert not find_identity_violations("prose", prose)


#: A `SELECT ... FROM entity WHERE ...` that exists to decide whether to CREATE.
_ENTITY_LOOKUP = re.compile(r"\bSELECT\b[^;]*\bFROM\s+entity\b(?!_)[^;]*\bWHERE\b", re.I)
#: The columns a lookup may key on. `name_key` is the identity; anything else is
#: a DISPLAY form, and migration 79 made those two different keys.
#: `hash_id` is `md5(f"{canonical_name}|{entity_type}")` — keying a lookup on it
#: is keying on the display name with an extra step, which review pointed out.
_DISPLAY_KEYS = ("canonical_name", "name", "hash_id")


def find_lookup_violations(path: str, source: str) -> List[str]:
    """A writer must look up on the key it writes.

    Migration 79 split identity (`name_key`) from display (`canonical_name` /
    `name`). A module that CREATEs entities but looks up on a display column now
    misses on any variant the identity rule folds — and then CREATEs, hitting
    `idx_entity_identity`. In `services/extraction/api.py` that exception unwound
    the whole per-document loop and the endpoint returned HTTP 200 having written
    nothing, relations included. Found by review, not by this suite, which is why
    the rule is here rather than in a comment.

    Scoped to modules that CREATE entities: a read-only consumer may legitimately
    look up a display name.
    """
    problems: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"{path}: unparseable ({exc})"]

    creates = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _CREATE_STMT.search(_literal_text(n))
    ]
    if not creates:
        return []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        sql = _sql_text(node)
        if not _ENTITY_LOOKUP.search(sql):
            continue
        where = _where_body(sql)
        # Keyed on the identity? Then fine. Asking whether `name_key` appears
        # ANYWHERE in the query was the previous rule, and review broke it with
        # `SELECT id, name_key FROM entity WHERE canonical_name = $n` — the
        # identity in the projection exempting a lookup on the display column,
        # which is blocker B2 exactly.
        if _assigns(where, "name_key"):
            continue
        keyed_on = [k for k in _DISPLAY_KEYS if _assigns(where, k)]
        if keyed_on:
            problems.append(
                f"{path}:{node.lineno}: this module CREATEs entities but looks "
                f"one up on {keyed_on} rather than `name_key`. Migration 79 made "
                f"those different keys, so the lookup misses on a variant and "
                f"the CREATE then collides with `idx_entity_identity`."
            )
    return problems


def test_every_entity_writer_looks_up_on_the_key_it_writes() -> None:
    sources = _production_sources()
    offenders: List[str] = []
    for path in sources:
        rel = str(path.relative_to(REPO_ROOT))
        if rel in _ALLOWED:
            continue
        offenders += find_lookup_violations(rel, path.read_text(encoding="utf-8"))
    assert not offenders, (
        "these decide whether to CREATE by asking about the DISPLAY name:\n"
        + "\n".join(offenders)
    )


def test_the_lookup_guard_fails_on_the_real_prefix_shape() -> None:
    """The two writers as review found them, verbatim."""
    for label, source in {
        "services/extraction/api.py": (
            "async def w(db, name, etype):\n"
            "    existing = await db.query(\n"
            '        "SELECT id FROM entity WHERE canonical_name = $name '
            'AND entity_type = $type LIMIT 1;",\n'
            '        {"name": name, "type": etype},\n'
            "    )\n"
            "    await db.query(\n"
            '        "CREATE entity CONTENT $data RETURN id;",\n'
            '        {"data": {"name_key": nk(name)}},\n'
            "    )\n"
        ),
        "vault_sync_service.py": (
            "async def w(name):\n"
            "    rows = await execute_query(\n"
            '        "SELECT id FROM entity WHERE name = $name LIMIT 1",\n'
            '        {"name": name},\n'
            "    )\n"
            "    await execute_query(\n"
            '        "CREATE entity SET name = $name, name_key = $name_key",\n'
            '        {"name": name, "name_key": nk(name)},\n'
            "    )\n"
        ),
    }.items():
        assert find_lookup_violations(label, source), f"not flagged: {label}"


def test_the_lookup_guard_accepts_a_module_that_only_reads() -> None:
    """A consumer that never CREATEs may look up whatever it likes."""
    reader = (
        "async def show(name):\n"
        '    return await execute_query(\n'
        '        "SELECT * FROM entity WHERE canonical_name = $n LIMIT 1", {"n": name}\n'
        "    )\n"
    )
    assert not find_lookup_violations("reader.py", reader)


#: The evasions review found in the NARROWED guards. Each satisfied the previous
#: version by putting the token somewhere that does not answer the question.
_EVASIONS = {
    "B1 — the identity in the PROJECTION, the lookup on the display column": (
        "async def w(name):\n"
        "    rows = await execute_query(\n"
        '        "SELECT id, name_key FROM entity WHERE canonical_name = $n LIMIT 1",\n'
        '        {"n": name},\n'
        "    )\n"
        "    await execute_query(\n"
        '        "CREATE entity SET canonical_name = $n, name_key = $k",\n'
        '        {"n": name, "k": nk(name)},\n'
        "    )\n",
        "lookup",
    ),
    "A1 — `name_key` in a RETURN clause, assigned nowhere": (
        "async def w(name):\n"
        "    await execute_query(\n"
        '        "CREATE entity SET canonical_name = $n RETURN id, name_key",\n'
        '        {"n": name},\n'
        "    )\n",
        "identity",
    ),
    "A3 — a SIBLING dict vouching for the CONTENT payload": (
        "async def w(name):\n"
        "    await execute_query(\n"
        '        "CREATE entity CONTENT $data RETURN id",\n'
        '        {"data": {"canonical_name": name},\n'
        '         "audit": {"name_key": nk(name)}},\n'
        "    )\n",
        "identity",
    ),
    "B2 — keyed on `hash_id`, the display name with an extra step": (
        "async def w(name, h):\n"
        "    rows = await execute_query(\n"
        '        "SELECT id FROM entity WHERE hash_id = $h LIMIT 1", {"h": h}\n'
        "    )\n"
        "    await execute_query(\n"
        '        "CREATE entity SET canonical_name = $n, name_key = $k",\n'
        '        {"n": name, "k": nk(name)},\n'
        "    )\n",
        "lookup",
    ),
}


def test_the_guards_catch_every_evasion_review_found() -> None:
    """The narrowing must not have traded one blind spot for a smaller one.

    Each of these satisfied the guard AFTER round 2's fix, because the token was
    present in a place that does not answer the question — a projection, a RETURN
    clause, a sibling dict, a derived column. B1 is the one that matters: it is
    blocker B2 wearing a projection.
    """
    for label, (source, which) in _EVASIONS.items():
        found = (
            find_identity_violations(label, source)
            if which == "identity"
            else find_lookup_violations(label, source)
        )
        assert found, f"guard did NOT flag: {label}"


def test_the_guards_still_accept_the_real_writers() -> None:
    """The mirror: narrowing must not start flagging correct code either."""
    for rel in (
        "packages/surrealdb-service/src/surrealdb_service/repositories/entity.py",
        "apps/app-main/src/app_main/services/vault_sync_service.py",
        "services/extraction/api.py",
        "packages/semantic-intelligence/scripts/test_pipeline.py",
    ):
        path = REPO_ROOT / rel
        source = path.read_text(encoding="utf-8")
        assert not find_identity_violations(rel, source), f"identity guard: {rel}"
        assert not find_lookup_violations(rel, source), f"lookup guard: {rel}"


def test_a_comparison_is_not_an_assignment() -> None:
    """`==` satisfied the predicate both guards rest on (review, round 3)."""
    assert _assigns("SET name_key = $k", "name_key")
    assert not _assigns("SET flag = (name_key == $k)", "name_key")
    assert not _assigns("WHERE name_key != $k", "name_key")
    assert not _assigns("WHERE name_key >= $k", "name_key")


def test_a_column_named_content_is_not_a_CONTENT_clause() -> None:
    """A correct SET writer must not be routed to the payload branch.

    `\\bCONTENT\\b` matched the column and refused correct code with a message
    about a payload the statement does not have.
    """
    src = (
        "async def w(name, c):\n"
        "    await execute_query(\n"
        '        "CREATE entity SET content = $c, canonical_name = $n, '
        'name_key = $k",\n'
        '        {"c": c, "n": name, "k": normalize_entity_name(name)},\n'
        "    )\n"
    )
    assert not find_identity_violations("content-column.py", src)
