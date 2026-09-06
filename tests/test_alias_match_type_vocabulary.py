"""Every writer of `entity_alias` states a `match_type` the schema accepts.

WHAT THIS EXISTS FOR. `entity_alias.match_type` is a closed vocabulary — three
values, asserted by the database (migration 80). Three of the four production
writers violated it, and the violations were invisible for two reasons: the
failure is a runtime rejection that two call sites swallow, and the field had
TWO definitions. Migration 78 declared it `option<string>` with `IF NOT EXISTS`,
which is a no-op wherever the field already exists, so a long-lived database kept
a stricter `TYPE string ASSERT $value INSIDE [...]` that appears in no migration.
The same write then succeeded on a fresh database and failed on the one holding
the data — which is why every test and container check passed.

THE TWO RULES THIS GUARD IS BUILT AGAINST (from PC.6):

1. **Derive the space, do not sample it.** The vocabulary is read from the
   migration, not typed here. The write sites are found by walking every
   production module, not listed here. Adding a fifth writer puts it in scope
   automatically; that is the whole point.
2. **Verify the guard in the state it forbids.** `test_guard_fails_on_the_real_
   historical_sources` feeds the detector the LITERAL pre-fix text of all three
   defective sites and asserts each is flagged. Not text of that shape — the text
   that was actually in the repository.

DECLARED LIMITS. A `match_type` whose value is computed at runtime from a
non-literal is out of reach: the guard checks literals. `kg_resolver` forwards a
parameter, and the literals that reach it (`"fuzzy"`, `"semantic"`) are caught at
its own call sites, which are literal. A writer that builds its SQL from a name
this module cannot resolve to a string is not covered, and no such writer exists
today.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO = Path(__file__).resolve().parents[1]

#: Production trees. Tests are excluded — a test may legitimately exercise a
#: rejected value.
_ROOTS = ("apps", "packages", "pipelines", "services")
_EXCLUDED_PARTS = {"tests", "test", "__pycache__", ".venv", "node_modules"}

_MIGRATION = REPO / "migrations" / "80.surrealql"

#: Any call that creates a row in the table. Both spellings are in the tree.
_WRITE_SQL = re.compile(r"\b(?:CREATE|INSERT\s+INTO)\s+entity_alias\b", re.I)
#: `match_type = 'x'` (SurrealQL) or `match_type: $x` / `match_type="x"` (Python).
_MATCH_TYPE_LITERAL = re.compile(r"match_type\s*[:=]\s*['\"]([A-Za-z_]+)['\"]")
_MATCH_TYPE_PRESENT = re.compile(r"\bmatch_type\b")


# --- deriving the vocabulary ---------------------------------------------------


def allowed_vocabulary() -> Set[str]:
    """Read the closed set from the migration that enforces it."""
    sql = _MIGRATION.read_text(encoding="utf-8")
    m = re.search(
        r"DEFINE FIELD OVERWRITE match_type ON entity_alias.*?INSIDE\s*\[([^\]]+)\]",
        sql,
        re.S,
    )
    assert m, (
        "migration 80 must define match_type with OVERWRITE and an INSIDE "
        "assertion — this guard derives its vocabulary from that line"
    )
    return {v.strip().strip("'\"") for v in m.group(1).split(",") if v.strip()}


# --- deriving the write sites --------------------------------------------------


def _production_modules() -> List[Path]:
    out: List[Path] = []
    for root in _ROOTS:
        for path in (REPO / root).rglob("*.py"):
            if _EXCLUDED_PARTS & set(path.parts):
                continue
            out.append(path)
    return out


def _call_text(node: ast.Call) -> str:
    """Every string literal anywhere inside a call, joined.

    Handles implicit concatenation across lines and the literal parts of
    f-strings, which is how every raw-SQL site in this repository is written.
    """
    parts: List[str] = []
    for sub in ast.walk(node):
        if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            parts.append(sub.value)
    return " ".join(parts)


def _callee_name(node: ast.Call) -> str:
    fn = node.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return ""


def find_violations(path: str, source: str, allowed: Set[str]) -> List[str]:
    """Return one message per offending site. Empty list = clean.

    Takes the source as text so the guard can be run against a state that is not
    on disk — see the historical-replay test.
    """
    problems: List[str] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:  # a file we cannot parse is a finding, not a pass
        return [f"{path}: unparseable ({exc})"]

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _callee_name(node)
        text = _call_text(node)
        line = getattr(node, "lineno", 0)

        # (a) raw SQL that creates an entity_alias row
        if _WRITE_SQL.search(text):
            if not _MATCH_TYPE_PRESENT.search(text):
                problems.append(
                    f"{path}:{line}: creates an entity_alias row without "
                    f"match_type — the field is TYPE string on a migrated "
                    f"database and the write is rejected there"
                )
            else:
                for value in _MATCH_TYPE_LITERAL.findall(text):
                    if value not in allowed:
                        problems.append(
                            f"{path}:{line}: match_type={value!r} is outside the "
                            f"vocabulary {sorted(allowed)}"
                        )

        # (b) register_alias(..., match_type="x", ...)
        # (c) _maybe_register_alias(id, text, "x", ...) — third positional
        if name in ("register_alias", "_maybe_register_alias"):
            literals: List[str] = []
            for kw in node.keywords:
                if kw.arg == "match_type" and isinstance(kw.value, ast.Constant):
                    if isinstance(kw.value.value, str):
                        literals.append(kw.value.value)
            if name == "_maybe_register_alias" and len(node.args) >= 3:
                third = node.args[2]
                if isinstance(third, ast.Constant) and isinstance(third.value, str):
                    literals.append(third.value)
            for value in literals:
                if value not in allowed:
                    problems.append(
                        f"{path}:{line}: {name}(match_type={value!r}) is outside "
                        f"the vocabulary {sorted(allowed)}"
                    )
    return problems


# --- controls ------------------------------------------------------------------


def test_walker_control_reaches_the_known_writers() -> None:
    """The walk must actually reach the modules this guard is about.

    A guard that scans nothing passes vacuously. These four are the writers
    measured against the live database; the point is not that the list is
    complete but that a walk returning none of them is broken.
    """
    seen = {str(p.relative_to(REPO)) for p in _production_modules()}
    for expected in (
        "apps/app-main/src/app_main/services/entity_resolution/"
        "recanonicalization_service.py",
        "apps/app-main/src/app_main/services/vault_sync_service.py",
        "pipelines/entity-filtering/src/entity_filtering/deduplication/"
        "canonical_entities.py",
        "services/extraction/api.py",
    ):
        assert expected in seen, f"walker never reached {expected}"


def test_detector_control_finds_the_sites_it_must_judge() -> None:
    """The detector must recognise real write sites, not merely fail to object.

    Without this, a detector whose regex matches nothing reports every file
    clean and the suite is green for the wrong reason.
    """
    found: List[Tuple[str, int]] = []
    for path in _production_modules():
        source = path.read_text(encoding="utf-8")
        if "entity_alias" not in source and "register_alias" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                text = _call_text(node)
                if _WRITE_SQL.search(text) or _callee_name(node) in (
                    "register_alias",
                    "_maybe_register_alias",
                ):
                    found.append((str(path.relative_to(REPO)), node.lineno))
    assert len(found) >= 5, (
        f"detector found only {len(found)} entity_alias write sites; the "
        f"measured tree holds at least five (two vault, one dedup, one "
        f"extraction, one curator merge plus the kg_resolver forwards): {found}"
    )


def test_vocabulary_is_derived_and_non_trivial() -> None:
    allowed = allowed_vocabulary()
    assert allowed == {"exact", "fuzzy", "semantic"}, (
        f"vocabulary drifted to {sorted(allowed)} — if that is intended, the "
        f"writers and this assertion move together"
    )


def test_migration_80_uses_overwrite_not_if_not_exists() -> None:
    """`IF NOT EXISTS` is what let two databases disagree in the first place."""
    sql = _MIGRATION.read_text(encoding="utf-8")
    assert "DEFINE FIELD OVERWRITE match_type ON entity_alias" in sql
    assert "DEFINE FIELD IF NOT EXISTS match_type ON entity_alias" not in sql


# --- the guard itself ----------------------------------------------------------


def test_every_production_alias_writer_states_an_accepted_match_type() -> None:
    allowed = allowed_vocabulary()
    problems: List[str] = []
    for path in _production_modules():
        source = path.read_text(encoding="utf-8")
        if "entity_alias" not in source and "register_alias" not in source:
            continue
        problems += find_violations(str(path.relative_to(REPO)), source, allowed)
    assert not problems, "entity_alias writers outside the vocabulary:\n" + "\n".join(
        problems
    )


# --- verification: the guard in the state it forbids ----------------------------

#: The LITERAL text of the three defective sites as they stood before this fix.
#: Not text of that shape — the text that was in the repository.
_HISTORICAL: Dict[str, str] = {
    "recanonicalization_service.py (curator merge)": '''
async def apply_merge(self, cluster):
    ok = await self.entity_repo.register_alias(
        cluster.winner_id,
        surface,
        match_type="retroactive_merge",
        similarity_score=1.0,
        method="recanonicalization",
    )
''',
    "vault_sync_service.py (vault import)": '''
async def run(self):
    await execute_query(
        "CREATE entity_alias SET "
        "canonical_entity = $entity_id, "
        "alias_text = $alias, "
        "match_type = 'vault_import', "
        "similarity_score = 1.0, "
        "method = 'obsidian_vault', "
        "verified = true",
        {"entity_id": entity_id, "alias": alias},
    )
''',
    "canonical_entities.py (dedup merge, no match_type at all)": '''
async def run(self):
    await execute_query(
        "CREATE entity_alias SET "
        "alias_text = $name, canonical_entity = $eid, confidence = 1.0",
        {"name": dup_name, "eid": canonical_id},
    )
''',
}


def test_guard_fails_on_the_real_historical_sources() -> None:
    """Replay: each defective site, verbatim, must be flagged.

    This is the half that a green suite does not give you. All three of these
    were live in the repository and every test passed.
    """
    allowed = allowed_vocabulary()
    for label, source in _HISTORICAL.items():
        problems = find_violations(label, source, allowed)
        assert problems, f"guard did NOT flag the historical defect: {label}"


def test_guard_accepts_the_corrected_sources() -> None:
    """The mirror: the same three sites, fixed, must pass.

    Without this, a detector that flags everything would satisfy the test above.
    """
    allowed = allowed_vocabulary()
    corrected = {
        "curator merge": (
            'x = repo.register_alias(w, s, match_type="exact", '
            'similarity_score=1.0, method="recanonicalization")'
        ),
        "vault import": (
            """await q("CREATE entity_alias SET match_type = 'exact', """
            """method = 'obsidian_vault'")"""
        ),
        "dedup merge": (
            """await q("CREATE entity_alias SET alias_text = $n, """
            """match_type = 'exact', method = 'dedup_merge'")"""
        ),
        "kg_resolver tier 2": (
            'await self._maybe_register_alias(cid, text, "fuzzy", s, r)'
        ),
    }
    for label, source in corrected.items():
        assert not find_violations(label, source, allowed), (
            f"guard wrongly flagged the corrected source: {label}"
        )
