"""Every record id interpolated into SurrealQL is validated first.

SurrealDB v2 cannot bind a record id as a `$param` in the positions these
endpoints need — `FROM {id}`, `UPDATE {id}`, `WHERE in = {id}` — so the id is
interpolated. An interpolated value nobody validates is SurrealQL injection, and
this repository has already lost a table to exactly that (Track Y.1). Two
families were open here: `MergeRequest.canonical_id`/`duplicate_id` (nine sites,
proxied from app-main) and the `{entity_id:path}` parameter, whose `:path`
converter accepts slashes and every metacharacter.

THE GUARD IS DERIVED. It does not list the sites it knows about — it walks the
module's AST for every f-string that interpolates a name into a query, and
requires each interpolated name to have passed the validator. A tenth
interpolation added later is in scope automatically, which is the only version of
this check worth having.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List

SERVICE = Path(__file__).resolve().parents[1]

#: EVERY module in the service, not just `api.py`. Review found four unvalidated
#: interpolations one file over, in `entity_validator.py`, which this guard did
#: not walk — and the precedent for widening was already in the next file in this
#: directory: `test_no_dead_env_constants.py` says "Every module in the service,
#: not just `api.py`. Review found a live instance one file over", for the same
#: sibling module. That lesson was learned before this guard was written and this
#: guard pinned a single file anyway.
SOURCES = sorted(p for p in SERVICE.glob("*.py") if p.name != "__init__.py")

API = SERVICE / "api.py"

#: The rule itself lives here so `entity_validator` can share it — see the module
#: docstring. `api.py` re-exports the pattern and wraps the raise in an
#: HTTPException at the boundary.
RECORD_IDS = SERVICE / "record_ids.py"

#: A statement that reaches the database. Interpolating into a `detail=` message
#: or a log line is not an injection, so the guard asks whether the f-string
#: looks like SurrealQL.
_QUERYISH = re.compile(
    r"\b(SELECT|UPDATE|CREATE|DELETE|RELATE|INSERT|REMOVE)\b", re.I
)


def _module(path: "Path" = None) -> ast.Module:
    return ast.parse((path or API).read_text(encoding="utf-8"))


def _interpolated_names(node: ast.JoinedStr) -> List[str]:
    out: List[str] = []
    for part in node.values:
        if not isinstance(part, ast.FormattedValue):
            continue
        v = part.value
        if isinstance(v, ast.Name):
            out.append(v.id)
        elif isinstance(v, ast.Attribute) and isinstance(v.value, ast.Name):
            out.append(f"{v.value.id}.{v.attr}")
        else:
            out.append(ast.dump(v)[:40])  # anything else is reported as-is
    return out


def _query_interpolations() -> Dict[int, List[str]]:
    """Every f-string in `api.py` that looks like SurrealQL, with its names."""
    tree = _module()
    found: Dict[int, List[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        literal = "".join(
            p.value for p in node.values
            if isinstance(p, ast.Constant) and isinstance(p.value, str)
        )
        if not _QUERYISH.search(literal):
            continue
        names = _interpolated_names(node)
        if names:
            found[node.lineno] = names
    return found


def test_detector_control_sees_the_interpolations() -> None:
    """A guard that finds no query f-strings would pass vacuously."""
    found = _query_interpolations()
    assert len(found) >= 10, (
        f"the walker found only {len(found)} interpolated query strings; the "
        f"module holds at least ten: {found}"
    )


def test_the_validator_rejects_what_it_must() -> None:
    """The pattern itself, against the shapes an attacker would send."""
    src = RECORD_IDS.read_text(encoding="utf-8")
    m = re.search(r"_RECORD_ID_RE = re\.compile\(\s*r\"([^\"]+)\"", src)
    assert m, "the validator's pattern is not where this guard expects it"
    # The captured group is already valid regex source (`\u27e8` is handled by
    # `re` itself); an `unicode_escape` round trip mangles `\-`.
    pattern = re.compile(m.group(1))

    for good in ("entity:abc123", "entity:a_b-c", "entity:⟨quoted⟩"):
        assert pattern.match(good), f"rejected a legitimate id: {good!r}"
    for bad in (
        "entity:x; REMOVE TABLE entity; --",
        "entity:x WHERE 1=1",
        "entity:x' OR '1'='1",
        "../../etc/passwd",
        "entity:",
        "",
        "nocolon",
    ):
        assert not pattern.match(bad), f"ACCEPTED an injection shape: {bad!r}"



#: Names that cannot carry SurrealQL because FastAPI has already coerced them.
#: `depth`, `limit` and `offset` are `int`-typed in every signature that takes
#: them; a non-numeric value is rejected with a 422 before the handler runs.
_INT_TYPED = frozenset({"depth", "limit", "offset"})

#: Names that hold a QUERY FRAGMENT rather than a value. These are in scope: a
#: fragment is safe only if it was assembled by binding, never by interpolation.
_FRAGMENTS = frozenset({"where"})


def _fragment_is_parameterised(scope: ast.AST, name: str) -> bool:
    """Was this fragment built without interpolating anything into it?

    Every assignment and augmented assignment to `name` inside the function must
    be a plain string. One f-string means a value was spliced into SurrealQL —
    which is the whole defect, wherever the fragment is later used.
    """
    for node in ast.walk(scope):
        target = None
        if isinstance(node, ast.Assign):
            target = next(
                (t.id for t in node.targets if isinstance(t, ast.Name)), None
            )
            value = node.value
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
            value = node.value
        else:
            continue
        if target != name:
            continue
        if isinstance(value, ast.JoinedStr) and any(
            isinstance(p, ast.FormattedValue) for p in value.values
        ):
            return False
    return True


#: Functions that REFUSE a value that is not a record id. `validate_record_id`
#: lives in `record_ids` and raises; `_validate_record_id` is `api.py`'s wrapper
#: translating that into a 400.
_VALIDATORS = frozenset({"validate_record_id", "_validate_record_id"})


def _functions(tree: ast.Module) -> List[ast.AST]:
    return [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _validated_in(scope: ast.AST) -> Dict[str, int]:
    """Names validated INSIDE this function, mapped to the line it happened on.

    Per function and per line, both deliberately. A module-wide name set was the
    previous version and it could not fail: `entity_id` is validated in
    `get_entity_graph`, so deleting the validation from `get_entity` left the
    name "validated" and the guard green. Verified by deleting it. A validation
    that runs in a different function protects a different call.
    """
    found: Dict[str, int] = {}
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            # Both names REFUSE — `validate_record_id` raises `InvalidRecordId`
            # and `api.py`'s `_validate_record_id` translates that into a 400 at
            # the HTTP boundary. Recognising both is a widening, not a loosening;
            # a name that merely returns the value would not belong here.
            if getattr(node.value.func, "id", "") in _VALIDATORS:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        found.setdefault(target.id, node.lineno)
    return found


#: Fields of a request model whose own validator refuses a non-record-id, so a
#: handler cannot receive one. Derived from the model rather than assumed: the
#: test below asserts the `field_validator` is really there.
_MODEL_VALIDATED = {"req.canonical_id", "req.duplicate_id"}


def test_the_request_model_really_validates_its_ids() -> None:
    """`_MODEL_VALIDATED` is a claim about MergeRequest — check it."""
    tree = _module()
    model = next(
        (n for n in ast.walk(tree)
         if isinstance(n, ast.ClassDef) and n.name == "MergeRequest"),
        None,
    )
    assert model is not None, "MergeRequest is gone; the exemption is stale"
    decorated = [
        d
        for fn in model.body
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
        for d in fn.decorator_list
        if getattr(getattr(d, "func", d), "id", "") == "field_validator"
    ]
    assert decorated, (
        "MergeRequest has no `field_validator`, so `req.canonical_id` / "
        "`req.duplicate_id` are exempted from the guard for a reason that no "
        "longer holds"
    )
    fields = {
        a.value
        for d in decorated
        for a in getattr(d, "args", [])
        if isinstance(a, ast.Constant)
    }
    assert {"canonical_id", "duplicate_id"} <= fields, (
        f"the validator covers {fields}, not both id fields"
    )


def test_every_id_interpolated_into_a_query_was_validated() -> None:
    offenders: List[str] = []

    for path in SOURCES:
        rel = path.name
        try:
            tree = _module(path)
        except SyntaxError as exc:
            offenders.append(f"{rel}: unparseable ({exc})")
            continue

        for scope in _functions(tree):
            validated = _validated_in(scope)
            for node in ast.walk(scope):
                if not isinstance(node, ast.JoinedStr):
                    continue
                literal = "".join(
                    p.value for p in node.values
                    if isinstance(p, ast.Constant) and isinstance(p.value, str)
                )
                if not _QUERYISH.search(literal):
                    continue
                for name in _interpolated_names(node):
                    if name in _INT_TYPED:
                        continue
                    if name in validated:
                        continue
                    if name in _MODEL_VALIDATED:
                        continue
                    if name in _FRAGMENTS:
                        if _fragment_is_parameterised(scope, name):
                            continue
                        offenders.append(
                            f"{rel}:{node.lineno}: the query fragment `{name}` "
                            f"is built by interpolating a value rather than "
                            f"binding it"
                        )
                        continue
                    line = validated.get(name)
                    if line is None:
                        offenders.append(
                            f"{rel}:{node.lineno}: `{name}` reaches SurrealQL "
                            f"without a record-id validator in `{scope.name}`"
                        )
                    elif line > node.lineno:
                        offenders.append(
                            f"{rel}:{node.lineno}: `{name}` is validated at line "
                            f"{line}, AFTER it is interpolated"
                        )

    assert not offenders, (
        "these interpolate a value into SurrealQL without validating it first:\n"
        + "\n".join(offenders)
    )


def test_walker_control_reaches_every_service_module() -> None:
    """A guard scoped to one file is how the third injection family survived."""
    names = {p.name for p in SOURCES}
    assert {"api.py", "entity_validator.py"} <= names, (
        f"the walk covers {sorted(names)}; it must reach every module the "
        f"Dockerfile ships, not the one the last finding was in"
    )


def test_the_guard_is_scoped_to_the_function_that_validates() -> None:
    """A validation in one function must not vouch for another.

    This is the state the previous version passed in: `entity_id` validated in
    `get_entity_graph`, not in `get_entity`, and both interpolate it.
    """
    src = (
        "def a(entity_id):\n"
        '    return q(f"SELECT * FROM {entity_id};")\n'
        "def b(entity_id):\n"
        '    entity_id = _validate_record_id(entity_id)\n'
        '    return q(f"SELECT * FROM {entity_id};")\n'
    )
    tree = ast.parse(src)
    a, b = _functions(tree)
    assert _validated_in(a) == {}, "an unvalidated function looked validated"
    assert "entity_id" in _validated_in(b)


def test_validation_must_precede_the_interpolation() -> None:
    """Validating AFTER the query has already run protects nothing."""
    src = (
        "def a(entity_id):\n"
        '    r = q(f"SELECT * FROM {entity_id};")\n'
        "    entity_id = _validate_record_id(entity_id)\n"
        "    return r\n"
    )
    tree = ast.parse(src)
    fn = _functions(tree)[0]
    validated = _validated_in(fn)
    join = next(n for n in ast.walk(fn) if isinstance(n, ast.JoinedStr))
    assert validated["entity_id"] > join.lineno, (
        "the guard cannot tell a late validation from a timely one"
    )


def test_the_duplicated_pattern_matches_the_canonical_one() -> None:
    """This service duplicates `_RECORD_ID_RE`; the copy must not drift.

    It cannot import the original — the container copies four `shared` modules
    and no `surrealdb_service` — so the comment says "keep these in sync". A
    comment does not keep anything in sync. This does.
    """
    import ast as _ast

    canonical = (
        Path(__file__).resolve().parents[3]
        / "packages/surrealdb-service/src/surrealdb_service/repositories/base.py"
    )
    assert canonical.exists(), f"the canonical pattern moved: {canonical}"

    def _pattern(path: Path) -> str:
        m = re.search(r"_RECORD_ID_RE = re\.compile\((.+)\)", path.read_text(encoding="utf-8"))
        assert m, f"no _RECORD_ID_RE in {path}"
        return _ast.literal_eval(m.group(1))

    assert _pattern(RECORD_IDS) == _pattern(canonical), (
        "the duplicated record-id pattern has drifted from the canonical one in "
        "surrealdb_service/repositories/base.py"
    )


def test_the_int_typed_exemption_is_read_from_the_signature() -> None:
    """`_INT_TYPED` was a prose claim; this checks it.

    Review's round-4 minor 1: the guard trusted that `depth`, `limit` and
    `offset` are `int` in every signature rather than reading the annotation. If
    one became `str` the exemption would go silent — which is the `where` finding
    one notch smaller, and it has the same fix.
    """
    offenders: List[str] = []
    for path in SOURCES:
        tree = _module(path)
        for fn in _functions(tree):
            for arg in list(fn.args.args) + list(fn.args.kwonlyargs):
                if arg.arg not in _INT_TYPED:
                    continue
                ann = arg.annotation
                name = getattr(ann, "id", None) or getattr(
                    getattr(ann, "value", None), "id", None
                )
                if name != "int":
                    offenders.append(
                        f"{path.name}:{fn.lineno}: `{arg.arg}` is exempted as "
                        f"int-typed but is annotated {ast.dump(ann) if ann else 'not at all'}"
                    )
    assert not offenders, (
        "these are exempted from the injection guard on the grounds that FastAPI "
        "coerces them to int, and they are not int:\n" + "\n".join(offenders)
    )


def test_the_container_ships_every_module_the_guard_walks() -> None:
    """A module the guard covers but the image lacks fails at import, not here.

    `record_ids.py` was added so `entity_validator` could share the validator;
    forgetting its COPY line would break the container while every test stayed
    green.
    """
    dockerfile = (SERVICE / "Dockerfile").read_text(encoding="utf-8")
    missing = [
        p.name
        for p in SOURCES
        if f"services/extraction/{p.name}" not in dockerfile
    ]
    assert not missing, (
        f"the Dockerfile does not COPY {missing}; the service imports every "
        f"module in this directory"
    )
