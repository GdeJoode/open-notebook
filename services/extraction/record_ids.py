"""One record-id validator for the extraction service.

SurrealDB v2 cannot bind a record id as a `$param` in the positions this service
needs — `FROM {id}`, `UPDATE {id}`, `WHERE in = {id}`, `RELATE {src}->…->{tgt}` —
so the id is interpolated, and an interpolated value nobody validates is
SurrealQL injection. This repository has already lost a table to exactly that
(Track Y.1). `RecordID.parse` / `startswith` are not sufficient: they split on the
first colon and accept whatever follows.

WHY IT LIVES HERE rather than in `api.py`. `api.py` imports `entity_validator`,
so `entity_validator` cannot import `api` — and both interpolate ids. Review
found four unvalidated sites in `entity_validator.py` after the same class had
been closed in `api.py`, because the fix was scoped to the file the finding was
in. A module both can import removes the choice.

The pattern is still duplicated once, deliberately, from
`surrealdb_service/repositories/base.py`: this service runs in a container that
copies three modules and no `surrealdb_service`. That copy is byte-identical and
`test_the_duplicated_pattern_matches_the_canonical_one` fails if the two drift.
"""

from __future__ import annotations

import re

_RECORD_ID_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*:[a-zA-Z0-9_\-⟨⟩]+$")


class InvalidRecordId(ValueError):
    """A value that would have been interpolated into SurrealQL is not an id."""


def validate_record_id(value: str, *, field: str = "record id") -> str:
    """Return `value` if it is a record id, else refuse.

    Refusing is the point: every caller interpolates the result into SurrealQL,
    so a value that reaches a query without passing here is a hole. Callers that
    are FastAPI handlers translate this into a 400; callers that are not let it
    propagate, because continuing past it means writing to an id nobody checked.
    """
    if not isinstance(value, str) or not _RECORD_ID_RE.match(value):
        raise InvalidRecordId(
            f"Invalid {field}: expected `table:id`, got {value!r}"
        )
    return value
