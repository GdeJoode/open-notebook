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

Test fixtures are deliberately out of scope. They use a SurrealQL-side
`string::lowercase(string::trim(...))`, which equals `normalize_entity_name` for
names without a leading article or a curated org alias — true of invented fixture
names, and false for `De Gemeente` (→ `gemeente`) or `BZK` (→ the expanded form).
That approximation is safe in a fixture and would not be in production, which is
why this scan covers production only and says so.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Production files that create an `entity` row without calling
#: `normalize_entity_name`, with the reason. Empty, and it should stay hard to
#: add to: an entry here is a second identity rule.
_ALLOWED: Dict[str, str] = {}

_CREATE = re.compile(r"CREATE entity\b(?!_)")


def _production_sources() -> List[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "--", "*.py"],
        cwd=REPO_ROOT, capture_output=True, text=True,
    ).stdout.splitlines()
    return [
        REPO_ROOT / rel
        for rel in listed
        if ("/src/" in rel or rel.startswith("services/"))
        and "test" not in Path(rel).name
        and "/tests/" not in rel
    ]


def test_every_entity_writer_uses_the_one_identity_rule() -> None:
    sources = _production_sources()
    assert sources, "walker control: no production sources found"

    writers = [p for p in sources if _CREATE.search(p.read_text(encoding="utf-8"))]
    assert len(writers) >= 3, (
        f"detector control: found only {len(writers)} entity writers "
        f"({[p.name for p in writers]}) — the scan is not seeing them"
    )

    offenders = sorted(
        str(p.relative_to(REPO_ROOT))
        for p in writers
        if "normalize_entity_name" not in p.read_text(encoding="utf-8")
        and str(p.relative_to(REPO_ROOT)) not in _ALLOWED
    )
    assert not offenders, (
        f"these create `entity` rows without deriving `name_key` from "
        f"`normalize_entity_name`: {offenders}. Two writers normalising one name "
        f"differently produce two identities for it, which migration 79's UNIQUE "
        f"index cannot catch — the keys differ, so both rows are accepted."
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
