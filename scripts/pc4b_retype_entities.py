"""Track PC.4b — retype existing entities with PC.4's typing.

PC.4 changed how a label is typed: against every ontology instead of the
document's selection, with 15 ontology types corrected, new NL aliases, literals
on ``other`` and ``Project`` as a programme. New extractions type the new way;
rows written before keep the old type. Identity is ``(name_key, entity_type)``,
so the next extraction that meets such a name writes a SECOND row next to the
old one. This retypes the old rows so that does not happen.

For every active/reference entity with a label it recomputes the type exactly
as persistence does — ``_resolve_entity_type`` over ``typing_vocabulary`` of the
RAW LLM label (``properties.raw_entity_type``; else ``primary_type``, which
persistence sets to the raw label whenever the raw label was not stored) — and:

* **changes** the type where it differs;
* flags a literal (``Date``, ``Budget``, …) with ``properties.non_type_label``;
* **refuses** a change onto a ``(name_key, entity_type)`` another active or
  reference row already holds, or another change claims. Two such rows are
  one entity: a merge for the curator, not a retype. Active pairs would also
  violate ``idx_entity_identity``; a reference pair would not, but
  ``upsert_entity`` looks up on ``(name_key, entity_type)`` without a status
  filter, so new mentions would land on either row.

Modes:

* default — DRY RUN: prints the plan, writes nothing;
* ``--apply`` — writes. Each update is guarded on the old type still being
  there; a row that fails or changed meanwhile is skipped and reported, never
  aborts the run. Every row actually written is appended to a rollback file in
  ``data/retype_rollback/`` (gitignored) with its previous type and flag;
* ``--rollback FILE`` — undoes exactly the rows in FILE (guarded on the type
  the retype wrote), restoring the previous type and flag.

Notebook schema edits are not projected (typing here has no notebook, like the
re-filter). If any notebook has accepted edits the script refuses unless
``--ignore-notebook-edits`` is given.

Run with SURREAL_* pointing at the target (the WSL venv, env from ``.env``)::

    python scripts/pc4b_retype_entities.py                   # dry run
    python scripts/pc4b_retype_entities.py --apply           # write
    python scripts/pc4b_retype_entities.py --rollback FILE   # undo
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from app_main.services.entity_persistence_service import (
    load_typing_ontologies,
    typing_vocabulary,
)
from app_main.services.retype_planner import plan_retype
from surrealdb_service.connection import ensure_record_id, execute_query

REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLBACK_DIR = REPO_ROOT / "data" / "retype_rollback"

_WRITE = (
    "UPDATE $id SET entity_type = $to, properties.non_type_label = $flag "
    "WHERE entity_type = $from RETURN id;"
)


async def _notebooks_with_edits() -> list[str]:
    rows = await execute_query(
        "SELECT notebook, accepted_extensions FROM notebook_schema;", {}
    )
    return [
        str(r["notebook"])
        for r in rows or []
        if any(
            isinstance(e, dict) and not str(e.get("type_name", "")).startswith("_")
            for e in (r.get("accepted_extensions") or [])
        )
    ]


async def plan() -> dict:
    """The plan under the current typing (``app_main.services.retype_planner``,
    shared with the ontology editor's preview)."""
    return await plan_retype(typing_vocabulary(None, await load_typing_ontologies()))


def _print(result: dict) -> None:
    changes, collisions = result["changes"], result["collisions"]
    print(f"entities read: {result['total']}")
    print(f"type changes: {len(changes)}")
    for (old, new), n in Counter((c["old"], c["new"]) for c in changes).most_common():
        print(f"  {old} -> {new}: {n}")
    print("by label:")
    for label, n in Counter(c["label"] for c in changes).most_common(30):
        print(f"  {label}: {n}")
    print(f"literals flagged without a type change: {len(result['flag_only'])}")
    print(f"refused (the new identity is already held; a curator merge): {len(collisions)}")
    for c in collisions:
        print(f"  {c['id']} {c['name']!r} [{c['status']}] {c['old']} -> {c['new']}, held by {c['held_by']}")


async def _write_rows(items, frm, to, flag_key, log_path=None) -> dict:
    written, skipped, failed = 0, [], []
    for c in items:
        try:
            rows = await execute_query(
                _WRITE,
                {"id": ensure_record_id(c["id"]), "from": c[frm], "to": c[to],
                 "flag": c[flag_key]},
            )
        except Exception as exc:  # one bad row must not abort the run
            failed.append({"id": c["id"], "error": str(exc)})
            continue
        if not rows:
            skipped.append(c["id"])  # changed since the plan was made
            continue
        written += 1
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    return {"written": written, "skipped": skipped, "failed": failed}


async def apply(result: dict, rollback_dir: Path = ROLLBACK_DIR) -> dict:
    rollback_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = rollback_dir / f"pc4b_retype_{stamp}.jsonl"
    print(f"rollback file (one line per row written): {log_path}")
    outcome = await _write_rows(
        result["changes"] + result["flag_only"], "old", "new", "new_flag", log_path
    )
    outcome["rollback_file"] = str(log_path)
    return outcome


async def rollback(log_path: Path) -> dict:
    items = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    return await _write_rows(items, "new", "old", "old_flag")


def _report(outcome: dict) -> None:
    print(f"written: {outcome['written']}; skipped (changed meanwhile): "
          f"{len(outcome['skipped'])}; failed: {len(outcome['failed'])}")
    for f in outcome["failed"]:
        print(f"  {f['id']}: {f['error']}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="write (default: dry run)")
    group.add_argument("--rollback", type=Path, help="undo the rows in this rollback file")
    parser.add_argument("--ignore-notebook-edits", action="store_true")
    args = parser.parse_args()

    if args.rollback:
        _report(await rollback(args.rollback))
        return 0
    edited = await _notebooks_with_edits()
    if edited and not args.ignore_notebook_edits:
        print(f"refusing: notebooks with accepted schema edits: {edited}")
        return 2
    result = await plan()
    _print(result)
    if args.apply:
        _report(await apply(result))
    else:
        print("DRY RUN: nothing written. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
