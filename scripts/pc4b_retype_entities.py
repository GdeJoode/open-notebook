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
from collections import Counter
from pathlib import Path

from app_main.services.entity_retype_service import (
    active_plan,
    new_rollback_path,
    notebooks_with_edits,
    rollback_file,
    write_plan,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLBACK_DIR = REPO_ROOT / "data" / "retype_rollback"


async def plan() -> dict:
    """The plan under the current typing (``app_main.services.entity_retype_service``,
    shared with the ontology editor)."""
    return await active_plan()


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


async def apply(result: dict, rollback_dir: Path = ROLLBACK_DIR) -> dict:
    log_path = new_rollback_path(rollback_dir)
    print(f"rollback file (one line per row written): {log_path}")
    return await write_plan(result, log_path)


async def rollback(log_path: Path) -> dict:
    return await rollback_file(log_path)


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
    edited = await notebooks_with_edits()
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
