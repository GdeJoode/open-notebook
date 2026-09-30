"""Track WB.3 — drop the 12-hex upload prefix from stored source titles.

The upload pipeline staged files as ``<12 hex>__<name>`` and the title was
taken from that name (``410a65bba3f8__Convenant Regio Deal Waterwegregio``).
The frontend already shows titles without it (``displaySourceTitle``); this
rewrites the stored ``title`` so every other reader (exports, MCP, Zotero push)
gets the clean one too. The owner decides whether to run it; it is not part of
a deploy.

For every source whose title starts with the prefix it plans the title without
it (``app_main.services.source_title.display_source_title``, the server's twin
of the frontend helper). A title that would be empty is left alone. Titles
need not be unique; the plan reports which clean titles several sources share
(two uploads of one file), but writes them all.

Modes:

* default — DRY RUN: prints the plan, writes nothing;
* ``--apply`` — writes. Each update is guarded on the old title still being
  there; a row that fails or changed meanwhile is skipped and reported, never
  aborts the run. Every row actually written is appended to a rollback file in
  ``data/title_backfill_rollback/`` (gitignored) with its previous title;
* ``--rollback FILE`` — undoes exactly the rows in FILE (guarded on the title
  the backfill wrote).

Note: ``source.updated`` is set on every write, so after ``--apply`` the
written sources read as just changed: for 15 minutes none of them counts as
stopped (``services/stopped_sources.py``), and lists by date move them up.

Run with SURREAL_* pointing at the target (the WSL venv, env from ``.env``)::

    python scripts/wb3_backfill_source_titles.py                   # dry run
    python scripts/wb3_backfill_source_titles.py --apply           # write
    python scripts/wb3_backfill_source_titles.py --rollback FILE   # undo
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from surrealdb_service.connection import ensure_record_id, execute_query

from app_main.services.source_title import display_source_title

REPO_ROOT = Path(__file__).resolve().parents[1]
ROLLBACK_DIR = REPO_ROOT / "data" / "title_backfill_rollback"

_WRITE = "UPDATE $id SET title = $to WHERE title = $from RETURN AFTER;"


async def plan() -> Dict[str, Any]:
    """``changes`` (``{id, old, new}``) for every prefixed title, and the clean
    titles several sources would share."""
    rows = await execute_query("SELECT id, title FROM source;", {}) or []
    changes: List[Dict[str, Any]] = []
    for r in rows:
        old = r.get("title")
        if not isinstance(old, str):
            continue
        new = display_source_title(old)
        if new != old and new.strip():
            changes.append({"id": str(r["id"]), "old": old, "new": new})
    new_by_id = {c["id"]: c["new"] for c in changes}
    after = Counter(new_by_id.get(str(r["id"]), r.get("title")) for r in rows)
    shared = sorted(t for t, n in after.items() if n > 1 and t)
    return {"total": len(rows), "changes": changes, "shared": shared}


async def _write_rows(
    items: List[Dict[str, Any]], frm: str, to: str, log_path: Optional[Path] = None
) -> Dict[str, Any]:
    written, skipped, failed = 0, [], []
    for c in items:
        try:
            rows = await execute_query(
                _WRITE, {"id": ensure_record_id(c["id"]), "from": c[frm], "to": c[to]}
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


def new_rollback_path(rollback_dir: Path = ROLLBACK_DIR) -> Path:
    rollback_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return rollback_dir / f"wb3_titles_{stamp}.jsonl"


async def apply(result: Dict[str, Any], rollback_dir: Path = ROLLBACK_DIR) -> Dict[str, Any]:
    log_path = new_rollback_path(rollback_dir)
    print(f"rollback file (one line per row written): {log_path}")
    outcome = await _write_rows(result["changes"], "old", "new", log_path)
    outcome["rollback_file"] = str(log_path)
    return outcome


async def rollback(log_path: Path) -> Dict[str, Any]:
    items = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
             if line]
    return await _write_rows(items, "new", "old")


def _print(result: Dict[str, Any]) -> None:
    print(f"sources read: {result['total']}")
    print(f"titles to clean: {len(result['changes'])}")
    for c in result["changes"]:
        print(f"  {c['id']}: {c['old']!r} -> {c['new']!r}")
    print(f"clean titles several sources share: {len(result['shared'])}")
    for t in result["shared"]:
        print(f"  {t!r}")


def _report(outcome: Dict[str, Any]) -> None:
    print(f"written: {outcome['written']}; skipped (changed meanwhile): "
          f"{len(outcome['skipped'])}; failed: {len(outcome['failed'])}")
    for f in outcome["failed"]:
        print(f"  {f['id']}: {f['error']}")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--apply", action="store_true", help="write (default: dry run)")
    group.add_argument("--rollback", type=Path, help="undo the rows in this rollback file")
    args = parser.parse_args()

    if args.rollback:
        _report(await rollback(args.rollback))
        return 0
    result = await plan()
    _print(result)
    if args.apply:
        _report(await apply(result))
    else:
        print("DRY RUN: nothing written. Re-run with --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
