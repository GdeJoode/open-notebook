"""Retype existing entities after the typing changed (PC.4b → OE.4).

A publish in the ontology editor changes how a label is typed, for NEW rows
only. Identity is ``(name_key, entity_type)``, so the next extraction that meets
an old row's name under its new type writes a SECOND row next to it. This moves
the old rows onto the type the ACTIVE ontologies give them, with the plan from
``retype_planner`` (conservative: collisions are refused and left for the
curator's merge on the resolution page, PC.5b ``/apply``).

Two readers:

* ``scripts/pc4b_retype_entities.py`` — the CLI (dry run, ``--apply``,
  ``--rollback FILE``), calling :func:`active_plan`, :func:`write_plan` and
  :func:`rollback_file` directly;
* the editor's API (``/api/ontologies/retype/…``) through
  :class:`EntityRetypeService`, which writes only the plan the owner confirmed
  (OE-D3 (a)): the dry run carries a :func:`fingerprint` of the plan, and the
  apply recomputes the plan and refuses (409) unless it still has that
  fingerprint. It runs under the editor's publish lock, so no publish or
  rollback can change the typing between that check and the writes.

Every row written is appended to a rollback file (JSONL, one row with its
previous type and flag per line); undoing it writes those rows back, guarded on
the type the retype wrote.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from surrealdb_service.connection import ensure_record_id, execute_query

from app_main.config import DATA_FOLDER
from app_main.services.entity_persistence_service import (
    load_typing_ontologies,
    typing_vocabulary,
)
from app_main.services.ontology_edit_service import OntologyEditError, _publish_lock
from app_main.services.retype_planner import plan_retype

ROLLBACK_DIR = Path(DATA_FOLDER) / "retype_rollback"
_ROLLBACK_NAME = re.compile(r"^pc4b_retype_\d{8}T\d{6}(\d{6})?Z\.jsonl$")

_WRITE = (
    "UPDATE $id SET entity_type = $to, properties.non_type_label = $flag "
    "WHERE entity_type = $from RETURN id;"
)


async def active_plan() -> Dict[str, Any]:
    """The plan under the typing extraction uses NOW (the registry's active set)."""
    return await plan_retype(typing_vocabulary(None, await load_typing_ontologies()))


def fingerprint(plan: Dict[str, Any]) -> str:
    """What the owner saw: every change, flag and collision, with its id, types
    and flags. ``total`` is left out, so a new unrelated entity does not
    invalidate a dry run."""
    body = {
        key: sorted(plan[key], key=lambda item: item["id"])
        for key in ("changes", "flag_only", "collisions")
    }
    text = json.dumps(body, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _write_rows(
    items: List[Dict[str, Any]], frm: str, to: str, flag_key: str,
    log_path: Optional[Path] = None,
) -> Dict[str, Any]:
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


def new_rollback_path(rollback_dir: Path = ROLLBACK_DIR) -> Path:
    rollback_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return rollback_dir / f"pc4b_retype_{stamp}.jsonl"


async def write_plan(plan: Dict[str, Any], log_path: Path) -> Dict[str, Any]:
    """Write the changes and flags of ``plan``, each guarded on the old type
    still being there; every row written is appended to ``log_path``."""
    outcome = await _write_rows(
        plan["changes"] + plan["flag_only"], "old", "new", "new_flag", log_path
    )
    outcome["rollback_file"] = str(log_path)
    return outcome


async def rollback_file(log_path: Path) -> Dict[str, Any]:
    """Undo exactly the rows in ``log_path``, guarded on the type the retype wrote."""
    items = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()
             if line]
    return await _write_rows(items, "new", "old", "old_flag")


class EntityRetypeService:
    """The retype as the ontology editor offers it: dry run, confirmed apply, undo."""

    def __init__(self, rollback_dir: Path = ROLLBACK_DIR):
        self.rollback_dir = rollback_dir

    async def dry_run(self) -> Dict[str, Any]:
        plan = await active_plan()
        return {
            **plan,
            "counts": {k: len(plan[k]) for k in ("changes", "collisions", "flag_only")},
            "fingerprint": fingerprint(plan),
        }

    async def apply(self, confirmed: str) -> Dict[str, Any]:
        """Write the plan whose fingerprint is ``confirmed``, or nothing."""
        async with _publish_lock():
            plan = await active_plan()
            if fingerprint(plan) != confirmed:
                raise OntologyEditError(409, (
                    "the retype plan changed since the dry run was shown; review the new "
                    "dry run"))
            outcome = await write_plan(plan, new_rollback_path(self.rollback_dir))
            path = Path(outcome.pop("rollback_file"))
            outcome["rollback_id"] = path.name if outcome["written"] else None
            return outcome

    async def undo(self, rollback_id: str) -> Dict[str, Any]:
        """Undo an apply by the rollback id it returned."""
        # A bare file name of the form the apply writes: nothing outside the
        # rollback directory can be named.
        if not _ROLLBACK_NAME.match(rollback_id):
            raise OntologyEditError(422, f"not a rollback id: {rollback_id!r}")
        path = self.rollback_dir / rollback_id
        if not path.is_file():
            raise OntologyEditError(404, f"no rollback file {rollback_id!r}")
        async with _publish_lock():
            return await rollback_file(path)


__all__ = [
    "EntityRetypeService", "ROLLBACK_DIR", "active_plan", "fingerprint",
    "new_rollback_path", "rollback_file", "write_plan",
]
