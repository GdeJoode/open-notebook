"""Edit the shared ontologies: draft → validate → preview → publish (Track OE.2).

The ontologies are shared by every notebook and decide how every extracted
entity is typed (owner, 2026-09-28). This is the only writer of their stored
versions besides the YAML seed:

* A **draft** per ontology (``ontology_draft``, migration 87) holds its OWN
  content being edited and the active version it started from. Every edit
  changes the draft only.
* **Validate / preview** run the OE.1 gate and impact preview on the candidate:
  every active ontology, with this draft in place of its own.
* **Publish** refuses a blocking finding (422) or a draft whose base is no
  longer the active version (409, someone published meanwhile), then stores the
  draft as a new active ``ontology_version`` and removes it. Publishes are
  serialised in-process (the API and the job worker share one process), so two
  can never leave two active versions (OE.0 review minor 2). The first publish of
  an ontology also stores the YAML seed as an inactive version, so it can be
  rolled back to.
* **Rollback** re-activates a stored version, through the same gate.

Rename keeps the old name as an alias, so the entities, labels and references
that use it keep resolving. The publish result reports how many entities the
edit moves; the editor then offers their retype (``entity_retype_service``,
OE.4), which shares the publish lock.
"""

from __future__ import annotations

import asyncio
import weakref
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from ontology_manager.registry import get_ontology_registry
from ontology_manager.schema import (
    EntityTypeDefinition,
    Ontology,
    RelationshipTypeDefinition,
)
from ontology_manager.validation import BLOCK
from surrealdb_service.connection import execute_query

from app_main.services.ontology_preview_service import load_own_ontologies, preview
from app_main.services.ontology_validation_service import validate_candidate

#: Who published. There is no authentication in the app (PC.7), so every
#: publish is recorded as the editor, not as a person.
EDITOR = "ontology-editor"


def _stamp() -> str:
    # Microseconds: two publishes from one base within a second must not collide.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
_publish_locks: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]" = (
    weakref.WeakKeyDictionary()
)


def _publish_lock() -> asyncio.Lock:
    """One lock per event loop: an asyncio.Lock is bound to the loop it is first
    used on, so a module-level one breaks the moment a second loop exists."""
    loop = asyncio.get_running_loop()
    lock = _publish_locks.get(loop)
    if lock is None:
        lock = _publish_locks[loop] = asyncio.Lock()
    return lock


class OntologyEditError(Exception):
    """An edit the service refuses; ``status`` is the HTTP status that fits."""

    def __init__(self, status: int, detail: Any):
        super().__init__(str(detail))
        self.status = status
        self.detail = detail


def _findings(findings) -> List[Dict[str, Any]]:
    return [f.__dict__ for f in findings]


class OntologyEditService:
    # -- the draft ---------------------------------------------------------------

    async def _active(self) -> Dict[str, Ontology]:
        return await load_own_ontologies()

    async def _draft_row(self, name: str) -> Optional[Dict[str, Any]]:
        rows = await execute_query(
            "SELECT * FROM ontology_draft WHERE name = $name LIMIT 1;", {"name": name}
        )
        return rows[0] if rows else None

    async def get_draft(self, name: str) -> Dict[str, Any]:
        """The draft, or — when there is none — the active content to start one."""
        row = await self._draft_row(name)
        if row:
            ontology = Ontology.from_yaml_string(row["yaml_content"])
            return {"name": name, "exists": True, "base_version": row["base_version"],
                    "ontology": ontology}
        active = (await self._active()).get(name)
        if active is None:
            raise OntologyEditError(404, f"Ontology '{name}' not found")
        return {"name": name, "exists": False, "base_version": active.metadata.version,
                "ontology": active}

    async def _save(self, name: str, ontology: Ontology, base_version: str) -> None:
        # SET, not CONTENT: CONTENT replaces the whole record, and on an
        # existing draft that left `created_at` NONE, which the schema refuses —
        # every second edit of a draft failed (review OE.2 B1).
        await execute_query(
            "UPSERT type::thing('ontology_draft', $name) SET"
            " name = $name, yaml_content = $yaml, base_version = $base,"
            " updated_at = time::now();",
            {"name": name, "yaml": ontology.to_yaml_string(), "base": base_version},
        )

    async def _edit(self, name: str, change) -> Dict[str, Any]:
        # Under the publish lock: an edit landing between a publish reading the
        # draft and deleting it was accepted, then silently lost (review M3).
        # Each edit is a read-modify-write of the CURRENT draft, so edits from
        # two tabs both apply.
        async with _publish_lock():
            draft = await self.get_draft(name)
            ontology = draft["ontology"]
            change(ontology)
            await self._save(name, ontology, draft["base_version"])
            return await self.get_draft(name)

    async def discard_draft(self, name: str) -> None:
        async with _publish_lock():
            await self._delete_draft(name)

    async def _delete_draft(self, name: str) -> None:
        await execute_query("DELETE ontology_draft WHERE name = $name;", {"name": name})

    # -- edits ---------------------------------------------------------------------

    async def put_entity_type(self, name: str, type_name: str, fields: Dict[str, Any]):
        try:
            definition = EntityTypeDefinition(**{**fields, "name": type_name})
        except Exception as exc:
            raise OntologyEditError(422, str(exc)) from exc
        return await self._edit(name, lambda o: o.entity_types.__setitem__(type_name, definition))

    async def delete_entity_type(self, name: str, type_name: str):
        def change(o: Ontology):
            if type_name not in o.entity_types:
                raise OntologyEditError(404, f"'{name}' defines no entity type '{type_name}'")
            del o.entity_types[type_name]
        return await self._edit(name, change)

    async def rename_entity_type(self, name: str, old: str, new: str):
        """Rename, keeping ``old`` as an alias so what uses it keeps resolving."""
        def change(o: Ontology):
            if old not in o.entity_types:
                raise OntologyEditError(404, f"'{name}' defines no entity type '{old}'")
            if new in o.entity_types:
                raise OntologyEditError(409, f"'{name}' already defines '{new}'")
            definition = o.entity_types.pop(old)
            definition.name = new
            aliases = [a for a in (definition.aliases or []) if a != new]
            definition.aliases = aliases + ([old] if old not in aliases else [])
            o.entity_types[new] = definition
            # This ontology's own references follow the new name; other
            # ontologies' references keep resolving through the alias.
            for other in o.entity_types.values():
                if other.parent_type == old:
                    other.parent_type = new
            for rel in o.relationship_types.values():
                for field in ("domain", "range", "source_types", "target_types"):
                    names = getattr(rel, field, None)
                    if names and old in names:
                        setattr(rel, field, [new if n == old else n for n in names])
        return await self._edit(name, change)

    async def rename_relationship_type(self, name: str, old: str, new: str):
        def change(o: Ontology):
            if old not in o.relationship_types:
                raise OntologyEditError(404, f"'{name}' defines no relationship type '{old}'")
            if new in o.relationship_types:
                raise OntologyEditError(409, f"'{name}' already defines '{new}'")
            definition = o.relationship_types.pop(old)
            definition.name = new
            o.relationship_types[new] = definition
        return await self._edit(name, change)

    async def put_relationship_type(self, name: str, rel_name: str, fields: Dict[str, Any]):
        try:
            definition = RelationshipTypeDefinition(**{**fields, "name": rel_name})
        except Exception as exc:
            raise OntologyEditError(422, str(exc)) from exc
        return await self._edit(
            name, lambda o: o.relationship_types.__setitem__(rel_name, definition))

    async def delete_relationship_type(self, name: str, rel_name: str):
        def change(o: Ontology):
            if rel_name not in o.relationship_types:
                raise OntologyEditError(404, f"'{name}' defines no relationship type '{rel_name}'")
            del o.relationship_types[rel_name]
        return await self._edit(name, change)

    # -- validate / preview ----------------------------------------------------------

    async def _candidate(self, name: str):
        draft = await self.get_draft(name)
        active = await self._active()
        candidate = {**active, name: draft["ontology"]}
        return draft, active, candidate

    async def validate_draft(self, name: str) -> List[Dict[str, Any]]:
        _draft, active, candidate = await self._candidate(name)
        return _findings(validate_candidate(candidate, active))

    async def preview_draft(self, name: str, include_detection: bool = True) -> Dict[str, Any]:
        _draft, active, candidate = await self._candidate(name)
        return await preview(candidate, active, include_detection=include_detection)

    # -- publish / versions / rollback --------------------------------------------------

    async def publish(self, name: str, note: Optional[str] = None) -> Dict[str, Any]:
        async with _publish_lock():
            row = await self._draft_row(name)
            if not row:
                raise OntologyEditError(404, f"'{name}' has no draft to publish")
            draft = Ontology.from_yaml_string(row["yaml_content"])
            active = await self._active()
            current = active.get(name)
            if current is None or current.metadata.version != row["base_version"]:
                raise OntologyEditError(409, (
                    f"'{name}' is at version "
                    f"{current.metadata.version if current else None!r} now, the draft was "
                    f"started from {row['base_version']!r}: someone published meanwhile"))
            candidate = {**active, name: draft}
            findings = validate_candidate(candidate, active)
            if any(f.severity == BLOCK for f in findings):
                raise OntologyEditError(422, {"message": "the draft has blocking findings",
                                              "findings": _findings(findings)})
            impact = await preview(candidate, active, include_detection=False)

            registry = get_ontology_registry()
            stored = {v["version"] for v in await registry.list_versions(name)}
            if await registry._load_from_db(name) is None:
                # The active content is the YAML seed (no active stored version,
                # even if inactive ones exist): store it, so it can be rolled
                # back to (review minor 6).
                seed = current.model_copy(deep=True)
                if seed.metadata.version in stored:
                    seed.metadata.version = f"{seed.metadata.version}+yaml-{_stamp()}"
                if not await registry.register(seed, activate=True, note="YAML seed",
                                               created_by=EDITOR):
                    raise OntologyEditError(500, f"could not store the seed of '{name}'")
                stored.add(seed.metadata.version)
            draft.metadata.version = f"{row['base_version'].split('+')[0]}+{_stamp()}"
            if draft.metadata.version in stored:
                raise OntologyEditError(409, f"version {draft.metadata.version!r} exists already")
            if not await registry.register(draft, activate=True, note=note, created_by=EDITOR):
                raise OntologyEditError(500, f"could not store version {draft.metadata.version!r}")
            warnings = _findings(findings)
            try:
                await self._delete_draft(name)
            except Exception as exc:  # published: report, do not fail the publish
                warnings.append({"rule": "draft_not_removed", "ontology": name, "subject": None,
                                 "detail": f"published, but the draft could not be removed: {exc}",
                                 "severity": "warn", "weight": 1})
            return {
                "name": name,
                "version": draft.metadata.version,
                "warnings": warnings,
                "retype": {k: impact["retype"][k] for k in
                           ("changes", "collisions", "flag_only", "caused_by_this_edit",
                            "collisions_caused_by_this_edit")},
            }

    async def versions(self, name: str) -> List[Dict[str, Any]]:
        return await get_ontology_registry().list_versions(name)

    async def rollback(self, name: str, version: str) -> Dict[str, Any]:
        async with _publish_lock():
            registry = get_ontology_registry()
            if version not in {v["version"] for v in await registry.list_versions(name)}:
                raise OntologyEditError(404, f"'{name}' has no stored version {version!r}")
            rows = await execute_query(
                "SELECT yaml_content FROM ontology_version WHERE name = $n AND version = $v;",
                {"n": name, "v": version},
            )
            restored = Ontology.from_yaml_string(rows[0]["yaml_content"])
            active = await self._active()
            findings = validate_candidate({**active, name: restored}, active)
            if any(f.severity == BLOCK for f in findings):
                raise OntologyEditError(422, {"message": "rolling back would introduce "
                                              "blocking findings", "findings": _findings(findings)})
            if not await registry.activate_version(name, version):
                raise OntologyEditError(500, f"could not activate {version!r}")
            return {"name": name, "version": version, "warnings": _findings(findings)}

    async def export_yaml(self, name: str) -> str:
        active = (await self._active()).get(name)
        if active is None:
            raise OntologyEditError(404, f"Ontology '{name}' not found")
        return active.to_yaml_string()


__all__ = ["OntologyEditError", "OntologyEditService"]
