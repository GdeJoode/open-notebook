"""
Bidirectional sync between an external vault (e.g. Obsidian) and the knowledge graph.

Reads entity notes (one .md per entity with YAML frontmatter aliases) from a
vault folder, presents them for review, and registers accepted aliases in the
entity_alias table. Also queues newly discovered entities for export as vault
note stubs.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger
from shared.utils.name_normalizer import normalize_entity_name
from surrealdb_service.connection import execute_query


def _parse_frontmatter(content: str) -> Dict[str, Any]:
    """Parse YAML frontmatter from a markdown file's content.

    Expects the standard --- delimited block at the top of the file.
    Returns an empty dict if no valid frontmatter is found.
    """
    match = re.match(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if not match:
        return {}

    try:
        import yaml

        return yaml.safe_load(match.group(1)) or {}
    except Exception as e:
        logger.warning(f"Failed to parse YAML frontmatter: {e}")
        return {}


class VaultSyncService:
    """Bidirectional sync between an external vault and the knowledge graph."""

    # ------------------------------------------------------------------
    # Inbound: Vault → App
    # ------------------------------------------------------------------

    async def scan_vault_entities(
        self, vault_path: str, entities_folder: str = "Entities"
    ) -> Dict[str, Any]:
        """Read entity notes from vault, diff against DB, create pending imports.

        Each .md file in the entities folder is treated as an entity note.
        The filename (without .md) is the canonical name. YAML frontmatter
        provides aliases and type.

        Returns:
            Stats dict with new, changed, unchanged, total_files counts.
        """
        folder = Path(vault_path) / entities_folder
        if not folder.exists():
            logger.warning(f"Vault entities folder not found: {folder}")
            return {"error": f"Folder not found: {folder}", "total_files": 0}

        md_files = sorted(folder.glob("*.md"))
        if not md_files:
            return {"new": 0, "changed": 0, "unchanged": 0, "total_files": 0}

        stats = {"new": 0, "changed": 0, "unchanged": 0, "total_files": len(md_files)}

        for md_file in md_files:
            canonical_name = md_file.stem
            content = md_file.read_text(encoding="utf-8")
            frontmatter = _parse_frontmatter(content)

            aliases = frontmatter.get("aliases", [])
            if isinstance(aliases, str):
                aliases = [aliases]
            entity_type = frontmatter.get("type", "UNKNOWN").upper()
            source_file = str(md_file.relative_to(vault_path))

            # Check if already imported
            existing = await execute_query(
                "SELECT * FROM vault_import WHERE canonical_name = $name LIMIT 1",
                {"name": canonical_name},
            )

            if existing:
                row = existing[0]
                old_aliases = row.get("aliases", [])
                old_type = row.get("entity_type", "")

                if set(old_aliases) == set(aliases) and old_type == entity_type:
                    stats["unchanged"] += 1
                    continue

                # Changed — update with pending status
                await execute_query(
                    "UPDATE vault_import SET "
                    "aliases = $aliases, entity_type = $entity_type, "
                    "source_file = $source_file, status = 'pending', "
                    "imported_at = time::now() "
                    "WHERE canonical_name = $name",
                    {
                        "name": canonical_name,
                        "aliases": aliases,
                        "entity_type": entity_type,
                        "source_file": source_file,
                    },
                )
                stats["changed"] += 1
            else:
                # New entry
                await execute_query(
                    "CREATE vault_import SET "
                    "canonical_name = $name, aliases = $aliases, "
                    "entity_type = $entity_type, source_file = $source_file, "
                    "status = 'pending', imported_at = time::now()",
                    {
                        "name": canonical_name,
                        "aliases": aliases,
                        "entity_type": entity_type,
                        "source_file": source_file,
                    },
                )
                stats["new"] += 1

        logger.info(f"Vault scan complete: {stats}")
        return stats

    async def get_pending_imports(self) -> List[Dict[str, Any]]:
        """Fetch vault_import records with status=pending."""
        return await execute_query(
            "SELECT * FROM vault_import WHERE status = 'pending' "
            "ORDER BY canonical_name ASC"
        )

    async def apply_imports(
        self,
        accepted_ids: List[str],
        rejected_ids: List[str],
    ) -> Dict[str, Any]:
        """Accept or reject pending vault imports.

        Accepted imports register each alias in the entity_alias table.
        Rejected imports are marked as rejected.
        """
        accepted_count = 0
        rejected_count = 0

        for import_id in accepted_ids:
            rows = await execute_query(
                "SELECT * FROM $id", {"id": import_id}
            )
            if not rows:
                continue

            row = rows[0]
            canonical_name = row.get("canonical_name", "")
            aliases = row.get("aliases", [])
            entity_type = row.get("entity_type", "UNKNOWN")

            # Ensure a canonical entity exists
            # BY IDENTITY, not by display name (PC.3) — the same split that
            # breaks the extraction service: this looked up on `name` while the
            # CREATE below writes `name_key`, and migration 79 made those two
            # different keys.
            vault_name_key = normalize_entity_name(canonical_name)
            entity_rows = await execute_query(
                "SELECT id FROM entity WHERE name_key = $name_key LIMIT 1",
                {"name_key": vault_name_key},
            )

            if entity_rows:
                entity_id = str(entity_rows[0]["id"])
            else:
                # Create the entity
                created = await execute_query(
                    # PC.3: `name_key` is required and is the identity column.
                    # `canonical_name` is set alongside `name` because migration
                    # 39's UNIQUE index and every canonical reader use it, and a
                    # vault-created row that lacked it was invisible to both.
                    "CREATE entity SET name = $name, canonical_name = $name, "
                    "name_key = $name_key, entity_type = $entity_type, "
                    "weight = 1, properties = {}, source_ids = [], "
                    "created = time::now(), updated = time::now()",
                    {
                        "name": canonical_name,
                        "name_key": vault_name_key,
                        "entity_type": entity_type,
                    },
                )
                entity_id = str(created[0]["id"]) if created else None

            if entity_id:
                # Register each alias
                for alias in aliases:
                    try:
                        await execute_query(
                            "CREATE entity_alias SET "
                            "canonical_entity = $entity_id, "
                            "alias_text = $alias, "
                            "match_type = 'exact', "  # closed vocabulary; see `method` below
                            "similarity_score = 1.0, "
                            "method = 'obsidian_vault', "
                            "verified = true",
                            {"entity_id": entity_id, "alias": alias},
                        )
                    except Exception as e:
                        # Alias might already exist
                        logger.debug(f"Alias '{alias}' may already exist: {e}")

                # Also register canonical name as self-alias
                try:
                    await execute_query(
                        "CREATE entity_alias SET "
                        "canonical_entity = $entity_id, "
                        "alias_text = $alias, "
                        "match_type = 'exact', "  # closed vocabulary; see `method` below
                        "similarity_score = 1.0, "
                        "method = 'obsidian_vault', "
                        "verified = true",
                        {"entity_id": entity_id, "alias": canonical_name},
                    )
                except Exception:
                    pass

            # Mark as accepted
            await execute_query(
                "UPDATE $id SET status = 'accepted'", {"id": import_id}
            )
            accepted_count += 1

        for import_id in rejected_ids:
            await execute_query(
                "UPDATE $id SET status = 'rejected'", {"id": import_id}
            )
            rejected_count += 1

        logger.info(
            f"Vault imports applied: {accepted_count} accepted, "
            f"{rejected_count} rejected"
        )
        return {"accepted": accepted_count, "rejected": rejected_count}

    # ------------------------------------------------------------------
    # Outbound: App → Vault
    # ------------------------------------------------------------------

    async def queue_export(self, entities: List[Dict[str, Any]]) -> int:
        """Queue newly discovered entities for export to vault.

        Only queues entities not already known from vault imports or
        previously queued for export.
        """
        queued = 0
        for entity in entities:
            name = entity.get("text", "") or entity.get("name", "")
            if not name or len(name) < 2:
                continue

            entity_type = entity.get("label", entity.get("entity_type", "UNKNOWN"))
            aliases = []

            # Check if already known from vault
            existing_import = await execute_query(
                "SELECT id FROM vault_import WHERE canonical_name = $name LIMIT 1",
                {"name": name},
            )
            if existing_import:
                continue

            # Check if already queued for export
            existing_export = await execute_query(
                "SELECT id FROM vault_export WHERE canonical_name = $name LIMIT 1",
                {"name": name},
            )
            if existing_export:
                continue

            # Queue for export
            source_ids = entity.get("source_ids", [])
            if not source_ids:
                chunk_id = entity.get("source_chunk_id")
                if chunk_id:
                    source_ids = [chunk_id]

            await execute_query(
                "CREATE vault_export SET "
                "canonical_name = $name, aliases = $aliases, "
                "entity_type = $entity_type, source_ids = $source_ids, "
                "status = 'pending', created_at = time::now()",
                {
                    "name": name,
                    "aliases": aliases,
                    "entity_type": entity_type,
                    "source_ids": source_ids,
                },
            )
            queued += 1

        if queued:
            logger.info(f"Queued {queued} entities for vault export")
        return queued

    async def get_pending_exports(self) -> List[Dict[str, Any]]:
        """Fetch vault_export records with status=pending."""
        return await execute_query(
            "SELECT * FROM vault_export WHERE status = 'pending' "
            "ORDER BY canonical_name ASC"
        )

    async def write_exports(
        self,
        accepted_ids: List[str],
        rejected_ids: List[str],
        vault_path: str,
        entities_folder: str = "Entities",
    ) -> Dict[str, Any]:
        """Write accepted entity stubs as .md files to the vault.

        Creates one .md file per entity with YAML frontmatter containing
        aliases and type. Rejected exports are marked as rejected.
        """
        folder = Path(vault_path) / entities_folder
        folder.mkdir(parents=True, exist_ok=True)

        written = 0
        rejected_count = 0

        for export_id in accepted_ids:
            rows = await execute_query(
                "SELECT * FROM $id", {"id": export_id}
            )
            if not rows:
                continue

            row = rows[0]
            name = row.get("canonical_name", "")
            aliases = row.get("aliases", [])
            entity_type = row.get("entity_type", "UNKNOWN")
            source_ids = row.get("source_ids", [])

            # Sanitize filename
            safe_name = re.sub(r'[<>:"/\\|?*]', "_", name)
            file_path = folder / f"{safe_name}.md"

            # Don't overwrite existing vault notes
            if file_path.exists():
                logger.info(f"Vault note already exists, skipping: {file_path}")
                await execute_query(
                    "UPDATE $id SET status = 'skipped'", {"id": export_id}
                )
                continue

            # Write the stub
            content_lines = [
                "---",
                f"aliases:",
            ]
            for alias in aliases:
                content_lines.append(f"  - {alias}")
            content_lines.extend([
                f"type: {entity_type.lower()}",
                "---",
                "",
                f"# {name}",
                "",
                f"*Generated by Open Notebook*",
                "",
            ])
            if source_ids:
                content_lines.append(f"Sources: {', '.join(str(s) for s in source_ids[:5])}")
                content_lines.append("")

            file_path.write_text("\n".join(content_lines), encoding="utf-8")

            await execute_query(
                "UPDATE $id SET status = 'written'", {"id": export_id}
            )
            written += 1

        for export_id in rejected_ids:
            await execute_query(
                "UPDATE $id SET status = 'rejected'", {"id": export_id}
            )
            rejected_count += 1

        logger.info(
            f"Vault export: {written} written, {rejected_count} rejected"
        )
        return {"written": written, "rejected": rejected_count}

    # ------------------------------------------------------------------
    # Dictionary
    # ------------------------------------------------------------------

    async def get_merged_dictionary(self) -> Dict[str, str]:
        """Build abbreviation dict from accepted vault imports + entity aliases.

        Returns {alias_text: canonical_name} for use by LLMMatcher.
        """
        result: Dict[str, str] = {}

        # Load from accepted vault_imports
        imports = await execute_query(
            "SELECT canonical_name, aliases FROM vault_import "
            "WHERE status = 'accepted'"
        )
        for row in imports:
            name = row.get("canonical_name", "")
            for alias in row.get("aliases", []):
                result[alias] = name
            # Also map canonical name to itself (for fast-path lookups)
            result[name] = name

        # Load from entity_alias table (may have more from KG resolution)
        aliases = await execute_query(
            "SELECT alias_text, canonical_entity.name AS canonical_name "
            "FROM entity_alias WHERE verified = true"
        )
        for row in aliases:
            alias = row.get("alias_text", "")
            canonical = row.get("canonical_name", "")
            if alias and canonical:
                result[alias] = canonical

        logger.info(f"Merged dictionary: {len(result)} entries")
        return result
