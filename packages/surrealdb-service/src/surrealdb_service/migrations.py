"""
Async migration system for SurrealDB.

Auto-discovers .surrealql files instead of hardcoding migration list.
Tracks applied versions in the ``_sbl_migrations`` table.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from surrealdb_service.connection import execute_query


# ---------------------------------------------------------------------------
# AsyncMigration
# ---------------------------------------------------------------------------

class AsyncMigration:
    """Single migration (up or down) represented as cleaned SQL text."""

    def __init__(self, version: int, sql: str) -> None:
        self.version = version
        self.sql = sql

    @classmethod
    def from_file(cls, path: Path) -> "AsyncMigration":
        """Create a migration from a ``.surrealql`` file.

        The filename must start with an integer version number, e.g.
        ``7.surrealql`` or ``7_down.surrealql``.
        """
        raw = path.read_text(encoding="utf-8")
        lines: list[str] = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if stripped and not stripped.startswith("--"):
                lines.append(stripped)
        sql = " ".join(lines)

        # Extract version number from filename (e.g. "10.surrealql" → 10)
        match = re.match(r"^(\d+)", path.stem)
        if not match:
            raise ValueError(f"Cannot parse version from filename: {path.name}")
        version = int(match.group(1))

        return cls(version=version, sql=sql)


# ---------------------------------------------------------------------------
# Discovery helper
# ---------------------------------------------------------------------------

def discover_migrations(
    migrations_dir: Path,
) -> tuple[Dict[int, AsyncMigration], Dict[int, AsyncMigration]]:
    """Glob ``migrations_dir`` for ``.surrealql`` files.

    Returns:
        (up_dict, down_dict) keyed by integer version.
    """
    up: Dict[int, AsyncMigration] = {}
    down: Dict[int, AsyncMigration] = {}

    if not migrations_dir.is_dir():
        logger.warning(f"Migrations directory not found: {migrations_dir}")
        return up, down

    for path in sorted(migrations_dir.glob("*.surrealql")):
        stem = path.stem  # e.g. "10" or "10_down"
        is_down = stem.endswith("_down")
        match = re.match(r"^(\d+)", stem)
        if not match:
            logger.warning(f"Skipping unrecognised migration file: {path.name}")
            continue
        version = int(match.group(1))
        migration = AsyncMigration.from_file(path)
        if is_down:
            down[version] = migration
        else:
            up[version] = migration

    logger.debug(
        f"Discovered {len(up)} up / {len(down)} down migrations in {migrations_dir}"
    )
    return up, down


# ---------------------------------------------------------------------------
# Version tracking (uses _sbl_migrations table)
# ---------------------------------------------------------------------------

async def _get_all_versions(config: Any = None) -> List[dict]:
    try:
        return await execute_query(
            "SELECT * FROM _sbl_migrations ORDER BY version;", config=config
        )
    except Exception:
        return []


async def _get_latest_version(config: Any = None) -> int:
    versions = await _get_all_versions(config)
    if not versions:
        return 0
    return max(v["version"] for v in versions)


async def _bump_version(version: int, config: Any = None) -> None:
    await execute_query(
        f"CREATE _sbl_migrations:{version} SET version = {version}, "
        f"applied_at = time::now();",
        config=config,
    )


async def _lower_version(version: int, config: Any = None) -> None:
    if version > 0:
        await execute_query(
            f"DELETE _sbl_migrations:{version};", config=config
        )


# ---------------------------------------------------------------------------
# AsyncMigrationRunner
# ---------------------------------------------------------------------------

class AsyncMigrationRunner:
    """Run ordered migrations (up or down)."""

    def __init__(
        self,
        up: Dict[int, AsyncMigration],
        down: Dict[int, AsyncMigration],
        config: Any = None,
    ) -> None:
        self.up = up
        self.down = down
        self.config = config

    async def run_all(self) -> None:
        """Run every pending *up* migration in version order."""
        current = await _get_latest_version(self.config)
        for version in sorted(self.up):
            if version <= current:
                continue
            logger.info(f"Running migration {version}")
            await execute_query(self.up[version].sql, config=self.config)
            await _bump_version(version, self.config)

    async def run_one_up(self) -> None:
        current = await _get_latest_version(self.config)
        pending = sorted(v for v in self.up if v > current)
        if pending:
            version = pending[0]
            logger.info(f"Running migration {version}")
            await execute_query(self.up[version].sql, config=self.config)
            await _bump_version(version, self.config)

    async def run_one_down(self) -> None:
        current = await _get_latest_version(self.config)
        if current > 0 and current in self.down:
            logger.info(f"Rolling back migration {current}")
            await execute_query(self.down[current].sql, config=self.config)
            await _lower_version(current, self.config)


# ---------------------------------------------------------------------------
# AsyncMigrationManager  (backward-compatible public API)
# ---------------------------------------------------------------------------

class AsyncMigrationManager:
    """High-level manager with the same API as the monolith version.

    Parameters
    ----------
    migrations_dir:
        Path to the directory containing ``.surrealql`` files.
        Defaults to ``<project-root>/migrations``.
    config:
        Optional :class:`SurrealDBConfig` — forwarded to ``execute_query``.
    """

    def __init__(
        self,
        migrations_dir: Optional[Path] = None,
        config: Any = None,
    ) -> None:
        if migrations_dir is None:
            migrations_dir = Path("migrations")
        self._dir = migrations_dir
        self._config = config
        self._up, self._down = discover_migrations(self._dir)
        self._runner = AsyncMigrationRunner(self._up, self._down, config)

    # -- public API (backward-compat) --

    async def get_current_version(self) -> int:
        return await _get_latest_version(self._config)

    async def needs_migration(self) -> bool:
        current = await self.get_current_version()
        return any(v > current for v in self._up)

    async def run_migration_up(self) -> None:
        current = await self.get_current_version()
        logger.info(f"Current version before migration: {current}")

        if await self.needs_migration():
            try:
                await self._runner.run_all()
                new_version = await self.get_current_version()
                logger.info(f"Migration successful. New version: {new_version}")
            except Exception as e:
                logger.error(f"Migration failed: {e}")
                raise
        else:
            logger.info("Database is already at the latest version")
