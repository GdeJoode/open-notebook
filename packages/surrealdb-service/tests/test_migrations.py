"""Tests for surrealdb_service.migrations."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from surrealdb_service.migrations import (
    AsyncMigration,
    AsyncMigrationManager,
    AsyncMigrationRunner,
    discover_migrations,
)


# ---------------------------------------------------------------------------
# AsyncMigration
# ---------------------------------------------------------------------------

class TestAsyncMigration:
    def test_from_file_strips_comments(self, tmp_path: Path):
        sql_file = tmp_path / "1.surrealql"
        sql_file.write_text(
            "-- comment\nCREATE TABLE foo;\n-- another comment\nINSERT INTO foo;\n"
        )
        migration = AsyncMigration.from_file(sql_file)
        assert migration.sql == "CREATE TABLE foo; INSERT INTO foo;"
        assert migration.version == 1

    def test_from_file_parses_version(self, tmp_path: Path):
        sql_file = tmp_path / "26.surrealql"
        sql_file.write_text("SELECT 1;")
        migration = AsyncMigration.from_file(sql_file)
        assert migration.version == 26

    def test_from_file_parses_down_version(self, tmp_path: Path):
        sql_file = tmp_path / "10_down.surrealql"
        sql_file.write_text("DROP TABLE foo;")
        migration = AsyncMigration.from_file(sql_file)
        assert migration.version == 10

    def test_from_file_raises_on_bad_name(self, tmp_path: Path):
        sql_file = tmp_path / "bad_name.surrealql"
        sql_file.write_text("SELECT 1;")
        with pytest.raises(ValueError, match="Cannot parse version"):
            AsyncMigration.from_file(sql_file)

    def test_empty_lines_skipped(self, tmp_path: Path):
        sql_file = tmp_path / "2.surrealql"
        sql_file.write_text("\n\n  \nSELECT 1;\n\n")
        migration = AsyncMigration.from_file(sql_file)
        assert migration.sql == "SELECT 1;"


# ---------------------------------------------------------------------------
# discover_migrations
# ---------------------------------------------------------------------------

class TestDiscoverMigrations:
    def test_discovers_up_and_down(self, tmp_path: Path):
        (tmp_path / "1.surrealql").write_text("UP 1;")
        (tmp_path / "1_down.surrealql").write_text("DOWN 1;")
        (tmp_path / "2.surrealql").write_text("UP 2;")
        (tmp_path / "10.surrealql").write_text("UP 10;")
        (tmp_path / "10_down.surrealql").write_text("DOWN 10;")

        up, down = discover_migrations(tmp_path)
        assert set(up.keys()) == {1, 2, 10}
        assert set(down.keys()) == {1, 10}

    def test_missing_dir_returns_empty(self, tmp_path: Path):
        up, down = discover_migrations(tmp_path / "nonexistent")
        assert up == {}
        assert down == {}

    def test_skips_non_numeric_files(self, tmp_path: Path):
        (tmp_path / "readme.surrealql").write_text("nope")
        (tmp_path / "3.surrealql").write_text("SELECT 1;")
        up, down = discover_migrations(tmp_path)
        assert set(up.keys()) == {3}
        assert down == {}


# ---------------------------------------------------------------------------
# AsyncMigrationRunner
# ---------------------------------------------------------------------------

class TestAsyncMigrationRunner:
    @pytest.mark.asyncio
    async def test_run_all_skips_applied(self):
        up = {
            1: AsyncMigration(1, "UP 1"),
            2: AsyncMigration(2, "UP 2"),
            3: AsyncMigration(3, "UP 3"),
        }
        runner = AsyncMigrationRunner(up=up, down={})

        executed: list[str] = []

        async def mock_execute(sql, params=None, config=None):
            executed.append(sql)
            return []

        # Current version is 2 → only migration 3 should run
        with (
            patch(
                "surrealdb_service.migrations._get_latest_version",
                new_callable=AsyncMock,
                return_value=2,
            ),
            patch(
                "surrealdb_service.migrations.execute_query",
                side_effect=mock_execute,
            ),
            patch(
                "surrealdb_service.migrations._bump_version",
                new_callable=AsyncMock,
            ) as mock_bump,
        ):
            await runner.run_all()

        assert "UP 3" in executed
        assert "UP 1" not in executed
        assert "UP 2" not in executed
        mock_bump.assert_awaited_once_with(3, None)

    @pytest.mark.asyncio
    async def test_run_one_up(self):
        up = {1: AsyncMigration(1, "UP 1"), 2: AsyncMigration(2, "UP 2")}
        runner = AsyncMigrationRunner(up=up, down={})

        with (
            patch(
                "surrealdb_service.migrations._get_latest_version",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch(
                "surrealdb_service.migrations.execute_query",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_exec,
            patch(
                "surrealdb_service.migrations._bump_version",
                new_callable=AsyncMock,
            ),
        ):
            await runner.run_one_up()

        # Should only run the first pending migration
        mock_exec.assert_awaited_once_with("UP 1", config=None)

    @pytest.mark.asyncio
    async def test_run_one_down(self):
        down = {1: AsyncMigration(1, "DOWN 1"), 2: AsyncMigration(2, "DOWN 2")}
        runner = AsyncMigrationRunner(up={}, down=down)

        with (
            patch(
                "surrealdb_service.migrations._get_latest_version",
                new_callable=AsyncMock,
                return_value=2,
            ),
            patch(
                "surrealdb_service.migrations.execute_query",
                new_callable=AsyncMock,
                return_value=[],
            ) as mock_exec,
            patch(
                "surrealdb_service.migrations._lower_version",
                new_callable=AsyncMock,
            ),
        ):
            await runner.run_one_down()

        mock_exec.assert_awaited_once_with("DOWN 2", config=None)


# ---------------------------------------------------------------------------
# AsyncMigrationManager
# ---------------------------------------------------------------------------

class TestAsyncMigrationManager:
    def _make_dir(self, tmp_path: Path) -> Path:
        (tmp_path / "1.surrealql").write_text("UP 1;")
        (tmp_path / "1_down.surrealql").write_text("DOWN 1;")
        (tmp_path / "2.surrealql").write_text("UP 2;")
        (tmp_path / "2_down.surrealql").write_text("DOWN 2;")
        return tmp_path

    @pytest.mark.asyncio
    async def test_needs_migration_true(self, tmp_path: Path):
        d = self._make_dir(tmp_path)
        mgr = AsyncMigrationManager(migrations_dir=d)

        with patch(
            "surrealdb_service.migrations._get_latest_version",
            new_callable=AsyncMock,
            return_value=0,
        ):
            assert await mgr.needs_migration() is True

    @pytest.mark.asyncio
    async def test_needs_migration_false(self, tmp_path: Path):
        d = self._make_dir(tmp_path)
        mgr = AsyncMigrationManager(migrations_dir=d)

        with patch(
            "surrealdb_service.migrations._get_latest_version",
            new_callable=AsyncMock,
            return_value=2,
        ):
            assert await mgr.needs_migration() is False

    @pytest.mark.asyncio
    async def test_get_current_version(self, tmp_path: Path):
        d = self._make_dir(tmp_path)
        mgr = AsyncMigrationManager(migrations_dir=d)

        with patch(
            "surrealdb_service.migrations._get_latest_version",
            new_callable=AsyncMock,
            return_value=5,
        ):
            assert await mgr.get_current_version() == 5

    @pytest.mark.asyncio
    async def test_run_migration_up_delegates(self, tmp_path: Path):
        d = self._make_dir(tmp_path)
        mgr = AsyncMigrationManager(migrations_dir=d)

        with (
            patch(
                "surrealdb_service.migrations._get_latest_version",
                new_callable=AsyncMock,
                return_value=0,
            ),
            patch.object(
                mgr._runner, "run_all", new_callable=AsyncMock
            ) as mock_run,
        ):
            await mgr.run_migration_up()

        mock_run.assert_awaited_once()
