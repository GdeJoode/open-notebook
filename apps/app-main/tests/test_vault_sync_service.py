"""Tests for VaultSyncService."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app_main.services.vault_sync_service import VaultSyncService, _parse_frontmatter


class TestParseFrontmatter:
    def test_valid_yaml(self):
        content = "---\naliases:\n  - Foo\n  - Bar\ntype: organization\n---\nBody text"
        result = _parse_frontmatter(content)
        assert result["aliases"] == ["Foo", "Bar"]
        assert result["type"] == "organization"

    def test_no_frontmatter(self):
        result = _parse_frontmatter("Just some text without frontmatter")
        assert result == {}

    def test_empty_frontmatter(self):
        result = _parse_frontmatter("---\n---\nBody")
        assert result == {}

    def test_single_alias_string(self):
        content = "---\naliases: Single Alias\ntype: person\n---\n"
        result = _parse_frontmatter(content)
        assert result["aliases"] == "Single Alias"

    def test_no_type_field(self):
        content = "---\naliases:\n  - A\n---\n"
        result = _parse_frontmatter(content)
        assert "type" not in result
        assert result["aliases"] == ["A"]


class TestScanVaultEntities:
    @pytest.mark.asyncio
    async def test_scan_nonexistent_folder(self):
        svc = VaultSyncService()
        result = await svc.scan_vault_entities("/nonexistent/path", "Entities")
        assert "error" in result
        assert result["total_files"] == 0

    @pytest.mark.asyncio
    async def test_scan_empty_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entities_dir = Path(tmpdir) / "Entities"
            entities_dir.mkdir()

            svc = VaultSyncService()
            result = await svc.scan_vault_entities(tmpdir, "Entities")
            assert result["new"] == 0
            assert result["total_files"] == 0

    @pytest.mark.asyncio
    async def test_scan_finds_new_entities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entities_dir = Path(tmpdir) / "Entities"
            entities_dir.mkdir()

            # Create test entity notes
            (entities_dir / "BZK.md").write_text(
                "---\naliases:\n  - Binnenlandse Zaken\ntype: organization\n---\n"
            )
            (entities_dir / "EZK.md").write_text(
                "---\naliases:\n  - Economische Zaken\ntype: organization\n---\n"
            )

            svc = VaultSyncService()

            # Mock DB queries to return no existing imports
            with patch(
                "app_main.services.vault_sync_service.execute_query",
                new_callable=AsyncMock,
            ) as mock_query:
                # First call per file: check existing → empty
                # Second call per file: CREATE
                mock_query.side_effect = [
                    [],  # BZK: no existing
                    [],  # BZK: CREATE
                    [],  # EZK: no existing
                    [],  # EZK: CREATE
                ]

                result = await svc.scan_vault_entities(tmpdir, "Entities")

            assert result["new"] == 2
            assert result["total_files"] == 2
            assert result["unchanged"] == 0

    @pytest.mark.asyncio
    async def test_scan_detects_unchanged(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entities_dir = Path(tmpdir) / "Entities"
            entities_dir.mkdir()

            (entities_dir / "BZK.md").write_text(
                "---\naliases:\n  - Binnenlandse Zaken\ntype: organization\n---\n"
            )

            svc = VaultSyncService()

            with patch(
                "app_main.services.vault_sync_service.execute_query",
                new_callable=AsyncMock,
            ) as mock_query:
                # Existing import matches current file
                mock_query.return_value = [{
                    "id": "vault_import:1",
                    "canonical_name": "BZK",
                    "aliases": ["Binnenlandse Zaken"],
                    "entity_type": "ORGANIZATION",
                }]

                result = await svc.scan_vault_entities(tmpdir, "Entities")

            assert result["unchanged"] == 1
            assert result["new"] == 0


class TestGetPendingImports:
    @pytest.mark.asyncio
    async def test_returns_pending(self):
        svc = VaultSyncService()
        with patch(
            "app_main.services.vault_sync_service.execute_query",
            new_callable=AsyncMock,
            return_value=[
                {"id": "vault_import:1", "canonical_name": "BZK", "status": "pending"},
            ],
        ):
            result = await svc.get_pending_imports()
        assert len(result) == 1
        assert result[0]["canonical_name"] == "BZK"


class TestApplyImports:
    @pytest.mark.asyncio
    async def test_accept_creates_entity_and_aliases(self):
        svc = VaultSyncService()

        call_count = 0

        async def mock_query(query, params=None, config=None):
            nonlocal call_count
            call_count += 1
            # SELECT vault_import by id
            if "FROM $id" in query and call_count == 1:
                return [{
                    "id": "vault_import:1",
                    "canonical_name": "BZK",
                    "aliases": ["Binnenlandse Zaken"],
                    "entity_type": "ORGANIZATION",
                }]
            # SELECT entity by name
            if "FROM entity WHERE" in query:
                return [{"id": "entity:abc"}]
            return []

        with patch(
            "app_main.services.vault_sync_service.execute_query",
            side_effect=mock_query,
        ):
            result = await svc.apply_imports(
                accepted_ids=["vault_import:1"],
                rejected_ids=[],
            )

        assert result["accepted"] == 1
        assert result["rejected"] == 0

    @pytest.mark.asyncio
    async def test_reject_marks_status(self):
        svc = VaultSyncService()

        with patch(
            "app_main.services.vault_sync_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await svc.apply_imports(
                accepted_ids=[],
                rejected_ids=["vault_import:2"],
            )

        assert result["rejected"] == 1


class TestQueueExport:
    @pytest.mark.asyncio
    async def test_queues_new_entities(self):
        svc = VaultSyncService()

        call_count = 0

        async def mock_query(query, params=None, config=None):
            nonlocal call_count
            call_count += 1
            return []  # No existing imports or exports

        with patch(
            "app_main.services.vault_sync_service.execute_query",
            side_effect=mock_query,
        ):
            count = await svc.queue_export([
                {"text": "New Entity", "label": "CONCEPT"},
            ])

        assert count == 1

    @pytest.mark.asyncio
    async def test_skips_already_known(self):
        svc = VaultSyncService()

        async def mock_query(query, params=None, config=None):
            if "vault_import" in query:
                return [{"id": "vault_import:1"}]  # Already known
            return []

        with patch(
            "app_main.services.vault_sync_service.execute_query",
            side_effect=mock_query,
        ):
            count = await svc.queue_export([
                {"text": "Known Entity", "label": "ORG"},
            ])

        assert count == 0

    @pytest.mark.asyncio
    async def test_skips_short_names(self):
        svc = VaultSyncService()
        with patch(
            "app_main.services.vault_sync_service.execute_query",
            new_callable=AsyncMock,
        ):
            count = await svc.queue_export([
                {"text": "X", "label": "UNKNOWN"},  # Too short
            ])
        assert count == 0


class TestWriteExports:
    @pytest.mark.asyncio
    async def test_writes_stub_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            svc = VaultSyncService()

            async def mock_query(query, params=None, config=None):
                if "FROM $id" in query:
                    return [{
                        "id": "vault_export:1",
                        "canonical_name": "TestEntity",
                        "aliases": ["TE", "Test"],
                        "entity_type": "CONCEPT",
                        "source_ids": ["source:1"],
                    }]
                return []

            with patch(
                "app_main.services.vault_sync_service.execute_query",
                side_effect=mock_query,
            ):
                result = await svc.write_exports(
                    accepted_ids=["vault_export:1"],
                    rejected_ids=[],
                    vault_path=tmpdir,
                    entities_folder="Entities",
                )

            assert result["written"] == 1
            stub = Path(tmpdir) / "Entities" / "TestEntity.md"
            assert stub.exists()
            content = stub.read_text()
            assert "aliases:" in content
            assert "type: concept" in content

    @pytest.mark.asyncio
    async def test_skips_existing_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            entities_dir = Path(tmpdir) / "Entities"
            entities_dir.mkdir()
            (entities_dir / "Existing.md").write_text("Already there")

            svc = VaultSyncService()

            async def mock_query(query, params=None, config=None):
                if "FROM $id" in query:
                    return [{
                        "id": "vault_export:1",
                        "canonical_name": "Existing",
                        "aliases": [],
                        "entity_type": "ORG",
                        "source_ids": [],
                    }]
                return []

            with patch(
                "app_main.services.vault_sync_service.execute_query",
                side_effect=mock_query,
            ):
                result = await svc.write_exports(
                    accepted_ids=["vault_export:1"],
                    rejected_ids=[],
                    vault_path=tmpdir,
                    entities_folder="Entities",
                )

            assert result["written"] == 0  # Skipped, not overwritten


class TestGetMergedDictionary:
    @pytest.mark.asyncio
    async def test_merges_imports_and_aliases(self):
        svc = VaultSyncService()

        call_count = 0

        async def mock_query(query, params=None, config=None):
            nonlocal call_count
            call_count += 1
            if "vault_import" in query:
                return [
                    {"canonical_name": "BZK", "aliases": ["Binnenlandse Zaken"]},
                ]
            if "entity_alias" in query:
                return [
                    {"alias_text": "EZK", "canonical_name": "Economische Zaken"},
                ]
            return []

        with patch(
            "app_main.services.vault_sync_service.execute_query",
            side_effect=mock_query,
        ):
            result = await svc.get_merged_dictionary()

        assert result["Binnenlandse Zaken"] == "BZK"
        assert result["BZK"] == "BZK"  # Self-mapping
        assert result["EZK"] == "Economische Zaken"
