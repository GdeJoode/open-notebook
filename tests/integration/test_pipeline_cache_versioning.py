"""
Pipeline cache versioning: multi-version save/load/archive cycles.

Validates that saving multiple versions of the same pipeline step
archives old files and preserves only the current one with the
canonical name.
"""

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


class TestPipelineCacheVersioning:
    """Multi-version cache behavior with real filesystem."""

    async def test_three_version_save_cycle(
        self, source_folder_service, pipeline_cache_service
    ):
        """Save v1, v2, v3 → only v3 has canonical name, v1/v2 archived with timestamp."""
        folder = await source_folder_service.create_folder(title="Version Test")

        # Save three versions of the same step
        entry_v1 = await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="entity_extraction",
            data={"version": 1, "entities": ["A"]},
        )
        entry_v2 = await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="entity_extraction",
            data={"version": 2, "entities": ["A", "B"]},
        )
        entry_v3 = await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="entity_extraction",
            data={"version": 3, "entities": ["A", "B", "C"]},
        )

        # The canonical file should contain v3 data
        canonical_path = Path(entry_v3.file_path)
        assert canonical_path.exists()
        content = json.loads(canonical_path.read_text())
        assert content["version"] == 3
        assert content["entities"] == ["A", "B", "C"]

        # There should be archived files (timestamp-suffixed)
        cache_dir = canonical_path.parent
        all_files = sorted(cache_dir.iterdir())
        # canonical + at least 1 archived (v1 might be archived twice or once)
        assert len(all_files) >= 2, (
            f"Expected at least 2 files (canonical + archive), got {len(all_files)}: "
            f"{[f.name for f in all_files]}"
        )

    async def test_different_steps_independent(
        self, source_folder_service, pipeline_cache_service
    ):
        """docling_parse and entity_extraction for same source → both have canonical names."""
        folder = await source_folder_service.create_folder(title="Steps Test")

        entry_parse = await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="docling_parse",
            data={"pages": 5},
        )
        entry_extract = await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="entity_extraction",
            data={"entities": ["Alice"]},
        )

        parse_path = Path(entry_parse.file_path)
        extract_path = Path(entry_extract.file_path)

        assert parse_path.exists()
        assert extract_path.exists()
        assert "docling_parse" in parse_path.name
        assert "entity_extraction" in extract_path.name

        # Both should be readable
        assert json.loads(parse_path.read_text())["pages"] == 5
        assert json.loads(extract_path.read_text())["entities"] == ["Alice"]

    async def test_content_changes_between_versions(
        self, source_folder_service, pipeline_cache_service
    ):
        """Save data_a then data_b → reading current returns data_b."""
        folder = await source_folder_service.create_folder(title="Content Change")

        await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="summarization",
            data="First summary: short",
            file_format="md",
        )
        await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="summarization",
            data="Second summary: much more detailed and comprehensive",
            file_format="md",
        )

        content = await pipeline_cache_service.get_output(
            folder.id, "summarization"
        )
        assert content is not None
        assert "Second summary" in content
        assert "First summary" not in content

    async def test_config_hash_differs_between_configs(
        self, source_folder_service, pipeline_cache_service
    ):
        """Save with config_a then config_b → different config_hash values."""
        folder = await source_folder_service.create_folder(title="Config Hash")

        entry_a = await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="entity_extraction",
            data={"entities": ["A"]},
            config_used={"model": "gpt-4", "temperature": 0.0},
        )
        entry_b = await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="entity_extraction",
            data={"entities": ["A", "B"]},
            config_used={"model": "claude-3", "temperature": 0.5},
        )

        assert entry_a.config_hash is not None
        assert entry_b.config_hash is not None
        assert entry_a.config_hash != entry_b.config_hash
