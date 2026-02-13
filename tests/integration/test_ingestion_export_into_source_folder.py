"""
Cross-package integration: ingestion DocumentExporter writes into
file-manager SourceFolder structure.

Validates that ingestion's export pipeline produces output compatible
with the source folder layout created by file-manager.
"""

import json
from pathlib import Path

import pytest

from ingestion.config import OutputConfig
from ingestion.exporters.document_exporter import DocumentExporter

pytestmark = pytest.mark.integration


class TestIngestionExportIntoSourceFolder:
    """DocumentExporter writes into a source folder created by SourceFolderService."""

    async def test_document_exporter_into_source_folder(
        self, source_folder_service, sample_document
    ):
        """Build ExtractedDocument, export with source_folder_path + source_id → files in ingestion/ with prefix."""
        folder = await source_folder_service.create_folder(title="Export Test")
        folder_path = Path(folder.folder_path)

        exporter = DocumentExporter()
        exported = exporter.export(
            document=sample_document,
            source_folder_path=folder_path,
            source_id=folder.source_id,
            copy_original=False,  # skip original copy for this test
        )

        # Check ingestion output directory was created
        ingestion_dir = folder_path / "ingestion"
        assert ingestion_dir.is_dir()

        # Check markdown was written with source_id prefix
        markdown_files = list(ingestion_dir.glob(f"{folder.source_id}_document.md"))
        assert len(markdown_files) == 1
        md_content = markdown_files[0].read_text()
        assert "Quantum Mechanics" in md_content

        # Check JSON was written
        json_files = list(ingestion_dir.glob(f"{folder.source_id}_document.json"))
        assert len(json_files) == 1

        # Check metadata was written
        meta_files = list(ingestion_dir.glob(f"{folder.source_id}_metadata.json"))
        assert len(meta_files) == 1

    def test_output_config_creates_dirs_in_source_folder(
        self, integration_temp_dir
    ):
        """OutputConfig.create_directories_in_source_folder() → all expected subdirs exist."""
        source_folder = integration_temp_dir / "test_source_folder"
        source_folder.mkdir()

        config = OutputConfig()
        dirs = config.create_directories_in_source_folder(source_folder)

        assert dirs["root"] == source_folder
        assert dirs["output"].is_dir()
        assert dirs["images"].is_dir()
        assert dirs["tables"].is_dir()

        # Verify actual paths
        assert (source_folder / "ingestion").is_dir()
        assert (source_folder / "ingestion" / "images").is_dir()
        assert (source_folder / "ingestion" / "tables").is_dir()

    async def test_source_folder_structure_matches_ingestion_expectations(
        self, source_folder_service
    ):
        """SourceFolderService dirs + OutputConfig dirs → complete compatible structure."""
        folder = await source_folder_service.create_folder(title="Compat Test")
        folder_path = Path(folder.folder_path)

        # file-manager creates these
        assert (folder_path / "original").is_dir()
        assert (folder_path / "ingestion").is_dir()
        assert (folder_path / "pipeline_cache").is_dir()

        # ingestion OutputConfig adds these inside
        config = OutputConfig()
        config.create_directories_in_source_folder(folder_path)

        assert (folder_path / "ingestion" / "images").is_dir()
        assert (folder_path / "ingestion" / "tables").is_dir()

    async def test_original_and_ingestion_output_coexist(
        self, source_folder_service, sample_document, integration_temp_dir
    ):
        """store_original() + export() → both original/ and ingestion/ populated correctly."""
        folder = await source_folder_service.create_folder(title="Coexist Test")
        folder_path = Path(folder.folder_path)

        # Store the original file
        await source_folder_service.store_original(
            folder_id=folder.id,
            source_file_path=str(sample_document.source_path),
        )

        # Export ingestion output
        exporter = DocumentExporter()
        exporter.export(
            document=sample_document,
            source_folder_path=folder_path,
            source_id=folder.source_id,
            copy_original=False,
        )

        # Both directories should have content
        original_files = list((folder_path / "original").iterdir())
        ingestion_files = list((folder_path / "ingestion").glob("*.md"))

        assert len(original_files) >= 1
        assert len(ingestion_files) >= 1

    async def test_exported_content_readable_as_cache(
        self, source_folder_service, pipeline_cache_service, sample_document
    ):
        """Export markdown → save via PipelineCacheService → read back matches."""
        folder = await source_folder_service.create_folder(title="Cache Read")
        folder_path = Path(folder.folder_path)

        # Export document
        exporter = DocumentExporter()
        exported = exporter.export(
            document=sample_document,
            source_folder_path=folder_path,
            source_id=folder.source_id,
            copy_original=False,
        )

        # Read the exported markdown
        md_path = exported["markdown"]
        md_content = md_path.read_text()

        # Save it as a pipeline cache entry
        await pipeline_cache_service.save_output(
            source_folder_id=folder.id,
            pipeline_step="ingestion_markdown",
            data=md_content,
            file_format="md",
        )

        # Read it back via cache service
        cached = await pipeline_cache_service.get_output(
            folder.id, "ingestion_markdown"
        )
        assert cached is not None
        assert "Quantum Mechanics" in cached
        assert cached == md_content
