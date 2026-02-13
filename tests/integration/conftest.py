"""
Shared fixtures for integration tests.

Provides mock repositories, realistic test data, and filesystem helpers
for cross-package integration testing without SurrealDB or LLM calls.
"""

import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from shared.models.file_tracking import PipelineCacheEntry, SourceFolder


# ---------------------------------------------------------------------------
# Factory functions (not fixtures — call directly in tests)
# ---------------------------------------------------------------------------


def make_source_folder(**overrides: Any) -> SourceFolder:
    """Build a SourceFolder with sensible defaults."""
    defaults: Dict[str, Any] = {
        "id": "source_folder:test1",
        "source_id": "ab12cd34",
        "title": "Test Document",
        "slug": "Test_Document",
        "folder_path": "/tmp/test_sources/Test_Document_ab12cd34",
        "source_type": "document",
        "total_cache_entries": 0,
        "total_folder_size_bytes": 0,
        "metadata": {},
        "created": datetime.now(timezone.utc),
        "updated": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return SourceFolder(**defaults)


def make_cache_entry(**overrides: Any) -> PipelineCacheEntry:
    """Build a PipelineCacheEntry with sensible defaults."""
    defaults: Dict[str, Any] = {
        "id": "pipeline_cache:test1",
        "source_folder_id": "source_folder:test1",
        "source_id": "ab12cd34",
        "pipeline_step": "docling_parse",
        "version": 1,
        "is_current": True,
        "file_path": "/tmp/test_cache/ab12cd34_docling_parse.json",
        "file_format": "json",
        "file_size_bytes": 1024,
        "metadata": {},
        "created": datetime.now(timezone.utc),
        "updated": datetime.now(timezone.utc),
    }
    defaults.update(overrides)
    return PipelineCacheEntry(**defaults)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def integration_temp_dir(tmp_path: Path) -> Path:
    """Root temp directory for integration tests (auto-cleaned by pytest)."""
    return tmp_path


@pytest.fixture
def storage_root(integration_temp_dir: Path) -> Path:
    """Simulated STORAGE_ROOT with sources/ subdir."""
    root = integration_temp_dir / "storage"
    (root / "sources").mkdir(parents=True)
    return root


@pytest.fixture
def mock_source_folder_repo(storage_root: Path) -> AsyncMock:
    """
    AsyncMock of SourceFolderRepository.

    - create() builds a real SourceFolder from input data
    - get_or_raise() returns the last-created folder
    - get() returns the last-created folder
    - update() returns a modified copy
    - delete() is a no-op
    """
    repo = AsyncMock()
    _created_folders: Dict[str, SourceFolder] = {}
    _id_counter = [0]

    async def _create(data: Dict[str, Any]) -> SourceFolder:
        _id_counter[0] += 1
        folder_id = f"source_folder:test{_id_counter[0]}"
        folder = SourceFolder(
            id=folder_id,
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            **data,
        )
        _created_folders[folder_id] = folder
        return folder

    async def _get_or_raise(folder_id: str) -> SourceFolder:
        if folder_id in _created_folders:
            return _created_folders[folder_id]
        raise ValueError(f"Not found: {folder_id}")

    async def _get(folder_id: str) -> Optional[SourceFolder]:
        return _created_folders.get(folder_id)

    async def _update(folder_id: str, data: Dict[str, Any]) -> SourceFolder:
        folder = _created_folders.get(folder_id)
        if not folder:
            raise ValueError(f"Not found: {folder_id}")
        updated = folder.model_copy(update=data)
        _created_folders[folder_id] = updated
        return updated

    async def _get_all(**kwargs: Any) -> List[SourceFolder]:
        return list(_created_folders.values())

    repo.create = AsyncMock(side_effect=_create)
    repo.get_or_raise = AsyncMock(side_effect=_get_or_raise)
    repo.get = AsyncMock(side_effect=_get)
    repo.update = AsyncMock(side_effect=_update)
    repo.delete = AsyncMock()
    repo.get_all = AsyncMock(side_effect=_get_all)
    repo.search_by_filename = AsyncMock(return_value=[])
    repo._created_folders = _created_folders

    return repo


@pytest.fixture
def mock_pipeline_cache_repo() -> AsyncMock:
    """
    AsyncMock of PipelineCacheRepository.

    - create() returns a PipelineCacheEntry with auto-incrementing IDs
    - get_current() tracks current entries per (folder_id, step)
    - get_all_for_source() returns all current entries for a folder
    - mark_not_current() marks entries as not current
    """
    repo = AsyncMock()
    _entries: Dict[str, PipelineCacheEntry] = {}
    _current: Dict[str, PipelineCacheEntry] = {}  # key = f"{folder_id}:{step}"
    _id_counter = [0]

    async def _create(data: Dict[str, Any]) -> PipelineCacheEntry:
        _id_counter[0] += 1
        entry_id = f"pipeline_cache:test{_id_counter[0]}"
        entry = PipelineCacheEntry(
            id=entry_id,
            created=datetime.now(timezone.utc),
            updated=datetime.now(timezone.utc),
            **data,
        )
        _entries[entry_id] = entry
        key = f"{data['source_folder_id']}:{data['pipeline_step']}"
        if data.get("is_current", True):
            _current[key] = entry
        return entry

    async def _get_current(
        source_folder_id: str, pipeline_step: str
    ) -> Optional[PipelineCacheEntry]:
        key = f"{source_folder_id}:{pipeline_step}"
        return _current.get(key)

    async def _get_all_for_source(
        source_folder_id: str,
    ) -> List[PipelineCacheEntry]:
        return [
            e
            for e in _current.values()
            if e.source_folder_id == source_folder_id
        ]

    async def _mark_not_current(
        source_folder_id: str, pipeline_step: str
    ) -> None:
        key = f"{source_folder_id}:{pipeline_step}"
        old = _current.pop(key, None)
        if old:
            # Update the stored entry's is_current flag
            _entries[old.id] = old.model_copy(update={"is_current": False})

    async def _get_version_history(
        source_folder_id: str, pipeline_step: str
    ) -> List[PipelineCacheEntry]:
        return [
            e
            for e in _entries.values()
            if e.source_folder_id == source_folder_id
            and e.pipeline_step == pipeline_step
        ]

    repo.create = AsyncMock(side_effect=_create)
    repo.get_current = AsyncMock(side_effect=_get_current)
    repo.get_all_for_source = AsyncMock(side_effect=_get_all_for_source)
    repo.mark_not_current = AsyncMock(side_effect=_mark_not_current)
    repo.get_version_history = AsyncMock(side_effect=_get_version_history)
    repo._entries = _entries
    repo._current = _current

    return repo


@pytest.fixture
def source_folder_service(
    mock_source_folder_repo: AsyncMock, storage_root: Path
) -> "SourceFolderService":
    """Real SourceFolderService with mock repo and real temp filesystem."""
    from file_manager.services.source_folder import SourceFolderService

    return SourceFolderService(
        repo=mock_source_folder_repo,
        storage_root=str(storage_root),
    )


@pytest.fixture
def pipeline_cache_service(
    mock_pipeline_cache_repo: AsyncMock,
    mock_source_folder_repo: AsyncMock,
) -> "PipelineCacheService":
    """Real PipelineCacheService with mock repos."""
    from file_manager.services.pipeline_cache import PipelineCacheService

    return PipelineCacheService(
        cache_repo=mock_pipeline_cache_repo,
        folder_repo=mock_source_folder_repo,
    )


@pytest.fixture
def realistic_extraction_result() -> ExtractionResult:
    """
    20 entities (3 noise, 2 near-duplicates), 15 relations.

    Mimics ontology-extraction output with realistic variety.
    """
    entities = [
        # Valid PERSON entities
        ExtractedEntity(text="Albert Einstein", label="PERSON", confidence=0.95),
        ExtractedEntity(text="Marie Curie", label="PERSON", confidence=0.92),
        ExtractedEntity(text="Niels Bohr", label="PERSON", confidence=0.88),
        ExtractedEntity(text="Max Planck", label="PERSON", confidence=0.91),
        ExtractedEntity(text="Werner Heisenberg", label="PERSON", confidence=0.87),
        # Duplicate of Einstein (different casing)
        ExtractedEntity(text="albert einstein", label="PERSON", confidence=0.80),
        # Near-duplicate of Einstein
        ExtractedEntity(text="Einstein", label="PERSON", confidence=0.75),
        # Valid ORG entities
        ExtractedEntity(text="University of Berlin", label="ORG", confidence=0.90),
        ExtractedEntity(text="Nobel Committee", label="ORG", confidence=0.85),
        ExtractedEntity(text="Kaiser Wilhelm Institute", label="ORG", confidence=0.88),
        # Valid concepts
        ExtractedEntity(text="quantum mechanics", label="CONCEPT", confidence=0.92),
        ExtractedEntity(text="general relativity", label="CONCEPT", confidence=0.90),
        ExtractedEntity(text="photoelectric effect", label="CONCEPT", confidence=0.88),
        # Valid LOC
        ExtractedEntity(text="Copenhagen", label="LOC", confidence=0.85),
        ExtractedEntity(text="Berlin", label="LOC", confidence=0.83),
        # Valid EVENT
        ExtractedEntity(text="Solvay Conference", label="EVENT", confidence=0.82),
        ExtractedEntity(text="Nobel Prize in Physics", label="EVENT", confidence=0.90),
        # Noise entities (should be removed)
        ExtractedEntity(text="the", label="MISC", confidence=0.3),
        ExtractedEntity(text="123", label="MISC", confidence=0.2),
        ExtractedEntity(text="p.", label="MISC", confidence=0.25),
    ]

    relations = [
        ExtractedRelation(
            source_entity="Albert Einstein",
            target_entity="general relativity",
            relation_type="DEVELOPED",
            confidence=0.92,
        ),
        ExtractedRelation(
            source_entity="Albert Einstein",
            target_entity="photoelectric effect",
            relation_type="EXPLAINED",
            confidence=0.90,
        ),
        ExtractedRelation(
            source_entity="Marie Curie",
            target_entity="Nobel Prize in Physics",
            relation_type="RECEIVED",
            confidence=0.88,
        ),
        ExtractedRelation(
            source_entity="Albert Einstein",
            target_entity="Nobel Prize in Physics",
            relation_type="RECEIVED",
            confidence=0.91,
        ),
        ExtractedRelation(
            source_entity="Albert Einstein",
            target_entity="University of Berlin",
            relation_type="AFFILIATED_WITH",
            confidence=0.85,
        ),
        ExtractedRelation(
            source_entity="Niels Bohr",
            target_entity="Copenhagen",
            relation_type="LOCATED_IN",
            confidence=0.82,
        ),
        ExtractedRelation(
            source_entity="Niels Bohr",
            target_entity="quantum mechanics",
            relation_type="CONTRIBUTED_TO",
            confidence=0.88,
        ),
        ExtractedRelation(
            source_entity="Max Planck",
            target_entity="quantum mechanics",
            relation_type="FOUNDED",
            confidence=0.90,
        ),
        ExtractedRelation(
            source_entity="Max Planck",
            target_entity="Kaiser Wilhelm Institute",
            relation_type="AFFILIATED_WITH",
            confidence=0.87,
        ),
        ExtractedRelation(
            source_entity="Werner Heisenberg",
            target_entity="quantum mechanics",
            relation_type="CONTRIBUTED_TO",
            confidence=0.85,
        ),
        ExtractedRelation(
            source_entity="Albert Einstein",
            target_entity="Solvay Conference",
            relation_type="PARTICIPATED_IN",
            confidence=0.80,
        ),
        ExtractedRelation(
            source_entity="Niels Bohr",
            target_entity="Solvay Conference",
            relation_type="PARTICIPATED_IN",
            confidence=0.78,
        ),
        # Relations referencing noise entities (should be dropped)
        ExtractedRelation(
            source_entity="the",
            target_entity="Albert Einstein",
            relation_type="UNKNOWN",
            confidence=0.1,
        ),
        ExtractedRelation(
            source_entity="123",
            target_entity="Berlin",
            relation_type="UNKNOWN",
            confidence=0.15,
        ),
        ExtractedRelation(
            source_entity="p.",
            target_entity="quantum mechanics",
            relation_type="UNKNOWN",
            confidence=0.12,
        ),
    ]

    return ExtractionResult(
        entities=entities,
        relations=relations,
        metadata={"source": "test", "pipeline": "ontology-extraction"},
    )


@pytest.fixture
def minimal_extraction_result() -> ExtractionResult:
    """2 entities, 1 relation for fast targeted tests."""
    return ExtractionResult(
        entities=[
            ExtractedEntity(text="Alice", label="PERSON", confidence=0.95),
            ExtractedEntity(text="ACME Corp", label="ORG", confidence=0.90),
        ],
        relations=[
            ExtractedRelation(
                source_entity="Alice",
                target_entity="ACME Corp",
                relation_type="WORKS_AT",
                confidence=0.88,
            ),
        ],
        metadata={"source": "minimal_test"},
    )


@pytest.fixture
def sample_document(integration_temp_dir: Path) -> "ExtractedDocument":
    """
    ExtractedDocument built without Docling.

    Has markdown content, 2 pages, no images/tables.
    """
    from ingestion.models.document import (
        ElementType,
        ExtractedDocument,
        ExtractedElement,
        PageContent,
    )

    source_file = integration_temp_dir / "sample.pdf"
    source_file.write_text("fake pdf content for testing")

    page1 = PageContent(
        page_number=1,
        elements=[
            ExtractedElement(
                content="Introduction to Quantum Mechanics",
                element_type=ElementType.TITLE,
                page=1,
            ),
            ExtractedElement(
                content="This document explores the foundations of quantum theory.",
                element_type=ElementType.PARAGRAPH,
                page=1,
            ),
        ],
        text_content="Introduction to Quantum Mechanics\n\nThis document explores the foundations of quantum theory.",
    )

    page2 = PageContent(
        page_number=2,
        elements=[
            ExtractedElement(
                content="Wave-Particle Duality",
                element_type=ElementType.HEADING,
                page=2,
            ),
            ExtractedElement(
                content="Light exhibits both wave-like and particle-like properties.",
                element_type=ElementType.PARAGRAPH,
                page=2,
            ),
        ],
        text_content="Wave-Particle Duality\n\nLight exhibits both wave-like and particle-like properties.",
    )

    return ExtractedDocument(
        source_path=source_file,
        title="Introduction to Quantum Mechanics",
        page_count=2,
        pages=[page1, page2],
        full_markdown="# Introduction to Quantum Mechanics\n\nThis document explores the foundations of quantum theory.\n\n## Wave-Particle Duality\n\nLight exhibits both wave-like and particle-like properties.",
        full_text="Introduction to Quantum Mechanics. This document explores the foundations of quantum theory. Wave-Particle Duality. Light exhibits both wave-like and particle-like properties.",
    )
