"""Tests for DuplicateDetector."""

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from file_manager.services.duplicate_detector import DuplicateDetector
from shared.models.file_tracking import SourceFolder


def make_source_folder(**overrides) -> SourceFolder:
    defaults = {
        "id": "source_folder:test_id",
        "source_id": "Ab3kX9q2",
        "title": "Test Document",
        "slug": "Test_Document",
        "folder_path": "/tmp/sources/Test_Document_Ab3kX9q2",
        "original_filename": "test_document.pdf",
        "file_size_bytes": 1024,
        "mime_type": "application/pdf",
        "source_type": "document",
        "total_cache_entries": 0,
        "total_folder_size_bytes": 0,
        "metadata": {},
        "created": datetime(2025, 1, 1),
        "updated": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    return SourceFolder(**defaults)


class TestDuplicateDetector:
    @pytest.fixture
    def mock_repo(self):
        return AsyncMock()

    @pytest.fixture
    def detector(self, mock_repo):
        return DuplicateDetector(repo=mock_repo)

    @pytest.mark.asyncio
    async def test_no_duplicates(self, detector, mock_repo):
        """No matches when DB is empty."""
        mock_repo.search_by_filename.return_value = []
        result = await detector.check_duplicate("unique_file.pdf")
        assert result.match_type == "none"
        assert len(result.matches) == 0

    @pytest.mark.asyncio
    async def test_definite_duplicate_all_criteria(self, detector, mock_repo):
        """All 3 criteria matching = definite duplicate."""
        existing = make_source_folder(
            original_filename="report_2024.pdf",
            file_size_bytes=10000,
            mime_type="application/pdf",
        )
        mock_repo.search_by_filename.return_value = [existing]

        result = await detector.check_duplicate(
            filename="report_2024.pdf",
            file_size_bytes=10000,
            mime_type="application/pdf",
        )

        assert result.match_type == "definite"
        assert len(result.matches) == 1
        assert "filename" in result.matches[0].matched_criteria
        assert "file_size" in result.matches[0].matched_criteria
        assert "mime_type" in result.matches[0].matched_criteria

    @pytest.mark.asyncio
    async def test_probable_duplicate_two_criteria(self, detector, mock_repo):
        """2 criteria matching = probable duplicate."""
        existing = make_source_folder(
            original_filename="report_2024.pdf",
            file_size_bytes=10000,
            mime_type="application/pdf",
        )
        mock_repo.search_by_filename.return_value = [existing]

        result = await detector.check_duplicate(
            filename="report_2024.pdf",
            file_size_bytes=50000,  # Different size
            mime_type="application/pdf",
        )

        assert result.match_type == "probable"
        assert len(result.matches) == 1
        assert "filename" in result.matches[0].matched_criteria
        assert "mime_type" in result.matches[0].matched_criteria

    @pytest.mark.asyncio
    async def test_no_match_one_criterion(self, detector, mock_repo):
        """Only 1 criterion matching = no duplicate."""
        existing = make_source_folder(
            original_filename="totally_different.docx",
            file_size_bytes=50000,
            mime_type="application/msword",
        )
        mock_repo.search_by_filename.return_value = [existing]

        result = await detector.check_duplicate(
            filename="report_2024.pdf",
            file_size_bytes=10000,
            mime_type="application/pdf",
        )

        assert result.match_type == "none"

    @pytest.mark.asyncio
    async def test_filename_similarity(self, detector, mock_repo):
        """Similar filenames are detected."""
        existing = make_source_folder(
            original_filename="quarterly_report_2024.pdf",
            file_size_bytes=10000,
            mime_type="application/pdf",
        )
        mock_repo.search_by_filename.return_value = [existing]

        result = await detector.check_duplicate(
            filename="quarterly_report_2024.pdf",
            file_size_bytes=10200,  # Within 5% tolerance
            mime_type="application/pdf",
        )

        assert result.match_type == "definite"

    @pytest.mark.asyncio
    async def test_file_size_tolerance(self, detector, mock_repo):
        """File sizes within 5% tolerance are considered matching."""
        existing = make_source_folder(
            original_filename="document.pdf",
            file_size_bytes=10000,
            mime_type="application/pdf",
        )
        mock_repo.search_by_filename.return_value = [existing]

        # 4.9% difference - within tolerance
        result = await detector.check_duplicate(
            filename="document.pdf",
            file_size_bytes=10490,
            mime_type="application/pdf",
        )
        assert result.match_type == "definite"

    @pytest.mark.asyncio
    async def test_file_size_outside_tolerance(self, detector, mock_repo):
        """File sizes outside 5% tolerance don't match."""
        existing = make_source_folder(
            original_filename="document.pdf",
            file_size_bytes=10000,
            mime_type="application/pdf",
        )
        mock_repo.search_by_filename.return_value = [existing]

        # 6% difference - outside tolerance
        result = await detector.check_duplicate(
            filename="document.pdf",
            file_size_bytes=10600,
            mime_type="application/pdf",
        )
        # Only filename and mime match, not size
        assert result.match_type == "probable"

    @pytest.mark.asyncio
    async def test_multiple_candidates(self, detector, mock_repo):
        """Multiple candidates are checked and best match returned first."""
        candidate1 = make_source_folder(
            id="source_folder:1",
            original_filename="report.pdf",
            file_size_bytes=10000,
            mime_type="application/pdf",
        )
        candidate2 = make_source_folder(
            id="source_folder:2",
            original_filename="report_v2.pdf",
            file_size_bytes=10000,
            mime_type="application/pdf",
        )
        mock_repo.search_by_filename.return_value = [candidate1, candidate2]

        result = await detector.check_duplicate(
            filename="report.pdf",
            file_size_bytes=10000,
            mime_type="application/pdf",
        )

        assert result.match_type == "definite"
        # First match should be the best (most criteria matched + highest similarity)
        assert result.matches[0].source_folder.id == "source_folder:1"

    @pytest.mark.asyncio
    async def test_no_original_filename_skipped(self, detector, mock_repo):
        """Candidates without original_filename are skipped."""
        existing = make_source_folder(original_filename=None)
        mock_repo.search_by_filename.return_value = [existing]

        result = await detector.check_duplicate("doc.pdf")
        assert result.match_type == "none"

    @pytest.mark.asyncio
    async def test_case_insensitive_filename(self, detector, mock_repo):
        """Filename comparison is case-insensitive."""
        existing = make_source_folder(
            original_filename="REPORT.PDF",
            file_size_bytes=1000,
            mime_type="application/pdf",
        )
        mock_repo.search_by_filename.return_value = [existing]

        result = await detector.check_duplicate(
            filename="report.pdf",
            mime_type="application/pdf",
        )

        assert result.match_type == "probable"
        assert any("filename" in m.matched_criteria for m in result.matches)
