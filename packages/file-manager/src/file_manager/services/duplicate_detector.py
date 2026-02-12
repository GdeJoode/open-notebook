"""
Duplicate detection service.

Metadata-based duplicate detection without content hashing.
Uses filename similarity, file size tolerance, and MIME type matching.
"""

import difflib
from dataclasses import dataclass, field
from typing import List, Optional

from shared.models.file_tracking import SourceFolder
from surrealdb_service.repositories.file_tracking import SourceFolderRepository


@dataclass
class DuplicateMatch:
    """A potential duplicate match."""
    source_folder: SourceFolder
    similarity_score: float
    matched_criteria: List[str]


@dataclass
class DuplicateCheckResult:
    """Result of a duplicate check."""
    match_type: str  # "definite", "probable", "none"
    matches: List[DuplicateMatch] = field(default_factory=list)
    message: str = ""


class DuplicateDetector:
    """
    Metadata-based duplicate detection (NO content hashing).

    Scoring:
    - Filename similarity (SequenceMatcher, threshold 0.85)
    - File size tolerance (5%)
    - MIME type exact match

    Classification:
    - All 3 criteria match = "definite"
    - 2 criteria match = "probable"
    - <2 = "none"
    """

    FILENAME_THRESHOLD = 0.85
    SIZE_TOLERANCE = 0.05

    def __init__(self, repo: SourceFolderRepository):
        self.repo = repo

    async def check_duplicate(
        self,
        filename: str,
        file_size_bytes: Optional[int] = None,
        mime_type: Optional[str] = None,
    ) -> DuplicateCheckResult:
        """
        Check for potential duplicates of a file.

        Args:
            filename: Filename to check.
            file_size_bytes: File size for comparison.
            mime_type: MIME type for comparison.

        Returns:
            DuplicateCheckResult with match type and matches.
        """
        # Get all source folders (could be optimized with search_by_filename
        # but for now we compare in-memory for accuracy)
        candidates = await self.repo.search_by_filename(filename, mime_type)

        # If no candidates from DB search, also try a broader search
        if not candidates and mime_type:
            candidates = await self.repo.search_by_filename(filename)

        matches: List[DuplicateMatch] = []

        for candidate in candidates:
            if not candidate.original_filename:
                continue

            matched_criteria: List[str] = []

            # Filename similarity
            similarity = difflib.SequenceMatcher(
                None,
                filename.lower(),
                candidate.original_filename.lower(),
            ).ratio()

            if similarity >= self.FILENAME_THRESHOLD:
                matched_criteria.append("filename")

            # File size tolerance
            if (
                file_size_bytes is not None
                and candidate.file_size_bytes is not None
                and candidate.file_size_bytes > 0
            ):
                size_diff = abs(file_size_bytes - candidate.file_size_bytes)
                tolerance = candidate.file_size_bytes * self.SIZE_TOLERANCE
                if size_diff <= tolerance:
                    matched_criteria.append("file_size")

            # MIME type exact match
            if mime_type and candidate.mime_type and mime_type == candidate.mime_type:
                matched_criteria.append("mime_type")

            if len(matched_criteria) >= 2:
                matches.append(DuplicateMatch(
                    source_folder=candidate,
                    similarity_score=similarity,
                    matched_criteria=matched_criteria,
                ))

        if not matches:
            return DuplicateCheckResult(
                match_type="none",
                message="No duplicates detected.",
            )

        # Sort by number of matched criteria (descending), then similarity
        matches.sort(
            key=lambda m: (len(m.matched_criteria), m.similarity_score),
            reverse=True,
        )

        best = matches[0]
        if len(best.matched_criteria) >= 3:
            match_type = "definite"
            message = (
                f"Definite duplicate: '{best.source_folder.original_filename}' "
                f"matches on {', '.join(best.matched_criteria)}"
            )
        else:
            match_type = "probable"
            message = (
                f"Probable duplicate: '{best.source_folder.original_filename}' "
                f"matches on {', '.join(best.matched_criteria)}"
            )

        return DuplicateCheckResult(
            match_type=match_type,
            matches=matches,
            message=message,
        )
