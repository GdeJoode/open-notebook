"""
Multi-strategy duplicate detection for Zotero items.

Cascade: DOI (exact) → ISBN (exact) → title+author (fuzzy).
"""

from difflib import SequenceMatcher
from typing import Any, Dict, Optional

from loguru import logger
from pyzotero import zotero


class DuplicateDetector:
    """Detect duplicate items in a Zotero library."""

    def __init__(self, zot: zotero.Zotero):
        self._zot = zot

    def find_duplicate(
        self,
        metadata: Dict[str, Any],
        threshold: float = 0.85,
    ) -> Optional[Dict[str, Any]]:
        """Run cascade duplicate detection.

        Priority:
        1. DOI (exact match, highest confidence)
        2. ISBN (exact match, for books)
        3. Title + author last name (fuzzy)

        Returns the existing Zotero item if found, None otherwise.
        """
        # 1. DOI match
        doi = metadata.get("DOI") or metadata.get("doi")
        if doi:
            result = self._find_by_doi(doi)
            if result:
                logger.info(f"Duplicate found by DOI: {doi} → {result['key']}")
                return result

        # 2. ISBN match
        isbn = metadata.get("ISBN") or metadata.get("isbn")
        if isbn:
            result = self._find_by_isbn(isbn)
            if result:
                logger.info(f"Duplicate found by ISBN: {isbn} → {result['key']}")
                return result

        # 3. Title + author fuzzy match
        title = metadata.get("title")
        if title:
            creators = metadata.get("creators", [])
            author_names = []
            for c in creators:
                if isinstance(c, dict):
                    last = c.get("lastName", c.get("last_name", ""))
                    if last:
                        author_names.append(last)
                elif isinstance(c, str):
                    author_names.append(c.split()[-1] if c else "")
            result = self._find_by_title_author(title, author_names, threshold)
            if result:
                logger.info(f"Duplicate found by title+author: {title[:50]} → {result['key']}")
                return result

        return None

    def _find_by_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Exact match on DOI."""
        try:
            items = self._zot.items(q=doi, qmode="everything", limit=10)
            doi_lower = doi.strip().lower()
            for item in items:
                item_doi = item.get("data", {}).get("DOI", "").strip().lower()
                if item_doi == doi_lower:
                    return item
        except Exception as e:
            logger.warning(f"DOI lookup failed: {e}")
        return None

    def _find_by_isbn(self, isbn: str) -> Optional[Dict[str, Any]]:
        """Exact match on ISBN (normalized, no hyphens)."""
        try:
            normalized = isbn.replace("-", "").replace(" ", "")
            items = self._zot.items(q=isbn, qmode="everything", limit=10)
            for item in items:
                item_isbn = (
                    item.get("data", {}).get("ISBN", "").replace("-", "").replace(" ", "")
                )
                if item_isbn == normalized:
                    return item
        except Exception as e:
            logger.warning(f"ISBN lookup failed: {e}")
        return None

    def _find_by_title_author(
        self,
        title: str,
        author_last_names: list[str],
        threshold: float = 0.85,
    ) -> Optional[Dict[str, Any]]:
        """Fuzzy match on title with author overlap check."""
        try:
            # Search by first 5 words to narrow candidates
            query_words = " ".join(title.split()[:5])
            candidates = self._zot.items(q=query_words, qmode="titleCreatorYear", limit=20)

            for item in candidates:
                item_title = item.get("data", {}).get("title", "")
                ratio = SequenceMatcher(None, title.lower(), item_title.lower()).ratio()

                if ratio < threshold:
                    continue

                # Check author overlap
                item_authors = [
                    c.get("lastName", "").lower()
                    for c in item.get("data", {}).get("creators", [])
                    if c.get("creatorType") == "author"
                ]
                query_authors = [a.lower() for a in author_last_names]
                overlap = len(set(item_authors) & set(query_authors))

                if overlap > 0 or not author_last_names:
                    return item
        except Exception as e:
            logger.warning(f"Title+author lookup failed: {e}")
        return None
