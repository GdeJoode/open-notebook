"""arXiv work resolver (Track V.4).

Routes when the reference carries an arXiv id (detected upstream by
:func:`shared.references.work_resolver.extract_arxiv_id`). Queries the arXiv Atom
API (``http://export.arxiv.org/api/query?id_list=<id>``), which returns **Atom
XML** — fetched through the shared fail-soft client's :meth:`get_text` and parsed
with the stdlib XML parser. An id-exact hit is certain (confidence 1.0), so no
title/author guard is needed: the id IS the precision signal.

# TODO(V.4-live): confirm the Atom element/namespace mapping (title, author name,
# published year, arxiv:doi) against a real arXiv response during the live smoke.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from xml.etree import ElementTree as ET

from loguru import logger

from shared.references.work_resolver import (
    DOI_MATCH_CONFIDENCE,
    ResolvedWork,
    extract_arxiv_id,
)
from shared.retrieval.cites_matching import ParsedReference, normalize_doi
from shared.vocabulary.http_client import VocabularyHTTPClient

_ARXIV_API_URL = "http://export.arxiv.org/api/query"
_ATOM_NS = "http://www.w3.org/2005/Atom"
_ARXIV_NS = "http://arxiv.org/schemas/atom"


class ArxivResolver:
    """Resolve an arXiv-id reference against the arXiv Atom API."""

    name = "arxiv"

    def __init__(
        self,
        *,
        http_client: Optional[VocabularyHTTPClient] = None,
    ) -> None:
        self._client = http_client or VocabularyHTTPClient(
            user_agent="open-notebook/0.1 (https://github.com/open-notebook)",
            timeout=10.0,
            min_interval=3.0,  # arXiv asks for ~1 request / 3s
            cache_ttl=3600.0,
        )

    async def resolve(self, ref: ParsedReference) -> Optional[ResolvedWork]:
        arxiv_id = extract_arxiv_id(ref)
        if not arxiv_id:
            return None
        body = await self._client.get_text(
            _ARXIV_API_URL, params={"id_list": arxiv_id, "max_results": "1"}
        )
        if not body:
            return None
        return self._parse_feed(body, arxiv_id)

    def _parse_feed(self, body: str, arxiv_id: str) -> Optional[ResolvedWork]:
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            logger.warning("arxiv: could not parse Atom feed: {exc}", exc=exc)
            return None
        entry = root.find(f"{{{_ATOM_NS}}}entry")
        if entry is None:
            return None

        entry_id = self._text(entry.find(f"{{{_ATOM_NS}}}id"))
        # arXiv answers an unknown id with an error pseudo-entry — never a match.
        if not entry_id or "api/errors" in entry_id:
            return None
        # Confirm the returned id is the one we asked for (guards a redirect/typo).
        if arxiv_id.lower() not in entry_id.lower():
            return None

        title = self._clean(self._text(entry.find(f"{{{_ATOM_NS}}}title")))
        if not title:
            return None
        authors = self._parse_authors(entry)
        year = self._parse_year(self._text(entry.find(f"{{{_ATOM_NS}}}published")))
        doi = normalize_doi(self._text(entry.find(f"{{{_ARXIV_NS}}}doi")))
        venue = self._clean(self._text(entry.find(f"{{{_ARXIV_NS}}}journal_ref")))

        return ResolvedWork(
            title=title,
            source=self.name,
            method="arxiv_id",
            confidence=DOI_MATCH_CONFIDENCE,
            doi=doi or None,
            authors=authors,
            year=year,
            venue=venue or "arXiv",
            external_id=f"arXiv:{arxiv_id}",
            external_uri=entry_id,
        )

    def _parse_authors(self, entry: ET.Element) -> Tuple[str, ...]:
        names: List[str] = []
        for author in entry.findall(f"{{{_ATOM_NS}}}author"):
            name = self._clean(self._text(author.find(f"{{{_ATOM_NS}}}name")))
            if name:
                names.append(name)
        return tuple(names)

    @staticmethod
    def _parse_year(published: str) -> Optional[int]:
        # arXiv `published` is ISO-8601, e.g. "2023-01-03T18:00:00Z".
        if published and len(published) >= 4 and published[:4].isdigit():
            return int(published[:4])
        return None

    @staticmethod
    def _text(element: Optional[ET.Element]) -> str:
        return element.text or "" if element is not None else ""

    @staticmethod
    def _clean(text: str) -> str:
        # Atom titles/journal_refs often carry embedded newlines + runs of spaces.
        return " ".join(text.split()).strip()

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["ArxivResolver"]
