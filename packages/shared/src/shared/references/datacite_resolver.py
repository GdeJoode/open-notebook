"""DataCite work resolver (Track V.4).

DataCite (``https://api.datacite.org/dois``) registers DOIs for datasets, theses,
preprints and software — the works Crossref does not cover. This leg is
**DOI-only**: a reference with a DOI that Crossref could not resolve falls through
to DataCite, where a normalized-DOI match is certain (confidence 1.0). No
title/author path — DataCite's free-text search is noisy and the DOI is the only
signal precise enough to trust here.

All HTTP goes through the shared fail-soft :class:`VocabularyHTTPClient`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from shared.references.work_resolver import DOI_MATCH_CONFIDENCE, ResolvedWork
from shared.retrieval.cites_matching import ParsedReference, normalize_doi
from shared.vocabulary.http_client import VocabularyHTTPClient

_DATACITE_DOIS_URL = "https://api.datacite.org/dois"


class DataCiteResolver:
    """Resolve a DOI-bearing reference against DataCite ``/dois/<doi>``."""

    name = "datacite"

    def __init__(
        self,
        *,
        http_client: Optional[VocabularyHTTPClient] = None,
    ) -> None:
        self._client = http_client or VocabularyHTTPClient(
            user_agent="open-notebook/0.1 (https://github.com/open-notebook)",
            timeout=10.0,
            min_interval=1.0,
            cache_ttl=3600.0,
        )

    async def resolve(self, ref: ParsedReference) -> Optional[ResolvedWork]:
        doi = normalize_doi(ref.doi)
        if not doi:
            return None
        payload = await self._client.get_json(f"{_DATACITE_DOIS_URL}/{doi}")
        if not payload or not isinstance(payload, dict):
            return None
        data = payload.get("data")
        if not isinstance(data, dict):
            return None
        attributes = data.get("attributes")
        if not isinstance(attributes, dict):
            return None
        resolved = self._attributes_to_resolved(attributes, data.get("id"))
        if resolved is None or normalize_doi(resolved.doi) != doi:
            return None
        return resolved

    def _attributes_to_resolved(
        self, attributes: Dict[str, Any], record_id: Optional[str]
    ) -> Optional[ResolvedWork]:
        """Map a DataCite ``data.attributes`` block to a :class:`ResolvedWork`."""
        doi = normalize_doi(attributes.get("doi") or record_id)
        title = self._first_title(attributes.get("titles") or [])
        if not title:
            return None
        authors = self._parse_creators(attributes.get("creators") or [])
        year = attributes.get("publicationYear")
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = None
        venue = attributes.get("publisher")
        # DataCite `publisher` may be a bare string or, in schema 4.5+, an object.
        if isinstance(venue, dict):
            venue = venue.get("name")
        return ResolvedWork(
            title=title,
            source=self.name,
            method="doi",
            confidence=DOI_MATCH_CONFIDENCE,
            doi=doi or None,
            authors=authors,
            year=year,
            venue=venue if isinstance(venue, str) else None,
            external_id=doi or "",
            external_uri=f"https://doi.org/{doi}" if doi else "",
        )

    @staticmethod
    def _first_title(titles: List[Dict[str, Any]]) -> str:
        for entry in titles:
            title = (entry.get("title") or "").strip()
            if title:
                return title
        return ""

    @staticmethod
    def _parse_creators(creators: List[Dict[str, Any]]) -> Tuple[str, ...]:
        names: List[str] = []
        for c in creators:
            name = (c.get("name") or "").strip()
            if not name:
                given = (c.get("givenName") or "").strip()
                family = (c.get("familyName") or "").strip()
                name = f"{given} {family}".strip()
            if name:
                names.append(name)
        return tuple(names)

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["DataCiteResolver"]
