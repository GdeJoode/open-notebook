"""Crossref work resolver (Track V.4).

Resolves a bibliographic reference to a canonical work in Crossref
(``https://api.crossref.org/works``). Distinct from the K.4
``vocabulary.CrossrefProvider`` — that resolves an *entity* surface form to a DOI
for reconciliation; this resolves a *whole reference* (a ``ParsedReference``) to a
:class:`ResolvedWork`. The K.4 provider is left untouched.

Two paths:

* **DOI-exact** — ``/works/<doi>`` returns the single record; a normalized-DOI
  match is certain (confidence 1.0, the gold signal).
* **Title/author** — ``/works?query.bibliographic=<title>&query.author=<authors>``;
  each candidate is gated by the shared title-overlap + author-confirmer guard
  (Crossref's relevance ``score`` is query-length dependent, not a probability, so
  it is not trusted on its own — the same K.4 lesson).

All HTTP goes through the shared fail-soft :class:`VocabularyHTTPClient`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from shared.references.work_resolver import (
    DOI_MATCH_CONFIDENCE,
    ResolvedWork,
    passes_title_author_guard,
)
from shared.retrieval.cites_matching import ParsedReference, normalize_doi
from shared.vocabulary.http_client import VocabularyHTTPClient

_CROSSREF_WORKS_URL = "https://api.crossref.org/works"


class CrossrefResolver:
    """Resolve a reference against Crossref ``/works`` (DOI or title/author)."""

    name = "crossref"

    def __init__(
        self,
        *,
        contact_email: str = "noreply@open-notebook.local",
        http_client: Optional[VocabularyHTTPClient] = None,
        rows: int = 5,
    ) -> None:
        self._rows = rows
        self._client = http_client or VocabularyHTTPClient(
            user_agent=(
                f"open-notebook/0.1 (https://github.com/open-notebook; "
                f"mailto:{contact_email})"
            ),
            timeout=10.0,
            min_interval=1.0,
            cache_ttl=3600.0,
        )

    async def resolve(self, ref: ParsedReference) -> Optional[ResolvedWork]:
        doi = normalize_doi(ref.doi)
        if doi:
            resolved = await self._resolve_by_doi(doi)
            if resolved is not None:
                return resolved
        return await self._resolve_by_title(ref)

    async def _resolve_by_doi(self, doi: str) -> Optional[ResolvedWork]:
        # Crossref's DOI in the path must be URL-safe; the client encodes params
        # but not the path, so a raw DOI (which contains '/') is passed as-is —
        # Crossref accepts the unencoded DOI on this route.
        payload = await self._client.get_json(f"{_CROSSREF_WORKS_URL}/{doi}")
        if not payload or not isinstance(payload, dict):
            return None
        message = payload.get("message")
        if not isinstance(message, dict):
            return None
        work = self._message_to_resolved(message, method="doi", confidence=DOI_MATCH_CONFIDENCE)
        if work is None or normalize_doi(work.doi) != doi:
            return None
        return work

    async def _resolve_by_title(self, ref: ParsedReference) -> Optional[ResolvedWork]:
        query = (ref.title or "").strip()
        if not query:
            return None
        params: Dict[str, str] = {
            "query.bibliographic": query,
            "rows": str(self._rows),
        }
        if ref.authors:
            params["query.author"] = " ".join(ref.authors)
        payload = await self._client.get_json(_CROSSREF_WORKS_URL, params=params)
        if not payload or not isinstance(payload, dict):
            return None
        items = (payload.get("message") or {}).get("items") or []
        for item in items:
            candidate = self._message_to_resolved(
                item, method="title_author", confidence=0.0
            )
            if candidate is None:
                continue
            accepted, confidence = passes_title_author_guard(
                ref, candidate.title, candidate.authors
            )
            if accepted:
                return ResolvedWork(
                    title=candidate.title,
                    source=self.name,
                    method="title_author",
                    confidence=confidence,
                    doi=candidate.doi,
                    authors=candidate.authors,
                    year=candidate.year,
                    venue=candidate.venue,
                    external_id=candidate.external_id,
                    external_uri=candidate.external_uri,
                )
        return None

    def _message_to_resolved(
        self, item: Dict[str, Any], *, method: str, confidence: float
    ) -> Optional[ResolvedWork]:
        doi = normalize_doi(item.get("DOI"))
        titles = item.get("title") or []
        title = titles[0] if titles else ""
        if not title:
            return None
        authors = self._parse_authors(item.get("author") or [])
        containers = item.get("container-title") or []
        venue = containers[0] if containers else item.get("publisher")
        return ResolvedWork(
            title=title,
            source=self.name,
            method=method,
            confidence=confidence,
            doi=doi or None,
            authors=authors,
            year=self._parse_year(item),
            venue=venue,
            external_id=doi or "",
            external_uri=f"https://doi.org/{doi}" if doi else "",
        )

    @staticmethod
    def _parse_authors(authors: List[Dict[str, Any]]) -> Tuple[str, ...]:
        names: List[str] = []
        for a in authors:
            family = (a.get("family") or "").strip()
            given = (a.get("given") or "").strip()
            if family and given:
                names.append(f"{given} {family}")
            elif family:
                names.append(family)
            elif a.get("name"):
                names.append(a["name"])
        return tuple(names)

    @staticmethod
    def _parse_year(item: Dict[str, Any]) -> Optional[int]:
        # Crossref dates are ``{"date-parts": [[year, month, day]]}``. Prefer
        # ``issued`` (publication), fall back to ``published``.
        for key in ("issued", "published", "published-print", "published-online"):
            block = item.get(key)
            if isinstance(block, dict):
                parts = block.get("date-parts") or []
                if parts and parts[0]:
                    try:
                        return int(parts[0][0])
                    except (TypeError, ValueError):
                        continue
        return None

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["CrossrefResolver"]
