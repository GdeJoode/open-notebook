"""OpenAlex work resolver (Track V.4) — the broad-recall fallback leg.

OpenAlex (https://openalex.org) is an open scholarly index with no API key. Two
paths:

* **DOI** — ``/works/https://doi.org/<doi>`` returns the single canonical work;
  a normalized-DOI match is a certain resolution.
* **Title/author** — ``/works?search=<title>`` (broadest recall). Each candidate
  is gated by the shared title-overlap + author-confirmer precision guard, so a
  generic near-miss stays unresolved rather than force-resolving.

OpenAlex payloads carry ORCID (per author) and ROR (per institution) inline, so
this leg needs no separate enrichment pass. All HTTP goes through the shared
fail-soft :class:`VocabularyHTTPClient`.
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

_OPENALEX_WORKS_URL = "https://api.openalex.org/works"


class OpenAlexResolver:
    """Resolve a reference against OpenAlex ``/works`` (DOI or title/author)."""

    name = "openalex"

    def __init__(
        self,
        *,
        contact_email: str = "noreply@open-notebook.local",
        http_client: Optional[VocabularyHTTPClient] = None,
        rows: int = 5,
    ) -> None:
        """Args:

        contact_email: Goes into the polite-pool ``User-Agent`` mailto (OpenAlex,
            like Crossref, routes a contactable client to its faster pool).
        http_client: Inject a client (tests pass one wired to a mock transport).
        rows: Max candidates to fetch on the title/author path.
        """
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
        """Resolve via DOI first (certain), else the guarded title/author path."""
        doi = normalize_doi(ref.doi)
        if doi:
            resolved = await self._resolve_by_doi(doi)
            if resolved is not None:
                return resolved
        return await self._resolve_by_title(ref)

    async def _resolve_by_doi(self, doi: str) -> Optional[ResolvedWork]:
        payload = await self._client.get_json(
            f"{_OPENALEX_WORKS_URL}/https://doi.org/{doi}"
        )
        if not payload or not isinstance(payload, dict):
            return None
        work = self._work_to_resolved(payload, method="doi")
        if work is None:
            return None
        # Confirm the authority returned the DOI we asked for before trusting it.
        if normalize_doi(work.doi) != doi:
            return None
        return work

    async def _resolve_by_title(self, ref: ParsedReference) -> Optional[ResolvedWork]:
        query = (ref.title or "").strip()
        if not query:
            return None
        payload = await self._client.get_json(
            _OPENALEX_WORKS_URL,
            params={"search": query, "per-page": str(self._rows)},
        )
        if not payload or not isinstance(payload, dict):
            return None
        for item in payload.get("results") or []:
            work = self._work_to_resolved(item, method="title_author")
            if work is None:
                continue
            accepted, confidence = passes_title_author_guard(
                ref, work.title, work.authors
            )
            if accepted:
                return ResolvedWork(
                    title=work.title,
                    source=self.name,
                    method="title_author",
                    confidence=confidence,
                    doi=work.doi,
                    authors=work.authors,
                    year=work.year,
                    venue=work.venue,
                    external_id=work.external_id,
                    external_uri=work.external_uri,
                    institutions=work.institutions,
                    author_ids=work.author_ids,
                    institution_ids=work.institution_ids,
                )
        return None

    def _work_to_resolved(
        self, work: Dict[str, Any], *, method: str
    ) -> Optional[ResolvedWork]:
        """Map an OpenAlex work object to a :class:`ResolvedWork`."""
        title = work.get("display_name") or work.get("title") or ""
        if not title:
            return None
        openalex_id = work.get("id") or ""
        authors, author_ids, institutions, institution_ids = self._parse_authorships(
            work.get("authorships") or []
        )
        venue = None
        primary = work.get("primary_location") or {}
        source = primary.get("source") or {}
        if isinstance(source, dict):
            venue = source.get("display_name")
        return ResolvedWork(
            title=title,
            source=self.name,
            method=method,
            confidence=DOI_MATCH_CONFIDENCE if method == "doi" else 0.0,
            doi=normalize_doi(work.get("doi")) or None,
            authors=authors,
            year=work.get("publication_year"),
            venue=venue,
            external_id=openalex_id,
            external_uri=openalex_id,
            institutions=institutions,
            author_ids=author_ids,
            institution_ids=institution_ids,
        )

    @staticmethod
    def _parse_authorships(
        authorships: List[Dict[str, Any]],
    ) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
        authors: List[str] = []
        author_ids: List[str] = []
        institutions: List[str] = []
        institution_ids: List[str] = []
        for entry in authorships:
            author = entry.get("author") or {}
            name = author.get("display_name")
            if name:
                authors.append(name)
            orcid = author.get("orcid")
            if orcid:
                author_ids.append(orcid)
            for inst in entry.get("institutions") or []:
                inst_name = inst.get("display_name")
                if inst_name and inst_name not in institutions:
                    institutions.append(inst_name)
                ror = inst.get("ror")
                if ror and ror not in institution_ids:
                    institution_ids.append(ror)
        return (
            tuple(authors),
            tuple(author_ids),
            tuple(institutions),
            tuple(institution_ids),
        )

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["OpenAlexResolver"]
