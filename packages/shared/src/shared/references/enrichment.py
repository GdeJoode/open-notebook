"""Opt-in ID enrichment for a resolved work (Track V.4) — ORCID / ROR.

A post-match, best-effort step that attaches author ORCID iDs and institution ROR
ids to an already-resolved :class:`ResolvedWork`. It is NEVER required: the
cascade returns a fully-usable work without it, and every lookup fails soft (a
down authority leaves the ids empty). OpenAlex already carries ORCID/ROR inline,
so enrichment mainly benefits the Crossref/DataCite/arXiv legs.

Precision: an ORCID is only attached when the returned record's family name
matches the author (a wrong ORCID is a false fact); a ROR is only attached when
ROR's affiliation matcher flags the candidate as ``chosen`` (its own confidence
gate).

# TODO(V.4-live): the ORCID public API needs an ``Accept: application/json``
# header to return JSON; confirm the shared HTTP client sends it (or add it) so
# the live ORCID leg does not degrade to XML → no-match.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shared.references.work_resolver import ResolvedWork
from shared.utils.name_normalizer import normalize_entity_name
from shared.vocabulary.http_client import VocabularyHTTPClient

_ORCID_SEARCH_URL = "https://pub.orcid.org/v3.0/expanded-search"
_ROR_URL = "https://api.ror.org/organizations"


class ReferenceEnricher:
    """Attach ORCID / ROR ids to a resolved work (opt-in, best-effort)."""

    def __init__(
        self,
        *,
        orcid_client: Optional[VocabularyHTTPClient] = None,
        ror_client: Optional[VocabularyHTTPClient] = None,
        enable_orcid: bool = True,
        enable_ror: bool = True,
    ) -> None:
        self._enable_orcid = enable_orcid
        self._enable_ror = enable_ror
        self._orcid = orcid_client or (
            VocabularyHTTPClient(
                user_agent="open-notebook/0.1 (https://github.com/open-notebook)",
                timeout=10.0,
                min_interval=1.0,
                cache_ttl=3600.0,
            )
            if enable_orcid
            else None
        )
        self._ror = ror_client or (
            VocabularyHTTPClient(
                user_agent="open-notebook/0.1 (https://github.com/open-notebook)",
                timeout=10.0,
                min_interval=1.0,
                cache_ttl=3600.0,
            )
            if enable_ror
            else None
        )

    async def enrich(self, work: ResolvedWork) -> ResolvedWork:
        """Return ``work`` augmented with any newly-found author/institution ids."""
        author_ids = work.author_ids
        institution_ids = work.institution_ids

        if self._enable_orcid and not author_ids and work.authors and self._orcid:
            found = await self._lookup_orcids(work.authors)
            if found:
                author_ids = found

        if (
            self._enable_ror
            and not institution_ids
            and work.institutions
            and self._ror
        ):
            found = await self._lookup_rors(work.institutions)
            if found:
                institution_ids = found

        if author_ids == work.author_ids and institution_ids == work.institution_ids:
            return work
        return replace(
            work, author_ids=author_ids, institution_ids=institution_ids
        )

    async def _lookup_orcids(self, authors: Sequence[str]) -> Tuple[str, ...]:
        ids: List[str] = []
        for author in authors:
            orcid = await self._lookup_one_orcid(author)
            if orcid:
                ids.append(orcid)
        return tuple(ids)

    async def _lookup_one_orcid(self, author: str) -> Optional[str]:
        if not author.strip() or self._orcid is None:
            return None
        payload = await self._orcid.get_json(
            _ORCID_SEARCH_URL, params={"q": author}
        )
        if not payload or not isinstance(payload, dict):
            return None
        results = payload.get("expanded-result") or []
        author_norm = normalize_entity_name(author)
        for result in results:
            orcid_id = result.get("orcid-id")
            family = (result.get("family-names") or "").strip()
            # Precision: only trust a hit whose family name is in the author string
            # (guards against attaching an unrelated ORCID on a common name).
            if orcid_id and family and normalize_entity_name(family) in author_norm:
                return str(orcid_id)
        return None

    async def _lookup_rors(self, institutions: Sequence[str]) -> Tuple[str, ...]:
        ids: List[str] = []
        for name in institutions:
            ror = await self._lookup_one_ror(name)
            if ror:
                ids.append(ror)
        return tuple(ids)

    async def _lookup_one_ror(self, name: str) -> Optional[str]:
        if not name.strip() or self._ror is None:
            return None
        payload = await self._ror.get_json(_ROR_URL, params={"affiliation": name})
        if not payload or not isinstance(payload, dict):
            return None
        for item in payload.get("items") or []:
            # ROR's affiliation matcher sets `chosen: true` only when confident.
            if item.get("chosen"):
                org = item.get("organization") or {}
                ror_id = org.get("id")
                if ror_id:
                    return str(ror_id)
        return None

    async def aclose(self) -> None:
        for client in (self._orcid, self._ror):
            if client is not None:
                await client.aclose()


__all__ = ["ReferenceEnricher"]
