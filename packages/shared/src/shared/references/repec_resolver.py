"""RePEc / CitEc work resolver (Track V.4) — CONFIG-GATED, never blocks.

RePEc (Research Papers in Economics) via the CitEc citation API is the authority
for economics working papers. The CitEc API requires an **access code emailed on
request** — which the user will obtain LATER. Until then this resolver is inert
by design:

* The access code is read from the ``REPEC_API_KEY`` environment variable (a
  constructor arg overrides it, for tests).
* When the code is unset/empty the resolver reports ``available == False``; the
  :class:`~shared.references.work_resolver.ResolverCascade` skips it SILENTLY.
* :meth:`resolve` on an unconfigured resolver returns ``None`` immediately — it
  NEVER raises, warns loudly, or blocks the cascade.

This is the ONLY intentionally-inert leg, and its inertness is a clean, tested
skip — not a stub. When a code is configured the resolver performs a real,
guarded title/author lookup.

# TODO(V.4-live): confirm the CitEc request contract (endpoint path, the access-
# code parameter name) and the JSON response shape once the emailed code arrives;
# the field mapping below follows the documented CitEc search response.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

from shared.references.work_resolver import (
    ResolvedWork,
    passes_title_author_guard,
)
from shared.retrieval.cites_matching import ParsedReference
from shared.vocabulary.http_client import VocabularyHTTPClient

_CITEC_SEARCH_URL = "https://citec.repec.org/api/search"
_REPEC_ENV_VAR = "REPEC_API_KEY"


class RePEcResolver:
    """Resolve an econ working-paper reference against CitEc — if configured."""

    name = "repec"

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        http_client: Optional[VocabularyHTTPClient] = None,
        rows: int = 5,
    ) -> None:
        """Args:

        api_key: The CitEc access code. When ``None`` it is read from the
            ``REPEC_API_KEY`` env var; an empty/unset value leaves the resolver
            unavailable (the silent-skip seam).
        http_client: Inject a client (tests pass a mock transport).
        rows: Max candidates to fetch.
        """
        self._api_key = (api_key if api_key is not None else os.getenv(_REPEC_ENV_VAR, "")) or ""
        self._api_key = self._api_key.strip()
        self._rows = rows
        # Only build a client when configured — an unconfigured resolver makes no
        # network calls at all.
        self._client = http_client
        if self._api_key and self._client is None:
            self._client = VocabularyHTTPClient(
                user_agent="open-notebook/0.1 (https://github.com/open-notebook)",
                timeout=10.0,
                min_interval=1.0,
                cache_ttl=3600.0,
            )

    @property
    def available(self) -> bool:
        """True only when an access code is configured (the cascade gate)."""
        return bool(self._api_key)

    async def resolve(self, ref: ParsedReference) -> Optional[ResolvedWork]:
        # Silent no-op when unconfigured: never raises, never warns, never blocks.
        if not self.available or self._client is None:
            return None
        query = (ref.title or "").strip()
        if not query:
            return None
        payload = await self._client.get_json(
            _CITEC_SEARCH_URL,
            params={"code": self._api_key, "q": query, "rows": str(self._rows)},
        )
        if not payload:
            return None
        for item in self._iter_items(payload):
            candidate = self._item_to_resolved(item)
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

    @staticmethod
    def _iter_items(payload: Any) -> List[Dict[str, Any]]:
        # CitEc search returns either a bare list or ``{"results": [...]}``.
        if isinstance(payload, list):
            return [i for i in payload if isinstance(i, dict)]
        if isinstance(payload, dict):
            results = payload.get("results") or payload.get("items") or []
            return [i for i in results if isinstance(i, dict)]
        return []

    def _item_to_resolved(self, item: Dict[str, Any]) -> Optional[ResolvedWork]:
        title = (item.get("title") or "").strip()
        if not title:
            return None
        handle = item.get("handle") or item.get("id") or ""
        authors = self._parse_authors(item.get("authors") or [])
        year = item.get("year")
        try:
            year = int(year) if year is not None else None
        except (TypeError, ValueError):
            year = None
        # A RePEc handle dereferences via the EconPapers redirect.
        uri = f"https://econpapers.repec.org/{handle}" if handle else ""
        return ResolvedWork(
            title=title,
            source=self.name,
            method="title_author",
            confidence=0.0,
            authors=authors,
            year=year,
            venue=item.get("series") or item.get("venue"),
            external_id=handle,
            external_uri=uri,
        )

    @staticmethod
    def _parse_authors(authors: List[Any]) -> Tuple[str, ...]:
        names: List[str] = []
        for a in authors:
            if isinstance(a, str) and a.strip():
                names.append(a.strip())
            elif isinstance(a, dict):
                name = (a.get("name") or "").strip()
                if name:
                    names.append(name)
        return tuple(names)

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()


__all__ = ["RePEcResolver"]
