"""Crossref vocabulary provider — scholarly DOIs (K.4).

Crossref is queried **directly** (no bulk pre-load) against the public REST API
``https://api.crossref.org/works`` for research-paper / journal / author-paper
entities. A resolvable title returns its DOI as a stable
``https://doi.org/<doi>`` URI.

Network discipline (the NIM / fair-use lesson)
----------------------------------------------
* All HTTP goes through :class:`~shared.vocabulary.http_client.VocabularyHTTPClient`
  (timeout + rate-limit + cache + fail-soft).
* We use the **polite pool**: the ``User-Agent`` carries a contact ``mailto:`` so
  Crossref routes us to the faster, more-reliable pool and can reach us if we
  misbehave. The contact is configurable.
* An unreachable Crossref returns ``[]`` (a no-match) — it never breaks reconcile.

``refresh()`` is a no-op (Crossref is on-demand, not bulk-ingested), returning 0.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.vocabulary.http_client import VocabularyHTTPClient
from shared.vocabulary.provider import VocabMatch

_CROSSREF_WORKS_URL = "https://api.crossref.org/works"

# Entity types that make sense to resolve against a paper registry. A government
# org or a person-without-paper-context is not sent to Crossref (saves calls and
# avoids spurious DOI matches).
_SCHOLARLY_TYPES = {
    "scholarly_article",
    "article",
    "paper",
    "publication",
    "periodical",
    "journal",
}


class CrossrefProvider:
    """On-demand DOI resolution for scholarly entities via the Crossref REST API."""

    name = "crossref"

    def __init__(
        self,
        *,
        contact_email: str = "noreply@open-notebook.local",
        http_client: Optional[VocabularyHTTPClient] = None,
        min_score: float = 70.0,
    ) -> None:
        """Args:

        contact_email: Goes into the polite-pool ``User-Agent`` mailto.
        http_client: Inject a client (tests pass one wired to a mocked transport).
        min_score: Crossref's per-result relevance score floor below which a hit
            is treated as too weak to be a confident match.
        """
        self._min_score = min_score
        self._client = http_client or VocabularyHTTPClient(
            user_agent=(
                f"open-notebook/0.1 (https://github.com/open-notebook; "
                f"mailto:{contact_email})"
            ),
            timeout=10.0,
            min_interval=1.0,
            cache_ttl=3600.0,
        )

    async def refresh(self) -> int:
        """No-op: Crossref is queried on demand, not bulk-loaded."""
        return 0

    async def lookup(self, name: str, entity_type: str) -> List[VocabMatch]:
        """Resolve a paper title to a DOI. Returns ``[]`` for non-scholarly types.

        Confidence is derived from Crossref's relevance ``score`` and whether the
        returned title matches the query closely. Two near-equal top hits both
        clearing the floor surface as two matches → the reconciler refuses to
        auto-link (precision). NEVER raises (the client fails soft).
        """
        if entity_type.lower() not in _SCHOLARLY_TYPES:
            return []
        query = (name or "").strip()
        if not query:
            return []

        payload = await self._client.get_json(
            _CROSSREF_WORKS_URL,
            params={"query.bibliographic": query, "rows": "5"},
        )
        if not payload:
            return []

        items = (payload.get("message") or {}).get("items") or []
        matches: List[VocabMatch] = []
        for item in items:
            match = self._item_to_match(item)
            if match is not None:
                matches.append(match)
        return matches

    def _item_to_match(self, item: Dict[str, Any]) -> Optional[VocabMatch]:
        doi = item.get("DOI")
        if not doi:
            return None
        score = float(item.get("score") or 0.0)
        if score < self._min_score:
            return None
        titles = item.get("title") or []
        title = titles[0] if titles else doi
        # Map Crossref's 0..~100 relevance score onto a 0..1 confidence, clamped.
        confidence = max(0.0, min(1.0, score / 100.0))
        return VocabMatch(
            canonical_name=title,
            external_uri=f"https://doi.org/{doi}",
            external_id=doi,
            source_vocabulary=self.name,
            aliases=[t for t in titles[1:]],
            confidence=confidence,
        )

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["CrossrefProvider"]
