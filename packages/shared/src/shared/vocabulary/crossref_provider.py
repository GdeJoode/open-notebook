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

from shared.utils.name_normalizer import normalize_entity_name
from shared.vocabulary.http_client import VocabularyHTTPClient
from shared.vocabulary.provider import VocabMatch

_CROSSREF_WORKS_URL = "https://api.crossref.org/works"

# Minimum token-overlap (Jaccard) between the query and the returned title
# before a Crossref hit may be treated as a confident match. Crossref's
# relevance ``score`` is a query-length-dependent BM25-style number, not a
# probability — a short/generic query can score >0.85 against an unrelated
# DOI. Requiring real lexical overlap on the normalized titles is a cheap
# precision gate that protects the single-candidate auto-link (the recurring
# Track-K over-link concern). 0.5 ⇒ at least half the union of tokens shared.
_MIN_TITLE_OVERLAP = 0.5

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


def _title_overlap(a: str, b: str) -> float:
    """Token-Jaccard overlap of two titles after normalization.

    Both sides run through :func:`normalize_entity_name` (lowercase, whitespace
    collapse, trailing-punct strip) before tokenizing on whitespace. Returns the
    size of the token intersection over the union (0.0 when either side has no
    tokens). Cheap and dependency-free — enough to reject a wholly-unrelated
    title without trying to be a real string-similarity metric.
    """
    a_tokens = set(normalize_entity_name(a).split())
    b_tokens = set(normalize_entity_name(b).split())
    if not a_tokens or not b_tokens:
        return 0.0
    union = a_tokens | b_tokens
    return len(a_tokens & b_tokens) / len(union)


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

        Confidence is derived from Crossref's relevance ``score`` AND a title-match
        gate: a hit whose returned title does not lexically overlap the query (by
        at least ``_MIN_TITLE_OVERLAP`` token Jaccard on the normalized forms) is
        rejected, because Crossref's score is query-length-dependent and a short
        query can score highly against an unrelated DOI. Two near-equal top hits
        both clearing score + overlap surface as two matches → the reconciler
        refuses to auto-link (precision). NEVER raises (the client fails soft).
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
            match = self._item_to_match(item, query)
            if match is not None:
                matches.append(match)
        return matches

    def _item_to_match(
        self, item: Dict[str, Any], query: str
    ) -> Optional[VocabMatch]:
        doi = item.get("DOI")
        if not doi:
            return None
        score = float(item.get("score") or 0.0)
        if score < self._min_score:
            return None
        titles = item.get("title") or []
        title = titles[0] if titles else doi
        # Title-match safeguard: reject hits whose title does not lexically
        # overlap the query. Guards against a high relevance score on an
        # unrelated DOI (Crossref score is not a probability).
        if titles and _title_overlap(query, title) < _MIN_TITLE_OVERLAP:
            logger.debug(
                "crossref: dropped DOI {} — title overlap below floor "
                "(query={!r} title={!r})",
                doi,
                query,
                title,
            )
            return None
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
