"""External-vocabulary providers for entity reconciliation (Phase K.4).

The vocabulary layer maps a persisted ``entity`` to stable cross-system
identifiers held in external authorities:

* **TOOI** — the Dutch government thesaurus (organisations / administrative
  areas) → ``https://identifier.overheid.nl/tooi/...`` URIs.
* **Crossref** — the scholarly DOI registry (papers / journals) → DOIs.

Both implement the :class:`~shared.vocabulary.provider.VocabularyProvider`
protocol so the reconciler (and future ORCID / arXiv / Wikidata providers) are
pluggable. Lookups go through a single shared HTTP client
(:mod:`shared.vocabulary.http_client`) that adds caching, rate-limiting,
timeouts and graceful failure so an unreachable authority degrades to a
no-match instead of breaking ingest/reconcile.
"""

from shared.vocabulary.provider import VocabMatch, VocabularyProvider

__all__ = ["VocabMatch", "VocabularyProvider"]
