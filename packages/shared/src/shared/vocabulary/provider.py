"""The ``VocabularyProvider`` protocol + the ``VocabMatch`` result type (K.4).

A provider is the thin adapter between an external authority (TOOI, Crossref,
later ORCID/arXiv/Wikidata) and the ``reference_entity`` table. Every provider:

* ``name`` — a stable short id, also used as ``source_vocabulary`` on rows.
* ``lookup(name, entity_type) -> list[VocabMatch]`` — resolve a surface form to
  zero-or-more candidate authority records. Returning **more than one** match is
  the signal the reconciler uses to *refuse* an auto-link (precision guard).
* ``refresh() -> int`` — (re-)sync the authority into ``reference_entity`` and
  return the number of rows upserted. For an on-demand authority (Crossref) this
  is a no-op returning 0; for a bulk authority (TOOI) it loads the dataset.

``VocabMatch`` is a plain dataclass (no DB coupling) so providers stay testable
without a database — the reconciler/repository turn matches into rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Protocol, runtime_checkable


@dataclass(frozen=True)
class VocabMatch:
    """One candidate authority record for a surface form.

    ``confidence`` is the provider's own estimate (exact-name hit = high; a
    fuzzy/secondary hit = lower). The reconciler only auto-links a SINGLE match
    whose confidence clears its threshold; two high-confidence matches are
    ambiguous and recorded as candidates, never auto-linked.
    """

    canonical_name: str
    external_uri: str
    external_id: str
    source_vocabulary: str
    aliases: List[str] = field(default_factory=list)
    confidence: float = 1.0


@runtime_checkable
class VocabularyProvider(Protocol):
    """Adapter protocol for an external vocabulary authority."""

    @property
    def name(self) -> str:
        """Stable short identifier (also the ``source_vocabulary`` value)."""
        ...

    async def lookup(self, name: str, entity_type: str) -> List[VocabMatch]:
        """Resolve a surface form to candidate authority records.

        Must NOT raise on a network/authority failure — a provider that cannot
        reach its backend logs and returns ``[]`` (a no-match), so reconcile/
        ingest never breaks because an authority is down.
        """
        ...

    async def refresh(self) -> int:
        """(Re-)sync the authority into ``reference_entity``; return rows upserted.

        Idempotent: a second refresh over unchanged data upserts the same keys
        and creates no duplicate rows.
        """
        ...


__all__ = ["VocabMatch", "VocabularyProvider"]
