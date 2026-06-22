"""Vocabulary reconciliation — link entities to external authorities (K.4).

``reconcile_entity`` runs the registered vocabulary providers (TOOI, Crossref,
later ORCID/arXiv/Wikidata) over one persisted ``entity`` and, IFF exactly one
high-confidence match exists, writes that authority's stable URI onto
``entity.external_ids`` and folds its canonical form + aliases into
``entity.aliases``.

THE PRECISION GUARD (the K-style over-merge backstop)
-----------------------------------------------------
Over-linking an entity into the WRONG vocabulary entry is a silent data error
the user cannot easily detect — worse than leaving it unlinked. So the rule is
strict:

* collect candidate matches from every provider whose confidence clears
  ``confidence_threshold``;
* **auto-link ONLY when there is exactly ONE such candidate.** Two (or more)
  qualifying candidates = ambiguous → record them as candidates (logged on the
  result), write NOTHING. A single match below threshold also does not link.

The reconciler never *merges* two persisted entities — it only annotates one
entity with external identifiers. That keeps it orthogonal to K.3's entity merge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Sequence

from loguru import logger

from shared.models.entity import Entity
from shared.vocabulary.provider import VocabMatch, VocabularyProvider


@dataclass
class ReconcileResult:
    """Outcome of reconciling one entity.

    ``linked`` is True only when a single high-confidence match was applied.
    ``candidates`` carries the qualifying matches when the link was REFUSED
    because more than one cleared the threshold (the ambiguous case), so a
    caller / UI can surface them for human review without auto-linking.
    """

    entity_id: Optional[str]
    linked: bool
    external_ids: List[str] = field(default_factory=list)
    added_aliases: List[str] = field(default_factory=list)
    candidates: List[VocabMatch] = field(default_factory=list)
    reason: str = ""


class VocabularyReconciler:
    """Reconcile a persisted entity against external vocabulary providers."""

    def __init__(
        self,
        providers: Sequence[VocabularyProvider],
        *,
        confidence_threshold: float = 0.85,
        entity_repo: Optional[Any] = None,
    ) -> None:
        """Args:

        providers: Ordered vocabulary providers to consult.
        confidence_threshold: A match must clear this to qualify as a link
            candidate. The single-candidate rule applies on top of this.
        entity_repo: A repository exposing ``async update_external_ids(id, uris,
            aliases)``-style persistence. Optional — when ``None`` the reconciler
            only mutates the in-memory ``Entity`` (and returns the result), which
            is what the unit tests assert; a batch caller supplies a repo to
            persist.
        """
        self._providers = list(providers)
        self._threshold = confidence_threshold
        self._repo = entity_repo

    async def reconcile_entity(self, entity: Entity) -> ReconcileResult:
        """Reconcile one entity. Mutates ``entity`` in place on a single match.

        Returns a :class:`ReconcileResult`. NEVER raises on a provider failure —
        each provider is fail-soft, so a down authority simply contributes no
        candidates.
        """
        entity_id = getattr(entity, "id", None)
        candidates = await self._collect_candidates(entity)

        qualifying = [m for m in candidates if m.confidence >= self._threshold]

        if not qualifying:
            return ReconcileResult(
                entity_id=entity_id,
                linked=False,
                candidates=[],
                reason="no_match_above_threshold",
            )

        # PRECISION GUARD: a single qualifying candidate is required to auto-link.
        # Distinct external URIs count as distinct candidates; two providers that
        # agree on the SAME URI are not ambiguous (collapse before counting).
        distinct = self._distinct_by_uri(qualifying)
        if len(distinct) > 1:
            logger.info(
                "reconcile_entity: {n} ambiguous candidates for {name!r} — "
                "NO auto-link (precision guard)",
                n=len(distinct),
                name=entity.canonical_name,
            )
            return ReconcileResult(
                entity_id=entity_id,
                linked=False,
                candidates=distinct,
                reason="ambiguous_multiple_candidates",
            )

        match = distinct[0]
        added_aliases = self._apply_match(entity, match)
        if self._repo is not None and entity_id is not None:
            await self._persist(entity_id, entity.external_ids, added_aliases)

        logger.info(
            "reconcile_entity: linked {name!r} → {uri} ({src})",
            name=entity.canonical_name,
            uri=match.external_uri,
            src=match.source_vocabulary,
        )
        return ReconcileResult(
            entity_id=entity_id,
            linked=True,
            external_ids=[match.external_uri],
            added_aliases=added_aliases,
            candidates=[match],
            reason="single_high_confidence_match",
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _collect_candidates(self, entity: Entity) -> List[VocabMatch]:
        matches: List[VocabMatch] = []
        for provider in self._providers:
            try:
                found = await provider.lookup(
                    entity.canonical_name, entity.entity_type
                )
            except Exception as exc:  # belt-and-braces; providers are fail-soft
                logger.warning(
                    "provider {p} raised during lookup (ignored): {e}",
                    p=getattr(provider, "name", provider),
                    e=exc,
                )
                continue
            matches.extend(found)
        return matches

    @staticmethod
    def _distinct_by_uri(matches: List[VocabMatch]) -> List[VocabMatch]:
        """Collapse matches that share an ``external_uri`` (keep highest conf)."""
        best: dict[str, VocabMatch] = {}
        for m in matches:
            existing = best.get(m.external_uri)
            if existing is None or m.confidence > existing.confidence:
                best[m.external_uri] = m
        return list(best.values())

    @staticmethod
    def _apply_match(entity: Entity, match: VocabMatch) -> List[str]:
        """Write the URI + aliases onto the entity (dedup-preserving)."""
        if match.external_uri not in entity.external_ids:
            entity.external_ids.append(match.external_uri)

        added: List[str] = []
        candidate_aliases = [match.canonical_name, *match.aliases]
        for alias in candidate_aliases:
            if alias and alias not in entity.aliases:
                entity.aliases.append(alias)
                added.append(alias)
        return added

    async def _persist(
        self, entity_id: str, external_ids: List[str], aliases: List[str]
    ) -> None:
        try:
            await self._repo.update_external_ids(  # type: ignore[union-attr]
                entity_id, external_ids, aliases
            )
        except Exception as exc:
            logger.error(
                "reconcile_entity: persistence failed for {id}: {e}",
                id=entity_id,
                e=exc,
            )


__all__ = ["VocabularyReconciler", "ReconcileResult"]
