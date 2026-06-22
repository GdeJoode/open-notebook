"""Unit tests for the K.4 vocabulary reconciler.

The reconciler is tested with fake in-memory providers (no DB, no network) so
the precision guard — the central over-merge backstop — is exercised directly:

* single high-confidence match → auto-link (external_ids + aliases populated);
* two qualifying matches → NO auto-link, candidates recorded (AC3);
* below-threshold / no match → no link.
"""

from __future__ import annotations

from typing import List

import pytest
from app_main.services.entity_resolution.vocabulary_reconciler import (
    VocabularyReconciler,
)
from shared.models.entity import Entity
from shared.vocabulary.provider import VocabMatch


class FakeProvider:
    """A provider that returns a fixed list of matches for any lookup."""

    def __init__(self, name: str, matches: List[VocabMatch]) -> None:
        self.name = name
        self._matches = matches

    async def lookup(self, name: str, entity_type: str) -> List[VocabMatch]:
        return list(self._matches)

    async def refresh(self) -> int:
        return 0


def _bzk_match(confidence: float = 0.99) -> VocabMatch:
    return VocabMatch(
        canonical_name="Binnenlandse Zaken en Koninkrijksrelaties",
        external_uri="https://identifier.overheid.nl/tooi/id/ministerie/mnre1034",
        external_id="mnre1034",
        source_vocabulary="tooi",
        aliases=["BZK", "ministerie van Binnenlandse Zaken en Koninkrijksrelaties"],
        confidence=confidence,
    )


@pytest.mark.asyncio
async def test_single_match_auto_links():
    """AC2: a BZK org entity gets the TOOI URI + aliases on a single match."""
    entity = Entity(
        canonical_name="Binnenlandse Zaken en Koninkrijksrelaties",
        entity_type="organization",
    )
    reconciler = VocabularyReconciler([FakeProvider("tooi", [_bzk_match()])])

    result = await reconciler.reconcile_entity(entity)

    assert result.linked is True
    assert entity.external_ids == [
        "https://identifier.overheid.nl/tooi/id/ministerie/mnre1034"
    ]
    assert "BZK" in entity.aliases
    assert "Binnenlandse Zaken en Koninkrijksrelaties" in entity.aliases
    assert result.reason == "single_high_confidence_match"


@pytest.mark.asyncio
async def test_two_equal_candidates_do_not_auto_link():
    """AC3 (precision): two equally-confident matches → NO link, candidates kept."""
    m1 = VocabMatch(
        canonical_name="Groningen (provincie)",
        external_uri="https://example/tooi/prov/groningen",
        external_id="prov-gr",
        source_vocabulary="tooi",
        confidence=0.95,
    )
    m2 = VocabMatch(
        canonical_name="Groningen (gemeente)",
        external_uri="https://example/tooi/gem/groningen",
        external_id="gem-gr",
        source_vocabulary="tooi",
        confidence=0.95,
    )
    entity = Entity(canonical_name="Groningen", entity_type="organization")
    reconciler = VocabularyReconciler([FakeProvider("tooi", [m1, m2])])

    result = await reconciler.reconcile_entity(entity)

    assert result.linked is False
    assert entity.external_ids == []  # nothing written
    assert entity.aliases == []
    assert len(result.candidates) == 2
    assert result.reason == "ambiguous_multiple_candidates"


@pytest.mark.asyncio
async def test_same_uri_from_two_providers_is_not_ambiguous():
    """Two providers agreeing on the SAME URI = one candidate → still links."""
    match = _bzk_match()
    entity = Entity(
        canonical_name="Binnenlandse Zaken en Koninkrijksrelaties",
        entity_type="organization",
    )
    reconciler = VocabularyReconciler(
        [FakeProvider("tooi", [match]), FakeProvider("mirror", [match])]
    )

    result = await reconciler.reconcile_entity(entity)
    assert result.linked is True
    assert entity.external_ids == [match.external_uri]


@pytest.mark.asyncio
async def test_below_threshold_does_not_link():
    entity = Entity(canonical_name="BZK", entity_type="organization")
    reconciler = VocabularyReconciler(
        [FakeProvider("tooi", [_bzk_match(confidence=0.50)])],
        confidence_threshold=0.85,
    )
    result = await reconciler.reconcile_entity(entity)
    assert result.linked is False
    assert entity.external_ids == []
    assert result.reason == "no_match_above_threshold"


@pytest.mark.asyncio
async def test_no_match_returns_unlinked():
    entity = Entity(canonical_name="Totally Unknown", entity_type="organization")
    reconciler = VocabularyReconciler([FakeProvider("tooi", [])])
    result = await reconciler.reconcile_entity(entity)
    assert result.linked is False
    assert entity.external_ids == []


@pytest.mark.asyncio
async def test_failing_provider_is_ignored():
    """A provider that raises must not break reconcile (fail-soft)."""

    class BoomProvider:
        name = "boom"

        async def lookup(self, name, entity_type):
            raise RuntimeError("down")

        async def refresh(self):
            return 0

    entity = Entity(
        canonical_name="Binnenlandse Zaken en Koninkrijksrelaties",
        entity_type="organization",
    )
    reconciler = VocabularyReconciler(
        [BoomProvider(), FakeProvider("tooi", [_bzk_match()])]
    )
    result = await reconciler.reconcile_entity(entity)
    # The healthy provider's single match still links despite the boom provider.
    assert result.linked is True


@pytest.mark.asyncio
async def test_persists_via_repo_when_supplied():
    captured = {}

    class FakeRepo:
        async def update_external_ids(self, entity_id, uris, aliases):
            captured["id"] = entity_id
            captured["uris"] = uris
            captured["aliases"] = aliases

    entity = Entity(
        id="entity:abc",
        canonical_name="Binnenlandse Zaken en Koninkrijksrelaties",
        entity_type="organization",
    )
    reconciler = VocabularyReconciler(
        [FakeProvider("tooi", [_bzk_match()])], entity_repo=FakeRepo()
    )
    result = await reconciler.reconcile_entity(entity)
    assert result.linked is True
    assert captured["id"] == "entity:abc"
    assert captured["uris"] == [
        "https://identifier.overheid.nl/tooi/id/ministerie/mnre1034"
    ]
