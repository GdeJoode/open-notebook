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


class _InMemoryReferenceRepo:
    """Reference repo keyed on (external_id, source) — migration 57's stable key.

    The same key the real ``ReferenceEntityRepository`` upsert uses, so two
    distinct orgs that share a display name persist as two rows and
    ``lookup_by_name`` returns both (the ambiguity the reconciler needs).
    """

    def __init__(self) -> None:
        self.rows: dict = {}

    async def bulk_load(self, records):
        n = 0
        for r in records:
            self.rows[(r.get("external_id"), r["source_vocabulary"])] = dict(r)
            n += 1
        return n

    async def lookup_by_name(self, name, source=None):
        return [
            row
            for (_eid, src), row in self.rows.items()
            if row.get("canonical_name") == name and (source is None or src == source)
        ]

    async def lookup_by_alias(self, alias, source=None):
        return [
            row
            for (_eid, src), row in self.rows.items()
            if (source is None or src == source) and alias in (row.get("aliases") or [])
        ]


class _BergenFetcher:
    """Bulk fetcher returning the two distinct "Bergen" gemeenten (no network)."""

    async def fetch_records(self):
        return [
            {
                "organisatiecode": "gm0373",
                "afkorting": "",
                "naam_excl_soort": "Bergen",
                "naam_incl_soort": "gemeente Bergen (NH.)",
                "soort": "gemeente",
                "aliases": ["Bergen (NH.)"],
            },
            {
                "organisatiecode": "gm0893",
                "afkorting": "",
                "naam_excl_soort": "Bergen",
                "naam_incl_soort": "gemeente Bergen (L.)",
                "soort": "gemeente",
                "aliases": ["Bergen (L.)"],
            },
        ]


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
async def test_two_same_named_orgs_via_tooi_provider_do_not_auto_link():
    """K-D2 rev2: two distinct-URI "Bergen" rows from the real TOOIProvider → NO link.

    Drives the full provider path (not a hand-rolled FakeProvider): the TOOI
    provider's ``lookup_by_name("Bergen")`` returns BOTH reference rows (distinct
    organisatiecodes gm0373 / gm0893), so the reconciler sees two distinct-URI
    candidates and correctly refuses to auto-link. This is the behaviour the
    ingest-level fix (migration 57) preserves: the ambiguity is no longer
    destroyed at load, so the precision guard fires.
    """
    from shared.vocabulary.tooi_provider import TOOIProvider

    repo = _InMemoryReferenceRepo()
    provider = TOOIProvider(repo, bulk_fetcher=_BergenFetcher())
    await provider.refresh()

    entity = Entity(canonical_name="Bergen", entity_type="organization")
    reconciler = VocabularyReconciler([provider])
    result = await reconciler.reconcile_entity(entity)

    assert result.linked is False
    assert entity.external_ids == []  # nothing written (ambiguous)
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
