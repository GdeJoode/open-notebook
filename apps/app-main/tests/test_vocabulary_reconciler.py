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

    async def lookup_by_key(self, key, source=None):
        return [
            row
            for (_eid, src), row in self.rows.items()
            if (source is None or src == source) and key in (row.get("lookup_keys") or [])
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
        aliases=["BZK", "ministerie van Binnenlandse Zaken en Koninkrijksrelaties",
                 "Binnenlandse Zaken"],  # the last: a historical name
        confidence=confidence,
        current_names=["Binnenlandse Zaken en Koninkrijksrelaties", "BZK",
                       "ministerie van Binnenlandse Zaken en Koninkrijksrelaties"],
    )


@pytest.mark.asyncio
async def test_single_match_auto_links():
    """AC2: a BZK org entity gets the TOOI URI + the record's CURRENT names (D2,
    owner decision: not the historical alias list)."""
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
    assert "Binnenlandse Zaken" not in entity.aliases
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
    # D1: kept for a curator rather than discarded.
    assert [c.external_uri for c in result.candidates] == [_bzk_match().external_uri]


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


# --- D1: the real provider behind the real reconciler --------------------------


async def _tooi_with(records):
    from shared.vocabulary.tooi_provider import TOOIProvider

    class _Fetcher:
        async def fetch_records(self):
            return records

    provider = TOOIProvider(_InMemoryReferenceRepo(), bulk_fetcher=_Fetcher())
    await provider.refresh()
    return provider


_D1_RECORDS = [
    {"organisatiecode": "mnre1171", "afkorting": "VRO",
     "naam_excl_soort": "Volkshuisvesting en Ruimtelijke Ordening",
     "naam_incl_soort": "ministerie van Volkshuisvesting en Ruimtelijke Ordening",
     "soort": "ministerie"},
    {"organisatiecode": "mnre1034", "afkorting": "BZK",
     "naam_excl_soort": "Binnenlandse Zaken en Koninkrijksrelaties",
     "naam_incl_soort": "ministerie van Binnenlandse Zaken en Koninkrijksrelaties",
     "soort": "ministerie"},
]


@pytest.mark.asyncio
async def test_a_decorated_name_whose_parts_agree_links():
    provider = await _tooi_with(_D1_RECORDS)
    entity = Entity(canonical_name="VRO (Volkshuisvesting en Ruimtelijke Ordening)",
                    entity_type="government_organization")
    result = await VocabularyReconciler([provider]).reconcile_entity(entity)
    assert result.linked is True
    assert entity.external_ids == ["https://identifier.overheid.nl/tooi/id/ministerie/mnre1171"]


@pytest.mark.asyncio
async def test_one_sided_evidence_is_recorded_for_a_curator_and_never_linked():
    """Owner decision 2026-09-26: only the abbreviation hits (constructed gloss
    that is no register name). A candidate, not a link."""
    provider = await _tooi_with(_D1_RECORDS)
    entity = Entity(canonical_name="BZK (Binnenlandse Zaken en Koninkrijk)",
                    entity_type="government_organization")
    result = await VocabularyReconciler([provider]).reconcile_entity(entity)
    assert result.linked is False
    assert entity.external_ids == []
    assert [c.external_id for c in result.candidates] == ["mnre1034"]
    assert "one-sided" in result.candidates[0].matched_on


# --- D2: the threshold guard and the batch caller -----------------------------


def test_a_threshold_that_would_link_one_sided_matches_is_refused():
    from shared.vocabulary.tooi_provider import ONE_SIDED_CONFIDENCE

    with pytest.raises(ValueError):
        VocabularyReconciler([], confidence_threshold=ONE_SIDED_CONFIDENCE)
    VocabularyReconciler([], confidence_threshold=0.85)  # the default is fine


def _row(eid, name, etype, **extra):
    return {"id": eid, "canonical_name": name, "entity_type": etype, "aliases": [], **extra}


@pytest.mark.asyncio
async def test_a_dry_run_reports_and_writes_nothing():
    from app_main.services.entity_resolution.vocabulary_reconcile_service import (
        reconcile_active_entities,
    )

    provider = await _tooi_with(_D1_RECORDS)
    rows = [
        _row("entity:a", "Volkshuisvesting en Ruimtelijke Ordening (Ministerie)",
             "government_organization"),
        # The staging defect: external_ids stored as an object.
        _row("entity:b", "BZK", "organization", external_ids={}),
        _row("entity:c", "Nergens Genoemd", "organization"),
    ]
    report = await reconcile_active_entities(
        VocabularyReconciler([provider]), rows=rows, dry_run=True)
    assert report.scanned == 3
    assert {r["id"] for r in report.linked} == {"entity:a", "entity:b"}
    assert report.by_type == {"government_organization": {"linked": 1},
                              "organization": {"linked": 1, "no_match": 1}}


@pytest.mark.asyncio
async def test_a_real_run_writes_only_the_links():
    from app_main.services.entity_resolution.vocabulary_reconcile_service import (
        reconcile_active_entities,
    )

    writes = []

    class Repo:
        async def update_external_ids(self, entity_id, uris, aliases):
            writes.append((entity_id, list(uris)))

    provider = await _tooi_with(_D1_RECORDS)
    rows = [
        _row("entity:a", "VRO", "government_organization"),
        # one-sided (constructed gloss): a candidate, so no write
        _row("entity:b", "BZK (Binnenlandse Zaken en Koninkrijk)", "government_organization"),
    ]
    report = await reconcile_active_entities(
        VocabularyReconciler([provider], entity_repo=Repo()), rows=rows, dry_run=False)
    assert writes == [("entity:a",
                       ["https://identifier.overheid.nl/tooi/id/ministerie/mnre1171"])]
    assert [c["id"] for c in report.candidates] == ["entity:b"]


# Records as the fixed loader builds them from the LIVE registers (gemeenten v5,
# provincies v1). TOOI writes `Bergen (NH)` / `Bergen (L)` — no bare `Bergen`, no
# dots — and `Groningen` is both a gemeente and a provincie.
_LIVE_AREAS = [
    {"organisatiecode": "gm0373", "afkorting": "", "naam_excl_soort": "Bergen (NH)",
     "naam_incl_soort": "gemeente Bergen (NH)", "soort": "gemeente"},
    {"organisatiecode": "gm0893", "afkorting": "", "naam_excl_soort": "Bergen (L)",
     "naam_incl_soort": "gemeente Bergen (L)", "soort": "gemeente"},
    {"organisatiecode": "gm0014", "afkorting": "", "naam_excl_soort": "Groningen",
     "naam_incl_soort": "gemeente Groningen", "soort": "gemeente"},
    {"organisatiecode": "pv20", "afkorting": "GR", "naam_excl_soort": "Groningen",
     "naam_incl_soort": "provincie Groningen", "soort": "provincie"},
    {"organisatiecode": "gm0091", "afkorting": "", "naam_excl_soort": "Sneek",
     "naam_incl_soort": "gemeente Sneek", "soort": "gemeente", "einddatum": "2010-12-31"},
    {"organisatiecode": "gm0432", "afkorting": "", "naam_excl_soort": "Opmeer",
     "naam_incl_soort": "gemeente Opmeer", "soort": "gemeente"},
]


async def _run(rows, *, dry_run=True, entity_repo=None, queue=None):
    from app_main.services.entity_resolution.vocabulary_reconcile_service import (
        reconcile_active_entities,
    )

    provider = await _tooi_with(_LIVE_AREAS + _D1_RECORDS)
    return await reconcile_active_entities(
        VocabularyReconciler([provider], entity_repo=entity_repo),
        rows=rows, dry_run=dry_run, queue=queue)


@pytest.mark.asyncio
async def test_areas_against_live_shaped_records():
    report = await _run([
        _row("e:1", "Gemeente Opmeer", "administrative_area", status="active"),
        _row("e:2", "Opmeer", "administrative_area", status="reference"),
        # The municipality and the province share the name: refused, both kept.
        _row("e:3", "Groningen", "administrative_area", status="active"),
        # TOOI has no bare `Bergen`: no match, not a guess.
        _row("e:4", "Bergen", "administrative_area", status="active"),
        # Ended 2010: a candidate, never a link (owner decision).
        _row("e:5", "Sneek", "administrative_area", status="active"),
    ])
    assert {r["id"] for r in report.linked} == {"e:1", "e:2"}
    reasons = {c["id"]: c["reason"] for c in report.candidates}
    assert reasons == {"e:3": "ambiguous_multiple_candidates", "e:5": "ended_organisation"}
    assert [c["candidates"][0]["valid_until"] for c in report.candidates if c["id"] == "e:5"]         == ["2010-12-31"]
    assert [r["id"] for r in report.no_match] == ["e:4"]
    assert report.scanned_by["administrative_area"] == {"active": 4, "reference": 1}
    assert report.scanned_by["organization"] == {}  # a type that scanned nothing is shown


@pytest.mark.asyncio
async def test_a_row_without_a_name_is_skipped_not_fatal():
    report = await _run([
        {"id": "e:9", "canonical_name": None, "entity_type": "organization",
         "status": "reference"},
        _row("e:1", "Gemeente Opmeer", "administrative_area", status="active"),
    ])
    assert report.skipped_no_name == 1
    assert [r["id"] for r in report.linked] == ["e:1"]


@pytest.mark.asyncio
async def test_a_refused_write_is_reported_as_failed_not_linked():
    """D2 review, B2: on staging's drifted schema the write failed, the repository
    returned False, and the link was reported anyway."""

    class RefusingRepo:
        async def update_external_ids(self, entity_id, uris, aliases=None):
            return False

    report = await _run([_row("e:1", "Gemeente Opmeer", "administrative_area",
                              status="active")],
                        dry_run=False, entity_repo=RefusingRepo())
    assert report.linked == []
    assert [r["id"] for r in report.write_failed] == ["e:1"]


@pytest.mark.asyncio
async def test_candidates_reach_the_curator_queue_on_a_real_run_only():
    calls = []

    class Queue:
        async def upsert_tooi(self, entity_id, **kw):
            calls.append({"entity_id": entity_id, **kw})
            return "triage_queue:x"

    rows = [_row("e:3", "Groningen", "administrative_area", status="active")]
    await _run(rows, dry_run=True, queue=Queue())
    assert calls == []  # a dry run writes nothing, the queue included

    class Repo:
        async def update_external_ids(self, entity_id, uris, aliases=None):
            return True

    report = await _run(rows, dry_run=False, entity_repo=Repo(), queue=Queue())
    assert report.queued == 1
    [call] = calls
    assert call["entity_id"] == "e:3"
    assert call["reason"].startswith("tooi: ambiguous_multiple_candidates: ")
    assert {c["uri"].rsplit("/", 2)[-2] for c in call["candidates"]} == {"gemeente", "provincie"}
    assert all("current_names" in c for c in call["candidates"])


@pytest.mark.asyncio
async def test_an_answered_candidate_is_counted_not_requeued():
    class Queue:
        async def upsert_tooi(self, entity_id, **kw):
            return None  # the repository's "already decided for this set"

    class Repo:
        async def update_external_ids(self, entity_id, uris, aliases=None):
            return True

    report = await _run([_row("e:3", "Groningen", "administrative_area", status="active")],
                        dry_run=False, entity_repo=Repo(), queue=Queue())
    assert (report.queued, report.already_decided) == (0, 1)
