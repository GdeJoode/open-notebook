"""Unit tests for the TOOI vocabulary provider (K.4).

No live network: the provider talks to a ``ReferenceEntityRepository``-shaped
fake (in-memory) so the bulk-load / lookup / idempotency logic is exercised
without a database. The bundled verified seed (real ministry records) is the
default source, so AC1-2 (BZK resolves to a TOOI URI) is covered offline.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import pytest
from shared.vocabulary.tooi_provider import TOOIProvider


class FakeReferenceRepo:
    """In-memory stand-in keyed on (external_id, source_vocabulary).

    Mirrors the real repo's UNIQUE (external_id, source_vocabulary) key from
    migration 57 — the org's STABLE identity, NOT its display name. Two distinct
    orgs that share a display name (the two "Bergen" gemeenten) therefore persist
    as TWO rows; ``lookup_by_name`` returns both. Keying on canonical_name here
    (the old migration-41 key) would silently collapse them and hide the bug the
    K-D2 review caught.
    """

    def __init__(self) -> None:
        self.rows: Dict[tuple, Dict[str, Any]] = {}

    async def bulk_load(self, records: List[Dict[str, Any]]) -> int:
        n = 0
        for r in records:
            key = (r.get("external_id"), r["source_vocabulary"])
            self.rows[key] = dict(r)
            n += 1
        return n

    async def lookup_by_name(
        self, name: str, source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        out = []
        for (_eid, src), row in self.rows.items():
            if row.get("canonical_name") == name and (source is None or src == source):
                out.append(row)
        return out

    async def lookup_by_alias(
        self, alias: str, source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        out = []
        for (_eid, src), row in self.rows.items():
            if source is not None and src != source:
                continue
            if alias in (row.get("aliases") or []):
                out.append(row)
        return out

    async def lookup_by_key(
        self, key: str, source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """The real query: `$key IN lookup_keys` — exact, on stored keys."""
        return [
            row for (_eid, src), row in self.rows.items()
            if (source is None or src == source) and key in (row.get("lookup_keys") or [])
        ]


@pytest.mark.asyncio
async def test_refresh_loads_seed():
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo)
    loaded = await provider.refresh()
    assert loaded >= 10  # the verified ministry seed
    assert ("mnre1034", "tooi") in repo.rows  # BZK, keyed on organisatiecode


@pytest.mark.asyncio
async def test_refresh_idempotent_no_duplicates():
    """AC6: a second refresh upserts the same keys — no duplicate rows."""
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo)
    first = await provider.refresh()
    n_after_first = len(repo.rows)
    await provider.refresh()
    assert len(repo.rows) == n_after_first
    assert first == n_after_first  # one row per record, no growth


@pytest.mark.asyncio
async def test_lookup_by_canonical_name_returns_uri():
    """AC1-2: BZK full name → a tooi reference with a non-null external_uri."""
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo)
    await provider.refresh()

    matches = await provider.lookup(
        "Binnenlandse Zaken en Koninkrijksrelaties", "organization"
    )
    assert len(matches) == 1
    m = matches[0]
    assert m.source_vocabulary == "tooi"
    assert m.external_uri.startswith("https://identifier.overheid.nl/tooi/id/")
    assert m.external_id == "mnre1034"
    assert "BZK" in m.aliases
    assert m.confidence > 0.9


@pytest.mark.asyncio
async def test_lookup_by_alias_abbreviation():
    """The abbreviation surface form resolves via the alias path."""
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo)
    await provider.refresh()

    matches = await provider.lookup("BZK", "organization")
    assert len(matches) == 1
    assert matches[0].external_id == "mnre1034"
    # alias path → slightly lower confidence than an exact canonical hit
    assert matches[0].confidence < 0.99


@pytest.mark.asyncio
async def test_lookup_miss_returns_empty():
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo)
    await provider.refresh()
    assert await provider.lookup("Totally Unknown Org", "organization") == []


@pytest.mark.asyncio
async def test_lookup_empty_name_returns_empty():
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo)
    await provider.refresh()
    assert await provider.lookup("", "organization") == []


@pytest.mark.asyncio
async def test_lookup_failsoft_on_repo_error():
    """A repo that raises must degrade to [] (never propagate)."""

    class BoomRepo(FakeReferenceRepo):
        async def lookup_by_name(self, name, source=None):
            raise RuntimeError("db down")

    provider = TOOIProvider(BoomRepo())
    assert await provider.lookup("BZK", "organization") == []


@pytest.mark.asyncio
async def test_refresh_from_bulk_file(tmp_path):
    """refresh(source=<file>) ingests the documented bulk JSON format."""
    bulk = [
        {
            "organisatiecode": "mnre9999",
            "afkorting": "TEST",
            "naam_excl_soort": "Test Ministerie",
            "naam_incl_soort": "ministerie van Test",
            "soort": "ministerie",
        }
    ]
    f = tmp_path / "tooi_organisations.json"
    f.write_text(json.dumps(bulk), encoding="utf-8")

    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo, source=str(f))
    loaded = await provider.refresh()
    assert loaded == 1
    matches = await provider.lookup("Test Ministerie", "organization")
    assert matches[0].external_id == "mnre9999"


@pytest.mark.asyncio
async def test_missing_bulk_file_falls_back_to_seed(tmp_path):
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo, source=str(tmp_path / "nope.json"))
    loaded = await provider.refresh()
    assert loaded >= 10  # fell back to the bundled seed


# --- remote bulk fetch (K-D2) ------------------------------------------------


class FakeBulkFetcher:
    """Stand-in for ``TOOIBulkFetcher`` — returns a fixed record set, no network."""

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records
        self.calls = 0

    async def fetch_records(self) -> List[Dict[str, Any]]:
        self.calls += 1
        return list(self._records)


_REMOTE_RECORDS = [
    {
        "organisatiecode": "mnre1058",
        "afkorting": "JenV",
        "naam_excl_soort": "Justitie en Veiligheid",
        "naam_incl_soort": "ministerie van Justitie en Veiligheid",
        "soort": "ministerie",
        "aliases": ["Veiligheid en Justitie", "VenJ", "MinJus"],
    },
    {
        "organisatiecode": "pv20",
        "afkorting": "GR",
        "naam_excl_soort": "Groningen",
        "naam_incl_soort": "provincie Groningen",
        "soort": "provincie",
        "aliases": [],
    },
]


@pytest.mark.asyncio
async def test_refresh_uses_remote_fetcher_when_attached():
    """A fetcher (no source file) feeds the full vocabulary into reference_entity."""
    repo = FakeReferenceRepo()
    fetcher = FakeBulkFetcher(_REMOTE_RECORDS)
    provider = TOOIProvider(repo, bulk_fetcher=fetcher)

    loaded = await provider.refresh()
    assert loaded == 2
    assert fetcher.calls == 1
    assert ("mnre1058", "tooi") in repo.rows  # Justitie en Veiligheid
    assert ("pv20", "tooi") in repo.rows  # provincie Groningen


@pytest.mark.asyncio
async def test_remote_records_carry_historical_aliases():
    """An org's historical names + abbreviations resolve via the alias path."""
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo, bulk_fetcher=FakeBulkFetcher(_REMOTE_RECORDS))
    await provider.refresh()

    for surface in ("VenJ", "Veiligheid en Justitie", "MinJus"):
        matches = await provider.lookup(surface, "organization")
        assert matches and matches[0].external_id == "mnre1058", surface


@pytest.mark.asyncio
async def test_refresh_remote_idempotent_no_duplicates():
    """AC6 at scale: re-running the remote refresh upserts, never duplicates."""
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo, bulk_fetcher=FakeBulkFetcher(_REMOTE_RECORDS))
    first = await provider.refresh()
    n = len(repo.rows)
    await provider.refresh()
    assert len(repo.rows) == n
    assert first == n


@pytest.mark.asyncio
async def test_refresh_empty_remote_falls_back_to_seed():
    """An unreachable register set (fetcher returns []) degrades to the seed."""
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo, bulk_fetcher=FakeBulkFetcher([]))
    loaded = await provider.refresh()
    assert loaded >= 10  # bundled seed
    assert ("mnre1034", "tooi") in repo.rows  # BZK


@pytest.mark.asyncio
async def test_refresh_file_source_overrides_remote(tmp_path):
    """An operator bulk file takes priority over the remote fetcher."""
    bulk = [
        {
            "organisatiecode": "mnre9999",
            "afkorting": "TEST",
            "naam_excl_soort": "Test Ministerie",
            "naam_incl_soort": "ministerie van Test",
            "soort": "ministerie",
        }
    ]
    f = tmp_path / "tooi_organisations.json"
    f.write_text(json.dumps(bulk), encoding="utf-8")

    fetcher = FakeBulkFetcher(_REMOTE_RECORDS)
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo, source=str(f), bulk_fetcher=fetcher)
    loaded = await provider.refresh()

    assert loaded == 1
    assert fetcher.calls == 0  # remote not consulted when a file is present
    assert ("mnre9999", "tooi") in repo.rows


@pytest.mark.asyncio
async def test_refresh_dedupes_same_code_distinct_surface_forms():
    """Two rows for one organisatiecode collapse to one reference row (aliases union)."""
    repo = FakeReferenceRepo()
    dup = [
        {
            "organisatiecode": "mnre1058",
            "afkorting": "JenV",
            "naam_excl_soort": "Justitie en Veiligheid",
            "naam_incl_soort": "ministerie van Justitie en Veiligheid",
            "soort": "ministerie",
            "aliases": ["JenV"],
        },
        {
            "organisatiecode": "mnre1058",
            "afkorting": "VenJ",
            "naam_excl_soort": "Veiligheid en Justitie",
            "naam_incl_soort": "ministerie van Veiligheid en Justitie",
            "soort": "ministerie",
            "aliases": ["VenJ"],
        },
    ]
    provider = TOOIProvider(repo, bulk_fetcher=FakeBulkFetcher(dup))
    loaded = await provider.refresh()
    assert loaded == 1  # one row for the org, not two
    row = next(iter(repo.rows.values()))
    assert "VenJ" in row["aliases"] and "JenV" in row["aliases"]


# --- duplicate display name across DISTINCT orgs (K-D2 rev2 blocker) ----------

# The real 605-gemeente register has TWO distinct "Bergen" municipalities:
# gm0373 (Noord-Holland) and gm0893 (Limburg). Both carry canonical_name="Bergen".
_TWO_BERGENS = [
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


@pytest.mark.asyncio
async def test_refresh_keeps_distinct_same_named_orgs_as_two_rows():
    """Two DISTINCT organisatiecodes sharing canonical_name="Bergen" → 2 rows.

    Regression for the K-D2 rev2 BLOCKER: keying ingest on the display name
    collapsed the two Bergen gemeenten into one reference row, which let the
    reconciler auto-link every "Bergen" entity to the surviving (wrong) URI.
    Keying on external_id (migration 57) keeps both.
    """
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo, bulk_fetcher=FakeBulkFetcher(_TWO_BERGENS))
    loaded = await provider.refresh()

    assert loaded == 2  # the count reflects BOTH persisted rows, not a collapse
    assert ("gm0373", "tooi") in repo.rows
    assert ("gm0893", "tooi") in repo.rows

    # lookup_by_name("Bergen") must surface BOTH distinct orgs (the ambiguity the
    # reconciler needs) — not a single silently-merged candidate.
    matches = await provider.lookup("Bergen", "organization")
    uris = sorted(m.external_uri for m in matches)
    assert len(matches) == 2
    assert uris == [
        "https://identifier.overheid.nl/tooi/id/gemeente/gm0373",
        "https://identifier.overheid.nl/tooi/id/gemeente/gm0893",
    ]


# ---------------------------------------------------------------------------
# D1: a decorated name resolves through its parts — linkably only when every part
# carries evidence and they agree
# ---------------------------------------------------------------------------

from shared.vocabulary.tooi_provider import ONE_SIDED_CONFIDENCE  # noqa: E402

GOV = "government_organization"
_D1_BULK = [
    {"organisatiecode": "mnre1171", "afkorting": "VRO",
     "naam_excl_soort": "Volkshuisvesting en Ruimtelijke Ordening",
     "naam_incl_soort": "ministerie van Volkshuisvesting en Ruimtelijke Ordening",
     "soort": "ministerie"},
    {"organisatiecode": "mnre1025", "afkorting": "VWS",
     "naam_excl_soort": "Volksgezondheid, Welzijn en Sport",
     "naam_incl_soort": "ministerie van Volksgezondheid, Welzijn en Sport",
     "soort": "ministerie"},
    {"organisatiecode": "mnre1130", "afkorting": "IenW",
     "naam_excl_soort": "Infrastructuur en Waterstaat",
     "naam_incl_soort": "ministerie van Infrastructuur en Waterstaat",
     "soort": "ministerie"},
    {"organisatiecode": "mnre1034", "afkorting": "BZK",
     "naam_excl_soort": "Binnenlandse Zaken en Koninkrijksrelaties",
     "naam_incl_soort": "ministerie van Binnenlandse Zaken en Koninkrijksrelaties",
     "soort": "ministerie"},
    {"organisatiecode": "pv21", "afkorting": "OV",
     "naam_excl_soort": "Overijssel", "naam_incl_soort": "provincie Overijssel",
     "soort": "provincie"},
]


async def _d1_provider(tmp_path, bulk=None) -> TOOIProvider:
    f = tmp_path / "tooi_d1.json"
    f.write_text(json.dumps(bulk or _D1_BULK), encoding="utf-8")
    provider = TOOIProvider(FakeReferenceRepo(), source=str(f))
    await provider.refresh()
    return provider


@pytest.mark.asyncio
@pytest.mark.parametrize("name, code", [
    ("Volkshuisvesting en Ruimtelijke Ordening (Ministerie)", "mnre1171"),
    ("VRO (Volkshuisvesting en Ruimtelijke Ordening)", "mnre1171"),
    ("IenW (Infrastructuur en Waterstaat)", "mnre1130"),
    ("ministeries van het Rijk: Volkshuisvesting en Ruimtelijke Ordening", "mnre1171"),
    # D2: the lookup key knows the corpus spelling `Koninkrijkrelaties`, so the
    # gloss now hits too and both parts agree.
    ("BZK (Binnenlandse Zaken en Koninkrijkrelaties)", "mnre1034"),
])
async def test_a_decorated_name_whose_parts_all_agree_is_linkable(tmp_path, name, code):
    """Corpus strings that K.4 as shipped resolved to nothing (D2 experiment)."""
    provider = await _d1_provider(tmp_path)
    matches = await provider.lookup(name, GOV)
    assert [m.external_id for m in matches] == [code]
    assert matches[0].confidence > ONE_SIDED_CONFIDENCE
    assert "one-sided" not in matches[0].matched_on
    assert matches[0].matched_on  # the rule that made the call is recorded


@pytest.mark.asyncio
async def test_all_agreeing_parts_are_named_in_the_provenance(tmp_path):
    provider = await _d1_provider(tmp_path)
    [m] = await provider.lookup("IenW (Infrastructuur en Waterstaat)", GOV)
    assert m.matched_on == "abbreviation: IenW + Infrastructuur en Waterstaat"


@pytest.mark.asyncio
@pytest.mark.parametrize("name, etype, code", [
    # constructed: the initials fit, the gloss is no register name.
    ("BZK (Binnenlandse Zaken en Koninkrijk)", GOV, "mnre1034"),
    # D1 review: the abbreviation hits a province, the gloss names public transport.
    ("OV (Openbaar Vervoer)", "concept", "pv21"),
])
async def test_evidence_from_one_part_is_a_candidate_never_a_link(tmp_path, name, etype, code):
    provider = await _d1_provider(tmp_path)
    [m] = await provider.lookup(name, etype)
    assert m.external_id == code
    assert m.confidence == ONE_SIDED_CONFIDENCE
    assert "one-sided" in m.matched_on


@pytest.mark.asyncio
async def test_a_gloss_with_the_right_initials_but_no_register_entry_is_one_sided(tmp_path):
    """Constructed: the initials check passes, the gloss names nothing registered."""
    provider = await _d1_provider(tmp_path)
    [m] = await provider.lookup("VRO (Volkshuisvesting Ruimte Ordening)", GOV)
    assert m.external_id == "mnre1171"
    assert m.confidence == ONE_SIDED_CONFIDENCE


@pytest.mark.asyncio
async def test_parts_that_disagree_return_both_so_the_reconciler_refuses(tmp_path):
    """Constructed: the abbreviation names one registered body, the gloss (same
    initials) names another. Both come back at full confidence — the reconciler
    links only a single match, so this is a refusal with both recorded."""
    bulk = _D1_BULK + [
        {"organisatiecode": "x9", "afkorting": "VRegO",
         "naam_excl_soort": "Vereniging Regionale Omroepen",
         "naam_incl_soort": "Vereniging Regionale Omroepen", "soort": "overige"},
    ]
    provider = await _d1_provider(tmp_path, bulk)
    matches = await provider.lookup("VRO (Vereniging Regionale Omroepen)", "organization")
    assert sorted(m.external_id for m in matches) == ["mnre1171", "x9"]
    assert all(m.confidence > ONE_SIDED_CONFIDENCE for m in matches)


@pytest.mark.asyncio
async def test_the_intersection_narrows_an_ambiguous_part_to_the_agreed_one(tmp_path):
    """Constructed: `EZ` is the abbreviation of two records; the gloss names one.
    Without the intersection both would come back and the reconciler would refuse."""
    bulk = [
        {"organisatiecode": "x1", "afkorting": "EZ", "naam_excl_soort": "Economische Zaken",
         "naam_incl_soort": "ministerie van Economische Zaken", "soort": "ministerie"},
        {"organisatiecode": "x2", "afkorting": "EZ", "naam_excl_soort": "Eerste Zaken",
         "naam_incl_soort": "ministerie van Eerste Zaken", "soort": "ministerie"},
    ]
    provider = await _d1_provider(tmp_path, bulk)
    matches = await provider.lookup("EZ (Economische Zaken)", GOV)
    assert [m.external_id for m in matches] == ["x1"]
    assert matches[0].confidence > ONE_SIDED_CONFIDENCE


@pytest.mark.asyncio
@pytest.mark.parametrize("name, etype", [
    ("De Staatssecretaris van Infrastructuur en Waterstaat (IenW)", GOV),
    ("ministeries van het Rijk: Volkshuisvesting en Ruimtelijke Ordening, "
     "Binnenlandse Zaken en Koninkrijkrelaties/Herstel Groningen, "
     "Sociale Zaken en Werkgelegenheid, Justitie en Veiligheid", GOV),
    # D1 review, wider graph: glosses that are not VRO's initials.
    ("VRO (Vereniging van Regio's)", "organization"),
    ("VRO (VROM, Volkshuisvesting, Ruimtelijke Ordening en Milieubeheer)", GOV),
    # Constructed by the review: a unit OF BZK, in the parenthesis position.
    ("Directoraat-generaal Volkshuisvesting (BZK)", GOV),
])
async def test_an_organ_a_list_or_a_foreign_gloss_resolves_to_nothing(tmp_path, name, etype):
    provider = await _d1_provider(tmp_path)
    assert await provider.lookup(name, etype) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("name, code", [
    ("VRO", "mnre1171"),
    # D2: case-insensitive — TOOI's alias is "ministerie van …", lowercase m.
    ("Ministerie van Volksgezondheid, Welzijn en Sport", "mnre1025"),
    ("volkshuisvesting en ruimtelijke ordening", "mnre1171"),
])
async def test_a_whole_name_hit_never_decomposes(tmp_path, name, code):
    provider = await _d1_provider(tmp_path)
    matches = await provider.lookup(name, GOV)
    assert [m.external_id for m in matches] == [code]
    assert matches[0].matched_on == ""


@pytest.mark.asyncio
async def test_a_canonical_key_hit_outranks_an_alias_key_hit(tmp_path):
    provider = await _d1_provider(tmp_path)
    [by_name] = await provider.lookup("infrastructuur en waterstaat", GOV)
    [by_alias] = await provider.lookup("IENW", GOV)
    assert by_name.confidence == 0.99 and by_alias.confidence == 0.9


@pytest.mark.asyncio
async def test_one_key_on_two_organisations_returns_both(tmp_path):
    """D2 dry run: `Economische Zaken` is mnre1040's name (ended 2010) and a past
    name of mnre1045 (EZK). Ranking name over alias picked the ended one; both
    must come back so the reconciler refuses. Records as the fixed loader builds
    them from the live register."""
    bulk = [
        {"organisatiecode": "mnre1040", "afkorting": "EZ",
         "naam_excl_soort": "Economische Zaken",
         "naam_incl_soort": "ministerie van Economische Zaken", "soort": "ministerie",
         "einddatum": "2010-10-13"},
        {"organisatiecode": "mnre1045", "afkorting": "EZK",
         "naam_excl_soort": "Economische Zaken en Klimaat",
         "naam_incl_soort": "ministerie van Economische Zaken en Klimaat",
         "soort": "ministerie", "begindatum": "2010-10-14",
         "aliases": ["Economische Zaken", "EZ", "ministerie van Economische Zaken"]},
    ]
    provider = await _d1_provider(tmp_path, bulk)
    for name in ("Economische Zaken", "Economische Zaken (Ministerie)"):
        matches = await provider.lookup(name, GOV)
        assert sorted(m.external_id for m in matches) == ["mnre1040", "mnre1045"], name
