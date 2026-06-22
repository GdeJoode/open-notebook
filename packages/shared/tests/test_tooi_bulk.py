"""Unit tests for the TOOI bulk-vocabulary fetcher (K-D2).

ALL HTTP IS MOCKED — no live network in CI. We inject an ``httpx.AsyncClient``
backed by ``httpx.MockTransport`` so the tests exercise the real JSON-LD parsing,
canonical-URI grouping, alias collection and fail-soft union logic without
touching the network. The fixtures below are trimmed but faithful copies of the
expanded JSON-LD the official registers actually return (verified live
2026-06-22): an org node carries ``ont:organisatiecode`` etc., a historical name
version points at the canonical entity via ``prov:specializationOf``, and
non-org metadata nodes (the waardelijst header) carry no organisatiecode.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import httpx
import pytest
from shared.vocabulary.http_client import VocabularyHTTPClient
from shared.vocabulary.tooi_bulk import (
    TOOIBulkFetcher,
    parse_register,
    register_json_url,
)

_ONT = "https://identifier.overheid.nl/tooi/def/ont/"
_ID = "https://identifier.overheid.nl/tooi/id"


def _lit(value: str) -> List[Dict[str, str]]:
    return [{"@value": value}]


def _org_node(
    *,
    node_id: str,
    soort_path: str,
    klass: str,
    code: str,
    afkorting: str,
    naam_excl: str,
    naam_incl: str,
    specialization_of: str | None = None,
    einddatum: str | None = None,
) -> Dict[str, Any]:
    node: Dict[str, Any] = {
        "@id": f"{_ID}/{soort_path}/{node_id}",
        "@type": [f"{_ONT}{klass}"],
        f"{_ONT}organisatiecode": _lit(code),
        f"{_ONT}afkorting": _lit(afkorting),
        f"{_ONT}officieleNaamExclSoort": _lit(naam_excl),
        f"{_ONT}officieleNaamInclSoort": _lit(naam_incl),
        f"{_ONT}voorkeursnaamExclSoort": _lit(naam_excl),
        f"{_ONT}voorkeursnaamInclSoort": _lit(naam_incl),
    }
    if specialization_of is not None:
        node["http://www.w3.org/ns/prov#specializationOf"] = [
            {"@id": f"{_ID}/{soort_path}/{specialization_of}"}
        ]
    if einddatum is not None:
        node[f"{_ONT}einddatum"] = _lit(einddatum)
    return node


def _metadata_node() -> Dict[str, Any]:
    """A non-org node (the waardelijst header) — must be ignored by the parser."""
    return {
        "@id": f"{_ID}/set/rwc_ministeries_compleet/6",
        "@type": [f"{_ONT}wl/RegisterwaardelijstCompleet"],
        "http://purl.org/dc/terms/description": _lit("Alle ministeries"),
    }


# --- A faithful, trimmed ministries register --------------------------------
# mnre1034 (BZK) — one current version.
# mnre1058 (Justitie) — three versions over time (VenJ → JenV), all pointing at
# the canonical mnre1058 via specializationOf, two of them ended.
_MINISTRIES: List[Dict[str, Any]] = [
    _metadata_node(),
    _org_node(
        node_id="mnre1034",
        soort_path="ministerie",
        klass="Ministerie",
        code="mnre1034",
        afkorting="BZK",
        naam_excl="Binnenlandse Zaken en Koninkrijksrelaties",
        naam_incl="ministerie van Binnenlandse Zaken en Koninkrijksrelaties",
    ),
    _org_node(
        node_id="hv_001",
        soort_path="ministerie",
        klass="Ministerie",
        code="mnre1058",
        afkorting="VenJ",
        naam_excl="Veiligheid en Justitie",
        naam_incl="ministerie van Veiligheid en Justitie",
        specialization_of="mnre1058",
        einddatum="2017-10-26",
    ),
    _org_node(
        node_id="hv_002",
        soort_path="ministerie",
        klass="Ministerie",
        code="mnre1058",
        afkorting="MinJus",
        naam_excl="Justitie",
        naam_incl="ministerie van Justitie",
        specialization_of="mnre1058",
        einddatum="2010-10-13",
    ),
    _org_node(
        node_id="hv_003",
        soort_path="ministerie",
        klass="Ministerie",
        code="mnre1058",
        afkorting="JenV",
        naam_excl="Justitie en Veiligheid",
        naam_incl="ministerie van Justitie en Veiligheid",
        specialization_of="mnre1058",
    ),
]

_PROVINCIES: List[Dict[str, Any]] = [
    _org_node(
        node_id="pv20",
        soort_path="provincie",
        klass="Provincie",
        code="pv20",
        afkorting="GR",
        naam_excl="Groningen",
        naam_incl="provincie Groningen",
    ),
]


def _fetcher_with_payloads(
    payloads: Dict[str, Any], *, registers: List[Dict[str, Any]]
) -> TOOIBulkFetcher:
    """A fetcher whose HTTP client serves ``payloads`` keyed by URL substring."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        for needle, body in payloads.items():
            if needle in url:
                return httpx.Response(200, json=body)
        return httpx.Response(404)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    http = VocabularyHTTPClient(
        user_agent="open-notebook/0.1 (mailto:test@example.com)",
        min_interval=0.0,
        cache_ttl=0.0,
        client=client,
    )
    return TOOIBulkFetcher(http_client=http, registers=registers)


# --- parse_register ----------------------------------------------------------


def test_parse_register_groups_versions_by_org():
    records = parse_register(_MINISTRIES, default_soort="ministerie")
    by_code = {r["organisatiecode"]: r for r in records}

    # One record per distinct organisatiecode, metadata node ignored.
    assert set(by_code) == {"mnre1034", "mnre1058"}

    bzk = by_code["mnre1034"]
    assert bzk["naam_excl_soort"] == "Binnenlandse Zaken en Koninkrijksrelaties"
    assert bzk["soort"] == "ministerie"


def test_parse_register_canonical_is_current_version():
    """The not-ended version supplies the canonical name; ended ones are aliases."""
    records = parse_register(_MINISTRIES, default_soort="ministerie")
    jus = next(r for r in records if r["organisatiecode"] == "mnre1058")

    # The current (no einddatum) version is JenV / "Justitie en Veiligheid".
    assert jus["naam_excl_soort"] == "Justitie en Veiligheid"
    assert jus["afkorting"] == "JenV"
    # Historical names + abbreviations are preserved as resolvable aliases.
    assert "Veiligheid en Justitie" in jus["aliases"]
    assert "VenJ" in jus["aliases"]
    assert "MinJus" in jus["aliases"]
    # Canonical name is not duplicated into aliases.
    assert "Justitie en Veiligheid" not in jus["aliases"]


def test_parse_register_tolerates_non_list():
    assert parse_register({"not": "a list"}, default_soort="ministerie") == []
    assert parse_register(None, default_soort="ministerie") == []


def test_register_json_url_shape():
    url = register_json_url("rwc_ministeries_compleet", 6)
    assert url == (
        "https://repository.officiele-overheidspublicaties.nl/waardelijsten/"
        "rwc_ministeries_compleet/6/json/rwc_ministeries_compleet_6.json"
    )


# --- TOOIBulkFetcher (mocked HTTP) ------------------------------------------


@pytest.mark.asyncio
async def test_fetch_records_unions_registers():
    """The full vocabulary is the union of the per-type registers."""
    registers = [
        {
            "set": "rwc_ministeries_compleet",
            "version": 6,
            "default_soort": "ministerie",
        },
        {"set": "rwc_provincies_compleet", "version": 1, "default_soort": "provincie"},
    ]
    fetcher = _fetcher_with_payloads(
        {
            "rwc_ministeries_compleet": _MINISTRIES,
            "rwc_provincies_compleet": _PROVINCIES,
        },
        registers=registers,
    )

    records = await fetcher.fetch_records()
    codes = {r["organisatiecode"] for r in records}
    assert codes == {"mnre1034", "mnre1058", "pv20"}
    prov = next(r for r in records if r["organisatiecode"] == "pv20")
    assert prov["soort"] == "provincie"


@pytest.mark.asyncio
async def test_fetch_records_failsoft_on_unreachable_register():
    """A 404/unreachable register contributes nothing — others still load."""
    registers = [
        {
            "set": "rwc_ministeries_compleet",
            "version": 6,
            "default_soort": "ministerie",
        },
        {"set": "rwc_gemeenten_compleet", "version": 5, "default_soort": "gemeente"},
    ]
    # Only ministries served; gemeenten 404s.
    fetcher = _fetcher_with_payloads(
        {"rwc_ministeries_compleet": _MINISTRIES},
        registers=registers,
    )
    records = await fetcher.fetch_records()
    assert {r["organisatiecode"] for r in records} == {"mnre1034", "mnre1058"}


@pytest.mark.asyncio
async def test_fetch_records_dedupes_across_registers():
    """A code appearing in two registers yields exactly one record."""
    registers = [
        {
            "set": "rwc_ministeries_compleet",
            "version": 6,
            "default_soort": "ministerie",
        },
        {"set": "rwc_provincies_compleet", "version": 1, "default_soort": "provincie"},
    ]
    fetcher = _fetcher_with_payloads(
        {
            "rwc_ministeries_compleet": _MINISTRIES,
            # provincies register accidentally repeats BZK
            "rwc_provincies_compleet": _PROVINCIES + [_MINISTRIES[1]],
        },
        registers=registers,
    )
    records = await fetcher.fetch_records()
    codes = [r["organisatiecode"] for r in records]
    assert codes.count("mnre1034") == 1
