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
    """In-memory stand-in keyed on (canonical_name, source_vocabulary).

    Mirrors the real repo's UNIQUE (canonical_name, source_vocabulary) key from
    migration 41 so we can assert refresh idempotency (no duplicate rows).
    """

    def __init__(self) -> None:
        self.rows: Dict[tuple, Dict[str, Any]] = {}

    async def bulk_load(self, records: List[Dict[str, Any]]) -> int:
        n = 0
        for r in records:
            key = (r["canonical_name"], r["source_vocabulary"])
            self.rows[key] = dict(r)
            n += 1
        return n

    async def lookup_by_name(
        self, name: str, source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        out = []
        for (cn, src), row in self.rows.items():
            if cn == name and (source is None or src == source):
                out.append(row)
        return out

    async def lookup_by_alias(
        self, alias: str, source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        out = []
        for (_cn, src), row in self.rows.items():
            if source is not None and src != source:
                continue
            if alias in (row.get("aliases") or []):
                out.append(row)
        return out


@pytest.mark.asyncio
async def test_refresh_loads_seed():
    repo = FakeReferenceRepo()
    provider = TOOIProvider(repo)
    loaded = await provider.refresh()
    assert loaded >= 10  # the verified ministry seed
    assert ("Binnenlandse Zaken en Koninkrijksrelaties", "tooi") in repo.rows


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
