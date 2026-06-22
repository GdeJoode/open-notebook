"""TOOI vocabulary provider — Dutch government organisations (K.4).

TOOI (the *Thesaurus en Ontologie Overheidsinformatie*) is the authoritative
Dutch-government linked-data registry. Each organisation has a stable URI under
``https://identifier.overheid.nl/tooi/id/...`` resolvable as RDF/Turtle with the
fields this provider keys off (verified live 2026-06-22):

    tooiont:organisatiecode        "mnre1034"
    tooiont:afkorting              "BZK"
    tooiont:officieleNaamExclSoort "Binnenlandse Zaken en Koninkrijksrelaties"
    tooiont:officieleNaamInclSoort "ministerie van Binnenlandse Zaken en Koninkrijksrelaties"

so e.g. ``mnre1034`` →
``https://identifier.overheid.nl/tooi/id/ministerie/mnre1034`` is the BZK URI.

Load model (decision K-D2 — see escalations.md)
-----------------------------------------------
``refresh()`` ingests a **bulk file** of TOOI organisation records (the
documented JSON shape below) into ``reference_entity`` (``source_vocabulary
= "tooi"``) and stamps ``last_validated``. A bundled representative seed
(``_SEED_RECORDS``, the verified ministries) is the default source so the
provider works out-of-the-box and the tests are deterministic without a network.
The exact production bulk-download URL + refresh cadence is the open K-D2 item;
when confirmed, point ``refresh(source=<url-or-path>)`` at it — the ingest format
and the lookup are unchanged.

The bulk file format (``tooi_organisations.json``) is a JSON list of records::

    [
      {
        "organisatiecode": "mnre1034",
        "afkorting": "BZK",
        "naam_excl_soort": "Binnenlandse Zaken en Koninkrijksrelaties",
        "naam_incl_soort": "ministerie van Binnenlandse Zaken en Koninkrijksrelaties",
        "soort": "ministerie",
        "entity_type": "organization"
      },
      ...
    ]

``lookup`` resolves a surface form (after the K.1/K.2 normalizer) against the
loaded rows by canonical name OR alias (abbreviation / incl-soort form).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from shared.utils.name_normalizer import normalize_entity_name
from shared.vocabulary.provider import VocabMatch

_TOOI_BASE_URI = "https://identifier.overheid.nl/tooi/id"

# Verified-live representative seed (subset of the ~25 core organisations).
# Each is the real TOOI organisatiecode + officiële naam fetched from
# identifier.overheid.nl on 2026-06-22. This is the deterministic default the
# tests and a fresh install use; refresh(source=...) replaces it with the full
# bulk dump once the K-D2 URL is confirmed.
_SEED_RECORDS: List[Dict[str, str]] = [
    {"organisatiecode": "mnre1010", "afkorting": "AZ", "naam_excl_soort": "Algemene Zaken", "naam_incl_soort": "ministerie van Algemene Zaken", "soort": "ministerie"},
    {"organisatiecode": "mnre1034", "afkorting": "BZK", "naam_excl_soort": "Binnenlandse Zaken en Koninkrijksrelaties", "naam_incl_soort": "ministerie van Binnenlandse Zaken en Koninkrijksrelaties", "soort": "ministerie"},
    {"organisatiecode": "mnre1013", "afkorting": "BZ", "naam_excl_soort": "Buitenlandse Zaken", "naam_incl_soort": "ministerie van Buitenlandse Zaken", "soort": "ministerie"},
    {"organisatiecode": "mnre1018", "afkorting": "Def", "naam_excl_soort": "Defensie", "naam_incl_soort": "ministerie van Defensie", "soort": "ministerie"},
    {"organisatiecode": "mnre1045", "afkorting": "EZK", "naam_excl_soort": "Economische Zaken en Klimaat", "naam_incl_soort": "ministerie van Economische Zaken en Klimaat", "soort": "ministerie"},
    {"organisatiecode": "mnre1130", "afkorting": "IenW", "naam_excl_soort": "Infrastructuur en Waterstaat", "naam_incl_soort": "ministerie van Infrastructuur en Waterstaat", "soort": "ministerie"},
    {"organisatiecode": "mnre1150", "afkorting": "LNV", "naam_excl_soort": "Landbouw, Natuur en Voedselkwaliteit", "naam_incl_soort": "ministerie van Landbouw, Natuur en Voedselkwaliteit", "soort": "ministerie"},
    {"organisatiecode": "mnre1153", "afkorting": "LVVN", "naam_excl_soort": "Landbouw, Visserij, Voedselzekerheid en Natuur", "naam_incl_soort": "ministerie van Landbouw, Visserij, Voedselzekerheid en Natuur", "soort": "ministerie"},
    {"organisatiecode": "mnre1109", "afkorting": "OCW", "naam_excl_soort": "Onderwijs, Cultuur en Wetenschap", "naam_incl_soort": "ministerie van Onderwijs, Cultuur en Wetenschap", "soort": "ministerie"},
    {"organisatiecode": "mnre1025", "afkorting": "VWS", "naam_excl_soort": "Volksgezondheid, Welzijn en Sport", "naam_incl_soort": "ministerie van Volksgezondheid, Welzijn en Sport", "soort": "ministerie"},
]


def _record_uri(record: Dict[str, str]) -> str:
    soort = record.get("soort", "ministerie")
    code = record["organisatiecode"]
    return f"{_TOOI_BASE_URI}/{soort}/{code}"


def _record_aliases(record: Dict[str, str]) -> List[str]:
    """All recognised surface forms for a TOOI organisation (deduped, ordered)."""
    candidates = [
        record.get("naam_incl_soort"),
        record.get("afkorting"),
        record.get("naam_excl_soort"),
    ]
    seen: List[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.append(c)
    return seen


class TOOIProvider:
    """Loads TOOI organisations into ``reference_entity`` and looks them up.

    The provider holds a repository handle (the ``ReferenceEntityRepository``,
    duck-typed so unit tests can inject an in-memory fake) and an optional bulk
    source. ``lookup`` queries the repo; ``refresh`` ingests the source.
    """

    name = "tooi"

    def __init__(
        self,
        repository: Any,
        *,
        source: Optional[str] = None,
    ) -> None:
        """Args:

        repository: A ``ReferenceEntityRepository``-shaped object exposing
            ``async bulk_load(records)``, ``async lookup_by_name(name, source)``
            and ``async lookup_by_alias(alias, source)``.
        source: Optional path to a bulk ``tooi_organisations.json`` file. When
            ``None``, ``refresh`` ingests the bundled verified seed.
        """
        self._repo = repository
        self._source = source

    # ------------------------------------------------------------------
    # refresh — bulk ingest into reference_entity (idempotent)
    # ------------------------------------------------------------------

    async def refresh(self, source: Optional[str] = None) -> int:
        """Ingest TOOI organisations into ``reference_entity``. Returns rows upserted.

        Idempotent: each record upserts on the ``(canonical_name,
        source_vocabulary)`` key (migration 41's UNIQUE index), so a second
        refresh over the same data creates no duplicates and only refreshes
        ``last_validated``.
        """
        records = self._read_source(source or self._source)
        rows = [self._to_reference_row(r) for r in records]
        loaded = await self._repo.bulk_load(rows)
        logger.info("TOOI refresh: ingested {n} organisations", n=loaded)
        return loaded

    @staticmethod
    def _read_source(source: Optional[str]) -> List[Dict[str, str]]:
        if source is None:
            return list(_SEED_RECORDS)
        path = Path(source)
        if not path.exists():
            logger.warning(
                "TOOI bulk source not found ({p}) — falling back to bundled seed",
                p=source,
            )
            return list(_SEED_RECORDS)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("TOOI bulk source unreadable ({p}): {e}", p=source, e=exc)
            return list(_SEED_RECORDS)
        if not isinstance(data, list):
            logger.error("TOOI bulk source is not a JSON list ({p})", p=source)
            return list(_SEED_RECORDS)
        return data

    def _to_reference_row(self, record: Dict[str, str]) -> Dict[str, Any]:
        canonical = record.get("naam_excl_soort") or record.get("afkorting") or ""
        return {
            "canonical_name": canonical,
            "entity_type": record.get("entity_type", "organization"),
            "source_vocabulary": self.name,
            "external_uri": _record_uri(record),
            "external_id": record["organisatiecode"],
            "aliases": _record_aliases(record),
            "properties": {"soort": record.get("soort", "ministerie")},
        }

    # ------------------------------------------------------------------
    # lookup — surface form -> candidate reference rows
    # ------------------------------------------------------------------

    async def lookup(self, name: str, entity_type: str) -> List[VocabMatch]:
        """Resolve a surface form against the loaded TOOI rows.

        Tries an exact canonical-name match first (high confidence), then an
        alias match (abbreviation / incl-soort form, slightly lower confidence).
        Returns ``[]`` on no hit or any repo error (fail-soft). NEVER raises.
        """
        normalized = normalize_entity_name(name)
        if not normalized:
            return []
        try:
            by_name = await self._repo.lookup_by_name(name, source=self.name)
            if by_name:
                return self._rows_to_matches(by_name, confidence=0.99)
            by_alias = await self._repo.lookup_by_alias(name, source=self.name)
            return self._rows_to_matches(by_alias, confidence=0.9)
        except Exception as exc:
            logger.warning("TOOI lookup failed for {n!r} (no-match): {e}", n=name, e=exc)
            return []

    def _rows_to_matches(
        self, rows: Sequence[Dict[str, Any]], *, confidence: float
    ) -> List[VocabMatch]:
        matches: List[VocabMatch] = []
        for row in rows:
            uri = row.get("external_uri")
            if not uri:
                continue
            matches.append(
                VocabMatch(
                    canonical_name=row.get("canonical_name", ""),
                    external_uri=uri,
                    external_id=row.get("external_id") or "",
                    source_vocabulary=self.name,
                    aliases=list(row.get("aliases") or []),
                    confidence=confidence,
                )
            )
        return matches


__all__ = ["TOOIProvider"]
