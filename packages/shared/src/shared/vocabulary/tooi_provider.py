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

Load model (decision K-D2 — RESOLVED, see escalations.md)
---------------------------------------------------------
``refresh()`` ingests TOOI organisation records into ``reference_entity``
(``source_vocabulary = "tooi"``) and stamps ``last_validated``. Three sources,
tried in priority order, each a fail-soft fallback for the next:

1. **Remote bulk fetch** (the resolved K-D2 source) — when a
   :class:`~shared.vocabulary.tooi_bulk.TOOIBulkFetcher` is attached, the FULL
   government-organisation vocabulary is pulled from the official versioned
   JSON-LD registers under ``repository.officiele-overheidspublicaties.nl``
   through the disciplined HTTP client (timeout + rate-limit + cache + fail-soft).
   See :mod:`shared.vocabulary.tooi_bulk` for the exact URLs/format.
2. **Operator bulk file** — ``refresh(source=<path>)`` reads a
   ``tooi_organisations.json`` dump (the documented JSON shape below) for an
   air-gapped or version-pinned deploy.
3. **Bundled verified seed** (``_SEED_RECORDS``, the real ministries) — the
   always-present floor so a fresh install and the offline tests resolve BZK
   without any network. Used when the remote fetch yields nothing and no file is
   given.

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
from shared.vocabulary.tooi_bulk import TOOIBulkFetcher

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


def _record_aliases(record: Dict[str, Any]) -> List[str]:
    """All recognised surface forms for a TOOI organisation (deduped, ordered).

    Combines the canonical-shape fields (incl-soort / afkorting / excl-soort) with
    any explicit ``aliases`` the source already carries (the bulk fetcher adds the
    org's historical names + abbreviations), preserving order and removing dupes.
    """
    candidates: List[Any] = [
        record.get("naam_incl_soort"),
        record.get("afkorting"),
        record.get("naam_excl_soort"),
    ]
    extra = record.get("aliases")
    if isinstance(extra, (list, tuple)):
        candidates.extend(extra)
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
        bulk_fetcher: Optional[TOOIBulkFetcher] = None,
    ) -> None:
        """Args:

        repository: A ``ReferenceEntityRepository``-shaped object exposing
            ``async bulk_load(records)``, ``async lookup_by_name(name, source)``
            and ``async lookup_by_alias(alias, source)``.
        source: Optional path to a bulk ``tooi_organisations.json`` file. Takes
            priority over the remote fetch when set (operator-pinned dump).
        bulk_fetcher: Optional :class:`TOOIBulkFetcher` that pulls the full
            vocabulary from the live TOOI registers. When attached (and no
            ``source`` file is given) ``refresh`` uses it, falling back to the
            bundled seed if nothing is reachable. ``None`` ⇒ no network; the seed
            (or the file) is the source — this keeps the unit tests offline.
        """
        self._repo = repository
        self._source = source
        self._bulk_fetcher = bulk_fetcher

    # ------------------------------------------------------------------
    # refresh — bulk ingest into reference_entity (idempotent)
    # ------------------------------------------------------------------

    async def refresh(self, source: Optional[str] = None) -> int:
        """Ingest TOOI organisations into ``reference_entity``. Returns rows upserted.

        Source priority (each fails soft into the next): an explicit/operator
        ``source`` file > the attached remote bulk fetcher > the bundled seed.

        Idempotent: each record upserts on the ``(canonical_name,
        source_vocabulary)`` key (migration 41's UNIQUE index), so a second
        refresh over the same data creates no duplicates and only refreshes
        ``last_validated``. A code that appears under two surface forms is
        deduped (by ``external_id``) before load so one org is never split into
        two reference rows.
        """
        records = await self._resolve_records(source or self._source)
        rows = self._dedupe_rows(self._to_reference_row(r) for r in records)
        loaded = int(await self._repo.bulk_load(rows))
        logger.info("TOOI refresh: ingested {n} organisations", n=loaded)
        return loaded

    async def _resolve_records(self, source: Optional[str]) -> List[Dict[str, str]]:
        """Pick the highest-priority reachable source. Never raises."""
        if source is not None:
            return self._read_file_source(source)
        if self._bulk_fetcher is not None:
            try:
                fetched = await self._bulk_fetcher.fetch_records()
            except Exception as exc:  # defensive — the fetcher is fail-soft already
                logger.error("TOOI remote bulk fetch raised ({e}) — using seed", e=exc)
                fetched = []
            if fetched:
                return fetched
            logger.warning("TOOI remote bulk fetch empty — falling back to seed")
        return list(_SEED_RECORDS)

    @staticmethod
    def _dedupe_rows(rows: Any) -> List[Dict[str, Any]]:
        """Collapse rows sharing an ``external_id`` (keep first, union aliases)."""
        by_id: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []
        for row in rows:
            ext_id = row.get("external_id") or row.get("canonical_name") or ""
            existing = by_id.get(ext_id)
            if existing is None:
                by_id[ext_id] = row
                order.append(ext_id)
                continue
            for alias in row.get("aliases") or []:
                if alias not in existing["aliases"]:
                    existing["aliases"].append(alias)
        return [by_id[k] for k in order]

    @staticmethod
    def _read_file_source(source: str) -> List[Dict[str, str]]:
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

    def _to_reference_row(self, record: Dict[str, Any]) -> Dict[str, Any]:
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
