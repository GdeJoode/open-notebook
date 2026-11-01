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

Load model (F2-28)
------------------
``refresh()`` ingests TOOI organisation records into ``reference_entity``
(``source_vocabulary = "tooi"``) and stamps ``last_validated``. The records come
from the caller: an attached fetcher (``shared.vocabulary.referentiegegevens``
hands it the registers from the inventory, SHA-256-checked) or a bulk file
(``refresh(source=<path>)``). There is NO fallback: no source, an empty fetch or
an unreadable file raises :class:`TOOIBronOntbreekt` and nothing is written. The
old fallback to a ten-ministry seed overwrote a full register with less (F2-28
review: aliases and end date of mnre1150 lost after a refresh without internet).

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

``lookup`` resolves a surface form against the loaded rows by lookup key — the
``normalize_alias_key`` of the canonical name and every alias (D2, migration 81) —
and, when the whole form misses, through the names it decomposes into (D1).
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from shared.utils.name_normalizer import normalize_entity_name
from shared.utils.nl_normalization import normalize_alias_key
from shared.utils.surface_parts import parse_surface
from shared.vocabulary.provider import VocabMatch

_TOOI_BASE_URI = "https://identifier.overheid.nl/tooi/id"

#: D1: the confidence of a match only SOME parts of a decorated name support.
#: Below ``VocabularyReconciler``'s default link threshold (0.85) on purpose: such
#: a match is a candidate for a curator, never a link. A caller that lowers the
#: threshold below this makes one-sided evidence link — keep them apart.
ONE_SIDED_CONFIDENCE = 0.8


class TOOIBronOntbreekt(RuntimeError):
    """No usable TOOI source: nothing to load, and nothing is written."""


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
        bulk_fetcher: Optional[Any] = None,
    ) -> None:
        """Args:

        repository: A ``ReferenceEntityRepository``-shaped object exposing
            ``async bulk_load(records)`` and ``async lookup_by_key(key, source)``
            (D2, migration 81).
        source: Optional path to a bulk ``tooi_organisations.json`` file. Takes
            priority over the remote fetch when set (operator-pinned dump).
        bulk_fetcher: Optional object with ``async fetch_records()`` (e.g. the
            inventory registers, ``shared.vocabulary.referentiegegevens``). Used
            when no ``source`` file is given.
        """
        self._repo = repository
        self._source = source
        self._bulk_fetcher = bulk_fetcher

    # ------------------------------------------------------------------
    # refresh — bulk ingest into reference_entity (idempotent)
    # ------------------------------------------------------------------

    async def refresh(self, source: Optional[str] = None) -> int:
        """Ingest TOOI organisations into ``reference_entity``. Returns rows upserted.

        Source: an explicit/operator ``source`` file, else the attached fetcher.
        Raises :class:`TOOIBronOntbreekt` (before writing anything) when neither
        yields records.

        Idempotent: each record upserts on the ``(external_id,
        source_vocabulary)`` key (migration 57's UNIQUE index — the org's stable
        code, NOT its display name), so a second refresh over the same data
        creates no duplicates and only refreshes ``last_validated``.

        The pre-load ``_dedupe_rows`` keys on the SAME ``external_id`` as the
        upsert, so the two stay consistent: a code that appears under two surface
        forms collapses to ONE row (aliases unioned), but two DISTINCT codes that
        happen to share a display name — e.g. the two "Bergen" gemeenten (gm0373
        / gm0893) — persist as TWO rows. Keying on the display name (the old
        migration-41 key) silently merged those distinct orgs, which let the
        reconciler auto-link "Bergen" to the wrong URI; the external_id key keeps
        the ambiguity so the reconciler refuses to link.

        Returns the number of rows actually persisted (``bulk_load`` counts
        successful upserts of the DEDUPED row set, so the count reflects the rows
        in ``reference_entity`` — it is not inflated by the pre-dedupe duplicates).
        """
        records = await self._resolve_records(source or self._source)
        rows = self._dedupe_rows(self._to_reference_row(r) for r in records)
        loaded = int(await self._repo.bulk_load(rows))
        logger.info("TOOI refresh: ingested {n} organisations", n=loaded)
        return loaded

    async def _resolve_records(self, source: Optional[str]) -> List[Dict[str, str]]:
        """The records to load; raises :class:`TOOIBronOntbreekt` when there are none."""
        if source is not None:
            return self._read_file_source(source)
        if self._bulk_fetcher is None:
            raise TOOIBronOntbreekt("no TOOI source (no file and no fetcher)")
        fetched = await self._bulk_fetcher.fetch_records()
        if not fetched:
            raise TOOIBronOntbreekt("TOOI fetch returned no records")
        return fetched

    @staticmethod
    def _dedupe_rows(rows: Any) -> List[Dict[str, Any]]:
        """Collapse rows sharing an ``external_id`` (keep first, union aliases).

        Keys on ``external_id`` — the SAME key the repository upsert uses
        (migration 57). Two rows for ONE org code collapse; two DISTINCT codes
        that share a display name (the two "Bergen" gemeenten) do NOT — they stay
        as two rows so the reconciler sees the ambiguity.
        """
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
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise TOOIBronOntbreekt(f"TOOI bulk source unreadable ({source}): {exc}") from exc
        if not isinstance(data, list) or not data:
            raise TOOIBronOntbreekt(f"TOOI bulk source is not a non-empty JSON list ({source})")
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
            "lookup_keys": sorted(
                {k for k in (normalize_alias_key(n) for n in [canonical, *_record_aliases(record)]) if k}
            ),
            "properties": {
                "soort": record.get("soort", "ministerie"),
                # In `properties` (FLEXIBLE): reference_entity is SCHEMAFULL and a
                # top-level field would be dropped without a migration.
                "current_names": [
                    n for n in dict.fromkeys(
                        [canonical, record.get("afkorting"), record.get("naam_incl_soort")]
                    ) if n
                ],
                "begindatum": record.get("begindatum") or "",
                "einddatum": record.get("einddatum") or "",
            },
        }

    # ------------------------------------------------------------------
    # lookup — surface form -> candidate reference rows
    # ------------------------------------------------------------------

    async def lookup(self, name: str, entity_type: str) -> List[VocabMatch]:
        """Resolve a surface form against the loaded TOOI rows.

        Tries an exact canonical-name match first (high confidence), then an
        alias match (abbreviation / incl-soort form, slightly lower confidence).
        When the name as written misses, the names it decomposes into are tried
        (D1, :meth:`_lookup_parts`). Returns ``[]`` on no hit or any repo error
        (fail-soft). NEVER raises.
        """
        normalized = normalize_entity_name(name)
        if not normalized:
            return []
        try:
            whole = await self._lookup_one(name)
            if whole:
                return whole
            return await self._lookup_parts(name, entity_type)
        except Exception as exc:
            logger.warning("TOOI lookup failed for {n!r} (no-match): {e}", n=name, e=exc)
            return []

    async def _lookup_one(self, name: str) -> List[VocabMatch]:
        """One string against the register, by lookup key (D2, migration 81).

        The key is :func:`normalize_alias_key` — case, whitespace, trailing
        punctuation, a leading article and curated spelling variants — the same
        function the writer applied to the canonical name and every alias. A row
        whose canonical name has this key is a name hit (0.99); a row that has it
        only through an alias is an alias hit (0.9), as K.4 ranked them.

        **Every row that carries the key comes back.** K.4 returned name hits and
        dropped alias hits when both existed, which decides between two
        ORGANISATIONS on whether the string is one's name or the other's alias.
        The D2 dry run caught what that does once labels are current:
        `Economische Zaken` is the name of mnre1040 (ended 2010) and a historical
        name of mnre1045 (EZK), and the ranking linked the corpus's 2024
        `Economische Zaken (Ministerie)` to the one that ended fourteen years
        earlier. Two organisations under one key is an ambiguity, and the
        reconciler's precision guard refuses it with both recorded.
        """
        key = normalize_alias_key(name)
        if not key:
            return []
        rows = await self._repo.lookup_by_key(key, source=self.name)
        return [
            m
            for row in rows
            for m in self._rows_to_matches(
                [row],
                confidence=0.99
                if normalize_alias_key(row.get("canonical_name", "")) == key
                else 0.9,
            )
        ]

    async def _lookup_parts(self, name: str, entity_type: str) -> List[VocabMatch]:
        """D1: when the name as written misses, try the names it decomposes into.

        ``VRO (Volkshuisvesting en Ruimtelijke Ordening)`` is in no register, but
        both of its parts are. Every part is looked up with the same exact rule,
        and the outcome depends on how many parts carry evidence:

        * **every part hits, and they agree** → the agreed match, at the part's
          own confidence, so the reconciler can link it;
        * **the parts that hit disagree** → every match, so the reconciler sees
          more than one and refuses, recording them as candidates;
        * **some part misses** → the matches at :data:`ONE_SIDED_CONFIDENCE`,
          below the reconciler's link threshold: recorded for a curator, never
          linked (owner decision, 2026-09-26). A miss is not silence — D1 review
          found `OV (Openbaar Vervoer)` linking to the province Overijssel on its
          abbreviation alone, the gloss naming something no register holds.

        A blocked form — an organ of something, or several referents in one
        string — carries no names and so resolves to nothing: `De
        Staatssecretaris van Infrastructuur en Waterstaat (IenW)` is an office.
        """
        parts = parse_surface(name, entity_type)
        pieces = parts.names[1:]
        hits: List[List[VocabMatch]] = []
        for part in pieces:
            found = await self._lookup_one(part)
            if found:
                hits.append(found)
        if not hits:
            return []
        agreed = set.intersection(*({m.external_uri for m in h} for h in hits))
        flat = [m for h in hits for m in h]
        if agreed:
            flat = [m for m in flat if m.external_uri in agreed]
        one_sided = len(hits) < len(pieces)
        # One match per URI, the most confident — the reconciler dedups the same
        # way, but a provider that returns its own duplicates is harder to read.
        best: Dict[str, VocabMatch] = {}
        for m in flat:
            if m.external_uri not in best or m.confidence > best[m.external_uri].confidence:
                best[m.external_uri] = m
        rule = f"{parts.shape}{' (one-sided)' if one_sided else ''}: " + " + ".join(pieces)
        return [
            replace(
                m,
                matched_on=rule,
                confidence=min(m.confidence, ONE_SIDED_CONFIDENCE) if one_sided else m.confidence,
            )
            for m in best.values()
        ]

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
                    valid_until=str((row.get("properties") or {}).get("einddatum") or ""),
                    current_names=list((row.get("properties") or {}).get("current_names") or []),
                )
            )
        return matches


__all__ = ["TOOIProvider"]
