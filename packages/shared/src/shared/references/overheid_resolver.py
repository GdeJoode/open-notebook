"""overheid.nl KOOP SRU work resolver (Track V.4).

Routes when the reference looks like a Dutch Kamerstuk / policy document
(detected upstream by :func:`shared.references.work_resolver.looks_like_nl_policy`).
Queries the KOOP SRU endpoint (``https://repository.overheid.nl/sru``), which
returns **SRU/XML** — fetched through the shared fail-soft client's
:meth:`get_text` and parsed with the stdlib XML parser.

Precision signal (in order of strength):

* **Dossier identifier** — a Kamerstuk citation carries a dossiernummer (e.g.
  "35 300"); when the SRU record's identifier contains those digits the match is
  strong (an official dossier id is near-unique), confidence 0.95.
* **Title overlap** — otherwise the returned title must clear the shared
  token-overlap floor against the reference text; below it the leg falls through.

Live-confirmed against KOOP SRU (2026-07-24)
============================================
Verified the record structure and query grammar against the live endpoint; three
corrections followed:

* **Query grammar** — ``cql.textAndIndexes="a b"`` is a PHRASE match (a multi-word
  citation string then matches ~nothing: "31305 mobiliteit" → 0 records). The
  ``all`` relation ANDs the words instead (→ 892 records), which is what we want.
* **Year** — records expose ``dcterms:date`` / ``available`` / ``modified``, NOT
  ``issued``; reading only ``issued`` always yielded ``year=None``. We now fall
  through those in order.
* **Dossiernummer** — a record carries a dedicated ``<dossiernummer>`` element
  (e.g. ``31305``). The dossier match now prefers it, keeping the
  digits-in-``identifier`` heuristic (true for ``kst-…`` ids) as a fallback.
"""

from __future__ import annotations

import re
from typing import Optional
from xml.etree import ElementTree as ET

from loguru import logger

from shared.references.work_resolver import (
    MIN_TITLE_OVERLAP,
    ResolvedWork,
    title_overlap,
)
from shared.retrieval.cites_matching import ParsedReference
from shared.vocabulary.http_client import VocabularyHTTPClient

_SRU_URL = "https://repository.overheid.nl/sru"

#: Confidence when the reference's dossiernummer is found in the record id.
_DOSSIER_CONFIDENCE = 0.95

# A Kamerstuk dossiernummer: 5 digits, optionally written "35 300" / "35-300".
_DOSSIER = re.compile(r"\b(\d{2})[\s\-]?(\d{3})\b")
#: Non-term characters to strip when building the CQL query. KOOP SRU expects
#: CQL with an index (``cql.textAndIndexes="..."``); passing the raw citation
#: string (commas, slashes, "nr.") is not valid CQL and returns nothing.
_CQL_TERM = re.compile(r"[^0-9A-Za-zÀ-ſ]+")


class OverheidResolver:
    """Resolve a Dutch policy-document reference against the KOOP SRU API."""

    name = "overheid"

    def __init__(
        self,
        *,
        http_client: Optional[VocabularyHTTPClient] = None,
        rows: int = 5,
    ) -> None:
        self._rows = rows
        self._client = http_client or VocabularyHTTPClient(
            user_agent="open-notebook/0.1 (https://github.com/open-notebook)",
            timeout=10.0,
            min_interval=1.0,
            cache_ttl=3600.0,
        )

    async def resolve(self, ref: ParsedReference) -> Optional[ResolvedWork]:
        query = self._build_cql_query(ref)
        if not query:
            return None
        body = await self._client.get_text(
            _SRU_URL,
            params={
                "operation": "searchRetrieve",
                "version": "2.0",
                "query": query,
                "maximumRecords": str(self._rows),
            },
        )
        if not body:
            return None
        return self._parse_response(body, ref)

    def _build_cql_query(self, ref: ParsedReference) -> Optional[str]:
        """Build a valid KOOP SRU CQL query from a reference.

        Terms are alphanumeric tokens (length > 2) from the title/raw text, capped
        to keep the search focused; the dossiernummer (the strongest signal) is
        naturally among them. Returns a ``cql.textAndIndexes="..."`` string, or
        ``None`` when there is nothing to search on.
        """
        raw = (ref.title or ref.raw_text or "").strip()
        if not raw:
            return None
        terms = [t for t in _CQL_TERM.sub(" ", raw).split() if len(t) > 2][:8]
        if not terms:
            return None
        # The ``all`` relation ANDs the words. Quoting them instead
        # (``textAndIndexes="a b"``) is an EXACT-PHRASE match that finds ~nothing
        # for a citation string — confirmed live (phrase → 0 records, all → 892).
        return f'cql.textAndIndexes all "{" ".join(terms)}"'

    def _parse_response(
        self, body: str, ref: ParsedReference
    ) -> Optional[ResolvedWork]:
        try:
            root = ET.fromstring(body)
        except ET.ParseError as exc:
            logger.warning("overheid: could not parse SRU response: {exc}", exc=exc)
            return None

        ref_dossier = self._reference_dossier(ref.raw_text)
        for record in self._iter_local(root, "recordData"):
            resolved = self._record_to_resolved(record, ref, ref_dossier)
            if resolved is not None:
                return resolved
        return None

    def _record_to_resolved(
        self,
        record: ET.Element,
        ref: ParsedReference,
        ref_dossier: Optional[str],
    ) -> Optional[ResolvedWork]:
        title = self._find_local_text(record, "title")
        identifier = self._find_local_text(record, "identifier")
        if not title and not identifier:
            return None
        doc_type = self._find_local_text(record, "type")
        # KOOP records carry the date as dcterms:date/available/modified, not
        # ``issued`` — try them in order of preference (confirmed live).
        year = self._parse_year(
            self._find_local_text(record, "issued")
            or self._find_local_text(record, "date")
            or self._find_local_text(record, "available")
            or self._find_local_text(record, "modified")
        )
        uri = self._find_local_text(record, "preferredUrl") or identifier

        # Strongest signal: the reference's dossiernummer matches the record's.
        # Prefer the dedicated <dossiernummer> element; fall back to digits in the
        # identifier (holds for ``kst-<dossier>-<nr>`` ids, not for ``blg-…``).
        record_dossier = self._find_local_text(record, "dossiernummer")
        if ref_dossier and (
            (record_dossier and ref_dossier == self._digits(record_dossier))
            or (identifier and ref_dossier in self._digits(identifier))
        ):
            return self._build(
                title or identifier, identifier, uri, doc_type, year,
                confidence=_DOSSIER_CONFIDENCE,
            )

        # Otherwise require a real title overlap against the reference text.
        query_text = ref.title or ref.raw_text or ""
        overlap = title_overlap(query_text, title) if title else 0.0
        if overlap >= MIN_TITLE_OVERLAP:
            return self._build(
                title, identifier, uri, doc_type, year,
                confidence=round(overlap, 4),
            )
        return None

    def _build(
        self,
        title: str,
        identifier: str,
        uri: str,
        doc_type: str,
        year: Optional[int],
        *,
        confidence: float,
    ) -> ResolvedWork:
        return ResolvedWork(
            title=title,
            source=self.name,
            method="sru",
            confidence=confidence,
            doi=None,
            year=year,
            venue=doc_type or "overheid.nl",
            external_id=identifier,
            external_uri=uri,
        )

    # -- dossier / digits helpers ------------------------------------------

    @staticmethod
    def _reference_dossier(raw_text: str) -> Optional[str]:
        m = _DOSSIER.search(raw_text or "")
        return f"{m.group(1)}{m.group(2)}" if m else None

    @staticmethod
    def _digits(text: str) -> str:
        return re.sub(r"\D", "", text or "")

    @staticmethod
    def _parse_year(issued: str) -> Optional[int]:
        if issued and len(issued) >= 4 and issued[:4].isdigit():
            return int(issued[:4])
        return None

    # -- namespace-robust XML traversal ------------------------------------
    #
    # KOOP SRU mixes several namespaces (sru, gzd, dcterms). Rather than pin
    # every prefix (which a live schema change could break), traverse by LOCAL
    # element name so the parser tolerates namespace/version drift.

    @staticmethod
    def _local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    def _iter_local(self, root: ET.Element, name: str):
        for el in root.iter():
            if self._local(el.tag) == name:
                yield el

    def _find_local_text(self, element: ET.Element, name: str) -> str:
        for el in element.iter():
            if self._local(el.tag) == name and el.text and el.text.strip():
                return el.text.strip()
        return ""

    async def aclose(self) -> None:
        await self._client.aclose()


__all__ = ["OverheidResolver"]
