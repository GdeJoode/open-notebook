"""`cites` (source → source) citation materialization service (Track U.3).

Wires the pure :func:`shared.retrieval.match_corpus_references` precision matcher
into the DB: loads the in-corpus sources' bibliographic projection via the
``SourceRepository``, runs the pure match over the references Track V supplies
(``{source_id: [ParsedReference]}``), then clears + RELATEs the ``cites`` edge
table idempotently for every confident intra-corpus match. External references
(no in-corpus match) are recorded as a COUNT (and logged) — not edged — keeping
the mechanism simple; Track V may later stub them as external nodes.

INFRASTRUCTURE, not live data
=============================
On the current corpus there are 0 intra-corpus citations (U.1) and no reference
input exists yet (Track V is unbuilt), so :meth:`materialize` called with no
references is a clean 0-edge NO-OP — it clears (0), matches (0), creates (0) and
returns, without crashing. The value is the working mechanism + the documented
``ParsedReference`` input interface for the upstream (Track V).

Precision over recall
=====================
The matcher (the pure half) only returns CONFIDENT, unambiguous, non-self
matches — a wrong match fabricates a citation, strictly worse than a missing one.
This service does not relax that: it persists exactly what the matcher returns.
Idempotent regenerator (clear + recreate, like the U.2 ``mentions`` service);
reversible; the canonical ``source`` rows are never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from loguru import logger
from shared.retrieval.cites_matching import (
    ParsedReference,
    SourceRecord,
    match_corpus_references,
)

from surrealdb_service.repositories import SourceRepository


@dataclass(frozen=True)
class MaterializeCitesResult:
    """Outcome of a ``cites`` materialization (telemetry + acceptance).

    Attributes:
        cleared: Edges removed by the clear step (the prior edge set size).
        created: DISTINCT ``cites`` (origin, target) edges persisted this run.
            Counts unique pairs, not raw RELATE calls — two confident matches for
            the same (origin, target) idempotently re-land one edge, so they count
            once (matching the de-duplicated edge table).
        failed: RELATE calls that failed (expected 0; non-zero is a partial
            write worth surfacing).
        external: References that matched NO in-corpus source (the external
            works — counted/logged, not edged; Track V may stub them later).
        ambiguous: References rejected by the precision ambiguity guard (cleared
            the floor but had no clear winner).
        self_citations_skipped: References whose only confident target was their
            own origin source (suppressed — no source cites itself).
        total_references: Total references considered across all sources.
        sources_in_corpus: In-corpus sources the references were matched against.
    """

    cleared: int
    created: int
    failed: int
    external: int
    ambiguous: int
    self_citations_skipped: int
    total_references: int
    sources_in_corpus: int


class CitesMaterializationService:
    """Materialize + read the ``cites`` document↔document citation graph (U.3)."""

    def __init__(self, source_repo: SourceRepository):
        self.source_repo = source_repo

    # Keys under which a Zotero / Docling enrichment might stash bibliographic
    # detail in the FLEXIBLE source metadata objects. Checked case-insensitively
    # across ``metadata`` / ``type_metadata`` / ``external_ids`` (the source table
    # is SCHEMAFULL with no top-level authors/doi column — those would be dropped).
    _AUTHOR_KEYS = ("authors", "author", "creators")
    _DOI_KEYS = ("doi",)

    @classmethod
    def _extract_bibliographic(
        cls, row: Dict[str, Any]
    ) -> "tuple[tuple[str, ...], str | None]":
        """Pull (authors, doi) from a source's flexible metadata objects.

        Scans ``metadata`` / ``type_metadata`` / ``external_ids`` for the author
        and DOI keys (case-insensitive). Returns empty authors / ``None`` doi when
        absent (the current-corpus reality) — the matcher then falls back to the
        title and the precision guard declines a title-only match, which is
        correct.
        """
        objs = [
            row.get("metadata"),
            row.get("type_metadata"),
            row.get("external_ids"),
        ]
        authors: List[str] = []
        doi: str | None = None
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            lowered = {str(k).lower(): v for k, v in obj.items()}
            for ak in cls._AUTHOR_KEYS:
                val = lowered.get(ak)
                if not val:
                    continue
                if isinstance(val, str):
                    authors.append(val)
                elif isinstance(val, (list, tuple)):
                    authors.extend(str(a) for a in val if a)
            if doi is None:
                for dk in cls._DOI_KEYS:
                    val = lowered.get(dk)
                    if val:
                        doi = str(val)
                        break
        return tuple(authors), doi

    @classmethod
    def _to_source_records(cls, rows: List[Dict[str, Any]]) -> List[SourceRecord]:
        """Project raw ``source`` rows into the pure matcher's input type.

        Builds the matcher's ``SourceRecord`` from ``title`` plus the ``authors`` /
        ``doi`` recovered from the flexible metadata objects (see
        :meth:`_extract_bibliographic`). On the current corpus only ``title`` is
        present, so records carry no authors/doi and the title-only precision
        guard declines — the clean 0-edge no-op.
        """
        records: List[SourceRecord] = []
        for row in rows or []:
            sid = row.get("id")
            if sid is None:
                continue
            authors, doi = cls._extract_bibliographic(row)
            records.append(
                SourceRecord(
                    source_id=str(sid),
                    title=str(row.get("title") or ""),
                    authors=authors,
                    doi=doi,
                )
            )
        return records

    async def load_corpus(self) -> List[SourceRecord]:
        """Load the in-corpus source records the matcher resolves references to.

        The read-only half — used both by :meth:`materialize` and by a dry-run /
        inspection. Read-only; no edge writes.
        """
        rows = await self.source_repo.load_source_records()
        return self._to_source_records(rows)

    async def materialize(
        self,
        references_by_source: Dict[str, Sequence[ParsedReference]],
    ) -> MaterializeCitesResult:
        """Clear + rebuild the ``cites`` edges from the matched references.

        Idempotent: clears every existing ``cites`` edge then RELATEs the
        freshly-matched confident set, so a re-run on the SAME references yields
        the SAME edge set with zero duplicates. Reversible — ``cites`` is a
        regenerated projection; the canonical ``source`` rows it matches against
        are untouched.

        Args:
            references_by_source: ``{origin_source_id: [ParsedReference, ...]}`` —
                the per-source reference lists Track V produces. Pass ``{}`` for
                the current-corpus no-op (clears to 0, creates 0).

        Returns:
            A :class:`MaterializeCitesResult` with the cleared/created counts and
            the precision-guard telemetry (external / ambiguous / self).
        """
        sources = await self.load_corpus()

        # Pure precision match — the decision (which references become edges) is
        # made here, with no I/O; this service only persists the result.
        match_result = match_corpus_references(references_by_source, sources)

        cleared = await self.source_repo.clear_cites()

        # De-dup (origin, target) pairs WITHIN this run: SurrealDB ``RELATE`` does
        # NOT collapse a repeated (in, out) — it creates a second edge row with a
        # fresh id. So if Track V lists the same cited work twice in one
        # bibliography (a duplicate entry), two matches for the same pair would
        # otherwise write two identical edges. Skipping an already-persisted pair
        # keeps the edge table one-row-per-(origin, target) — true idempotency —
        # and ``created`` (the size of this set) then matches the edge table.
        # Distinct origins citing the same target stay separate (the KEY is the
        # PAIR, not just the target). ``failed`` counts every failed RELATE.
        persisted_pairs: set[tuple[str, str]] = set()
        failed = 0
        for match in match_result.matches:
            # The citing origin is carried THROUGH the match (set by
            # match_corpus_references), so distinct origins citing the same target
            # with a value-equal reference are attributed correctly — no fragile
            # reverse lookup that a value-equality fallback could mis-resolve.
            origin = match.from_source_id
            if origin is None:
                # Defensive: a corpus match always carries its origin. A missing
                # origin means a single-reference match leaked in without one;
                # skip rather than RELATE a dangling edge with no citing side.
                logger.warning(
                    "cites: matched reference has no origin source, skipping: {r}",
                    r=match.reference.raw_text[:80],
                )
                failed += 1
                continue
            if (origin, match.source_id) in persisted_pairs:
                # Already RELATEd this (origin, target) this run — skip the
                # duplicate so the edge table stays one-row-per-pair.
                continue
            try:
                ok = await self.source_repo.relate_cites(
                    origin,
                    match.source_id,
                    confidence=match.confidence,
                    reference_text=match.reference.raw_text,
                    match_method=match.method,
                )
            except Exception as e:  # one bad edge must not abort the rebuild
                logger.warning(
                    "relate_cites raised for {s}->{t}: {e}",
                    s=origin,
                    t=match.source_id,
                    e=e,
                )
                ok = False
            if ok:
                persisted_pairs.add((origin, match.source_id))
            else:
                failed += 1

        created = len(persisted_pairs)
        logger.info(
            "cites materialized: cleared={cleared} created={created} "
            "failed={failed} external={ext} ambiguous={amb} self={self_} "
            "(refs={refs}, sources={src})",
            cleared=cleared,
            created=created,
            failed=failed,
            ext=match_result.external,
            amb=match_result.ambiguous,
            self_=match_result.self_citations_skipped,
            refs=match_result.total_references,
            src=len(sources),
        )
        return MaterializeCitesResult(
            cleared=cleared,
            created=created,
            failed=failed,
            external=match_result.external,
            ambiguous=match_result.ambiguous,
            self_citations_skipped=match_result.self_citations_skipped,
            total_references=match_result.total_references,
            sources_in_corpus=len(sources),
        )

    async def get_citation_edges(
        self,
        *,
        min_confidence: float = 0.0,
        source_id: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Fetch the materialized document↔document citation edges (U.4 viz).

        Returns the persisted ``cites`` edges with endpoints + confidence + the
        per-edge "why" (the cited reference text + match method), ordered by
        confidence descending. Empty on the current corpus (0 edges).
        """
        edges: List[Dict[str, Any]] = await self.source_repo.load_cites_edges(
            min_confidence=min_confidence, source_id=source_id
        )
        return edges
