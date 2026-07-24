"""Reference-extraction orchestration (Track V.5 / G.3) — the GROBID → U.3 bridge.

Track G.2 built the style-agnostic producer (:class:`GrobidReferenceService`: a
source PDF -> ``List[ParsedReference]`` via GROBID's CRF models). Track U.3 built
the ``cites`` materializer (``CitesMaterializationService.materialize``: a
``{source_id: [ParsedReference]}`` map -> the confident intra-corpus
``source -> cites -> source`` edge set). This service (G.3) connects the two: it
reads each source's ORIGINAL PDF (``source.asset.file_path``), runs GROBID, and
hands the assembled corpus map to U.3's EXISTING entry point.

GROBID is the sole engine (decision G-D2)
=========================================
The earlier V.1/V.2/V.3 heuristic (region-locator + segmenter + entry-parser over
docling chunks) had to *understand* each bibliography's formatting to slice and
parse it. GROBID's trained models already own that knowledge, so the heuristic is
gone: the ONLY reference engine is GROBID, and its input is the source PDF
(decision G-D1), not chunks. A source with no PDF (URL/note sources) yields ``[]``.

Why the whole corpus, not one source
====================================
U.3's ``materialize`` is a WHOLE-CORPUS regenerator: it ``DELETE cites`` (all
edges) then rebuilds from the map it is given (``SourceRepository.clear_cites``
is a bare ``DELETE cites``). Feeding it a single source's references would wipe
every OTHER source's edges. References are not persisted anywhere (the producer
is stateless), so the only correct, non-destructive way to drive U.3's existing
public entry point is to (re)assemble the full corpus map — extract references
for every source — and materialize once. That is what :meth:`materialize_corpus`
does; the per-source :meth:`extract_source_references` is its inner unit.

Idempotent + non-destructive
============================
Because U.3 is clear-before-relate over the whole corpus, re-running the pass on
an unchanged corpus yields the identical edge set with zero duplicates — no
second dedup is added here (the constraint). The canonical ``source`` rows are
never mutated; ``cites`` is a regenerated projection U.3 owns.

External resolution (opt-in, offline by default)
================================================
When ``resolve_external`` is set the pass additionally routes each extracted
reference through the V.4 :class:`ResolverCascade` (OpenAlex / Crossref / …) and
counts the confident external resolutions. This is OFF by default to keep the
core path free of extra network round-trips; the cascade fails soft per leg (a
network error is a no-match, never a raise). Resolution runs BEFORE
materialization and is independent of it — the cascade is V.4's external-authority
resolver, whereas the ``cites`` edges are U.3's separate intra-corpus concern;
resolved works are a count today (persisting a ``ResolvedWork`` into the graph is
a later phase).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from loguru import logger
from shared.references.grobid_reference_service import GrobidReferenceService
from shared.references.work_resolver import ResolverCascade
from shared.retrieval.cites_matching import ParsedReference
from surrealdb_service.repositories import SourceRepository

from app_main.services.cites_materialization_service import (
    CitesMaterializationService,
)


@dataclass(frozen=True)
class ReferenceMaterializationSummary:
    """Outcome of a G.3 corpus reference pass (telemetry + acceptance).

    Attributes:
        sources_scanned: Sources enumerated from the corpus (the candidates the
            pass extracted references from).
        sources_with_references: Sources that yielded at least one reference (a
            PDF was present and GROBID parsed a bibliography).
        refs_extracted: Total references extracted across all sources — the size
            of the map handed to U.3.
        edges_materialized: Confident intra-corpus ``cites`` edges U.3 created
            from that map (``MaterializeCitesResult.created``). 0 is legitimate
            on a corpus with no in-corpus citations.
        external_resolved: References the V.4 cascade resolved to a confident
            external work (0 when ``resolve_external`` is off / no cascade).
    """

    sources_scanned: int
    sources_with_references: int
    refs_extracted: int
    edges_materialized: int
    external_resolved: int


class ReferenceExtractionService:
    """Extract corpus references via GROBID and materialize U.3's ``cites`` (G.3)."""

    def __init__(
        self,
        source_repo: SourceRepository,
        cites_service: CitesMaterializationService,
        grobid_service: GrobidReferenceService,
        *,
        resolver_cascade: Optional[ResolverCascade] = None,
    ) -> None:
        self.source_repo = source_repo
        self.cites_service = cites_service
        self.grobid_service = grobid_service
        self._resolver_cascade = resolver_cascade

    async def extract_source_references(
        self, source_id: str
    ) -> List[ParsedReference]:
        """Extract one source's references from its ORIGINAL PDF via GROBID.

        Reads the source row, takes the uploaded file at ``asset.file_path``, and
        hands the PDF to GROBID (decision G-D1). Best-effort: a source with no
        asset, no ``file_path``, or a non-PDF path (URL/note sources) yields ``[]``
        — never raises. The ``file_path`` is passed through verbatim (it is
        repo-root-relative in local runs); path resolution is the caller's/env's
        concern.
        """
        source = await self.source_repo.get(source_id)
        asset = getattr(source, "asset", None) if source else None
        file_path = getattr(asset, "file_path", None) if asset else None
        if not file_path or not str(file_path).lower().endswith(".pdf"):
            logger.debug(
                "G.3: source {sid} has no PDF asset (file_path={fp}); no references",
                sid=source_id,
                fp=file_path,
            )
            return []
        refs = await self.grobid_service.extract_references(file_path)
        return list(refs)

    async def materialize_corpus(
        self, *, resolve_external: bool = False
    ) -> ReferenceMaterializationSummary:
        """Extract the whole corpus' references and rebuild U.3's ``cites`` edges.

        Enumerates every source, extracts its references (GROBID per source PDF),
        assembles the ``{source_id: [ParsedReference]}`` map U.3 expects,
        optionally resolves external works (opt-in), then calls U.3's whole-corpus
        :meth:`CitesMaterializationService.materialize` ONCE. Idempotent
        (clear-before-relate) and non-destructive (canonical ``source`` rows are
        untouched).

        Args:
            resolve_external: When ``True``, route each extracted reference through
                the V.4 cascade and count confident external resolutions. Default
                ``False`` skips the external round-trips.

        Returns:
            A :class:`ReferenceMaterializationSummary` of the pass.
        """
        records = await self.source_repo.load_source_records()
        references_by_source: Dict[str, Sequence[ParsedReference]] = {}
        refs_extracted = 0
        for row in records or []:
            sid = row.get("id")
            if sid is None:
                continue
            sid = str(sid)
            refs = await self.extract_source_references(sid)
            if refs:
                references_by_source[sid] = refs
                refs_extracted += len(refs)

        external_resolved = 0
        if resolve_external:
            external_resolved = await self._resolve_external(references_by_source)

        result = await self.cites_service.materialize(references_by_source)

        summary = ReferenceMaterializationSummary(
            sources_scanned=len(records or []),
            sources_with_references=len(references_by_source),
            refs_extracted=refs_extracted,
            edges_materialized=result.created,
            external_resolved=external_resolved,
        )
        logger.info(
            "G.3 reference pass: scanned={scanned} with_refs={with_refs} "
            "refs={refs} cites_edges={edges} external_resolved={ext}",
            scanned=summary.sources_scanned,
            with_refs=summary.sources_with_references,
            refs=summary.refs_extracted,
            edges=summary.edges_materialized,
            ext=summary.external_resolved,
        )
        return summary

    async def _resolve_external(
        self, references_by_source: Dict[str, Sequence[ParsedReference]]
    ) -> int:
        """Route every extracted reference through the V.4 cascade; count hits.

        Best-effort per reference: a cascade failure is logged and treated as a
        no-match (never propagates). Returns 0 (and warns once) when the flag was
        set but no cascade was injected.
        """
        if self._resolver_cascade is None:
            logger.warning(
                "G.3: resolve_external requested but no ResolverCascade is "
                "configured; skipping external resolution"
            )
            return 0
        resolved = 0
        for refs in references_by_source.values():
            for ref in refs:
                try:
                    work = await self._resolver_cascade.resolve(ref)
                except Exception as exc:  # noqa: BLE001 — cascade must fail soft
                    logger.warning(
                        "G.3: external resolution raised (treated as no-match) "
                        "for {r}: {exc}",
                        r=ref.raw_text[:80],
                        exc=exc,
                    )
                    work = None
                if work is not None:
                    resolved += 1
        return resolved
