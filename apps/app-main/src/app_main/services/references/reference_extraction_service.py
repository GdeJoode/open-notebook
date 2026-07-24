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

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from loguru import logger
from shared.references.footnote_reference_extractor import (
    FootnoteReferenceExtractor,
)
from shared.references.grobid_reference_service import GrobidReferenceService
from shared.references.work_resolver import ResolverCascade
from shared.retrieval.cites_matching import ParsedReference
from surrealdb_service.repositories import SourceRepository

from app_main.services.cites_materialization_service import (
    CitesMaterializationService,
)


def _dedup_key(ref: ParsedReference) -> str:
    """A coarse identity for a reference: its DOI, else its normalized raw text.

    Bibliography and footnote refs rarely overlap (one is a scholarly biblStruct,
    the other a government identifier), but a DOI-carrying footnote citation could
    duplicate a bibliography entry — collapse those on the shared DOI, otherwise on
    the lowercased/whitespace-collapsed verbatim text.
    """
    if ref.doi:
        return f"doi:{ref.doi}"
    return "raw:" + " ".join((ref.raw_text or "").lower().split())


#: The compose mount `./notebook_data:/app/data`: a source's stored
#: `data/uploads/…` path is container-relative. In-container it resolves as-is;
#: for a local run the `data/` prefix maps to the host `notebook_data/`.
_MOUNT_PREFIX = "data/"
_HOST_UPLOADS_ROOT = "notebook_data/"


def _resolve_asset_pdf(file_path: str) -> str:
    """Resolve a source's recorded asset path to an on-disk PDF across environments.

    Sources store a container-relative path (``data/uploads/X.pdf``, the
    ``./notebook_data:/app/data`` mount). Tries, in order: the path as-is
    (in-container, and local papers under ``data/uploads``); an explicit
    ``OPEN_NOTEBOOK_UPLOADS_ROOT`` override (root + basename); the mount inverse
    ``data/`` → ``notebook_data/`` (local dev). Returns the first existing
    candidate, else the original path unchanged (GROBID then reports it not found
    and the pass yields ``[]`` — best-effort).
    """
    candidates = [file_path]
    root = os.environ.get("OPEN_NOTEBOOK_UPLOADS_ROOT")
    if root:
        candidates.append(str(Path(root) / Path(file_path).name))
    if file_path.startswith(_MOUNT_PREFIX):
        candidates.append(_HOST_UPLOADS_ROOT + file_path[len(_MOUNT_PREFIX):])
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    return file_path


def _merge_references(
    bibliography: List[ParsedReference], footnotes: List[ParsedReference]
) -> List[ParsedReference]:
    """Bibliography refs plus footnote refs, dropping obvious duplicates.

    Bibliography refs keep their order and priority; a footnote ref is appended
    only when its :func:`_dedup_key` was not already seen.
    """
    merged: List[ParsedReference] = list(bibliography)
    seen = {_dedup_key(ref) for ref in bibliography}
    for ref in footnotes:
        key = _dedup_key(ref)
        if key in seen:
            continue
        seen.add(key)
        merged.append(ref)
    return merged


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
        footnote_extractor: Optional[FootnoteReferenceExtractor] = None,
    ) -> None:
        self.source_repo = source_repo
        self.cites_service = cites_service
        self.grobid_service = grobid_service
        self._resolver_cascade = resolver_cascade
        self._footnote_extractor = footnote_extractor

    async def extract_source_references(
        self, source_id: str
    ) -> List[ParsedReference]:
        """Extract one source's references from its ORIGINAL PDF via GROBID.

        Reads the source row, takes the uploaded file at ``asset.file_path``, and
        hands the PDF to GROBID (decision G-D1). Best-effort: a source with no
        asset, no ``file_path``, or a non-PDF path (URL/note sources) yields ``[]``
        — never raises. The stored ``file_path`` is container-relative
        (``data/uploads/…``, the ``notebook_data`` mount); :func:`_resolve_asset_pdf`
        maps it to an on-disk PDF across environments before GROBID reads it.

        When a :class:`FootnoteReferenceExtractor` is wired (GF.4), the source's
        references are the GROBID **bibliography** refs PLUS the **footnote** refs
        (the policy-document citation type GROBID's bibliography model misses),
        deduplicated on obvious overlaps. Without one, only the bibliography refs
        are returned (the existing G.* behaviour, unchanged).
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
        pdf_path = _resolve_asset_pdf(str(file_path))
        refs = list(await self.grobid_service.extract_references(pdf_path))
        if self._footnote_extractor is not None:
            footnote_refs = await self._footnote_extractor.extract(pdf_path)
            refs = _merge_references(refs, footnote_refs)
        return refs

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
