"""Reference-extraction orchestration (Track V.5) — the V.1-V.3 → U.3 bridge.

Track V.1-V.3 built a PURE producer (``shared.references.extract_references``:
chunks -> ``List[ParsedReference]``). Track U.3 built the ``cites`` materializer
(``CitesMaterializationService.materialize``: a ``{source_id: [ParsedReference]}``
map -> the confident intra-corpus ``source -> cites -> source`` edge set). V.5 is
the integration that connects the two: it reads each source's persisted chunks,
runs the producer, and hands the assembled map to U.3's EXISTING entry point.

Why the whole corpus, not one source
====================================
U.3's ``materialize`` is a WHOLE-CORPUS regenerator: it ``DELETE cites`` (all
edges) then rebuilds from the map it is given (``SourceRepository.clear_cites``
is a bare ``DELETE cites``). Feeding it a single source's references would wipe
every OTHER source's edges. References are not persisted anywhere (the producer
is pure), so the only correct, non-destructive way to drive U.3's existing
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
core path offline + fast; the cascade fails soft per leg (a network error is a
no-match, never a raise). Resolution runs BEFORE materialization and is
independent of it — the cascade is V.4's external-authority resolver, whereas the
``cites`` edges are U.3's separate intra-corpus concern; resolved works are a
count today (persisting a ``ResolvedWork`` into the graph is a later phase).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
from shared.references import extract_references
from shared.references.region_locator import ReferenceChunk
from shared.references.work_resolver import ResolverCascade
from shared.retrieval.cites_matching import ParsedReference
from surrealdb_service.repositories import SourceRepository

from app_main.services.cites_materialization_service import (
    CitesMaterializationService,
)


@dataclass(frozen=True)
class ReferenceMaterializationSummary:
    """Outcome of a V.5 corpus reference pass (telemetry + acceptance).

    Attributes:
        sources_scanned: Sources enumerated from the corpus (the candidates the
            pass extracted references from).
        sources_with_references: Sources that yielded at least one reference (a
            bibliography was located and parsed).
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
    """Extract corpus references and materialize U.3's ``cites`` edges (V.5)."""

    def __init__(
        self,
        source_repo: SourceRepository,
        cites_service: CitesMaterializationService,
        *,
        resolver_cascade: Optional[ResolverCascade] = None,
    ) -> None:
        self.source_repo = source_repo
        self.cites_service = cites_service
        self._resolver_cascade = resolver_cascade

    @staticmethod
    def _to_reference_chunk(chunk: Any) -> ReferenceChunk:
        """Project a persisted ``Chunk`` into the producer's DB-free input type.

        Maps only the fields V.1's region locator needs (text / section_path /
        element_type / order / page). Kept tolerant of a plain object so callers
        can pass either a :class:`shared.models.source.Chunk` or a lightweight
        stand-in.
        """
        return ReferenceChunk(
            text=getattr(chunk, "text", "") or "",
            section_path=tuple(getattr(chunk, "section_path", ()) or ()),
            element_type=getattr(chunk, "element_type", "text") or "text",
            order=int(getattr(chunk, "order", 0) or 0),
            page=getattr(chunk, "physical_page", None),
        )

    async def extract_source_references(
        self, source_id: str
    ) -> List[ParsedReference]:
        """Extract one source's references from its persisted chunks (V.1-V.3).

        Reads the source's chunks (ordered) via the existing repository layer plus
        its ``full_text`` (the V.1 fallback + span source), then runs the pure
        producer. Returns ``[]`` for a source with no locatable bibliography —
        never raises.
        """
        chunks = await self.source_repo.get_chunks(source_id)
        source = await self.source_repo.get(source_id)
        full_text = (getattr(source, "full_text", "") if source else "") or ""
        ref_chunks = [self._to_reference_chunk(c) for c in chunks or []]
        refs: List[ParsedReference] = list(extract_references(ref_chunks, full_text))
        return refs

    async def materialize_corpus(
        self, *, resolve_external: bool = False
    ) -> ReferenceMaterializationSummary:
        """Extract the whole corpus' references and rebuild U.3's ``cites`` edges.

        Enumerates every source, extracts its references, assembles the
        ``{source_id: [ParsedReference]}`` map U.3 expects, optionally resolves
        external works (opt-in), then calls U.3's whole-corpus
        :meth:`CitesMaterializationService.materialize` ONCE. Idempotent
        (clear-before-relate) and non-destructive (canonical ``source`` rows are
        untouched).

        Args:
            resolve_external: When ``True``, route each extracted reference through
                the V.4 cascade and count confident external resolutions. Default
                ``False`` keeps the pass fully offline.

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
            "V.5 reference pass: scanned={scanned} with_refs={with_refs} "
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
                "V.5: resolve_external requested but no ResolverCascade is "
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
                        "V.5: external resolution raised (treated as no-match) "
                        "for {r}: {exc}",
                        r=ref.raw_text[:80],
                        exc=exc,
                    )
                    work = None
                if work is not None:
                    resolved += 1
        return resolved
