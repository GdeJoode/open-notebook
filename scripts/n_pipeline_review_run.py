"""Live end-to-end measurement of the extraction pipeline (Track N review).

Ingests ONE real document through the production path and records how many
entities and relations survive each stage, from the LLM's raw output to the
persisted graph. Written for the pipeline review requested after N.4d: the
question is not whether a stage is correct — each has its own tests — but
whether the chain as a whole is coherent, and where things fall away.

What is real here: the document, the docling parse, the schema detection, the
LLM extraction, every filtering stage, the canonical bridge, and the writes to
the live database. What is NOT the full production path: chunks are built from
docling's text elements directly rather than through the ingestion service's
parser layer, because that layer reaches docling over HTTP and the service is
not running. Chunk TEXT is what Pass 2 consumes, so the LLM-facing input is
faithful; the bbox positions this skips affect neither extraction nor filtering.

Run (writes to the live DB — use a scratch notebook):

    SURREAL_DATABASE=staging ENABLE_CONCEPT_ALIGNMENT=true \
        uv run --project apps/app-main python scripts/n_pipeline_review_run.py \
        --pdf docling_input/Convenant_Oost-Groningen.pdf --chunks 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("SURREAL_DATABASE", "staging")

# A SurrealDB record id, strictly: table:id with no separators that could end
# the statement. Guards the one place this script must interpolate.
_RECORD_ID_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*:[A-Za-z0-9_]+")

STAGES: List[Dict[str, Any]] = []


def _record(stage: str, kind: str, before: int, after: int, note: str = "") -> None:
    STAGES.append(
        {"stage": stage, "kind": kind, "before": before, "after": after,
         "delta": after - before, "note": note}
    )


def _instrument():
    """Wrap each filtering component so the chain reports its own attrition.

    Patched at CLASS level: the workflow instance is built inside the service,
    so there is no object to wrap from here. Every wrapper is transparent — it
    records lengths and returns the original result untouched.
    """
    from unittest.mock import patch

    from entity_filtering.deduplication.embedding_deduplicator import (
        EmbeddingDeduplicator,
    )
    from entity_filtering.deduplication.entity_deduplicator import EntityDeduplicator
    from entity_filtering.deduplication.fuzzy_resolver import FuzzyResolver
    from entity_filtering.filters.noise_filter import NoiseFilter
    from entity_filtering.filters.normalizer import EntityNormalizer
    from entity_filtering.filters.reclassifier import EntityReclassifier

    patches = []

    def wrap(cls, method, stage, kind="entities", pair=False):
        original = getattr(cls, method)

        def wrapper(self, items, *a, **kw):
            before = len(items)
            out = original(self, items, *a, **kw)
            seq = out[0] if pair else out
            _record(stage, kind, before, len(seq))
            return out

        patches.append(patch.object(cls, method, wrapper))

    wrap(NoiseFilter, "filter_entities", "1 noise filter")
    wrap(NoiseFilter, "filter_relations", "1 noise filter", kind="relations")
    wrap(EntityNormalizer, "normalize", "2 normalize")
    wrap(EntityReclassifier, "reclassify", "3 reclassify")
    wrap(EntityDeduplicator, "deduplicate", "4 string dedup", pair=True)
    wrap(FuzzyResolver, "resolve", "5 fuzzy dedup", pair=True)
    wrap(EmbeddingDeduplicator, "deduplicate", "6 embedding dedup", pair=True)
    for p in patches:
        p.start()
    return patches


async def _graph_totals(execute_query) -> Dict[str, int]:
    """Running graph size, so accumulation ACROSS documents is visible.

    A single document cannot show whether the pipeline recognises a person it
    has already seen; the whole point of a multi-document profile is that this
    number stops growing linearly if anything is matching.
    """
    out: Dict[str, int] = {}
    for table in ("entity", "relation", "ontology_gap"):
        rows = await execute_query(f"SELECT count() AS n FROM {table} GROUP ALL", {})
        out[table] = rows[0]["n"] if rows else 0
    return out


def _chunks_from_pdf(pdf: Path, limit: int, min_chars: int = 40) -> List[Dict[str, Any]]:
    from docling.document_converter import DocumentConverter

    # NOTE a limitation of this harness, stated rather than hidden: it reads only
    # `doc.texts`. The production chunker (`chunking.chunk_builder.from_document`)
    # also emits a chunk per TABLE and per image, so a table-heavy document gives
    # production more to work with than it gives this script.
    doc = DocumentConverter().convert(pdf).document
    chunks: List[Dict[str, Any]] = []
    chapter = None
    for item in getattr(doc, "texts", []):
        text = (getattr(item, "text", "") or "").strip()
        label = str(getattr(item, "label", "") or "paragraph")
        if label in ("section_header", "title"):
            chapter = text
        if len(text) < min_chars:
            continue  # headers and stray fragments carry no extractable claim
        prov = getattr(item, "prov", None)
        page = (prov[0].page_no - 1) if prov else 0
        chunks.append(
            {
                "text": text,
                "order": len(chunks),
                "physical_page": page,
                "element_type": "paragraph",
                "chapter": chapter,
                "positions": [],
                "metadata": {},
            }
        )
        if len(chunks) >= limit:
            break
    return chunks


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", required=True, nargs="+")
    parser.add_argument("--chunks", type=int, default=12)
    parser.add_argument(
        "--min-chars", type=int, default=40,
        help="Skip text elements shorter than this. The default drops headers and "
             "stray fragments; set 0 for a document docling line-fragments, where "
             "EVERY element is short and the default would skip the whole file.",
    )
    parser.add_argument("--out", default="claudedocs/pipeline-review-run.json")
    parser.add_argument(
        "--notebook",
        help="Reuse an existing notebook instead of creating one, so a document "
             "added later accumulates into the SAME graph — which is the only "
             "way cross-document behaviour stays measurable.",
    )
    args = parser.parse_args()

    from app_main.services.entity_extraction_service import EntityExtractionService
    from loguru import logger
    from surrealdb_service.connection import execute_query
    from surrealdb_service.repositories.source import ChunkRepository, SourceRepository

    started = time.time()
    logger.remove()

    if args.notebook:
        notebook_id = args.notebook
        if not _RECORD_ID_RE.fullmatch(notebook_id):
            raise ValueError(f"not a record id: {notebook_id!r}")
        existing = await execute_query(
            f"SELECT id FROM notebook WHERE id = {notebook_id}", {}
        )
        if not existing:
            raise ValueError(f"no such notebook: {notebook_id}")
    else:
        nb = await execute_query(
            "CREATE notebook SET name=$n, description='pipeline review', archived=false",
            {"n": f"Pipeline review {int(started)}"},
        )
        notebook_id = str(nb[0]["id"])
        if not _RECORD_ID_RE.fullmatch(notebook_id):
            raise ValueError(f"refusing to interpolate a non-record id: {notebook_id!r}")
        # `notebook_schema.notebook` is typed `record<notebook>`, and a bound
        # string does not satisfy that — the same reason the RELATE below
        # interpolates.
        await execute_query(
            f"CREATE notebook_schema SET notebook={notebook_id}, base_ontology='deals', "
            "accepted_extensions=[], pending_extensions=[], excluded_types=[], "
            "coverage_pct=0.0, review_required=false, soft_nudge_dismissed=false",
            {},
        )
    print(f"notebook={notebook_id}")

    _instrument()
    service = EntityExtractionService(source_repo=SourceRepository())
    documents: List[Dict[str, Any]] = []

    for pdf_path in args.pdf:
        doc_started = time.time()
        chunks = _chunks_from_pdf(Path(pdf_path), args.chunks, args.min_chars)
        src = await execute_query(
            "CREATE source SET title=$t, full_text=$x, asset={}, insights=[]",
            {"t": Path(pdf_path).name, "x": "\n\n".join(c["text"] for c in chunks)},
        )
        source_id = str(src[0]["id"])
        if not _RECORD_ID_RE.fullmatch(source_id):
            raise ValueError(f"refusing to interpolate a non-record id: {source_id!r}")
        # The edge table types `in` as record<source>: source->reference->notebook.
        await execute_query(f"RELATE {source_id}->reference->{notebook_id}", {})
        for c in chunks:
            c["source"] = source_id
        await ChunkRepository().bulk_create(chunks)

        before = len(STAGES)
        result = await service.run_extraction(
            source_id=source_id, notebook_id=notebook_id, run_filtering=True
        )
        totals = await _graph_totals(execute_query)
        documents.append({
            "pdf": pdf_path, "source": source_id, "chunks": len(chunks),
            "elapsed_s": round(time.time() - doc_started, 1),
            "result": result, "stages": STAGES[before:],
            "graph_after": totals,
        })
        print(f"  {Path(pdf_path).name[:44]:44} "
              f"{result.get('entity_count', 0):3} ents  "
              f"{result.get('relation_count', 0):3} rels  "
              f"graph={totals['entity']}/{totals['relation']}  "
              f"({documents[-1]['elapsed_s']}s)", flush=True)

    totals = await _graph_totals(execute_query)
    gaps = await execute_query("SELECT * FROM ontology_gap", {})
    entities = await execute_query(
        "SELECT name, entity_type, primary_type FROM entity ORDER BY name", {}
    )

    report = {
        "documents": documents,
        "chunks_per_document": args.chunks,
        "elapsed_s": round(time.time() - started, 1),
        "notebook": notebook_id,
        "graph": totals,
        "entities": entities,
        "ontology_gaps": [
            {k: g.get(k) for k in
             ("entity_text", "entity_type_guess", "frequency", "ontology_name")}
            for g in (gaps or [])
        ],
    }
    Path(args.out).write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )

    print(f"\n{'document':46} {'ents':>5} {'rels':>5} {'graph e/r':>12}")
    for d in documents:
        print(f"{Path(d['pdf']).name[:46]:46} "
              f"{d['result'].get('entity_count', 0):5} "
              f"{d['result'].get('relation_count', 0):5} "
              f"{d['graph_after']['entity']:5}/{d['graph_after']['relation']:<6}")
    print(f"\nfinal graph: {totals['entity']} entities, {totals['relation']} "
          f"relations, {totals['ontology_gap']} gaps")
    print(f"report -> {args.out}  ({report['elapsed_s']}s)")


if __name__ == "__main__":
    asyncio.run(main())
