"""Track PC.4 — does ontology detection select what a document's labels need?

Detection selects up to three ontologies per document for the extraction prompt
(``detect_applicable_schemas``, plus the notebook's declared base). For every
source this prints the selection and classifies each entity's label:

* **in selected** — some selected ontology declares it;
* **only in unselected** — only an ontology detection did NOT select declares it:
  a detection miss (typing still resolves it, since PC.4 types against every
  shipped ontology, but the prompt did not offer the type);
* **in no ontology** — no shipped ontology declares it: a vocabulary gap, which
  better detection cannot fix.

The selection is read from ``extraction_result.metadata.applied_ontologies``
(recorded since PC.4). For a source extracted before that, the detection is
REPLAYED — it is deterministic (content overlap + document type, no model) — with
today's ontologies and notebook schema, which may differ from what ran then; the
output marks each source ``recorded`` or ``replayed``.

Run with SURREAL_* pointing at the target (the WSL venv, env from ``.env``)::

    python scripts/pc4_detection_probe.py

Read-only: it writes nothing.
"""

from __future__ import annotations

import asyncio
from collections import Counter, defaultdict

from app_main.services.entity_extraction_service import (
    EntityExtractionService,
    _applicability_sample,
)
from ontology_extraction.multi_schema_orchestrator import detect_applicable_schemas
from ontology_manager import get_ontology_manager
from ontology_manager.canonical_bridge import _find_definition
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories import NotebookSchemaRepository


async def _replay(source, candidates, notebook_schema):
    chunks = await execute_query(
        "SELECT text, order FROM chunk WHERE source = $id ORDER BY order ASC;",
        {"id": ensure_record_id(str(source["id"]))},
    )
    detected = await detect_applicable_schemas(
        document_type=source.get("dt"),
        document_text=_applicability_sample(chunks),
        ontologies=candidates,
        top_k=3,
    )
    if notebook_schema is not None and notebook_schema.base_ontology:
        svc = EntityExtractionService.__new__(EntityExtractionService)
        detected = svc._apply_notebook_schema_default(detected, candidates, notebook_schema)
    return [o.metadata.name for o, _ in detected]


async def main() -> int:
    manager = get_ontology_manager()
    onts = {}
    for name in await manager.list_ontologies():
        ontology = await manager.get_ontology(name)
        if ontology is not None:
            onts[name] = ontology
    candidates = list(onts.values())

    sources = await execute_query(
        "SELECT id, metadata.document_type AS dt FROM source;", {}
    )
    recorded = {
        str(r["source_id"]): r.get("applied")
        for r in await execute_query(
            "SELECT source_id, metadata.applied_ontologies AS applied FROM extraction_result;",
            {},
        )
    }
    notebook_of = {
        str(r["s"]): str(r["nb"])
        for r in await execute_query("SELECT in AS s, out AS nb FROM reference;", {})
    }
    labels_of = defaultdict(list)
    for e in await execute_query(
        "SELECT primary_type, properties.raw_entity_type AS raw, source_documents "
        "FROM entity WHERE status IN ['active', 'reference'];",
        {},
    ):
        label = (e.get("primary_type") or e.get("raw") or "").strip()
        for s in e.get("source_documents") or []:
            labels_of[str(s)].append(label)

    home_cache: dict = {}

    def home(label):
        if label not in home_cache:
            home_cache[label] = {n for n, o in onts.items() if _find_definition(label, [o])}
        return home_cache[label]

    schemas = NotebookSchemaRepository()
    total, missed = Counter(), Counter()
    for source in sources:
        sid = str(source["id"])
        if recorded.get(sid):
            selection, how = list(recorded[sid]), "recorded"
        else:
            nbs = await schemas.get_by_notebook(notebook_of[sid]) if sid in notebook_of else None
            selection, how = await _replay(source, candidates, nbs), "replayed"
        counts = Counter()
        for label in labels_of.get(sid, []):
            if not label:
                counts["no label"] += 1
            elif not home(label):
                counts["in no ontology"] += 1
            elif home(label) & set(selection):
                counts["in selected"] += 1
            else:
                counts["only in unselected"] += 1
                missed[label] += 1
        total.update(counts)
        print(f"{sid} [{how}] {selection} {dict(counts)}")

    print("TOTAL", dict(total))
    print("missed labels", missed.most_common(20))
    if not total:
        print("no labelled entities: nothing measured (not a pass)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
