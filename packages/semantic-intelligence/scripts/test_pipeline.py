"""
Test script for the semantic intelligence layer.

Reads docling-parsed JSON documents, extracts entities and relations
via Ollama LLM, writes them to SurrealDB entity/relation tables with
provenance, then runs all semantic layer modules.

Usage:
    python -m semantic_intelligence.test_pipeline /data/output/Convenant_Oost-Groningen
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger


# ============================================================================
# Entity extraction via Ollama (simplified — uses the LLM directly)
# ============================================================================


async def extract_entities_from_text(
    text: str,
    source_id: str = "",
) -> tuple[List[Dict], List[Dict]]:
    """Extract entities and relations from text using Ollama.

    Returns:
        Tuple of (entities, relations) as dicts.
    """
    import httpx

    from semantic_intelligence.config import OLLAMA_LLM_MODEL, OLLAMA_URL

    prompt = f"""Extract entities and relations from this Dutch government document text.

Return valid JSON with this structure:
{{
  "entities": [
    {{"name": "exact name", "type": "one of: Gemeente|Provincie|Regio|Organisatie|Persoon|Beleidsinstrument|Thema|Programma", "description": "one sentence"}}
  ],
  "relations": [
    {{"source": "entity name", "target": "entity name", "type": "relation type", "description": "one sentence"}}
  ]
}}

Focus on: geographical entities (gemeenten, provincies, regio's), organizations (ministeries, koepels), policy instruments (regio deals, convenanten), and thematic concepts (brede welvaart, werkgelegenheid, leefbaarheid).

TEXT:
{text[:6000]}

Return ONLY valid JSON, no other text."""

    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": OLLAMA_LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_ctx": 8192},
            },
        )
        resp.raise_for_status()
        response_text = resp.json().get("response", "")

    # Parse JSON from response
    try:
        # Find JSON in the response
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response_text[start:end])
            entities = data.get("entities", [])
            relations = data.get("relations", [])
            return entities, relations
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse LLM response as JSON: {e}")

    return [], []


# ============================================================================
# Write entities to SurrealDB
# ============================================================================


async def write_entities_to_db(
    entities: List[Dict],
    relations: List[Dict],
    source_document: str = "",
    extraction_method: str = "llm",
) -> tuple[int, int]:
    """Write extracted entities and relations to SurrealDB.

    Creates canonical entity records and RELATE edges with provenance.
    Handles deduplication by canonical_name + entity_type (UNIQUE index).

    Returns:
        Tuple of (entities_created, relations_created).
    """
    from semantic_intelligence.config import embed_text
    from surrealdb_service.connection import execute_query

    entity_count = 0
    relation_count = 0

    # Map of name → entity ID for relation creation
    entity_map: Dict[str, str] = {}

    for ent in entities:
        name = ent.get("name", "").strip()
        etype = ent.get("type", "Thing").strip()
        desc = ent.get("description", "")

        if not name:
            continue

        # Check if entity already exists
        existing = await execute_query(
            "SELECT id FROM entity WHERE canonical_name = $name AND entity_type = $type LIMIT 1",
            {"name": name, "type": etype},
        )

        if existing:
            eid = str(existing[0]["id"])
            # Update source_documents if not already there
            await execute_query(
                "UPDATE $id SET source_documents += $src, updated_at = time::now()",
                {"id": eid, "src": source_document},
            )
            entity_map[name] = eid
            continue

        # Generate embedding
        try:
            embedding = await embed_text(f"{etype}: {name}. {desc}")
        except Exception:
            embedding = []

        # Create entity
        result = await execute_query(
            "CREATE entity CONTENT $data RETURN id",
            {
                "data": {
                    "canonical_name": name,
                    "entity_type": etype,
                    "description": desc,
                    "source_documents": [source_document] if source_document else [],
                    "extraction_method": extraction_method,
                    "confidence": 0.85,
                    "embedding": embedding,
                    "properties": ent.get("properties", {}),
                    "status": "active",
                }
            },
        )
        if result:
            eid = str(result[0]["id"])
            entity_map[name] = eid
            entity_count += 1

            # Create alias
            await execute_query(
                "CREATE entity_alias SET alias_text = $name, canonical_entity = $eid, confidence = 1.0",
                {"name": name, "eid": eid},
            )

    # Create relations
    for rel in relations:
        source_name = rel.get("source", "").strip()
        target_name = rel.get("target", "").strip()
        rel_type = rel.get("type", "related_to").strip()

        source_id = entity_map.get(source_name)
        target_id = entity_map.get(target_name)

        if not source_id or not target_id:
            continue

        # Check if relation already exists
        existing = await execute_query(
            "SELECT id FROM relation WHERE in = $from AND out = $to AND relation_type = $type LIMIT 1",
            {"from": source_id, "to": target_id, "type": rel_type},
        )
        if existing:
            continue

        await execute_query(
            f"RELATE {source_id}->relation->{target_id} SET "
            f"relation_type = $type, "
            f"source_documents = $docs, "
            f"extraction_method = $method, "
            f"confidence = 0.8",
            {
                "type": rel_type,
                "docs": [source_document] if source_document else [],
                "method": extraction_method,
            },
        )
        relation_count += 1

    return entity_count, relation_count


# ============================================================================
# Main test pipeline
# ============================================================================


async def process_document(doc_dir: Path) -> Dict[str, Any]:
    """Process a single docling output directory: extract entities → write to DB."""
    json_path = doc_dir / "output" / "extracted_info" / "document.json"
    if not json_path.exists():
        logger.error(f"No document.json found in {doc_dir}")
        return {}

    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)

    # Combine all page text
    full_text = ""
    for page in doc.get("pages", []):
        for elem in page.get("elements", []):
            content = elem.get("content", "")
            if content:
                full_text += content + "\n"

    doc_name = doc_dir.name
    logger.info(f"Processing {doc_name}: {len(full_text)} chars")

    # Extract entities from first ~6000 chars (LLM context limit)
    entities, relations = await extract_entities_from_text(
        full_text, source_id=doc_name
    )
    logger.info(f"  Extracted: {len(entities)} entities, {len(relations)} relations")

    # Write to SurrealDB
    ent_count, rel_count = await write_entities_to_db(
        entities, relations,
        source_document=doc_name,
        extraction_method="llm",
    )
    logger.info(f"  Written: {ent_count} new entities, {rel_count} new relations")

    return {
        "document": doc_name,
        "extracted_entities": len(entities),
        "extracted_relations": len(relations),
        "new_entities": ent_count,
        "new_relations": rel_count,
    }


async def run_test_pipeline(doc_dirs: List[Path]):
    """Run the full test pipeline on multiple documents."""
    from surrealdb_service.connection import execute_query

    logger.info("=" * 60)
    logger.info("SEMANTIC LAYER TEST PIPELINE")
    logger.info("=" * 60)

    # Step 1: Extract entities from all documents
    logger.info("\n--- Step 1: Entity extraction ---")
    results = []
    for doc_dir in doc_dirs:
        result = await process_document(doc_dir)
        if result:
            results.append(result)

    # Show entity/relation counts
    counts = await execute_query("SELECT count() FROM entity GROUP ALL")
    ent_total = counts[0].get("count", 0) if counts else 0
    counts = await execute_query("SELECT count() FROM relation GROUP ALL")
    rel_total = counts[0].get("count", 0) if counts else 0
    logger.info(f"\nTotal in DB: {ent_total} entities, {rel_total} relations")

    if ent_total == 0:
        logger.warning("No entities extracted — cannot test semantic layer modules")
        return

    # Step 2: Graph algorithms
    logger.info("\n--- Step 2: Graph algorithms ---")
    from semantic_intelligence.graph_algorithms import run_all
    graph_result = await run_all()
    logger.info(f"Graph: {graph_result.get('nodes')} nodes, {graph_result.get('edges')} edges, "
                f"{graph_result.get('communities')} communities")
    if graph_result.get("top_entities"):
        for ent in graph_result["top_entities"][:5]:
            logger.info(f"  Top: {ent['name']} (PR={ent['pagerank']:.4f}, community={ent['community']})")

    # Step 3: Decision tracker
    logger.info("\n--- Step 3: Decision intelligence ---")
    from semantic_intelligence.decision_tracker import (
        find_precedents,
        link_decisions,
        record_decision,
    )

    dec1 = await record_decision(
        category="regiodeal",
        scenario="Convenant Oost-Groningen",
        reasoning="Oost-Groningen is een NPVR-regio met uitdagingen op het gebied van "
        "bevolkingskrimp, werkgelegenheid, en bereikbaarheid van voorzieningen.",
        outcome="Regio Deal toegekend",
        source_documents=["Convenant_Oost-Groningen"],
    )

    dec2 = await record_decision(
        category="regiodeal",
        scenario="Convenant Zuidwest-Friesland",
        reasoning="Zuidwest-Friesland kampt met vergrijzing, ontgroening en afname "
        "van voorzieningen. Brede welvaart indicatoren lager dan gemiddeld.",
        outcome="Regio Deal toegekend",
        source_documents=["Convenant_Zuidwest-Friesland"],
    )

    await link_decisions(dec1, dec2, "related_policy")

    precedents = await find_precedents("bevolkingskrimp regio voorzieningen", limit=3)
    logger.info(f"Precedents found: {len(precedents)}")
    for p in precedents:
        logger.info(f"  {p.get('category')}: {p.get('scenario', '')[:50]} (sim={p.get('similarity', 0):.2f})")

    # Step 4: Deduplication
    logger.info("\n--- Step 4: Deduplication ---")
    from semantic_intelligence.deduplication import detect_conflicts, find_duplicates
    candidates = await find_duplicates(review_threshold=0.4)
    logger.info(f"Duplicate candidates: {len(candidates)}")
    for c in candidates[:5]:
        logger.info(f"  '{c.name_a}' <-> '{c.name_b}' (score={c.combined_score:.2f})")

    conflicts = await detect_conflicts()
    logger.info(f"Conflicts: {len(conflicts)}")

    # Step 5: Provenance + SPARQL
    logger.info("\n--- Step 5: SPARQL reasoning ---")
    from semantic_intelligence.reasoning import sparql_query
    results = await sparql_query(
        "PREFIX on: <urn:on:prop:> "
        "PREFIX ont: <urn:on:type:> "
        "SELECT ?name ?type WHERE { "
        "  ?s on:canonical_name ?name . "
        "  ?s a ?type "
        "} LIMIT 10"
    )
    logger.info(f"SPARQL results: {len(results)}")
    for r in results[:5]:
        logger.info(f"  {r}")

    # Step 6: Ontology management
    logger.info("\n--- Step 6: Ontology management ---")
    from semantic_intelligence.ontology_manager import (
        create_brede_welvaart_skos,
        export_ontology,
        generate_shacl_shapes,
        load_all_ontologies,
    )

    ont_dir = Path("/mnt/e/Repos/Private/open-notebook/packages/ontology-manager/ontologies")
    if ont_dir.exists():
        merged = load_all_ontologies(ont_dir)
        shapes = generate_shacl_shapes(merged)
        logger.info(f"Ontology: {len(merged)} triples, {len(shapes)} SHACL triples")
    else:
        logger.warning("Ontology directory not found")

    bw = create_brede_welvaart_skos()
    logger.info(f"Brede Welvaart SKOS: {len(bw)} triples")

    logger.info("\n" + "=" * 60)
    logger.info("TEST PIPELINE COMPLETE")
    logger.info("=" * 60)


# ============================================================================
# CLI
# ============================================================================


if __name__ == "__main__":
    # Default: use docling output directories
    doc_dirs = [
        Path("/mnt/e/Repos/Private/open-notebook/docling_output/Convenant_Oost-Groningen"),
        Path("/mnt/e/Repos/Private/open-notebook/docling_output/Convenant_Zuid-Oost-Drenthe-II"),
        Path("/mnt/e/Repos/Private/open-notebook/docling_output/Convenant_Zuidwest-Friesland"),
    ]

    # Allow override from command line
    if len(sys.argv) > 1:
        doc_dirs = [Path(p) for p in sys.argv[1:]]

    # Filter to existing dirs
    existing = [d for d in doc_dirs if d.exists()]
    if not existing:
        logger.error(f"No document directories found: {doc_dirs}")
        sys.exit(1)

    asyncio.run(run_test_pipeline(existing))
