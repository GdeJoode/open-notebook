"""
Decision intelligence module for the semantic intelligence layer.

Records decisions as first-class objects in SurrealDB with:
- Category, scenario, reasoning, outcome, confidence
- Embedding of reasoning text for precedent search
- Causal links between decisions via RELATE edges
- Graph traversal for impact analysis

Usage:
    from semantic_intelligence.decision_tracker import (
        record_decision, link_decisions, find_precedents,
        trace_causal_chain, analyse_impact,
    )

    # Record a new decision
    dec_id = await record_decision(
        category="regiodeal",
        scenario="Toekenning regiodeal Zuid-Limburg",
        reasoning="Hoge score op brede welvaart indicatoren...",
        outcome="Toegekend, 25M EUR",
        confidence=0.9,
    )

    # Find similar past decisions
    precedents = await find_precedents("brede welvaart Limburg", limit=5)

    # Link two decisions causally
    await link_decisions(dec_a, dec_b, "informed_by")

    # Trace downstream impact
    chain = await trace_causal_chain(dec_id, depth=3)

    python -m semantic_intelligence.decision_tracker
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

from loguru import logger


# ============================================================================
# Record decisions
# ============================================================================


async def record_decision(
    category: str,
    reasoning: str,
    scenario: Optional[str] = None,
    outcome: Optional[str] = None,
    confidence: float = 1.0,
    source_documents: Optional[List[str]] = None,
    related_entities: Optional[List[str]] = None,
    properties: Optional[Dict[str, Any]] = None,
    embed: bool = True,
) -> str:
    """Record a decision as a first-class object in SurrealDB.

    Args:
        category: Decision category (e.g. "regiodeal", "beleid", "investering").
        reasoning: The reasoning behind the decision (will be embedded for search).
        scenario: Description of the scenario/context.
        outcome: What was decided.
        confidence: Confidence score (0-1).
        source_documents: SurrealDB record IDs of related source documents.
        related_entities: SurrealDB record IDs of related entities.
        properties: Additional metadata.
        embed: Whether to generate an embedding of the reasoning text.

    Returns:
        The SurrealDB record ID of the created decision.
    """
    from semantic_intelligence.config import execute, embed_text

    # Build record
    params: Dict[str, Any] = {
        "category": category,
        "reasoning": reasoning,
        "confidence": confidence,
        "source_documents": source_documents or [],
        "related_entities": related_entities or [],
        "properties": properties or {},
        "extraction_method": "manual",
    }
    if scenario:
        params["scenario"] = scenario
    if outcome:
        params["outcome"] = outcome

    # Generate embedding
    if embed:
        embed_text_str = f"{category}: {scenario or ''} {reasoning}"
        embedding = await embed_text(embed_text_str)
        params["embedding"] = embedding

    # Create record
    result = await execute(
        "CREATE decision CONTENT $data RETURN id",
        {"data": params},
    )

    dec_id = str(result[0]["id"]) if result else ""
    logger.info(f"Recorded decision: {dec_id} ({category})")
    return dec_id


# ============================================================================
# Link decisions
# ============================================================================


async def link_decisions(
    decision_a: str,
    decision_b: str,
    relationship_type: str = "caused",
    description: Optional[str] = None,
    confidence: float = 1.0,
) -> str:
    """Create a causal link between two decisions.

    Args:
        decision_a: Source decision ID (the cause).
        decision_b: Target decision ID (the effect).
        relationship_type: Type of causal link (e.g. "caused", "informed_by", "superseded").
        description: Optional description of the relationship.
        confidence: Confidence in the causal link.

    Returns:
        The RELATE edge record ID.
    """
    from semantic_intelligence.config import execute

    params: Dict[str, Any] = {
        "from": decision_a,
        "to": decision_b,
        "rel_type": relationship_type,
        "confidence": confidence,
    }
    if description:
        params["desc"] = description

    query = (
        f"RELATE {decision_a}->caused->{decision_b} SET "
        "relationship_type = $rel_type, "
        "confidence = $confidence"
    )
    if description:
        query += ", description = $desc"

    result = await execute(query, params)
    edge_id = str(result[0].get("id", "")) if result else ""
    logger.info(f"Linked decisions: {decision_a} -[{relationship_type}]-> {decision_b}")
    return edge_id


# ============================================================================
# Find precedents (vector similarity search)
# ============================================================================


async def find_precedents(
    reasoning_text: str,
    limit: int = 5,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Find similar past decisions via vector similarity on reasoning text.

    Args:
        reasoning_text: The reasoning to search for similar decisions.
        limit: Maximum number of results.
        category: Optional filter by decision category.

    Returns:
        List of decision records with similarity scores, sorted by relevance.
    """
    from semantic_intelligence.config import execute, embed_text

    # Generate embedding for the query text
    query_embedding = await embed_text(reasoning_text)

    # Build SurrealQL query with cosine similarity
    where_clause = "WHERE status = 'active' AND embedding != NONE"
    if category:
        where_clause += f" AND category = '{category}'"

    query = (
        f"SELECT *, vector::similarity::cosine(embedding, $emb) AS similarity "
        f"FROM decision {where_clause} "
        f"ORDER BY similarity DESC LIMIT $limit"
    )

    results = await execute(query, {"emb": query_embedding, "limit": limit})

    # Clean up results
    precedents = []
    for r in results:
        r.pop("embedding", None)  # Don't return the embedding vector
        precedents.append(r)

    logger.info(f"Found {len(precedents)} precedents for query")
    return precedents


# ============================================================================
# Trace causal chains (graph traversal)
# ============================================================================


async def trace_causal_chain(
    decision_id: str,
    depth: int = 5,
    direction: str = "downstream",
) -> List[Dict[str, Any]]:
    """Trace the causal chain from a decision.

    Args:
        decision_id: Starting decision ID.
        depth: Maximum traversal depth.
        direction: "downstream" (effects) or "upstream" (causes).

    Returns:
        List of decisions in the causal chain, ordered by distance.
    """
    from semantic_intelligence.config import execute

    if direction == "downstream":
        # Follow outgoing caused edges: what did this decision cause?
        arrow = "->caused->"
    else:
        # Follow incoming caused edges: what caused this decision?
        arrow = "<-caused<-"

    # SurrealDB graph traversal with depth
    query = f"SELECT * FROM $start{arrow}decision..{depth}"
    results = await execute(query, {"start": decision_id})

    # Flatten nested traversal results
    chain = []
    _flatten_traversal(results, chain, set())

    logger.info(f"Causal chain ({direction}): {len(chain)} decisions from {decision_id}")
    return chain


def _flatten_traversal(data: Any, out: List[Dict], seen: set) -> None:
    """Recursively flatten SurrealDB graph traversal results."""
    if isinstance(data, dict):
        did = str(data.get("id", ""))
        if did and did not in seen:
            seen.add(did)
            clean = {k: v for k, v in data.items() if k != "embedding"}
            out.append(clean)
    elif isinstance(data, list):
        for item in data:
            _flatten_traversal(item, out, seen)


# ============================================================================
# Impact analysis
# ============================================================================


async def analyse_impact(
    decision_id: str,
    depth: int = 5,
) -> Dict[str, Any]:
    """Analyse the downstream impact of a decision.

    Combines causal chain traversal with entity relationships
    to show what was affected by this decision.

    Args:
        decision_id: Decision to analyse.
        depth: Maximum traversal depth.

    Returns:
        Dict with: downstream_decisions, affected_entities, total_reach.
    """
    from semantic_intelligence.config import execute

    # Get the decision itself
    decision_result = await execute("SELECT * FROM $id", {"id": decision_id})
    decision = decision_result[0] if decision_result else {}

    # Get downstream decisions
    downstream = await trace_causal_chain(decision_id, depth, "downstream")

    # Get related entities (from the decision + all downstream)
    all_entity_ids = set(decision.get("related_entities", []))
    for dec in downstream:
        all_entity_ids.update(dec.get("related_entities", []))

    # Fetch entity details
    affected_entities = []
    if all_entity_ids:
        for eid in all_entity_ids:
            ent = await execute("SELECT canonical_name, entity_type FROM $id", {"id": eid})
            if ent:
                affected_entities.append(ent[0])

    return {
        "decision": {
            "id": decision_id,
            "category": decision.get("category", ""),
            "outcome": decision.get("outcome", ""),
        },
        "downstream_decisions": len(downstream),
        "affected_entities": affected_entities,
        "total_reach": len(downstream) + len(affected_entities),
    }


# ============================================================================
# CLI entry point
# ============================================================================


async def _demo():
    """Demo: record decisions, link them, search precedents."""
    from semantic_intelligence.config import execute

    # Check existing decisions
    result = await execute("SELECT count() FROM decision GROUP ALL")
    count = result[0].get("count", 0) if result else 0
    logger.info(f"Existing decisions: {count}")

    # Record sample decisions
    dec1 = await record_decision(
        category="regiodeal",
        scenario="Regio Deal Zuid-Limburg 2024",
        reasoning="Zuid-Limburg scoort laag op brede welvaart indicatoren: "
        "gezondheid, arbeidsparticipatie, en bereikbaarheid. "
        "NPVR-regio met structurele uitdagingen.",
        outcome="Toegekend, 25M EUR voor 4 jaar",
        confidence=0.95,
    )

    dec2 = await record_decision(
        category="regiodeal",
        scenario="Regio Deal Zeeuws-Vlaanderen 2024",
        reasoning="Zeeuws-Vlaanderen kampt met bevolkingskrimp, vergrijzing, "
        "en bereikbaarheidsproblemen. Grensregio met specifieke uitdagingen "
        "rond grensoverschrijdende samenwerking.",
        outcome="Toegekend, 15M EUR voor 4 jaar",
        confidence=0.9,
    )

    # Link them
    await link_decisions(dec1, dec2, "related_policy", "Beide NPVR regiodeals 2024")

    # Search for precedents
    precedents = await find_precedents("brede welvaart krimp regio", limit=3)
    for p in precedents:
        logger.info(
            f"  Precedent: {p.get('category')} - {p.get('scenario', '')[:50]} "
            f"(similarity={p.get('similarity', 0):.2f})"
        )

    # Analyse impact
    impact = await analyse_impact(dec1)
    logger.info(f"Impact analysis: {impact}")


if __name__ == "__main__":
    asyncio.run(_demo())
