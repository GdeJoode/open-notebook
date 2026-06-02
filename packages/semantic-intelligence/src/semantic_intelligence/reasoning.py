"""
Reasoning and provenance tracking for the semantic intelligence layer.

Three capabilities:
1. **Provenance tracking**: dataclasses and helpers for recording how facts
   were derived — source document, extraction method, confidence, chain of
   derivation. Used by all other modules.

2. **Forward chaining**: simple rule engine that evaluates Python-defined
   rules against SurrealDB facts and writes inferred facts back with
   provenance.

3. **SPARQL reasoning**: exports a SurrealDB subgraph to rdflib, runs
   SPARQL CONSTRUCT/SELECT queries, and writes inferred triples back.

Usage:
    # Provenance
    from semantic_intelligence.reasoning import Provenance, attach_provenance

    prov = Provenance(
        source_document="source:abc123",
        extraction_method="llm",
        confidence=0.85,
    )

    # Forward chaining
    from semantic_intelligence.reasoning import Rule, run_forward_chaining

    rules = [
        Rule(
            name="org_in_gemeente",
            condition="SELECT * FROM relation WHERE relation_type = 'gevestigd_in' "
                      "AND out.entity_type = 'Gemeente'",
            action=create_inferred_fact,
        ),
    ]
    await run_forward_chaining(rules)

    # SPARQL
    from semantic_intelligence.reasoning import sparql_query, sparql_construct

    results = await sparql_query("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 10")

    python -m semantic_intelligence.reasoning
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Coroutine, Dict, List, Optional

from loguru import logger


# ============================================================================
# Provenance dataclasses — used by ALL modules
# ============================================================================


@dataclass
class Provenance:
    """Provenance metadata for a fact, entity, relation, or decision.

    Every record produced by the semantic layer should carry provenance
    so we can trace where information came from.
    """

    source_document: Optional[str] = None  # SurrealDB record ID, e.g. "source:abc123"
    source_documents: List[str] = field(default_factory=list)  # Multiple sources
    extracted_at: datetime = field(default_factory=datetime.now)
    extraction_method: str = "unknown"  # "llm", "docling+ollama", "rule_engine", "sparql", "manual"
    confidence: float = 1.0
    provenance_chain: List[str] = field(default_factory=list)  # IDs of facts this was derived from

    def to_dict(self) -> Dict[str, Any]:
        docs = self.source_documents
        if self.source_document and self.source_document not in docs:
            docs = [self.source_document] + docs
        return {
            "source_documents": docs,
            "extracted_at": self.extracted_at.isoformat(),
            "extraction_method": self.extraction_method,
            "confidence": self.confidence,
            "provenance_chain": self.provenance_chain,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Provenance":
        docs = data.get("source_documents", [])
        return cls(
            source_document=docs[0] if docs else None,
            source_documents=docs,
            extracted_at=datetime.fromisoformat(data["extracted_at"])
            if "extracted_at" in data
            else datetime.now(),
            extraction_method=data.get("extraction_method", "unknown"),
            confidence=data.get("confidence", 1.0),
            provenance_chain=data.get("provenance_chain", []),
        )

    def derive(
        self,
        method: str,
        confidence: Optional[float] = None,
        parent_ids: Optional[List[str]] = None,
    ) -> "Provenance":
        """Create a derived provenance from this one.

        Used when a new fact is inferred from an existing fact.
        The new provenance inherits source documents and adds
        this fact's ID to the chain.
        """
        return Provenance(
            source_documents=list(self.source_documents),
            extraction_method=method,
            confidence=confidence if confidence is not None else self.confidence * 0.9,
            provenance_chain=(parent_ids or []) + self.provenance_chain,
        )


def attach_provenance(record: Dict[str, Any], prov: Provenance) -> Dict[str, Any]:
    """Merge provenance fields into a SurrealDB record dict."""
    record.update(prov.to_dict())
    return record


# ============================================================================
# Forward chaining rule engine
# ============================================================================


@dataclass
class Rule:
    """A forward chaining rule.

    - condition: SurrealQL query that finds matching facts.
      Should return records that satisfy the rule's precondition.
    - action: async callable that receives the matched records and
      produces new facts (writes them to SurrealDB).
    - name: human-readable rule name for logging.
    """

    name: str
    condition: str  # SurrealQL SELECT query
    action: Callable[[List[Dict[str, Any]]], Coroutine[Any, Any, int]]
    enabled: bool = True


async def run_forward_chaining(
    rules: List[Rule],
    max_iterations: int = 10,
) -> Dict[str, int]:
    """Run forward chaining rules until no new facts are produced.

    Each iteration evaluates all rules. If any rule's action produces
    new facts, we iterate again (up to max_iterations). This handles
    cascading inferences where rule A's output triggers rule B.

    Args:
        rules: List of Rule objects to evaluate.
        max_iterations: Safety limit to prevent infinite loops.

    Returns:
        Dict mapping rule name to number of facts produced.
    """
    from surrealdb_service.connection import execute_query

    stats: Dict[str, int] = {r.name: 0 for r in rules}
    active_rules = [r for r in rules if r.enabled]

    for iteration in range(max_iterations):
        new_facts_total = 0
        logger.info(f"Forward chaining iteration {iteration + 1}")

        for rule in active_rules:
            try:
                # Evaluate condition
                matches = await execute_query(rule.condition)
                if not matches:
                    continue

                # Run action
                new_count = await rule.action(matches)
                stats[rule.name] += new_count
                new_facts_total += new_count

                if new_count > 0:
                    logger.info(f"  Rule '{rule.name}': {new_count} new facts from {len(matches)} matches")

            except Exception as e:
                logger.error(f"  Rule '{rule.name}' failed: {e}")

        if new_facts_total == 0:
            logger.info(f"Forward chaining converged after {iteration + 1} iterations")
            break
    else:
        logger.warning(f"Forward chaining hit max_iterations ({max_iterations})")

    return stats


# ============================================================================
# SPARQL reasoning via rdflib
# ============================================================================


async def _export_subgraph_to_rdflib(
    entity_filter: str = "",
    limit: int = 10000,
) -> "rdflib.Graph":
    """Export a subgraph from SurrealDB into an rdflib Graph.

    Converts entity records and relation edges into RDF triples
    using a simple namespace scheme:
      - Entities: <urn:on:entity:{id}> rdf:type <urn:on:type:{entity_type}>
      - Relations: <urn:on:entity:{from}> <urn:on:rel:{type}> <urn:on:entity:{to}>
      - Properties: <urn:on:entity:{id}> <urn:on:prop:{key}> Literal(value)
    """
    import rdflib
    from rdflib import URIRef, Literal, Namespace, RDF

    from surrealdb_service.connection import execute_query

    ON = Namespace("urn:on:")
    ONT = Namespace("urn:on:type:")
    REL = Namespace("urn:on:rel:")
    PROP = Namespace("urn:on:prop:")

    g = rdflib.Graph()
    g.bind("on", ON)
    g.bind("ont", ONT)
    g.bind("rel", REL)
    g.bind("prop", PROP)

    # Export entities
    entity_query = "SELECT * FROM entity WHERE status = 'active'"
    if entity_filter:
        entity_query += f" AND {entity_filter}"
    entity_query += f" LIMIT {limit}"

    entities = await execute_query(entity_query)
    for ent in entities:
        eid = str(ent.get("id", "")).replace(":", "_")
        subj = URIRef(f"urn:on:entity:{eid}")
        g.add((subj, RDF.type, ONT[ent.get("entity_type", "Thing")]))
        g.add((subj, PROP["canonical_name"], Literal(ent.get("canonical_name", ""))))

        for key, val in ent.get("properties", {}).items():
            if isinstance(val, (str, int, float, bool)):
                g.add((subj, PROP[key], Literal(val)))

    # Export relations
    relation_query = f"SELECT * FROM relation WHERE status = 'active' LIMIT {limit}"
    relations = await execute_query(relation_query)
    for rel in relations:
        from_id = str(rel.get("in", "")).replace(":", "_")
        to_id = str(rel.get("out", "")).replace(":", "_")
        rel_type = rel.get("relation_type", "related_to")
        g.add((
            URIRef(f"urn:on:entity:{from_id}"),
            REL[rel_type],
            URIRef(f"urn:on:entity:{to_id}"),
        ))

    logger.info(f"Exported {len(entities)} entities and {len(relations)} relations to rdflib ({len(g)} triples)")
    return g


async def sparql_query(
    sparql: str,
    entity_filter: str = "",
) -> List[Dict[str, str]]:
    """Run a SPARQL SELECT query over the SurrealDB entity graph.

    Exports the graph to rdflib, runs the query, and returns results
    as a list of dicts.

    Args:
        sparql: SPARQL SELECT query string.
        entity_filter: Optional SurrealQL WHERE clause to limit the export.

    Returns:
        List of result dicts with variable bindings.
    """
    g = await _export_subgraph_to_rdflib(entity_filter)
    results = g.query(sparql)
    return [
        {str(var): str(val) for var, val in zip(results.vars, row)}
        for row in results
    ]


async def sparql_construct(
    sparql: str,
    write_back: bool = False,
    entity_filter: str = "",
) -> List[tuple]:
    """Run a SPARQL CONSTRUCT query and optionally write inferred triples back.

    Args:
        sparql: SPARQL CONSTRUCT query string.
        write_back: If True, writes inferred relations back to SurrealDB.
        entity_filter: Optional SurrealQL WHERE clause.

    Returns:
        List of (subject, predicate, object) triples.
    """
    from rdflib import URIRef

    from surrealdb_service.connection import execute_query

    g = await _export_subgraph_to_rdflib(entity_filter)
    inferred = g.query(sparql)

    triples = [(str(s), str(p), str(o)) for s, p, o in inferred]
    logger.info(f"SPARQL CONSTRUCT produced {len(triples)} triples")

    if write_back and triples:
        written = 0
        for s, p, o in triples:
            # Parse URIs back to entity IDs
            if "urn:on:entity:" in s and "urn:on:entity:" in o and "urn:on:rel:" in p:
                from_id = s.replace("urn:on:entity:", "").replace("_", ":")
                to_id = o.replace("urn:on:entity:", "").replace("_", ":")
                rel_type = p.split("urn:on:rel:")[-1]

                # Check if relation already exists
                check = await execute_query(
                    "SELECT * FROM relation WHERE in = $from AND out = $to "
                    "AND relation_type = $type LIMIT 1",
                    {"from": from_id, "to": to_id, "type": rel_type},
                )
                if not check:
                    await execute_query(
                        "RELATE $from->relation->$to SET "
                        "relation_type = $type, "
                        "extraction_method = 'sparql', "
                        "confidence = 0.8, "
                        "provenance_chain = [$query]",
                        {
                            "from": from_id,
                            "to": to_id,
                            "type": rel_type,
                            "query": sparql[:200],
                        },
                    )
                    written += 1
        logger.info(f"Wrote {written} inferred relations to SurrealDB")

    return triples


# ============================================================================
# CLI entry point
# ============================================================================


async def _demo():
    """Demo: show provenance tracking and SPARQL query."""
    from surrealdb_service.connection import execute_query

    # Check entity count
    result = await execute_query("SELECT count() FROM entity GROUP ALL")
    count = result[0].get("count", 0) if result else 0
    logger.info(f"Entity count: {count}")

    if count > 0:
        # Run a simple SPARQL query
        results = await sparql_query(
            "SELECT ?name ?type WHERE { "
            "  ?s <urn:on:prop:canonical_name> ?name . "
            "  ?s a ?type "
            "} LIMIT 5"
        )
        for r in results:
            logger.info(f"  {r}")
    else:
        logger.info("No entities found. Run entity extraction first.")

    # Demo: provenance
    prov = Provenance(
        source_document="source:test123",
        extraction_method="demo",
        confidence=0.95,
    )
    derived = prov.derive("rule_engine", parent_ids=["entity:abc"])
    logger.info(f"Original provenance: {prov.to_dict()}")
    logger.info(f"Derived provenance: {derived.to_dict()}")


if __name__ == "__main__":
    asyncio.run(_demo())
