"""
Entity and relation deduplication module.

Extends the existing entity-filtering pipeline with canonical entity
resolution. Works on the `entity` table in SurrealDB.

Strategies:
1. **Blocking**: reduce candidate pairs via token blocking, sorted
   neighbourhood, and vector blocking (nearest neighbours via embeddings).
2. **Similarity scoring**: Jaro-Winkler + semantic cosine similarity.
3. **Merge/link decisions**: above threshold → merge; below → keep; middle → flag.
4. **Triplet deduplication**: remove duplicate (subject, predicate, object).
5. **Conflict detection**: flag contradictions between sources.

Usage:
    from semantic_layer.deduplication import (
        find_duplicates, merge_entities, detect_conflicts,
    )

    candidates = await find_duplicates(entity_type="Gemeente")
    await merge_entities(canonical_id, duplicate_id)
    conflicts = await detect_conflicts()

    python -m semantic_layer.deduplication
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

try:
    from rapidfuzz import fuzz
    from rapidfuzz.distance import JaroWinkler
except ImportError:
    fuzz = None
    JaroWinkler = None


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class DuplicateCandidate:
    """A pair of entities that may be duplicates."""

    entity_a: str  # SurrealDB record ID
    entity_b: str
    name_a: str
    name_b: str
    string_similarity: float = 0.0
    semantic_similarity: float = 0.0
    combined_score: float = 0.0
    method: str = ""  # "token_block", "sorted_neighbourhood", "vector_block"


@dataclass
class ConflictReport:
    """A detected conflict between two facts about the same entity."""

    entity_id: str
    entity_name: str
    property_name: str
    value_a: Any
    value_b: Any
    source_a: str
    source_b: str
    conflict_type: str  # "value", "type", "relationship", "temporal"


# ============================================================================
# Blocking strategies
# ============================================================================


async def _token_blocking(
    entities: List[Dict[str, Any]],
    min_shared_tokens: int = 1,
) -> List[Tuple[str, str]]:
    """Generate candidate pairs via token blocking.

    Entities that share at least min_shared_tokens in their canonical_name
    are considered candidates.
    """
    # Build inverted index: token → set of entity IDs
    token_index: Dict[str, Set[str]] = {}
    for ent in entities:
        eid = str(ent["id"])
        name = ent.get("canonical_name", "").lower()
        tokens = set(name.split())
        for token in tokens:
            if len(token) < 2:
                continue
            token_index.setdefault(token, set()).add(eid)

    # Generate pairs from shared tokens
    pairs: Set[Tuple[str, str]] = set()
    for token, eids in token_index.items():
        if len(eids) > 100:
            continue  # Skip very common tokens
        eid_list = sorted(eids)
        for i in range(len(eid_list)):
            for j in range(i + 1, len(eid_list)):
                pairs.add((eid_list[i], eid_list[j]))

    return list(pairs)


async def _sorted_neighbourhood_blocking(
    entities: List[Dict[str, Any]],
    window_size: int = 5,
) -> List[Tuple[str, str]]:
    """Generate candidate pairs via sorted neighbourhood.

    Sort entities by name, then slide a window and pair entities within it.
    """
    sorted_ents = sorted(entities, key=lambda e: e.get("canonical_name", "").lower())
    pairs: Set[Tuple[str, str]] = set()

    for i in range(len(sorted_ents)):
        for j in range(i + 1, min(i + window_size, len(sorted_ents))):
            a = str(sorted_ents[i]["id"])
            b = str(sorted_ents[j]["id"])
            pairs.add((a, b) if a < b else (b, a))

    return list(pairs)


async def _vector_blocking(
    entities: List[Dict[str, Any]],
    top_k: int = 5,
    min_similarity: float = 0.7,
) -> List[Tuple[str, str]]:
    """Generate candidate pairs via vector similarity.

    For each entity with an embedding, find the top-k nearest neighbours.
    """
    from semantic_layer.config import execute

    pairs: Set[Tuple[str, str]] = set()

    for ent in entities:
        eid = str(ent["id"])
        embedding = ent.get("embedding")
        if not embedding:
            continue

        # Use SurrealDB vector similarity to find nearest
        results = await execute(
            "SELECT id, vector::similarity::cosine(embedding, $emb) AS sim "
            "FROM entity WHERE id != $id AND status = 'active' AND embedding != NONE "
            "ORDER BY sim DESC LIMIT $k",
            {"emb": embedding, "id": eid, "k": top_k},
        )

        for r in results:
            sim = r.get("sim", 0)
            if sim >= min_similarity:
                other_id = str(r["id"])
                pair = (eid, other_id) if eid < other_id else (other_id, eid)
                pairs.add(pair)

    return list(pairs)


# ============================================================================
# Similarity scoring
# ============================================================================


def _string_similarity(name_a: str, name_b: str) -> float:
    """Compute string similarity using Jaro-Winkler.

    Falls back to simple ratio if rapidfuzz is not installed.
    """
    if JaroWinkler:
        return JaroWinkler.normalized_similarity(name_a.lower(), name_b.lower())
    # Fallback: simple token overlap
    tokens_a = set(name_a.lower().split())
    tokens_b = set(name_b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))


def _compute_combined_score(
    string_sim: float,
    semantic_sim: float,
    string_weight: float = 0.4,
    semantic_weight: float = 0.6,
) -> float:
    """Weighted combination of string and semantic similarity."""
    return string_sim * string_weight + semantic_sim * semantic_weight


# ============================================================================
# Main deduplication pipeline
# ============================================================================


async def find_duplicates(
    entity_type: Optional[str] = None,
    merge_threshold: float = 0.85,
    review_threshold: float = 0.6,
    blocking_methods: Optional[List[str]] = None,
) -> List[DuplicateCandidate]:
    """Find duplicate entity candidates.

    Args:
        entity_type: Filter by entity type (e.g. "Gemeente", "PERSON").
        merge_threshold: Above this → auto-merge recommended.
        review_threshold: Above this but below merge → flag for review.
        blocking_methods: Which blocking strategies to use.
            Default: ["token", "sorted", "vector"].

    Returns:
        List of DuplicateCandidate, sorted by combined_score descending.
    """
    from semantic_layer.config import execute

    blocking_methods = blocking_methods or ["token", "sorted", "vector"]

    # Load entities
    query = "SELECT * FROM entity WHERE status = 'active'"
    if entity_type:
        query += f" AND entity_type = '{entity_type}'"
    entities = await execute(query)

    if len(entities) < 2:
        return []

    # Build entity lookup
    entity_map = {str(e["id"]): e for e in entities}

    # Run blocking strategies
    all_pairs: Set[Tuple[str, str]] = set()

    if "token" in blocking_methods:
        pairs = await _token_blocking(entities)
        all_pairs.update(pairs)
        logger.info(f"Token blocking: {len(pairs)} candidates")

    if "sorted" in blocking_methods:
        pairs = await _sorted_neighbourhood_blocking(entities)
        all_pairs.update(pairs)
        logger.info(f"Sorted neighbourhood: {len(pairs)} candidates")

    if "vector" in blocking_methods:
        pairs = await _vector_blocking(entities)
        all_pairs.update(pairs)
        logger.info(f"Vector blocking: {len(pairs)} candidates")

    logger.info(f"Total unique candidate pairs: {len(all_pairs)}")

    # Score each pair
    candidates: List[DuplicateCandidate] = []
    for id_a, id_b in all_pairs:
        ent_a = entity_map.get(id_a, {})
        ent_b = entity_map.get(id_b, {})
        name_a = ent_a.get("canonical_name", "")
        name_b = ent_b.get("canonical_name", "")

        # String similarity
        str_sim = _string_similarity(name_a, name_b)

        # Semantic similarity (if both have embeddings)
        sem_sim = 0.0
        emb_a = ent_a.get("embedding")
        emb_b = ent_b.get("embedding")
        if emb_a and emb_b:
            # Cosine similarity
            import math
            dot = sum(a * b for a, b in zip(emb_a, emb_b))
            norm_a = math.sqrt(sum(a * a for a in emb_a))
            norm_b = math.sqrt(sum(b * b for b in emb_b))
            if norm_a > 0 and norm_b > 0:
                sem_sim = dot / (norm_a * norm_b)

        combined = _compute_combined_score(str_sim, sem_sim)

        if combined >= review_threshold:
            candidates.append(DuplicateCandidate(
                entity_a=id_a,
                entity_b=id_b,
                name_a=name_a,
                name_b=name_b,
                string_similarity=str_sim,
                semantic_similarity=sem_sim,
                combined_score=combined,
            ))

    candidates.sort(key=lambda c: c.combined_score, reverse=True)
    logger.info(
        f"Duplicate candidates: {len(candidates)} above review threshold, "
        f"{sum(1 for c in candidates if c.combined_score >= merge_threshold)} above merge threshold"
    )
    return candidates


# ============================================================================
# Entity merging
# ============================================================================


async def merge_entities(
    canonical_id: str,
    duplicate_id: str,
    merge_aliases: bool = True,
) -> None:
    """Merge a duplicate entity into a canonical entity.

    - Transfers aliases from duplicate to canonical
    - Redirects all relations (in and out) from duplicate to canonical
    - Marks duplicate as merged (status="merged", merged_into=canonical)
    - Preserves provenance from both entities

    Args:
        canonical_id: The entity to keep.
        duplicate_id: The entity to merge into canonical.
        merge_aliases: Whether to create an alias from the duplicate's name.
    """
    from semantic_layer.config import execute

    # Get both entities
    canonical = (await execute("SELECT * FROM $id", {"id": canonical_id}))
    duplicate = (await execute("SELECT * FROM $id", {"id": duplicate_id}))

    if not canonical or not duplicate:
        logger.error(f"Cannot merge: one or both entities not found")
        return

    canonical = canonical[0]
    duplicate = duplicate[0]

    # Merge source_documents
    merged_docs = list(set(
        canonical.get("source_documents", []) + duplicate.get("source_documents", [])
    ))
    await execute(
        "UPDATE $id SET source_documents = $docs, updated_at = time::now()",
        {"id": canonical_id, "docs": merged_docs},
    )

    # Create alias from duplicate name
    if merge_aliases:
        dup_name = duplicate.get("canonical_name", "")
        if dup_name:
            await execute(
                "CREATE entity_alias SET "
                "alias_text = $name, canonical_entity = $eid, confidence = 1.0",
                {"name": dup_name, "eid": canonical_id},
            )

    # Transfer existing aliases
    await execute(
        "UPDATE entity_alias SET canonical_entity = $canonical "
        "WHERE canonical_entity = $duplicate",
        {"canonical": canonical_id, "duplicate": duplicate_id},
    )

    # Redirect relations: incoming
    await execute(
        "UPDATE relation SET out = $canonical WHERE out = $duplicate",
        {"canonical": canonical_id, "duplicate": duplicate_id},
    )

    # Redirect relations: outgoing
    await execute(
        "UPDATE relation SET in = $canonical WHERE in = $duplicate",
        {"canonical": canonical_id, "duplicate": duplicate_id},
    )

    # Mark duplicate as merged
    await execute(
        "UPDATE $id SET status = 'merged', merged_into = $canonical, updated_at = time::now()",
        {"id": duplicate_id, "canonical": canonical_id},
    )

    logger.info(
        f"Merged entity '{duplicate.get('canonical_name')}' into "
        f"'{canonical.get('canonical_name')}'"
    )


# ============================================================================
# Triplet deduplication
# ============================================================================


async def deduplicate_relations() -> int:
    """Remove duplicate relations (same from, to, and type).

    Keeps the relation with highest confidence. Removes others.

    Returns:
        Number of duplicate relations removed.
    """
    from semantic_layer.config import execute

    # Find duplicates: group by (in, out, relation_type)
    dupes = await execute(
        "SELECT in, out, relation_type, count() AS cnt, "
        "array::group(id) AS ids "
        "FROM relation WHERE status = 'active' "
        "GROUP BY in, out, relation_type "
        "HAVING cnt > 1"
    )

    removed = 0
    for group in dupes:
        ids = group.get("ids", [])
        if len(ids) < 2:
            continue

        # Fetch all, keep the one with highest confidence
        rels = []
        for rid in ids:
            r = await execute("SELECT * FROM $id", {"id": rid})
            if r:
                rels.append(r[0])

        rels.sort(key=lambda r: r.get("confidence", 0), reverse=True)

        # Delete all except the first (highest confidence)
        for rel in rels[1:]:
            await execute("DELETE $id", {"id": str(rel["id"])})
            removed += 1

    logger.info(f"Triplet deduplication: removed {removed} duplicate relations")
    return removed


# ============================================================================
# Conflict detection
# ============================================================================


async def detect_conflicts(
    property_names: Optional[List[str]] = None,
) -> List[ConflictReport]:
    """Detect contradicting facts about the same entity from different sources.

    Finds entities where the same property has different values from
    different source documents.

    Args:
        property_names: Specific properties to check. If None, checks all.

    Returns:
        List of ConflictReport objects.
    """
    from semantic_layer.config import execute

    # Get entities with multiple source documents
    entities = await execute(
        "SELECT * FROM entity WHERE status = 'active' "
        "AND array::len(source_documents) > 1"
    )

    conflicts: List[ConflictReport] = []

    for ent in entities:
        eid = str(ent["id"])
        name = ent.get("canonical_name", "")
        props = ent.get("properties", {})
        sources = ent.get("source_documents", [])

        if not props or len(sources) < 2:
            continue

        # Check for property-level conflicts
        # This is a simplified version — in reality, you'd track which
        # source provided which property value
        for key, value in props.items():
            if property_names and key not in property_names:
                continue

            # Look for aliases of this entity with different property values
            aliases_result = await execute(
                "SELECT * FROM entity_alias WHERE canonical_entity = $eid",
                {"eid": eid},
            )

            # Check if other entities (now merged) had different values
            # This is tracked via the provenance_chain
            if isinstance(value, list) and len(value) > 1:
                # Multiple values for the same property — potential conflict
                for i in range(len(value)):
                    for j in range(i + 1, len(value)):
                        if value[i] != value[j]:
                            conflicts.append(ConflictReport(
                                entity_id=eid,
                                entity_name=name,
                                property_name=key,
                                value_a=value[i],
                                value_b=value[j],
                                source_a=sources[0] if sources else "",
                                source_b=sources[1] if len(sources) > 1 else "",
                                conflict_type="value",
                            ))

    logger.info(f"Conflict detection: {len(conflicts)} conflicts found")
    return conflicts


# ============================================================================
# CLI entry point
# ============================================================================


async def _demo():
    """Demo: find duplicates and run deduplication."""
    from semantic_layer.config import execute

    result = await execute("SELECT count() FROM entity GROUP ALL")
    count = result[0].get("count", 0) if result else 0
    logger.info(f"Entities: {count}")

    if count >= 2:
        candidates = await find_duplicates(review_threshold=0.5)
        for c in candidates[:5]:
            logger.info(
                f"  Candidate: '{c.name_a}' <-> '{c.name_b}' "
                f"(str={c.string_similarity:.2f}, sem={c.semantic_similarity:.2f}, "
                f"combined={c.combined_score:.2f})"
            )

        # Deduplicate relations
        removed = await deduplicate_relations()

        # Detect conflicts
        conflicts = await detect_conflicts()
        for c in conflicts[:5]:
            logger.info(f"  Conflict: {c.entity_name}.{c.property_name}: {c.value_a} vs {c.value_b}")
    else:
        logger.info("Not enough entities for deduplication demo")


if __name__ == "__main__":
    asyncio.run(_demo())
