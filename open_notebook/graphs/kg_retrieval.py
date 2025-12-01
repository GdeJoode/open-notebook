"""
Knowledge Graph Retrieval Pipeline

Implements HippoRAG-style retrieval:
1. Fact scoring (embedding search + claim matching)
2. Entity extraction from query
3. PPR-based passage ranking
4. Dense fallback retrieval
5. Optional LLM reranking

See docs/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md Phase 4 for documentation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from loguru import logger

from open_notebook.database.repository import repo_query
from open_notebook.processors.embeddings import (
    EmbeddingConfig,
    KnowledgeGraphEmbeddings,
    get_kg_embeddings,
)
from open_notebook.graphs.kg_analysis import PersonalizedPageRank, PPRConfig
from open_notebook.graphs.kg_loader import KnowledgeGraphLoader, LoaderConfig


# =============================================================================
# CONFIGURATION
# =============================================================================


class RetrievalMode(str, Enum):
    """Retrieval mode selection."""
    HIPPORAG = "hipporag"  # Full HippoRAG with PPR
    DENSE = "dense"  # Dense retrieval only
    HYBRID = "hybrid"  # Combine dense + PPR scores


@dataclass
class RetrievalConfig:
    """Configuration for the retrieval pipeline."""

    # Retrieval mode
    mode: RetrievalMode = RetrievalMode.HIPPORAG

    # Fact scoring weights
    fact_embedding_weight: float = 0.5
    claim_similarity_weight: float = 0.3
    citation_weight: float = 0.2

    # PPR settings (HippoRAG defaults)
    ppr_damping: float = 0.5  # HippoRAG uses 0.5 for aggressive teleportation
    passage_weight: float = 0.05  # Low weight for passage nodes in reset prob

    # Retrieval limits
    top_k_facts: int = 50  # Initial fact candidates
    top_k_entities: int = 20  # Entities for PPR seeding
    top_k_results: int = 10  # Final results

    # Dense fallback threshold
    min_entity_matches: int = 1  # Minimum entities to use PPR

    # Reranking (optional)
    enable_reranking: bool = False
    rerank_top_k: int = 20  # Candidates for reranking

    # Embedding config
    embedding_config: Optional[EmbeddingConfig] = None


@dataclass
class RetrievalResult:
    """Result from the retrieval pipeline."""

    source_id: str
    score: float
    title: Optional[str] = None
    content_preview: Optional[str] = None

    # Score components (for debugging/transparency)
    ppr_score: float = 0.0
    dense_score: float = 0.0
    rerank_score: float = 0.0

    # Metadata
    matched_entities: List[str] = field(default_factory=list)
    matched_facts: List[str] = field(default_factory=list)


@dataclass
class FactScore:
    """Score for a fact/relationship."""

    fact_id: str  # Edge ID (e.g., "mentions:abc123")
    fact_text: str
    score: float

    # Components
    embedding_similarity: float = 0.0
    claim_similarity: float = 0.0
    citation_score: float = 0.0

    # Related entities
    source_entity: str = ""
    target_entity: str = ""


# =============================================================================
# FACT SCORER
# =============================================================================


class FactScorer:
    """
    Scores facts (relationships) based on query relevance.

    Uses three scoring methods:
    1. Embedding similarity (vector search)
    2. Claim similarity (for claim nodes)
    3. Citation-based scoring (for citation relationships)
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self._embeddings = get_kg_embeddings(self.config.embedding_config)

    async def score_facts(
        self,
        query: str,
        limit: int = 50,
    ) -> List[FactScore]:
        """
        Score all facts by relevance to query.

        Args:
            query: The search query
            limit: Maximum number of facts to return

        Returns:
            List of FactScore objects sorted by score
        """
        # Generate query embedding
        query_embedding = await self._embeddings.embed_query(query)
        if not query_embedding:
            logger.warning("Failed to generate query embedding")
            return []

        # Get scores from different sources
        embedding_scores = await self._score_by_embedding(query_embedding, limit * 2)
        claim_scores = await self._score_by_claim_similarity(query_embedding, limit)
        citation_scores = await self._score_by_citations(query, limit)

        # Combine scores
        all_facts: Dict[str, FactScore] = {}

        for fact_id, score, fact_text, src, tgt in embedding_scores:
            all_facts[fact_id] = FactScore(
                fact_id=fact_id,
                fact_text=fact_text,
                score=0.0,
                embedding_similarity=score,
                source_entity=src,
                target_entity=tgt,
            )

        for fact_id, score, fact_text in claim_scores:
            if fact_id in all_facts:
                all_facts[fact_id].claim_similarity = score
            else:
                all_facts[fact_id] = FactScore(
                    fact_id=fact_id,
                    fact_text=fact_text,
                    score=0.0,
                    claim_similarity=score,
                )

        for fact_id, score in citation_scores:
            if fact_id in all_facts:
                all_facts[fact_id].citation_score = score

        # Calculate combined scores
        w_emb = self.config.fact_embedding_weight
        w_claim = self.config.claim_similarity_weight
        w_cite = self.config.citation_weight

        for fact in all_facts.values():
            fact.score = (
                w_emb * fact.embedding_similarity +
                w_claim * fact.claim_similarity +
                w_cite * fact.citation_score
            )

        # Sort by combined score and return top-k
        sorted_facts = sorted(
            all_facts.values(),
            key=lambda x: x.score,
            reverse=True
        )

        return sorted_facts[:limit]

    async def _score_by_embedding(
        self,
        query_embedding: List[float],
        limit: int,
    ) -> List[Tuple[str, float, str, str, str]]:
        """
        Score facts by embedding similarity.

        Returns: List of (fact_id, score, fact_text, source_entity, target_entity)
        """
        try:
            # Search fact embeddings on edges (mentions, relates_to)
            # First, try mentions edges with fact_text
            results = await repo_query(
                """
                SELECT
                    id,
                    fact_text,
                    in as source_entity,
                    out as target_entity,
                    vector::similarity::cosine(fact_embedding, $query_emb) AS score
                FROM mentions
                WHERE fact_embedding != NONE
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"query_emb": query_embedding, "limit": limit}
            )

            scored = []
            for r in results:
                if r.get("score") and r["score"] > 0:
                    scored.append((
                        str(r["id"]),
                        float(r["score"]),
                        r.get("fact_text", ""),
                        str(r.get("source_entity", "")),
                        str(r.get("target_entity", "")),
                    ))

            # Also search relates_to edges
            results2 = await repo_query(
                """
                SELECT
                    id,
                    fact_text,
                    in as source_entity,
                    out as target_entity,
                    vector::similarity::cosine(fact_embedding, $query_emb) AS score
                FROM relates_to
                WHERE fact_embedding != NONE
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"query_emb": query_embedding, "limit": limit}
            )

            for r in results2:
                if r.get("score") and r["score"] > 0:
                    scored.append((
                        str(r["id"]),
                        float(r["score"]),
                        r.get("fact_text", ""),
                        str(r.get("source_entity", "")),
                        str(r.get("target_entity", "")),
                    ))

            # Sort combined results
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:limit]

        except Exception as e:
            logger.error(f"Embedding fact search failed: {e}")
            return []

    async def _score_by_claim_similarity(
        self,
        query_embedding: List[float],
        limit: int,
    ) -> List[Tuple[str, float, str]]:
        """
        Score claims by embedding similarity.

        Returns: List of (claim_id, score, statement)
        """
        try:
            results = await repo_query(
                """
                SELECT
                    id,
                    statement,
                    vector::similarity::cosine(embedding, $query_emb) AS score
                FROM claim
                WHERE embedding != NONE
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"query_emb": query_embedding, "limit": limit}
            )

            return [
                (str(r["id"]), float(r["score"]), r.get("statement", ""))
                for r in results
                if r.get("score") and r["score"] > 0
            ]

        except Exception as e:
            logger.error(f"Claim similarity search failed: {e}")
            return []

    async def _score_by_citations(
        self,
        query: str,
        limit: int,
    ) -> List[Tuple[str, float]]:
        """
        Score citation relationships based on query relevance.

        Citations with matching context get higher scores.

        Returns: List of (citation_id, score)
        """
        try:
            # Full-text search on citation context
            results = await repo_query(
                """
                SELECT
                    id,
                    search::score(1) AS score
                FROM cites
                WHERE citation_context @1@ $query
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"query": query, "limit": limit}
            )

            # Normalize scores to 0-1 range
            if results:
                max_score = max(r.get("score", 0) for r in results) or 1
                return [
                    (str(r["id"]), float(r.get("score", 0)) / max_score)
                    for r in results
                    if r.get("score")
                ]
            return []

        except Exception as e:
            logger.debug(f"Citation search failed (may not exist): {e}")
            return []


# =============================================================================
# ENTITY EXTRACTOR
# =============================================================================


class EntityExtractor:
    """
    Extracts relevant entities from query for PPR seeding.

    Methods:
    1. Direct entity matching (exact/fuzzy name match)
    2. Embedding-based entity retrieval
    3. Fact-derived entity extraction
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self._embeddings = get_kg_embeddings(self.config.embedding_config)

    async def extract_entities(
        self,
        query: str,
        fact_scores: Optional[List[FactScore]] = None,
        limit: int = 20,
    ) -> Dict[str, float]:
        """
        Extract entities relevant to the query.

        Args:
            query: The search query
            fact_scores: Pre-computed fact scores (optional)
            limit: Maximum entities to return

        Returns:
            Dict mapping entity_id -> relevance score
        """
        entity_scores: Dict[str, float] = {}

        # Method 1: Direct entity name matching
        name_matches = await self._match_by_name(query, limit)
        for entity_id, score in name_matches:
            entity_scores[entity_id] = max(
                entity_scores.get(entity_id, 0),
                score * 0.8  # Weight for name match
            )

        # Method 2: Embedding similarity
        query_embedding = await self._embeddings.embed_query(query)
        if query_embedding:
            emb_matches = await self._match_by_embedding(query_embedding, limit)
            for entity_id, score in emb_matches:
                entity_scores[entity_id] = max(
                    entity_scores.get(entity_id, 0),
                    score * 0.9  # Weight for embedding match
                )

        # Method 3: Extract from fact scores
        if fact_scores:
            for fact in fact_scores[:limit]:
                if fact.source_entity:
                    entity_scores[fact.source_entity] = max(
                        entity_scores.get(fact.source_entity, 0),
                        fact.score * 0.7  # Weight for fact-derived
                    )
                if fact.target_entity:
                    entity_scores[fact.target_entity] = max(
                        entity_scores.get(fact.target_entity, 0),
                        fact.score * 0.7
                    )

        # Normalize scores
        if entity_scores:
            max_score = max(entity_scores.values())
            if max_score > 0:
                entity_scores = {
                    k: v / max_score for k, v in entity_scores.items()
                }

        # Return top-k
        sorted_entities = sorted(
            entity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

        return dict(sorted_entities)

    async def _match_by_name(
        self,
        query: str,
        limit: int,
    ) -> List[Tuple[str, float]]:
        """Match entities by name (fuzzy/full-text search)."""
        try:
            # Full-text search on entity names
            results = await repo_query(
                """
                SELECT
                    id,
                    name,
                    search::score(1) AS score
                FROM entity
                WHERE name @1@ $query
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"query": query, "limit": limit}
            )

            # Normalize scores
            if results:
                max_score = max(r.get("score", 0) for r in results) or 1
                return [
                    (str(r["id"]), float(r.get("score", 0)) / max_score)
                    for r in results
                    if r.get("score")
                ]
            return []

        except Exception as e:
            logger.debug(f"Entity name search failed: {e}")
            return []

    async def _match_by_embedding(
        self,
        query_embedding: List[float],
        limit: int,
    ) -> List[Tuple[str, float]]:
        """Match entities by embedding similarity."""
        try:
            results = await repo_query(
                """
                SELECT
                    id,
                    vector::similarity::cosine(embedding, $query_emb) AS score
                FROM entity
                WHERE embedding != NONE
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"query_emb": query_embedding, "limit": limit}
            )

            return [
                (str(r["id"]), float(r["score"]))
                for r in results
                if r.get("score") and r["score"] > 0
            ]

        except Exception as e:
            logger.error(f"Entity embedding search failed: {e}")
            return []


# =============================================================================
# PPR RETRIEVER
# =============================================================================


class PPRRetriever:
    """
    HippoRAG-style retrieval using Personalized PageRank.

    Seeds PPR with query-relevant entities and propagates
    relevance through the knowledge graph to rank passages.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self._loader: Optional[KnowledgeGraphLoader] = None
        self._ppr = PersonalizedPageRank(
            PPRConfig(alpha=self.config.ppr_damping)
        )

    async def _get_graph(self):
        """Get or load the knowledge graph."""
        if self._loader is None:
            self._loader = KnowledgeGraphLoader()

        if self._loader.graph is None:
            await self._loader.load_full_graph()

        return self._loader.graph

    async def retrieve(
        self,
        entity_scores: Dict[str, float],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve passages using PPR from entity seeds.

        Args:
            entity_scores: Dict mapping entity_id -> relevance score
            top_k: Number of results to return

        Returns:
            List of (source_id, ppr_score) tuples
        """
        if not entity_scores:
            logger.warning("No entity seeds for PPR, returning empty")
            return []

        graph = await self._get_graph()
        if graph is None or graph.number_of_nodes() == 0:
            logger.warning("Knowledge graph is empty")
            return []

        # Build reset probability vector
        reset_prob: Dict[str, float] = {}

        # Add entity weights
        for entity_id, score in entity_scores.items():
            if entity_id in graph:
                reset_prob[entity_id] = score

        # Add low weight for source nodes (HippoRAG style)
        for node in graph.nodes():
            if graph.nodes[node].get("node_type") == "source":
                reset_prob[node] = self.config.passage_weight

        if not reset_prob:
            logger.warning("No valid seeds found in graph")
            return []

        # Compute PPR
        ppr_scores = self._ppr.compute(graph, reset_prob)

        # Filter to source nodes only
        source_scores = []
        for node_id, score in ppr_scores.items():
            if graph.nodes[node_id].get("node_type") == "source":
                source_scores.append((node_id, score))

        # Sort and return top-k
        source_scores.sort(key=lambda x: x[1], reverse=True)
        return source_scores[:top_k]


# =============================================================================
# DENSE RETRIEVER
# =============================================================================


class DenseRetriever:
    """
    Dense retrieval using embedding similarity.

    Used as fallback when PPR doesn't find good results
    or as a component in hybrid retrieval.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self._embeddings = get_kg_embeddings(self.config.embedding_config)

    async def retrieve(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Retrieve passages by embedding similarity.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (source_id, similarity_score) tuples
        """
        # Generate query embedding
        query_embedding = await self._embeddings.embed_query(query)
        if not query_embedding:
            logger.warning("Failed to generate query embedding")
            return []

        try:
            results = await repo_query(
                """
                SELECT
                    id,
                    vector::similarity::cosine(embedding, $query_emb) AS score
                FROM source
                WHERE embedding != NONE
                ORDER BY score DESC
                LIMIT $limit
                """,
                {"query_emb": query_embedding, "limit": top_k}
            )

            return [
                (str(r["id"]), float(r["score"]))
                for r in results
                if r.get("score") and r["score"] > 0
            ]

        except Exception as e:
            logger.error(f"Dense retrieval failed: {e}")
            return []


# =============================================================================
# RERANKER (OPTIONAL)
# =============================================================================


class LLMReranker:
    """
    Optional LLM-based reranking for improved precision.

    Takes top candidates and re-scores them using LLM judgment.
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()

    async def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, float]],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidates using LLM scoring.

        Args:
            query: Original search query
            candidates: List of (source_id, score) to rerank
            top_k: Number of results after reranking

        Returns:
            Reranked list of (source_id, new_score) tuples
        """
        if not candidates:
            return []

        # Fetch content for candidates
        source_ids = [c[0] for c in candidates[:self.config.rerank_top_k]]

        try:
            sources = await repo_query(
                """
                SELECT id, title, content
                FROM source
                WHERE id IN $ids
                """,
                {"ids": source_ids}
            )

            source_content = {
                str(s["id"]): {
                    "title": s.get("title", ""),
                    "content": s.get("content", "")[:1000],  # Limit content
                }
                for s in sources
            }

        except Exception as e:
            logger.error(f"Failed to fetch source content: {e}")
            return candidates[:top_k]

        # For now, return original scores
        # TODO: Implement actual LLM reranking when needed
        logger.info("LLM reranking not yet implemented, returning original order")
        return candidates[:top_k]


# =============================================================================
# UNIFIED RETRIEVAL PIPELINE
# =============================================================================


class KnowledgeGraphRetriever:
    """
    Unified retrieval pipeline combining all components.

    Pipeline:
    1. Score facts by query relevance
    2. Extract entities from query and facts
    3. Run PPR from entity seeds
    4. Optionally combine with dense retrieval
    5. Optionally rerank with LLM
    """

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()

        self._fact_scorer = FactScorer(config)
        self._entity_extractor = EntityExtractor(config)
        self._ppr_retriever = PPRRetriever(config)
        self._dense_retriever = DenseRetriever(config)
        self._reranker = LLMReranker(config) if config and config.enable_reranking else None

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant passages for a query.

        Args:
            query: Search query
            top_k: Number of results (overrides config)

        Returns:
            List of RetrievalResult objects
        """
        top_k = top_k or self.config.top_k_results

        logger.info(f"Retrieving for query: {query[:100]}...")

        # Step 1: Score facts
        fact_scores = await self._fact_scorer.score_facts(
            query,
            limit=self.config.top_k_facts
        )
        logger.debug(f"Found {len(fact_scores)} relevant facts")

        # Step 2: Extract entities
        entity_scores = await self._entity_extractor.extract_entities(
            query,
            fact_scores=fact_scores,
            limit=self.config.top_k_entities,
        )
        logger.debug(f"Extracted {len(entity_scores)} entities")

        # Step 3: Determine retrieval strategy
        use_ppr = (
            self.config.mode in [RetrievalMode.HIPPORAG, RetrievalMode.HYBRID]
            and len(entity_scores) >= self.config.min_entity_matches
        )

        ppr_results: List[Tuple[str, float]] = []
        dense_results: List[Tuple[str, float]] = []

        if use_ppr:
            # Step 3a: PPR retrieval
            ppr_results = await self._ppr_retriever.retrieve(
                entity_scores,
                top_k=self.config.rerank_top_k if self._reranker else top_k,
            )
            logger.debug(f"PPR returned {len(ppr_results)} results")

        if (
            self.config.mode == RetrievalMode.DENSE
            or self.config.mode == RetrievalMode.HYBRID
            or not ppr_results  # Fallback
        ):
            # Step 3b: Dense retrieval
            dense_results = await self._dense_retriever.retrieve(
                query,
                top_k=self.config.rerank_top_k if self._reranker else top_k,
            )
            logger.debug(f"Dense returned {len(dense_results)} results")

        # Step 4: Combine results
        combined = self._combine_results(ppr_results, dense_results)

        # Step 5: Optional reranking
        if self._reranker and self.config.enable_reranking:
            combined = await self._reranker.rerank(query, combined, top_k)

        # Step 6: Build result objects
        results = await self._build_results(
            combined[:top_k],
            entity_scores,
            fact_scores,
            ppr_results,
            dense_results,
        )

        logger.info(f"Retrieved {len(results)} results")
        return results

    def _combine_results(
        self,
        ppr_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
    ) -> List[Tuple[str, float]]:
        """Combine PPR and dense results."""
        if self.config.mode == RetrievalMode.HIPPORAG:
            if ppr_results:
                return ppr_results
            return dense_results

        if self.config.mode == RetrievalMode.DENSE:
            return dense_results

        # HYBRID: combine scores
        scores: Dict[str, Tuple[float, float]] = {}

        # Normalize PPR scores
        if ppr_results:
            max_ppr = max(s for _, s in ppr_results) or 1
            for source_id, score in ppr_results:
                scores[source_id] = (score / max_ppr, 0.0)

        # Normalize dense scores
        if dense_results:
            max_dense = max(s for _, s in dense_results) or 1
            for source_id, score in dense_results:
                ppr_score = scores.get(source_id, (0.0, 0.0))[0]
                scores[source_id] = (ppr_score, score / max_dense)

        # Combine with weights (0.6 PPR, 0.4 dense)
        combined = [
            (source_id, 0.6 * ppr + 0.4 * dense)
            for source_id, (ppr, dense) in scores.items()
        ]
        combined.sort(key=lambda x: x[1], reverse=True)

        return combined

    async def _build_results(
        self,
        ranked: List[Tuple[str, float]],
        entity_scores: Dict[str, float],
        fact_scores: List[FactScore],
        ppr_results: List[Tuple[str, float]],
        dense_results: List[Tuple[str, float]],
    ) -> List[RetrievalResult]:
        """Build RetrievalResult objects with metadata."""
        if not ranked:
            return []

        # Fetch source metadata
        source_ids = [r[0] for r in ranked]
        try:
            sources = await repo_query(
                """
                SELECT id, title, content
                FROM source
                WHERE id IN $ids
                """,
                {"ids": source_ids}
            )
            source_meta = {
                str(s["id"]): {
                    "title": s.get("title"),
                    "content": s.get("content", "")[:200],  # Preview
                }
                for s in sources
            }
        except Exception:
            source_meta = {}

        # Build lookup dicts
        ppr_lookup = dict(ppr_results)
        dense_lookup = dict(dense_results)

        # Get matched entities for each source
        # (This would require graph traversal in production)
        matched_entities_lookup: Dict[str, List[str]] = {}

        results = []
        for source_id, score in ranked:
            meta = source_meta.get(source_id, {})
            results.append(RetrievalResult(
                source_id=source_id,
                score=score,
                title=meta.get("title"),
                content_preview=meta.get("content"),
                ppr_score=ppr_lookup.get(source_id, 0.0),
                dense_score=dense_lookup.get(source_id, 0.0),
                matched_entities=matched_entities_lookup.get(source_id, []),
            ))

        return results


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def hipporag_retrieve(
    query: str,
    top_k: int = 10,
    config: Optional[RetrievalConfig] = None,
) -> List[RetrievalResult]:
    """
    Convenience function for HippoRAG retrieval.

    Args:
        query: Search query
        top_k: Number of results
        config: Optional retrieval configuration

    Returns:
        List of RetrievalResult objects
    """
    config = config or RetrievalConfig(mode=RetrievalMode.HIPPORAG)
    retriever = KnowledgeGraphRetriever(config)
    return await retriever.retrieve(query, top_k)


async def dense_retrieve(
    query: str,
    top_k: int = 10,
    config: Optional[RetrievalConfig] = None,
) -> List[RetrievalResult]:
    """
    Convenience function for dense retrieval.

    Args:
        query: Search query
        top_k: Number of results
        config: Optional retrieval configuration

    Returns:
        List of RetrievalResult objects
    """
    config = config or RetrievalConfig(mode=RetrievalMode.DENSE)
    retriever = KnowledgeGraphRetriever(config)
    return await retriever.retrieve(query, top_k)


async def hybrid_retrieve(
    query: str,
    top_k: int = 10,
    config: Optional[RetrievalConfig] = None,
) -> List[RetrievalResult]:
    """
    Convenience function for hybrid retrieval (PPR + dense).

    Args:
        query: Search query
        top_k: Number of results
        config: Optional retrieval configuration

    Returns:
        List of RetrievalResult objects
    """
    config = config or RetrievalConfig(mode=RetrievalMode.HYBRID)
    retriever = KnowledgeGraphRetriever(config)
    return await retriever.retrieve(query, top_k)
