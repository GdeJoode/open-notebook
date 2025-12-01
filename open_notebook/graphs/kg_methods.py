"""
Knowledge Graph Analysis Methods

High-level analysis methods for the knowledge graph:
- find_experts(): Find authoritative sources/entities on a topic
- trace_claim(): Trace the provenance and support for a claim
- compute_influence(): Calculate influence scores

See docs/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md Phase 3.3 for documentation.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
from loguru import logger

from open_notebook.graphs.kg_analysis import (
    PersonalizedPageRank,
    CentralityCalculator,
    PathFinder,
    PPRConfig,
)
from open_notebook.graphs.kg_loader import (
    KnowledgeGraphLoader,
    LoaderConfig,
)


# =============================================================================
# RESULT MODELS
# =============================================================================


@dataclass
class ExpertResult:
    """Result from find_experts()."""

    entity_id: str
    name: str
    entity_type: str
    score: float  # Combined expertise score
    centrality: float  # PageRank centrality
    mention_count: int  # Number of sources mentioning this entity
    connected_topics: List[str]  # Related topic entities


@dataclass
class ClaimTraceResult:
    """Result from trace_claim()."""

    claim_id: str
    statement: str
    confidence: float
    supporting_sources: List[Dict[str, Any]]  # Sources that support this claim
    related_claims: List[Dict[str, Any]]  # Similar or related claims
    evidence_chain: List[List[str]]  # Paths from claim to supporting entities
    entities_involved: List[str]


@dataclass
class InfluenceResult:
    """Result from influence score computation."""

    node_id: str
    node_type: str
    influence_score: float
    component_scores: Dict[str, float]  # Breakdown of score components
    influenced_nodes: List[str]  # Nodes this entity influences
    influenced_by: List[str]  # Nodes that influence this entity


# =============================================================================
# EXPERT FINDING
# =============================================================================


class ExpertFinder:
    """
    Finds authoritative entities (experts) on a given topic.

    Uses a combination of:
    - Centrality scores (PageRank, degree)
    - Connection to topic entities
    - Source citation patterns
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.centrality = CentralityCalculator()
        self.ppr = PersonalizedPageRank()
        self.path_finder = PathFinder()

    async def find_experts(
        self,
        topic_entity_ids: List[str],
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[ExpertResult]:
        """
        Find expert entities on a topic.

        Args:
            topic_entity_ids: Entity IDs representing the topic of interest
            entity_types: Filter by entity types (e.g., ['person', 'organization'])
            limit: Maximum number of experts to return

        Returns:
            List of ExpertResult sorted by expertise score
        """
        if not topic_entity_ids:
            return []

        # 1. Compute PPR starting from topic entities
        ppr_scores = self.ppr.compute(self.graph, topic_entity_ids)

        # 2. Compute global centrality
        pagerank_scores = self.centrality.pagerank(self.graph)

        # 3. Count mentions for each entity
        mention_counts = self._count_mentions()

        # 4. Score and rank entities
        candidates = []
        for node_id, node_data in self.graph.nodes(data=True):
            if node_data.get("node_type") != "entity":
                continue

            # Filter by entity type if specified
            if entity_types:
                if node_data.get("entity_type") not in entity_types:
                    continue

            # Skip topic entities themselves
            if node_id in topic_entity_ids:
                continue

            ppr_score = ppr_scores.get(node_id, 0)
            pagerank = pagerank_scores.get(node_id, 0)
            mentions = mention_counts.get(node_id, 0)

            # Combined expertise score
            # Weight: PPR (topic relevance) + PageRank (authority) + mentions
            expertise_score = (
                0.5 * ppr_score +
                0.3 * pagerank +
                0.2 * (mentions / max(mention_counts.values(), default=1))
            )

            if expertise_score > 0:
                # Find connected topics
                connected_topics = self._get_connected_topics(
                    node_id, topic_entity_ids
                )

                candidates.append(ExpertResult(
                    entity_id=node_id,
                    name=node_data.get("name", ""),
                    entity_type=node_data.get("entity_type", ""),
                    score=expertise_score,
                    centrality=pagerank,
                    mention_count=mentions,
                    connected_topics=connected_topics,
                ))

        # Sort by score and return top results
        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:limit]

    def _count_mentions(self) -> Dict[str, int]:
        """Count how many sources mention each entity."""
        mention_counts: Dict[str, int] = {}

        for source, target, edge_data in self.graph.edges(data=True):
            if edge_data.get("edge_type") == "mentions":
                entity_id = target
                mention_counts[entity_id] = mention_counts.get(entity_id, 0) + 1

        return mention_counts

    def _get_connected_topics(
        self,
        entity_id: str,
        topic_ids: List[str],
    ) -> List[str]:
        """Get topic entities connected to this entity."""
        connected = []

        for topic_id in topic_ids:
            # Check if there's a path
            path = self.path_finder.shortest_path(
                self.graph, entity_id, topic_id
            )
            if path and len(path) <= 3:  # Within 2 hops
                topic_data = self.graph.nodes.get(topic_id, {})
                connected.append(topic_data.get("name", topic_id))

        return connected


# =============================================================================
# CLAIM TRACING
# =============================================================================


class ClaimTracer:
    """
    Traces the provenance and support for claims in the knowledge graph.

    Helps answer questions like:
    - What sources support this claim?
    - What entities are involved?
    - Are there contradicting claims?
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.path_finder = PathFinder()

    async def trace_claim(
        self,
        claim_id: str,
        max_depth: int = 3,
    ) -> Optional[ClaimTraceResult]:
        """
        Trace the provenance of a claim.

        Args:
            claim_id: The claim node ID to trace
            max_depth: Maximum depth for path finding

        Returns:
            ClaimTraceResult with supporting sources and evidence chains
        """
        if claim_id not in self.graph:
            logger.warning(f"Claim {claim_id} not found in graph")
            return None

        claim_data = self.graph.nodes[claim_id]

        # 1. Find supporting sources
        supporting_sources = self._find_supporting_sources(claim_id)

        # 2. Find related claims
        related_claims = self._find_related_claims(claim_id)

        # 3. Build evidence chains
        evidence_chains = self._build_evidence_chains(claim_id, max_depth)

        # 4. Extract involved entities
        entities_involved = self._extract_involved_entities(claim_id)

        return ClaimTraceResult(
            claim_id=claim_id,
            statement=claim_data.get("statement", ""),
            confidence=claim_data.get("confidence", 0.5),
            supporting_sources=supporting_sources,
            related_claims=related_claims,
            evidence_chain=evidence_chains,
            entities_involved=entities_involved,
        )

    def _find_supporting_sources(self, claim_id: str) -> List[Dict[str, Any]]:
        """Find sources that support this claim."""
        sources = []

        for source_id, _, edge_data in self.graph.in_edges(claim_id, data=True):
            if edge_data.get("edge_type") == "supports":
                source_data = self.graph.nodes.get(source_id, {})
                sources.append({
                    "source_id": source_id,
                    "title": source_data.get("title", ""),
                    "source_type": source_data.get("source_type", ""),
                    "support_strength": edge_data.get("weight", 1.0),
                })

        return sources

    def _find_related_claims(self, claim_id: str) -> List[Dict[str, Any]]:
        """Find claims related to this one (same sources or entities)."""
        related = []

        # Get sources supporting this claim
        supporting_source_ids = {
            source_id
            for source_id, _, edge_data in self.graph.in_edges(claim_id, data=True)
            if edge_data.get("edge_type") == "supports"
        }

        # Find other claims supported by the same sources
        for source_id in supporting_source_ids:
            for _, other_claim_id, edge_data in self.graph.out_edges(source_id, data=True):
                if (
                    edge_data.get("edge_type") == "supports"
                    and other_claim_id != claim_id
                    and self.graph.nodes.get(other_claim_id, {}).get("node_type") == "claim"
                ):
                    claim_data = self.graph.nodes[other_claim_id]
                    related.append({
                        "claim_id": other_claim_id,
                        "statement": claim_data.get("statement", ""),
                        "claim_type": claim_data.get("claim_type", ""),
                        "shared_source": source_id,
                    })

        return related

    def _build_evidence_chains(
        self,
        claim_id: str,
        max_depth: int,
    ) -> List[List[str]]:
        """Build paths from claim through sources to supporting entities."""
        chains = []

        # Get supporting sources
        for source_id, _, edge_data in self.graph.in_edges(claim_id, data=True):
            if edge_data.get("edge_type") != "supports":
                continue

            # Get entities mentioned by this source
            for _, entity_id, mention_data in self.graph.out_edges(source_id, data=True):
                if mention_data.get("edge_type") == "mentions":
                    chain = [claim_id, source_id, entity_id]
                    chains.append(chain)

        return chains[:20]  # Limit chains

    def _extract_involved_entities(self, claim_id: str) -> List[str]:
        """Extract entities involved in a claim via its supporting sources."""
        entities = set()

        # Get supporting sources
        for source_id, _, edge_data in self.graph.in_edges(claim_id, data=True):
            if edge_data.get("edge_type") != "supports":
                continue

            # Get entities from these sources
            for _, entity_id, mention_data in self.graph.out_edges(source_id, data=True):
                if mention_data.get("edge_type") == "mentions":
                    entities.add(entity_id)

        return list(entities)


# =============================================================================
# INFLUENCE SCORING
# =============================================================================


class InfluenceCalculator:
    """
    Calculates influence scores for nodes in the knowledge graph.

    Influence combines:
    - Centrality (structural importance)
    - Citation patterns (how often cited/mentioned)
    - Reach (how many nodes are influenced)
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph
        self.centrality = CentralityCalculator()

    async def compute_influence_scores(
        self,
        node_type: Optional[str] = None,
    ) -> Dict[str, InfluenceResult]:
        """
        Compute influence scores for all nodes (or filtered by type).

        Args:
            node_type: Filter to specific node type ('entity', 'source', 'claim')

        Returns:
            Dict mapping node_id to InfluenceResult
        """
        # Compute centralities
        degree = self.centrality.degree_centrality(self.graph)
        pagerank = self.centrality.pagerank(self.graph)

        results = {}

        for node_id, node_data in self.graph.nodes(data=True):
            if node_type and node_data.get("node_type") != node_type:
                continue

            # Component scores
            degree_score = degree.get(node_id, 0)
            pagerank_score = pagerank.get(node_id, 0)

            # Citation score (incoming edges)
            citation_score = self.graph.in_degree(node_id) / max(self.graph.number_of_nodes(), 1)

            # Reach score (nodes reachable within 2 hops)
            reach = self._compute_reach(node_id, max_hops=2)
            reach_score = reach / max(self.graph.number_of_nodes(), 1)

            # Combined influence score
            influence_score = (
                0.3 * pagerank_score +
                0.3 * degree_score +
                0.2 * citation_score +
                0.2 * reach_score
            )

            # Get influenced nodes and influencers
            influenced = list(self.graph.successors(node_id))[:10]
            influencers = list(self.graph.predecessors(node_id))[:10]

            results[node_id] = InfluenceResult(
                node_id=node_id,
                node_type=node_data.get("node_type", ""),
                influence_score=influence_score,
                component_scores={
                    "pagerank": pagerank_score,
                    "degree": degree_score,
                    "citations": citation_score,
                    "reach": reach_score,
                },
                influenced_nodes=influenced,
                influenced_by=influencers,
            )

        return results

    def _compute_reach(self, node_id: str, max_hops: int = 2) -> int:
        """Count nodes reachable within max_hops."""
        if node_id not in self.graph:
            return 0

        reachable = set()
        frontier = {node_id}

        for _ in range(max_hops):
            next_frontier = set()
            for n in frontier:
                for neighbor in self.graph.successors(n):
                    if neighbor not in reachable and neighbor != node_id:
                        reachable.add(neighbor)
                        next_frontier.add(neighbor)
            frontier = next_frontier

        return len(reachable)

    async def get_top_influencers(
        self,
        node_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[InfluenceResult]:
        """Get the most influential nodes."""
        all_scores = await self.compute_influence_scores(node_type)

        sorted_results = sorted(
            all_scores.values(),
            key=lambda x: x.influence_score,
            reverse=True,
        )

        return sorted_results[:limit]


# =============================================================================
# UNIFIED ANALYSIS SERVICE
# =============================================================================


class KnowledgeGraphAnalyzer:
    """
    Unified service for knowledge graph analysis.

    Combines all analysis capabilities:
    - Graph loading
    - Expert finding
    - Claim tracing
    - Influence scoring
    """

    def __init__(self):
        self.loader = KnowledgeGraphLoader()
        self._graph: Optional[nx.DiGraph] = None

    async def load_graph(self, full: bool = True) -> None:
        """Load the knowledge graph."""
        if full:
            self._graph = await self.loader.load_full_graph()
        logger.info(f"Loaded graph: {self._graph.number_of_nodes()} nodes, {self._graph.number_of_edges()} edges")

    async def load_for_entities(self, entity_ids: List[str]) -> None:
        """Load subgraph for specific entities."""
        self._graph = await self.loader.load_entity_subgraph(entity_ids)

    @property
    def graph(self) -> Optional[nx.DiGraph]:
        return self._graph

    async def find_experts(
        self,
        topic_entity_ids: List[str],
        entity_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[ExpertResult]:
        """Find experts on a topic."""
        if self._graph is None:
            await self.load_for_entities(topic_entity_ids)

        finder = ExpertFinder(self._graph)
        return await finder.find_experts(topic_entity_ids, entity_types, limit)

    async def trace_claim(
        self,
        claim_id: str,
        max_depth: int = 3,
    ) -> Optional[ClaimTraceResult]:
        """Trace a claim's provenance."""
        if self._graph is None:
            await self.load_graph()

        tracer = ClaimTracer(self._graph)
        return await tracer.trace_claim(claim_id, max_depth)

    async def compute_influence(
        self,
        node_type: Optional[str] = None,
    ) -> Dict[str, InfluenceResult]:
        """Compute influence scores."""
        if self._graph is None:
            await self.load_graph()

        calculator = InfluenceCalculator(self._graph)
        return await calculator.compute_influence_scores(node_type)

    async def get_top_influencers(
        self,
        node_type: Optional[str] = None,
        limit: int = 10,
    ) -> List[InfluenceResult]:
        """Get most influential nodes."""
        if self._graph is None:
            await self.load_graph()

        calculator = InfluenceCalculator(self._graph)
        return await calculator.get_top_influencers(node_type, limit)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def find_experts(
    topic_entity_ids: List[str],
    entity_types: Optional[List[str]] = None,
    limit: int = 10,
) -> List[ExpertResult]:
    """Convenience function to find experts on a topic."""
    analyzer = KnowledgeGraphAnalyzer()
    await analyzer.load_for_entities(topic_entity_ids)
    return await analyzer.find_experts(topic_entity_ids, entity_types, limit)


async def trace_claim(claim_id: str) -> Optional[ClaimTraceResult]:
    """Convenience function to trace a claim."""
    analyzer = KnowledgeGraphAnalyzer()
    await analyzer.load_graph()
    return await analyzer.trace_claim(claim_id)


async def get_influence_scores(
    node_type: Optional[str] = None,
) -> Dict[str, InfluenceResult]:
    """Convenience function to compute influence scores."""
    analyzer = KnowledgeGraphAnalyzer()
    await analyzer.load_graph()
    return await analyzer.compute_influence(node_type)
