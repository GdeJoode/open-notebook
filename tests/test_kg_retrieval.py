"""
Tests for Knowledge Graph Retrieval Pipeline.

These tests verify:
1. Fact scoring (embedding, claim, citation)
2. Entity extraction
3. PPR-based retrieval
4. Dense retrieval
5. Hybrid retrieval
6. End-to-end pipeline

Run with: pytest tests/test_kg_retrieval.py -v
"""

import pytest
import networkx as nx
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Dict, Any

from open_notebook.graphs.kg_retrieval import (
    FactScorer,
    FactScore,
    EntityExtractor,
    PPRRetriever,
    DenseRetriever,
    LLMReranker,
    KnowledgeGraphRetriever,
    RetrievalConfig,
    RetrievalMode,
    RetrievalResult,
    hipporag_retrieve,
    dense_retrieve,
    hybrid_retrieve,
)


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def mock_embedding():
    """Mock embedding vector."""
    return [0.1] * 1024  # 1024-dimensional embedding


@pytest.fixture
def mock_embeddings_service(mock_embedding):
    """Mock the embeddings service."""
    with patch(
        "open_notebook.graphs.kg_retrieval.get_kg_embeddings"
    ) as mock_get:
        mock_service = MagicMock()

        async def mock_embed_query(text):
            return mock_embedding

        mock_service.embed_query = mock_embed_query
        mock_get.return_value = mock_service
        yield mock_service


@pytest.fixture
def mock_repo_query():
    """Mock the repository query function."""
    with patch(
        "open_notebook.graphs.kg_retrieval.repo_query"
    ) as mock_query:
        yield mock_query


@pytest.fixture
def sample_fact_scores():
    """Sample fact scores for testing."""
    return [
        FactScore(
            fact_id="mentions:1",
            fact_text="Climate change affects biodiversity",
            score=0.9,
            embedding_similarity=0.85,
            source_entity="source:1",
            target_entity="entity:climate",
        ),
        FactScore(
            fact_id="mentions:2",
            fact_text="WHO reports on health impacts",
            score=0.8,
            embedding_similarity=0.75,
            source_entity="source:2",
            target_entity="entity:who",
        ),
        FactScore(
            fact_id="relates_to:1",
            fact_text="Temperature relates to weather",
            score=0.7,
            embedding_similarity=0.7,
            source_entity="entity:temp",
            target_entity="entity:weather",
        ),
    ]


@pytest.fixture
def sample_entity_scores():
    """Sample entity scores for PPR seeding."""
    return {
        "entity:climate": 0.9,
        "entity:who": 0.7,
        "entity:biodiversity": 0.6,
    }


@pytest.fixture
def sample_graph():
    """Create a sample knowledge graph for testing."""
    G = nx.DiGraph()

    # Add source nodes
    G.add_node("source:1", node_type="source", title="Climate Report")
    G.add_node("source:2", node_type="source", title="Health Report")
    G.add_node("source:3", node_type="source", title="Weather Analysis")

    # Add entity nodes
    G.add_node("entity:climate", node_type="entity", name="Climate Change")
    G.add_node("entity:who", node_type="entity", name="WHO")
    G.add_node("entity:biodiversity", node_type="entity", name="Biodiversity")

    # Add edges
    G.add_edge("source:1", "entity:climate", edge_type="mentions", weight=1.0)
    G.add_edge("source:1", "entity:biodiversity", edge_type="mentions", weight=1.0)
    G.add_edge("source:2", "entity:who", edge_type="mentions", weight=1.0)
    G.add_edge("source:2", "entity:climate", edge_type="mentions", weight=0.8)
    G.add_edge("entity:climate", "entity:biodiversity", edge_type="relates_to", weight=0.9)

    return G


# =============================================================================
# FACT SCORER TESTS
# =============================================================================


class TestFactScorer:
    """Tests for fact scoring."""

    @pytest.mark.asyncio
    async def test_score_facts_empty_query(self, mock_embeddings_service, mock_repo_query):
        """Test scoring with empty embedding result."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=[])

        scorer = FactScorer()
        scores = await scorer.score_facts("test query")

        assert scores == []

    @pytest.mark.asyncio
    async def test_score_facts_combines_sources(
        self, mock_embeddings_service, mock_repo_query, mock_embedding
    ):
        """Test that fact scoring combines multiple sources."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)

        # Mock different query results
        async def mock_query(query, params=None):
            if "mentions" in query and "fact_embedding" in query:
                return [
                    {
                        "id": "mentions:1",
                        "fact_text": "Test fact",
                        "score": 0.9,
                        "source_entity": "source:1",
                        "target_entity": "entity:1",
                    }
                ]
            if "relates_to" in query:
                return []
            if "claim" in query:
                return [
                    {"id": "claim:1", "statement": "Test claim", "score": 0.8}
                ]
            if "cites" in query:
                return []
            return []

        mock_repo_query.side_effect = mock_query

        scorer = FactScorer()
        scores = await scorer.score_facts("test query", limit=10)

        # Should have combined results from different sources
        assert len(scores) >= 1

    @pytest.mark.asyncio
    async def test_score_facts_with_weights(
        self, mock_embeddings_service, mock_repo_query, mock_embedding
    ):
        """Test that scoring weights are applied correctly."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)

        async def mock_query(query, params=None):
            if "mentions" in query and "fact_embedding" in query:
                return [
                    {
                        "id": "fact:1",
                        "fact_text": "Test",
                        "score": 1.0,
                        "source_entity": "s",
                        "target_entity": "t",
                    }
                ]
            return []

        mock_repo_query.side_effect = mock_query

        config = RetrievalConfig(
            fact_embedding_weight=0.5,
            claim_similarity_weight=0.3,
            citation_weight=0.2,
        )
        scorer = FactScorer(config)
        scores = await scorer.score_facts("test")

        # Score should be 0.5 * 1.0 = 0.5 (only embedding similarity)
        if scores:
            assert scores[0].score == pytest.approx(0.5, rel=0.1)


# =============================================================================
# ENTITY EXTRACTOR TESTS
# =============================================================================


class TestEntityExtractor:
    """Tests for entity extraction."""

    @pytest.mark.asyncio
    async def test_extract_from_facts(
        self, mock_embeddings_service, mock_repo_query, sample_fact_scores, mock_embedding
    ):
        """Test entity extraction from fact scores."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)
        mock_repo_query.return_value = []

        extractor = EntityExtractor()
        entities = await extractor.extract_entities(
            "climate change",
            fact_scores=sample_fact_scores,
            limit=10,
        )

        # Should extract entities from fact scores
        assert len(entities) > 0
        # Source and target entities from facts should be included
        assert any("climate" in e for e in entities.keys()) or any(
            "who" in e for e in entities.keys()
        )

    @pytest.mark.asyncio
    async def test_extract_by_embedding(
        self, mock_embeddings_service, mock_repo_query, mock_embedding
    ):
        """Test entity extraction by embedding similarity."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)

        async def mock_query(query, params=None):
            if "entity" in query and "embedding" in query:
                return [
                    {"id": "entity:1", "score": 0.95},
                    {"id": "entity:2", "score": 0.85},
                ]
            return []

        mock_repo_query.side_effect = mock_query

        extractor = EntityExtractor()
        entities = await extractor.extract_entities("test query")

        assert len(entities) > 0
        assert "entity:1" in entities

    @pytest.mark.asyncio
    async def test_extract_normalizes_scores(
        self, mock_embeddings_service, mock_repo_query, mock_embedding
    ):
        """Test that entity scores are normalized."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)

        async def mock_query(query, params=None):
            if "entity" in query and "embedding" in query:
                return [
                    {"id": "entity:1", "score": 0.8},
                    {"id": "entity:2", "score": 0.4},
                ]
            return []

        mock_repo_query.side_effect = mock_query

        extractor = EntityExtractor()
        entities = await extractor.extract_entities("test")

        # All scores should be <= 1.0
        for score in entities.values():
            assert score <= 1.0


# =============================================================================
# PPR RETRIEVER TESTS
# =============================================================================


class TestPPRRetriever:
    """Tests for PPR-based retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_empty_seeds(self):
        """Test retrieval with no entity seeds."""
        retriever = PPRRetriever()
        results = await retriever.retrieve({})

        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_with_seeds(self, sample_graph, sample_entity_scores):
        """Test PPR retrieval with entity seeds."""
        retriever = PPRRetriever()

        # Mock the graph loader
        with patch.object(retriever, "_get_graph", return_value=sample_graph):
            results = await retriever.retrieve(sample_entity_scores, top_k=5)

        # Should return source nodes
        assert all(r[0].startswith("source:") for r in results)
        # Should be sorted by score descending
        if len(results) > 1:
            assert results[0][1] >= results[1][1]

    @pytest.mark.asyncio
    async def test_retrieve_filters_to_sources(self, sample_graph, sample_entity_scores):
        """Test that only source nodes are returned."""
        retriever = PPRRetriever()

        with patch.object(retriever, "_get_graph", return_value=sample_graph):
            results = await retriever.retrieve(sample_entity_scores, top_k=10)

        # All results should be source nodes
        for source_id, score in results:
            assert sample_graph.nodes[source_id].get("node_type") == "source"


# =============================================================================
# DENSE RETRIEVER TESTS
# =============================================================================


class TestDenseRetriever:
    """Tests for dense retrieval."""

    @pytest.mark.asyncio
    async def test_retrieve_basic(
        self, mock_embeddings_service, mock_repo_query, mock_embedding
    ):
        """Test basic dense retrieval."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)
        mock_repo_query.return_value = [
            {"id": "source:1", "score": 0.95},
            {"id": "source:2", "score": 0.85},
        ]

        retriever = DenseRetriever()
        results = await retriever.retrieve("test query", top_k=5)

        assert len(results) == 2
        assert results[0][0] == "source:1"
        assert results[0][1] == 0.95

    @pytest.mark.asyncio
    async def test_retrieve_empty_embedding(self, mock_embeddings_service):
        """Test retrieval with failed embedding."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=[])

        retriever = DenseRetriever()
        results = await retriever.retrieve("test")

        assert results == []


# =============================================================================
# RERANKER TESTS
# =============================================================================


class TestLLMReranker:
    """Tests for LLM reranking."""

    @pytest.mark.asyncio
    async def test_rerank_returns_original_order(self, mock_repo_query):
        """Test that reranker returns original order (placeholder implementation)."""
        mock_repo_query.return_value = [
            {"id": "source:1", "title": "Test", "content": "Content"},
        ]

        candidates = [
            ("source:1", 0.9),
            ("source:2", 0.8),
        ]

        reranker = LLMReranker()
        results = await reranker.rerank("test query", candidates, top_k=2)

        assert len(results) == 2
        # Should maintain order (placeholder implementation)
        assert results[0][0] == "source:1"

    @pytest.mark.asyncio
    async def test_rerank_empty_candidates(self):
        """Test reranking with no candidates."""
        reranker = LLMReranker()
        results = await reranker.rerank("test", [], top_k=5)

        assert results == []


# =============================================================================
# UNIFIED RETRIEVER TESTS
# =============================================================================


class TestKnowledgeGraphRetriever:
    """Tests for the unified retrieval pipeline."""

    @pytest.mark.asyncio
    async def test_retrieve_hipporag_mode(
        self,
        mock_embeddings_service,
        mock_repo_query,
        sample_graph,
        mock_embedding,
    ):
        """Test HippoRAG retrieval mode."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)

        # Mock query responses
        async def mock_query(query, params=None):
            if "mentions" in query and "fact_embedding" in query:
                return [
                    {
                        "id": "mentions:1",
                        "fact_text": "Test",
                        "score": 0.9,
                        "source_entity": "source:1",
                        "target_entity": "entity:climate",
                    }
                ]
            if "entity" in query and "embedding" in query:
                return [{"id": "entity:climate", "score": 0.9}]
            if "source" in query and "embedding" in query:
                return [{"id": "source:1", "score": 0.85}]
            if "SELECT id, title, content" in query:
                return [{"id": "source:1", "title": "Test", "content": "Content"}]
            return []

        mock_repo_query.side_effect = mock_query

        config = RetrievalConfig(mode=RetrievalMode.HIPPORAG)
        retriever = KnowledgeGraphRetriever(config)

        # Mock the PPR retriever's graph
        with patch.object(
            retriever._ppr_retriever, "_get_graph", return_value=sample_graph
        ):
            results = await retriever.retrieve("climate change", top_k=5)

        assert isinstance(results, list)
        # Results should be RetrievalResult objects
        for r in results:
            assert isinstance(r, RetrievalResult)

    @pytest.mark.asyncio
    async def test_retrieve_dense_mode(
        self, mock_embeddings_service, mock_repo_query, mock_embedding
    ):
        """Test dense retrieval mode."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)

        async def mock_query(query, params=None):
            if "source" in query and "embedding" in query:
                return [
                    {"id": "source:1", "score": 0.95},
                    {"id": "source:2", "score": 0.85},
                ]
            if "SELECT id, title, content" in query:
                return [
                    {"id": "source:1", "title": "Test 1", "content": "Content 1"},
                    {"id": "source:2", "title": "Test 2", "content": "Content 2"},
                ]
            return []

        mock_repo_query.side_effect = mock_query

        config = RetrievalConfig(mode=RetrievalMode.DENSE)
        retriever = KnowledgeGraphRetriever(config)
        results = await retriever.retrieve("test query", top_k=5)

        assert len(results) > 0
        # First result should have highest score
        assert results[0].source_id == "source:1"

    @pytest.mark.asyncio
    async def test_retrieve_hybrid_mode(
        self,
        mock_embeddings_service,
        mock_repo_query,
        sample_graph,
        mock_embedding,
    ):
        """Test hybrid retrieval mode."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)

        async def mock_query(query, params=None):
            if "mentions" in query and "fact_embedding" in query:
                return []
            if "entity" in query and "embedding" in query:
                return [{"id": "entity:1", "score": 0.8}]
            if "source" in query and "embedding" in query:
                return [
                    {"id": "source:1", "score": 0.9},
                    {"id": "source:2", "score": 0.7},
                ]
            if "SELECT id, title, content" in query:
                return [
                    {"id": "source:1", "title": "Test", "content": "Content"},
                    {"id": "source:2", "title": "Test 2", "content": "Content 2"},
                ]
            return []

        mock_repo_query.side_effect = mock_query

        config = RetrievalConfig(mode=RetrievalMode.HYBRID)
        retriever = KnowledgeGraphRetriever(config)

        with patch.object(
            retriever._ppr_retriever, "_get_graph", return_value=sample_graph
        ):
            results = await retriever.retrieve("test", top_k=5)

        assert isinstance(results, list)

    def test_combine_results_hipporag(self):
        """Test result combination in HippoRAG mode."""
        config = RetrievalConfig(mode=RetrievalMode.HIPPORAG)
        retriever = KnowledgeGraphRetriever(config)

        ppr_results = [("source:1", 0.9), ("source:2", 0.7)]
        dense_results = [("source:3", 0.95)]

        combined = retriever._combine_results(ppr_results, dense_results)

        # In HippoRAG mode, PPR results take precedence
        assert combined[0][0] == "source:1"

    def test_combine_results_dense(self):
        """Test result combination in dense mode."""
        config = RetrievalConfig(mode=RetrievalMode.DENSE)
        retriever = KnowledgeGraphRetriever(config)

        ppr_results = [("source:1", 0.9)]
        dense_results = [("source:3", 0.95)]

        combined = retriever._combine_results(ppr_results, dense_results)

        # In dense mode, dense results are returned
        assert combined[0][0] == "source:3"

    def test_combine_results_hybrid(self):
        """Test result combination in hybrid mode."""
        config = RetrievalConfig(mode=RetrievalMode.HYBRID)
        retriever = KnowledgeGraphRetriever(config)

        ppr_results = [("source:1", 0.8), ("source:2", 0.6)]
        dense_results = [("source:1", 0.9), ("source:3", 0.7)]

        combined = retriever._combine_results(ppr_results, dense_results)

        # source:1 should have combined score
        source1_score = next(s for id, s in combined if id == "source:1")
        # Combined: 0.6 * (0.8/0.8) + 0.4 * (0.9/0.9) = 0.6 + 0.4 = 1.0
        assert source1_score == pytest.approx(1.0, rel=0.1)


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    @pytest.mark.asyncio
    async def test_hipporag_retrieve(
        self, mock_embeddings_service, mock_repo_query, mock_embedding
    ):
        """Test hipporag_retrieve convenience function."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)
        mock_repo_query.return_value = []

        results = await hipporag_retrieve("test query", top_k=5)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_dense_retrieve(
        self, mock_embeddings_service, mock_repo_query, mock_embedding
    ):
        """Test dense_retrieve convenience function."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)

        async def mock_query(query, params=None):
            if "source" in query and "embedding" in query:
                return [{"id": "source:1", "score": 0.9}]
            if "SELECT id, title, content" in query:
                return [{"id": "source:1", "title": "Test", "content": "Content"}]
            return []

        mock_repo_query.side_effect = mock_query

        results = await dense_retrieve("test query", top_k=5)

        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_hybrid_retrieve(
        self, mock_embeddings_service, mock_repo_query, mock_embedding
    ):
        """Test hybrid_retrieve convenience function."""
        mock_embeddings_service.embed_query = AsyncMock(return_value=mock_embedding)

        async def mock_query(query, params=None):
            if "source" in query and "embedding" in query:
                return [{"id": "source:1", "score": 0.9}]
            if "SELECT id, title, content" in query:
                return [{"id": "source:1", "title": "Test", "content": "Content"}]
            return []

        mock_repo_query.side_effect = mock_query

        results = await hybrid_retrieve("test query", top_k=5)

        assert isinstance(results, list)


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestRetrievalConfig:
    """Tests for retrieval configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RetrievalConfig()

        assert config.mode == RetrievalMode.HIPPORAG
        assert config.ppr_damping == 0.5
        assert config.top_k_results == 10
        assert config.enable_reranking is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = RetrievalConfig(
            mode=RetrievalMode.HYBRID,
            ppr_damping=0.7,
            top_k_results=20,
            enable_reranking=True,
        )

        assert config.mode == RetrievalMode.HYBRID
        assert config.ppr_damping == 0.7
        assert config.top_k_results == 20
        assert config.enable_reranking is True


# =============================================================================
# RESULT DATACLASS TESTS
# =============================================================================


class TestRetrievalResult:
    """Tests for RetrievalResult dataclass."""

    def test_result_creation(self):
        """Test creating a retrieval result."""
        result = RetrievalResult(
            source_id="source:1",
            score=0.95,
            title="Test Document",
            content_preview="This is a preview...",
            ppr_score=0.9,
            dense_score=0.85,
        )

        assert result.source_id == "source:1"
        assert result.score == 0.95
        assert result.ppr_score == 0.9
        assert result.dense_score == 0.85

    def test_result_defaults(self):
        """Test default values in retrieval result."""
        result = RetrievalResult(source_id="source:1", score=0.5)

        assert result.title is None
        assert result.content_preview is None
        assert result.ppr_score == 0.0
        assert result.dense_score == 0.0
        assert result.matched_entities == []


class TestFactScore:
    """Tests for FactScore dataclass."""

    def test_fact_score_creation(self):
        """Test creating a fact score."""
        fact = FactScore(
            fact_id="mentions:1",
            fact_text="Climate affects weather",
            score=0.85,
            embedding_similarity=0.9,
            claim_similarity=0.7,
            source_entity="source:1",
            target_entity="entity:climate",
        )

        assert fact.fact_id == "mentions:1"
        assert fact.score == 0.85
        assert fact.embedding_similarity == 0.9
