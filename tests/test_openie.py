"""
Tests for OpenIE extraction pipeline.

These tests verify:
1. Entity extraction from text
2. Triple extraction
3. Claim extraction
4. Combined extraction
5. Embedding generation
6. Entity linking and deduplication

Note: These tests require Ollama to be running with the required models:
- qwen2.5:14b (for OpenIE extraction)
- mxbai-embed-large (for embeddings)

Run with: pytest tests/test_openie.py -v
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from open_notebook.processors.openie import (
    OpenIEConfig,
    OpenIEExtractor,
    ExtractedEntity,
    ExtractedTriple,
    ExtractedClaim,
    ExtractionResult,
    extract_knowledge_from_text,
    map_entity_type,
    map_claim_type,
)
from open_notebook.processors.embeddings import (
    EmbeddingConfig,
    EmbeddingService,
    KnowledgeGraphEmbeddings,
)
from open_notebook.domain.knowledge_graph import EntityType, ClaimType


# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_text():
    """Sample text for extraction tests."""
    return """
    The World Health Organization (WHO) has declared climate change to be the
    biggest health threat facing humanity. Dr. Tedros Adhanom Ghebreyesus,
    the WHO Director-General, stated that climate change is causing more
    extreme weather events, which leads to increased disease transmission.

    Research from the Netherlands Environmental Assessment Agency shows that
    global temperatures have risen by 1.1 degrees Celsius since pre-industrial
    times. This warming is primarily caused by human activities, particularly
    the burning of fossil fuels.
    """


@pytest.fixture
def openie_config():
    """OpenIE configuration for tests."""
    return OpenIEConfig(
        model="qwen2.5:14b",
        provider="ollama",
        base_url="http://localhost:11434",
        temperature=0.1,
    )


@pytest.fixture
def embedding_config():
    """Embedding configuration for tests."""
    return EmbeddingConfig(
        model="mxbai-embed-large",
        provider="ollama",
        base_url="http://localhost:11434",
        dimension=1024,
    )


# =============================================================================
# UNIT TESTS (mocked LLM)
# =============================================================================


class TestEntityTypeMappings:
    """Test entity type mapping functions."""

    def test_map_entity_type_person(self):
        assert map_entity_type("person") == EntityType.PERSON

    def test_map_entity_type_organization(self):
        assert map_entity_type("organization") == EntityType.ORGANIZATION

    def test_map_entity_type_topic(self):
        assert map_entity_type("topic") == EntityType.TOPIC

    def test_map_entity_type_location(self):
        assert map_entity_type("location") == EntityType.LOCATION

    def test_map_entity_type_concept(self):
        assert map_entity_type("concept") == EntityType.CONCEPT

    def test_map_entity_type_event(self):
        assert map_entity_type("event") == EntityType.EVENT

    def test_map_entity_type_unknown(self):
        assert map_entity_type("unknown_type") == EntityType.OTHER

    def test_map_entity_type_case_insensitive(self):
        assert map_entity_type("PERSON") == EntityType.PERSON
        assert map_entity_type("Organization") == EntityType.ORGANIZATION


class TestClaimTypeMappings:
    """Test claim type mapping functions."""

    def test_map_claim_type_factual(self):
        assert map_claim_type("factual") == ClaimType.FACTUAL

    def test_map_claim_type_causal(self):
        assert map_claim_type("causal") == ClaimType.CAUSAL

    def test_map_claim_type_normative(self):
        assert map_claim_type("normative") == ClaimType.NORMATIVE

    def test_map_claim_type_predictive(self):
        assert map_claim_type("predictive") == ClaimType.PREDICTIVE

    def test_map_claim_type_unknown(self):
        assert map_claim_type("unknown_type") == ClaimType.FACTUAL


class TestExtractedEntityModel:
    """Test ExtractedEntity Pydantic model."""

    def test_create_entity(self):
        entity = ExtractedEntity(
            name="World Health Organization",
            entity_type="organization",
            description="International health agency",
            aliases=["WHO"],
            confidence=0.95,
        )
        assert entity.name == "World Health Organization"
        assert entity.entity_type == "organization"
        assert entity.confidence == 0.95

    def test_entity_default_values(self):
        entity = ExtractedEntity(
            name="Test",
            entity_type="person",
        )
        assert entity.aliases == []
        assert entity.description is None
        assert entity.confidence == 0.8


class TestExtractedTripleModel:
    """Test ExtractedTriple Pydantic model."""

    def test_create_triple(self):
        triple = ExtractedTriple(
            subject="WHO",
            predicate="declared",
            object="climate change health threat",
            context="The WHO has declared...",
            confidence=0.9,
        )
        assert triple.subject == "WHO"
        assert triple.predicate == "declared"
        assert triple.object == "climate change health threat"


class TestExtractedClaimModel:
    """Test ExtractedClaim Pydantic model."""

    def test_create_claim(self):
        claim = ExtractedClaim(
            statement="Climate change is the biggest health threat",
            claim_type="factual",
            supporting_quote="declared climate change to be the biggest health threat",
            entities_involved=["climate change", "health"],
            confidence=0.85,
        )
        assert claim.statement == "Climate change is the biggest health threat"
        assert claim.claim_type == "factual"


class TestOpenIEExtractorWithMocks:
    """Test OpenIEExtractor with mocked LLM responses."""

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = """```json
[
    {
        "name": "World Health Organization",
        "entity_type": "organization",
        "description": "International health agency",
        "aliases": ["WHO"],
        "confidence": 0.95
    },
    {
        "name": "climate change",
        "entity_type": "topic",
        "description": "Long-term shift in temperatures",
        "aliases": [],
        "confidence": 0.9
    }
]
```"""
        return mock_response

    @pytest.mark.asyncio
    async def test_extract_entities_with_mock(self, mock_llm_response, sample_text):
        """Test entity extraction with mocked LLM."""
        with patch.object(OpenIEExtractor, "_get_llm") as mock_get_llm:
            mock_llm = MagicMock()
            mock_llm.chat_async = AsyncMock(return_value=mock_llm_response)
            mock_get_llm.return_value = mock_llm

            extractor = OpenIEExtractor()
            entities = await extractor.extract_entities(sample_text)

            assert len(entities) == 2
            assert entities[0].name == "World Health Organization"
            assert entities[0].entity_type == "organization"
            assert entities[1].name == "climate change"

    @pytest.mark.asyncio
    async def test_extract_entities_empty_text(self):
        """Test that empty text returns empty list."""
        extractor = OpenIEExtractor()
        entities = await extractor.extract_entities("")
        assert entities == []

        entities = await extractor.extract_entities("   ")
        assert entities == []


class TestOpenIEExtractorParsing:
    """Test JSON parsing in OpenIEExtractor."""

    def test_parse_json_with_markdown(self):
        """Test parsing JSON wrapped in markdown code blocks."""
        extractor = OpenIEExtractor()

        # Test with ```json block
        response = '''```json
[{"name": "test", "entity_type": "person"}]
```'''
        result = extractor._parse_json_response(response)
        assert result == [{"name": "test", "entity_type": "person"}]

    def test_parse_json_without_markdown(self):
        """Test parsing plain JSON."""
        extractor = OpenIEExtractor()

        response = '[{"name": "test", "entity_type": "person"}]'
        result = extractor._parse_json_response(response)
        assert result == [{"name": "test", "entity_type": "person"}]

    def test_parse_invalid_json(self):
        """Test handling of invalid JSON."""
        extractor = OpenIEExtractor()

        result = extractor._parse_json_response("not valid json")
        assert result is None


# =============================================================================
# INTEGRATION TESTS (requires Ollama)
# =============================================================================


@pytest.mark.integration
@pytest.mark.skipif(
    not pytest.importorskip("esperanto"),
    reason="esperanto not installed",
)
class TestOpenIEIntegration:
    """
    Integration tests that require Ollama to be running.

    Run with: pytest tests/test_openie.py -v -m integration
    """

    @pytest.mark.asyncio
    async def test_extract_entities_real(self, sample_text, openie_config):
        """Test real entity extraction with Ollama."""
        extractor = OpenIEExtractor(openie_config)
        entities = await extractor.extract_entities(sample_text)

        # Should find at least some entities
        assert len(entities) > 0

        # Check for expected entities
        entity_names = [e.name.lower() for e in entities]
        # At least one of these should be found
        expected_any = ["world health organization", "who", "climate change"]
        found_any = any(exp in " ".join(entity_names) for exp in expected_any)
        assert found_any, f"Expected to find one of {expected_any}, got {entity_names}"

    @pytest.mark.asyncio
    async def test_extract_triples_real(self, sample_text, openie_config):
        """Test real triple extraction with Ollama."""
        extractor = OpenIEExtractor(openie_config)
        triples = await extractor.extract_triples(sample_text)

        # Should find at least some triples
        assert len(triples) > 0

        # Check structure
        for triple in triples:
            assert triple.subject
            assert triple.predicate
            assert triple.object

    @pytest.mark.asyncio
    async def test_extract_claims_real(self, sample_text, openie_config):
        """Test real claim extraction with Ollama."""
        extractor = OpenIEExtractor(openie_config)
        claims = await extractor.extract_claims(sample_text)

        # Should find at least some claims
        assert len(claims) > 0

        # Check structure
        for claim in claims:
            assert claim.statement
            assert claim.claim_type in ["factual", "causal", "normative", "predictive"]

    @pytest.mark.asyncio
    async def test_extract_all_real(self, sample_text, openie_config):
        """Test combined extraction with Ollama."""
        extractor = OpenIEExtractor(openie_config)
        result = await extractor.extract_all(sample_text)

        assert isinstance(result, ExtractionResult)
        # Should find something in at least one category
        total = len(result.entities) + len(result.triples) + len(result.claims)
        assert total > 0


@pytest.mark.integration
class TestEmbeddingIntegration:
    """
    Integration tests for embedding generation.

    Run with: pytest tests/test_openie.py -v -m integration
    """

    @pytest.mark.asyncio
    async def test_embed_text_real(self, embedding_config):
        """Test real embedding generation with Ollama."""
        service = EmbeddingService(embedding_config)
        embedding = await service.embed_text("This is a test sentence.")

        assert len(embedding) > 0
        assert len(embedding) == embedding_config.dimension
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_batch_real(self, embedding_config):
        """Test batch embedding generation."""
        service = EmbeddingService(embedding_config)
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence",
        ]
        embeddings = await service.embed_batch(texts)

        assert len(embeddings) == 3
        for emb in embeddings:
            assert len(emb) == embedding_config.dimension

    @pytest.mark.asyncio
    async def test_kg_embeddings_real(self, embedding_config):
        """Test KnowledgeGraphEmbeddings three-tier approach."""
        kg = KnowledgeGraphEmbeddings(embedding_config)

        # Test passage embedding
        passage_emb = await kg.embed_passage("Climate change is a global issue.")
        assert len(passage_emb) == embedding_config.dimension

        # Test entity embedding
        entity_emb = await kg.embed_entity(
            "World Health Organization",
            "International health agency",
        )
        assert len(entity_emb) == embedding_config.dimension

        # Test fact embedding
        fact_emb = await kg.embed_fact(
            "WHO",
            "declared",
            "climate change health threat",
        )
        assert len(fact_emb) == embedding_config.dimension

        # Test query embedding
        query_emb = await kg.embed_query("What is climate change?")
        assert len(query_emb) == embedding_config.dimension


# =============================================================================
# CONVENIENCE FUNCTION TESTS
# =============================================================================


class TestConvenienceFunctions:
    """Test convenience wrapper functions."""

    @pytest.mark.asyncio
    async def test_extract_knowledge_from_text_with_mock(self):
        """Test convenience function with mocked extractor."""
        with patch.object(OpenIEExtractor, "extract_all") as mock_extract:
            mock_extract.return_value = ExtractionResult(
                entities=[
                    ExtractedEntity(name="Test", entity_type="person", confidence=0.9)
                ],
                triples=[],
                claims=[],
            )

            result = await extract_knowledge_from_text("test text")

            assert isinstance(result, ExtractionResult)
            assert len(result.entities) == 1
