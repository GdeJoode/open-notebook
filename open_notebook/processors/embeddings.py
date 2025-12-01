"""
Embedding Service for Knowledge Graph

Provides embeddings using local Ollama model (mxbai-embed-large).
Supports three-tier embedding generation:
1. Passage embeddings (for sources/chunks)
2. Entity embeddings (for entities)
3. Fact embeddings (for relationship edges)

See docs/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md Phase 2.3 for documentation.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from esperanto import AIFactory, EmbeddingModel
from loguru import logger


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation."""

    model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
    )
    provider: str = "ollama"
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "EMBEDDING_OLLAMA_BASE_URL", "http://localhost:11434"
        )
    )
    dimension: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSION", "1024"))
    )
    batch_size: int = 32  # Number of texts to embed in one batch


# =============================================================================
# EMBEDDING SERVICE
# =============================================================================


class EmbeddingService:
    """
    Service for generating embeddings using local Ollama model.

    Supports:
    - Single text embedding
    - Batch embedding
    - Instruction-prefixed embeddings (for retrieval optimization)
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self._model: Optional[EmbeddingModel] = None

    def _get_model(self) -> EmbeddingModel:
        """Get or create the embedding model instance."""
        if self._model is None:
            logger.info(
                f"Initializing embedding model: {self.config.provider}/{self.config.model}"
            )
            self._model = AIFactory.create_embedding(
                provider=self.config.provider,
                model_name=self.config.model,
                config={"base_url": self.config.base_url},
            )
        return self._model

    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.

        Args:
            text: The text to embed

        Returns:
            List of floats representing the embedding vector
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return []

        model = self._get_model()

        try:
            result = await model.embed_async([text])
            if result and result.data and len(result.data) > 0:
                return result.data[0].embedding
            return []
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        # Filter empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_texts.append(text)
                valid_indices.append(i)

        if not valid_texts:
            return [[] for _ in texts]

        model = self._get_model()
        embeddings = [[] for _ in texts]

        try:
            # Process in batches
            for batch_start in range(0, len(valid_texts), self.config.batch_size):
                batch_end = min(batch_start + self.config.batch_size, len(valid_texts))
                batch = valid_texts[batch_start:batch_end]
                batch_indices = valid_indices[batch_start:batch_end]

                result = await model.embed_async(batch)
                if result and result.data:
                    for j, emb_data in enumerate(result.data):
                        original_idx = batch_indices[j]
                        embeddings[original_idx] = emb_data.embedding

            logger.info(f"Generated embeddings for {len(valid_texts)} texts")
            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding generation failed: {e}")
            return [[] for _ in texts]

    async def embed_with_instruction(
        self, text: str, instruction: str = "query"
    ) -> List[float]:
        """
        Generate embedding with instruction prefix for retrieval optimization.

        Some embedding models support instruction-prefixed embeddings that
        optimize for specific use cases (query vs document).

        Args:
            text: The text to embed
            instruction: The instruction type (e.g., "query", "document")

        Returns:
            List of floats representing the embedding vector
        """
        # For mxbai-embed-large, we can prepend instruction
        # Format: "Represent this [instruction] for retrieval: [text]"
        if instruction == "query":
            prefixed_text = f"Represent this sentence for searching relevant passages: {text}"
        elif instruction == "document":
            prefixed_text = f"Represent this document for retrieval: {text}"
        else:
            prefixed_text = text

        return await self.embed_text(prefixed_text)


# =============================================================================
# THREE-TIER EMBEDDING FUNCTIONS
# =============================================================================


class KnowledgeGraphEmbeddings:
    """
    Three-tier embedding generation for Knowledge Graph.

    Tier 1: Passage embeddings (sources/chunks) - optimized for document retrieval
    Tier 2: Entity embeddings (entities) - semantic similarity for deduplication
    Tier 3: Fact embeddings (edges) - relationship/claim retrieval
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.service = EmbeddingService(config)

    async def embed_passage(self, text: str) -> List[float]:
        """
        Generate passage embedding for a source document or chunk.

        Optimized for document retrieval scenarios.
        """
        return await self.service.embed_with_instruction(text, "document")

    async def embed_entity(self, name: str, description: Optional[str] = None) -> List[float]:
        """
        Generate entity embedding for deduplication and similarity.

        Combines name with description for richer semantic representation.
        """
        content = name
        if description:
            content = f"{name}: {description}"
        return await self.service.embed_text(content)

    async def embed_fact(
        self,
        subject: str,
        predicate: str,
        obj: str,
        context: Optional[str] = None,
    ) -> List[float]:
        """
        Generate fact embedding for a triple/relationship.

        HippoRAG-style fact embedding on edges for relationship retrieval.
        """
        # Format fact as natural language
        fact_text = f"{subject} {predicate} {obj}"
        if context:
            fact_text = f"{fact_text}. Context: {context}"
        return await self.service.embed_with_instruction(fact_text, "document")

    async def embed_claim(self, statement: str) -> List[float]:
        """
        Generate embedding for a claim statement.
        """
        return await self.service.embed_with_instruction(statement, "document")

    async def embed_query(self, query: str) -> List[float]:
        """
        Generate query embedding for retrieval.

        Optimized for searching passages, entities, and facts.
        """
        return await self.service.embed_with_instruction(query, "query")

    async def embed_passages_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate passage embeddings for multiple texts.
        """
        # Add instruction prefix to each text
        prefixed = [
            f"Represent this document for retrieval: {text}"
            for text in texts
        ]
        return await self.service.embed_batch(prefixed)

    async def embed_entities_batch(
        self,
        entities: List[Dict[str, Any]],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple entities.

        Args:
            entities: List of dicts with 'name' and optional 'description'
        """
        texts = []
        for entity in entities:
            name = entity.get("name", "")
            description = entity.get("description")
            if description:
                texts.append(f"{name}: {description}")
            else:
                texts.append(name)
        return await self.service.embed_batch(texts)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def generate_embedding(
    text: str,
    config: Optional[EmbeddingConfig] = None,
) -> List[float]:
    """
    Convenience function to generate a single embedding.

    Args:
        text: The text to embed
        config: Optional embedding configuration

    Returns:
        List of floats representing the embedding vector
    """
    service = EmbeddingService(config)
    return await service.embed_text(text)


async def generate_embeddings_batch(
    texts: List[str],
    config: Optional[EmbeddingConfig] = None,
) -> List[List[float]]:
    """
    Convenience function to generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        config: Optional embedding configuration

    Returns:
        List of embedding vectors
    """
    service = EmbeddingService(config)
    return await service.embed_batch(texts)


# Singleton instance for reuse
_kg_embeddings: Optional[KnowledgeGraphEmbeddings] = None


def get_kg_embeddings(config: Optional[EmbeddingConfig] = None) -> KnowledgeGraphEmbeddings:
    """Get or create a singleton KnowledgeGraphEmbeddings instance."""
    global _kg_embeddings
    if _kg_embeddings is None:
        _kg_embeddings = KnowledgeGraphEmbeddings(config)
    return _kg_embeddings
