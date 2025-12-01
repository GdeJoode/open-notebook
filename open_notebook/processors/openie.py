"""
OpenIE (Open Information Extraction) Pipeline

Extracts structured knowledge from text using local LLM (qwen2.5:14b via Ollama):
- Named Entity Recognition (NER)
- Triple extraction (subject-predicate-object)
- Claim extraction with evidence chains

See docs/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md Phase 2 for full documentation.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from esperanto import AIFactory, LanguageModel
from loguru import logger
from pydantic import BaseModel, Field

from open_notebook.domain.knowledge_graph import (
    ClaimType,
    Entity,
    EntityType,
    compute_entity_hash,
)


# =============================================================================
# CONFIGURATION
# =============================================================================


@dataclass
class OpenIEConfig:
    """Configuration for OpenIE extraction."""

    model: str = field(
        default_factory=lambda: os.getenv("OPENIE_MODEL", "qwen2.5:14b")
    )
    provider: str = "ollama"
    base_url: str = field(
        default_factory=lambda: os.getenv(
            "OPENIE_OLLAMA_BASE_URL", "http://localhost:11434"
        )
    )
    temperature: float = 0.1  # Low for consistent extraction
    max_tokens: int = 4096

    # Extraction thresholds
    entity_confidence_threshold: float = 0.6
    triple_confidence_threshold: float = 0.5
    claim_confidence_threshold: float = 0.5


# =============================================================================
# EXTRACTION SCHEMAS (Pydantic models for structured output)
# =============================================================================


class ExtractedEntity(BaseModel):
    """An entity extracted from text."""

    name: str = Field(..., description="The entity name as it appears in text")
    entity_type: str = Field(
        ..., description="Type: person, organization, topic, location, concept, event"
    )
    description: Optional[str] = Field(
        None, description="Brief description if inferrable"
    )
    aliases: List[str] = Field(
        default_factory=list, description="Alternative names mentioned"
    )
    confidence: float = Field(
        default=0.8, ge=0.0, le=1.0, description="Confidence score"
    )


class ExtractedTriple(BaseModel):
    """A subject-predicate-object triple extracted from text."""

    subject: str = Field(..., description="The subject entity")
    predicate: str = Field(..., description="The relationship/action")
    object: str = Field(..., description="The object entity or value")
    context: Optional[str] = Field(
        None, description="Surrounding context from source"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence score"
    )


class ExtractedClaim(BaseModel):
    """A claim/statement extracted from text."""

    statement: str = Field(..., description="The claim statement")
    claim_type: str = Field(
        default="factual",
        description="Type: factual, causal, normative, predictive",
    )
    supporting_quote: Optional[str] = Field(
        None, description="Direct quote supporting this claim"
    )
    entities_involved: List[str] = Field(
        default_factory=list, description="Entities mentioned in the claim"
    )
    confidence: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Confidence score"
    )


class ExtractionResult(BaseModel):
    """Complete extraction result from a text passage."""

    entities: List[ExtractedEntity] = Field(default_factory=list)
    triples: List[ExtractedTriple] = Field(default_factory=list)
    claims: List[ExtractedClaim] = Field(default_factory=list)
    source_text_hash: Optional[str] = Field(
        None, description="Hash of source text for deduplication"
    )


# =============================================================================
# PROMPTS
# =============================================================================

ENTITY_EXTRACTION_PROMPT = """You are an expert at extracting named entities from text.

Extract all named entities from the following text. For each entity, identify:
1. The entity name (exactly as it appears or normalized)
2. The entity type (person, organization, topic, location, concept, event, product)
3. A brief description if inferrable from context
4. Any aliases or alternative names mentioned
5. Your confidence in the extraction (0.0-1.0)

TEXT:
{text}

Respond with a JSON array of entities. Example format:
```json
[
  {{
    "name": "World Health Organization",
    "entity_type": "organization",
    "description": "International public health agency",
    "aliases": ["WHO"],
    "confidence": 0.95
  }},
  {{
    "name": "climate change",
    "entity_type": "topic",
    "description": "Long-term shift in global temperatures and weather patterns",
    "aliases": ["global warming"],
    "confidence": 0.9
  }}
]
```

Return ONLY the JSON array, no other text."""


TRIPLE_EXTRACTION_PROMPT = """You are an expert at extracting relationships from text.

Extract subject-predicate-object triples that represent factual relationships.
Focus on meaningful relationships, not trivial ones.

TEXT:
{text}

For each triple, provide:
1. subject: The entity performing the action or being described
2. predicate: The relationship or action (use simple verb phrases)
3. object: The target entity or value
4. context: A brief quote showing where this relationship is stated
5. confidence: Your confidence (0.0-1.0)

Respond with a JSON array. Example format:
```json
[
  {{
    "subject": "Einstein",
    "predicate": "developed",
    "object": "theory of relativity",
    "context": "Einstein developed his famous theory of relativity in 1905",
    "confidence": 0.95
  }},
  {{
    "subject": "Netherlands",
    "predicate": "is member of",
    "object": "European Union",
    "context": "The Netherlands, as an EU member state...",
    "confidence": 0.9
  }}
]
```

Return ONLY the JSON array, no other text."""


CLAIM_EXTRACTION_PROMPT = """You are an expert at extracting claims and statements from text.

Extract claims that could be verified, supported, or contested. Focus on:
- Factual claims (statements of fact)
- Causal claims (X causes Y)
- Normative claims (should/ought statements)
- Predictive claims (forecasts about the future)

TEXT:
{text}

For each claim, provide:
1. statement: The claim in clear, standalone form
2. claim_type: factual, causal, normative, or predictive
3. supporting_quote: Direct quote from the text
4. entities_involved: List of entity names mentioned
5. confidence: Your confidence this is a genuine claim (0.0-1.0)

Respond with a JSON array. Example format:
```json
[
  {{
    "statement": "Climate change increases the frequency of extreme weather events",
    "claim_type": "causal",
    "supporting_quote": "Research shows climate change is driving more frequent storms",
    "entities_involved": ["climate change", "extreme weather events"],
    "confidence": 0.85
  }}
]
```

Return ONLY the JSON array, no other text."""


COMBINED_EXTRACTION_PROMPT = """You are an expert knowledge extractor. Analyze the following text and extract:

1. ENTITIES: Named entities (people, organizations, topics, locations, concepts, events)
2. TRIPLES: Subject-predicate-object relationships
3. CLAIMS: Verifiable statements or assertions

TEXT:
{text}

Respond with a JSON object containing all extractions:
```json
{{
  "entities": [
    {{
      "name": "entity name",
      "entity_type": "person|organization|topic|location|concept|event|product",
      "description": "brief description or null",
      "aliases": ["alternative names"],
      "confidence": 0.8
    }}
  ],
  "triples": [
    {{
      "subject": "subject entity",
      "predicate": "relationship verb",
      "object": "object entity or value",
      "context": "quote from text",
      "confidence": 0.7
    }}
  ],
  "claims": [
    {{
      "statement": "the claim statement",
      "claim_type": "factual|causal|normative|predictive",
      "supporting_quote": "quote from text",
      "entities_involved": ["entity1", "entity2"],
      "confidence": 0.7
    }}
  ]
}}
```

Return ONLY the JSON object, no other text."""


# =============================================================================
# OPENIE EXTRACTOR
# =============================================================================


class OpenIEExtractor:
    """
    Extracts structured knowledge from text using local LLM.

    Uses qwen2.5:14b via Ollama for:
    - Named Entity Recognition
    - Triple extraction
    - Claim extraction
    """

    def __init__(self, config: Optional[OpenIEConfig] = None):
        self.config = config or OpenIEConfig()
        self._llm: Optional[LanguageModel] = None

    def _get_llm(self) -> LanguageModel:
        """Get or create the LLM instance."""
        if self._llm is None:
            logger.info(
                f"Initializing OpenIE LLM: {self.config.provider}/{self.config.model}"
            )
            self._llm = AIFactory.create_language(
                provider=self.config.provider,
                model_name=self.config.model,
                config={
                    "base_url": self.config.base_url,
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            )
        return self._llm

    def _parse_json_response(self, response: str) -> Any:
        """Parse JSON from LLM response, handling markdown code blocks."""
        # Remove markdown code blocks if present
        text = response.strip()
        if text.startswith("```"):
            # Find the end of the opening fence
            first_newline = text.find("\n")
            if first_newline > 0:
                text = text[first_newline + 1 :]
            # Remove closing fence
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response[:500]}...")
            return None

    async def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """Extract named entities from text."""
        if not text or not text.strip():
            return []

        llm = self._get_llm()
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)

        try:
            response = await llm.chat_async([{"role": "user", "content": prompt}])
            content = response.choices[0].message.content

            parsed = self._parse_json_response(content)
            if not parsed or not isinstance(parsed, list):
                logger.warning("Failed to parse entity extraction response")
                return []

            entities = []
            for item in parsed:
                try:
                    entity = ExtractedEntity(**item)
                    if entity.confidence >= self.config.entity_confidence_threshold:
                        entities.append(entity)
                except Exception as e:
                    logger.debug(f"Skipping invalid entity: {e}")

            logger.info(f"Extracted {len(entities)} entities from text")
            return entities

        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return []

    async def extract_triples(self, text: str) -> List[ExtractedTriple]:
        """Extract subject-predicate-object triples from text."""
        if not text or not text.strip():
            return []

        llm = self._get_llm()
        prompt = TRIPLE_EXTRACTION_PROMPT.format(text=text)

        try:
            response = await llm.chat_async([{"role": "user", "content": prompt}])
            content = response.choices[0].message.content

            parsed = self._parse_json_response(content)
            if not parsed or not isinstance(parsed, list):
                logger.warning("Failed to parse triple extraction response")
                return []

            triples = []
            for item in parsed:
                try:
                    triple = ExtractedTriple(**item)
                    if triple.confidence >= self.config.triple_confidence_threshold:
                        triples.append(triple)
                except Exception as e:
                    logger.debug(f"Skipping invalid triple: {e}")

            logger.info(f"Extracted {len(triples)} triples from text")
            return triples

        except Exception as e:
            logger.error(f"Triple extraction failed: {e}")
            return []

    async def extract_claims(self, text: str) -> List[ExtractedClaim]:
        """Extract claims/statements from text."""
        if not text or not text.strip():
            return []

        llm = self._get_llm()
        prompt = CLAIM_EXTRACTION_PROMPT.format(text=text)

        try:
            response = await llm.chat_async([{"role": "user", "content": prompt}])
            content = response.choices[0].message.content

            parsed = self._parse_json_response(content)
            if not parsed or not isinstance(parsed, list):
                logger.warning("Failed to parse claim extraction response")
                return []

            claims = []
            for item in parsed:
                try:
                    claim = ExtractedClaim(**item)
                    if claim.confidence >= self.config.claim_confidence_threshold:
                        claims.append(claim)
                except Exception as e:
                    logger.debug(f"Skipping invalid claim: {e}")

            logger.info(f"Extracted {len(claims)} claims from text")
            return claims

        except Exception as e:
            logger.error(f"Claim extraction failed: {e}")
            return []

    async def extract_all(self, text: str) -> ExtractionResult:
        """
        Extract all knowledge types from text in a single LLM call.

        More efficient than calling extract_entities, extract_triples,
        and extract_claims separately.
        """
        if not text or not text.strip():
            return ExtractionResult()

        llm = self._get_llm()
        prompt = COMBINED_EXTRACTION_PROMPT.format(text=text)

        try:
            response = await llm.chat_async([{"role": "user", "content": prompt}])
            content = response.choices[0].message.content

            parsed = self._parse_json_response(content)
            if not parsed or not isinstance(parsed, dict):
                logger.warning("Failed to parse combined extraction response")
                return ExtractionResult()

            # Parse entities
            entities = []
            for item in parsed.get("entities", []):
                try:
                    entity = ExtractedEntity(**item)
                    if entity.confidence >= self.config.entity_confidence_threshold:
                        entities.append(entity)
                except Exception as e:
                    logger.debug(f"Skipping invalid entity: {e}")

            # Parse triples
            triples = []
            for item in parsed.get("triples", []):
                try:
                    triple = ExtractedTriple(**item)
                    if triple.confidence >= self.config.triple_confidence_threshold:
                        triples.append(triple)
                except Exception as e:
                    logger.debug(f"Skipping invalid triple: {e}")

            # Parse claims
            claims = []
            for item in parsed.get("claims", []):
                try:
                    claim = ExtractedClaim(**item)
                    if claim.confidence >= self.config.claim_confidence_threshold:
                        claims.append(claim)
                except Exception as e:
                    logger.debug(f"Skipping invalid claim: {e}")

            logger.info(
                f"Combined extraction: {len(entities)} entities, "
                f"{len(triples)} triples, {len(claims)} claims"
            )

            return ExtractionResult(
                entities=entities,
                triples=triples,
                claims=claims,
            )

        except Exception as e:
            logger.error(f"Combined extraction failed: {e}")
            return ExtractionResult()


# =============================================================================
# ENTITY TYPE MAPPING
# =============================================================================


def map_entity_type(extracted_type: str) -> EntityType:
    """Map extracted entity type string to EntityType enum."""
    type_map = {
        "person": EntityType.PERSON,
        "organization": EntityType.ORGANIZATION,
        "topic": EntityType.TOPIC,
        "location": EntityType.LOCATION,
        "concept": EntityType.CONCEPT,
        "event": EntityType.EVENT,
        "product": EntityType.PRODUCT,
    }
    return type_map.get(extracted_type.lower(), EntityType.OTHER)


def map_claim_type(extracted_type: str) -> ClaimType:
    """Map extracted claim type string to ClaimType enum."""
    type_map = {
        "factual": ClaimType.FACTUAL,
        "causal": ClaimType.CAUSAL,
        "normative": ClaimType.NORMATIVE,
        "predictive": ClaimType.PREDICTIVE,
    }
    return type_map.get(extracted_type.lower(), ClaimType.FACTUAL)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


async def extract_knowledge_from_text(
    text: str,
    config: Optional[OpenIEConfig] = None,
) -> ExtractionResult:
    """
    Convenience function to extract all knowledge from text.

    Args:
        text: The text to extract knowledge from
        config: Optional OpenIE configuration

    Returns:
        ExtractionResult with entities, triples, and claims
    """
    extractor = OpenIEExtractor(config)
    return await extractor.extract_all(text)


async def extract_entities_from_text(
    text: str,
    config: Optional[OpenIEConfig] = None,
) -> List[ExtractedEntity]:
    """
    Convenience function to extract only entities from text.

    Args:
        text: The text to extract entities from
        config: Optional OpenIE configuration

    Returns:
        List of extracted entities
    """
    extractor = OpenIEExtractor(config)
    return await extractor.extract_entities(text)
