"""
Knowledge Graph Domain Models

This module contains the domain models for the Knowledge Graph system,
implementing Option C (Full Ontology-Driven KG) with HippoRAG enhancements.

See docs/KNOWLEDGE_GRAPH_IMPLEMENTATION_PLAN.md for full documentation.
"""

import hashlib
from enum import Enum
from typing import Any, ClassVar, Dict, List, Optional

from loguru import logger
from pydantic import Field, field_validator

from open_notebook.database.repository import ensure_record_id, repo_query
from open_notebook.domain.base import ObjectModel
from open_notebook.domain.models import model_manager
from open_notebook.exceptions import DatabaseOperationError, InvalidInputError


# =============================================================================
# ENUMS
# =============================================================================


class SourceType(str, Enum):
    """Classification of source document types."""
    ACADEMIC_PAPER = "academic_paper"
    POLICY_DOCUMENT = "policy_document"
    POLICY_ADVICE = "policy_advice"
    SOCIAL_MEDIA = "social_media"
    NEWS_ARTICLE = "news_article"
    LEGAL_DOCUMENT = "legal_document"
    REPORT = "report"
    PRESENTATION = "presentation"
    OTHER = "other"


class EntityType(str, Enum):
    """Types of entities that can be extracted from sources."""
    PERSON = "person"
    ORGANIZATION = "organization"
    TOPIC = "topic"
    LOCATION = "location"
    CONCEPT = "concept"
    EVENT = "event"
    PRODUCT = "product"
    OTHER = "other"


class ClaimType(str, Enum):
    """Types of claims that can be extracted from sources."""
    FACTUAL = "factual"
    CAUSAL = "causal"
    NORMATIVE = "normative"
    PREDICTIVE = "predictive"


class VerificationStatus(str, Enum):
    """Verification status for claims."""
    UNVERIFIED = "unverified"
    SUPPORTED = "supported"
    CONTESTED = "contested"
    REFUTED = "refuted"


class EvidenceType(str, Enum):
    """Types of evidence supporting claims."""
    STATISTICAL = "statistical"
    QUALITATIVE = "qualitative"
    EXPERIMENTAL = "experimental"
    OBSERVATIONAL = "observational"
    EXPERT_OPINION = "expert_opinion"
    CASE_STUDY = "case_study"


class EvidenceStrength(str, Enum):
    """Strength levels for evidence."""
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"


class OrganizationType(str, Enum):
    """Types of organizations."""
    UNIVERSITY = "university"
    RESEARCH_INSTITUTE = "research_institute"
    THINK_TANK = "think_tank"
    ADVISORY_BODY = "advisory_body"
    GOVERNMENT = "government"
    NGO = "ngo"
    COMPANY = "company"
    INTERNATIONAL_ORG = "international_org"
    OTHER = "other"


class TopicLevel(str, Enum):
    """Hierarchical level for topics."""
    BROAD = "broad"
    SPECIFIC = "specific"
    NARROW = "narrow"


class AuthorRole(str, Enum):
    """Role of an author in a source."""
    AUTHOR = "author"
    LEAD_AUTHOR = "lead_author"
    CORRESPONDING = "corresponding"
    CONTRIBUTOR = "contributor"


class CitationSentiment(str, Enum):
    """Sentiment of a citation."""
    SUPPORTIVE = "supportive"
    CRITICAL = "critical"
    NEUTRAL = "neutral"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def compute_entity_hash(name: str) -> str:
    """
    Compute a hash ID for an entity based on its normalized name.

    This is used for deduplication (HippoRAG style).
    """
    normalized = name.lower().strip()
    return hashlib.md5(normalized.encode()).hexdigest()


# =============================================================================
# ENTITY MODEL
# =============================================================================


class Entity(ObjectModel):
    """
    Represents an entity extracted from sources.

    Entities are the building blocks of the knowledge graph, representing
    real-world objects like people, organizations, topics, locations, etc.

    HippoRAG Enhancement: Each entity has an embedding for KNN-based
    deduplication and semantic similarity search.
    """
    table_name: ClassVar[str] = "entity"

    name: str = Field(..., description="Primary name of the entity")
    entity_type: EntityType = Field(..., description="Type classification")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    description: Optional[str] = Field(None, description="Description of the entity")

    # External knowledge base links
    external_ids: Dict[str, str] = Field(
        default_factory=dict,
        description="External KB links (wikidata, orcid, ror, etc.)"
    )

    # Hash ID for deduplication (HippoRAG style)
    hash_id: Optional[str] = Field(None, description="MD5 hash of normalized name")

    # Embedding for semantic similarity
    embedding: Optional[List[float]] = Field(None, description="Entity embedding vector")

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise InvalidInputError("Entity name cannot be empty")
        return v.strip()

    @field_validator("entity_type", mode="before")
    @classmethod
    def validate_entity_type(cls, v):
        if isinstance(v, str):
            return EntityType(v)
        return v

    def _prepare_save_data(self) -> Dict[str, Any]:
        """Prepare data for saving, computing hash_id if not set."""
        data = super()._prepare_save_data()

        # Always compute hash_id from name
        if data.get("name"):
            data["hash_id"] = compute_entity_hash(data["name"])

        # Convert enum to string
        if "entity_type" in data and isinstance(data["entity_type"], EntityType):
            data["entity_type"] = data["entity_type"].value

        return data

    def needs_embedding(self) -> bool:
        return True

    def get_embedding_content(self) -> Optional[str]:
        """Content to embed: name + description + aliases."""
        parts = [self.name]
        if self.description:
            parts.append(self.description)
        if self.aliases:
            parts.extend(self.aliases)
        return " ".join(parts)

    async def find_similar(
        self,
        threshold: float = 0.8,
        limit: int = 10
    ) -> List["Entity"]:
        """
        Find similar entities using embedding similarity.

        This is the KNN-based deduplication from HippoRAG.
        """
        if not self.embedding:
            return []

        try:
            results = await repo_query(
                """
                SELECT *,
                    vector::similarity::cosine(embedding, $embedding) AS similarity
                FROM entity
                WHERE id != $id
                    AND embedding != NONE
                    AND vector::similarity::cosine(embedding, $embedding) >= $threshold
                ORDER BY similarity DESC
                LIMIT $limit
                """,
                {
                    "id": ensure_record_id(self.id) if self.id else None,
                    "embedding": self.embedding,
                    "threshold": threshold,
                    "limit": limit
                }
            )
            return [Entity(**r) for r in results]
        except Exception as e:
            logger.error(f"Error finding similar entities: {e}")
            raise DatabaseOperationError(e)

    async def get_mentions(self) -> List[Dict[str, Any]]:
        """Get all sources that mention this entity."""
        try:
            results = await repo_query(
                """
                SELECT
                    in AS source_id,
                    in.title AS source_title,
                    context,
                    confidence
                FROM mentions
                WHERE out = $id
                """,
                {"id": ensure_record_id(self.id)}
            )
            return results
        except Exception as e:
            logger.error(f"Error getting mentions for entity {self.id}: {e}")
            raise DatabaseOperationError(e)

    async def create_same_as_link(
        self,
        other_entity_id: str,
        similarity: float,
        method: str = "embedding_knn"
    ) -> Any:
        """Create a same_as relationship to another entity."""
        if not self.id:
            raise InvalidInputError("Entity must be saved before creating relationships")

        return await self.relate(
            "same_as",
            other_entity_id,
            {"similarity": similarity, "method": method, "verified": False}
        )

    @classmethod
    async def get_by_hash(cls, hash_id: str) -> Optional["Entity"]:
        """Find an entity by its hash_id."""
        try:
            results = await repo_query(
                "SELECT * FROM entity WHERE hash_id = $hash_id LIMIT 1",
                {"hash_id": hash_id}
            )
            if results:
                return cls(**results[0])
            return None
        except Exception as e:
            logger.error(f"Error getting entity by hash: {e}")
            raise DatabaseOperationError(e)

    @classmethod
    async def find_or_create(
        cls,
        name: str,
        entity_type: EntityType,
        **kwargs
    ) -> "Entity":
        """
        Find an existing entity by hash or create a new one.

        This ensures deduplication at creation time.
        """
        hash_id = compute_entity_hash(name)
        existing = await cls.get_by_hash(hash_id)

        if existing:
            return existing

        entity = cls(name=name, entity_type=entity_type, **kwargs)
        await entity.save()
        return entity


# =============================================================================
# CLAIM MODEL
# =============================================================================


class Claim(ObjectModel):
    """
    Represents a claim extracted from sources.

    Claims are statements that can be verified, supported, or contradicted
    by evidence from various sources.
    """
    table_name: ClassVar[str] = "claim"

    statement: str = Field(..., description="The claim statement")
    claim_type: ClaimType = Field(..., description="Type of claim")
    verification_status: VerificationStatus = Field(
        default=VerificationStatus.UNVERIFIED,
        description="Current verification status"
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence score (0-1)"
    )
    first_appearance: Optional[str] = Field(
        None,
        description="Date when claim first appeared"
    )

    embedding: Optional[List[float]] = Field(None, description="Claim embedding vector")

    @field_validator("statement")
    @classmethod
    def statement_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise InvalidInputError("Claim statement cannot be empty")
        return v.strip()

    @field_validator("claim_type", mode="before")
    @classmethod
    def validate_claim_type(cls, v):
        if isinstance(v, str):
            return ClaimType(v)
        return v

    @field_validator("verification_status", mode="before")
    @classmethod
    def validate_verification_status(cls, v):
        if isinstance(v, str):
            return VerificationStatus(v)
        return v

    def _prepare_save_data(self) -> Dict[str, Any]:
        data = super()._prepare_save_data()

        # Convert enums to strings
        if "claim_type" in data and isinstance(data["claim_type"], ClaimType):
            data["claim_type"] = data["claim_type"].value
        if "verification_status" in data and isinstance(data["verification_status"], VerificationStatus):
            data["verification_status"] = data["verification_status"].value

        return data

    def needs_embedding(self) -> bool:
        return True

    def get_embedding_content(self) -> Optional[str]:
        return self.statement

    async def get_supporting_sources(self) -> List[Dict[str, Any]]:
        """Get sources that support this claim."""
        try:
            results = await repo_query(
                """
                SELECT
                    in AS source_id,
                    in.title AS source_title,
                    in.source_type AS source_type,
                    strength,
                    quote
                FROM supports
                WHERE out = $id
                """,
                {"id": ensure_record_id(self.id)}
            )
            return results
        except Exception as e:
            logger.error(f"Error getting supporting sources for claim {self.id}: {e}")
            raise DatabaseOperationError(e)

    async def get_contradicting_sources(self) -> List[Dict[str, Any]]:
        """Get sources that contradict this claim."""
        try:
            results = await repo_query(
                """
                SELECT
                    in AS source_id,
                    in.title AS source_title,
                    in.source_type AS source_type,
                    strength,
                    quote
                FROM contradicts
                WHERE out = $id
                """,
                {"id": ensure_record_id(self.id)}
            )
            return results
        except Exception as e:
            logger.error(f"Error getting contradicting sources for claim {self.id}: {e}")
            raise DatabaseOperationError(e)

    async def get_evidence_summary(self) -> Dict[str, Any]:
        """Get a summary of evidence for and against this claim."""
        supporting = await self.get_supporting_sources()
        contradicting = await self.get_contradicting_sources()

        return {
            "claim_id": self.id,
            "statement": self.statement,
            "verification_status": self.verification_status.value if isinstance(self.verification_status, VerificationStatus) else self.verification_status,
            "supporting_count": len(supporting),
            "contradicting_count": len(contradicting),
            "supporting_sources": supporting,
            "contradicting_sources": contradicting
        }

    @classmethod
    async def find_similar(
        cls,
        query_embedding: List[float],
        threshold: float = 0.7,
        limit: int = 10
    ) -> List["Claim"]:
        """Find claims similar to a query embedding."""
        try:
            results = await repo_query(
                """
                SELECT *,
                    vector::similarity::cosine(embedding, $embedding) AS similarity
                FROM claim
                WHERE embedding != NONE
                    AND vector::similarity::cosine(embedding, $embedding) >= $threshold
                ORDER BY similarity DESC
                LIMIT $limit
                """,
                {
                    "embedding": query_embedding,
                    "threshold": threshold,
                    "limit": limit
                }
            )
            return [cls(**r) for r in results]
        except Exception as e:
            logger.error(f"Error finding similar claims: {e}")
            raise DatabaseOperationError(e)


# =============================================================================
# EVIDENCE MODEL
# =============================================================================


class Evidence(ObjectModel):
    """
    Represents evidence supporting or contradicting claims.
    """
    table_name: ClassVar[str] = "evidence"

    description: str = Field(..., description="Description of the evidence")
    evidence_type: EvidenceType = Field(..., description="Type of evidence")
    strength: EvidenceStrength = Field(
        default=EvidenceStrength.MODERATE,
        description="Strength of the evidence"
    )
    methodology: Optional[str] = Field(None, description="Methodology used")
    sample_size: Optional[int] = Field(None, ge=0, description="Sample size if applicable")

    source_id: Optional[str] = Field(None, description="Reference to source document")
    embedding: Optional[List[float]] = Field(None, description="Evidence embedding vector")

    @field_validator("description")
    @classmethod
    def description_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise InvalidInputError("Evidence description cannot be empty")
        return v.strip()

    @field_validator("evidence_type", mode="before")
    @classmethod
    def validate_evidence_type(cls, v):
        if isinstance(v, str):
            return EvidenceType(v)
        return v

    @field_validator("strength", mode="before")
    @classmethod
    def validate_strength(cls, v):
        if isinstance(v, str):
            return EvidenceStrength(v)
        return v

    def _prepare_save_data(self) -> Dict[str, Any]:
        data = super()._prepare_save_data()

        # Convert enums to strings
        if "evidence_type" in data and isinstance(data["evidence_type"], EvidenceType):
            data["evidence_type"] = data["evidence_type"].value
        if "strength" in data and isinstance(data["strength"], EvidenceStrength):
            data["strength"] = data["strength"].value

        # Ensure source_id is RecordID format
        if data.get("source_id"):
            data["source_id"] = ensure_record_id(data["source_id"])

        return data

    def needs_embedding(self) -> bool:
        return True

    def get_embedding_content(self) -> Optional[str]:
        parts = [self.description]
        if self.methodology:
            parts.append(self.methodology)
        return " ".join(parts)


# =============================================================================
# PERSON MODEL
# =============================================================================


class Person(ObjectModel):
    """
    Represents a person (author, expert, etc.) in the knowledge graph.
    """
    table_name: ClassVar[str] = "person"

    name: str = Field(..., description="Full name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    orcid: Optional[str] = Field(None, description="ORCID identifier")
    email: Optional[str] = Field(None, description="Email address")
    bio: Optional[str] = Field(None, description="Biography")
    expertise_areas: List[str] = Field(
        default_factory=list,
        description="Areas of expertise"
    )
    h_index: Optional[int] = Field(None, ge=0, description="H-index")
    current_position: Optional[str] = Field(None, description="Current position/title")

    external_ids: Dict[str, str] = Field(
        default_factory=dict,
        description="External IDs (linkedin, twitter, google_scholar)"
    )

    embedding: Optional[List[float]] = Field(None, description="Person embedding vector")

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise InvalidInputError("Person name cannot be empty")
        return v.strip()

    def needs_embedding(self) -> bool:
        return True

    def get_embedding_content(self) -> Optional[str]:
        parts = [self.name]
        if self.bio:
            parts.append(self.bio)
        if self.expertise_areas:
            parts.extend(self.expertise_areas)
        if self.current_position:
            parts.append(self.current_position)
        return " ".join(parts)

    async def get_publications(self) -> List[Dict[str, Any]]:
        """Get all sources authored by this person."""
        try:
            results = await repo_query(
                """
                SELECT
                    in AS source_id,
                    in.title AS title,
                    in.source_type AS source_type,
                    role,
                    position
                FROM authored_by
                WHERE out = $id
                ORDER BY in.created DESC
                """,
                {"id": ensure_record_id(self.id)}
            )
            return results
        except Exception as e:
            logger.error(f"Error getting publications for person {self.id}: {e}")
            raise DatabaseOperationError(e)

    async def get_affiliations(self) -> List[Dict[str, Any]]:
        """Get all organizations this person is affiliated with."""
        try:
            results = await repo_query(
                """
                SELECT
                    out AS org_id,
                    out.name AS org_name,
                    out.org_type AS org_type,
                    role,
                    department,
                    is_current
                FROM affiliated_with
                WHERE in = $id
                """,
                {"id": ensure_record_id(self.id)}
            )
            return results
        except Exception as e:
            logger.error(f"Error getting affiliations for person {self.id}: {e}")
            raise DatabaseOperationError(e)

    async def get_citation_count(self) -> int:
        """Get total citation count for this person's publications."""
        try:
            results = await repo_query(
                """
                SELECT count() AS citations FROM cites
                WHERE out IN (
                    SELECT in FROM authored_by WHERE out = $id
                )
                GROUP ALL
                """,
                {"id": ensure_record_id(self.id)}
            )
            return results[0]["citations"] if results else 0
        except Exception as e:
            logger.error(f"Error getting citation count for person {self.id}: {e}")
            raise DatabaseOperationError(e)

    @classmethod
    async def find_by_orcid(cls, orcid: str) -> Optional["Person"]:
        """Find a person by their ORCID."""
        try:
            results = await repo_query(
                "SELECT * FROM person WHERE orcid = $orcid LIMIT 1",
                {"orcid": orcid}
            )
            if results:
                return cls(**results[0])
            return None
        except Exception as e:
            logger.error(f"Error finding person by ORCID: {e}")
            raise DatabaseOperationError(e)


# =============================================================================
# ORGANIZATION MODEL
# =============================================================================


class Organization(ObjectModel):
    """
    Represents an organization in the knowledge graph.
    """
    table_name: ClassVar[str] = "organization"

    name: str = Field(..., description="Organization name")
    aliases: List[str] = Field(default_factory=list, description="Alternative names")
    org_type: OrganizationType = Field(..., description="Type of organization")
    country: Optional[str] = Field(None, description="Country")
    city: Optional[str] = Field(None, description="City")
    website: Optional[str] = Field(None, description="Website URL")
    description: Optional[str] = Field(None, description="Description")
    ror_id: Optional[str] = Field(None, description="Research Organization Registry ID")

    embedding: Optional[List[float]] = Field(None, description="Organization embedding")

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise InvalidInputError("Organization name cannot be empty")
        return v.strip()

    @field_validator("org_type", mode="before")
    @classmethod
    def validate_org_type(cls, v):
        if isinstance(v, str):
            return OrganizationType(v)
        return v

    def _prepare_save_data(self) -> Dict[str, Any]:
        data = super()._prepare_save_data()

        if "org_type" in data and isinstance(data["org_type"], OrganizationType):
            data["org_type"] = data["org_type"].value

        return data

    def needs_embedding(self) -> bool:
        return True

    def get_embedding_content(self) -> Optional[str]:
        parts = [self.name]
        if self.description:
            parts.append(self.description)
        if self.aliases:
            parts.extend(self.aliases)
        return " ".join(parts)

    async def get_members(self) -> List[Dict[str, Any]]:
        """Get all people affiliated with this organization."""
        try:
            results = await repo_query(
                """
                SELECT
                    in AS person_id,
                    in.name AS person_name,
                    role,
                    department,
                    is_current
                FROM affiliated_with
                WHERE out = $id
                """,
                {"id": ensure_record_id(self.id)}
            )
            return results
        except Exception as e:
            logger.error(f"Error getting members for organization {self.id}: {e}")
            raise DatabaseOperationError(e)

    async def get_publications(self) -> List[Dict[str, Any]]:
        """Get publications from members of this organization."""
        try:
            results = await repo_query(
                """
                SELECT DISTINCT
                    source.id AS source_id,
                    source.title AS title,
                    source.source_type AS source_type
                FROM (
                    SELECT in AS person FROM affiliated_with WHERE out = $id
                )
                LET source = (SELECT in FROM authored_by WHERE out = person)
                """,
                {"id": ensure_record_id(self.id)}
            )
            return results
        except Exception as e:
            logger.error(f"Error getting publications for organization {self.id}: {e}")
            raise DatabaseOperationError(e)

    @classmethod
    async def find_by_ror(cls, ror_id: str) -> Optional["Organization"]:
        """Find an organization by its ROR ID."""
        try:
            results = await repo_query(
                "SELECT * FROM organization WHERE ror_id = $ror_id LIMIT 1",
                {"ror_id": ror_id}
            )
            if results:
                return cls(**results[0])
            return None
        except Exception as e:
            logger.error(f"Error finding organization by ROR: {e}")
            raise DatabaseOperationError(e)


# =============================================================================
# TOPIC MODEL
# =============================================================================


class Topic(ObjectModel):
    """
    Represents a topic/concept in the knowledge graph.

    Topics form a hierarchical taxonomy with broader/narrower relationships.
    """
    table_name: ClassVar[str] = "topic"

    name: str = Field(..., description="Topic name")
    description: Optional[str] = Field(None, description="Topic description")
    level: TopicLevel = Field(
        default=TopicLevel.SPECIFIC,
        description="Hierarchical level"
    )
    domain: Optional[str] = Field(None, description="Domain (e.g., healthcare, education)")
    wikidata_id: Optional[str] = Field(None, description="Wikidata Q-identifier")

    embedding: Optional[List[float]] = Field(None, description="Topic embedding")

    @field_validator("name")
    @classmethod
    def name_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise InvalidInputError("Topic name cannot be empty")
        return v.strip()

    @field_validator("level", mode="before")
    @classmethod
    def validate_level(cls, v):
        if isinstance(v, str):
            return TopicLevel(v)
        return v

    def _prepare_save_data(self) -> Dict[str, Any]:
        data = super()._prepare_save_data()

        if "level" in data and isinstance(data["level"], TopicLevel):
            data["level"] = data["level"].value

        return data

    def needs_embedding(self) -> bool:
        return True

    def get_embedding_content(self) -> Optional[str]:
        parts = [self.name]
        if self.description:
            parts.append(self.description)
        if self.domain:
            parts.append(self.domain)
        return " ".join(parts)

    async def get_sources(self) -> List[Dict[str, Any]]:
        """Get all sources that discuss this topic."""
        try:
            results = await repo_query(
                """
                SELECT
                    in AS source_id,
                    in.title AS title,
                    in.source_type AS source_type,
                    relevance,
                    is_primary
                FROM discusses
                WHERE out = $id
                ORDER BY relevance DESC
                """,
                {"id": ensure_record_id(self.id)}
            )
            return results
        except Exception as e:
            logger.error(f"Error getting sources for topic {self.id}: {e}")
            raise DatabaseOperationError(e)

    async def get_broader_topics(self) -> List["Topic"]:
        """Get broader (parent) topics in the hierarchy."""
        try:
            results = await repo_query(
                """
                SELECT out.* FROM broader_than WHERE in = $id
                """,
                {"id": ensure_record_id(self.id)}
            )
            return [Topic(**r["out"]) for r in results if r.get("out")]
        except Exception as e:
            logger.error(f"Error getting broader topics for {self.id}: {e}")
            raise DatabaseOperationError(e)

    async def get_narrower_topics(self) -> List["Topic"]:
        """Get narrower (child) topics in the hierarchy."""
        try:
            results = await repo_query(
                """
                SELECT in.* FROM broader_than WHERE out = $id
                """,
                {"id": ensure_record_id(self.id)}
            )
            return [Topic(**r["in"]) for r in results if r.get("in")]
        except Exception as e:
            logger.error(f"Error getting narrower topics for {self.id}: {e}")
            raise DatabaseOperationError(e)

    async def get_related_topics(self) -> List[Dict[str, Any]]:
        """Get related topics."""
        try:
            results = await repo_query(
                """
                SELECT
                    out AS topic_id,
                    out.name AS name,
                    out.domain AS domain,
                    strength
                FROM related_to_topic
                WHERE in = $id
                ORDER BY strength DESC
                """,
                {"id": ensure_record_id(self.id)}
            )
            return results
        except Exception as e:
            logger.error(f"Error getting related topics for {self.id}: {e}")
            raise DatabaseOperationError(e)

    async def get_experts(self, min_publications: int = 2, limit: int = 10) -> List[Dict[str, Any]]:
        """Find experts on this topic based on publication count."""
        try:
            results = await repo_query(
                """
                SELECT
                    person.id AS person_id,
                    person.name AS name,
                    person.h_index AS h_index,
                    count(source) AS publications
                FROM (
                    SELECT in AS source FROM discusses WHERE out = $id
                )
                LET person = (SELECT out FROM authored_by WHERE in = source)
                GROUP BY person.id
                HAVING publications >= $min_pubs
                ORDER BY publications DESC
                LIMIT $limit
                """,
                {
                    "id": ensure_record_id(self.id),
                    "min_pubs": min_publications,
                    "limit": limit
                }
            )
            return results
        except Exception as e:
            logger.error(f"Error finding experts for topic {self.id}: {e}")
            raise DatabaseOperationError(e)

    @classmethod
    async def find_by_name(cls, name: str) -> Optional["Topic"]:
        """Find a topic by exact name match."""
        try:
            results = await repo_query(
                "SELECT * FROM topic WHERE name = $name LIMIT 1",
                {"name": name}
            )
            if results:
                return cls(**results[0])
            return None
        except Exception as e:
            logger.error(f"Error finding topic by name: {e}")
            raise DatabaseOperationError(e)

    @classmethod
    async def search(
        cls,
        query: str,
        domain: Optional[str] = None,
        limit: int = 20
    ) -> List["Topic"]:
        """Search topics by name or description."""
        try:
            if domain:
                results = await repo_query(
                    """
                    SELECT * FROM topic
                    WHERE (name @@ $query OR description @@ $query)
                        AND domain = $domain
                    ORDER BY relevance DESC
                    LIMIT $limit
                    """,
                    {"query": query, "domain": domain, "limit": limit}
                )
            else:
                results = await repo_query(
                    """
                    SELECT * FROM topic
                    WHERE name @@ $query OR description @@ $query
                    ORDER BY relevance DESC
                    LIMIT $limit
                    """,
                    {"query": query, "limit": limit}
                )
            return [cls(**r) for r in results]
        except Exception as e:
            logger.error(f"Error searching topics: {e}")
            raise DatabaseOperationError(e)
