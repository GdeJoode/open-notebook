"""
Ontology Evolution Agent for self-improvement.

Tracks gaps in the ontology (unmapped entities), analyzes frequency patterns,
and generates schema proposals for new entity types or relationships.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel, Field


class GapStatus(str, Enum):
    """Status of an ontology gap."""

    PENDING = "pending"  # Not yet reviewed
    REVIEWING = "reviewing"  # Under review
    PROPOSED = "proposed"  # Schema proposal created
    RESOLVED = "resolved"  # Added to ontology
    IGNORED = "ignored"  # Intentionally not added


class ProposalStatus(str, Enum):
    """Status of a schema proposal."""

    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"


class ProposalType(str, Enum):
    """Type of schema proposal."""

    NEW_ENTITY_TYPE = "new_entity_type"
    NEW_RELATIONSHIP_TYPE = "new_relationship_type"
    NEW_PROPERTY = "new_property"
    MODIFY_ENTITY_TYPE = "modify_entity_type"
    MODIFY_RELATIONSHIP_TYPE = "modify_relationship_type"


@dataclass
class OntologyGap:
    """Represents a gap in the ontology - an entity that doesn't match known types."""

    id: Optional[str] = None
    entity_text: str = ""
    entity_type_guess: Optional[str] = None
    context: Optional[str] = None
    frequency: int = 1
    source_ids: List[str] = field(default_factory=list)
    ontology_name: str = "general"
    status: GapStatus = GapStatus.PENDING
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "entity_text": self.entity_text,
            "entity_type_guess": self.entity_type_guess,
            "context": self.context,
            "frequency": self.frequency,
            "source_ids": self.source_ids,
            "ontology_name": self.ontology_name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class SchemaProposal(BaseModel):
    """A proposal for ontology schema changes."""

    id: Optional[str] = None
    ontology_name: str
    proposal_type: ProposalType
    name: str  # Name of proposed entity type, relationship, etc.
    definition: Dict[str, Any] = Field(default_factory=dict)
    rationale: str
    evidence_count: int = 0
    related_gap_ids: List[str] = Field(default_factory=list)
    status: ProposalStatus = ProposalStatus.PROPOSED
    reviewed_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class OntologyEvolutionAgent:
    """
    Agent for tracking ontology gaps and generating schema proposals.

    Responsibilities:
    1. Track entities that don't match known types (gaps)
    2. Analyze frequency patterns in gaps
    3. Generate schema proposals when frequency threshold reached
    4. Manage proposal workflow (approve/reject/implement)
    """

    def __init__(
        self,
        frequency_threshold: int = 5,
        auto_propose: bool = True,
    ):
        """
        Initialize evolution agent.

        Args:
            frequency_threshold: Number of occurrences before auto-proposing
            auto_propose: Whether to automatically create proposals
        """
        self.frequency_threshold = frequency_threshold
        self.auto_propose = auto_propose

    async def record_gap(
        self,
        entity_text: str,
        entity_type_guess: Optional[str] = None,
        context: Optional[str] = None,
        source_id: Optional[str] = None,
        ontology_name: str = "general",
    ) -> OntologyGap:
        """
        Record a gap in the ontology.

        Args:
            entity_text: The entity text that didn't match
            entity_type_guess: LLM's guess at what type it might be
            context: Context where the entity was found
            source_id: ID of source document
            ontology_name: Which ontology had the gap

        Returns:
            The created or updated OntologyGap
        """
        try:
            from surrealdb_service.connection import execute_query as repo_query

            # Check if gap already exists
            existing = await repo_query(
                """
                SELECT * FROM ontology_gap
                WHERE entity_text = $entity_text AND ontology_name = $ontology_name
                LIMIT 1
                """,
                {"entity_text": entity_text, "ontology_name": ontology_name},
            )

            if existing:
                # Update existing gap
                gap_id = existing[0].get("id")
                current_frequency = existing[0].get("frequency", 1)
                current_sources = existing[0].get("source_documents", [])

                if source_id and source_id not in current_sources:
                    current_sources.append(source_id)

                await repo_query(
                    """
                    UPDATE ontology_gap
                    SET frequency = $frequency, source_documents = $sources
                    WHERE id = $id
                    """,
                    {
                        "id": gap_id,
                        "frequency": current_frequency + 1,
                        "sources": current_sources,
                    },
                )

                gap = OntologyGap(
                    id=gap_id,
                    entity_text=entity_text,
                    entity_type_guess=entity_type_guess or existing[0].get("entity_type_guess"),
                    context=context or existing[0].get("context"),
                    frequency=current_frequency + 1,
                    source_ids=current_sources,
                    ontology_name=ontology_name,
                    status=GapStatus(existing[0].get("status", "pending")),
                )

                logger.debug(
                    f"Updated gap '{entity_text}' (frequency: {gap.frequency})"
                )

                # Check if we should auto-propose
                if self.auto_propose and gap.frequency >= self.frequency_threshold:
                    await self._maybe_create_proposal(gap)

                return gap

            else:
                # Create new gap
                sources = [source_id] if source_id else []
                result = await repo_query(
                    """
                    CREATE ontology_gap SET
                        entity_text = $entity_text,
                        entity_type_guess = $entity_type_guess,
                        context = $context,
                        frequency = 1,
                        source_documents = $sources,
                        ontology_name = $ontology_name,
                        status = 'pending',
                        created_at = time::now()
                    """,
                    {
                        "entity_text": entity_text,
                        "entity_type_guess": entity_type_guess,
                        "context": context,
                        "sources": sources,
                        "ontology_name": ontology_name,
                    },
                )

                gap = OntologyGap(
                    id=result[0].get("id") if result else None,
                    entity_text=entity_text,
                    entity_type_guess=entity_type_guess,
                    context=context,
                    frequency=1,
                    source_ids=sources,
                    ontology_name=ontology_name,
                    status=GapStatus.PENDING,
                )

                logger.info(f"Recorded new ontology gap: '{entity_text}'")
                return gap

        except Exception as e:
            logger.error(f"Failed to record gap: {e}")
            return OntologyGap(
                entity_text=entity_text,
                entity_type_guess=entity_type_guess,
                context=context,
                ontology_name=ontology_name,
            )

    async def _maybe_create_proposal(self, gap: OntologyGap) -> Optional[SchemaProposal]:
        """Create a proposal if threshold reached and no existing proposal."""
        try:
            from surrealdb_service.connection import execute_query as repo_query

            # Check if proposal already exists
            existing = await repo_query(
                """
                SELECT * FROM schema_proposal
                WHERE name = $name AND ontology_name = $ontology_name
                LIMIT 1
                """,
                {"name": gap.entity_text, "ontology_name": gap.ontology_name},
            )

            if existing:
                return None  # Already has proposal

            # Create new proposal
            proposal = await self.create_proposal_from_gap(gap)
            return proposal

        except Exception as e:
            logger.debug(f"Could not check/create proposal: {e}")
            return None

    async def create_proposal_from_gap(
        self,
        gap: OntologyGap,
    ) -> Optional[SchemaProposal]:
        """
        Create a schema proposal from a gap.

        Args:
            gap: The ontology gap to propose a solution for

        Returns:
            The created SchemaProposal or None
        """
        try:
            from surrealdb_service.connection import execute_query as repo_query, ensure_record_id

            # Determine proposal type based on entity_type_guess
            proposal_type = ProposalType.NEW_ENTITY_TYPE

            # Generate definition based on gap analysis
            definition = {
                "name": gap.entity_text.title().replace(" ", ""),  # PascalCase
                "description": f"Entity type for '{gap.entity_text}' (auto-generated)",
                "examples": [gap.entity_text],
                "properties": [
                    {
                        "name": "name",
                        "data_type": "string",
                        "description": "Name of the entity",
                        "required": True,
                    }
                ],
            }

            if gap.entity_type_guess:
                definition["parent_type"] = gap.entity_type_guess

            rationale = (
                f"Entity '{gap.entity_text}' appeared {gap.frequency} times "
                f"across {len(gap.source_ids)} source(s) but didn't match any "
                f"existing entity type in the {gap.ontology_name} ontology."
            )

            # Create in database
            result = await repo_query(
                """
                CREATE schema_proposal SET
                    ontology_name = $ontology_name,
                    proposal_type = $proposal_type,
                    name = $name,
                    definition = $definition,
                    rationale = $rationale,
                    evidence_count = $evidence_count,
                    related_gaps = $related_gaps,
                    status = 'proposed',
                    created_at = time::now()
                """,
                {
                    "ontology_name": gap.ontology_name,
                    "proposal_type": proposal_type.value,
                    "name": gap.entity_text,
                    "definition": definition,
                    "rationale": rationale,
                    "evidence_count": gap.frequency,
                    "related_gaps": [ensure_record_id(gap.id)] if gap.id else [],
                },
            )

            if result:
                # Update gap status
                if gap.id:
                    await repo_query(
                        "UPDATE ontology_gap SET status = 'proposed' WHERE id = $id",
                        {"id": ensure_record_id(gap.id)},
                    )

                logger.info(
                    f"Created schema proposal for '{gap.entity_text}' "
                    f"(evidence: {gap.frequency} occurrences)"
                )

                return SchemaProposal(
                    id=result[0].get("id"),
                    ontology_name=gap.ontology_name,
                    proposal_type=proposal_type,
                    name=gap.entity_text,
                    definition=definition,
                    rationale=rationale,
                    evidence_count=gap.frequency,
                    related_gap_ids=[gap.id] if gap.id else [],
                )

        except Exception as e:
            logger.error(f"Failed to create proposal: {e}")

        return None

    async def list_gaps(
        self,
        ontology_name: str = "general",
        status: Optional[GapStatus] = None,
        min_frequency: int = 1,
        limit: int = 100,
    ) -> List[OntologyGap]:
        """
        List ontology gaps.

        Args:
            ontology_name: Filter by ontology
            status: Filter by status
            min_frequency: Minimum occurrence count
            limit: Maximum results

        Returns:
            List of OntologyGap objects
        """
        try:
            from surrealdb_service.connection import execute_query as repo_query

            query = """
                SELECT * FROM ontology_gap
                WHERE ontology_name = $ontology_name
                AND frequency >= $min_frequency
            """
            params: Dict[str, Any] = {
                "ontology_name": ontology_name,
                "min_frequency": min_frequency,
            }

            if status:
                query += " AND status = $status"
                params["status"] = status.value

            query += " ORDER BY frequency DESC LIMIT $limit"
            params["limit"] = limit

            result = await repo_query(query, params)

            return [
                OntologyGap(
                    id=r.get("id"),
                    entity_text=r.get("entity_text", ""),
                    entity_type_guess=r.get("entity_type_guess"),
                    context=r.get("context"),
                    frequency=r.get("frequency", 1),
                    source_ids=r.get("source_documents", []),
                    ontology_name=r.get("ontology_name", "general"),
                    status=GapStatus(r.get("status", "pending")),
                )
                for r in (result or [])
            ]

        except Exception as e:
            logger.error(f"Failed to list gaps: {e}")
            return []

    async def list_proposals(
        self,
        ontology_name: str = "general",
        status: Optional[ProposalStatus] = None,
        limit: int = 100,
    ) -> List[SchemaProposal]:
        """
        List schema proposals.

        Args:
            ontology_name: Filter by ontology
            status: Filter by status
            limit: Maximum results

        Returns:
            List of SchemaProposal objects
        """
        try:
            from surrealdb_service.connection import execute_query as repo_query

            query = """
                SELECT * FROM schema_proposal
                WHERE ontology_name = $ontology_name
            """
            params: Dict[str, Any] = {"ontology_name": ontology_name}

            if status:
                query += " AND status = $status"
                params["status"] = status.value

            query += " ORDER BY evidence_count DESC LIMIT $limit"
            params["limit"] = limit

            result = await repo_query(query, params)

            return [
                SchemaProposal(
                    id=r.get("id"),
                    ontology_name=r.get("ontology_name", "general"),
                    proposal_type=ProposalType(r.get("proposal_type", "new_entity_type")),
                    name=r.get("name", ""),
                    definition=r.get("definition", {}),
                    rationale=r.get("rationale", ""),
                    evidence_count=r.get("evidence_count", 0),
                    related_gap_ids=r.get("related_gaps", []),
                    status=ProposalStatus(r.get("status", "proposed")),
                )
                for r in (result or [])
            ]

        except Exception as e:
            logger.error(f"Failed to list proposals: {e}")
            return []

    async def approve_proposal(
        self,
        proposal_id: str,
    ) -> bool:
        """
        Approve a schema proposal.

        Args:
            proposal_id: ID of proposal to approve

        Returns:
            True if successfully approved
        """
        try:
            from surrealdb_service.connection import execute_query as repo_query, ensure_record_id

            await repo_query(
                """
                UPDATE schema_proposal
                SET status = 'approved', reviewed_at = time::now()
                WHERE id = $id
                """,
                {"id": ensure_record_id(proposal_id)},
            )

            logger.info(f"Approved schema proposal: {proposal_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to approve proposal: {e}")
            return False

    async def reject_proposal(
        self,
        proposal_id: str,
    ) -> bool:
        """
        Reject a schema proposal.

        Args:
            proposal_id: ID of proposal to reject

        Returns:
            True if successfully rejected
        """
        try:
            from surrealdb_service.connection import execute_query as repo_query, ensure_record_id

            await repo_query(
                """
                UPDATE schema_proposal
                SET status = 'rejected', reviewed_at = time::now()
                WHERE id = $id
                """,
                {"id": ensure_record_id(proposal_id)},
            )

            # Update related gaps to ignored
            result = await repo_query(
                "SELECT related_gaps FROM schema_proposal WHERE id = $id",
                {"id": ensure_record_id(proposal_id)},
            )

            if result and result[0].get("related_gaps"):
                for gap_id in result[0]["related_gaps"]:
                    await repo_query(
                        "UPDATE ontology_gap SET status = 'ignored' WHERE id = $id",
                        {"id": ensure_record_id(gap_id)},
                    )

            logger.info(f"Rejected schema proposal: {proposal_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to reject proposal: {e}")
            return False

    async def get_gap_statistics(
        self,
        ontology_name: str = "general",
    ) -> Dict[str, Any]:
        """
        Get statistics about ontology gaps.

        Args:
            ontology_name: Filter by ontology

        Returns:
            Dictionary with gap statistics
        """
        try:
            from surrealdb_service.connection import execute_query as repo_query

            # Total gaps
            total_result = await repo_query(
                "SELECT count() AS total FROM ontology_gap WHERE ontology_name = $name GROUP ALL",
                {"name": ontology_name},
            )
            total = total_result[0].get("total", 0) if total_result else 0

            # By status
            status_result = await repo_query(
                """
                SELECT status, count() AS count
                FROM ontology_gap
                WHERE ontology_name = $name
                GROUP BY status
                """,
                {"name": ontology_name},
            )

            by_status = {
                r.get("status", "unknown"): r.get("count", 0)
                for r in (status_result or [])
            }

            # High frequency gaps
            high_freq_result = await repo_query(
                """
                SELECT entity_text, frequency
                FROM ontology_gap
                WHERE ontology_name = $name AND frequency >= $threshold
                ORDER BY frequency DESC
                LIMIT 10
                """,
                {"name": ontology_name, "threshold": self.frequency_threshold},
            )

            high_frequency = [
                {"entity": r.get("entity_text"), "frequency": r.get("frequency")}
                for r in (high_freq_result or [])
            ]

            return {
                "ontology_name": ontology_name,
                "total_gaps": total,
                "by_status": by_status,
                "high_frequency_gaps": high_frequency,
                "frequency_threshold": self.frequency_threshold,
            }

        except Exception as e:
            logger.error(f"Failed to get gap statistics: {e}")
            return {"ontology_name": ontology_name, "error": str(e)}
