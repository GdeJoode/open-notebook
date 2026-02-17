"""
Summarization service - business logic for document summarization.

Wraps the SummarizationWorkflow pipeline and persists results to the
summary table via SummaryRepository.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from surrealdb_service.repositories.summary import SummaryRepository


# All known strategies and their implementation status.
# Matches SummarizationStrategy enum in summarization.models.result.
_STRATEGIES = [
    {"name": "raptor", "implemented": True},
    {"name": "treekg", "implemented": True},
    {"name": "naive", "implemented": True},
    {"name": "map_reduce", "implemented": False},
    {"name": "refine", "implemented": False},
    {"name": "walking_tree", "implemented": False},
]

_IMPLEMENTED = {s["name"] for s in _STRATEGIES if s["implemented"]}


class SummarizationService:
    """Service for summarization operations."""

    def __init__(self, summary_repo: SummaryRepository):
        self.summary_repo = summary_repo

    async def list_strategies(self) -> List[Dict[str, Any]]:
        """Return all strategies with their implementation status."""
        return list(_STRATEGIES)

    async def list_summaries(
        self, source_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List generated summaries, optionally filtered by source."""
        return await self.summary_repo.list_summaries(source_id=source_id)

    async def get_summary(self, summary_id: str) -> Optional[Dict[str, Any]]:
        """Get a single summary with full content."""
        return await self.summary_repo.get_summary(summary_id)

    async def delete_summary(self, summary_id: str) -> bool:
        """Delete a summary."""
        return await self.summary_repo.delete_summary(summary_id)

    async def generate_summary(
        self,
        source_id: str,
        strategy: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a summary for a source.

        Loads chunks from the DB, runs the summarization workflow,
        and persists the result.
        """
        from summarization.models.result import SummarizationStrategy
        from summarization.workflow import SummarizationWorkflow
        from summarization.config import SummarizationConfig
        from summarization.models.result import ChunkInput
        from surrealdb_service.repositories import ChunkRepository

        # Validate strategy
        try:
            strat_enum = SummarizationStrategy(strategy)
        except ValueError:
            raise ValueError(f"Unknown strategy: {strategy}")

        if strategy not in _IMPLEMENTED:
            raise ValueError(f"Strategy '{strategy}' is not yet implemented")

        # Load source chunks
        chunk_repo = ChunkRepository()
        chunks_raw = await chunk_repo.get_by_source(source_id)
        if not chunks_raw:
            raise ValueError(f"No chunks found for source '{source_id}'")

        chunks = [
            ChunkInput(
                text=c.text,
                chunk_id=str(c.id or ""),
                order=c.order,
            )
            for c in chunks_raw
        ]

        # Build config
        cfg = SummarizationConfig(strategy=strategy, **(config or {}))
        workflow = SummarizationWorkflow(config=cfg)
        result = await workflow.process(chunks)

        # Persist
        record = {
            "source_id": source_id,
            "strategy": result.strategy.value,
            "document_summary": result.document_summary,
            "summary_nodes": [n.model_dump() for n in result.summary_nodes],
            "num_layers": result.num_layers,
            "num_input_chunks": result.num_input_chunks,
            "metadata": result.metadata,
        }
        saved = await self.summary_repo.create_summary(record)
        return saved
