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

    async def _build_routed_model(self, source_id: str):
        """Build the J.4 failover-routed model to inject into summarization.

        Resolves the layered privacy mode from the source (a PRIVATE source pins
        summarization local) and returns a
        :class:`FailoverSummarizationModel` the strategies use in place of the
        env-driven ``AIFactory`` ollama model. Best-effort: on any resolution
        hiccup returns ``None`` so summarization falls back to the legacy
        env-driven path rather than failing.
        """
        try:
            from app_main.dependencies import get_source_repo
            from app_main.services.model_routing.privacy_resolver import (
                resolve_privacy_mode,
            )
            from app_main.services.model_routing.summarization_model import (
                FailoverSummarizationModel,
            )

            source_private: Optional[bool] = None
            notebook_id: Optional[str] = None
            try:
                source = await get_source_repo().get(source_id)
                if source is not None:
                    source_private = bool(getattr(source, "private", False))
            except Exception as e:  # noqa: BLE001 — degrade, never fail summary
                logger.warning(
                    f"summarization routing: could not read source.private for "
                    f"{source_id} ({e}); assuming not private"
                )

            mode = await resolve_privacy_mode(
                source_private=source_private, notebook_id=notebook_id
            )
            logger.info(
                "summarization routing (J.4): source={src} mode={mode}",
                src=source_id,
                mode=mode.value,
            )
            return FailoverSummarizationModel(
                mode=mode, source_id=source_id, notebook_id=notebook_id
            )
        except Exception as e:  # noqa: BLE001 — fall back to env-driven path
            logger.warning(
                f"summarization routing: failed to build routed model "
                f"({e}); using legacy env-driven summarization path."
            )
            return None

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

        # Build config, then inject the J.4 privacy-aware routed model so
        # summarization honors the same cloud/local routing + failover as
        # extraction/chat (instead of the hard-pinned env-driven ollama path).
        cfg = SummarizationConfig(strategy=strategy, **(config or {}))
        cfg.language_model = await self._build_routed_model(source_id)
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
