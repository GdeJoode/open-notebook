"""
SourceSummarizationOrchestrator — runs transformation graph on a source.

Drives the LangGraph transformation pipeline for one or more transformations.
The transformation graph is lazy-imported because it pulls in ai_prompter
and other optional deps that not every dev env has.

Pulled out of SourceProcessingService in Phase 3 of the refactor.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import Source
from surrealdb_service.repositories import SourceRepository, TransformationRepository


class SourceSummarizationOrchestrator:
    """Runs summarization/transformation graphs for a source."""

    def __init__(
        self,
        source_repo: SourceRepository,
        transformation_repo: TransformationRepository,
    ) -> None:
        self.source_repo = source_repo
        self.transformation_repo = transformation_repo

    async def run_summaries(
        self,
        source_id: str,
        transformation_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Run summarization transformations on a source.

        When ``transformation_ids`` is omitted the source-level defaults
        are used. Raises ValueError when the source or a specified
        transformation id does not exist.
        """
        source = await self.source_repo.get(source_id)
        if source is None:
            raise ValueError(f"Source '{source_id}' not found")

        count = await self._run_transformations(
            source, transformation_ids=transformation_ids,
        )
        insights_list = await self.source_repo.get_insights(source_id)
        return {
            "source_id": str(source.id),
            "insights_created": len(insights_list),
            "transformations_run": count,
        }

    async def _run_transformations(
        self,
        source: Source,
        *,
        transformation_ids: Optional[List[str]] = None,
    ) -> int:
        """Apply the transformation graph to each requested transformation.

        Returns the number of transformations that completed without raising.
        """
        from app_main.graphs.transformation import graph as transform_graph

        transformations = []
        if transformation_ids:
            for tid in transformation_ids:
                t = await self.transformation_repo.get(tid)
                if t is None:
                    raise ValueError(f"Transformation '{tid}' not found")
                transformations.append(t)
        else:
            transformations = await self.transformation_repo.get_defaults()

        if not transformations:
            return 0

        content = source.full_text
        if not content:
            logger.warning(
                f"Source {source.id} has no content for transformations"
            )
            return 0

        count = 0
        for transformation in transformations:
            logger.debug(f"Applying transformation {transformation.name}")
            try:
                await transform_graph.ainvoke(
                    dict(
                        input_text=content,
                        source=source,
                        transformation=transformation,
                    )
                )
                count += 1
            except Exception as e:
                logger.error(
                    f"Transformation '{transformation.name}' failed: {e}"
                )

        return count
