"""
ContextService — builds context dicts from sources, insights and notes.

Replaces the monolith ``ContextBuilder`` by using injected surrealdb-service
repositories instead of ORM-style ``Source.get()`` calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from loguru import logger

from shared.utils import token_count

from surrealdb_service.repositories import (
    NoteRepository,
    NotebookRepository,
    SourceInsightRepository,
    SourceRepository,
)


# ---------------------------------------------------------------------------
# Supporting dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ContextItem:
    """A single item in the context with auto-calculated token count."""

    id: str
    type: Literal["source", "note", "insight"]
    content: Dict[str, Any]
    priority: int = 0
    tokens: Optional[int] = None

    def __post_init__(self) -> None:
        if self.tokens is None:
            self.tokens = token_count(str(self.content))


@dataclass
class ContextConfig:
    """Configuration knobs for context building."""

    include_insights: bool = True
    include_notes: bool = True
    max_tokens: Optional[int] = None
    priority_weights: Dict[str, int] = field(
        default_factory=lambda: {"source": 100, "note": 50, "insight": 75}
    )


# ---------------------------------------------------------------------------
# ContextService
# ---------------------------------------------------------------------------

class ContextService:
    """Build context dicts suitable for chat prompts.

    The output format matches the monolith ``ContextBuilder`` so existing
    consumers (``source_chat.py``'s ``_format_source_context``) work without
    changes.
    """

    def __init__(
        self,
        source_repo: SourceRepository,
        insight_repo: SourceInsightRepository,
        notebook_repo: NotebookRepository,
        note_repo: NoteRepository,
    ) -> None:
        self._source_repo = source_repo
        self._insight_repo = insight_repo
        self._notebook_repo = notebook_repo
        self._note_repo = note_repo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def build_source_context(
        self,
        source_id: str,
        include_insights: bool = True,
        include_notes: bool = False,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Build context for a single source.

        Returns a dict with keys ``sources``, ``insights``, ``notes``,
        ``total_tokens``, ``total_items``, and ``metadata``.
        """
        items: List[ContextItem] = []

        full_source_id = (
            source_id
            if source_id.startswith("source:")
            else f"source:{source_id}"
        )

        source = await self._source_repo.get(full_source_id)
        if source is None:
            logger.warning(f"Source {source_id} not found")
            return self._empty_response()

        # Build source content dict
        source_dict: Dict[str, Any] = {
            "id": source.id,
            "title": getattr(source, "title", None),
            "full_text": getattr(source, "full_text", None),
        }
        items.append(
            ContextItem(
                id=source.id or full_source_id,
                type="source",
                content=source_dict,
                priority=100,
            )
        )

        # Insights
        if include_insights:
            insights = await self._insight_repo.get_by_source(full_source_id)
            for insight in insights:
                insight_dict: Dict[str, Any] = {
                    "id": insight.id,
                    "source_id": full_source_id,
                    "insight_type": getattr(insight, "insight_type", None),
                    "content": getattr(insight, "content", None),
                }
                items.append(
                    ContextItem(
                        id=insight.id or "",
                        type="insight",
                        content=insight_dict,
                        priority=75,
                    )
                )

        # De-duplicate, prioritize, truncate
        items = self._deduplicate(items)
        items.sort(key=lambda x: x.priority, reverse=True)
        if max_tokens:
            items = self._truncate(items, max_tokens)

        return self._format(items)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_response() -> Dict[str, Any]:
        return {
            "sources": [],
            "insights": [],
            "notes": [],
            "total_tokens": 0,
            "total_items": 0,
            "metadata": {
                "source_count": 0,
                "note_count": 0,
                "insight_count": 0,
            },
        }

    @staticmethod
    def _deduplicate(items: List[ContextItem]) -> List[ContextItem]:
        seen: set[str] = set()
        result: list[ContextItem] = []
        for item in items:
            if item.id not in seen:
                result.append(item)
                seen.add(item.id)
        return result

    @staticmethod
    def _truncate(
        items: List[ContextItem], max_tokens: int
    ) -> List[ContextItem]:
        total = sum(i.tokens or 0 for i in items)
        if total <= max_tokens:
            return items
        # Remove lowest-priority items from the end
        while total > max_tokens and items:
            removed = items.pop()
            total -= removed.tokens or 0
        return items

    @staticmethod
    def _format(items: List[ContextItem]) -> Dict[str, Any]:
        sources: list[dict] = []
        notes: list[dict] = []
        insights: list[dict] = []

        for item in items:
            if item.type == "source":
                sources.append(item.content)
            elif item.type == "note":
                notes.append(item.content)
            elif item.type == "insight":
                insights.append(item.content)

        total_tokens = sum(i.tokens or 0 for i in items)

        return {
            "sources": sources,
            "insights": insights,
            "notes": notes,
            "total_tokens": total_tokens,
            "total_items": len(items),
            "metadata": {
                "source_count": len(sources),
                "note_count": len(notes),
                "insight_count": len(insights),
            },
        }
