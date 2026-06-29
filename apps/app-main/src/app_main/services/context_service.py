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
    ChunkRepository,
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
        chunk_repo: Optional[ChunkRepository] = None,
    ) -> None:
        self._source_repo = source_repo
        self._insight_repo = insight_repo
        self._notebook_repo = notebook_repo
        self._note_repo = note_repo
        # Optional — only the chunk-level (page-cited) context path needs it
        # (Track X.2). Kept optional so existing constructions are unaffected.
        self._chunk_repo = chunk_repo

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

    async def build_source_chunks(
        self,
        source_id: str,
        max_chunks: Optional[int] = None,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Chunk-level context for a single source (Track X.2).

        Returns the source's content chunks in document order, each carrying
        its provenance (``chunk_id``/``physical_page``/``printed_page``/
        ``section_path``/``element_type``/``source``) so ``source_chat`` can
        cite the exact page/section rather than only the source title. Noise
        chunks (``is_content == False``) are skipped.

        Degrades gracefully: without an injected ``chunk_repo`` (or on a lookup
        failure) returns ``[]`` — the caller falls back to source-level
        ``full_text`` and still answers.

        Args:
            source_id: Source id (``source:`` prefix optional).
            max_chunks: Cap on the number of chunks returned (by document order).
            max_tokens: Token budget; chunks are accumulated in order until the
                budget would be exceeded.
        """
        if self._chunk_repo is None:
            return []

        full_source_id = (
            source_id
            if source_id.startswith("source:")
            else f"source:{source_id}"
        )

        try:
            chunks = await self._chunk_repo.get_by_source(full_source_id)
        except Exception as e:  # provenance is additive — never break the chat
            logger.warning(f"Chunk context lookup failed for {source_id}: {e}")
            return []

        out: List[Dict[str, Any]] = []
        running_tokens = 0
        for chunk in chunks:
            if getattr(chunk, "is_content", True) is False:
                continue
            text = getattr(chunk, "text", "") or ""
            record: Dict[str, Any] = {
                "chunk_id": getattr(chunk, "id", None),
                "source": full_source_id,
                "text": text,
                "physical_page": getattr(chunk, "physical_page", None),
                "printed_page": getattr(chunk, "printed_page", None),
                "section_path": getattr(chunk, "section_path", None),
                "element_type": getattr(chunk, "element_type", None),
                "order": getattr(chunk, "order", None),
            }
            if max_tokens is not None:
                running_tokens += token_count(text)
                if running_tokens > max_tokens and out:
                    break
            out.append(record)
            if max_chunks is not None and len(out) >= max_chunks:
                break
        return out

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
