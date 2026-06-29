import asyncio
from typing import Annotated, Any, Dict, List, Optional

from ai_prompter import Prompter
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from shared.models import Source
from shared.models.source import SourceInsight
from app_main.config import LANGGRAPH_CHECKPOINT_FILE
from app_main.dependencies import get_context_service
from app_main.graphs.citations import citations_from_hits, format_citation_tag
from app_main.graphs.utils import provision_langchain_model

MODEL_INVOKE_TIMEOUT = 300  # seconds

# Cap the number of chunk blocks injected into the single-source prompt so a
# long document does not blow the context window. Source-level full_text is
# still available as a fallback when there are no chunks.
MAX_CHUNK_BLOCKS = 40
MAX_CHUNK_TOKENS = 40000


class SourceChatState(TypedDict):
    messages: Annotated[list, add_messages]
    source_id: str
    source: Optional[Source]
    insights: Optional[List[SourceInsight]]
    context: Optional[str]
    model_override: Optional[str]
    context_indicators: Optional[Dict[str, List[str]]]
    # Chunk-level source citations ({source, page, chunk_id, section}) derived
    # from the chunk context fed to the model (Track X.2). Additive.
    citations: Optional[List[Dict[str, Any]]]


async def call_model_with_source_context(
    state: SourceChatState, config: RunnableConfig
) -> dict:
    source_id = state.get("source_id")
    if not source_id:
        raise ValueError("source_id is required in state")

    svc = get_context_service()
    context_data = await svc.build_source_context(
        source_id=source_id,
        include_insights=True,
        include_notes=False,
        max_tokens=50000,
    )
    # Chunk-level context with page/section provenance (Track X.2). Empty when
    # the source has no chunks (audio/plain text) — the prompt then falls back
    # to source-level full_text, and citations stay source-level.
    chunk_context = await svc.build_source_chunks(
        source_id=source_id,
        max_chunks=MAX_CHUNK_BLOCKS,
        max_tokens=MAX_CHUNK_TOKENS,
    )

    source = None
    insights = []
    context_indicators: dict[str, list[str | None]] = {
        "sources": [],
        "insights": [],
        "notes": [],
    }

    if context_data.get("sources"):
        source_info = context_data["sources"][0]
        source = Source(**source_info) if isinstance(source_info, dict) else source_info
        context_indicators["sources"].append(source.id)

    if context_data.get("insights"):
        for insight_data in context_data["insights"]:
            insight = (
                SourceInsight(**insight_data)
                if isinstance(insight_data, dict)
                else insight_data
            )
            insights.append(insight)
            context_indicators["insights"].append(insight.id)

    formatted_context = _format_source_context(context_data, chunk_context)

    # Citation set = provenance of the chunk context fed to the model
    # (chunk-level when chunks exist; otherwise the source-level fallback so the
    # answer still cites the source). Deterministic; X.3 adds the membership
    # guard.
    if chunk_context:
        citations = citations_from_hits(chunk_context)
    elif source is not None:
        citations = citations_from_hits(
            [{"source": source.id, "physical_page": None}]
        )
    else:
        citations = []

    prompt_data = {
        "source": source.model_dump() if source else None,
        "insights": [insight.model_dump() for insight in insights] if insights else [],
        "context": formatted_context,
        "context_indicators": context_indicators,
    }

    system_prompt = Prompter(prompt_template="source_chat").render(data=prompt_data)
    payload = [SystemMessage(content=system_prompt)] + state.get("messages", [])

    model_id = (
        config.get("configurable", {}).get("model_id")
        or state.get("model_override")
    )
    model = await provision_langchain_model(
        str(payload), model_id, "chat", max_tokens=8192
    )

    ai_message = await asyncio.wait_for(
        model.ainvoke(payload), timeout=MODEL_INVOKE_TIMEOUT
    )

    return {
        "messages": ai_message,
        "source": source,
        "insights": insights,
        "context": formatted_context,
        "context_indicators": context_indicators,
        "citations": citations,
    }


def _format_chunk_context(chunk_context: List[Dict[str, Any]]) -> str:
    """Render chunk-level context blocks, each prefixed with a provenance tag
    ``[source: <id> | p.<page> | <section>]`` (Track X.2) so the model can cite
    the exact page/section. Returns ``""`` when there are no chunks."""
    if not chunk_context:
        return ""
    parts = ["## SOURCE PASSAGES (page-cited)"]
    for chunk in chunk_context:
        tag = format_citation_tag(chunk)
        if tag:
            parts.append(tag)
        text = chunk.get("text") or ""
        parts.append(text)
        parts.append("")
    return "\n".join(parts)


def _format_source_context(
    context_data: Dict,
    chunk_context: Optional[List[Dict[str, Any]]] = None,
) -> str:
    context_parts = []

    # Prefer chunk-level, page-cited passages when available; the source-level
    # full_text below remains as a fallback/overview.
    chunk_block = _format_chunk_context(chunk_context or [])
    if chunk_block:
        context_parts.append(chunk_block)

    if context_data.get("sources"):
        context_parts.append("## SOURCE CONTENT")
        for source in context_data["sources"]:
            if isinstance(source, dict):
                context_parts.append(f"**Source ID:** {source.get('id', 'Unknown')}")
                context_parts.append(f"**Title:** {source.get('title', 'No title')}")
                if source.get("full_text"):
                    full_text = source["full_text"]
                    if len(full_text) > 5000:
                        full_text = full_text[:5000] + "...\n[Content truncated]"
                    context_parts.append(f"**Content:**\n{full_text}")
                context_parts.append("")

    if context_data.get("insights"):
        context_parts.append("## SOURCE INSIGHTS")
        for insight in context_data["insights"]:
            if isinstance(insight, dict):
                context_parts.append(f"**Insight ID:** {insight.get('id', 'Unknown')}")
                context_parts.append(
                    f"**Type:** {insight.get('insight_type', 'Unknown')}"
                )
                context_parts.append(
                    f"**Content:** {insight.get('content', 'No content')}"
                )
                context_parts.append("")

    if context_data.get("metadata"):
        metadata = context_data["metadata"]
        context_parts.append("## CONTEXT METADATA")
        context_parts.append(f"- Source count: {metadata.get('source_count', 0)}")
        context_parts.append(f"- Insight count: {metadata.get('insight_count', 0)}")
        context_parts.append(f"- Total tokens: {context_data.get('total_tokens', 0)}")
        context_parts.append("")

    return "\n".join(context_parts)


memory = SqliteSaver.from_conn_string(LANGGRAPH_CHECKPOINT_FILE)

source_chat_state = StateGraph(SourceChatState)
source_chat_state.add_node("source_chat_agent", call_model_with_source_context)
source_chat_state.add_edge(START, "source_chat_agent")
source_chat_state.add_edge("source_chat_agent", END)
source_chat_graph = source_chat_state.compile(checkpointer=memory)
