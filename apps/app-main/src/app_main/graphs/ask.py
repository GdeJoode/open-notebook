import asyncio
import operator
from typing import Annotated, List

from ai_prompter import Prompter
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from loguru import logger
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from surrealdb_service.repositories import SearchRepository
from app_main.graphs.citations import (
    citations_from_hits,
    format_citation_tag,
    guard_answer_citations,
    guard_citation_array,
    merge_citations,
)
from app_main.graphs.utils import provision_langchain_model
from shared.utils.text import clean_thinking_content

MODEL_INVOKE_TIMEOUT = 300  # seconds


class SubGraphState(TypedDict):
    question: str
    term: str
    instructions: str
    results: dict
    answer: str
    ids: list


class Search(BaseModel):
    term: str
    instructions: str = Field(
        description="Tell the answering LLM what information you need extracted from this search"
    )


class Strategy(BaseModel):
    reasoning: str
    searches: List[Search] = Field(
        default_factory=list,
        description="You can add up to five searches to this strategy",
    )


class ThreadState(TypedDict):
    question: str
    strategy: Strategy
    answers: Annotated[list, operator.add]
    # Per-sub-answer citation lists, accumulated across the fan-out branches
    # (Track X.2). Merged/de-duplicated into ``citations`` by the final node.
    citation_groups: Annotated[list, operator.add]
    # Retrieval hits per sub-answer (Track X.3): the context the final node
    # validates the emitted citations against (the array safety-net).
    retrieval_hits: Annotated[list, operator.add]
    final_answer: str
    citations: list


async def call_model_with_messages(state: ThreadState, config: RunnableConfig) -> dict:
    parser = PydanticOutputParser(pydantic_object=Strategy)
    system_prompt = Prompter(prompt_template="ask/entry", parser=parser).render(
        data=state
    )
    model = await provision_langchain_model(
        system_prompt,
        config.get("configurable", {}).get("strategy_model"),
        "tools",
        max_tokens=2000,
        structured=dict(type="json"),
    )
    ai_message = await asyncio.wait_for(
        model.ainvoke(system_prompt), timeout=MODEL_INVOKE_TIMEOUT
    )

    message_content = ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)
    cleaned_content = clean_thinking_content(message_content)

    strategy = parser.parse(cleaned_content)

    return {"strategy": strategy}


async def trigger_queries(state: ThreadState, config: RunnableConfig):
    return [
        Send(
            "provide_answer",
            {
                "question": state["question"],
                "instructions": s.instructions,
                "term": s.term,
            },
        )
        for s in state["strategy"].searches
    ]


def _format_results_with_provenance(results: list) -> str:
    """Render retrieval hits as prompt blocks, each prefixed with a provenance
    tag ``[source: <id> | p.<page> | <section>]`` (Track X.2).

    The tag lets the answering model attribute each claim to the exact
    source/page/section. Page-less hits (insights/notes/plain text) still carry
    a ``[source: ...]`` tag. The full hit dict is included after the tag so the
    existing answer behaviour (which consumed the raw ``results``) is preserved.
    """
    blocks = []
    for hit in results:
        tag = format_citation_tag(hit)
        prefix = f"{tag}\n" if tag else ""
        blocks.append(f"{prefix}{hit}")
    return "\n\n".join(blocks)


async def provide_answer(state: SubGraphState, config: RunnableConfig) -> dict:
    payload = dict(state)
    from retrieval.service import RetrievalService

    retrieval_svc = RetrievalService(SearchRepository())
    # Try vector search (embeds the query text); falls back to text search
    # if no embedding model is configured. Opt into provenance hydration
    # (Track X.2) so each hit carries its chunk page/section for citation.
    try:
        from app_main.dependencies import get_embedding_service

        embed_svc = await get_embedding_service()
        retrieval_svc = RetrievalService(
            SearchRepository(), embedding_model=embed_svc.embedding_model
        )
        results = await retrieval_svc.vector_search(state["term"], 10, hydrate=True)
    except Exception:
        results = await retrieval_svc.text_search(state["term"], 10, hydrate=True)
    if len(results) == 0:
        return {"answers": [], "citation_groups": [], "retrieval_hits": []}
    payload["results"] = _format_results_with_provenance(results)
    ids = [r["id"] for r in results]
    payload["ids"] = ids
    # Deterministic citation set = provenance of the context hits fed to the
    # LLM (Track X.2; X.3 adds the membership/faithfulness guard).
    citations = citations_from_hits(results)
    system_prompt = Prompter(prompt_template="ask/query_process").render(data=payload)
    model = await provision_langchain_model(
        system_prompt,
        config.get("configurable", {}).get("answer_model"),
        "tools",
        max_tokens=2000,
    )
    ai_message = await asyncio.wait_for(
        model.ainvoke(system_prompt), timeout=MODEL_INVOKE_TIMEOUT
    )
    ai_content = ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)
    answer_text = clean_thinking_content(ai_content)
    # The Track X.3 inline-marker guard runs on the FINAL synthesized answer
    # (``write_final_answer``) — the user-visible text — against the accumulated
    # retrieval set, not here on the intermediate sub-answers (which are never
    # surfaced). We only thread the per-sub-answer ``results`` into state so the
    # final node can validate against the full union of hits.
    return {
        "answers": [answer_text],
        "citation_groups": [citations],
        "retrieval_hits": results,
    }


async def write_final_answer(state: ThreadState, config: RunnableConfig) -> dict:
    system_prompt = Prompter(prompt_template="ask/final_answer").render(data=state)
    model = await provision_langchain_model(
        system_prompt,
        config.get("configurable", {}).get("final_answer_model"),
        "tools",
        max_tokens=2000,
    )
    ai_message = await asyncio.wait_for(
        model.ainvoke(system_prompt), timeout=MODEL_INVOKE_TIMEOUT
    )
    final_content = ai_message.content if isinstance(ai_message.content, str) else str(ai_message.content)
    final_answer = clean_thinking_content(final_content)
    # Emit the merged, de-duplicated citation set across all sub-answers
    # (Track X.2). Additive — consumers that ignore ``citations`` are
    # unaffected; the answer text is produced exactly as before.
    citations = merge_citations(state.get("citation_groups", []))
    # The retrieval set that informed the final answer = the union of every
    # sub-answer's hits (accumulated in ``retrieval_hits``). Both X.3 guards
    # validate against it.
    retrieval_hits = state.get("retrieval_hits", [])
    # Track X.3 faithfulness guard (PRIMARY): validate the LLM's inline
    # ``[id, p.X]`` markers in the FINAL, user-visible answer — the synthesized
    # text streamed to the client — against the retrieval set. A hallucinated
    # marker (a source/page the model never saw) is flagged/logged, not stripped
    # (``strip=False``): the answer text is kept byte-identical, same
    # least-destructive policy. Membership check only, not semantic support.
    marker_guard = guard_answer_citations(final_answer, retrieval_hits, strip=False)
    if marker_guard["dropped_count"]:
        logger.warning(
            "ask: {} hallucinated inline citation marker(s) in the final "
            "answer flagged (kept {}): {}",
            marker_guard["dropped_count"],
            marker_guard["kept_count"],
            [d["marker"] for d in marker_guard["dropped"]],
        )
    # Track X.3 array safety-net (SECONDARY): membership-check the chunk-bearing
    # citations against the same retrieval set. A no-op today (citations are
    # context-derived); regression insurance if that ever stops being true.
    guarded = guard_citation_array(citations, retrieval_hits)
    if guarded["dropped_count"]:
        logger.warning(
            "ask: {} citation(s) dropped by the array safety-net "
            "(non-context-derived chunk_id): {}",
            guarded["dropped_count"],
            guarded["dropped"],
        )
    return {
        "final_answer": final_answer,
        "citations": guarded["citations"],
    }


agent_state = StateGraph(ThreadState)
agent_state.add_node("agent", call_model_with_messages)
agent_state.add_node("provide_answer", provide_answer)
agent_state.add_node("write_final_answer", write_final_answer)
agent_state.add_edge(START, "agent")
agent_state.add_conditional_edges("agent", trigger_queries, ["provide_answer"])
agent_state.add_edge("provide_answer", "write_final_answer")
agent_state.add_edge("write_final_answer", END)

graph = agent_state.compile()
