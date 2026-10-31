"""Search router - text/vector search and ask endpoints."""

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger
from shared.models.classification import ClassificationScope

from app_main.api.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse
from app_main.dependencies import get_model_service, get_search_service
from app_main.exceptions import InvalidInputError
from app_main.services.model_service import ModelService
from app_main.services.search_service import SearchService
from app_main.api.agent_auth import RECHT_BEHEERDER, RECHT_LEZEN_EXTERN, current_scope
from app_main.api.fouten import fout_zonder_inhoud, log_fout, melding_met_id

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse, dependencies=[RECHT_LEZEN_EXTERN])
async def search_knowledge_base(
    search_request: SearchRequest,
    search_svc: SearchService = Depends(get_search_service),
    model_svc: ModelService = Depends(get_model_service),
    scope: ClassificationScope = Depends(current_scope),
):
    """Search the source chunks using text, vector, or hybrid search.

    Zoekt alleen in fragmenten, nooit in notities of inzichten (ADR 0004), met
    het classificatiebereik van de sleutel in de query (ADR 0002 C) en
    standaard alleen de huidige bronversie (``history`` voor oudere).
    """
    try:
        if search_request.type == "vector":
            # Verify embedding model is available
            defaults = model_svc.model_manager.get_defaults()
            if not defaults or not defaults.default_embedding_model:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Vector search requires an embedding model. "
                        "Please configure one in the Models section."
                    ),
                )
            results = await search_svc.vector_search(
                keyword=search_request.query,
                results=search_request.limit,
                minimum_score=search_request.minimum_score,
                scope=scope,
                history=search_request.history,
            )
        elif search_request.type == "hybrid":
            # Hybrid (BM25 + vector) fusion. Unlike pure vector search this
            # degrades gracefully: with no embedding model the retrieval layer
            # falls back to text-only search rather than erroring, so the
            # endpoint stays usable in unconfigured environments.
            results = await search_svc.hybrid_search(
                keyword=search_request.query,
                results=search_request.limit,
                minimum_score=search_request.minimum_score,
                text_weight=search_request.text_weight,
                scope=scope,
                history=search_request.history,
            )

            # Optional cross-encoder rerank of the top-N fused results. Default
            # (rerank=false) leaves the fused output byte-identical. On a
            # reranker-service outage this falls back to the heuristic reranker
            # internally — it never raises, so search stays up.
            if search_request.rerank and results:
                from app_main.services.rerank_orchestrator import (
                    rerank_hybrid_results,
                )

                from surrealdb_service.repositories.source import ChunkRepository

                # De tekst van de getroffen fragmenten, binnen het bereik.
                fragment_ids = [
                    str(c)
                    for r in results[: search_request.rerank_top_n]
                    for c in r.get("chunks") or []
                ]
                chunk_texts = await ChunkRepository().get_texts(fragment_ids, scope=scope)
                results = await rerank_hybrid_results(
                    query=search_request.query,
                    results=results,
                    top_n=search_request.rerank_top_n,
                    chunk_texts=chunk_texts,
                )
        else:
            results = await search_svc.text_search(
                keyword=search_request.query,
                results=search_request.limit,
                scope=scope,
                history=search_request.history,
            )

        return SearchResponse(
            results=results or [],
            total_count=len(results) if results else 0,
            search_type=search_request.type,
        )

    except InvalidInputError as e:
        raise fout_zonder_inhoud(e, "Ongeldige zoekvraag", status_code=400)
    except Exception as e:
        raise fout_zonder_inhoud(e, "Zoeken mislukt")


async def stream_ask_response(
    question: str,
    strategy_model_id: str,
    answer_model_id: str,
    final_answer_model_id: str,
) -> AsyncGenerator[str, None]:
    """Stream the ask response as Server-Sent Events."""
    try:
        from app_main.graphs.ask import graph as ask_graph

        final_answer = None

        async for chunk in ask_graph.astream(
            input=dict(question=question),
            config=dict(
                configurable=dict(
                    strategy_model=strategy_model_id,
                    answer_model=answer_model_id,
                    final_answer_model=final_answer_model_id,
                )
            ),
            stream_mode="updates",
        ):
            if "agent" in chunk:
                strategy_data = {
                    "type": "strategy",
                    "reasoning": chunk["agent"]["strategy"].reasoning,
                    "searches": [
                        {"term": s.term, "instructions": s.instructions}
                        for s in chunk["agent"]["strategy"].searches
                    ],
                }
                yield f"data: {json.dumps(strategy_data)}\n\n"

            elif "provide_answer" in chunk:
                for answer in chunk["provide_answer"]["answers"]:
                    answer_data = {"type": "answer", "content": answer}
                    yield f"data: {json.dumps(answer_data)}\n\n"

            elif "write_final_answer" in chunk:
                final_answer = chunk["write_final_answer"]["final_answer"]
                citations = chunk["write_final_answer"].get("citations", [])
                final_data = {
                    "type": "final_answer",
                    "content": final_answer,
                    "citations": citations,
                }
                yield f"data: {json.dumps(final_data)}\n\n"

        completion_data = {"type": "complete", "final_answer": final_answer}
        yield f"data: {json.dumps(completion_data)}\n\n"

    except Exception as e:
        logger.error(f"Error in ask streaming: {e}")
        fout_id = log_fout(e, "Vraag beantwoorden mislukt")
        error_data = {"type": "error", "message": melding_met_id("Vraag beantwoorden mislukt", fout_id)}
        yield f"data: {json.dumps(error_data)}\n\n"


@router.post("/search/ask", dependencies=[RECHT_BEHEERDER])
async def ask_knowledge_base(
    ask_request: AskRequest,
    model_svc: ModelService = Depends(get_model_service),
):
    """Ask the knowledge base a question using AI models (streaming)."""
    try:
        strategy_model = await model_svc.get(ask_request.strategy_model)
        answer_model = await model_svc.get(ask_request.answer_model)
        final_answer_model = await model_svc.get(ask_request.final_answer_model)

        if not strategy_model:
            raise HTTPException(
                status_code=400,
                detail=f"Strategy model {ask_request.strategy_model} not found",
            )
        if not answer_model:
            raise HTTPException(
                status_code=400,
                detail=f"Answer model {ask_request.answer_model} not found",
            )
        if not final_answer_model:
            raise HTTPException(
                status_code=400,
                detail=f"Final answer model {ask_request.final_answer_model} not found",
            )

        defaults = model_svc.model_manager.get_defaults()
        if not defaults or not defaults.default_embedding_model:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Ask feature requires an embedding model. "
                    "Please configure one in the Models section."
                ),
            )

        return StreamingResponse(
            stream_ask_response(
                ask_request.question,
                strategy_model.id,
                answer_model.id,
                final_answer_model.id,
            ),
            media_type="text/plain",
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask endpoint: {e}")
        raise fout_zonder_inhoud(e, "Ask operation failed")


@router.post("/search/ask/simple", response_model=AskResponse, dependencies=[RECHT_BEHEERDER])
async def ask_knowledge_base_simple(
    ask_request: AskRequest,
    model_svc: ModelService = Depends(get_model_service),
):
    """Ask the knowledge base a question (non-streaming)."""
    try:
        from app_main.graphs.ask import graph as ask_graph

        strategy_model = await model_svc.get(ask_request.strategy_model)
        answer_model = await model_svc.get(ask_request.answer_model)
        final_answer_model = await model_svc.get(ask_request.final_answer_model)

        if not strategy_model:
            raise HTTPException(
                status_code=400, detail="Strategy model not found"
            )
        if not answer_model:
            raise HTTPException(
                status_code=400, detail="Answer model not found"
            )
        if not final_answer_model:
            raise HTTPException(
                status_code=400, detail="Final answer model not found"
            )

        defaults = model_svc.model_manager.get_defaults()
        if not defaults or not defaults.default_embedding_model:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Ask feature requires an embedding model. "
                    "Please configure one in the Models section."
                ),
            )

        final_answer = None
        citations: list = []
        async for chunk in ask_graph.astream(
            input=dict(question=ask_request.question),
            config=dict(
                configurable=dict(
                    strategy_model=strategy_model.id,
                    answer_model=answer_model.id,
                    final_answer_model=final_answer_model.id,
                )
            ),
            stream_mode="updates",
        ):
            if "write_final_answer" in chunk:
                final_answer = chunk["write_final_answer"]["final_answer"]
                citations = chunk["write_final_answer"].get("citations", [])

        if not final_answer:
            raise HTTPException(status_code=500, detail="No answer generated")

        return AskResponse(
            answer=final_answer,
            question=ask_request.question,
            citations=citations,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask simple endpoint: {e}")
        raise fout_zonder_inhoud(e, "Ask operation failed")
