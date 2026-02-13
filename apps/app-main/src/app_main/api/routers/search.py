"""Search router - text/vector search and ask endpoints."""

import json
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from app_main.api.schemas import AskRequest, AskResponse, SearchRequest, SearchResponse
from app_main.dependencies import get_model_service, get_search_service
from app_main.exceptions import DatabaseOperationError, InvalidInputError
from app_main.services.model_service import ModelService
from app_main.services.search_service import SearchService

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search_knowledge_base(
    search_request: SearchRequest,
    search_svc: SearchService = Depends(get_search_service),
    model_svc: ModelService = Depends(get_model_service),
):
    """Search the knowledge base using text or vector search."""
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
                include_sources=search_request.search_sources,
                include_notes=search_request.search_notes,
                minimum_score=search_request.minimum_score,
            )
        else:
            results = await search_svc.text_search(
                keyword=search_request.query,
                results=search_request.limit,
                include_sources=search_request.search_sources,
                include_notes=search_request.search_notes,
            )

        return SearchResponse(
            results=results or [],
            total_count=len(results) if results else 0,
            search_type=search_request.type,
        )

    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except DatabaseOperationError as e:
        logger.error(f"Database error during search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error during search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


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
                final_data = {"type": "final_answer", "content": final_answer}
                yield f"data: {json.dumps(final_data)}\n\n"

        completion_data = {"type": "complete", "final_answer": final_answer}
        yield f"data: {json.dumps(completion_data)}\n\n"

    except Exception as e:
        logger.error(f"Error in ask streaming: {e}")
        error_data = {"type": "error", "message": str(e)}
        yield f"data: {json.dumps(error_data)}\n\n"


@router.post("/search/ask")
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
        raise HTTPException(
            status_code=500, detail=f"Ask operation failed: {str(e)}"
        )


@router.post("/search/ask/simple", response_model=AskResponse)
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

        if not final_answer:
            raise HTTPException(status_code=500, detail="No answer generated")

        return AskResponse(answer=final_answer, question=ask_request.question)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in ask simple endpoint: {e}")
        raise HTTPException(
            status_code=500, detail=f"Ask operation failed: {str(e)}"
        )
