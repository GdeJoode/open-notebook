"""Source chat router - source-specific chat session management."""

import asyncio
import json
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Path
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field

from app_main.dependencies import get_chat_service, get_source_service
from app_main.services.chat_service import ChatService
from app_main.services.source_service import SourceService

router = APIRouter(tags=["source-chat"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateSourceChatSessionRequest(BaseModel):
    source_id: str = Field(..., description="Source ID to create chat session for")
    title: Optional[str] = Field(None, description="Optional session title")
    model_override: Optional[str] = Field(
        None, description="Optional model override for this session"
    )


class UpdateSourceChatSessionRequest(BaseModel):
    title: Optional[str] = Field(None, description="New session title")
    model_override: Optional[str] = Field(
        None, description="Model override for this session"
    )


class ChatMessage(BaseModel):
    id: str = Field(..., description="Message ID")
    type: str = Field(..., description="Message type (human|ai)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class ContextIndicator(BaseModel):
    sources: List[str] = Field(
        default_factory=list, description="Source IDs used in context"
    )
    insights: List[str] = Field(
        default_factory=list, description="Insight IDs used in context"
    )
    notes: List[str] = Field(
        default_factory=list, description="Note IDs used in context"
    )


class SourceChatSessionResponse(BaseModel):
    id: str = Field(..., description="Session ID")
    title: str = Field(..., description="Session title")
    source_id: str = Field(..., description="Source ID")
    model_override: Optional[str] = Field(
        None, description="Model override for this session"
    )
    created: str = Field(..., description="Creation timestamp")
    updated: str = Field(..., description="Last update timestamp")
    message_count: Optional[int] = Field(
        None, description="Number of messages in session"
    )


class SourceChatSessionWithMessagesResponse(SourceChatSessionResponse):
    messages: List[ChatMessage] = Field(
        default_factory=list, description="Session messages"
    )
    context_indicators: Optional[ContextIndicator] = Field(
        None, description="Context indicators from last response"
    )


class SendMessageRequest(BaseModel):
    message: str = Field(..., description="User message content")
    model_override: Optional[str] = Field(
        None, description="Optional model override for this message"
    )


class SuccessResponse(BaseModel):
    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_source_id(source_id: str) -> str:
    if source_id.startswith("source:"):
        return source_id
    return f"source:{source_id}"


def _ensure_session_id(session_id: str) -> str:
    if session_id.startswith("chat_session:"):
        return session_id
    return f"chat_session:{session_id}"


def _get_source_chat_graph():
    """Lazy import of the source chat graph."""
    from app_main.graphs.source_chat import source_chat_graph

    return source_chat_graph


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post(
    "/sources/{source_id}/chat/sessions",
    response_model=SourceChatSessionResponse,
)
async def create_source_chat_session(
    request: CreateSourceChatSessionRequest,
    source_id: str = Path(..., description="Source ID"),
    source_svc: SourceService = Depends(get_source_service),
    chat_svc: ChatService = Depends(get_chat_service),
):
    """Create a new chat session for a source."""
    try:
        full_source_id = _ensure_source_id(source_id)
        source = await source_svc.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        session = await chat_svc.create_session(source_id=full_source_id)

        # Update title/model_override if provided
        if request.title or request.model_override:
            update_data = {}
            if request.title:
                update_data["title"] = request.title
            else:
                update_data["title"] = (
                    f"Source Chat {asyncio.get_event_loop().time():.0f}"
                )
            if request.model_override:
                update_data["model_override"] = request.model_override

            if session.id and update_data:
                from surrealdb_service.connection import execute_query

                set_clauses = ", ".join(
                    f"{k} = ${k}" for k in update_data.keys()
                )
                await execute_query(
                    f"UPDATE $session_id SET {set_clauses}",
                    {"session_id": session.id, **update_data},
                )

        return SourceChatSessionResponse(
            id=session.id or "",
            title=request.title
            or f"Source Chat {asyncio.get_event_loop().time():.0f}",
            source_id=source_id,
            model_override=request.model_override,
            created=str(session.created) if session.created else "",
            updated=str(session.updated) if session.updated else "",
            message_count=0,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating source chat session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error creating source chat session: {str(e)}",
        )


@router.get(
    "/sources/{source_id}/chat/sessions",
    response_model=List[SourceChatSessionResponse],
)
async def get_source_chat_sessions(
    source_id: str = Path(..., description="Source ID"),
    source_svc: SourceService = Depends(get_source_service),
):
    """Get all chat sessions for a source."""
    try:
        full_source_id = _ensure_source_id(source_id)
        source = await source_svc.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        from surrealdb_service.connection import execute_query, ensure_record_id

        relations = await execute_query(
            "SELECT in FROM refers_to WHERE out = $source_id",
            {"source_id": ensure_record_id(full_source_id)},
        )

        sessions = []
        for relation in relations:
            session_id = relation.get("in")
            if session_id:
                session_result = await execute_query(
                    f"SELECT * FROM {session_id}"
                )
                if session_result and len(session_result) > 0:
                    sd = session_result[0]
                    sessions.append(
                        SourceChatSessionResponse(
                            id=sd.get("id") or "",
                            title=sd.get("title") or "Untitled Session",
                            source_id=source_id,
                            model_override=sd.get("model_override"),
                            created=str(sd.get("created", "")),
                            updated=str(sd.get("updated", "")),
                            message_count=0,
                        )
                    )

        sessions.sort(key=lambda x: x.created, reverse=True)
        return sessions
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching source chat sessions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching source chat sessions: {str(e)}",
        )


@router.get(
    "/sources/{source_id}/chat/sessions/{session_id}",
    response_model=SourceChatSessionWithMessagesResponse,
)
async def get_source_chat_session(
    source_id: str = Path(..., description="Source ID"),
    session_id: str = Path(..., description="Session ID"),
    source_svc: SourceService = Depends(get_source_service),
    chat_svc: ChatService = Depends(get_chat_service),
):
    """Get a specific source chat session with its messages."""
    try:
        full_source_id = _ensure_source_id(source_id)
        source = await source_svc.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        full_session_id = _ensure_session_id(session_id)
        session = await chat_svc.get_session(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Verify session belongs to this source
        from surrealdb_service.connection import execute_query, ensure_record_id

        relation_query = await execute_query(
            "SELECT * FROM refers_to WHERE in = $session_id AND out = $source_id",
            {
                "session_id": ensure_record_id(full_session_id),
                "source_id": ensure_record_id(full_source_id),
            },
        )
        if not relation_query:
            raise HTTPException(
                status_code=404, detail="Session not found for this source"
            )

        # Get messages from LangGraph state
        source_chat_graph = _get_source_chat_graph()
        thread_state = source_chat_graph.get_state(
            config=RunnableConfig(configurable={"thread_id": session_id})
        )

        messages: list[ChatMessage] = []
        context_indicators = None

        if thread_state and thread_state.values:
            if "messages" in thread_state.values:
                for msg in thread_state.values["messages"]:
                    messages.append(
                        ChatMessage(
                            id=getattr(msg, "id", f"msg_{len(messages)}"),
                            type=(
                                msg.type
                                if hasattr(msg, "type")
                                else "unknown"
                            ),
                            content=(
                                msg.content
                                if hasattr(msg, "content")
                                else str(msg)
                            ),
                            timestamp=None,
                        )
                    )

            if "context_indicators" in thread_state.values:
                ci = thread_state.values["context_indicators"]
                context_indicators = ContextIndicator(
                    sources=ci.get("sources", []),
                    insights=ci.get("insights", []),
                    notes=ci.get("notes", []),
                )

        return SourceChatSessionWithMessagesResponse(
            id=session.id or "",
            title=getattr(session, "title", None) or "Untitled Session",
            source_id=source_id,
            model_override=getattr(session, "model_override", None),
            created=str(session.created) if session.created else "",
            updated=str(session.updated) if session.updated else "",
            message_count=len(messages),
            messages=messages,
            context_indicators=context_indicators,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching source chat session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching source chat session: {str(e)}",
        )


@router.put(
    "/sources/{source_id}/chat/sessions/{session_id}",
    response_model=SourceChatSessionResponse,
)
async def update_source_chat_session(
    request: UpdateSourceChatSessionRequest,
    source_id: str = Path(..., description="Source ID"),
    session_id: str = Path(..., description="Session ID"),
    source_svc: SourceService = Depends(get_source_service),
    chat_svc: ChatService = Depends(get_chat_service),
):
    """Update source chat session title and/or model override."""
    try:
        full_source_id = _ensure_source_id(source_id)
        source = await source_svc.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        full_session_id = _ensure_session_id(session_id)
        session = await chat_svc.get_session(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Verify session belongs to this source
        from surrealdb_service.connection import execute_query, ensure_record_id

        relation_query = await execute_query(
            "SELECT * FROM refers_to WHERE in = $session_id AND out = $source_id",
            {
                "session_id": ensure_record_id(full_session_id),
                "source_id": ensure_record_id(full_source_id),
            },
        )
        if not relation_query:
            raise HTTPException(
                status_code=404, detail="Session not found for this source"
            )

        update_data = request.model_dump(exclude_unset=True)
        if update_data:
            set_clauses = ", ".join(f"{k} = ${k}" for k in update_data.keys())
            await execute_query(
                f"UPDATE $session_id SET {set_clauses}",
                {
                    "session_id": ensure_record_id(full_session_id),
                    **update_data,
                },
            )

        return SourceChatSessionResponse(
            id=session.id or "",
            title=request.title
            or getattr(session, "title", None)
            or "Untitled Session",
            source_id=source_id,
            model_override=request.model_override
            or getattr(session, "model_override", None),
            created=str(session.created) if session.created else "",
            updated=str(session.updated) if session.updated else "",
            message_count=0,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating source chat session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating source chat session: {str(e)}",
        )


@router.delete(
    "/sources/{source_id}/chat/sessions/{session_id}",
    response_model=SuccessResponse,
)
async def delete_source_chat_session(
    source_id: str = Path(..., description="Source ID"),
    session_id: str = Path(..., description="Session ID"),
    source_svc: SourceService = Depends(get_source_service),
    chat_svc: ChatService = Depends(get_chat_service),
):
    """Delete a source chat session."""
    try:
        full_source_id = _ensure_source_id(source_id)
        source = await source_svc.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        full_session_id = _ensure_session_id(session_id)
        session = await chat_svc.get_session(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Verify session belongs to this source
        from surrealdb_service.connection import execute_query, ensure_record_id

        relation_query = await execute_query(
            "SELECT * FROM refers_to WHERE in = $session_id AND out = $source_id",
            {
                "session_id": ensure_record_id(full_session_id),
                "source_id": ensure_record_id(full_source_id),
            },
        )
        if not relation_query:
            raise HTTPException(
                status_code=404, detail="Session not found for this source"
            )

        await chat_svc.delete_session(full_session_id)
        return SuccessResponse(
            success=True, message="Source chat session deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting source chat session: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting source chat session: {str(e)}",
        )


async def stream_source_chat_response(
    session_id: str,
    source_id: str,
    message: str,
    model_override: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """Stream the source chat response as Server-Sent Events."""
    try:
        source_chat_graph = _get_source_chat_graph()

        # Get current state
        current_state = source_chat_graph.get_state(
            config=RunnableConfig(configurable={"thread_id": session_id})
        )

        state_values = current_state.values if current_state else {}
        state_values["messages"] = state_values.get("messages", [])
        state_values["source_id"] = source_id
        state_values["model_override"] = model_override

        # Add user message
        user_message = HumanMessage(content=message)
        state_values["messages"].append(user_message)

        # Send user message event
        user_event = {
            "type": "user_message",
            "content": message,
            "timestamp": None,
        }
        yield f"data: {json.dumps(user_event)}\n\n"

        # Execute source chat graph
        result = source_chat_graph.invoke(
            input=state_values,
            config=RunnableConfig(
                configurable={
                    "thread_id": session_id,
                    "model_id": model_override,
                }
            ),
        )

        # Stream AI messages
        if "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, "type") and msg.type == "ai":
                    ai_event = {
                        "type": "ai_message",
                        "content": (
                            msg.content
                            if hasattr(msg, "content")
                            else str(msg)
                        ),
                        "timestamp": None,
                    }
                    yield f"data: {json.dumps(ai_event)}\n\n"

        # Stream context indicators
        if "context_indicators" in result:
            context_event = {
                "type": "context_indicators",
                "data": result["context_indicators"],
            }
            yield f"data: {json.dumps(context_event)}\n\n"

        # Stream chunk-level source citations (Track X.2). Additive — clients
        # that ignore this event are unaffected.
        if result.get("citations"):
            citations_event = {
                "type": "citations",
                "data": result["citations"],
            }
            yield f"data: {json.dumps(citations_event)}\n\n"

        # Completion signal
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"

    except Exception as e:
        logger.error(f"Error in source chat streaming: {e}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"


@router.post(
    "/sources/{source_id}/chat/sessions/{session_id}/messages",
)
async def send_message_to_source_chat(
    request: SendMessageRequest,
    source_id: str = Path(..., description="Source ID"),
    session_id: str = Path(..., description="Session ID"),
    source_svc: SourceService = Depends(get_source_service),
    chat_svc: ChatService = Depends(get_chat_service),
):
    """Send a message to source chat session with SSE streaming response."""
    try:
        full_source_id = _ensure_source_id(source_id)
        source = await source_svc.get(full_source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        full_session_id = _ensure_session_id(session_id)
        session = await chat_svc.get_session(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Verify session belongs to this source
        from surrealdb_service.connection import execute_query, ensure_record_id

        relation_query = await execute_query(
            "SELECT * FROM refers_to WHERE in = $session_id AND out = $source_id",
            {
                "session_id": ensure_record_id(full_session_id),
                "source_id": ensure_record_id(full_source_id),
            },
        )
        if not relation_query:
            raise HTTPException(
                status_code=404, detail="Session not found for this source"
            )

        if not request.message:
            raise HTTPException(
                status_code=400, detail="Message content is required"
            )

        # Determine model override (request > session)
        model_override = request.model_override or getattr(
            session, "model_override", None
        )

        return StreamingResponse(
            stream_source_chat_response(
                session_id=session_id,
                source_id=full_source_id,
                message=request.message,
                model_override=model_override,
            ),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/plain; charset=utf-8",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending message to source chat: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error sending message: {str(e)}",
        )
