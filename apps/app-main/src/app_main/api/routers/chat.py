"""Chat router - chat session management with LangGraph integration."""

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from loguru import logger
from pydantic import BaseModel, Field

from app_main.dependencies import get_chat_service, get_notebook_service
from app_main.services.chat_service import ChatService
from app_main.services.notebook_service import NotebookService

router = APIRouter(prefix="/chat", tags=["chat"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    notebook_id: str = Field(..., description="Notebook ID to create session for")
    title: Optional[str] = Field(None, description="Optional session title")
    model_override: Optional[str] = Field(
        None, description="Optional model override for this session"
    )


class UpdateSessionRequest(BaseModel):
    title: Optional[str] = Field(None, description="New session title")
    model_override: Optional[str] = Field(
        None, description="Model override for this session"
    )


class ChatMessage(BaseModel):
    id: str = Field(..., description="Message ID")
    type: str = Field(..., description="Message type (human|ai)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[str] = Field(None, description="Message timestamp")


class ChatSessionResponse(BaseModel):
    id: str = Field(..., description="Session ID")
    title: str = Field(..., description="Session title")
    notebook_id: Optional[str] = Field(None, description="Notebook ID")
    created: str = Field(..., description="Creation timestamp")
    updated: str = Field(..., description="Last update timestamp")
    message_count: Optional[int] = Field(
        None, description="Number of messages in session"
    )
    model_override: Optional[str] = Field(
        None, description="Model override for this session"
    )


class ChatSessionWithMessagesResponse(ChatSessionResponse):
    messages: List[ChatMessage] = Field(
        default_factory=list, description="Session messages"
    )


class ExecuteChatRequest(BaseModel):
    session_id: str = Field(..., description="Chat session ID")
    message: str = Field(..., description="User message content")
    context: Dict[str, Any] = Field(
        ..., description="Chat context with sources and notes"
    )
    model_override: Optional[str] = Field(
        None, description="Optional model override for this message"
    )


class ExecuteChatResponse(BaseModel):
    session_id: str = Field(..., description="Session ID")
    messages: List[ChatMessage] = Field(..., description="Updated message list")


class BuildContextRequest(BaseModel):
    notebook_id: str = Field(..., description="Notebook ID")
    context_config: Dict[str, Any] = Field(..., description="Context configuration")


class BuildContextResponse(BaseModel):
    context: Dict[str, Any] = Field(..., description="Built context data")
    token_count: int = Field(..., description="Estimated token count")
    char_count: int = Field(..., description="Character count")


class SuccessResponse(BaseModel):
    success: bool = Field(True, description="Operation success status")
    message: str = Field(..., description="Success message")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_session_id(session_id: str) -> str:
    """Ensure session_id has proper table prefix."""
    if session_id.startswith("chat_session:"):
        return session_id
    return f"chat_session:{session_id}"


def _get_chat_graph():
    """Lazy import of the chat graph to avoid circular imports."""
    from app_main.graphs.chat import graph as chat_graph

    return chat_graph


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/sessions", response_model=List[ChatSessionResponse])
async def get_sessions(
    notebook_id: str = Query(..., description="Notebook ID"),
    notebook_svc: NotebookService = Depends(get_notebook_service),
    chat_svc: ChatService = Depends(get_chat_service),
):
    """Get all chat sessions for a notebook."""
    try:
        notebook = await notebook_svc.get(notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        sessions_data = await notebook_svc.get_chat_sessions(notebook_id)

        return [
            ChatSessionResponse(
                id=s.get("id", "") if isinstance(s, dict) else (s.id or ""),
                title=(s.get("title") if isinstance(s, dict) else s.title)
                or "Untitled Session",
                notebook_id=notebook_id,
                created=str(
                    s.get("created", "") if isinstance(s, dict) else s.created
                ),
                updated=str(
                    s.get("updated", "") if isinstance(s, dict) else s.updated
                ),
                message_count=0,
                model_override=(
                    s.get("model_override")
                    if isinstance(s, dict)
                    else getattr(s, "model_override", None)
                ),
            )
            for s in sessions_data
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching chat sessions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching chat sessions: {str(e)}"
        )


@router.post("/sessions", response_model=ChatSessionResponse)
async def create_session(
    request: CreateSessionRequest,
    notebook_svc: NotebookService = Depends(get_notebook_service),
    chat_svc: ChatService = Depends(get_chat_service),
):
    """Create a new chat session."""
    try:
        notebook = await notebook_svc.get(request.notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        session = await chat_svc.create_session(notebook_id=request.notebook_id)

        # Update title and model_override if provided
        if request.title or request.model_override:
            update_data = {}
            if request.title:
                update_data["title"] = request.title
            else:
                update_data["title"] = (
                    f"Chat Session {asyncio.get_event_loop().time():.0f}"
                )
            if request.model_override:
                update_data["model_override"] = request.model_override

            if session.id:
                from surrealdb_service.connection import execute_query

                set_clauses = ", ".join(
                    f"{k} = ${k}" for k in update_data.keys()
                )
                await execute_query(
                    f"UPDATE $session_id SET {set_clauses}",
                    {"session_id": session.id, **update_data},
                )

        return ChatSessionResponse(
            id=session.id or "",
            title=request.title
            or f"Chat Session {asyncio.get_event_loop().time():.0f}",
            notebook_id=request.notebook_id,
            created=str(session.created) if session.created else "",
            updated=str(session.updated) if session.updated else "",
            message_count=0,
            model_override=request.model_override,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error creating chat session: {str(e)}"
        )


@router.get(
    "/sessions/{session_id}",
    response_model=ChatSessionWithMessagesResponse,
)
async def get_session(
    session_id: str,
    chat_svc: ChatService = Depends(get_chat_service),
):
    """Get a specific session with its messages."""
    try:
        full_session_id = _ensure_session_id(session_id)
        session = await chat_svc.get_session(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Get messages from LangGraph state
        chat_graph = _get_chat_graph()
        thread_state = chat_graph.get_state(
            config=RunnableConfig(configurable={"thread_id": session_id})
        )

        messages: list[ChatMessage] = []
        if thread_state and thread_state.values and "messages" in thread_state.values:
            for msg in thread_state.values["messages"]:
                messages.append(
                    ChatMessage(
                        id=getattr(msg, "id", f"msg_{len(messages)}"),
                        type=msg.type if hasattr(msg, "type") else "unknown",
                        content=(
                            msg.content if hasattr(msg, "content") else str(msg)
                        ),
                        timestamp=None,
                    )
                )

        # Find notebook_id via relationship query
        from surrealdb_service.connection import execute_query, ensure_record_id

        notebook_query = await execute_query(
            "SELECT out FROM refers_to WHERE in = $session_id",
            {"session_id": ensure_record_id(full_session_id)},
        )
        notebook_id = notebook_query[0]["out"] if notebook_query else None

        if not notebook_id:
            logger.warning(
                f"No notebook relationship found for session {session_id}"
            )

        return ChatSessionWithMessagesResponse(
            id=session.id or "",
            title=getattr(session, "title", None) or "Untitled Session",
            notebook_id=notebook_id,
            created=str(session.created) if session.created else "",
            updated=str(session.updated) if session.updated else "",
            message_count=len(messages),
            messages=messages,
            model_override=getattr(session, "model_override", None),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching session: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching session: {str(e)}"
        )


@router.put("/sessions/{session_id}", response_model=ChatSessionResponse)
async def update_session(
    session_id: str,
    request: UpdateSessionRequest,
    chat_svc: ChatService = Depends(get_chat_service),
):
    """Update session title and/or model override."""
    try:
        full_session_id = _ensure_session_id(session_id)
        session = await chat_svc.get_session(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        update_data = request.model_dump(exclude_unset=True)

        if update_data:
            from surrealdb_service.connection import execute_query, ensure_record_id

            set_clauses = ", ".join(f"{k} = ${k}" for k in update_data.keys())
            await execute_query(
                f"UPDATE $session_id SET {set_clauses}",
                {"session_id": ensure_record_id(full_session_id), **update_data},
            )

        # Find notebook_id via relationship query
        from surrealdb_service.connection import execute_query, ensure_record_id

        notebook_query = await execute_query(
            "SELECT out FROM refers_to WHERE in = $session_id",
            {"session_id": ensure_record_id(full_session_id)},
        )
        notebook_id = notebook_query[0]["out"] if notebook_query else None

        return ChatSessionResponse(
            id=session.id or "",
            title=request.title or getattr(session, "title", "") or "",
            notebook_id=notebook_id,
            created=str(session.created) if session.created else "",
            updated=str(session.updated) if session.updated else "",
            message_count=0,
            model_override=request.model_override
            or getattr(session, "model_override", None),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error updating session: {str(e)}"
        )


@router.delete("/sessions/{session_id}", response_model=SuccessResponse)
async def delete_session(
    session_id: str,
    chat_svc: ChatService = Depends(get_chat_service),
):
    """Delete a chat session."""
    try:
        full_session_id = _ensure_session_id(session_id)
        session = await chat_svc.get_session(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        await chat_svc.delete_session(full_session_id)
        return SuccessResponse(success=True, message="Session deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error deleting session: {str(e)}"
        )


@router.post("/execute", response_model=ExecuteChatResponse)
async def execute_chat(
    request: ExecuteChatRequest,
    chat_svc: ChatService = Depends(get_chat_service),
):
    """Execute a chat request and get AI response."""
    try:
        full_session_id = _ensure_session_id(request.session_id)
        session = await chat_svc.get_session(full_session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        # Determine model override (per-request takes precedence)
        model_override = request.model_override or getattr(
            session, "model_override", None
        )

        chat_graph = _get_chat_graph()

        # Get current state
        current_state = chat_graph.get_state(
            config=RunnableConfig(
                configurable={"thread_id": request.session_id}
            )
        )

        state_values = current_state.values if current_state else {}
        state_values["messages"] = state_values.get("messages", [])
        state_values["context"] = request.context
        state_values["model_override"] = model_override

        # Add user message
        user_message = HumanMessage(content=request.message)
        state_values["messages"].append(user_message)

        # Execute chat graph
        result = chat_graph.invoke(
            input=state_values,
            config=RunnableConfig(
                configurable={
                    "thread_id": request.session_id,
                    "model_id": model_override,
                }
            ),
        )

        # Convert messages to response format
        messages: list[ChatMessage] = []
        for msg in result.get("messages", []):
            messages.append(
                ChatMessage(
                    id=getattr(msg, "id", f"msg_{len(messages)}"),
                    type=msg.type if hasattr(msg, "type") else "unknown",
                    content=(
                        msg.content if hasattr(msg, "content") else str(msg)
                    ),
                    timestamp=None,
                )
            )

        return ExecuteChatResponse(
            session_id=request.session_id, messages=messages
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing chat: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error executing chat: {str(e)}"
        )


@router.post("/context", response_model=BuildContextResponse)
async def build_context(
    request: BuildContextRequest,
    notebook_svc: NotebookService = Depends(get_notebook_service),
):
    """Build context for a notebook based on context configuration."""
    try:
        notebook = await notebook_svc.get(request.notebook_id)
        if not notebook:
            raise HTTPException(status_code=404, detail="Notebook not found")

        from app_main.dependencies import get_source_service, get_note_service

        source_svc = get_source_service()
        note_svc = get_note_service()

        context_data: dict[str, list[dict]] = {"sources": [], "notes": []}
        total_content = ""

        if request.context_config:
            # Process sources
            for source_id, status in request.context_config.get(
                "sources", {}
            ).items():
                if "not in" in status:
                    continue
                try:
                    full_source_id = (
                        source_id
                        if source_id.startswith("source:")
                        else f"source:{source_id}"
                    )
                    source = await source_svc.get(full_source_id)
                    if not source:
                        continue

                    if "insights" in status:
                        insights = await source_svc.get_insights(full_source_id)
                        source_context = {
                            "id": source.id,
                            "title": source.title,
                            "insights": [
                                {"type": i.insight_type, "content": i.content}
                                for i in insights
                            ],
                        }
                    elif "full content" in status:
                        source_context = {
                            "id": source.id,
                            "title": source.title,
                            "full_text": source.full_text,
                        }
                    else:
                        continue

                    context_data["sources"].append(source_context)
                    total_content += str(source_context)
                except Exception as e:
                    logger.warning(f"Error processing source {source_id}: {e}")
                    continue

            # Process notes
            for note_id, status in request.context_config.get(
                "notes", {}
            ).items():
                if "not in" in status:
                    continue
                try:
                    full_note_id = (
                        note_id
                        if note_id.startswith("note:")
                        else f"note:{note_id}"
                    )
                    note = await note_svc.get(full_note_id)
                    if not note:
                        continue

                    if "full content" in status:
                        note_context = {
                            "id": note.id,
                            "title": note.title,
                            "content": note.content,
                        }
                        context_data["notes"].append(note_context)
                        total_content += str(note_context)
                except Exception as e:
                    logger.warning(f"Error processing note {note_id}: {e}")
                    continue
        else:
            # Default: include all with short context
            sources_data = await notebook_svc.get_sources(request.notebook_id)
            for sd in sources_data:
                try:
                    sid = (
                        sd.get("id") if isinstance(sd, dict) else sd.id
                    )
                    source = await source_svc.get(sid)
                    if not source:
                        continue
                    insights = await source_svc.get_insights(sid)
                    source_context = {
                        "id": sid,
                        "title": source.title,
                        "insights": [
                            {"type": i.insight_type, "content": i.content}
                            for i in insights
                        ],
                    }
                    context_data["sources"].append(source_context)
                    total_content += str(source_context)
                except Exception as e:
                    logger.warning(f"Error processing source: {e}")
                    continue

            notes_data = await notebook_svc.get_notes(request.notebook_id)
            for nd in notes_data:
                try:
                    note_context = {
                        "id": nd.get("id") if isinstance(nd, dict) else nd.id,
                        "title": nd.get("title")
                        if isinstance(nd, dict)
                        else nd.title,
                        "content": nd.get("content")
                        if isinstance(nd, dict)
                        else nd.content,
                    }
                    context_data["notes"].append(note_context)
                    total_content += str(note_context)
                except Exception as e:
                    logger.warning(f"Error processing note: {e}")
                    continue

        char_count = len(total_content)
        from shared.utils import token_count

        estimated_tokens = token_count(total_content) if total_content else 0

        return BuildContextResponse(
            context=context_data,
            token_count=estimated_tokens,
            char_count=char_count,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building context: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error building context: {str(e)}"
        )
