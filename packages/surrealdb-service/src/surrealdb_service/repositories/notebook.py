"""
Notebook repository with specialized operations.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import ChatMessage, ChatSession, Note, Notebook
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.base import BaseRepository, _validate_record_id


class NotebookRepository(BaseRepository[Notebook]):
    """Repository for Notebook operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Notebook, config)

    async def get_sources(self, notebook_id: str) -> List[Dict[str, Any]]:
        """
        Get all sources for a notebook.

        Args:
            notebook_id: Notebook ID.

        Returns:
            List of source data (without full_text for efficiency).
        """
        try:
            result = await execute_query(
                """
                SELECT * OMIT source.full_text FROM (
                    SELECT in AS source FROM reference WHERE out=$id
                    FETCH source
                ) ORDER BY source.updated DESC
                """,
                {"id": ensure_record_id(notebook_id)},
                self.config,
            )
            return [item.get("source", {}) for item in result] if result else []
        except Exception as e:
            logger.error(f"Failed to get sources for notebook {notebook_id}: {e}")
            return []

    async def get_notes(self, notebook_id: str) -> List[Dict[str, Any]]:
        """
        Get all notes for a notebook.

        Args:
            notebook_id: Notebook ID.

        Returns:
            List of note data (without content/embedding for efficiency).
        """
        try:
            result = await execute_query(
                """
                SELECT * OMIT note.content, note.embedding FROM (
                    SELECT in AS note FROM artifact WHERE out=$id
                    FETCH note
                ) ORDER BY note.updated DESC
                """,
                {"id": ensure_record_id(notebook_id)},
                self.config,
            )
            return [item.get("note", {}) for item in result] if result else []
        except Exception as e:
            logger.error(f"Failed to get notes for notebook {notebook_id}: {e}")
            return []

    async def get_chat_sessions(self, notebook_id: str) -> List[Dict[str, Any]]:
        """
        Get all chat sessions for a notebook.

        Args:
            notebook_id: Notebook ID.

        Returns:
            List of chat session data.
        """
        try:
            result = await execute_query(
                """
                SELECT * FROM (
                    SELECT <- chat_session AS chat_session
                    FROM refers_to
                    WHERE out=$id
                    FETCH chat_session
                ) ORDER BY chat_session.updated DESC
                """,
                {"id": ensure_record_id(notebook_id)},
                self.config,
            )
            return [
                item.get("chat_session", [{}])[0]
                for item in result
                if item.get("chat_session")
            ] if result else []
        except Exception as e:
            logger.error(f"Failed to get chat sessions for notebook {notebook_id}: {e}")
            return []


class NoteRepository(BaseRepository[Note]):
    """Repository for Note operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Note, config)

    async def add_to_notebook(self, note_id: str, notebook_id: str) -> bool:
        """
        Add a note to a notebook.

        Args:
            note_id: Note ID.
            notebook_id: Notebook ID.

        Returns:
            True if successful.
        """
        try:
            await self.relate(note_id, "artifact", notebook_id)
            return True
        except Exception as e:
            logger.error(f"Failed to add note to notebook: {e}")
            return False

    async def create_with_embedding(
        self,
        data: Dict[str, Any],
        embedding: Optional[List[float]] = None,
    ) -> Note:
        """
        Create a note with optional embedding.

        Args:
            data: Note data.
            embedding: Optional pre-computed embedding.

        Returns:
            Created note.
        """
        if embedding:
            data["embedding"] = embedding
        return await self.create(data)

    async def find_related_by_embedding(
        self, note_id: str, k: int
    ) -> List[Dict[str, Any]]:
        """Return the top-``k`` other notes by cosine similarity (Track Y.1).

        Note-level mirror of ``SourceRepository.find_related_by_embedding``:
        ranks every *other* note that has a populated ``note.embedding`` against
        this note's embedding using SurrealDB's native
        ``vector::similarity::cosine`` — the same operator the chunk/source
        search paths use. Ranking server-side keeps every vector in the DB (no
        bulk pull into Python) and reuses the proven cosine path.

        Behaviour:
          * The query note's own row is excluded (``id != $id``).
          * Notes with an empty ``embedding`` (``array<float>`` is strict and
            non-optional, so an unembedded note holds ``[]`` not NONE — see
            [[note-embedding-non-optional]]) are excluded: an empty vector can
            never be a result and the dimension guard would drop it anyway.
          * If the *query* note itself has no/empty embedding, returns ``[]``
            (nothing to compare). The caller distinguishes this from "note not
            found" via a prior existence check; Track Y.2 treats it as the
            needs-embedding signal (embed first, then re-rank).
          * Ordering is ``score DESC`` with a stable ``id ASC`` tie-break, so
            equal-similarity notes come back deterministically.
          * ``k`` bounds the LIMIT; requesting more than exist returns all.

        The embedding dimension is never hardcoded — cosine reads whatever
        length the stored vectors are. The
        ``array::len(embedding) = array::len($q)`` predicate guards the cosine
        call (SurrealDB errors on a length mismatch) and, as a side effect,
        excludes the empty-embedding notes (``array::len([]) = 0`` never equals
        a populated query vector's length).

        Returns a list of ``{"id", "title", "score"}`` dicts (ids stringified
        by ``execute_query``), or ``[]`` on error / no query embedding.
        """
        try:
            rid = ensure_record_id(note_id)
            query_vec = await execute_query(
                "SELECT VALUE embedding FROM $id",
                {"id": rid},
                self.config,
            )
            # ``SELECT VALUE embedding`` yields [vector] for a populated field,
            # [[]] for the strict-but-empty unembedded note, and [] when the
            # note is absent. ``not query_vec[0]`` covers both the empty-vector
            # and absent cases — nothing to compare against.
            if not query_vec or not query_vec[0]:
                return []

            rows = await execute_query(
                "SELECT id, title, "
                "vector::similarity::cosine(embedding, $q) AS score "
                "FROM note "
                "WHERE array::len(embedding) > 0 AND id != $id "
                "AND array::len(embedding) = array::len($q) "
                "ORDER BY score DESC, id ASC "
                "LIMIT $k",
                {"q": query_vec[0], "id": rid, "k": int(k)},
                self.config,
            )
            return [
                {
                    "id": str(r["id"]),
                    "title": r.get("title"),
                    "score": float(r["score"]),
                }
                for r in (rows or [])
                if r.get("score") is not None
            ]
        except Exception as e:
            logger.error(f"Failed to find related notes for {note_id}: {e}")
            return []

    async def relate_note(
        self,
        from_note: str,
        to_note: str,
        *,
        similarity_score: float,
        method: str = "embedding",
    ) -> bool:
        """Idempotently RELATE one note to another via ``related_note`` (Y.1).

        RELATE is *not* idempotent: a repeated ``RELATE a -> related_note -> b``
        writes a SECOND row (Track W.2/W.3 lesson), so notes would accrue
        duplicate edges on every re-link. This helper clears any existing
        ``(from_note, to_note)`` edge first, then RELATEs exactly once — so the
        edge set for a pair is always a single, current row carrying the latest
        ``similarity_score``.

        Guards (all BEFORE any interpolation):
          * both ids are validated against the strict ``_RECORD_ID_RE`` pattern
            (the same one ``BaseRepository.relate`` uses) — a ``;``-bearing /
            ``REMOVE TABLE``-bearing / otherwise malformed id is REFUSED, not
            interpolated. ``RecordID.parse`` splits only on the first colon and
            round-trips an injection payload verbatim, so the SDK's own parsing
            is NOT a validator; the regex is. The endpoints are then interpolated
            as literal record ids (SurrealDB's RELATE graph syntax does not bind a
            ``$param`` in the in/out position — a parameterized
            ``RELATE $from->...->$to`` silently writes nothing), which is only
            safe because every accepted id matches ``table:id`` with no SurrealQL
            metacharacters;
          * both ids must be in the ``note`` table (a wrong-table id is rejected
            rather than silently mis-linked);
          * a self-edge (``from_note == to_note``) is refused — a note is not
            "related" to itself, and the cosine ranking already excludes self.

        Args:
            from_note: source note record id.
            to_note: target note record id.
            similarity_score: the cosine that drove the link (stored on the edge).
            method: how the link was derived (default ``"embedding"``).

        Returns:
            True on success; False if refused (invalid/unsafe id, wrong table,
            self-edge) or on a DB error.
        """
        # Strict-validate the RAW input strings BEFORE the SDK touches them.
        # ``RecordID.parse`` would happily accept ``note:x; REMOVE TABLE note --``
        # (it splits on the first colon and keeps the rest as the id, round-tripped
        # verbatim by ``str()``), so validation must run on the raw string against
        # ``_RECORD_ID_RE`` — which rejects every SurrealQL metacharacter.
        try:
            from_str = _validate_record_id(str(from_note))
            to_str = _validate_record_id(str(to_note))
        except ValueError as e:
            logger.error(
                f"relate_note: refusing unsafe/invalid note id "
                f"({from_note!r}/{to_note!r}): {e}"
            )
            return False

        if not (from_str.startswith("note:") and to_str.startswith("note:")):
            logger.error(
                f"relate_note: both ids must be notes (got {from_str} -> {to_str})"
            )
            return False

        if from_str == to_str:
            logger.warning(f"relate_note: refusing self-edge for {from_str}")
            return False

        from_rid = ensure_record_id(from_str)
        to_rid = ensure_record_id(to_str)

        try:
            # Clear-before-relate: drop any prior edge for this exact (in, out)
            # pair so the re-link replaces rather than duplicates (RELATE is not
            # idempotent). Scoped to the directed pair — other edges untouched.
            await execute_query(
                "DELETE related_note WHERE in = $from AND out = $to",
                {"from": from_rid, "to": to_rid},
                self.config,
            )
            # Endpoints interpolated as literal record ids (RELATE graph syntax
            # won't bind a ``$param`` in the in/out position — see the docstring).
            # Safe because ``from_str``/``to_str`` passed ``_RECORD_ID_RE`` above:
            # ``table:id`` only, no SurrealQL metacharacters. Only the SET *values*
            # stay parameterized.
            await execute_query(
                f"RELATE {from_str}->related_note->{to_str} "
                "SET similarity_score = $score, method = $method",
                {
                    "score": float(similarity_score),
                    "method": method,
                },
                self.config,
            )
            return True
        except Exception as e:
            logger.error(
                f"relate_note: failed to relate {from_str} -> {to_str}: {e}"
            )
            return False


class ChatSessionRepository(BaseRepository[ChatSession]):
    """Repository for ChatSession operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(ChatSession, config)

    async def relate_to_notebook(self, session_id: str, notebook_id: str) -> bool:
        """
        Relate a chat session to a notebook.

        Args:
            session_id: Chat session ID.
            notebook_id: Notebook ID.

        Returns:
            True if successful.
        """
        try:
            await self.relate(session_id, "refers_to", notebook_id)
            return True
        except Exception as e:
            logger.error(f"Failed to relate session to notebook: {e}")
            return False

    async def relate_to_source(self, session_id: str, source_id: str) -> bool:
        """
        Relate a chat session to a source.

        Args:
            session_id: Chat session ID.
            source_id: Source ID.

        Returns:
            True if successful.
        """
        try:
            await self.relate(session_id, "refers_to", source_id)
            return True
        except Exception as e:
            logger.error(f"Failed to relate session to source: {e}")
            return False


class ChatMessageRepository(BaseRepository[ChatMessage]):
    """Repository for ChatMessage operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(ChatMessage, config)

    async def get_session_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[ChatMessage]:
        """
        Get all messages for a chat session.

        Args:
            session_id: Chat session ID.
            limit: Optional message limit.

        Returns:
            List of messages ordered by creation time.
        """
        return await self.query(
            "session=$session_id",
            {"session_id": ensure_record_id(session_id)},
            order_by="created ASC",
            limit=limit,
        )
