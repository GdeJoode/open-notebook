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

    async def get_auto_insights(self, notebook_id: str) -> bool:
        """Read the per-notebook auto-INSIGHTS toggle (Track PL.3).

        Returns the notebook's ``auto_insights`` flag — the gate the PL.3 embed
        auto-chain consults before enqueuing the LLM-cost INSIGHTS stage. Default
        ON: a missing/legacy row (no toggle field), an unknown notebook, or any
        read error returns ``True`` so the conservative behaviour is "run
        insights" (matches the migration-72 default and the track's full-auto
        choice). The caller treats ``False`` as an explicit opt-out.

        Args:
            notebook_id: The ``notebook:...`` id to read the toggle from.

        Returns:
            The notebook's ``auto_insights`` value, or ``True`` when it cannot be
            resolved.
        """
        try:
            rows = await execute_query(
                "SELECT VALUE auto_insights FROM $id;",
                {"id": ensure_record_id(notebook_id)},
                self.config,
            )
            if not rows or rows[0] is None:
                # Legacy row predating the field, or unknown notebook -> default ON.
                return True
            return bool(rows[0])
        except Exception as e:
            logger.error(
                f"Failed to read auto_insights for notebook {notebook_id} "
                f"(defaulting ON): {e}"
            )
            return True

    async def get_librarian_enabled(self, notebook_id: str) -> bool:
        """Read the per-notebook librarian opt-in flag (Track F.5).

        Default OFF: a missing/legacy row, an unknown notebook, or a read error
        returns ``False`` so the periodic audit never runs for a notebook that
        never opted in.
        """
        try:
            rows = await execute_query(
                "SELECT VALUE librarian_enabled FROM $id;",
                {"id": ensure_record_id(notebook_id)},
                self.config,
            )
            if not rows or rows[0] is None:
                return False
            return bool(rows[0])
        except Exception as e:
            logger.error(
                f"Failed to read librarian_enabled for {notebook_id} "
                f"(defaulting OFF): {e}"
            )
            return False

    async def list_librarian_notebook_ids(self) -> List[str]:
        """Stringified ids of every non-archived notebook opted into the
        librarian audit (Track F.5). Empty on none / read error."""
        try:
            rows = await execute_query(
                "SELECT VALUE id FROM notebook "
                "WHERE librarian_enabled = true AND archived != true;",
                None,
                self.config,
            )
            return [str(r) for r in (rows or []) if r is not None]
        except Exception as e:
            logger.error(f"Failed to list librarian-enabled notebooks: {e}")
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
          * Notes with an empty/absent ``embedding`` are excluded by the shared
            ``embedding != NONE AND array::len(embedding) > 0`` predicate (the
            same form the note→source path uses — NS.2 alignment). ``note.embedding``
            is strict ``array<float>`` so an unembedded note holds ``[]`` not NONE
            ([[note-embedding-non-optional]]); the ``> 0`` clause drops it, and the
            ``!= NONE`` clause guards the cross-type case where ``array::len`` would
            otherwise raise on a NONE field.
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
                "WHERE embedding != NONE AND array::len(embedding) > 0 "
                "AND id != $id "
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

    async def find_related_sources_by_embedding(
        self, note_id: str, k: int
    ) -> List[Dict[str, Any]]:
        """Return the top-``k`` sources by cosine similarity to a note (NS.1).

        Cross-type analogue of ``find_related_by_embedding``: ranks every
        ``source`` whose aggregate ``source.embedding`` is populated against
        this *note's* ``note.embedding`` using SurrealDB's native
        ``vector::similarity::cosine`` — the link "this note is about these
        sources". Ranking server-side keeps every vector in the DB (no bulk
        pull into Python) and reuses the proven cosine path.

        Cross-type validity (NS.1 verification): the comparison is only
        meaningful because both vectors live in the SAME embedding space.
        ``note.embedding`` is the note's content embedded by
        ``EmbeddingService._embed_with_retry`` (``embedding_model.aembed``);
        ``source.embedding`` is the mean-pool (R.0) of that same model's chunk
        vectors (each 1024-dim mxbai-embed-large — probed on staging). Same
        model, same dimension → cosine(note, source mean-pool) is the standard
        query↔document retrieval signal. The ``array::len`` guard below is the
        hard safety net: a dimension mismatch is excluded, never crashed.

        Behaviour (mirrors the note→note path):
          * Sources with an empty/absent ``embedding`` (not yet aggregated / no
            chunk vectors) are excluded by the shared ``embedding != NONE AND
            array::len(embedding) > 0`` predicate — the same form the note→note
            path uses (NS.2 alignment). A source's ``embedding`` is genuinely
            NONE when unaggregated (unlike a note's strict ``[]``), so the
            ``!= NONE`` clause is load-bearing here: ``array::len(NONE)`` raises
            in SurrealDB, so it must be short-circuited before the length check.
          * If the *query* note has no/empty embedding (``note.embedding`` is
            strict ``array<float>``, so an unembedded note holds ``[]`` not
            NONE — [[note-embedding-non-optional]]), returns ``[]`` (nothing to
            compare). Track NS.2 treats this as the needs-embedding signal
            (embed first, then re-rank) — no crash.
          * Ordering is ``score DESC`` with a stable ``id ASC`` tie-break, so
            equal-similarity sources come back deterministically.
          * ``k`` bounds the LIMIT; requesting more than exist returns all.

        The embedding dimension is never hardcoded — cosine reads whatever
        length the stored vectors are. The
        ``array::len(embedding) = array::len($q)`` predicate guards the cosine
        call (SurrealDB errors on a length mismatch) and, as a side effect,
        keeps the ranking robust to any legacy / mixed-dim source rows instead
        of letting one bad vector fail the whole query.

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
                "FROM source "
                "WHERE embedding != NONE AND array::len(embedding) > 0 "
                "AND array::len(embedding) = array::len($q) "
                "ORDER BY score DESC, id ASC "
                "LIMIT $k",
                {"q": query_vec[0], "k": int(k)},
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
            logger.error(
                f"Failed to find related sources for note {note_id}: {e}"
            )
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

    async def relate_note_source(
        self,
        note_id: str,
        source_id: str,
        *,
        similarity_score: float,
        method: str = "embedding",
    ) -> bool:
        """Idempotently RELATE a ``note -> note_about -> source`` edge (NS.1).

        Persists "this note is about this source" as a real graph edge so the
        auto-link feature (Track NS) can link notes to the SOURCES they resemble
        by embedding similarity, alongside the existing note→note
        ``related_note`` links. The precision decision (which similarities are
        high enough to persist) lives in the orchestrator (NS.2), not here.

        RELATE is *not* idempotent: a repeated ``RELATE n -> note_about -> s``
        writes a SECOND row (the Track W/Y/Z lesson), so a re-link would accrue
        duplicate edges. This helper clears any existing ``(note, source)`` edge
        first, then RELATEs exactly once — so the edge set for a directed pair
        is always a single, current row carrying the latest ``similarity_score``.

        Guards (all BEFORE any interpolation — the Y.1/Z.1 injection lesson,
        [[surrealdb-relate-id-injection]]):
          * both ids are validated against the strict ``_RECORD_ID_RE`` pattern
            via :func:`_validate_record_id`. A ``;``-bearing / ``REMOVE TABLE``-
            bearing / otherwise malformed id is REFUSED, not interpolated.
            ``RecordID.parse`` splits only on the first colon and round-trips an
            injection payload verbatim, so the SDK's own parsing is NOT a
            validator; the regex is. The endpoints are then interpolated as
            literal record ids because SurrealDB's RELATE graph syntax does not
            bind a ``$param`` in the in/out position (a parameterized
            ``RELATE $from->...->$to`` silently writes 0 rows), which is only
            safe because every accepted id matches ``table:id`` with no SurrealQL
            metacharacters. Only the SET *values* stay parameterized;
          * the from-id must be in the ``note`` table and the to-id in the
            ``source`` table — a wrong-table id (or a swapped note/source) is
            rejected rather than silently mis-linked. A self-edge check is N/A
            here (the endpoints are different types and can never be equal).

        Args:
            note_id: the note record id (edge ``in``).
            source_id: the source record id (edge ``out``).
            similarity_score: the cosine that drove the link (stored on the edge).
            method: how the link was derived (default ``"embedding"``).

        Returns:
            True on success; False if refused (invalid/unsafe id, wrong table)
            or on a DB error.
        """
        # Strict-validate the RAW input strings BEFORE the SDK touches them.
        # ``RecordID.parse`` would happily accept ``note:x; REMOVE TABLE note --``
        # (it splits on the first colon and keeps the rest as the id, round-tripped
        # verbatim by ``str()``), so validation must run on the raw string against
        # ``_RECORD_ID_RE`` — which rejects every SurrealQL metacharacter.
        try:
            note_str = _validate_record_id(str(note_id))
            source_str = _validate_record_id(str(source_id))
        except ValueError as e:
            logger.error(
                f"relate_note_source: refusing unsafe/invalid id "
                f"({note_id!r}/{source_id!r}): {e}"
            )
            return False

        if not note_str.startswith("note:"):
            logger.error(
                f"relate_note_source: 'in' must be a note (got {note_str})"
            )
            return False
        if not source_str.startswith("source:"):
            logger.error(
                f"relate_note_source: 'out' must be a source (got {source_str})"
            )
            return False

        note_rid = ensure_record_id(note_str)
        source_rid = ensure_record_id(source_str)

        try:
            # Clear-before-relate: drop any prior edge for this exact (in, out)
            # pair so the re-link replaces rather than duplicates (RELATE is not
            # idempotent). Scoped to the directed pair — other edges untouched.
            await execute_query(
                "DELETE note_about WHERE in = $from AND out = $to",
                {"from": note_rid, "to": source_rid},
                self.config,
            )
            # Endpoints interpolated as literal record ids (RELATE graph syntax
            # won't bind a ``$param`` in the in/out position — see the docstring).
            # Safe because ``note_str``/``source_str`` passed ``_RECORD_ID_RE``
            # above: ``table:id`` only, no SurrealQL metacharacters. Only the SET
            # *values* stay parameterized.
            await execute_query(
                f"RELATE {note_str}->note_about->{source_str} "
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
                f"relate_note_source: failed to relate {note_str} -> "
                f"{source_str}: {e}"
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
