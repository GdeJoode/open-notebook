"""Auto-link orchestrator for notes (Track Y.2).

Given a note, find its most-related notes by embedding similarity and persist the
links as ``related_note`` RELATE edges. This is the orchestration layer over the
Y.1 primitives: it ensures the note is embedded, ranks candidates
(``NoteRepository.find_related_by_embedding``), applies a precision gate
(``min_similarity`` + top-``k``), and writes idempotent edges
(``NoteRepository.relate_note``).

Layering: the embed step (``EmbeddingService.embed_note``) lives in app-main's
local-Ollama pipeline, so this orchestrator lives here too — NOT in
surrealdb-service, which is deliberately embedding-free (the W.3 MCP design).
The MCP ``auto_link_note`` tool therefore does NOT call this service; it calls
the embedding-free repo primitives directly and requires the note to already
have an embedding (see ``mcp/server.py``).

Reversibility: only ``related_note`` edges are written; canonical ``note`` rows
are never mutated by linking. The embed step may populate ``note.embedding`` (a
needed, non-destructive backfill of a field that was ``[]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger

from surrealdb_service.repositories import NoteRepository

# Conservative defaults (Y.2 decision — avoid graph explosion, R.6 discipline).
#
# ``min_similarity`` is a cosine score in roughly [-1, 1]. 0.75 keeps only
# strongly-related notes: weak/incidental overlap (the bulk of any corpus once
# notes accumulate) is dropped, so the graph stays sparse and meaningful rather
# than near-complete. ``k`` caps the fan-out per note so even a cluster of very
# similar notes links to at most a handful, not all of them.
DEFAULT_MIN_SIMILARITY: float = 0.75
DEFAULT_K: int = 5

# Validation bounds (defense-in-depth — also enforced at the route layer).
MAX_K: int = 50
MIN_SIMILARITY_FLOOR: float = -1.0
MIN_SIMILARITY_CEIL: float = 1.0


@dataclass
class AutoLinkResult:
    """Summary of an auto-link run for a single note.

    ``status`` is the high-level outcome:
      * ``"linked"`` — ranking ran and edges were (re)written as needed;
      * ``"needs_embedding"`` — the note could not be embedded (no content, or
        the embed step produced nothing), so similarity could not run. This is a
        clean signal, never an error/crash.
      * ``"not_found"`` — no such note.
    """

    note_id: str
    status: str
    created: int = 0
    skipped_existing: int = 0
    below_threshold: int = 0
    candidates_considered: int = 0
    embedded: bool = False
    min_similarity: float = DEFAULT_MIN_SIMILARITY
    k: int = DEFAULT_K
    linked_note_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "note_id": self.note_id,
            "status": self.status,
            "created": self.created,
            "skipped_existing": self.skipped_existing,
            "below_threshold": self.below_threshold,
            "candidates_considered": self.candidates_considered,
            "embedded": self.embedded,
            "min_similarity": self.min_similarity,
            "k": self.k,
            "linked_note_ids": self.linked_note_ids,
        }


class NoteAutoLinkService:
    """Orchestrate note → related-notes → ``related_note`` edges (Y.2)."""

    def __init__(
        self,
        note_repo: NoteRepository,
        embedding_service: Any,
    ):
        """
        Args:
            note_repo: the Y.1 note repository (similarity + idempotent relate).
            embedding_service: an ``EmbeddingService`` with ``embed_note`` —
                injected (not imported) so this service stays testable without
                a live embedding model and so the embed dependency is explicit.
        """
        self.note_repo = note_repo
        self.embedding_service = embedding_service

    @staticmethod
    def _clamp_params(
        k: Optional[int], min_similarity: Optional[float]
    ) -> tuple[int, float]:
        """Normalise/clamp params to safe bounds (defense-in-depth).

        The route layer validates these at the boundary, but the service also
        clamps so a direct caller (the job in Y.3, a test) can never request an
        unbounded ``k`` or a nonsensical threshold.
        """
        eff_k = DEFAULT_K if k is None else int(k)
        eff_k = max(1, min(eff_k, MAX_K))

        eff_min = (
            DEFAULT_MIN_SIMILARITY if min_similarity is None else float(min_similarity)
        )
        eff_min = max(MIN_SIMILARITY_FLOOR, min(eff_min, MIN_SIMILARITY_CEIL))
        return eff_k, eff_min

    async def _ensure_embedding(self, note_id: str) -> tuple[bool, bool, bool]:
        """Ensure the note has a usable embedding, embedding it if needed.

        ``note.embedding`` is strict ``array<float>`` so an unembedded note holds
        ``[]`` (never NONE) — see [[note-embedding-non-optional]]. We embed when
        the field is missing/empty, then re-check.

        Returns ``(found, has_embedding, embedded_now)``. ``found`` is False for
        a missing note (so the caller can report ``not_found`` without a second
        ``get`` — the Y.2 follow-up). ``has_embedding`` is False — never a
        crash — when the note has no content to embed or the embed step produced
        nothing; the caller turns that into a ``needs_embedding`` result.
        """
        note = await self.note_repo.get(note_id)
        if note is None:
            return (False, False, False)

        if note.embedding:
            return (True, True, False)

        # No/empty embedding → embed via the injected EmbeddingService. embed_note
        # is a no-op (embeddings_created=0) for a note with no content; it never
        # raises for that case, but guard the whole call so a model/transport
        # failure becomes a clean needs-embedding result rather than a 500.
        try:
            embed_result = await self.embedding_service.embed_note(note_id)
        except Exception as e:  # noqa: BLE001 — degrade to needs-embedding, never crash
            logger.warning(f"auto-link: embed_note failed for {note_id}: {e}")
            return (True, False, False)

        if not embed_result or embed_result.embeddings_created < 1:
            logger.info(
                f"auto-link: note {note_id} could not be embedded "
                "(no content / nothing produced) — needs-embedding"
            )
            return (True, False, False)

        return (True, True, True)

    async def auto_link(
        self,
        note_id: str,
        *,
        k: Optional[int] = None,
        min_similarity: Optional[float] = None,
    ) -> AutoLinkResult:
        """Auto-link ``note_id`` to its most-related notes.

        Flow: ensure-embedding → rank by cosine (``find_related_by_embedding``)
        → keep only candidates at/above ``min_similarity`` (top-``k`` is enforced
        by the ranking LIMIT) → idempotently RELATE each via ``relate_note``
        (carrying the cosine ``similarity_score``).

        Idempotent: ``relate_note`` clears-before-relates each pair, so re-running
        yields the identical edge set (no duplicate rows). Self-links are
        impossible (cosine excludes self; ``relate_note`` refuses a self-edge).

        Args:
            note_id: the note to link from.
            k: max related notes to consider/link (clamped to [1, ``MAX_K``];
                default ``DEFAULT_K``).
            min_similarity: minimum cosine to link (clamped to [-1, 1]; default
                ``DEFAULT_MIN_SIMILARITY``).

        Returns:
            An ``AutoLinkResult`` summary. Never raises for the ordinary
            no-embedding / not-found cases — those are reported via ``status``.
        """
        eff_k, eff_min = self._clamp_params(k, min_similarity)

        found, has_embedding, embedded_now = await self._ensure_embedding(note_id)
        if not has_embedding:
            # not_found vs needs_embedding: ``found`` is carried out of
            # ``_ensure_embedding`` so we don't re-``get`` the note here (Y.2
            # follow-up — the redundant double get is gone).
            status = "needs_embedding" if found else "not_found"
            return AutoLinkResult(
                note_id=note_id,
                status=status,
                embedded=embedded_now,
                min_similarity=eff_min,
                k=eff_k,
            )

        candidates = await self.note_repo.find_related_by_embedding(note_id, eff_k)

        result = AutoLinkResult(
            note_id=note_id,
            status="linked",
            candidates_considered=len(candidates),
            embedded=embedded_now,
            min_similarity=eff_min,
            k=eff_k,
        )

        for cand in candidates:
            score = cand.get("score")
            target = cand.get("id")
            if score is None or target is None:
                continue
            if float(score) < eff_min:
                result.below_threshold += 1
                continue

            ok = await self.note_repo.relate_note(
                note_id, target, similarity_score=float(score), method="embedding"
            )
            if ok:
                result.created += 1
                result.linked_note_ids.append(target)
            else:
                # relate_note refuses self/wrong-table/unsafe ids or hit a DB
                # error — count it as skipped rather than failing the whole run.
                result.skipped_existing += 1

        logger.info(
            "auto-link {}: created={} below_threshold={} skipped={} "
            "(k={}, min_sim={}, embedded={})",
            note_id,
            result.created,
            result.below_threshold,
            result.skipped_existing,
            eff_k,
            eff_min,
            embedded_now,
        )
        return result
