"""Unit tests for the auto-link orchestrator (Track Y.2).

The orchestration logic (ensure-embedding → rank → threshold/top-k gate →
idempotent relate → summary) is exercised against an AsyncMock NoteRepository and
embedding service — no DB. The repo primitives themselves (cosine ranking,
clear-before-relate idempotency, self-edge refusal) are proven against a real
container in the Y.1 / Y.2-MCP roundtrip suites; here we prove the SERVICE drives
them correctly:

* below-threshold candidates are NOT linked (precision gate);
* top-k is honoured (the service passes k to the ranking and never links more);
* self is never a candidate (cosine excludes it; we assert the service never
  relates a note to itself);
* a note with an empty/absent embedding is embedded FIRST, then linked;
* a note that cannot be embedded → needs_embedding result, no crash, no edges;
* the run is idempotency-driving: every link goes through relate_note (which is
  itself clear-before-relate), so re-running re-issues the same idempotent calls.

NS.2 extends this to a SECOND pass — note→source — over the same embedding:
``find_related_sources_by_embedding`` → same threshold/top-k gate →
``relate_note_source``. The dedicated tests at the bottom prove the source pass
applies the gate, folds its (separate) counts into the summary, is additive to
the note pass (both run in one call), and never crosses wires with the note
counters.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from app_main.services.note_auto_link_service import (
    DEFAULT_K,
    DEFAULT_MIN_SIMILARITY,
    MAX_K,
    NoteAutoLinkService,
)

pytestmark = pytest.mark.asyncio


def _note(note_id: str, embedding):
    n = MagicMock()
    n.id = note_id
    n.embedding = embedding
    return n


def _embed_result(created: int):
    r = MagicMock()
    r.embeddings_created = created
    return r


def _make_service(
    *,
    note,
    candidates,
    source_candidates=None,
    relate_ok=True,
    relate_source_ok=True,
    embed_created=1,
):
    """Build a service whose repo returns ``note`` from get(), ``candidates``
    from find_related_by_embedding(), and ``source_candidates`` from
    find_related_sources_by_embedding(). embedding_service.embed_note flips the
    note to an embedded state.
    """
    repo = AsyncMock()
    embed_svc = AsyncMock()

    # get() returns the (mutable) note; embed_note populates its embedding so a
    # second get() after embedding reflects the change (mirrors the real flow).
    repo.get.return_value = note

    async def _embed_note(note_id):
        if embed_created >= 1:
            note.embedding = [1.0, 0.0, 0.0]
        return _embed_result(embed_created)

    embed_svc.embed_note.side_effect = _embed_note
    repo.find_related_by_embedding.return_value = candidates
    repo.relate_note.return_value = relate_ok
    # NS.2 source pass: default to no source candidates so existing note-only
    # tests stay note-focused; source-specific tests pass an explicit list.
    repo.find_related_sources_by_embedding.return_value = source_candidates or []
    repo.relate_note_source.return_value = relate_source_ok
    return NoteAutoLinkService(note_repo=repo, embedding_service=embed_svc), repo, embed_svc


# ---------------------------------------------------------------------------
# Threshold gate: only candidates at/above min_similarity are linked.
# ---------------------------------------------------------------------------


async def test_below_threshold_candidates_not_linked():
    note = _note("note:q", [1.0, 0.0, 0.0])
    candidates = [
        {"id": "note:a", "title": "a", "score": 0.95},  # >= 0.75 -> link
        {"id": "note:b", "title": "b", "score": 0.80},  # >= 0.75 -> link
        {"id": "note:c", "title": "c", "score": 0.60},  # <  0.75 -> drop
        {"id": "note:d", "title": "d", "score": 0.10},  # <  0.75 -> drop
    ]
    svc, repo, _ = _make_service(note=note, candidates=candidates)

    result = await svc.auto_link("note:q", min_similarity=0.75, k=10)

    assert result.status == "linked"
    assert result.created == 2
    assert result.below_threshold == 2
    assert set(result.linked_note_ids) == {"note:a", "note:b"}
    # relate_note was called ONLY for the above-threshold pairs.
    linked = {c.args[1] for c in repo.relate_note.await_args_list}
    assert linked == {"note:a", "note:b"}
    # The cosine score is carried onto the edge.
    scores = {c.args[1]: c.kwargs["similarity_score"] for c in repo.relate_note.await_args_list}
    assert scores["note:a"] == 0.95 and scores["note:b"] == 0.80


async def test_threshold_boundary_is_inclusive():
    note = _note("note:q", [1.0, 0.0])
    candidates = [{"id": "note:a", "title": "a", "score": 0.75}]
    svc, repo, _ = _make_service(note=note, candidates=candidates)

    result = await svc.auto_link("note:q", min_similarity=0.75)
    # Exactly at the threshold links (>= not >).
    assert result.created == 1
    assert result.below_threshold == 0


# ---------------------------------------------------------------------------
# Top-k: the service passes k to the ranking and never links beyond it.
# ---------------------------------------------------------------------------


async def test_top_k_passed_to_ranking_and_capped():
    note = _note("note:q", [1.0, 0.0])
    # Ranking already LIMITs to k; simulate it returning exactly k=2 here.
    candidates = [
        {"id": "note:a", "title": "a", "score": 0.99},
        {"id": "note:b", "title": "b", "score": 0.98},
    ]
    svc, repo, _ = _make_service(note=note, candidates=candidates)

    result = await svc.auto_link("note:q", k=2, min_similarity=0.5)

    # k was forwarded to the ranking call (the LIMIT is enforced there).
    repo.find_related_by_embedding.assert_awaited_once_with("note:q", 2)
    assert result.created == 2
    assert result.k == 2


async def test_k_clamped_to_bounds():
    note = _note("note:q", [1.0, 0.0])
    svc, repo, _ = _make_service(note=note, candidates=[])

    # k above MAX_K is clamped down; k below 1 is clamped up.
    await svc.auto_link("note:q", k=9999)
    assert repo.find_related_by_embedding.await_args_list[-1].args[1] == MAX_K
    await svc.auto_link("note:q", k=0)
    assert repo.find_related_by_embedding.await_args_list[-1].args[1] == 1


# ---------------------------------------------------------------------------
# Self-exclusion: the service never relates a note to itself.
# ---------------------------------------------------------------------------


async def test_never_links_to_self():
    note = _note("note:q", [1.0, 0.0])
    # find_related_by_embedding already excludes self (Y.1). Even so, assert the
    # service never issues a self relate.
    candidates = [{"id": "note:a", "title": "a", "score": 0.9}]
    svc, repo, _ = _make_service(note=note, candidates=candidates)

    await svc.auto_link("note:q", min_similarity=0.5)
    for call in repo.relate_note.await_args_list:
        assert call.args[0] != call.args[1], "service issued a self-edge relate"
        assert call.args[1] != "note:q"


# ---------------------------------------------------------------------------
# No-embedding path: embed first, or report needs_embedding (no crash).
# ---------------------------------------------------------------------------


async def test_unembedded_note_is_embedded_then_linked():
    # Note starts with the strict-but-empty [] embedding (needs embedding).
    note = _note("note:q", [])
    candidates = [{"id": "note:a", "title": "a", "score": 0.9}]
    svc, repo, embed_svc = _make_service(
        note=note, candidates=candidates, embed_created=1
    )

    result = await svc.auto_link("note:q", min_similarity=0.5)

    embed_svc.embed_note.assert_awaited_once_with("note:q")
    assert result.embedded is True
    assert result.status == "linked"
    assert result.created == 1


async def test_note_that_cannot_be_embedded_returns_needs_embedding():
    note = _note("note:q", [])  # empty, and embedding produces nothing
    svc, repo, embed_svc = _make_service(
        note=note, candidates=[], embed_created=0
    )

    result = await svc.auto_link("note:q")

    assert result.status == "needs_embedding"
    assert result.created == 0
    # Ranking/relate never ran — nothing to compare.
    repo.find_related_by_embedding.assert_not_awaited()
    repo.relate_note.assert_not_awaited()


async def test_embed_failure_degrades_to_needs_embedding_no_crash():
    note = _note("note:q", [])
    repo = AsyncMock()
    repo.get.return_value = note
    embed_svc = AsyncMock()
    embed_svc.embed_note.side_effect = RuntimeError("ollama down")
    svc = NoteAutoLinkService(note_repo=repo, embedding_service=embed_svc)

    result = await svc.auto_link("note:q")

    assert result.status == "needs_embedding"
    repo.relate_note.assert_not_awaited()


async def test_missing_note_returns_not_found():
    repo = AsyncMock()
    repo.get.return_value = None
    embed_svc = AsyncMock()
    svc = NoteAutoLinkService(note_repo=repo, embedding_service=embed_svc)

    result = await svc.auto_link("note:ghost")

    assert result.status == "not_found"
    repo.find_related_by_embedding.assert_not_awaited()
    embed_svc.embed_note.assert_not_awaited()


# ---------------------------------------------------------------------------
# Already-embedded note skips the embed step.
# ---------------------------------------------------------------------------


async def test_already_embedded_note_skips_embedding():
    note = _note("note:q", [1.0, 0.0])
    svc, repo, embed_svc = _make_service(note=note, candidates=[])

    result = await svc.auto_link("note:q")

    embed_svc.embed_note.assert_not_awaited()
    assert result.embedded is False


# ---------------------------------------------------------------------------
# Idempotency-driving: a relate that returns False is counted as skipped, never
# a crash; every link is routed through the idempotent relate_note.
# ---------------------------------------------------------------------------


async def test_relate_refused_counts_as_skipped():
    note = _note("note:q", [1.0, 0.0])
    candidates = [{"id": "note:a", "title": "a", "score": 0.9}]
    svc, repo, _ = _make_service(
        note=note, candidates=candidates, relate_ok=False
    )

    result = await svc.auto_link("note:q", min_similarity=0.5)
    assert result.created == 0
    assert result.skipped_existing == 1


async def test_defaults_applied_when_params_omitted():
    note = _note("note:q", [1.0, 0.0])
    svc, repo, _ = _make_service(note=note, candidates=[])

    result = await svc.auto_link("note:q")
    assert result.k == DEFAULT_K
    assert result.min_similarity == DEFAULT_MIN_SIMILARITY
    repo.find_related_by_embedding.assert_awaited_once_with("note:q", DEFAULT_K)


# ===========================================================================
# NS.2 — the note → source pass (note_about edges).
# ===========================================================================


async def test_source_pass_threshold_gate_only_links_above():
    """Only sources at/above min_similarity become note_about edges."""
    note = _note("note:q", [1.0, 0.0, 0.0])
    sources = [
        {"id": "source:a", "title": "a", "score": 0.95},  # >= 0.75 -> link
        {"id": "source:b", "title": "b", "score": 0.78},  # >= 0.75 -> link
        {"id": "source:c", "title": "c", "score": 0.50},  # <  0.75 -> drop
    ]
    svc, repo, _ = _make_service(
        note=note, candidates=[], source_candidates=sources
    )

    result = await svc.auto_link("note:q", min_similarity=0.75, k=10)

    assert result.status == "linked"
    assert result.source_links_created == 2
    assert result.source_below_threshold == 1
    assert result.source_candidates_considered == 3
    assert set(result.linked_source_ids) == {"source:a", "source:b"}
    # relate_note_source called ONLY for above-threshold sources, carrying score.
    related = {
        c.args[1]: c.kwargs["similarity_score"]
        for c in repo.relate_note_source.await_args_list
    }
    assert related == {"source:a": 0.95, "source:b": 0.78}
    # The note pass had no candidates → note counters stay zero (no cross-wiring).
    assert result.created == 0 and result.below_threshold == 0


async def test_source_top_k_forwarded_to_source_ranking():
    note = _note("note:q", [1.0, 0.0])
    svc, repo, _ = _make_service(
        note=note, candidates=[], source_candidates=[]
    )

    await svc.auto_link("note:q", k=3, min_similarity=0.5)
    # k is forwarded to BOTH rankings (the LIMIT enforces top-k in each).
    repo.find_related_sources_by_embedding.assert_awaited_once_with("note:q", 3)


async def test_source_relate_refused_counts_as_source_skipped():
    note = _note("note:q", [1.0, 0.0])
    sources = [{"id": "source:a", "title": "a", "score": 0.9}]
    svc, repo, _ = _make_service(
        note=note,
        candidates=[],
        source_candidates=sources,
        relate_source_ok=False,
    )

    result = await svc.auto_link("note:q", min_similarity=0.5)
    assert result.source_links_created == 0
    assert result.source_skipped_existing == 1


async def test_both_passes_run_in_one_call_and_counts_are_separate():
    """A single auto_link produces BOTH related_note and note_about edges, with
    the two link types reported in distinct counters (additive, no conflation)."""
    note = _note("note:q", [1.0, 0.0, 0.0])
    notes = [
        {"id": "note:a", "title": "a", "score": 0.95},
        {"id": "note:b", "title": "b", "score": 0.60},  # below 0.75
    ]
    sources = [
        {"id": "source:x", "title": "x", "score": 0.90},
        {"id": "source:y", "title": "y", "score": 0.92},
        {"id": "source:z", "title": "z", "score": 0.40},  # below 0.75
    ]
    svc, repo, _ = _make_service(
        note=note, candidates=notes, source_candidates=sources
    )

    result = await svc.auto_link("note:q", min_similarity=0.75, k=10)

    # Note pass.
    assert result.created == 1
    assert result.below_threshold == 1
    assert result.candidates_considered == 2
    assert result.linked_note_ids == ["note:a"]
    # Source pass.
    assert result.source_links_created == 2
    assert result.source_below_threshold == 1
    assert result.source_candidates_considered == 3
    assert set(result.linked_source_ids) == {"source:x", "source:y"}
    # Both relate primitives were driven from the one call.
    repo.relate_note.assert_awaited_once()
    assert repo.relate_note_source.await_count == 2
    # The summary dict carries both link types.
    d = result.to_dict()
    assert d["created"] == 1 and d["source_links_created"] == 2
    assert d["linked_source_ids"] and d["linked_note_ids"]


async def test_note_pass_unaffected_when_no_sources():
    """Additivity: with zero source candidates, the note pass is identical to
    Y.2 and status is still 'linked' (source pass is a clean no-op)."""
    note = _note("note:q", [1.0, 0.0])
    notes = [{"id": "note:a", "title": "a", "score": 0.9}]
    svc, repo, _ = _make_service(
        note=note, candidates=notes, source_candidates=[]
    )

    result = await svc.auto_link("note:q", min_similarity=0.5)

    assert result.status == "linked"
    assert result.created == 1
    assert result.source_links_created == 0
    assert result.source_candidates_considered == 0
    repo.relate_note_source.assert_not_awaited()


async def test_needs_embedding_runs_neither_pass():
    """A note that cannot be embedded reports needs_embedding and neither the
    note nor the source ranking runs (NS.2 extends the Y.2 guard symmetrically)."""
    note = _note("note:q", [])
    svc, repo, _ = _make_service(
        note=note, candidates=[], source_candidates=[], embed_created=0
    )

    result = await svc.auto_link("note:q")

    assert result.status == "needs_embedding"
    assert result.source_links_created == 0
    repo.find_related_by_embedding.assert_not_awaited()
    repo.find_related_sources_by_embedding.assert_not_awaited()
    repo.relate_note_source.assert_not_awaited()
