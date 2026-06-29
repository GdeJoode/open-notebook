"""Unit tests for ``SearchRepository`` chunk-provenance hydration (Track X.1).

These exercise ``hydrate_provenance`` with the DB join (``_best_chunk_per_source``)
mocked, so they assert the Python-side contract:

* every hit gets the full provenance key set (stable shape);
* a source hit with a matching chunk carries that chunk's real page/section;
* notes / non-source hits and lookup misses degrade to ``None`` on every key;
* a DB failure during hydration never raises and never mutates existing keys.

The hit→chunk *mapping* itself (cosine top-1 == fn::vector_search's collapsed
``math::max``) is proven separately against staging in
``test_search_provenance_staging.py``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

import surrealdb_service.repositories.search as search_mod
from surrealdb_service.repositories import SearchRepository
from surrealdb_service.repositories.search import (
    _PROVENANCE_KEYS,
    _hit_is_chunk_backed,
    _hit_parent_source,
)


def _repo_with_best(best: dict) -> SearchRepository:
    repo = SearchRepository()
    repo._best_chunk_per_source = AsyncMock(return_value=best)
    return repo


class TestHitClassDistinguishers:
    """The source-level anchor (``parent_id``) vs the chunk-backed decision
    (own ``id`` prefix). A ``source_insight`` hit has a ``source:`` parent but
    is NOT chunk-backed — that distinction is the whole point of the X.1 fix.
    """

    def test_parent_source_from_parent_id(self):
        assert (
            _hit_parent_source({"id": "source:a", "parent_id": "source:a"})
            == "source:a"
        )

    def test_insight_parent_is_its_source(self):
        # fn:: emits source_insight hits with parent_id == owning source.
        hit = {"id": "source_insight:i1", "parent_id": "source:s1"}
        assert _hit_parent_source(hit) == "source:s1"

    def test_note_has_no_source_parent(self):
        assert _hit_parent_source({"id": "note:n1", "parent_id": "note:n1"}) is None

    def test_chunk_backed_only_for_source_own_id(self):
        # source_embedding hit -> own id is source: (fn:: SELECT source.id AS id)
        assert _hit_is_chunk_backed({"id": "source:s1", "parent_id": "source:s1"})

    def test_insight_is_not_chunk_backed(self):
        # own id is source_insight: even though parent_id is source:
        assert not _hit_is_chunk_backed(
            {"id": "source_insight:i1", "parent_id": "source:s1"}
        )

    def test_note_is_not_chunk_backed(self):
        assert not _hit_is_chunk_backed({"id": "note:n1", "parent_id": "note:n1"})

    def test_non_string_id_not_chunk_backed(self):
        assert not _hit_is_chunk_backed({"id": None})


class TestHydrateProvenance:
    @pytest.mark.asyncio
    async def test_attaches_real_chunk_provenance_to_source_hit(self):
        repo = _repo_with_best(
            {
                "source:s1": {
                    "chunk_id": "chunk:c1",
                    "physical_page": 10,
                    "printed_page": 11,
                    "section_path": ["Intro", "Background"],
                    "element_type": "text",
                }
            }
        )
        hits = [{"id": "source:s1", "parent_id": "source:s1", "similarity": 0.9}]

        out = await repo.hydrate_provenance(hits, embedding=[0.1, 0.2])

        hit = out[0]
        assert hit["chunk_id"] == "chunk:c1"
        assert hit["physical_page"] == 10
        assert hit["printed_page"] == 11
        assert hit["section_path"] == ["Intro", "Background"]
        assert hit["element_type"] == "text"
        # existing fields untouched (additive)
        assert hit["similarity"] == 0.9
        assert hit["source"] == "source:s1"

    @pytest.mark.asyncio
    async def test_all_keys_present_even_without_chunk(self):
        """Shape is stable: a source with no matching chunk still gets every
        provenance key, set to None (callers can rely on ``hit['physical_page']``).
        """
        repo = _repo_with_best({})  # no chunk found for any source
        hits = [{"id": "source:s1", "parent_id": "source:s1"}]

        out = await repo.hydrate_provenance(hits, embedding=[0.1])

        for key in _PROVENANCE_KEYS:
            assert key in out[0]
            assert out[0][key] is None

    @pytest.mark.asyncio
    async def test_note_hit_gets_null_provenance(self):
        """Notes have no page — they degrade to None without error, and we
        never look them up as a source.
        """
        repo = _repo_with_best({})
        hits = [{"id": "note:n1", "parent_id": "note:n1", "relevance": 5.0}]

        out = await repo.hydrate_provenance(hits, embedding=None)

        assert out[0]["relevance"] == 5.0
        assert out[0]["source"] is None
        for key in _PROVENANCE_KEYS:
            assert out[0][key] is None
        # a note id must never be queried as a source
        repo._best_chunk_per_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_mixed_source_and_note_hits(self):
        repo = _repo_with_best(
            {"source:s1": {"chunk_id": "chunk:c1", "physical_page": 3}}
        )
        hits = [
            {"id": "source:s1", "parent_id": "source:s1"},
            {"id": "note:n1", "parent_id": "note:n1"},
        ]

        out = await repo.hydrate_provenance(hits, embedding=[0.5])

        by_id = {h["id"]: h for h in out}
        assert by_id["source:s1"]["physical_page"] == 3
        assert by_id["source:s1"]["chunk_id"] == "chunk:c1"
        # the chunk had no section/printed page in the lookup -> None
        assert by_id["source:s1"]["section_path"] is None
        assert by_id["note:n1"]["physical_page"] is None

    @pytest.mark.asyncio
    async def test_source_insight_hit_gets_no_chunk_provenance(self):
        """BLOCKER fix: a source_insight hit (own id ``source_insight:``, parent
        ``source:``) must NOT be stamped with a chunk's page — an insight has no
        single originating chunk. All chunk keys stay None; ``source`` is set
        from parent_id; it is never routed through the chunk lookup.
        """
        repo = _repo_with_best(
            {"source:s1": {"chunk_id": "chunk:c1", "physical_page": 42}}
        )
        hits = [
            {
                "id": "source_insight:i1",
                "parent_id": "source:s1",
                "similarity": 0.88,
            }
        ]

        out = await repo.hydrate_provenance(hits, embedding=[0.1, 0.2])

        hit = out[0]
        # source-level anchor preserved
        assert hit["source"] == "source:s1"
        assert hit["similarity"] == 0.88
        # NO chunk-level provenance — not the source's top embedding chunk
        assert hit["chunk_id"] is None
        assert hit["physical_page"] is None
        assert hit["printed_page"] is None
        assert hit["section_path"] is None
        assert hit["element_type"] is None
        # the insight must never be looked up as a chunk-backed source
        repo._best_chunk_per_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_insight_alongside_embedding_hit_for_same_source(self):
        """An insight and an embedding hit for the SAME source: only the
        embedding (``source:``) hit gets the page; the insight stays page-less.
        """
        repo = _repo_with_best(
            {"source:s1": {"chunk_id": "chunk:c1", "physical_page": 7}}
        )
        hits = [
            {"id": "source:s1", "parent_id": "source:s1"},  # embedding hit
            {"id": "source_insight:i1", "parent_id": "source:s1"},  # insight
        ]

        out = await repo.hydrate_provenance(hits, embedding=[0.5])

        by_id = {h["id"]: h for h in out}
        assert by_id["source:s1"]["physical_page"] == 7
        assert by_id["source:s1"]["chunk_id"] == "chunk:c1"
        assert by_id["source_insight:i1"]["physical_page"] is None
        assert by_id["source_insight:i1"]["chunk_id"] is None
        assert by_id["source_insight:i1"]["source"] == "source:s1"
        # only the embedding hit's source id was looked up
        repo._best_chunk_per_source.assert_awaited_once()
        looked_up = repo._best_chunk_per_source.call_args.args[0]
        assert looked_up == ["source:s1"]

    @pytest.mark.asyncio
    async def test_text_only_attaches_no_chunk_keys(self):
        """Text-only path (embedding=None): assert NO chunk-level provenance is
        attached for any hit — not even an arbitrary first chunk's
        section_path/element_type. Source-level only.
        """
        repo = _repo_with_best({})
        hits = [
            {"id": "source:s1", "parent_id": "source:s1", "relevance": 9.0},
        ]

        out = await repo.hydrate_provenance(hits, embedding=None)

        hit = out[0]
        assert hit["source"] == "source:s1"
        assert hit["relevance"] == 9.0
        for key in _PROVENANCE_KEYS:
            assert hit[key] is None
        # text path must not run the chunk lookup at all
        repo._best_chunk_per_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_lookup_failure_degrades_gracefully(self):
        """A DB error during hydration must not raise and must not corrupt the
        hits — provenance keys are seeded to None, existing keys intact.
        """
        repo = SearchRepository()
        repo._best_chunk_per_source = AsyncMock(
            side_effect=RuntimeError("db down")
        )
        hits = [{"id": "source:s1", "parent_id": "source:s1", "similarity": 0.7}]

        out = await repo.hydrate_provenance(hits, embedding=[0.1])

        assert out[0]["similarity"] == 0.7
        for key in _PROVENANCE_KEYS:
            assert out[0][key] is None

    @pytest.mark.asyncio
    async def test_empty_hits_no_lookup(self):
        repo = _repo_with_best({})
        out = await repo.hydrate_provenance([], embedding=[0.1])
        assert out == []
        repo._best_chunk_per_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_none_value_kept_when_lookup_gives_none(self):
        """If a hit already carries a provenance key, a None from the lookup
        must not clobber it; a real value does update it.
        """
        repo = _repo_with_best(
            {"source:s1": {"chunk_id": "chunk:c1", "physical_page": None}}
        )
        hits = [
            {
                "id": "source:s1",
                "parent_id": "source:s1",
                "physical_page": 99,  # pre-existing
            }
        ]

        out = await repo.hydrate_provenance(hits, embedding=[0.1])

        # chunk_id filled from lookup; physical_page kept (lookup gave None)
        assert out[0]["chunk_id"] == "chunk:c1"
        assert out[0]["physical_page"] == 99


class TestSearchMethodsWireHydration:
    """The three search entry points must hydrate, and ``hybrid_search`` must
    hydrate the fused set once (legs called with ``hydrate=False``).
    """

    @pytest.mark.asyncio
    async def test_vector_search_hydrates_with_embedding(self):
        repo = SearchRepository()
        repo.hydrate_provenance = AsyncMock(side_effect=lambda hits, embedding=None: hits)
        with patch.object(
            search_mod, "execute_query", AsyncMock(return_value=[{"id": "source:s1"}])
        ):
            await repo.vector_search([0.1, 0.2], results=5)

        repo.hydrate_provenance.assert_awaited_once()
        _, kwargs = repo.hydrate_provenance.call_args
        assert kwargs["embedding"] == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_text_search_hydrates_without_embedding(self):
        repo = SearchRepository()
        repo.hydrate_provenance = AsyncMock(side_effect=lambda hits, embedding=None: hits)
        with patch.object(
            search_mod, "execute_query", AsyncMock(return_value=[{"id": "source:s1"}])
        ):
            await repo.text_search("kw", results=5)

        repo.hydrate_provenance.assert_awaited_once()
        _, kwargs = repo.hydrate_provenance.call_args
        assert kwargs["embedding"] is None

    @pytest.mark.asyncio
    async def test_text_search_hydrate_false_skips(self):
        repo = SearchRepository()
        repo.hydrate_provenance = AsyncMock()
        with patch.object(
            search_mod, "execute_query", AsyncMock(return_value=[{"id": "source:s1"}])
        ):
            out = await repo.text_search("kw", hydrate=False)

        repo.hydrate_provenance.assert_not_called()
        assert out == [{"id": "source:s1"}]

    @pytest.mark.asyncio
    async def test_hybrid_hydrates_fused_set_once_with_embedding(self):
        repo = SearchRepository()
        repo.text_search = AsyncMock(return_value=[{"id": "source:s1"}])
        repo.vector_search = AsyncMock(return_value=[{"id": "source:s1"}])
        repo.hydrate_provenance = AsyncMock(side_effect=lambda hits, embedding=None: hits)

        await repo.hybrid_search("kw", [0.3, 0.4], results=10)

        # legs called with hydrate=False so we hydrate the fused set once
        assert repo.text_search.call_args.kwargs.get("hydrate") is False
        assert repo.vector_search.call_args.kwargs.get("hydrate") is False
        repo.hydrate_provenance.assert_awaited_once()
        assert repo.hydrate_provenance.call_args.kwargs["embedding"] == [0.3, 0.4]
