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
from surrealdb_service.repositories.search import _PROVENANCE_KEYS, _hit_source_id


def _repo_with_best(best: dict) -> SearchRepository:
    repo = SearchRepository()
    repo._best_chunk_per_source = AsyncMock(return_value=best)
    return repo


class TestHitSourceId:
    def test_source_hit_resolves_via_parent_id(self):
        assert (
            _hit_source_id({"id": "source:a", "parent_id": "source:a"})
            == "source:a"
        )

    def test_source_embedding_hit_uses_parent_id_not_id(self):
        # fn:: sets parent_id to the source even when id differs.
        hit = {"id": "source:s1", "parent_id": "source:s1"}
        assert _hit_source_id(hit) == "source:s1"

    def test_note_hit_is_not_source_scoped(self):
        assert _hit_source_id({"id": "note:n1", "parent_id": "note:n1"}) is None

    def test_falls_back_to_id_when_no_parent(self):
        assert _hit_source_id({"id": "source:x"}) == "source:x"

    def test_non_string_id_is_none(self):
        assert _hit_source_id({"id": None}) is None


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
    async def test_text_only_path_passes_no_embedding(self):
        """Text-search hydration must call the lookup with embedding=None so
        the source-level (page-less) branch is used.
        """
        repo = _repo_with_best({})
        hits = [{"id": "source:s1", "parent_id": "source:s1"}]

        await repo.hydrate_provenance(hits, embedding=None)

        repo._best_chunk_per_source.assert_awaited_once()
        _, kwargs = repo._best_chunk_per_source.call_args
        # second positional arg is the embedding
        args = repo._best_chunk_per_source.call_args.args
        assert args[1] is None

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
