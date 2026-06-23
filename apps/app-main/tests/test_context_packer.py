"""Unit tests for the Track M context-aware chunk packer (M.3 + M.4 guard)."""

from app_main.services.extraction_chunking.context_packer import (
    DEFAULT_CONTEXT_WINDOW,
    estimate_tokens,
    input_budget_tokens,
    pack_chunks_for_model,
)


def _chunks(n: int, *, chars: int = 2000, source_id: str = "source:doc") -> list:
    """n ingestion-sized chunks of ~``chars`` each with stable ids/metadata."""
    return [
        {
            "text": "word " * (chars // 5),  # ~chars characters
            "id": f"chunk:{i}",
            "section_path": [f"section {i}"],
            "section_level": 1,
            "physical_page": i,
            "element_type": "paragraph",
            "source_id": source_id,
            "section_heading": f"section {i}",
        }
        for i in range(n)
    ]


def _max_window_tokens(packed: list) -> int:
    return max(estimate_tokens(w["text"]) for w in packed)


class TestEstimateTokens:
    def test_empty_is_zero(self):
        assert estimate_tokens("") == 0

    def test_rounds_up(self):
        # 5 chars -> ceil(5/4) == 2, never truncates downward.
        assert estimate_tokens("abcde") == 2


class TestInputBudget:
    def test_big_context_big_budget(self):
        budget = input_budget_tokens(
            context_window=256_000, max_output_tokens=4096, prompt_overhead_tokens=2000
        )
        # ~(256000 - 4096 - 2000) * 0.85
        assert budget > 200_000

    def test_null_context_degrades_to_default(self):
        budget = input_budget_tokens(
            context_window=None, max_output_tokens=None, prompt_overhead_tokens=2000
        )
        # Uses DEFAULT_CONTEXT_WINDOW, stays positive and bounded.
        assert 0 < budget < DEFAULT_CONTEXT_WINDOW

    def test_pathological_config_floors(self):
        # overhead > context would go negative without the floor.
        budget = input_budget_tokens(
            context_window=1000, max_output_tokens=2000, prompt_overhead_tokens=5000
        )
        assert budget >= 256


class TestPackCountDropsForBigContext:
    def test_big_context_few_windows(self):
        chunks = _chunks(60)
        packed = pack_chunks_for_model(
            chunks,
            context_window=128_000,
            max_output_tokens=4096,
            prompt_overhead_tokens=2000,
        )
        # 60 tiny chunks collapse into a handful of windows for a 128K model.
        assert len(packed) <= 5
        assert len(packed) < len(chunks)

    def test_pack_count_strictly_drops_vs_per_chunk(self):
        chunks = _chunks(28)  # the live ~28-batch case
        packed = pack_chunks_for_model(
            chunks,
            context_window=256_000,
            max_output_tokens=4096,
            prompt_overhead_tokens=2000,
        )
        # 256K model reads the whole document in a single call.
        assert len(packed) <= 2
        assert len(packed) < 28


class TestCallCountDropMeasurement:
    """The headline Track M improvement: a big-context model's effective
    LLM-call count drops from the live ~28 (one per 2000-char chunk) to a
    handful, while a small fallback re-packs without overflow."""

    def test_28_chunk_doc_drops_from_28_calls_to_few_for_128k(self):
        chunks = _chunks(28)  # the live case: 28 persisted 2000-char chunks
        old_call_count = len(chunks)  # pre-M: one LLM call per chunk
        packed = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        new_call_count = len(packed)
        assert old_call_count == 28
        assert new_call_count <= 3
        # The whole point: a >5x reduction in LLM calls for a big-context model.
        assert new_call_count < old_call_count / 5

    def test_same_doc_small_fallback_repacks_without_overflow(self):
        chunks = _chunks(28)
        fb_budget = input_budget_tokens(context_window=8192, max_output_tokens=512)
        primary = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        fallback = pack_chunks_for_model(
            chunks, context_window=8192, max_output_tokens=512
        )
        # Re-pack happened (more, smaller windows than the primary) and none
        # overflow the 8K fallback budget.
        assert len(fallback) > len(primary)
        assert _max_window_tokens(fallback) <= fb_budget


class TestManyBoundedWindowsForSmallContext:
    def test_small_context_many_windows_none_overflow(self):
        chunks = _chunks(40)
        budget = input_budget_tokens(
            context_window=8192, max_output_tokens=512, prompt_overhead_tokens=2000
        )
        packed = pack_chunks_for_model(
            chunks,
            context_window=8192,
            max_output_tokens=512,
            prompt_overhead_tokens=2000,
        )
        assert len(packed) >= 5
        assert _max_window_tokens(packed) <= budget


class TestOversizedChunkGuard:
    def test_single_oversized_chunk_is_resplit(self):
        # One 40K-char table chunk vs an 8K context: must re-split, no overflow.
        big = [
            {
                "text": "x" * 40_000,
                "id": "chunk:big",
                "section_path": [],
                "section_level": 0,
                "physical_page": None,
                "element_type": "table",
                "source_id": "source:doc",
            }
        ]
        budget = input_budget_tokens(
            context_window=8192, max_output_tokens=512, prompt_overhead_tokens=2000
        )
        packed = pack_chunks_for_model(
            big,
            context_window=8192,
            max_output_tokens=512,
            prompt_overhead_tokens=2000,
        )
        assert len(packed) >= 2
        assert _max_window_tokens(packed) <= budget

    def test_resplit_for_small_fallback_vs_packed_for_primary(self):
        # The M.4 invariant in miniature: the SAME chunks packed for a 256K
        # primary make few windows; re-packed for an 8K fallback make many,
        # none overflowing the fallback.
        chunks = _chunks(30)
        primary = pack_chunks_for_model(
            chunks, context_window=256_000, max_output_tokens=4096
        )
        fallback = pack_chunks_for_model(
            chunks, context_window=8192, max_output_tokens=512
        )
        assert len(primary) < len(fallback)
        fb_budget = input_budget_tokens(
            context_window=8192, max_output_tokens=512
        )
        assert _max_window_tokens(fallback) <= fb_budget


class TestOrderAndProvenancePreserved:
    def test_constituent_ids_cover_all_chunks_in_order(self):
        chunks = _chunks(10)
        packed = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        flattened = [
            cid for w in packed for cid in w["constituent_chunk_ids"]
        ]
        assert flattened == [f"chunk:{i}" for i in range(10)]

    def test_window_carries_first_chunk_metadata(self):
        chunks = _chunks(4)
        packed = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        first = packed[0]
        assert first["source_id"] == "source:doc"
        assert first["id"] == "chunk:0"
        assert first["section_path"] == ["section 0"]

    def test_empty_and_whitespace_chunks_dropped(self):
        chunks = [
            {"text": "real content here", "id": "chunk:0"},
            {"text": "   ", "id": "chunk:1"},
            {"text": "", "id": "chunk:2"},
        ]
        packed = pack_chunks_for_model(
            chunks, context_window=128_000, max_output_tokens=4096
        )
        flattened = [cid for w in packed for cid in w["constituent_chunk_ids"]]
        assert flattened == ["chunk:0"]


class TestDeterminism:
    def test_same_input_same_windows(self):
        chunks = _chunks(20)
        a = pack_chunks_for_model(chunks, context_window=32_000, max_output_tokens=2048)
        b = pack_chunks_for_model(chunks, context_window=32_000, max_output_tokens=2048)
        assert [w["text"] for w in a] == [w["text"] for w in b]
        assert [w["constituent_chunk_ids"] for w in a] == [
            w["constituent_chunk_ids"] for w in b
        ]
