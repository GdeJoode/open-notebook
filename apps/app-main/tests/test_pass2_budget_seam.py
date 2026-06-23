"""Seam regression for the Track M Pass-2 token-budget reconciliation (M rev2).

The bug: the service threaded the PACKING budget (``input_budget_tokens`` =
``(ctx - max_output - overhead) * 0.85``) into ``run_pass2``'s per-prompt guard,
but that guard measures the WHOLE prompt (ontology block + chunk_text). Since the
packer already filled chunk_text up to ``input_budget`` (overhead removed),
re-threading the packing budget made Pass-2 re-add the ontology overhead on top
of a budget that already removed it -> a near-ceiling window falsely raised
``Pass2TokenBudgetExceeded`` even though ``ontology + window`` fit the real
context.

The fix threads ``pass2_token_cap`` (= ``context_window - max_output_tokens``)
instead. This test exercises the real seam end to end: pack a window up to
``input_budget`` for a 128K model, prepend a realistic ontology block via the
real ``build_pass2_prompt``, and assert ``run_pass2`` does NOT raise under the
threaded cap (it WOULD under the old double-subtraction) — while a genuinely
oversized window still raises.
"""

from __future__ import annotations

import json

import pytest
from app_main.services.extraction_chunking.context_packer import (
    input_budget_tokens,
    pack_chunks_for_model,
    pass2_token_cap,
)
from ontology_extraction.pass2_typed_extraction import (
    Pass2TokenBudgetExceeded,
    _estimate_tokens,
    run_pass2,
)
from ontology_extraction.prompts.pass2 import build_pass2_prompt
from ontology_manager.schema import (
    EntityTypeDefinition,
    Ontology,
    OntologyMetadata,
    RelationshipTypeDefinition,
)

# Model dimensions for the headline big-context path (the reproduced abort).
CTX = 128_000
MAX_OUT = 4096


def _realistic_ontology(n_types: int = 40) -> Ontology:
    """An ontology whose serialized block runs ~1-1.5K tokens (the overhead the
    packer reserves and the Pass-2 guard re-measures)."""
    types = {
        f"Type{i:03d}": EntityTypeDefinition(
            name=f"Type{i:03d}",
            description=("a descriptive sentence about this entity type. " * 4),
        )
        for i in range(n_types)
    }
    rels = {
        f"REL{i:02d}": RelationshipTypeDefinition(
            name=f"REL{i:02d}",
            description="a relationship between two of the entity types above.",
            domain=[f"Type{i:03d}"],
            range=[f"Type{(i + 1) % n_types:03d}"],
        )
        for i in range(10)
    }
    return Ontology(
        metadata=OntologyMetadata(name="seam_ontology", version="1.0"),
        entity_types=types,
        relationship_types=rels,
    )


def _ingestion_chunks(total_chars: int, *, chunk_chars: int = 2000) -> list:
    """Ingestion-sized chunks summing to ~``total_chars`` of real words."""
    n = max(1, total_chars // chunk_chars)
    return [
        {
            "text": "word " * (chunk_chars // 5),
            "id": f"chunk:{i}",
            "section_path": ["s"],
            "section_level": 1,
            "physical_page": i,
            "element_type": "paragraph",
            "source_id": "source:doc",
            "section_heading": "s",
        }
        for i in range(n)
    ]


def _valid_response() -> str:
    return json.dumps({"entities": [], "relations": []})


@pytest.mark.asyncio
async def test_near_ceiling_packed_window_does_not_abort_under_threaded_cap():
    """A window packed to ~``input_budget`` for a 128K model + a realistic
    ontology block must pass ``run_pass2`` when threaded with ``pass2_token_cap``.

    Under the OLD double-subtraction (threading ``input_budget``) this same input
    raised ``Pass2TokenBudgetExceeded``; this test pins the fix.
    """
    pack_budget = input_budget_tokens(context_window=CTX, max_output_tokens=MAX_OUT)
    cap = pass2_token_cap(context_window=CTX, max_output_tokens=MAX_OUT)

    # Enough ingestion text that the packer fills the first window right up to
    # the input budget (a near-ceiling window, the reproduced abort case).
    chunks = _ingestion_chunks(pack_budget * 4 * 2)
    # No cap: this seam is about a window packed to the FULL input_budget (the
    # near-ceiling abort case). The M.3 default cap would otherwise shrink the
    # window well below the budget and the seam would not be exercised.
    packed = pack_chunks_for_model(
        chunks, context_window=CTX, max_output_tokens=MAX_OUT, max_window_tokens=None
    )
    window = packed[0]

    ontology = _realistic_ontology()

    # Prove this is genuinely the seam the reviewer flagged: ontology + window
    # measured TOGETHER exceeds the packing budget (so the old threading would
    # abort), yet sits comfortably under the threaded cap (so the fix admits it).
    whole_prompt = build_pass2_prompt(ontology, window["text"], [])
    measured = _estimate_tokens(whole_prompt)
    assert measured > pack_budget, (
        "test invalid: window+ontology must exceed the packing budget to "
        "reproduce the double-subtraction abort"
    )
    assert measured <= cap, "window+ontology must fit the threaded cap"

    called = {"n": 0}

    def fake_llm(sys_p: str, user_p: str, model: str) -> str:
        called["n"] += 1
        return _valid_response()

    # Must NOT raise — this is the regression. (Under the old threaded
    # ``input_budget`` it raised Pass2TokenBudgetExceeded here.)
    await run_pass2(
        [window],
        ontology,
        accepted_extensions=[],
        llm_caller=fake_llm,
        token_budget=cap,
    )
    assert called["n"] == 1


@pytest.mark.asyncio
async def test_old_double_subtraction_would_have_aborted():
    """Lock in the bug's shape: threading the PACKING budget (the old behavior)
    on the same near-ceiling window DOES raise — proving the fix is load-bearing,
    not a no-op."""
    pack_budget = input_budget_tokens(context_window=CTX, max_output_tokens=MAX_OUT)
    chunks = _ingestion_chunks(pack_budget * 4 * 2)
    # No cap: reproduce the full-context near-ceiling window (see above).
    packed = pack_chunks_for_model(
        chunks, context_window=CTX, max_output_tokens=MAX_OUT, max_window_tokens=None
    )
    window = packed[0]
    ontology = _realistic_ontology()

    def fake_llm(sys_p: str, user_p: str, model: str) -> str:  # pragma: no cover
        raise AssertionError("guard should fire before the LLM is called")

    with pytest.raises(Pass2TokenBudgetExceeded):
        await run_pass2(
            [window],
            ontology,
            accepted_extensions=[],
            llm_caller=fake_llm,
            token_budget=pack_budget,  # the OLD (buggy) threaded value
        )


@pytest.mark.asyncio
async def test_genuinely_oversized_window_still_raises_under_cap():
    """The real overflow guard is preserved: a prompt that ACTUALLY exceeds
    ``context_window - max_output_tokens`` still raises under the threaded cap."""
    cap = pass2_token_cap(context_window=CTX, max_output_tokens=MAX_OUT)
    # A single chunk whose text alone blows past the cap (~cap*4 chars + margin).
    oversized_text = "x" * (cap * 4 + 100_000)
    ontology = _realistic_ontology()

    measured = _estimate_tokens(build_pass2_prompt(ontology, oversized_text, []))
    assert measured > cap, "test invalid: input must genuinely exceed the cap"

    def fake_llm(sys_p: str, user_p: str, model: str) -> str:  # pragma: no cover
        raise AssertionError("guard should fire before the LLM is called")

    with pytest.raises(Pass2TokenBudgetExceeded):
        await run_pass2(
            [{"text": oversized_text, "id": "huge"}],
            ontology,
            accepted_extensions=[],
            llm_caller=fake_llm,
            token_budget=cap,
        )
