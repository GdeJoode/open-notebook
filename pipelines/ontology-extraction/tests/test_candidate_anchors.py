"""Track N.1 — the candidate-anchor block in the Pass-2 prompt.

Asserts the anchor section is present + correctly framed when anchors are
supplied, absent (back-compat) when not, and budget-capped so it can't crowd out
the chunk/ontology.
"""

from __future__ import annotations

from types import SimpleNamespace

from ontology_extraction.prompts.pass2 import _MAX_ANCHOR_CHARS, build_pass2_prompt


def _ontology():
    # build_pass2_prompt only reads .metadata.name + .entity_types/.relationship_types
    return SimpleNamespace(
        metadata=SimpleNamespace(name="TestSchema"),
        entity_types={},
        relationship_types={},
    )


def test_no_anchor_section_when_none():
    p = build_pass2_prompt(_ontology(), "some text", None, candidate_anchors=None)
    assert "Candidate anchors" not in p


def test_no_anchor_section_when_empty_list():
    p = build_pass2_prompt(_ontology(), "some text", None, candidate_anchors=[])
    assert "Candidate anchors" not in p


def test_anchor_section_present_and_framed_as_precision_aid():
    p = build_pass2_prompt(
        _ontology(), "text", None, candidate_anchors=["Audit Trail", "Backup Service"]
    )
    assert "## Candidate anchors (precision aid — NOT a shortlist)" in p
    assert "- Audit Trail" in p
    assert "- Backup Service" in p
    # exhaustive-recall framing preserved: anchors are not a filter
    assert "neither exhaustive nor a filter" in p


def test_anchor_block_is_budget_capped():
    many = [f"candidate-term-{i:04d}" for i in range(500)]
    p = build_pass2_prompt(_ontology(), "text", None, candidate_anchors=many)
    rendered = [line for line in p.splitlines() if line.startswith("- candidate-term-")]
    # not all 500 anchors survive the char budget
    assert 0 < len(rendered) < 500
    anchor_chars = sum(len(line[2:]) for line in rendered)
    assert anchor_chars <= _MAX_ANCHOR_CHARS


def test_blank_anchors_are_dropped():
    p = build_pass2_prompt(
        _ontology(), "text", None, candidate_anchors=["  ", "", "Real Term"]
    )
    assert "- Real Term" in p
    assert "- \n" not in p


# -- run_pass2 budget safety: anchors must never break the run ---------------

import asyncio  # noqa: E402

from ontology_extraction.pass2_typed_extraction import (  # noqa: E402
    _estimate_tokens,
    run_pass2,
)


def test_anchors_dropped_when_they_would_breach_budget():
    """Anchors are best-effort: a chunk that fits WITHOUT them must not be turned
    into a Pass2TokenBudgetExceeded by adding them — they get dropped instead."""
    ont = _ontology()
    chunk_text = "Regio Deal Ministerie van Economische Zaken " * 20
    chunks = [
        {"text": chunk_text, "id": "c1"},
        {"text": chunk_text + " Extra Term", "id": "c2"},
    ]
    base = _estimate_tokens(build_pass2_prompt(ont, chunk_text, []))
    budget = base + 5  # base fits; base + the ~150-token anchor block does not

    captured: dict = {}

    def caller(system: str, user: str, model: str) -> str:
        captured["prompt"] = user
        return '{"entities": [], "relations": []}'

    # Must NOT raise Pass2TokenBudgetExceeded — anchors are dropped to fit.
    result = asyncio.run(
        run_pass2(chunks, ont, llm_caller=caller, token_budget=budget)
    )
    assert result is not None
    assert "Candidate anchors" not in captured["prompt"]


def test_anchors_present_when_budget_allows():
    ont = _ontology()
    chunk_text = "Regio Deal and the Backup Service handle records."
    chunks = [
        {"text": chunk_text, "id": "c1"},
        {"text": "An unrelated Ministerie chunk about Beleid.", "id": "c2"},
    ]
    captured: dict = {}

    def caller(system: str, user: str, model: str) -> str:
        captured["prompt"] = user
        return '{"entities": [], "relations": []}'

    asyncio.run(run_pass2(chunks, ont, llm_caller=caller, token_budget=100_000))
    assert "Candidate anchors" in captured["prompt"]  # generous budget → kept
