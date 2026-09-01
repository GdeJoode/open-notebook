"""Track N.3 — abstention prompt + the not-a-concept gate wired into run_pass2.

The deterministic tier runs pure; the LLM-judge is mocked at the caller seam (the
run_pass2 caller is invoked a SECOND time with ``JUDGE_SYSTEM_PROMPT`` for the
ambiguous middle). Asserts the abstention clause is in the prompt, page-furniture
is dropped, real entities survive, chunks with no entity are counted as
abstentions, and the raw over-generation counts land in metadata.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

from ontology_extraction.not_a_concept import JUDGE_SYSTEM_PROMPT
from ontology_extraction.pass2_typed_extraction import run_pass2
from ontology_extraction.prompts.pass2 import build_pass2_prompt


def _ontology():
    return SimpleNamespace(
        metadata=SimpleNamespace(name="TestSchema"),
        entity_types={},
        relationship_types={},
    )


def _extract_payload(entities, relations=None):
    return json.dumps({"entities": entities, "relations": relations or []})


# --- abstention prompt clause ----------------------------------------------


def test_prompt_carries_abstention_clause():
    p = build_pass2_prompt(_ontology(), "some text", None)
    assert "INSUFFICIENT_EVIDENCE" in p
    assert 'return an EMPTY "entities" array' in p
    # still an EXCEPTION to exhaustive recall, not a licence to under-extract
    assert "single exception to exhaustive recall" in p


# --- deterministic gate in run_pass2 ---------------------------------------


def test_gate_drops_furniture_keeps_real_entity():
    ent = [
        {"text": "Regio Deal", "label": "RegioDeal", "confidence": 0.9},
        {"text": "Click here", "label": "other", "confidence": 0.9},
        {"text": "Figure 3", "label": "other", "confidence": 0.9},
    ]

    def caller(system: str, user: str, model: str) -> str:
        return _extract_payload(ent)

    result = asyncio.run(run_pass2([{"text": "t", "id": "c1"}], _ontology(),
                                   llm_caller=caller))
    texts = {e.text for e in result.entities}
    assert texts == {"Regio Deal"}
    assert result.metadata["entities_extracted"] == 3
    assert result.metadata["entities_kept"] == 1
    assert result.metadata["not_a_concept_removed"] == 2
    assert result.metadata["not_a_concept_judged"] == 0  # no ambiguous → no judge


def test_gate_drops_relation_referencing_removed_entity():
    ent = [
        {"text": "Regio Deal", "label": "RegioDeal", "confidence": 0.9},
        {"text": "Click here", "label": "other", "confidence": 0.9},
    ]
    rels = [
        {"source": "Regio Deal", "target": "Click here", "type": "mentions",
         "confidence": 0.8},
    ]

    def caller(system: str, user: str, model: str) -> str:
        return _extract_payload(ent, rels)

    result = asyncio.run(run_pass2([{"text": "t", "id": "c1"}], _ontology(),
                                   llm_caller=caller))
    # the relation's target was removed → the relation is dropped too
    assert result.relations == []


# --- abstention counting ----------------------------------------------------


def test_abstained_chunk_counted():
    payloads = [_extract_payload([{"text": "Regio Deal", "label": "RegioDeal",
                                   "confidence": 0.9}]),
                _extract_payload([])]  # furniture-only chunk → LLM abstains
    idx = {"n": 0}

    def caller(system: str, user: str, model: str) -> str:
        i = idx["n"]
        idx["n"] += 1
        return payloads[i]

    result = asyncio.run(run_pass2(
        [{"text": "real", "id": "c1"}, {"text": "furniture", "id": "c2"}],
        _ontology(), llm_caller=caller,
    ))
    assert result.metadata["abstained_chunks"] == 1
    assert result.metadata["chunk_count"] == 2


# --- LLM-judge on the ambiguous middle -------------------------------------


def test_judge_rejects_ambiguous_entity():
    ent = [
        {"text": "Regio Deal", "label": "RegioDeal", "confidence": 0.9},  # accept
        {"text": "governance", "label": "other", "confidence": 0.9},      # ambiguous
    ]

    def caller(system: str, user: str, model: str) -> str:
        if system == JUDGE_SYSTEM_PROMPT:
            return '{"verdicts": [{"text": "governance", "is_concept": false}]}'
        return _extract_payload(ent)

    result = asyncio.run(run_pass2([{"text": "t", "id": "c1"}], _ontology(),
                                   llm_caller=caller))
    assert {e.text for e in result.entities} == {"Regio Deal"}
    assert result.metadata["not_a_concept_judged"] == 1
    assert result.metadata["not_a_concept_removed"] == 1


def test_judge_keeps_ambiguous_entity():
    ent = [{"text": "governance", "label": "other", "confidence": 0.9}]

    def caller(system: str, user: str, model: str) -> str:
        if system == JUDGE_SYSTEM_PROMPT:
            return '{"verdicts": [{"text": "governance", "is_concept": true}]}'
        return _extract_payload(ent)

    result = asyncio.run(run_pass2([{"text": "t", "id": "c1"}], _ontology(),
                                   llm_caller=caller))
    assert {e.text for e in result.entities} == {"governance"}
    assert result.metadata["not_a_concept_removed"] == 0


def test_judge_disabled_keeps_ambiguous(monkeypatch):
    monkeypatch.setenv("EXTRACTION_NOT_A_CONCEPT_JUDGE", "0")
    ent = [{"text": "governance", "label": "other", "confidence": 0.9}]
    calls = {"n": 0}

    def caller(system: str, user: str, model: str) -> str:
        calls["n"] += 1
        assert system != JUDGE_SYSTEM_PROMPT, "judge must not be called when disabled"
        return _extract_payload(ent)

    result = asyncio.run(run_pass2([{"text": "t", "id": "c1"}], _ontology(),
                                   llm_caller=caller))
    assert {e.text for e in result.entities} == {"governance"}  # kept (no drop on guess)
    assert result.metadata["not_a_concept_judged"] == 0
    assert calls["n"] == 1  # only the extraction call


def test_gate_disabled_keeps_everything(monkeypatch):
    monkeypatch.setenv("EXTRACTION_NOT_A_CONCEPT", "0")
    ent = [{"text": "Click here", "label": "other", "confidence": 0.9}]

    def caller(system: str, user: str, model: str) -> str:
        return _extract_payload(ent)

    result = asyncio.run(run_pass2([{"text": "t", "id": "c1"}], _ontology(),
                                   llm_caller=caller))
    assert {e.text for e in result.entities} == {"Click here"}
    assert result.metadata["not_a_concept_removed"] == 0
