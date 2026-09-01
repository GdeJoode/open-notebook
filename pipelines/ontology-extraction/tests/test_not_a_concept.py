"""Track N.3 — the extraction-time not-a-concept gate.

Deterministic tier (high-precision reject / fast accept / ambiguous) + the pure
judge prompt/parse. No LLM needed — the judge call is orchestrated at the
run_pass2 seam (see test_abstention.py).
"""

from __future__ import annotations

from ontology_extraction.not_a_concept import (
    build_judge_prompt,
    classify_deterministic,
    parse_judge_response,
    partition_deterministic,
)

# --- deterministic REJECT (high precision) ---------------------------------


def test_rejects_ui_and_boilerplate_labels():
    # unconditional furniture — rejected regardless of label
    for junk in ["Click here", "Read more", "Back to top", "Download",
                 "All rights reserved", "Privacy Policy", "Lees meer",
                 "Inhoudsopgave"]:
        assert classify_deterministic(junk, "other") is True, junk
        assert classify_deterministic(junk, "Ministerie") is True, junk  # even specific


def test_field_word_rejected_under_generic_but_kept_under_specific_label():
    # N.3 review MAJOR: a homograph field word is furniture under a generic label
    # but a KEPT real entity when the LLM committed to a specific type.
    for word in ["Total", "Page", "Note", "Datum", "Tabel"]:
        assert classify_deterministic(word, "other") is True, word
        assert classify_deterministic(word, "Organisatie") is False, word


def test_ui_homograph_under_generic_label_defers_to_judge():
    # "Next"/"Home"/"Index" could be a real entity (the retailer Next, a stock
    # Index) — never hard-rejected; they defer to the judge under a generic label.
    for word in ["Next", "Home", "Volgende", "Menu", "Index"]:
        assert classify_deterministic(word, "other") is None, word
        assert classify_deterministic(word, "Organisatie") is False, word


def test_rejects_references_and_numbers():
    for junk in ["Figure 3", "Table 2a", "Page 12", "Hoofdstuk 4", "Bijlage 2"]:
        assert classify_deterministic(junk, "other") is True, junk
    for num in ["12345", "3.14", "42%", "€ 1.200", "2021", "-", "•", "()"]:
        assert classify_deterministic(num, "other") is True, num


def test_rejects_too_short():
    assert classify_deterministic("A", "Researcher") is True
    assert classify_deterministic("X", "other") is True


# --- deterministic ACCEPT --------------------------------------------------


def test_accepts_specific_label():
    # The LLM committed to a specific schema type → trust it, no judge call.
    assert classify_deterministic("Regio Deal", "RegioDeal") is False
    assert classify_deterministic("subsidies", "Instrument") is False


def test_accepts_multiword_proper_name_even_under_generic_label():
    assert classify_deterministic("Ministerie van BZK", "other") is False
    assert classify_deterministic("Regio Deal Midden-Limburg", "other") is False


# --- AMBIGUOUS (defer to judge) --------------------------------------------


def test_generic_word_under_generic_label_is_ambiguous():
    assert classify_deterministic("governance", "other") is None
    assert classify_deterministic("challenges", "other") is None
    # a specific label flips the same word to accept
    assert classify_deterministic("governance", "BeleidsThema") is False


def test_extra_reject_exact_is_honoured():
    extra = frozenset({"quarterly report"})
    assert classify_deterministic("Quarterly Report", "other",
                                  extra_reject_exact=extra) is True


# --- partition -------------------------------------------------------------


def test_partition_three_buckets_preserves_order():
    ents = [
        {"text": "Regio Deal", "label": "RegioDeal"},   # accept
        {"text": "Click here", "label": "other"},         # reject
        {"text": "governance", "label": "other"},         # ambiguous
        {"text": "Ministerie van BZK", "label": "other"}, # accept (proper name)
    ]
    kept, rejected, ambiguous = partition_deterministic(ents)
    assert [e["text"] for e in kept] == ["Regio Deal", "Ministerie van BZK"]
    assert [e["text"] for e in rejected] == ["Click here"]
    assert [e["text"] for e in ambiguous] == ["governance"]


def test_partition_reads_object_entities():
    from types import SimpleNamespace

    ents = [SimpleNamespace(text="Regio Deal", label="RegioDeal"),
            SimpleNamespace(text="Page 3", label="other")]
    kept, rejected, ambiguous = partition_deterministic(ents)
    assert len(kept) == 1 and len(rejected) == 1 and not ambiguous


# --- judge prompt / parse (pure) -------------------------------------------


def test_build_judge_prompt_lists_candidates():
    prompt = build_judge_prompt([("governance", "other"), ("challenges", "BeleidsThema")])
    assert '"governance" (type: other)' in prompt
    assert '"challenges" (type: BeleidsThema)' in prompt
    assert "is_concept" in prompt


def test_parse_judge_response_maps_verdicts():
    items = [("governance", "other"), ("challenges", "other")]
    raw = '{"verdicts": [{"text": "governance", "is_concept": false}, ' \
          '{"text": "challenges", "is_concept": true}]}'
    got = parse_judge_response(raw, items)
    assert got == {"governance": False, "challenges": True}


def test_parse_judge_tolerates_fences_and_omits_silent_items():
    # Only EXPLICIT verdicts are returned; a silent candidate is absent (the caller
    # defaults it to keep). This lets the caller count what was truly arbitrated.
    items = [("governance", "other"), ("challenges", "other")]
    raw = "```json\n{\"verdicts\": [{\"text\": \"governance\", \"is_concept\": false}]}\n```"
    got = parse_judge_response(raw, items)
    assert got == {"governance": False}
    assert got.get("challenges", True) is True  # missing → caller keeps


def test_parse_judge_garbage_returns_empty():
    items = [("governance", "other")]
    assert parse_judge_response("not json at all", items) == {}
    assert parse_judge_response("", items) == {}
