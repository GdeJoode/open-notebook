"""Tests for the Track N.1 pre-LLM candidate layer.

Deterministic + spaCy-free: the spaCy path is exercised via a stub ``nlp`` object,
the fallback + TF-IDF + merge are pure. No model download needed.
"""

from __future__ import annotations

import re

from ontology_extraction.candidates import (
    Candidate,
    domain_ner_candidates,
    extract_candidates,
    noun_phrase_candidates,
    tfidf_salient_terms,
)


class _NC:
    def __init__(self, text: str) -> None:
        self.text = text


class _Doc:
    def __init__(self, phrases, ents=None) -> None:
        self.noun_chunks = [_NC(p) for p in phrases]
        self.ents = [_NC(e) for e in (ents or [])]


class _Ruler:
    def add_patterns(self, patterns) -> None:  # spaCy EntityRuler shape
        pass


class _FakeNlp:
    """Minimal spaCy-shaped stub: callable → doc with .noun_chunks / .ents,
    plus the EntityRuler wiring (pipe_names / add_pipe) the domain stub touches."""

    def __init__(self, phrases, ents=None) -> None:
        self._phrases = phrases
        self._ents = ents or []
        self.pipe_names: list[str] = []

    def add_pipe(self, name: str) -> _Ruler:
        self.pipe_names.append(name)
        return _Ruler()

    def __call__(self, text: str) -> _Doc:
        return _Doc(self._phrases, self._ents)


# -- TF-IDF -----------------------------------------------------------------


def test_tfidf_needs_at_least_two_corpus_chunks():
    assert tfidf_salient_terms("hello world", ["hello world"]) == []


def test_tfidf_surfaces_distinctive_terms_drops_common():
    corpus = [
        "gemeente beleid indicator subsidie",
        "gemeente beleid subsidie programma",
        "quantum entanglement gemeente",
    ]
    terms = dict(tfidf_salient_terms("quantum entanglement gemeente", corpus, top_k=5))
    assert "quantum" in terms and "entanglement" in terms
    # 'gemeente' appears in every corpus chunk → idf≈0 → not salient
    assert "gemeente" not in terms


# -- noun-phrase (fallback + spaCy stub) ------------------------------------


def test_regex_noun_phrase_fallback():
    got = noun_phrase_candidates(
        'The Regio Deal met "audit trail" en Ministerie van Economische Zaken.'
    )
    joined = " | ".join(got)
    assert "Regio Deal" in joined
    assert "audit trail" in joined  # quoted term


def test_spacy_path_uses_noun_chunks_when_nlp_injected():
    nlp = _FakeNlp(["audit trail management", "backup service"])
    assert noun_phrase_candidates("irrelevant", nlp=nlp) == [
        "audit trail management",
        "backup service",
    ]


# -- domain-NER stub (inert in N.1) -----------------------------------------


def test_domain_ner_inert_unless_enabled_and_populated():
    assert domain_ner_candidates("Regio Deal", enabled=False, patterns=None) == []
    assert domain_ner_candidates("Regio Deal", enabled=True, patterns=[]) == []
    # enabled + patterns + a stub ruler → returns the stub's ents
    nlp = _FakeNlp([], ents=["Regio Deal"])
    got = domain_ner_candidates(
        "Regio Deal",
        enabled=True,
        patterns=[{"label": "DEAL", "pattern": "Regio Deal"}],
        nlp=nlp,
    )
    assert got == ["Regio Deal"]


# -- merge / rank / dedup / cap ---------------------------------------------


def test_extract_merges_ranks_dedups_and_filters():
    nlp = _FakeNlp(["Audit Trail", "audit trail", "Backup Service"])
    corpus = ["audit trail service extra", "backup restore config", "quantum note here"]
    cands = extract_candidates(
        "Audit Trail and the Backup Service handle 12345 items",
        corpus_chunks=corpus,
        top_k=10,
        nlp=nlp,
    )
    texts = [c.text.lower() for c in cands]
    # case-insensitive dedup
    assert texts.count("audit trail") == 1
    # pure numbers are not candidates
    assert not any(re.fullmatch(r"\d+", t) for t in texts)
    # a multi-word phrase outranks a bare tfidf unigram
    assert cands[0].source in ("noun_chunk", "domain_ner")
    assert all(isinstance(c, Candidate) for c in cands)


def test_top_k_cap_and_empty_input():
    nlp = _FakeNlp([f"Phrase Number {i}" for i in range(50)])
    cands = extract_candidates("x", corpus_chunks=["a", "b"], top_k=7, nlp=nlp)
    assert len(cands) <= 7
    assert extract_candidates("") == []


def test_contractions_do_not_produce_spurious_quoted_anchors():
    # Straight apostrophes in contractions must NOT be read as quote delimiters
    # (the N.1-review MINOR): only double-quoted spans are candidates.
    got = noun_phrase_candidates("They don't like it's style but \"Audit Trail\" is real.")
    joined = " | ".join(got)
    assert "Audit Trail" in joined  # the real double-quoted span survives
    assert not any("'" in g and len(g.split()) > 1 and g[0].islower() for g in got)
    # no spurious lowercase apostrophe-run like "t like it"
    assert "t like it" not in joined


def test_real_spacy_noun_chunks_when_available():
    # Runs the REAL spaCy path where the model is installed (CI/container); skips
    # where it isn't (local WSL: compiled-extension install blocked by /mnt I/O).
    import pytest

    from ontology_extraction.candidates import _load_spacy

    _load_spacy.cache_clear()
    nlp = _load_spacy()
    if nlp is None:
        pytest.skip("spaCy/en_core_web_sm not functional in this env")
    got = noun_phrase_candidates(
        "The Regio Deal is a policy instrument for the Ministry.", nlp=nlp
    )
    # spaCy noun-chunks should surface at least one multi-word phrase.
    assert any(len(g.split()) >= 2 for g in got)
