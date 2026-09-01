"""Track N.2 — Hearst is-a pattern miner (spaCy noun-chunk based).

The miner works on REAL spaCy noun-chunk boundaries + POS (no regex NP-guessing).
The core boundary/POS logic is exercised here with a token-level spaCy STUB — a
doc that yields tokens (``.text``/``.pos_``/``.is_space``), supports
``doc[start:stop]`` slicing, and exposes ``.noun_chunks`` spans
(``.start``/``.end``/``.text``) — so the edge cases (verb-gap stop, "other"-drop,
dedup) are fast, deterministic, and pinned to an exact token layout, with no
34 MB model download per run. One gated test below runs the REAL model where it
is installed (it does run under this repo's WSL /mnt venv).
"""

from __future__ import annotations

from ontology_extraction.candidates import mine_hearst_isa

# --- token-level spaCy stub ------------------------------------------------


class _Tok:
    def __init__(self, text: str, pos: str) -> None:
        self.text = text
        self.pos_ = pos
        self.is_space = False


class _Span:
    def __init__(self, doc: "_Doc", start: int, end: int) -> None:
        self._doc = doc
        self.start = start
        self.end = end

    @property
    def text(self) -> str:
        return " ".join(t.text for t in self._doc._toks[self.start : self.end])


class _Doc:
    def __init__(self, toks, chunks, sents=None) -> None:
        self._toks = toks
        self._chunks = chunks
        self._sents = sents

    def __iter__(self):
        return iter(self._toks)

    def __getitem__(self, s):
        return self._toks[s]

    @property
    def noun_chunks(self):
        return [_Span(self, a, b) for a, b in self._chunks]

    @property
    def sents(self):
        # Raise like a spaCy Doc with no sentence boundaries when unset, so the
        # miner exercises its whole-doc fallback; expose spans when provided.
        if self._sents is None:
            raise ValueError("sentence boundaries unset")
        return [_Span(self, a, b) for a, b in self._sents]


def _nlp(spec, chunks, sents=None):
    """spec = [(text, pos), …]; chunks/sents = [(start, end), …] → callable stub."""
    doc = _Doc([_Tok(t, p) for t, p in spec], chunks, sents)
    return lambda _text="": doc


# --- broad-FIRST: "<broad> such as <hyponyms…>" ----------------------------


def test_such_as_en():
    # components such as bolts and screws are cheap.
    nlp = _nlp(
        [
            ("components", "NOUN"), ("such", "ADJ"), ("as", "SCONJ"),
            ("bolts", "NOUN"), ("and", "CCONJ"), ("screws", "NOUN"),
            ("are", "AUX"), ("cheap", "ADJ"), (".", "PUNCT"),
        ],
        [(0, 1), (3, 4), (5, 6)],
    )
    got = mine_hearst_isa("components such as bolts and screws are cheap.", nlp=nlp)
    assert ("bolts", "components") in got
    assert ("screws", "components") in got


def test_zoals_nl():
    # instrumenten zoals subsidies en convenanten worden ingezet.
    nlp = _nlp(
        [
            ("instrumenten", "NOUN"), ("zoals", "SCONJ"), ("subsidies", "NOUN"),
            ("en", "CCONJ"), ("convenanten", "NOUN"), ("worden", "AUX"),
            ("ingezet", "VERB"), (".", "PUNCT"),
        ],
        [(0, 1), (2, 3), (4, 5)],
    )
    got = mine_hearst_isa("instrumenten zoals subsidies en convenanten.", nlp=nlp)
    assert ("subsidies", "instrumenten") in got
    assert ("convenanten", "instrumenten") in got


def test_list_stops_at_verb_gap():
    # tools such as hammers work and drills → the VERB "work" ends the list, so
    # only "hammers" is mined, never "drills" across the clause boundary.
    nlp = _nlp(
        [
            ("tools", "NOUN"), ("such", "ADJ"), ("as", "SCONJ"),
            ("hammers", "NOUN"), ("work", "VERB"), ("and", "CCONJ"),
            ("drills", "NOUN"),
        ],
        [(0, 1), (3, 4), (6, 7)],
    )
    got = mine_hearst_isa("tools such as hammers work and drills", nlp=nlp)
    assert ("hammers", "tools") in got
    assert ("drills", "tools") not in got


def test_including_and_especially():
    nlp_inc = _nlp(
        [("metals", "NOUN"), ("including", "VERB"), ("iron", "NOUN")],
        [(0, 1), (2, 3)],
    )
    assert ("iron", "metals") in mine_hearst_isa("metals including iron", nlp=nlp_inc)
    nlp_esp = _nlp(
        [("metals", "NOUN"), (",", "PUNCT"), ("especially", "ADV"), ("copper", "NOUN")],
        [(0, 1), (3, 4)],
    )
    assert ("copper", "metals") in mine_hearst_isa("metals, especially copper", nlp=nlp_esp)


# --- broad-LAST: "<hyponyms…> and other <broad>" ---------------------------


def test_and_other_en():
    # bolts and other fasteners → ("bolts","fasteners"); "other" is dropped off
    # the hypernym chunk, not treated as part of the type name.
    nlp = _nlp(
        [
            ("bolts", "NOUN"), ("and", "CCONJ"), ("other", "ADJ"),
            ("fasteners", "NOUN"),
        ],
        [(0, 1), (2, 4)],
    )
    assert ("bolts", "fasteners") in mine_hearst_isa("bolts and other fasteners", nlp=nlp)


def test_en_andere_nl():
    # gemeenten en andere overheden → ("gemeenten","overheden").
    nlp = _nlp(
        [
            ("gemeenten", "NOUN"), ("en", "CCONJ"), ("andere", "ADJ"),
            ("overheden", "NOUN"),
        ],
        [(0, 1), (2, 4)],
    )
    assert ("gemeenten", "overheden") in mine_hearst_isa(
        "gemeenten en andere overheden", nlp=nlp
    )


# --- clause / sentence bounding (N.2 review MAJOR) -------------------------


def test_broad_first_anchor_not_crossing_sentence():
    # "Governance matters here. Such as subsidies and grants were used." — the
    # hypernym must NOT be pulled from the previous sentence.
    nlp = _nlp(
        [
            ("Governance", "NOUN"), ("matters", "VERB"), ("here", "ADV"),
            (".", "PUNCT"), ("Such", "ADJ"), ("as", "SCONJ"),
            ("subsidies", "NOUN"), ("and", "CCONJ"), ("grants", "NOUN"),
            ("were", "AUX"), ("used", "VERB"), (".", "PUNCT"),
        ],
        [(0, 1), (6, 7), (8, 9)],
        sents=[(0, 4), (4, 12)],
    )
    assert mine_hearst_isa("Governance matters here. Such as subsidies…", nlp=nlp) == []


def test_broad_last_anchor_not_crossing_sentence():
    # "screws and other. Fasteners are common." — the hypernym must NOT be pulled
    # from the next sentence via the following-chunk fallback.
    nlp = _nlp(
        [
            ("screws", "NOUN"), ("and", "CCONJ"), ("other", "ADJ"), (".", "PUNCT"),
            ("Fasteners", "NOUN"), ("are", "AUX"), ("common", "ADJ"), (".", "PUNCT"),
        ],
        [(0, 1), (4, 5)],
        sents=[(0, 4), (4, 8)],
    )
    assert mine_hearst_isa("screws and other. Fasteners are common.", nlp=nlp) == []


def test_within_sentence_verb_blocks_anchor():
    # "Governance matters, such as subsidies." — a VERB between the noun and the
    # cue blocks the anchor even within one sentence (gap discipline). No .sents
    # here → the whole-doc fallback still applies the gap check.
    nlp = _nlp(
        [
            ("Governance", "NOUN"), ("matters", "VERB"), (",", "PUNCT"),
            ("such", "ADJ"), ("as", "SCONJ"), ("subsidies", "NOUN"), (".", "PUNCT"),
        ],
        [(0, 1), (5, 6)],
    )
    assert mine_hearst_isa("Governance matters, such as subsidies.", nlp=nlp) == []


def test_comma_before_cue_still_anchors():
    # "metals, such as iron" — a bare comma (PUNCT) between noun and cue is a clean
    # gap, so the anchor is still found (guards against over-blocking).
    nlp = _nlp(
        [
            ("metals", "NOUN"), (",", "PUNCT"), ("such", "ADJ"), ("as", "SCONJ"),
            ("iron", "NOUN"),
        ],
        [(0, 1), (4, 5)],
    )
    assert ("iron", "metals") in mine_hearst_isa("metals, such as iron", nlp=nlp)


def test_real_spacy_does_not_cross_sentences():
    # REAL model: the two cross-sentence layouts the N.2 review reproduced must
    # seed NO pair pulling a hypernym from a neighbouring sentence.
    import pytest
    from ontology_extraction.candidates import _load_spacy

    _load_spacy.cache_clear()
    if _load_spacy("en") is None:
        pytest.skip("spaCy/en_core_web_sm not functional in this env")
    got1 = mine_hearst_isa("Governance matters here. Such as subsidies and grants were used.")
    assert ("subsidies", "Governance") not in got1
    assert ("grants", "Governance") not in got1
    got2 = mine_hearst_isa("screws and other. Fasteners are common in industry.")
    assert ("screws", "Fasteners") not in got2


# --- precision / hygiene ---------------------------------------------------


def test_no_cue_no_pair():
    nlp = _nlp(
        [
            ("The", "DET"), ("minister", "NOUN"), ("signed", "VERB"),
            ("the", "DET"), ("convenant", "NOUN"), ("today", "NOUN"),
        ],
        [(0, 2), (3, 5)],
    )
    assert mine_hearst_isa("The minister signed the convenant today.", nlp=nlp) == []


def test_self_reference_excluded():
    # tools such as tools → narrow == broad → dropped.
    nlp = _nlp(
        [("tools", "NOUN"), ("such", "ADJ"), ("as", "SCONJ"), ("tools", "NOUN")],
        [(0, 1), (3, 4)],
    )
    assert mine_hearst_isa("tools such as tools", nlp=nlp) == []


def test_deduped():
    # metals such as iron and iron → same pair once.
    nlp = _nlp(
        [
            ("metals", "NOUN"), ("such", "ADJ"), ("as", "SCONJ"),
            ("iron", "NOUN"), ("and", "CCONJ"), ("iron", "NOUN"),
        ],
        [(0, 1), (3, 4), (5, 6)],
    )
    got = mine_hearst_isa("metals such as iron and iron", nlp=nlp)
    assert got.count(("iron", "metals")) == 1


def test_endpoints_are_edge_stripped():
    # the components such as the bolts → articles stripped from both endpoints.
    nlp = _nlp(
        [
            ("the", "DET"), ("components", "NOUN"), ("such", "ADJ"), ("as", "SCONJ"),
            ("the", "DET"), ("bolts", "NOUN"),
        ],
        [(0, 2), (4, 6)],
    )
    got = mine_hearst_isa("the components such as the bolts", nlp=nlp)
    assert ("bolts", "components") in got


def test_no_spacy_returns_empty():
    # No model available and no injected nlp → [] (no garbage regex fallback).
    import ontology_extraction.candidates as c

    orig = c._load_spacy
    c._load_spacy = lambda lang="en": None
    try:
        assert c.mine_hearst_isa("components such as bolts and screws") == []
    finally:
        c._load_spacy = orig


def test_real_spacy_when_available():
    # REAL model path — runs wherever the model is installed (this repo's WSL
    # /mnt venv included); skips only where the model is genuinely absent.
    import pytest
    from ontology_extraction.candidates import _load_spacy

    _load_spacy.cache_clear()
    if _load_spacy("en") is None:
        pytest.skip("spaCy/en_core_web_sm not functional in this env")
    got = mine_hearst_isa("policy instruments such as subsidies and covenants.")
    hypernyms = {b.lower() for _, b in got}
    assert any("instrument" in h for h in hypernyms)
