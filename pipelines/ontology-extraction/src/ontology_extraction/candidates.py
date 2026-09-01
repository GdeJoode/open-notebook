"""Pre-LLM candidate/evidence extraction (Track N.1).

Distils a chunk into a small set of salient CANDIDATE terms BEFORE the Pass-2
LLM call, so the prompt can carry them as *precision anchors* ("make sure you did
not miss these; extract everything else too; ignore any that are spurious"). This
does NOT replace the LLM's exhaustive extraction — it only nudges recall on the
terms cheap NLP can already surface with high confidence.

Three deterministic-first sources are merged (Decision N-D2):

1. **TF-IDF salience** over the source's OWN chunks (pure stdlib — no model): the
   terms distinctive to THIS chunk vs. the rest of the document.
2. **Noun-phrase candidates**: spaCy ``en_core_web_sm`` noun-chunks when the model
   is installed (the core linguistic source, default ON), else a dependency-free
   regex fallback (Title-Case runs + quoted terms). spaCy is lazy-loaded and
   guarded — a missing library/model degrades to the fallback, never crashes.
3. **Domain-NER stub** (spaCy rule-based ``EntityRuler``): a wired-but-inert
   gazetteer hook — returns nothing until a domain ``patterns`` list is supplied
   and it is explicitly enabled. This is the seam a future domain term list plugs
   into; N.1 ships it empty.

All pure/deterministic given the same input (spaCy noun-chunks included), so the
whole layer is fast to unit-test without the model.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple

from loguru import logger

# Language-split stopword sets — kept separate so a cheap NL-vs-EN language guess
# (:func:`_detect_lang`) can pick the right spaCy model per document, while their
# union still guards TF-IDF salience against function words (the corpus is
# Dutch/English mixed: Gemeente, Ministerie, …).
_NL_STOPWORDS = frozenset(
    """
    de het een van en in op te dat die deze der den des aan met voor is was zijn
    er om ook naar bij uit als maar of dan door over onder tussen tot per wij zij
    """.split()
)
_EN_STOPWORDS = frozenset(
    """
    the a an of and in on to that this these for is are was were be by with from at
    as but or than then it its their your our we they he she his her not no yes
    """.split()
)
_STOPWORDS = _NL_STOPWORDS | _EN_STOPWORDS

# A candidate must be 2..60 chars and not a pure number / single stopword.
_MIN_LEN = 2
_MAX_LEN = 60
_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ0-9'’-]+")
# Title-case runs (proper-noun phrases) — a capitalized word followed by runs of
# (another capitalized word | a lowercase connector). The regex noun-phrase
# fallback when spaCy is unavailable. Captures "Regio Deal", "Ministerie van
# Economische Zaken"; a leading article ("The") is harmless noise dropped later.
_TITLECASE_RUN_RE = re.compile(
    r"\b[A-ZÀ-Ý][a-zà-ÿ]+"
    # {0,7} (not {0,5}) so a long article-led title like "De Minister van
    # Volkshuisvesting en Ruimtelijke Ordening" is captured whole — a leading
    # article consumes a slot, and _strip_edge_words trims it back afterwards.
    r"(?:\s+(?:van|de|der|den|voor|en|of|the|het|[A-ZÀ-Ý][a-zà-ÿ]+)){0,7}"
)
# Only DOUBLE quotes (straight + curly) — the straight single quote is
# overwhelmingly a contraction/possessive apostrophe in prose ("don't", "it's"),
# so matching it as a quote delimiter over-captured spurious spans.
_QUOTED_RE = re.compile(r"[\"“”]([^\"“”]{2,60})[\"“”]")


@dataclass(frozen=True)
class Candidate:
    """One pre-LLM candidate term with its provenance + salience score."""

    text: str
    source: str  # "tfidf" | "noun_chunk" | "domain_ner"
    score: float = 0.0


# ---------------------------------------------------------------------------
# TF-IDF salience (pure stdlib)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "")]


def tfidf_salient_terms(
    chunk_text: str,
    corpus_chunks: Sequence[str],
    *,
    top_k: int = 12,
) -> List[Tuple[str, float]]:
    """Terms distinctive to ``chunk_text`` vs. the rest of the source.

    ``tf * log(N / (1 + df))`` over the source's own chunks. Needs ≥2 corpus
    chunks to have any discriminative power; with fewer it returns ``[]`` (the
    noun-phrase source still supplies anchors). Deterministic.
    """
    corpus = [c for c in corpus_chunks if c and c.strip()]
    if len(corpus) < 2:
        return []
    n_docs = len(corpus)
    # Document frequency across the corpus.
    df: dict[str, int] = {}
    for doc in corpus:
        for term in set(_tokenize(doc)):
            df[term] = df.get(term, 0) + 1

    tokens = _tokenize(chunk_text)
    if not tokens:
        return []
    total = len(tokens)
    tf: dict[str, int] = {}
    for term in tokens:
        tf[term] = tf.get(term, 0) + 1

    scored: List[Tuple[str, float]] = []
    for term, count in tf.items():
        if term in _STOPWORDS or len(term) < _MIN_LEN or term.isdigit():
            continue
        idf = math.log(n_docs / (1 + df.get(term, 0)))
        if idf <= 0:
            continue
        scored.append((term, (count / total) * idf))
    scored.sort(key=lambda kv: kv[1], reverse=True)
    return scored[:top_k]


# ---------------------------------------------------------------------------
# Noun-phrase candidates (spaCy noun-chunks, else regex fallback)
# ---------------------------------------------------------------------------


# spaCy model per detected language. Both are URL-pinned deps + lazy-loaded.
_SPACY_MODEL = {"nl": "nl_core_news_sm", "en": "en_core_web_sm"}


def _detect_lang(text: str) -> str:
    """Cheap NL-vs-EN language guess by stopword ratio (dependency-free).

    Picks the spaCy model for the document. Ties / no signal → ``"en"``. Robust
    enough for a whole document/chunk; the regex source (which runs alongside
    spaCy regardless) doesn't care about language.
    """
    tokens = _tokenize(text)
    if not tokens:
        return "en"
    nl = sum(1 for t in tokens if t in _NL_STOPWORDS)
    en = sum(1 for t in tokens if t in _EN_STOPWORDS)
    return "nl" if nl > en else "en"


@lru_cache(maxsize=2)
def _load_spacy(lang: str = "en") -> Optional[Any]:
    """Lazy-load the spaCy model for ``lang`` (nl/en); return None if unavailable.

    Guarded so a missing library OR model degrades to the regex source — the
    extraction path must never crash on a parsing dependency (Decision N-D2).
    Cached per language (maxsize=2 → both models load once per process). NOTE: the
    cache also pins a transient load FAILURE for the process lifetime — acceptable
    because spaCy availability doesn't change mid-process; a deploy that installs a
    model takes effect on the next process.
    """
    model = _SPACY_MODEL.get(lang, "en_core_web_sm")
    try:
        import spacy  # type: ignore
    except Exception as exc:  # noqa: BLE001 — library not installed → fallback
        logger.debug("candidates: spaCy not importable ({e}); using regex source", e=exc)
        return None
    try:
        # Disable NER + lemmatizer — we only want the parser's noun-chunks; the
        # EntityRuler (domain NER) is added explicitly when enabled.
        return spacy.load(model, disable=["ner", "lemmatizer"])
    except Exception as exc:  # noqa: BLE001 — model not downloaded → fallback
        logger.info(
            "candidates: spaCy model {m!r} unavailable ({e}); using regex source",
            m=model,
            e=exc,
        )
        return None


def _regex_noun_phrases(text: str) -> List[str]:
    """Dependency-free noun-phrase fallback: Title-Case runs + quoted terms."""
    out: List[str] = []
    for m in _TITLECASE_RUN_RE.finditer(text or ""):
        out.append(m.group(0))
    for m in _QUOTED_RE.finditer(text or ""):
        out.append(m.group(1))
    return out


def noun_phrase_candidates(
    text: str, *, nlp: Optional[Any] = None, lang: Optional[str] = None
) -> List[str]:
    """Noun-phrase candidates: spaCy noun-chunks + regex Title-Case runs, MERGED.

    The two sources are COMPLEMENTARY (live-validated on Dutch policy docs): the
    per-language spaCy model reads grammar (generic phrases like "de gemeente")
    but FRAGMENTS long compound proper names, while the regex captures the long
    Title-Case runs ("Minister van Volkshuisvesting en Ruimtelijke Ordening")
    spaCy splits. Running BOTH (dedup'd downstream) gives the union. spaCy loads
    the model for ``lang`` (nl/en; auto-detected when None). ``nlp`` may be
    injected (tests). When spaCy is unavailable the regex source stands alone.
    Deterministic, order-stable.
    """
    phrases: List[str] = []
    engine = nlp if nlp is not None else _load_spacy(lang or _detect_lang(text))
    if engine is not None:
        try:
            phrases.extend(nc.text for nc in engine(text or "").noun_chunks)
        except Exception as exc:  # noqa: BLE001 — spaCy hiccup → regex source only
            logger.warning(
                "candidates: spaCy noun_chunks failed ({e}); regex source only", e=exc
            )
    phrases.extend(_regex_noun_phrases(text))
    return phrases


# ---------------------------------------------------------------------------
# Domain-NER stub (spaCy EntityRuler gazetteer) — wired but inert in N.1
# ---------------------------------------------------------------------------


def domain_ner_candidates(
    text: str,
    *,
    patterns: Optional[List[dict]] = None,
    enabled: bool = False,
    nlp: Optional[Any] = None,
    lang: Optional[str] = None,
) -> List[str]:
    """Domain entities via a rule-based spaCy ``EntityRuler`` gazetteer (stub).

    Returns ``[]`` unless BOTH ``enabled`` and a non-empty ``patterns`` list are
    supplied — the N.1 gazetteer is empty, so this is inert. The wiring exists so
    a future domain term list (products, standards, domain entities) plugs in via
    ``EXTRACTION_DOMAIN_NER_ENABLED`` + a ``domain_patterns`` file, WITHOUT a code
    change here (Decision N-D2). Deterministic (rule-based, no statistical NER).
    """
    if not enabled or not patterns:
        return []
    engine = nlp if nlp is not None else _load_spacy(lang or _detect_lang(text))
    if engine is None:
        return []
    try:
        if "entity_ruler" not in engine.pipe_names:
            ruler = engine.add_pipe("entity_ruler")
            ruler.add_patterns(patterns)
        doc = engine(text or "")
        return [ent.text for ent in doc.ents]
    except Exception as exc:  # noqa: BLE001 — ruler unavailable → no domain hits
        logger.warning("candidates: EntityRuler stub failed ({e})", e=exc)
        return []


# ---------------------------------------------------------------------------
# Merge → ranked candidate list
# ---------------------------------------------------------------------------


# Articles/connectors that are noise at the START or END of a candidate phrase
# (a leading "De Minister…" or a trailing "…Regio Deal de"). Stripped so the
# anchor is the entity itself. Live-validated on the Regio Deal convenants where
# the regex fallback otherwise emitted "De Partners en de" / "Regio Deal de".
_EDGE_WORDS = frozenset(
    {"de", "het", "een", "the", "a", "an", "van", "der", "den", "en", "of",
     "voor", "in", "op", "and", "or"}
)


def _strip_edge_words(text: str) -> str:
    """Drop leading/trailing articles + connectors from a phrase (case-insensitive)."""
    words = text.split()
    while words and words[0].lower() in _EDGE_WORDS:
        words.pop(0)
    while words and words[-1].lower() in _EDGE_WORDS:
        words.pop()
    return " ".join(words)


def _normalize(text: str) -> str:
    return _strip_edge_words(re.sub(r"\s+", " ", (text or "").strip()))


def _acceptable(text: str) -> bool:
    t = _normalize(text)
    if not (_MIN_LEN <= len(t) <= _MAX_LEN):
        return False
    if t.lower() in _STOPWORDS:
        return False
    if re.fullmatch(r"[\d\s.,%€$-]+", t):  # pure number / amount → not a concept
        return False
    return True


def extract_candidates(
    chunk_text: str,
    *,
    corpus_chunks: Optional[Sequence[str]] = None,
    top_k: int = 20,
    domain_ner_enabled: bool = False,
    domain_patterns: Optional[List[dict]] = None,
    nlp: Optional[Any] = None,
) -> List[Candidate]:
    """Merge TF-IDF + noun-phrase + (stub) domain-NER candidates, ranked, capped.

    Order-stable and dedup'd by normalized lowercase text. Noun-phrases and
    domain hits are weighted above bare TF-IDF unigrams (a multi-word phrase is a
    stronger anchor). Returns at most ``top_k`` candidates.
    """
    corpus = list(corpus_chunks) if corpus_chunks is not None else [chunk_text]
    # Detect the document language ONCE (over the whole corpus for stability) so
    # spaCy picks the nl/en model consistently across the source's chunks.
    lang = _detect_lang(" ".join(corpus)[:20000]) if corpus else "en"

    merged: dict[str, Candidate] = {}

    def _add(text: str, source: str, score: float) -> None:
        t = _normalize(text)
        if not _acceptable(t):
            return
        key = t.lower()
        prev = merged.get(key)
        # Keep the highest-scoring provenance; phrases (noun_chunk/domain) win ties.
        if prev is None or score > prev.score:
            merged[key] = Candidate(text=t, source=source, score=score)

    # Noun-phrases first (strongest anchors) — a small length bonus favours
    # specific multi-word terms over single tokens.
    for phrase in noun_phrase_candidates(chunk_text, nlp=nlp, lang=lang):
        merged_len = len(_normalize(phrase).split())
        _add(phrase, "noun_chunk", 1.0 + 0.1 * min(merged_len, 5))
    for hit in domain_ner_candidates(
        chunk_text,
        patterns=domain_patterns,
        enabled=domain_ner_enabled,
        nlp=nlp,
        lang=lang,
    ):
        _add(hit, "domain_ner", 2.0)  # a gazetteer hit is the strongest signal
    for term, score in tfidf_salient_terms(chunk_text, corpus):
        _add(term, "tfidf", score)  # bare unigram salience, below phrases

    ranked = sorted(merged.values(), key=lambda c: c.score, reverse=True)
    return ranked[:top_k]


__all__ = [
    "Candidate",
    "extract_candidates",
    "tfidf_salient_terms",
    "noun_phrase_candidates",
    "domain_ner_candidates",
]
