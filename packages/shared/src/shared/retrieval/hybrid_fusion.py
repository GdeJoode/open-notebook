"""Pure hybrid-fusion ranker for source↔source retrieval (Track R.3).

This module fuses the two heterogeneous retrieval signals into ONE ranking:

- the **dense** signal (R.1): cosine similarity ``∈ [0, 1]`` between aggregate
  source embeddings, and
- the **KG** signal (R.2): an *unbounded* weighted sum over shared active
  canonical entities (``type_salience × IDF``), e.g. ``2.5..4.9`` on staging.

The two are on DIFFERENT scales, so naively adding raw scores would let the
KG signal's larger magnitude silently dominate (or, after a clumsy rescale,
let outliers swing the order). We therefore fuse by **Reciprocal Rank Fusion
(RRF)** — scale-independent because it consumes each signal's *rank*, not its
raw score::

    fused(doc) = Σ_signal   w_signal · 1 / (k + rank_signal(doc))

with ``rank`` 1-based (best = 1). RRF is the standard, robust hybrid-retrieval
combiner (Cormack et al. 2009): it is insensitive to the heterogeneous scales,
degrades gracefully when one signal is missing, and exposes a single intuitive
knob per signal (the weight). The user steer — "the KG must play a LARGE role
in search" — is honoured by a **KG-prominent default**: ``w_kg ≥ w_dense`` (see
:data:`KG_HEAVY` / :data:`BALANCED` and the locked Decision in
``docs/tracks/R-hybrid-search/plan.md``).

Why RRF over min-max / z-score normalize-then-sum
-------------------------------------------------
Normalize-then-weighted-sum is the obvious alternative, but it is more
sensitive to the very scale heterogeneity we have here: min-max is dominated by
the extreme values (one outlier KG score compresses every other source toward
0), and z-score assumes a roughly comparable spread the two signals do not
share. RRF sidesteps both by discarding magnitudes entirely and ranking on
ordinal position — exactly the property we want when one signal is bounded
``[0, 1]`` and the other is an unbounded sum. The cost is that RRF cannot
reward a *runaway* top hit more than a merely-good one within a signal; for
source-linking (where "is B in the same cluster as Q?" matters more than
"by exactly how much") that trade-off is acceptable and was chosen
deliberately.

Missing-signal handling
-----------------------
A source present in only ONE signal must still rank (the carry-in AC: never
dropped). RRF makes this natural: a doc absent from a signal simply contributes
``0`` from that signal (it has no rank there), so its fused score is its single
signal's RRF term. We do NOT impute a phantom worst-rank; absence contributes
nothing rather than a small positive term, which keeps a one-signal source
strictly below an otherwise-equal two-signal source — the desired bias toward
corroborated results — while still ranking it ahead of nothing.

Provenance
----------
Every fused result carries, per signal, the raw score, the 1-based rank, and
that signal's exact contribution to the fused total (``w · 1/(k+rank)``), plus
the KG driving-entities explanation passed through from R.2. This is the
"why matched" lineage (the Purview linking-explanation lesson; feeds R.5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# RRF constant.
# ---------------------------------------------------------------------------
# The ``k`` in ``1/(k + rank)`` damps the influence of very high ranks so the
# top few results don't dominate purely on ordinal position. 60 is the value
# from the original RRF paper (Cormack, Clarke & Buettcher, SIGIR 2009) and the
# de-facto default in hybrid-search systems (Elasticsearch, OpenSearch). It is
# surfaced as a parameter so the ablation harness can probe its effect, but the
# default matches the literature.
DEFAULT_RRF_K: int = 60


@dataclass(frozen=True)
class FusionWeights:
    """Per-signal RRF weights + the RRF ``k`` constant (the tunable config).

    Attributes:
        dense: Weight on the dense (cosine) signal's RRF term.
        kg: Weight on the KG-proximity signal's RRF term. KG-prominent presets
            keep ``kg >= dense`` per the locked Track R decision.
        rrf_k: The ``k`` in ``1/(k + rank)``. Defaults to :data:`DEFAULT_RRF_K`.
    """

    dense: float = 1.0
    kg: float = 1.0
    rrf_k: int = DEFAULT_RRF_K

    @property
    def kg_prominent(self) -> bool:
        """True when the KG weight is at least the dense weight (the steer)."""
        return self.kg >= self.dense


# ---------------------------------------------------------------------------
# Named presets (the config surface — AC5).
# ---------------------------------------------------------------------------
# KG-PROMINENT default per the locked Decision: the KG signal must play a large
# role, so its weight is >= dense in every shipped preset, and the default IS
# the kg-heavy preset.
KG_HEAVY = FusionWeights(dense=1.0, kg=3.0)
"""KG-heavy preset: KG weight 3× dense. The DEFAULT — honours the user steer."""

BALANCED = FusionWeights(dense=1.0, kg=1.0)
"""Balanced preset: equal weight to both signals (still KG-prominent: kg==dense)."""

PRESETS: Dict[str, FusionWeights] = {
    "kg-heavy": KG_HEAVY,
    "balanced": BALANCED,
}

DEFAULT_PRESET = "kg-heavy"


def get_preset(name: Optional[str]) -> FusionWeights:
    """Resolve a preset name to its :class:`FusionWeights` (default kg-heavy).

    ``None`` / empty / unknown names fall back to the KG-heavy default rather
    than raising — the ranker must never fail closed on a bad query param; an
    unknown preset just gets the safe, steer-honouring default.
    """
    if not name:
        return PRESETS[DEFAULT_PRESET]
    return PRESETS.get(str(name).strip().lower(), PRESETS[DEFAULT_PRESET])


@dataclass(frozen=True)
class SignalProvenance:
    """One signal's contribution to a fused result (the per-signal "why").

    Attributes:
        present: Whether the source appeared in this signal at all.
        score: The raw signal score (cosine for dense; weighted sum for KG).
            ``None`` when the source is absent from this signal.
        rank: The 1-based rank within this signal (1 = best). ``None`` when
            absent.
        contribution: This signal's exact additive term in the fused score
            (``weight · 1/(rrf_k + rank)``), ``0.0`` when absent.
    """

    present: bool
    score: Optional[float]
    rank: Optional[int]
    contribution: float


@dataclass
class FusedResult:
    """A fused, ranked source with full per-signal provenance.

    Attributes:
        source_id: The candidate source id.
        title: Source title (hydrated by the service; ``None`` in pure use).
        fused_score: The total RRF score (sum of signal contributions).
        dense: Dense-signal provenance (score, rank, contribution).
        kg: KG-signal provenance (score, rank, contribution).
        kg_entities: The KG driving-entities explanation passed through from
            R.2 (each a dict ``{entity_id, name, type, document_frequency,
            weight, via_relation}``). Empty when the source had no KG signal.
    """

    source_id: str
    title: Optional[str]
    fused_score: float
    dense: SignalProvenance
    kg: SignalProvenance
    kg_entities: List[Dict[str, Any]] = field(default_factory=list)


def _rank_map(
    ranked: Sequence[Tuple[str, float]],
) -> Dict[str, Tuple[int, float]]:
    """Map source id -> (1-based rank, raw score) for a single signal's list.

    The input is assumed already sorted best-first (R.1 returns cosine DESC,
    R.2 returns KG-score DESC). First occurrence wins on a duplicate id so a
    malformed list never inflates a rank.
    """
    out: Dict[str, Tuple[int, float]] = {}
    rank = 0
    for sid, score in ranked:
        sid = str(sid)
        if sid in out:
            continue
        rank += 1
        out[sid] = (rank, float(score))
    return out


def _rrf_term(weight: float, rrf_k: int, rank: Optional[int]) -> float:
    """RRF contribution ``weight / (rrf_k + rank)``; ``0`` when rank is absent.

    A source missing from a signal contributes nothing from it (not a phantom
    worst-rank term) — see the module docstring on missing-signal handling.
    """
    if rank is None:
        return 0.0
    return float(weight) / (float(rrf_k) + float(rank))


def fuse_rankings(
    dense: Sequence[Tuple[str, float]],
    kg: Sequence[Tuple[str, float, List[Dict[str, Any]]]],
    *,
    weights: FusionWeights = KG_HEAVY,
    titles: Optional[Dict[str, Optional[str]]] = None,
    limit: Optional[int] = None,
) -> List[FusedResult]:
    """Fuse the dense + KG ranked lists into one RRF ranking (pure, no I/O).

    Args:
        dense: The dense signal as ``[(source_id, cosine_score), ...]``, already
            sorted best-first (R.1's ``find_related_by_embedding`` output).
        kg: The KG signal as ``[(source_id, kg_score, explanation), ...]``,
            already sorted best-first (R.2's ``score_related_sources`` output,
            with ``explanation`` the driving-entities lineage). ``explanation``
            may be an empty list.
        weights: The RRF weights + ``k`` constant. Defaults to the KG-heavy
            preset (the steer-honouring default). Pass :data:`BALANCED` or a
            custom :class:`FusionWeights` to tune; a higher ``kg`` weight
            provably reorders KG-favoured sources upward.
        titles: Optional ``{source_id: title}`` hydration map. Sources without
            an entry get ``title=None``.
        limit: Optional cap on results returned (the top ``limit`` fused).

    Returns:
        A list of :class:`FusedResult` sorted by ``fused_score`` descending with
        a deterministic ``source_id`` ascending tie-break. The union of both
        signals' sources is returned — a source present in only ONE signal
        still appears (never dropped), ranked by its single signal's RRF term.
    """
    titles = titles or {}

    dense_ranks = _rank_map([(sid, sc) for sid, sc in dense])
    kg_ranks: Dict[str, Tuple[int, float]] = {}
    kg_explanations: Dict[str, List[Dict[str, Any]]] = {}
    seen_kg = 0
    for entry in kg:
        sid = str(entry[0])
        score = float(entry[1])
        explanation = list(entry[2]) if len(entry) > 2 and entry[2] else []
        if sid in kg_ranks:
            continue
        seen_kg += 1
        kg_ranks[sid] = (seen_kg, score)
        kg_explanations[sid] = explanation

    all_ids = set(dense_ranks) | set(kg_ranks)

    results: List[FusedResult] = []
    for sid in all_ids:
        d_rank_score = dense_ranks.get(sid)
        k_rank_score = kg_ranks.get(sid)

        d_rank = d_rank_score[0] if d_rank_score else None
        d_score = d_rank_score[1] if d_rank_score else None
        k_rank = k_rank_score[0] if k_rank_score else None
        k_score = k_rank_score[1] if k_rank_score else None

        d_contrib = _rrf_term(weights.dense, weights.rrf_k, d_rank)
        k_contrib = _rrf_term(weights.kg, weights.rrf_k, k_rank)
        fused_score = d_contrib + k_contrib

        results.append(
            FusedResult(
                source_id=sid,
                title=titles.get(sid),
                fused_score=fused_score,
                dense=SignalProvenance(
                    present=d_rank is not None,
                    score=d_score,
                    rank=d_rank,
                    contribution=d_contrib,
                ),
                kg=SignalProvenance(
                    present=k_rank is not None,
                    score=k_score,
                    rank=k_rank,
                    contribution=k_contrib,
                ),
                kg_entities=kg_explanations.get(sid, []),
            )
        )

    results.sort(key=lambda r: (-r.fused_score, r.source_id))
    if limit is not None:
        results = results[: max(int(limit), 0)]
    return results
