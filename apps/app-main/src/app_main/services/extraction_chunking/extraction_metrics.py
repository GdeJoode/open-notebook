"""Over-generation & abstention measurement for typed extraction (Track N.3).

The falsifiable gate for the N.3 not-a-concept filter + abstention prompt: given
the raw counts ``run_pass2`` records in its result metadata, derive two rates —

* ``over_generation_rate`` = ``1 − survivors / extracted`` — the share of LLM-
  emitted entities the deterministic + judge not-a-concept gate rejected as page-
  furniture. A pre-filter that does its job pushes this ABOVE 0 without dropping
  golden entities (the N.5 regression gate asserts the recall floor separately).
* ``abstain_rate`` = ``abstained_chunks / total_chunks`` — the share of chunks
  that yielded no entity at all (pure boilerplate / no domain content). The
  abstention prompt should push this above 0 on furniture-heavy corpora instead
  of manufacturing plausible-but-empty concepts.

Pure functions over the counts — no LLM, no DB — so this doubles as a CI gate the
N.5 heterogeneous-corpus test can assert on directly, exactly like M.5's
``chunking_metrics.measure_packing``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class ExtractionMetrics:
    """Derived over-generation / abstention rates for one extraction run.

    * ``extracted`` — entities the LLM emitted, BEFORE the not-a-concept gate.
    * ``survivors`` — entities that passed the gate (what reaches downstream).
    * ``rejected`` — ``extracted − survivors`` (page-furniture the gate removed).
    * ``over_generation_rate`` — ``rejected / extracted`` ∈ [0, 1] (0 when the LLM
      emitted nothing).
    * ``abstained_chunks`` / ``total_chunks`` — chunks that produced no entity.
    * ``abstain_rate`` — ``abstained_chunks / total_chunks`` ∈ [0, 1].
    """

    extracted: int
    survivors: int
    rejected: int
    over_generation_rate: float
    abstained_chunks: int
    total_chunks: int
    abstain_rate: float


def over_generation_rate(extracted: int, survivors: int) -> float:
    """``1 − survivors / extracted``; 0.0 when nothing was extracted.

    Clamped to [0, 1] and guarded against ``survivors > extracted`` (which would
    signal a caller bug — the gate can only remove, never add).
    """
    if extracted <= 0:
        return 0.0
    survivors = max(0, min(survivors, extracted))
    return 1.0 - survivors / extracted


def abstain_rate(abstained_chunks: int, total_chunks: int) -> float:
    """``abstained_chunks / total_chunks``; 0.0 when there are no chunks."""
    if total_chunks <= 0:
        return 0.0
    abstained = max(0, min(abstained_chunks, total_chunks))
    return abstained / total_chunks


def measure_extraction(metadata: Mapping[str, Any]) -> ExtractionMetrics:
    """Derive :class:`ExtractionMetrics` from ``run_pass2`` result metadata.

    Reads ``entities_extracted``, ``entities_kept`` (falls back to
    ``total_entities``), ``abstained_chunks``, and ``chunk_count`` — all missing
    keys degrade to 0, so a run from before N.3 (no counts) measures as an all-
    zero, no-over-generation baseline rather than raising.
    """
    extracted = int(metadata.get("entities_extracted", 0) or 0)
    survivors = int(
        metadata.get("entities_kept", metadata.get("total_entities", 0)) or 0
    )
    total_chunks = int(metadata.get("chunk_count", 0) or 0)
    abstained = int(metadata.get("abstained_chunks", 0) or 0)
    survivors = max(0, min(survivors, extracted)) if extracted else survivors
    return ExtractionMetrics(
        extracted=extracted,
        survivors=survivors,
        rejected=max(0, extracted - survivors),
        over_generation_rate=over_generation_rate(extracted, survivors),
        abstained_chunks=abstained,
        total_chunks=total_chunks,
        abstain_rate=abstain_rate(abstained, total_chunks),
    )


__all__ = [
    "ExtractionMetrics",
    "over_generation_rate",
    "abstain_rate",
    "measure_extraction",
]
