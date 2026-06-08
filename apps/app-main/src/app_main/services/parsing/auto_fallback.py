"""Auto-fallback orchestrator: docling first, MinerU if confidence is low.

Implements the ``parser_engine="auto"`` branch of ``SourceExtractor``
(Phase A.1c). Wraps the dual-extraction control flow so it is testable
in isolation without going through the full extractor surface.

Decision flow (mirroring the plan):

1. Run Docling.
2. If Docling raises or returns ``success=False``: try MinerU. Return
   the MinerU result with ``engine_used="mineru"`` and
   ``decision="fallback"``.
3. Score the Docling result. If ``overall >= threshold``: keep Docling,
   ``decision="accept"``.
4. Else: run MinerU. Return MinerU with ``engine_used="mineru"`` and
   ``decision="fallback"`` (Q-A-2 — no comparative scoring of MinerU).

If MinerU itself fails after a Docling fallback decision, the Docling
result is returned as a degraded best-effort (with a WARNING log) so
that auto-mode never strands the user — the badge will say "docling"
and "fallback_triggered=true" but the source will still process.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

from loguru import logger
from shared.services.metrics import record_metric

from app_main.services.parsing.confidence import (
    DEFAULT_THRESHOLD,
    DoclingConfidenceScore,
    score_docling_extraction,
)


async def extract_with_auto_fallback(
    file_path: Path,
    *,
    docling_client: Any,
    mineru_client: Any,
    threshold: float = DEFAULT_THRESHOLD,
    source_id: Optional[str] = None,
    notebook_id: Optional[str] = None,
) -> Tuple[Any, str, DoclingConfidenceScore]:
    """Run docling, score it, fall back to MinerU when below ``threshold``.

    ``docling_client`` and ``mineru_client`` must both expose
    ``async def process(path: Path) -> IngestionResult``. They are
    injected so SourceExtractor (which already has them lazy-imported)
    can pass concrete instances and tests can pass mocks.

    Returns ``(chosen_result, engine_used, score)``:

    * ``chosen_result`` — the ``IngestionResult`` to persist downstream
      (docling on accept, MinerU on fallback, docling as last resort
      if MinerU also crashes).
    * ``engine_used`` — ``"docling"`` or ``"mineru"`` — the engine
      whose output is in ``chosen_result``.
    * ``score`` — the docling confidence score; ``decision`` field
      indicates whether the fallback fired.

    ``source_id`` / ``notebook_id`` are optional context for the
    telemetry hook (B.4 / RETRO #5). When supplied, one
    ``extraction.auto_fallback`` metric is emitted per call regardless
    of which engine wins — closes the gap where Track A's auto-fallback
    decisions only existed in Loguru. Defaults preserve backward-compat
    with existing callers + tests that don't yet know about telemetry.
    """
    file_path = Path(file_path)

    # Step 1: try Docling. Treat both exceptions and ``success=False``
    # as "Docling can't deal with this" and immediately try MinerU.
    docling_result, docling_failed = await _safe_run(
        docling_client, file_path, engine_label="docling"
    )

    if docling_failed:
        logger.warning(
            f"[auto-fallback] Docling failed for {file_path.name}; "
            f"falling back to MinerU"
        )
        mineru_result, mineru_failed = await _safe_run(
            mineru_client, file_path, engine_label="mineru"
        )
        if mineru_failed:
            # Both failed — re-raise the Docling error as the primary
            # since that's what the user's "auto" mode initially asked
            # for. ``_safe_run`` returned None for the result; bubble.
            # Telemetry intentionally not emitted here: a both-failed
            # outcome is the rare hard error path the caller handles
            # separately, and the source row will carry its own error.
            raise RuntimeError(
                f"Both Docling and MinerU failed for {file_path.name}"
            )

        score = _failed_docling_score(threshold)
        chosen, engine_used = mineru_result, "mineru"
    else:
        # Step 3: score the docling output.
        score = score_docling_extraction(docling_result, threshold=threshold)
        if score.decision == "accept":
            chosen, engine_used = docling_result, "docling"
        else:
            # Step 4: docling succeeded but confidence is below the bar.
            # Run MinerU and trust its output (Q-A-2).
            logger.info(
                f"[auto-fallback] Docling confidence {score.overall:.3f} < "
                f"{threshold:.3f} for {file_path.name}; trying MinerU"
            )
            mineru_result, mineru_failed = await _safe_run(
                mineru_client, file_path, engine_label="mineru"
            )
            if mineru_failed:
                # MinerU couldn't help. Return docling as best-effort so
                # the source still processes. The downstream badge will
                # read "docling" + "fallback_triggered=true" so the user
                # knows the confidence-driven fallback was attempted.
                logger.warning(
                    f"[auto-fallback] MinerU fallback also failed for "
                    f"{file_path.name}; keeping low-confidence docling "
                    "result"
                )
                chosen, engine_used = docling_result, "docling"
            else:
                chosen, engine_used = mineru_result, "mineru"

    # Single metric emit per call — closes RETRO #5 (Track A flying
    # blind on telemetry). ``record_metric`` is failure-tolerant so
    # this never crashes the parser. Context (source_id, notebook_id)
    # is optional; tests that don't supply it still exercise the path
    # and produce a context-less row (option<string> schema allows it).
    await record_metric(
        "extraction.auto_fallback",
        {
            "confidence": score.overall,
            "decision": score.decision,
            "engine_used": engine_used,
            "threshold": score.threshold,
        },
        source=source_id,
        notebook=notebook_id,
    )

    return chosen, engine_used, score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _safe_run(
    client: Any,
    file_path: Path,
    *,
    engine_label: str,
) -> Tuple[Optional[Any], bool]:
    """Call ``client.process(file_path)``, normalising both exception
    paths and ``IngestionResult.success=False`` into a single
    ``(result_or_none, failed: bool)`` return.

    A ``failed`` outcome is logged at WARNING; the caller decides
    whether to escalate (raise) or to try the other engine.
    """
    try:
        result = await client.process(file_path)
    except Exception as exc:
        logger.warning(
            f"[auto-fallback] {engine_label} raised for {file_path.name}: {exc}"
        )
        return None, True

    if not getattr(result, "success", True):
        logger.warning(
            f"[auto-fallback] {engine_label} reported failure for "
            f"{file_path.name}: {getattr(result, 'error_message', '')}"
        )
        return result, True

    return result, False


def _failed_docling_score(threshold: float) -> DoclingConfidenceScore:
    """Sentinel score for the "docling raised" branch.

    Confidence is 0.0 so the UI tooltip surfaces "docling failed" to
    the user; decision is ``fallback`` regardless of threshold.
    """
    return DoclingConfidenceScore(
        overall=0.0,
        signals={},
        decision="fallback",
        threshold=threshold,
    )


__all__ = ["extract_with_auto_fallback"]
