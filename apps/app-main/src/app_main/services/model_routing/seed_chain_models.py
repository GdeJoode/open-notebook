"""Idempotent startup backfill of chain-model context params (Track M.1).

The ``model`` record already carries ``context_window`` and ``max_output_tokens``
(no migration needed). They are nullable and were unpopulated for some
failover-chain models. Track M's context-aware chunk packer
(``extraction_chunking.context_packer``) reads ``context_window`` from the
active model to size its LLM-call windows, so an unpopulated value forces the
packer into its conservative default and wastes a big model's context.

This backfill fills those nulls — **fill-nulls-only**, so an operator who has
manually tuned a value is never overwritten (the idempotency contract). It is
best-effort and non-blocking: a failure is logged and swallowed so a seed hiccup
never blocks API startup (the packer's null-context degrade is the safety net).

Per-model values (sourced from provider docs; operator-overridable via the model
row — Decision M-D5):

  * NVIDIA NIM mistral-medium  context 128_000, output 16_384 (the J.4 seed
    already sets these; re-asserted here only if null).
  * Local Ollama llama3.1:8b   context **8_192** — deliberately the model's
    SAFE effective window, NOT its 128K theoretical max. Ollama's ``num_ctx``
    defaults to ~2-4K and is the BINDING constraint for the fallback; an 8_192
    floor keeps a pack sized for this candidate within reach of the local
    runtime (Decision: cap the model row's context_window low rather than rely
    on a runtime num_ctx the seed cannot observe — see M.4 guard). output 4_096.
  * Mistral-large-3 (256K) / Ministral (131072): seeded elsewhere; left as-is.
"""

from __future__ import annotations

from typing import Optional

from loguru import logger

# Conservative SAFE context window for a local Ollama model. NOT the model's
# theoretical max — Ollama's effective num_ctx is the binding constraint, so a
# pack sized for this candidate must stay small. See M.4 guard.
_LOCAL_SAFE_CONTEXT_WINDOW = 8192
_LOCAL_DEFAULT_MAX_OUTPUT = 4096


async def _backfill_model_row(
    *,
    name: str,
    provider: str,
    context_window: int,
    max_output_tokens: int,
) -> Optional[str]:
    """Fill ``context_window`` / ``max_output_tokens`` on a model row if null.

    Matches the row by ``name`` + ``provider``. Fills ONLY the fields that are
    currently null (operator-set values win). Returns the row id, or ``None``
    when no such row exists (a model the operator has not configured — nothing
    to backfill).
    """
    from surrealdb_service.connection import execute_query

    rows = await execute_query(
        "SELECT * FROM model WHERE name = $name AND provider = $provider LIMIT 1;",
        {"name": name, "provider": provider},
    )
    if not rows:
        return None

    row = rows[0]
    model_id = row.get("id")
    updates: dict = {}
    if row.get("context_window") is None:
        updates["context_window"] = context_window
    if row.get("max_output_tokens") is None:
        updates["max_output_tokens"] = max_output_tokens

    if not updates:
        logger.debug(
            f"chain-model backfill: {provider}/{name} already populated "
            f"(context_window={row.get('context_window')}, "
            f"max_output_tokens={row.get('max_output_tokens')})"
        )
        return str(model_id) if model_id is not None else None

    set_clause = ", ".join(f"{k} = ${k}" for k in updates)
    await execute_query(
        f"UPDATE $id SET {set_clause};",
        {"id": model_id, **updates},
    )
    logger.info(
        f"chain-model backfill: {provider}/{name} filled {updates}"
    )
    return str(model_id) if model_id is not None else None


async def backfill_chain_model_params() -> None:
    """Backfill chain-model context params (idempotent, fill-nulls-only).

    Safe to run on every startup: a populated row is a no-op, an operator-tuned
    value is preserved. Never raises — a failure is logged and swallowed.
    """
    try:
        # Local Ollama fallback — the binding constraint for the failover chain.
        # A conservative context_window keeps a pack sized for THIS candidate
        # within the local runtime's effective context (M.4 guard).
        await _backfill_model_row(
            name="llama3.1:8b",
            provider="ollama",
            context_window=_LOCAL_SAFE_CONTEXT_WINDOW,
            max_output_tokens=_LOCAL_DEFAULT_MAX_OUTPUT,
        )
        # NVIDIA NIM mistral-medium — re-assert only if the J.4 seed left a null
        # (defence-in-depth; the J.4 CREATE already sets these).
        await _backfill_model_row(
            name="mistralai/mistral-medium-3.5-128b",
            provider="nvidia",
            context_window=128_000,
            max_output_tokens=16_384,
        )
    except Exception as e:  # noqa: BLE001 — never block startup on the backfill
        logger.warning(
            f"chain-model param backfill skipped ({e}); the context packer "
            "degrades to its conservative default for unpopulated models."
        )
