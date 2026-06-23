"""P.1 backfill: embed active KG entities that have an empty ``embedding``.

The corpus persisted before the P.1 forward fix stored ``embedding=[]`` on every
entity, which silently disables K.5's semantic (embedding) dedup band — the
operator cannot reduce the exhaustively-extracted entity set because the fuzzy/
embedding resolver has no vectors to compare. This script embeds the
``canonical_name`` of each such entity through the SAME configured embedding
model the extraction pipeline uses (resolved via ``get_embedding_service`` →
``ModelManager`` → the DB default; the I.G 768-dim pin — never hardcoded) and
writes the vector onto ``entity.embedding``.

Properties:

* **Idempotent** — it only ever loads entities whose embedding is empty/missing
  (``EntityRepository.list_active_entities_missing_embedding``), so re-running
  skips everything already done. Safe to run repeatedly.
* **Resumable** — each batch is committed immediately, then the next batch is
  re-queried from the live "still missing" set. A crash mid-run loses at most
  the in-flight batch; the next run picks up exactly where it stopped.
* **Batched** — embeds ``--batch-size`` texts per model call (default 64) to
  bound memory and amortise the round-trip.
* **Logs progress** — one line per batch (done / total-seen / failures).

Usage (run inside the open_notebook app environment, e.g. the app container or
``uv run --project apps/app-main``)::

    python scripts/backfill_entity_embeddings.py
    python scripts/backfill_entity_embeddings.py --batch-size 128
    python scripts/backfill_entity_embeddings.py --limit 200   # cap one run
    python scripts/backfill_entity_embeddings.py --dry-run     # count only

Args:
    --batch-size N  Entities embedded per model call (default 64).
    --limit N       Stop after backfilling at most N entities (default: all).
    --dry-run       Report how many entities are missing a vector; write nothing.
"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger
from surrealdb_service.repositories.entity import EntityRepository


async def _backfill(
    batch_size: int, limit: int | None, dry_run: bool
) -> dict[str, int]:
    """Embed + persist vectors for active entities missing an embedding.

    Returns a summary dict (``backfilled`` / ``failed`` / ``remaining``).
    """
    repo = EntityRepository()

    # Initial count of the work to do (a cheap read; also the dry-run answer).
    pending = await repo.list_active_entities_missing_embedding()
    total_missing = len(pending)
    logger.info(
        f"backfill_entity_embeddings: {total_missing} active entit(ies) "
        f"with an empty/missing embedding"
    )
    if dry_run:
        return {"backfilled": 0, "failed": 0, "remaining": total_missing}
    if total_missing == 0:
        logger.info("Nothing to backfill — every active entity has a vector.")
        return {"backfilled": 0, "failed": 0, "remaining": 0}

    # Resolve the SAME embedding model the pipeline uses (the I.G 768-dim pin,
    # configured in the DB defaults). Imported lazily so a --dry-run never needs
    # the app DI container / model layer wired up.
    from app_main.dependencies import get_embedding_service

    embed_svc = await get_embedding_service()
    model = embed_svc.embedding_model

    backfilled = 0
    failed = 0
    dim_seen: int | None = None

    while True:
        # Re-query the live "still missing" set each iteration so the run stays
        # resumable and idempotent — never relies on a stale in-memory cursor.
        remaining_budget = (
            None if limit is None else max(0, limit - backfilled)
        )
        if remaining_budget == 0:
            logger.info(f"Reached --limit {limit}; stopping.")
            break
        fetch = (
            batch_size
            if remaining_budget is None
            else min(batch_size, remaining_budget)
        )
        rows = await repo.list_active_entities_missing_embedding(limit=fetch)
        if not rows:
            break

        names = [str(r.get("canonical_name") or "") for r in rows]
        try:
            vectors = await model.aembed(names)
        except Exception as e:
            # A model-call failure is fatal for this batch; abort rather than
            # write partial/garbage. The next run retries the same rows (they
            # are still "missing").
            logger.error(f"Embedding call failed for a batch of {len(names)}: {e}")
            failed += len(names)
            break

        for row, vec in zip(rows, vectors):
            entity_id = str(row.get("id"))
            if not vec:
                logger.warning(
                    f"Empty vector for entity {entity_id} "
                    f"('{row.get('canonical_name')}') — skipping"
                )
                failed += 1
                continue
            if dim_seen is None:
                dim_seen = len(vec)
                logger.info(f"Embedding dimension = {dim_seen}")
            ok = await repo.update_entity_embedding(entity_id, list(vec))
            if ok:
                backfilled += 1
            else:
                failed += 1

        logger.info(
            f"Progress: backfilled={backfilled} failed={failed} "
            f"(of {total_missing} initially missing)"
        )

    remaining = len(await repo.list_active_entities_missing_embedding())
    logger.info(
        f"backfill_entity_embeddings DONE: backfilled={backfilled} "
        f"failed={failed} remaining={remaining} "
        f"(embedding dim={dim_seen})"
    )
    return {"backfilled": backfilled, "failed": failed, "remaining": remaining}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill empty entity embeddings (P.1)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Entities embedded per model call (default 64).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Backfill at most N entities this run (default: all).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count entities missing a vector; write nothing.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(
        _backfill(
            batch_size=args.batch_size,
            limit=args.limit,
            dry_run=args.dry_run,
        )
    )


if __name__ == "__main__":
    main()
