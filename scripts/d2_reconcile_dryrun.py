"""D2 — what `POST /vocabulary/reconcile` would do on a database, without touching it.

The route needs migration 81 and a loaded register on the target database. This
script needs neither: it fetches the live TOOI registers with the repository's
own `TOOIBulkFetcher`, builds the register IN MEMORY with the provider's own
`_to_reference_row`, reads the entities with one SELECT, and runs the real
`reconcile_active_entities` with no repository and no queue.

Writes under claudedocs/d2-reconcile/:

    tooi_records.json    the register as the fixed loader parses it
    dryrun-staging.json  the full report: scanned per type and status, every link,
                         every candidate (with end dates), every no-match by name

Usage (WSL venv; SELECT only)::

    SURREAL_URL=ws://localhost:8000 SURREAL_DATABASE=staging \\
        .venv/bin/python scripts/d2_reconcile_dryrun.py
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from pathlib import Path

from app_main.services.entity_resolution.vocabulary_reconcile_service import (
    DEFAULT_STATUSES,
    DEFAULT_TYPES,
    reconcile_active_entities,
)
from app_main.services.entity_resolution.vocabulary_reconciler import (
    VocabularyReconciler,
)
from shared.vocabulary.tooi_bulk import TOOIBulkFetcher
from shared.vocabulary.tooi_provider import TOOIProvider
from surrealdb_service.connection import execute_query

OUT = Path("claudedocs/d2-reconcile")


class MemRegister:
    """`reference_entity`'s lookup contract, in memory: `lookup_keys CONTAINSANY [$key]`."""

    rows: list = []

    async def bulk_load(self, rows):
        self.rows = list(rows)
        return len(self.rows)

    async def lookup_by_key(self, key, source=None):
        return [r for r in self.rows if key in (r.get("lookup_keys") or [])]


async def main() -> None:
    fetcher = TOOIBulkFetcher()
    try:
        records = await fetcher.fetch_records()
    finally:
        await fetcher.aclose()
    register = MemRegister()
    provider = TOOIProvider(register)
    register.rows = [provider._to_reference_row(r) for r in records]

    rows = await execute_query(
        "SELECT id, canonical_name, entity_type, status, external_ids, aliases FROM entity "
        "WHERE status IN $statuses AND entity_type IN $types;",
        {"types": list(DEFAULT_TYPES), "statuses": list(DEFAULT_STATUSES)},
    )
    report = await reconcile_active_entities(
        VocabularyReconciler([provider]), rows=rows, dry_run=True
    )
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "tooi_records.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=0), encoding="utf-8")
    (OUT / "dryrun-staging.json").write_text(
        json.dumps({"register_rows": len(records), **asdict(report)},
                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({"register_rows": len(records), "scanned": report.scanned,
                      "skipped_no_name": report.skipped_no_name,
                      "scanned_by": report.scanned_by, "by_type": report.by_type},
                     indent=1))


if __name__ == "__main__":
    asyncio.run(main())
