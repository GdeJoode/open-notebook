"""Track L.4 — seed a per-notebook schema default so policy_themes is applied.

The 44% generic ``concept``/``topic`` bucket is a schema-APPLICATION gap. L.4
adds two mechanisms so the keyword-poor ``policy_themes`` schema gets applied to
Regio-Deal documents:

  1. schema-affinity co-selection in ``detect_applicable_schemas`` (automatic —
     fires for ALL gov/policy docs once ``deals``/``government`` is selected), and
  2. a per-notebook ``notebook_schema`` default (this script — the explicit
     operator override for the Regio-Deal notebook).

A full settings UI for the base ontology is OUT OF SCOPE for L.4 (the existing
``schemas.py`` router exposes review-flags / extensions / type-edits, but not a
``base_ontology`` setter). This documented seed is the V1 way to SET the default
for a specific notebook; it reuses ``NotebookSchemaRepository.upsert`` (singleton
per notebook — rewrites in place).

Usage (inside the app container / with the workspace env active)::

    uv run --project apps/app-main python scripts/l4_seed_notebook_schema.py \
        notebook:abc123 --base deals

The configured base + its affinity bundle (``deals`` -> ``policy_themes``) are
then forced by ``entity_extraction_service`` on the next ingest regardless of the
keyword score — config beats auto-detection (L.4 AC3).
"""

from __future__ import annotations

import argparse
import asyncio

from loguru import logger
from shared.models.notebook_schema import NotebookSchema
from surrealdb_service.repositories.notebook_schema import NotebookSchemaRepository


async def seed(notebook_id: str, base_ontology: str) -> str:
    """Create or rewrite the notebook_schema row for ``notebook_id``.

    Preserves any existing accepted/pending extensions and review flags — only
    the ``base_ontology`` is (re)set. Returns the persisted record id.
    """
    repo = NotebookSchemaRepository()
    existing = await repo.get_by_notebook(notebook_id)
    if existing is not None:
        existing.base_ontology = base_ontology
        record_id = await repo.upsert(existing)
        logger.info(
            "Updated notebook_schema {rid}: base_ontology={base}",
            rid=record_id,
            base=base_ontology,
        )
        return record_id

    schema = NotebookSchema(notebook=notebook_id, base_ontology=base_ontology)
    record_id = await repo.upsert(schema)
    logger.info(
        "Created notebook_schema {rid} for {nb}: base_ontology={base}",
        rid=record_id,
        nb=notebook_id,
        base=base_ontology,
    )
    return record_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "notebook_id",
        help="Record id of the notebook (e.g. 'notebook:abc123').",
    )
    parser.add_argument(
        "--base",
        default="deals",
        help=(
            "Base ontology to force (default: 'deals'). Its affinity bundle "
            "(deals -> policy_themes) is co-applied automatically."
        ),
    )
    args = parser.parse_args()
    record_id = asyncio.run(seed(args.notebook_id, args.base))
    print(record_id)


if __name__ == "__main__":
    main()
