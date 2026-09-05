"""
CLI entry point for the entity-filtering pipeline.

Provides a simple command-line interface for running the filtering
pipeline on JSON extraction results.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger
from shared.models.extraction import ExtractionResult

from entity_filtering.config import FilteringConfig, KGResolutionConfig
from entity_filtering.workflow import FilteringWorkflow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entity filtering pipeline - filter, normalize, "
        "deduplicate, and score extracted entities and relations.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to JSON file containing an ExtractionResult.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Path to write the FilteredResult JSON. "
        "Defaults to stdout.",
    )
    parser.add_argument(
        "--min-entity-length",
        type=int,
        default=2,
        help="Minimum character length for entities (default: 2).",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication.",
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.85,
        help="Deduplication similarity threshold (default: 0.85).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a JSON config file for FilteringConfig.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


async def _run(args: argparse.Namespace) -> None:
    # Configure logging
    logger.remove()
    level = "DEBUG" if args.verbose else "INFO"
    logger.add(sys.stderr, level=level)

    # Load config
    if args.config and args.config.exists():
        with open(args.config) as f:
            config_data = json.load(f)
        config = FilteringConfig(**config_data)
    else:
        config = FilteringConfig(
            min_entity_length=args.min_entity_length,
            dedup_enabled=not args.no_dedup,
            dedup_similarity_threshold=args.dedup_threshold,
            # PC.3: this branch is the CLI's OWN default — nobody chose it — so
            # it states the choice instead of inheriting one. Off, matching the
            # app: stage 10's verdict has no consumer, so running it costs a
            # candidate fetch plus Levenshtein and cosine per entity and changes
            # nothing. A `--config` file overrides everything here and is the
            # operator's own decision.
            kg_resolution=KGResolutionConfig(enabled=False),
        )

    # Load input
    if not args.input.exists():
        logger.error("Input file does not exist: {}", args.input)
        sys.exit(1)

    with open(args.input) as f:
        raw = json.load(f)
    extraction_result = ExtractionResult(**raw)

    logger.info(
        "Loaded extraction result: {} entities, {} relations",
        extraction_result.entity_count,
        extraction_result.relation_count,
    )

    # Run pipeline
    workflow = FilteringWorkflow(config=config)
    result = await workflow.process(extraction_result)

    # Output
    output_json = result.model_dump_json(indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_json)
        logger.info("Wrote result to {}", args.output)
    else:
        print(output_json)


def main() -> None:
    """CLI entry point."""
    args = _parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
