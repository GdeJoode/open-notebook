"""CLI entry point for ontology-extraction pipeline."""

import argparse
import asyncio
import json
import sys

from loguru import logger


def main():
    parser = argparse.ArgumentParser(
        description="Ontology-guided extraction pipeline"
    )
    parser.add_argument("--ontology", default="general", help="Ontology to use")
    parser.add_argument("--input", type=str, help="Input JSON file with chunks")
    parser.add_argument(
        "--output", type=str, help="Output JSON file for results"
    )
    args = parser.parse_args()

    from .config import ExtractionConfig
    from .workflow import ExtractionWorkflow

    config = ExtractionConfig(ontology_name=args.ontology)
    workflow = ExtractionWorkflow(config)

    if args.input:
        with open(args.input) as f:
            chunks = json.load(f)
    else:
        text = sys.stdin.read()
        chunks = [{"text": text, "id": "stdin"}]

    result = asyncio.run(workflow.extract(chunks))

    output = result.model_dump_json(indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        logger.info(f"Results written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
