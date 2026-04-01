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
    parser.add_argument(
        "--extractor",
        default="llm",
        choices=["llm", "langextract"],
        help="Extraction backend to use (default: llm)",
    )
    parser.add_argument(
        "--langextract-model",
        default="qwen2.5:latest",
        help="Model ID for LangExtract (default: qwen2.5:latest)",
    )
    parser.add_argument(
        "--langextract-examples-dir",
        default=None,
        help="Directory containing LangExtract YAML example overrides",
    )
    parser.add_argument(
        "--langextract-model-url",
        default=None,
        help="Model URL for Ollama (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--langextract-passes",
        type=int,
        default=None,
        help="Number of extraction passes (default: 1)",
    )
    parser.add_argument(
        "--langextract-max-workers",
        type=int,
        default=None,
        help="Max parallel workers (default: 4)",
    )
    parser.add_argument(
        "--langextract-max-char-buffer",
        type=int,
        default=None,
        help="Max chars per chunk (default: 5000)",
    )
    parser.add_argument(
        "--langextract-batch-length",
        type=int,
        default=None,
        help="Chunks per batch (>= max_workers)",
    )
    parser.add_argument(
        "--langextract-temperature",
        type=float,
        default=None,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--langextract-max-output-tokens",
        type=int,
        default=None,
        help="Token limit for model output",
    )
    parser.add_argument(
        "--langextract-top-p",
        type=float,
        default=None,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--langextract-top-k",
        type=int,
        default=None,
        help="Top-k sampling parameter",
    )
    parser.add_argument(
        "--langextract-no-schema",
        action="store_true",
        default=False,
        help="Disable schema constraints (sets use_schema_constraints=False)",
    )
    parser.add_argument(
        "--langextract-no-fence",
        action="store_true",
        default=False,
        help="Disable output fencing (sets fence_output=False)",
    )
    parser.add_argument(
        "--langextract-api-key",
        default=None,
        help="API key for cloud providers (OpenAI, Gemini)",
    )
    parser.add_argument(
        "--langextract-provider",
        default=None,
        help="Explicit provider selection (ollama, openai, gemini, vertexai)",
    )
    parser.add_argument(
        "--langextract-save-jsonl",
        action="store_true",
        default=False,
        help="Save raw results to JSONL",
    )
    parser.add_argument(
        "--langextract-jsonl-dir",
        default=None,
        help="JSONL output directory",
    )
    parser.add_argument(
        "--langextract-visualize",
        action="store_true",
        default=False,
        help="Generate HTML visualization",
    )
    parser.add_argument(
        "--langextract-visualize-dir",
        default=None,
        help="Visualization output directory",
    )
    args = parser.parse_args()

    from .config import ExtractionConfig
    from .workflow import ExtractionWorkflow

    config_kwargs = {
        "ontology_name": args.ontology,
        "extractor_type": args.extractor,
        "langextract_model_id": args.langextract_model,
        "langextract_examples_dir": args.langextract_examples_dir,
        "langextract_use_schema_constraints": not args.langextract_no_schema,
        "langextract_fence_output": not args.langextract_no_fence,
        "langextract_save_jsonl": args.langextract_save_jsonl,
        "langextract_visualize": args.langextract_visualize,
    }

    # Only set optional fields when explicitly provided
    if args.langextract_model_url is not None:
        config_kwargs["langextract_model_url"] = args.langextract_model_url
    if args.langextract_passes is not None:
        config_kwargs["langextract_extraction_passes"] = args.langextract_passes
    if args.langextract_max_workers is not None:
        config_kwargs["langextract_max_workers"] = args.langextract_max_workers
    if args.langextract_max_char_buffer is not None:
        config_kwargs["langextract_max_char_buffer"] = args.langextract_max_char_buffer
    if args.langextract_batch_length is not None:
        config_kwargs["langextract_batch_length"] = args.langextract_batch_length
    if args.langextract_temperature is not None:
        config_kwargs["langextract_temperature"] = args.langextract_temperature
    if args.langextract_max_output_tokens is not None:
        config_kwargs["langextract_max_output_tokens"] = args.langextract_max_output_tokens
    if args.langextract_top_p is not None:
        config_kwargs["langextract_top_p"] = args.langextract_top_p
    if args.langextract_top_k is not None:
        config_kwargs["langextract_top_k"] = args.langextract_top_k
    if args.langextract_api_key is not None:
        config_kwargs["langextract_api_key"] = args.langextract_api_key
    if args.langextract_provider is not None:
        config_kwargs["langextract_provider"] = args.langextract_provider
    if args.langextract_jsonl_dir is not None:
        config_kwargs["langextract_jsonl_output_dir"] = args.langextract_jsonl_dir
    if args.langextract_visualize_dir is not None:
        config_kwargs["langextract_visualize_output_dir"] = args.langextract_visualize_dir

    config = ExtractionConfig(**config_kwargs)
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
