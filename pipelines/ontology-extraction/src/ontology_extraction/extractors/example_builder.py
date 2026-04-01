"""Build LangExtract ExampleData from ontology definitions and YAML overrides."""

from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from ontology_manager.schema import Ontology

try:
    import langextract as lx
except ImportError:
    lx = None  # type: ignore[assignment]


class ExampleBuilder:
    """Build LangExtract ExampleData from ontology definitions and YAML overrides."""

    def __init__(self, examples_dir: Path | None = None):
        self._examples_dir = examples_dir

    def build_examples(
        self, ontology: Ontology, override_file: Path | None = None
    ) -> tuple[str, list[Any]]:
        """Return (prompt_description, examples) for lx.extract().

        1. Generate baseline examples from ontology entity types
        2. Load and merge YAML overrides if file exists
        3. Return combined prompt + examples
        """
        if lx is None:
            raise ImportError(
                "langextract is required for LangExtractExtractor. "
                "Install it with: pip install langextract"
            )

        prompt, override_examples = None, []

        # Try to load YAML overrides
        resolved_override = override_file
        if resolved_override is None and self._examples_dir is not None:
            candidate = (
                self._examples_dir
                / f"{ontology.metadata.name}_langextract_examples.yaml"
            )
            if candidate.exists():
                resolved_override = candidate

        if resolved_override is not None and resolved_override.exists():
            prompt, override_examples = self._load_yaml_overrides(resolved_override)

        # Generate baseline examples from ontology
        ontology_examples = self._examples_from_ontology(ontology)

        # Build default prompt if not provided by overrides
        if not prompt:
            entity_names = ", ".join(ontology.entity_types.keys())
            prompt = (
                f"Extract all entities of the following types from the given text: "
                f"{entity_names}. Be thorough and precise. Use exact text from the "
                f"input for extraction_text."
            )

        # Override examples take precedence; append ontology-generated ones
        combined = override_examples + ontology_examples
        return prompt, combined

    def _examples_from_ontology(self, ontology: Ontology) -> list[Any]:
        """Generate ExampleData from entity type definitions.

        For each entity type that has examples and extraction hints, we
        construct a synthetic example text and corresponding extractions
        so LangExtract understands the extraction classes.
        """
        if lx is None:
            return []

        examples: list[Any] = []

        for type_name, entity_type in ontology.entity_types.items():
            if not entity_type.examples:
                continue

            # Build a synthetic example text from the entity's examples
            example_names = entity_type.examples[:3]  # Use up to 3 examples
            hints = entity_type.extraction_hints or []
            description = entity_type.description or f"a {type_name}"

            # Create a simple text that mentions the example entities
            text_parts = []
            if hints:
                text_parts.append(hints[0])
            text_parts.extend(
                f"{name} is {description}." for name in example_names
            )
            text = " ".join(text_parts)

            # Build extractions for each example entity name
            extractions = []
            for name in example_names:
                attrs: dict[str, Any] = {}
                # Add property examples if available
                for prop in entity_type.properties:
                    if prop.examples:
                        attrs[prop.name] = prop.examples[0]
                extractions.append(
                    lx.data.Extraction(
                        extraction_class=type_name,
                        extraction_text=name,
                        attributes=attrs if attrs else None,
                    )
                )

            if extractions:
                examples.append(
                    lx.data.ExampleData(text=text, extractions=extractions)
                )

        return examples

    def _load_yaml_overrides(
        self, path: Path
    ) -> tuple[str | None, list[Any]]:
        """Load override examples from a YAML file.

        Expected format:
            prompt_description: "..."
            examples:
              - text: "..."
                extractions:
                  - extraction_class: "Person"
                    extraction_text: "Dr. Jane Smith"
                    attributes:
                      role: "researcher"
        """
        if lx is None:
            return None, []

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load LangExtract examples from {path}: {e}")
            return None, []

        if not isinstance(data, dict):
            logger.warning(f"Invalid LangExtract examples file format: {path}")
            return None, []

        prompt = data.get("prompt_description")
        raw_examples = data.get("examples", [])
        examples: list[Any] = []

        for raw in raw_examples:
            text = raw.get("text", "")
            extractions = []
            for ext in raw.get("extractions", []):
                extractions.append(
                    lx.data.Extraction(
                        extraction_class=ext.get("extraction_class", ""),
                        extraction_text=ext.get("extraction_text", ""),
                        attributes=ext.get("attributes"),
                    )
                )
            if text and extractions:
                examples.append(
                    lx.data.ExampleData(text=text, extractions=extractions)
                )

        logger.info(
            f"Loaded {len(examples)} LangExtract example(s) from {path}"
        )
        return prompt, examples
