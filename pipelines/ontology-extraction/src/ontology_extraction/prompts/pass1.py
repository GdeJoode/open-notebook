"""Prompt template for Pass-1 schema validation (B.1c).

Single-schema only — the multi-schema orchestrator lands in B.1e.

Structure
=========

The prompt has three sections, sized so the full assembled prompt
stays inside the 3000-token Pass-1 budget for a typical scholarly-grade
ontology (~30 entity types) + a 1500-token text sample:

1. **Schema summary** (~500 tokens). One bullet per entity-type:
   ``name — description``. Properties / relationships are intentionally
   omitted — Pass-1 only needs the *type names* to judge coverage.
2. **Text sample** (~1500 tokens). The first 3–5 chunks of the source.
3. **Output schema** (~200 tokens). A compact JSON-schema sketch so
   the LLM emits parseable structured output on the first attempt.

The actual token budget is enforced by the caller via the
``len(text) // 4`` heuristic (see Q-B-2 in the plan).
"""

from __future__ import annotations

from typing import Any, Dict

# Output schema documented inline so the LLM emits parseable JSON on
# the first attempt. Field names exactly match ``Pass1Output``. The
# top-level wrapping object is required so the response is a single
# parse target.
_OUTPUT_FORMAT = """\
## Output Format

Return ONLY a single JSON object with this exact shape (no other text):

```json
{
  "detected_schema": "scholarly",
  "confidence_in_choice": 0.87,
  "coverage_pct": 0.74,
  "uncovered_concepts": [
    {"surface_form": "deep neural network", "suggested_type": "Method"}
  ],
  "proposed_extensions": [
    {"type_name": "Method", "parent_type": "Concept", "rationale": "..."}
  ],
  "alternative_schemas": ["general", "policy"]
}
```

## Field Rules

- ``detected_schema``: the SCHEMA NAME you think best fits this text.
  If the attempted schema fits, return its name. Otherwise return the
  name of a better-fit schema.
- ``confidence_in_choice``: float in [0.0, 1.0]. Your confidence that
  ``detected_schema`` is the right choice.
- ``coverage_pct``: float in [0.0, 1.0]. Fraction of important
  text concepts that map cleanly onto the ATTEMPTED schema's entity
  types.
- ``uncovered_concepts``: surface forms from the text that did NOT
  fit any attempted-schema type. Each dict carries the surface form
  as it appeared in text, plus a one-word ``suggested_type``.
  Maximum 10 entries.
- ``proposed_extensions``: new entity types you'd add to the schema
  to cover the gaps. Each dict carries ``type_name``, ``parent_type``
  (an existing schema type to inherit from), and a short
  ``rationale``. Maximum 5 entries.
- ``alternative_schemas``: up to 3 alternative schema NAMES (from
  this list of candidates: scholarly, general, policy, government,
  regiodeal, deals, instruments, schema_core) that might fit the
  text better than the attempted one. Empty list if the attempted
  schema is clearly the best.

Return ONLY the JSON object, no commentary, no markdown wrapping.
"""


def build_schema_summary(ontology: Dict[str, Any]) -> str:
    """Render an ontology to a compact ``name — description`` bullet list.

    ~500 tokens for a 30-type ontology. Intentionally omits properties
    and relationships — Pass-1 only judges *whether* a concept fits
    *some* type, not which properties it would fill.

    Accepts a plain ``dict`` rather than an ``Ontology`` instance so
    the caller can pass either the YAML-parsed form or a ``dict()``
    of a Pydantic model without an import cycle.

    Args:
        ontology: Dict with at least ``{"metadata": {"name": str},
            "entity_types": dict|list}``. Both dict and list-of-dicts
            shapes are accepted for ``entity_types`` since YAML files
            in this repo use the list shape.

    Returns:
        Markdown-style bulleted summary suitable for prompt
        injection. Returns an empty-but-labeled stub if no entity
        types are present, so the LLM still receives a parseable
        prompt.
    """
    metadata = ontology.get("metadata", {}) or {}
    schema_name = metadata.get("name", "unknown")

    entity_types = ontology.get("entity_types", {}) or {}

    # Normalize both dict and list shapes to ``[(name, description), ...]``
    pairs: list[tuple[str, str]] = []
    if isinstance(entity_types, dict):
        for name, defn in entity_types.items():
            desc = ""
            if isinstance(defn, dict):
                desc = defn.get("description", "") or ""
            elif hasattr(defn, "description"):
                desc = getattr(defn, "description", "") or ""
            pairs.append((str(name), str(desc)))
    elif isinstance(entity_types, list):
        for et in entity_types:
            if isinstance(et, dict):
                pairs.append(
                    (str(et.get("name", "")), str(et.get("description", "") or ""))
                )
            elif hasattr(et, "name"):
                pairs.append(
                    (
                        str(getattr(et, "name", "")),
                        str(getattr(et, "description", "") or ""),
                    )
                )

    pairs = [(n, d) for n, d in pairs if n]

    lines = [f"## Schema: {schema_name}", "", "Entity types in this schema:", ""]
    if not pairs:
        lines.append("(none)")
    else:
        for name, desc in pairs:
            # Trim long descriptions to keep the token budget tight.
            short_desc = desc.strip().replace("\n", " ")
            if len(short_desc) > 160:
                short_desc = short_desc[:157] + "..."
            if short_desc:
                lines.append(f"- **{name}** — {short_desc}")
            else:
                lines.append(f"- **{name}**")
    return "\n".join(lines)


def build_pass1_prompt(ontology: Dict[str, Any], text_sample: str) -> str:
    """Assemble the complete Pass-1 prompt body.

    The output is a single string the caller hands to the LLM as the
    *user* prompt (a separate system prompt setting role/persona is
    fine to layer on top — Pass-1 needs no special system context).

    Args:
        ontology: See ``build_schema_summary``.
        text_sample: ~1500-token text excerpt from the source. The
            caller is responsible for sampling; this function does
            not truncate (truncation belongs at the budget guard so
            the failure is loud, not silent).

    Returns:
        A multi-section prompt string.
    """
    schema_summary = build_schema_summary(ontology)
    return (
        "# Pass-1 Schema Validation\n"
        "\n"
        "You are validating whether an ontology schema fits a piece of text.\n"
        "Judge how well the schema below covers the concepts in the text,\n"
        "and propose extensions for any important concepts it misses.\n"
        "\n"
        f"{schema_summary}\n"
        "\n"
        "## Text Sample\n"
        "\n"
        f"{text_sample}\n"
        "\n"
        f"{_OUTPUT_FORMAT}"
    )
