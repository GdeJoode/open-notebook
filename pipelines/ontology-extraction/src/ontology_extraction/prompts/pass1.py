"""Prompt template for Pass-1 schema validation (B.1c).

Single-schema only — the multi-schema orchestrator lands in B.1e.

Structure
=========

The prompt has three sections, sized so the full assembled prompt
stays inside the 3000-token Pass-1 budget for a typical scholarly-grade
ontology (~8 entity types) + a 1500-token text sample:

1. **Schema summary** (~500 tokens for small ontologies). One bullet per
   entity-type: ``name — description``. For ontologies with more than
   ``COMPACT_SUMMARY_THRESHOLD`` entity types, descriptions are dropped
   and the names are emitted as a single comma-separated group so the
   token budget is not blown by repeated long descriptions
   (Major 3 of the B.1c attempt-1 review: a 100-type ontology would
   otherwise overflow the 2400-token target).
2. **Text sample** (~1500 tokens). The first 3–5 chunks of the source.
3. **Output schema** (~200 tokens). A compact JSON-schema sketch so
   the LLM emits parseable structured output on the first attempt.

The actual token budget is enforced by the caller via the
``len(text) // 4`` heuristic (see Q-B-2 in the plan).
"""

from __future__ import annotations

from typing import Any, Dict

# When the ontology has more than this many entity types, the schema
# summary drops descriptions and emits a single comma-separated list
# of type names. Driven by the reviewer's worked example: a 100-type
# ontology with average ~80-char descriptions blows past the 2400-token
# budget under the verbose format. Names alone are the load-bearing
# signal for Pass-1's schema-fit judgement.
COMPACT_SUMMARY_THRESHOLD = 30

# Canonical Pass-1 system prompt. Moved here from the validator module
# (Minor #5 of the attempt-1 review: all prompt strings live under
# ``prompts/``).
PASS1_SYSTEM_PROMPT = (
    "You are a meticulous ontology curator. Judge schema fit "
    "honestly — under-confidence is better than over-confidence."
)

# Closed list of schema names this codebase recognises as Pass-1
# alternatives. Sourced from ``packages/ontology-manager/ontologies/``
# at the time of writing (B.1c attempt 2). Adding a new ontology YAML
# means adding the name here so the LLM sees the option. Drives the
# ``alternative_schemas`` hint in the prompt. Minor #2 of the
# attempt-1 review noted ``base``, ``policy_themes``, and
# ``social_profiles`` were missing — they are included now.
CANDIDATE_SCHEMA_NAMES = [
    "base",
    "deals",
    "general",
    "government",
    "instruments",
    "policy",
    "policy_themes",
    "regiodeal",
    "schema_core",
    "scholarly",
    "social_profiles",
]


def _build_output_format_section() -> str:
    """Render the closed-list output-format section.

    Inlined so the candidate-schema list is rendered from
    ``CANDIDATE_SCHEMA_NAMES`` at prompt time. Keeping this as a
    function (not a constant) means a new ontology gets picked up
    without re-templating the docstring.
    """
    candidates = ", ".join(CANDIDATE_SCHEMA_NAMES)
    # Compact output-format section: the example JSON is the
    # load-bearing part; one-line field rules are sufficient. The
    # original verbose version (attempt 1) was ~465 tokens which
    # combined with the schema summary blew the 100-type stress
    # budget. This version is ~280 tokens (Major 3 follow-up).
    return f"""\
## Output Format

Return ONLY this JSON object (no prose, no markdown fence):

```json
{{
  "detected_schema": "scholarly",
  "confidence_in_choice": 0.87,
  "coverage_pct": 0.74,
  "uncovered_concepts": [
    {{"surface_form": "deep neural network", "suggested_type": "Method"}}
  ],
  "proposed_extensions": [
    {{"type_name": "Method", "parent_type": "Concept", "rationale": "..."}}
  ],
  "alternative_schemas": [
    {{"name": "general", "confidence": 0.4}}
  ]
}}
```

Rules:
- ``detected_schema``: best-fit schema name (use attempted name if it fits).
- ``confidence_in_choice``, ``coverage_pct``: floats in [0.0, 1.0].
- ``uncovered_concepts``: up to 10 surface forms not covered by attempted schema.
- ``proposed_extensions``: up to 5 new type proposals.
- ``alternative_schemas``: up to 3 objects with ``name`` + ``confidence``. Names from: {candidates}.
"""


def build_schema_summary(ontology: Dict[str, Any]) -> str:
    """Render an ontology to a compact ``name — description`` bullet list.

    ~500 tokens for a small ontology. Intentionally omits properties
    and relationships — Pass-1 only judges *whether* a concept fits
    *some* type, not which properties it would fill.

    When the ontology has more than ``COMPACT_SUMMARY_THRESHOLD``
    entity types, descriptions are dropped (Major 3 of the B.1c
    attempt-1 review). The LLM gets the type names as a single
    comma-separated group plus a one-line note explaining the
    compression, which preserves the schema-fit signal while keeping
    the prompt under the 2400-token budget for ontologies with up to
    ~150 types.

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
        return "\n".join(lines)

    if len(pairs) > COMPACT_SUMMARY_THRESHOLD:
        # Compressed format: names only, single group, plus a note so
        # the LLM does not assume the descriptions were stripped by
        # accident. Reviewer's worked example (100-type ontology +
        # 1500-token sample) sits under the budget at this density.
        names = ", ".join(n for n, _ in pairs)
        lines.append(f"{{{names}}}")
        lines.append("")
        lines.append(
            "Types compressed: descriptions omitted to fit the prompt "
            "budget. Ask for details on specific types if needed."
        )
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
        f"{_build_output_format_section()}"
    )
