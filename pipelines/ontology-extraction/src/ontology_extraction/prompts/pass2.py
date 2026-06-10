"""Prompt template for Pass-2 typed extraction (B.1d).

Single-schema only. Pass-2 takes an ontology, an optional list of
"accepted extensions" (extra entity types accepted by the curator in
B.3c, conceptually) and a chunk of source text, and asks the LLM to
return ``{entities: [...], relations: [...]}`` with a confidence
score for **every** entity and relation.

Why a dedicated module
======================

The existing ``LLMExtractor.extract`` path uses
``OntologyPromptGenerator.generate_combined_extraction_prompt`` from
``ontology-manager``. That prompt is richer (concepts + claims) but
does not know how to inject extension types or how to enforce the
B4 "confidence on every element" contract. Rather than fork the
generator we build a focused per-chunk prompt here and let B.1f
swap callers.

Token budget
============
Per Q-B-2 the assembled prompt + chunk text is capped at 3000
estimated tokens (``len(text) // 4``). With a 20% safety margin the
target is ≤ 2400 tokens — same envelope as Pass-1. For very large
ontologies the entity-type summary auto-compresses (see
``COMPACT_TYPE_THRESHOLD``) so a 100-type ontology + 1500-token
chunk still fits comfortably.

Structure
=========
1. **Schema header** (~100 tokens). Ontology name + the
   "confidence is mandatory" reminder.
2. **Entity types** (~600 tokens for small ontologies; auto-compressed
   above the threshold). One bullet per type: ``name — description``.
3. **Accepted extensions** (~150 tokens, only when non-empty). New
   entity types the curator accepted for this schema. Each rendered
   as ``name (parent: parent_type) — rationale``.
4. **Relationship types** (~300 tokens). Single-line ``name`` plus
   any domain/range hints.
5. **Text chunk** (~1500 tokens for a typical chunk).
6. **Output schema** (~250 tokens). A compact JSON-schema sketch +
   strict rules pinned to the B4 contract.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ontology_manager.schema import Ontology

# Above this many entity types, drop descriptions and emit the names
# as a single comma-separated group (mirrors Pass-1's
# COMPACT_SUMMARY_THRESHOLD). 30 is the same headroom B.1c proved fits
# a 100-type ontology + 1500-token chunk under a 2400-token budget.
COMPACT_TYPE_THRESHOLD = 30

# Same threshold for relationship types — they tend to be terser, but
# a schema with hundreds of relations still blows the budget under
# the verbose render. Compression is symmetrical.
COMPACT_RELATION_THRESHOLD = 30


PASS2_SYSTEM_PROMPT = (
    "You are a precise knowledge extractor. For every entity and "
    "every relation you extract you MUST return a confidence value "
    "in the closed interval [0.0, 1.0] reflecting how certain you "
    "are that the extraction is correct. Under-confidence is "
    "preferred over over-confidence."
)


def _format_entity_types(ontology: Ontology) -> List[str]:
    """Render the ontology's entity types as a Markdown bullet list.

    Compresses to names-only above ``COMPACT_TYPE_THRESHOLD``. Returns
    a list of lines (not a joined string) so the caller can splice the
    section into a larger prompt without extra ``join`` calls.
    """
    types = list(ontology.entity_types.values())
    if not types:
        return ["(no entity types defined)"]

    if len(types) > COMPACT_TYPE_THRESHOLD:
        names = ", ".join(t.name for t in types)
        return [
            f"{{{names}}}",
            "",
            "Types compressed: descriptions omitted to fit the prompt "
            "budget. Use the names verbatim as ``label`` values.",
        ]

    lines: List[str] = []
    for t in types:
        desc = (t.description or "").strip().replace("\n", " ")
        if len(desc) > 160:
            desc = desc[:157] + "..."
        if desc:
            lines.append(f"- **{t.name}** — {desc}")
        else:
            lines.append(f"- **{t.name}**")
    return lines


def _format_relationship_types(ontology: Ontology) -> List[str]:
    """Render the ontology's relationship types as a Markdown bullet list.

    Symmetric compression policy to ``_format_entity_types``.
    Domain/range are joined with ``|`` so an entry stays on one line.
    """
    rels = list(ontology.relationship_types.values())
    if not rels:
        return ["(no relationship types defined)"]

    if len(rels) > COMPACT_RELATION_THRESHOLD:
        names = ", ".join(r.name for r in rels)
        return [
            f"{{{names}}}",
            "",
            "Relationship types compressed: descriptions omitted. Use "
            "the names verbatim as ``type`` values.",
        ]

    lines: List[str] = []
    for r in rels:
        domain = "|".join(r.domain) if r.domain else "Any"
        range_ = "|".join(r.range) if r.range else "Any"
        desc = (r.description or "").strip().replace("\n", " ")
        if len(desc) > 100:
            desc = desc[:97] + "..."
        head = f"- **{r.name}** ({domain} → {range_})"
        lines.append(f"{head} — {desc}" if desc else head)
    return lines


def _format_accepted_extensions(extensions: List[Dict[str, Any]]) -> List[str]:
    """Render the curator-accepted extension types as bullets.

    Each extension dict is expected to carry at least
    ``type_name``; ``parent_type`` and ``rationale`` are surfaced
    when present so the LLM understands the relationship to the base
    ontology. Missing fields render as ``(unknown parent)`` / no
    rationale to keep the prompt parseable even if upstream sends
    sparse data.

    B.3c — Defence-in-depth: resume-sentinel entries (carrying
    ``is_resume_sentinel=True``) are filtered out here as well as at
    the extraction-service seam. They are infrastructure markers for
    the pause/resume flow — letting them through would render
    ``_resumed_without_extensions`` into the LLM prompt as a real
    entity type. If the upstream filter ever regresses, the prompt
    builder still scrubs the leak before it reaches the LLM.
    """
    if not extensions:
        return []

    # Filter sentinels up-front so the empty-result branch below
    # behaves correctly when the entire list is sentinels.
    visible = [ext for ext in extensions if not ext.get("is_resume_sentinel")]
    if not visible:
        return []

    lines = [
        "",
        "## Accepted Extension Types",
        "",
        (
            "These are additional entity types the schema curator "
            "accepted for this notebook. Treat them as first-class "
            "members of the schema and use the exact ``type_name`` as "
            "the ``label`` value when you extract instances."
        ),
        "",
    ]
    for ext in visible:
        name = str(ext.get("type_name", "")).strip()
        if not name:
            # Skip malformed entries — no usable label.
            continue
        parent = str(ext.get("parent_type", "")).strip()
        rationale = str(ext.get("rationale", "")).strip().replace("\n", " ")
        if len(rationale) > 160:
            rationale = rationale[:157] + "..."

        parent_part = f"parent: {parent}" if parent else "no parent"
        head = f"- **{name}** ({parent_part})"
        lines.append(f"{head} — {rationale}" if rationale else head)
    return lines


_OUTPUT_FORMAT_SECTION = """\
## Output Format

Return ONLY this JSON object (no prose, no markdown fence):

```json
{
  "entities": [
    {
      "text": "Albert Einstein",
      "label": "Person",
      "confidence": 0.95,
      "properties": {}
    }
  ],
  "relations": [
    {
      "source": "Albert Einstein",
      "target": "Theory of Relativity",
      "type": "AUTHORED",
      "confidence": 0.9,
      "properties": {}
    }
  ]
}
```

Rules:
- ``entities``: every item MUST include a ``confidence`` ∈ [0.0, 1.0].
- ``relations``: every item MUST include a ``confidence`` ∈ [0.0, 1.0].
- ``label`` and relation ``type`` MUST come from the schema or the
  accepted-extension list above. Do not invent new type names.
- ``source``/``target`` must reference an entity ``text`` you also
  extracted in this same response.
- ``properties`` may be ``{}`` if you have nothing to add.
- Confidence is mandatory; entries missing it will be dropped.
"""


def build_pass2_prompt(
    ontology: Ontology,
    chunk_text: str,
    accepted_extensions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Assemble the Pass-2 user prompt for one chunk.

    Args:
        ontology: The ontology the chunk should be extracted against.
        chunk_text: The text of a single chunk. The caller controls
            chunking — this function does not truncate (so a budget
            miss in the validator is loud, not silent).
        accepted_extensions: Optional list of extension dicts from
            ``Pass1Output.proposed_extensions`` that the curator
            (B.3c, conceptually) approved. Empty list / ``None`` is
            equivalent — preserves back-compat with callers that only
            pass an ontology.

    Returns:
        A multi-section prompt string.
    """
    extensions = accepted_extensions or []

    schema_name = ontology.metadata.name if ontology.metadata else "unknown"
    parts: List[str] = [
        "# Pass-2 Typed Extraction",
        "",
        (
            f"Extract entities and relations from the text below "
            f"according to the **{schema_name}** schema."
        ),
        "",
        "## Entity Types",
        "",
    ]
    parts.extend(_format_entity_types(ontology))
    parts.append("")
    parts.append("## Relationship Types")
    parts.append("")
    parts.extend(_format_relationship_types(ontology))

    parts.extend(_format_accepted_extensions(extensions))

    parts.extend(
        [
            "",
            "## Text",
            "",
            chunk_text,
            "",
            _OUTPUT_FORMAT_SECTION,
        ]
    )
    return "\n".join(parts)
