"""Prompt template for orphan-connector relation confirmation (B.5a).

Given two entities that co-occur in a single chunk, ask the LLM whether
there is a directed relationship between them and, if so, what its
type is. The LLM must return a strict JSON object so the parser stays
trivial.

Token budget
============
Per Q-B-2 the assembled prompt + chunk text is capped at a 1500 estimated
tokens (``len(text) // 4`` heuristic). The cap is **lower** than the
Pass-1 / Pass-2 cap (2400) because:

- Orphan-connector calls fire once per orphan-partner-chunk triple,
  i.e. potentially many per source. A tighter budget caps the LLM
  spend and surfaces oversize evidence chunks early.
- The prompt body is tiny (a system reminder + two entity names + one
  evidence chunk) — the only realistic way to blow the cap is a
  pathological evidence chunk; we want that to be a hard fail, not a
  silent truncation.

Structure
=========
1. **System prompt** (~70 tokens). "You are a precise relation
   extractor … return strict JSON …".
2. **User prompt body** (~50 tokens of scaffolding).
3. **Two entity names** (~10 tokens).
4. **Evidence chunk** (~variable; dominates the budget).
5. **Output schema reminder** (~80 tokens).
"""

from __future__ import annotations


ORPHAN_CONFIRM_SYSTEM_PROMPT = (
    "You are a precise relation extractor. Given two entities that "
    "co-occur in a text passage, decide whether the passage asserts a "
    "directed relationship from the first entity to the second. If yes, "
    "return a concise UPPER_SNAKE_CASE relation_type label (e.g. "
    "WORKS_AT, AUTHORED, PART_OF). If no relation is asserted, return "
    "null for relation_type. Always return a confidence value in the "
    "closed interval [0.0, 1.0]. Output strict JSON only — no prose, no "
    "code fences."
)


def build_orphan_confirm_prompt(
    entity_a: str,
    entity_b: str,
    chunk_text: str,
) -> str:
    """Render the orphan-confirmation user prompt.

    Args:
        entity_a: Surface form of the candidate source entity (the
            orphan in most call sites).
        entity_b: Surface form of the candidate target entity (the
            co-occurring partner).
        chunk_text: The evidence chunk where both entities appear.

    Returns:
        A single string ready to hand to the injected LLM caller. The
        output schema example is included verbatim so the LLM has a
        deterministic target.
    """
    return (
        "# Relation Confirmation\n\n"
        "Two entities co-occur in the passage below. Decide whether "
        "the passage asserts a directed relationship from entity A to "
        "entity B.\n\n"
        f"- Entity A: {entity_a}\n"
        f"- Entity B: {entity_b}\n\n"
        "## Passage\n"
        f"{chunk_text}\n\n"
        "## Output Format\n"
        "Return strict JSON with exactly two keys:\n"
        '- ``relation_type``: an UPPER_SNAKE_CASE label, or `null` when no '
        "relationship is asserted.\n"
        '- ``confidence``: a float in [0.0, 1.0].\n\n'
        "Example (positive):\n"
        '{"relation_type": "WORKS_AT", "confidence": 0.82}\n\n'
        "Example (negative):\n"
        '{"relation_type": null, "confidence": 0.91}\n'
    )
