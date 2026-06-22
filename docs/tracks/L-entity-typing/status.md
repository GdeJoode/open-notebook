# Track L — Entity typing fidelity — status

Append-only ledger. One row per phase attempt.

| Phase | Title | PR | Branch | Date | Status |
|---|---|---|---|---|---|
| — | (planning) | — | — | 2026-06-22 | track-planner producing plan.md |

## Basis
- Analysis: `claudedocs/entity-typing-analysis.md` — 27% of entities are `other`, 44% generic concept/topic, because persistence flattens the rich ontology types onto a 20-type enum with no bridge + an English-only alias map (Dutch corpus). type_tags/primary_type empty (rich type lost).
- KEY mechanism: ontologies already declare `parent_type` chains to schema_core schema.org bases → a small fixed schema.org→canonical-enum map (language-agnostic) bridges them. Preserve the ontology type in primary_type/type_tags.
- LANGUAGE: architecture is language-agnostic (typing flows through ontology-declared types); residual free-form-label cleanup stays curated EN+NL per user (98% need). No LLM/embedding fallback for the 2%.
