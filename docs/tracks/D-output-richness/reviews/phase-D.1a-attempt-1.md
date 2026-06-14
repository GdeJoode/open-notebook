# Review — Track D Phase D.1a attempt 1

**Branch**: `track/d-obsidian-zip` (HEAD `55516da`)
**Decision**: **REVISIONS_NEEDED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-14

## Summary

Service well-structured; D.3 precedents (status post-filter, counts-only telemetry) consistently applied; snapshot+inversion genuinely strong (proven by mental probe: key reorder + internal whitespace + blank lines all break). 22 new tests pass, full app-main suite 552 passing. 9 plan ACs mechanically met in narrow fixtures. **BUT one realistic edge case breaks AC #2's flat-vault promise**: `/` `\` `:` in `canonical_name` produce nested zip paths.

## Acceptance criteria

8/9 PASS; AC #2 FAIL for filename-unsafe chars; AC #9 PARTIAL (no boundary test).

## Snapshot pattern assessment (per RETRO #6)

**Genuinely strong**, not pseudo-coverage. Mental probes verified:
- Frontmatter key reorder → FAILS
- Trailing whitespace on internal heading → FAILS
- Mid-body blank lines → FAILS

`rstrip()` normalization documented; only swallows trailing whitespace from editor auto-trim. Does NOT mask internal whitespace drift or key reordering. Materially stronger than `== model_default` pseudo-coverage RETRO #6 warned about.

## Majors (2)

### M1: Per-entity filenames not sanitized for filesystem-unsafe chars

`obsidian_export_service.py:452-467` (`_build_filename_map`) + `:257-260` (`archive.writestr(filename, ...)`). `normalize_entity_name` only lowercases + collapses whitespace + strips trailing punct — does NOT strip `/`, `\`, internal `:`, control chars, or NUL.

**Reproduction (verified live)**:
```
Entity(canonical_name="Hello/World") → zip entry: "hello/world.md"
```
User extracts zip → gets phantom `hello/` directory containing `world.md`. Violates plan §116 "flat Obsidian vault directory" + AC #2 "filenames follow `normalize_entity_name(canonical_name) + ".md"`" (implicitly flat).

Realistic entities with slashes exist: URLs (`openai.com/gpt-4`), paths, fractions, schema namespaces.

Wikilinks in body become `[[hello/world]]` — Obsidian partly handles as folder-note reference, different semantics than intended.

**Fix recommendation**: B.2b TTL exporter solved this with `_FILENAME_UNSAFE_RE`. Apply same sanitization to per-entity filenames (after `normalize_entity_name`, before zip-write). Rebuild collision tracking on sanitized stem. Add regression test covering `/` and `:` in `canonical_name`.

### M2: AC #9 not tested at the boundary

Test uses `min_connections=1` with 5 entities + 1 isolate. Self-review flags as "same code path". TRUE for simple isolate case but does NOT exercise off-by-one boundary at degree==`n-1` vs degree==`n`.

A buggy `>` (strict greater-than) implementation would still pass the current test on degree-1 vs degree-0 split.

**Fix**: extend test with entity at degree `n-1` vs degree `n` (e.g., `min_connections=2`, one entity at degree 1, another at degree 2). Pins the boundary explicitly.

## Minors (6, non-blocking)

1. Wasted recomputation in `finally` block: `_build_filename_map` + `_count_rendered_relations` called twice. Both idempotent so OK for correctness; meaningful cost at 10K entities.
2. `type_tags`/property values with embedded delimiters not escaped — `type_tags=["foo, bar"]` renders as `[foo, bar]` (strict YAML parses as 2 tags). Acceptable V1, document.
3. Filename length not capped — 300-char `canonical_name` → 303-char filename. ext4/NTFS cap at 255. Truncate to ~200 + collision suffix.
4. `min_connections` degree includes relations to filtered-out endpoints. Semantically borderline; document choice.
5. `_render_entity_markdown` `all_entities` param unused — drop from signature or wire up.
6. Notebook name falls back to ID — flagged in self-review, acceptable V1.

## Kudos

- Recursive `_no_ids` walker in telemetry test defends Q-D-8 against nested-dict ID leaks
- Failure-path telemetry test (`partial: True`) — thoughtful addition not strictly required
- 3-way Smith collision + `smith.` post-normalize collision work without explicit tests (robust beyond what's tested)
- `ExportArtifact` already has `vault_dir` slot + `NotImplementedError` paths — D.1b is genuinely body-only
- Snapshot+inversion strong (proven by 3-way mental probe)
- Self-review honestly flags gaps (min_connections=1 vs =5, recompute cost, notebook-name fallback)

## Decision rationale

0 blockers; 2 majors (slash-in-filename violates flat-vault promise; AC#9 boundary not tested). ≥1 major → REVISIONS_NEEDED. Implementation quality otherwise high.

## Next steps

1. Sanitize per-entity filenames before zip-write (apply `_FILENAME_UNSAFE_RE` after `normalize_entity_name`, rebuild collision tracking on sanitized stem). Regression test covering `/` and `:` in `canonical_name`.
2. Extend `min_connections` test with boundary case (degree `n-1` vs `n`).
3. Optionally address minors (especially wasted recomputation).
