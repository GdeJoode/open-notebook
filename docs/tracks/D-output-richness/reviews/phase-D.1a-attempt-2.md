# Review — Track D Phase D.1a attempt 2

**Branch**: `track/d-obsidian-zip` (HEAD `bd29285`)
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-14

## Summary

Both attempt-1 majors are genuinely fixed. M1 (`_safe_entity_stem` layering `_ENTITY_FILENAME_UNSAFE` on top of `normalize_entity_name`, collision counter on sanitized stem, stem cap at 200) — mental inversion verified: revert the regex and three layers of the regression test would fail (zip namelist contains `hello/world.md`, expected stems absent, wikilink in neighbour body would be `[[hello/world]]`). M2 (`min_connections` boundary pin at degree==N) — mental inversion verified: flip `>=` to `>` and `carol.md in files` assertion fails. Minors 1–5 closed; Minor 6 (notebook name plumbing) defensibly deferred to D.1c. Test count 552 → 554 (+2 regression-pin tests). `apps/app-main`: 554 passed. `packages/shared`: 199 passed.

## Acceptance criteria check

All 9 ACs from plan §D.1a now PASS, including the boundary-pin for AC #9 and the flat-vault promise for AC #2 (previously violated).

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `POST /export-obsidian` zip mode → 200 + `application/zip` + zipfile-parseable | PASS | `test_obsidian_zip_happy_path` (router) + `test_three_entities_two_relations_render_correctly` (service) |
| 2 | Flat zip: `README.md` + one `.md` per entity, filenames = `normalize_entity_name(canonical_name) + ".md"` | PASS | `test_filename_sanitization_strips_slash_and_colon` now closes the slash-leak gap from attempt-1; `normalize_entity_name` is layered with `_ENTITY_FILENAME_UNSAFE` so the flat-vault promise holds for unsafe chars too |
| 3 | Frontmatter has roadmap §340 keys (id/type/confidence/external_ids/aliases/sources) | PASS | Snapshot + inversion against `obsidian_export_golden.md` |
| 4 | Wikilinks resolve; broken-target relations silently dropped (Q-D-4) | PASS | `test_broken_wikilinks_silently_dropped` + sanitized-stem assertion inside `test_filename_sanitization_strips_slash_and_colon` |
| 5 | README has filter snapshot + counts + top-20 list | PASS | `test_readme_index_contains_required_sections` |
| 6 | `metrics{event_type: "export.obsidian"}` fires exactly once per export | PASS | `test_telemetry_emits_export_obsidian_with_counts_only` + `test_telemetry_records_failure_partial` (failure-path) |
| 7 | Snapshot vs golden; inversion fails | PASS | `test_snapshot_against_golden` + `test_snapshot_inversion_detects_drift` (verified strong in attempt-1 review with mental probes) |
| 8 | Filename collision: `Smith` + `smith.` → `smith.md` + `smith-2.md` | PASS | `test_filename_collision_appends_suffix`; also live-verified 3-way collision `Hello/World` + `hello/world` + `Hello_World` → `hello_world.md` + `hello_world-2.md` + `hello_world-3.md` |
| 9 | `min_connections=N` excludes degree<N | PASS | `test_min_connections_boundary_at_exact_threshold` pins inclusive `>=` semantic; degree==2 INCLUDED, degree==1 EXCLUDED at threshold==2 |

## Test status

```
apps/app-main:
  tests/test_obsidian_export_service.py: 14 passed
  tests/test_exports_router.py: 10 passed (6 networkx + 4 obsidian)
  full suite: 554 passed in 114.58s (0:01:54)
packages/shared: 199 passed in 3.80s
```

## Mental inversion verification (M1 + M2)

### M1 — filename sanitization

Performed two-step mental inversion of `test_filename_sanitization_strips_slash_and_colon`:

**Inversion A — revert `_safe_entity_stem` to `normalize_entity_name`**:
1. Entity `Hello/World` → stem `hello/world` → zip entry `hello/world.md`.
2. Assertion `"hello_world.md" in files` → would FAIL (entry is `hello/world.md`).
3. Assertion `assert "/" not in name` inside the file-list loop → would FAIL for `hello/world.md`.
4. Wikilink rendering: stem fed to `[[…]]` would be `hello/world` (other_filename = `hello/world.md`; `other_filename[:-3]` = `hello/world`).
5. Assertion `assert "[[hello_world]] (knows)" in neighbour_body` → would FAIL.
6. Assertion `assert "[[hello/world]]" not in neighbour_body` → would FAIL.

Three independent assertions in the test fire on the buggy code. Test is a real regression pin, not pseudo-coverage.

**Inversion B — keep regex but do not extend to backslash**: `Path\To\Note` → assertion `assert "\\" not in name` would FAIL. Each separator class is independently asserted.

Also live-verified end-to-end:
- `_safe_entity_stem("Hello/World")` → `"hello_world"`
- `_safe_entity_stem("///")` → `"___"` (3 underscores; valid stem; no fallback needed)
- `_safe_entity_stem("")` → `""` → `_build_filename_map` substitutes `"unnamed"`
- 3-way collision `Hello/World` + `hello/world` + `Hello_World` → `hello_world.md` + `hello_world-2.md` + `hello_world-3.md` (correct)
- 250-char stem → truncated to 200 chars (cap honoured)

### M2 — `min_connections` boundary

Performed mental inversion of `test_min_connections_boundary_at_exact_threshold`:

**Inversion — flip `>=` to `>` at line 532**:
- Setup: alice degree=1, bob degree=3, carol degree=2; `min_connections=2`.
- Buggy `>`: alice (1>2 → False, excluded), bob (3>2 → True, included), carol (2>2 → False, EXCLUDED).
- Assertion `assert "carol.md" in files` → would FAIL with the explicit "Boundary regression" message.

Live-verified: `_apply_min_connections_filter(ents, rels, min_connections=2)` returns `{entity:bob, entity:carol}`; same call with `min_connections=3` returns `{entity:bob}` only. Boundary inclusion confirmed.

## Minors check (all 6 from attempt-1)

1. **Wasted recomputation in `finally`** — RESOLVED. `filename_map`, `rendered_relation_count`, `files_written` are pre-bound locals (lines 309–311) and cached after the success-path computation (lines 360–363). The `finally` block reads from locals (line 425–426). `_build_filename_map` and `_count_rendered_relations` are now called exactly once per successful export. On failure paths the locals retain their pre-bound defaults (0/empty) so the payload still reflects "how far we got".
2. **`type_tags` / property YAML escaping** — DOCUMENTED (module docstring lines 60–67) as a V1 limitation with explicit deferral to D.2's YAML frontmatter pass. Acceptable.
3. **Filename length cap** — RESOLVED. `_MAX_STEM_LEN=200` defined at line 134, applied inside `_safe_entity_stem` at lines 160–161. Leaves 55 bytes for `-99` collision suffix + `.md` extension under the typical 255-byte filesystem limit.
4. **`min_connections` degree semantics** — DOCUMENTED (module docstring lines 69–80). The choice (degree computed over all relations including those whose other endpoint dropped) is explicit so an operator interpreting the filter knows which graph they're counting against.
5. **Unused `all_entities` param** — RESOLVED. Signature dropped (`_render_entity_markdown(entity, relations, filename_map)`). Only a doc-back-reference remains at line 650 explaining the removal. Snapshot tests at lines 536 and 563 use the new 3-arg form correctly.
6. **Notebook name in README** — DEFERRED to D.1c with defensible rationale (the FE dialog is the natural place for notebook-name plumbing through the router; D.1c plan explicitly opens that dialog and threads notebook context). Acceptable.

## Edge-case probes (live-verified, none failed)

- 3-way collision after sanitization: `Hello/World` + `hello/world` + `Hello_World` → `hello_world.md` + `hello_world-2.md` + `hello_world-3.md`.
- Empty `canonical_name` → `_safe_entity_stem` returns `""` → `_build_filename_map` substitutes `"unnamed"` → `unnamed.md`. Multiple empty names collide on `unnamed`, get `unnamed-2.md`, `unnamed-3.md` correctly.
- `canonical_name="///"` → `_safe_entity_stem` returns `"___"` (3 underscores; valid stem; no fallback).
- 250-char `canonical_name` → stem truncated to 200 chars.
- Tab character in name → `normalize_entity_name` collapses to space before the sanitization regex ever sees the `\t`. No surprise.
- Windows-unsafe chars (`<`, `>`, `|`, `?`, `*`) — explicitly included in `_ENTITY_FILENAME_UNSAFE`; the regex extends B.2b's pattern. Live-verified `a<b>c|d?e*f` → `a_b_c_d_e_f`.
- Null byte (`\x00`) — included in regex; live-verified replaced with underscore.

## Kudos

- The regression test asserts THREE separate aspects of the M1 fix: (a) zip namelist is flat, (b) specific sanitized stems are present, (c) wikilink body uses sanitized stem. Any one of them on its own would catch the regression; together they document the contract crisply.
- The M2 boundary test layout is genuinely surgical: three relations carefully chosen to produce explicit degrees 1/3/2 so the test stresses exactly the `==` boundary, not the `>>` case. The docstring also explicitly explains the inversion-failure mode ("a buggy `>` would still pass `test_min_connections_filter_excludes_isolates`...") — exactly the RETRO #6 inversion-test pattern.
- Minor #1 (wasted recomputation) was addressed structurally — pre-binding locals — not by adding a memo decorator or recompute guard. The pattern reads cleanly: failure-path still produces a well-formed payload because the locals carry their defaults.
- `_safe_entity_stem` regex extends B.2b with `< > | ? *` so the stem survives Windows + POSIX + `Content-Disposition` uniformly. The choice to replace with `_` (not strip) is documented inline ("visible round-trip instead of silent fusion") — exactly the kind of decision-explanation that survives reviewer turnover.
- The `_render_entity_markdown` docstring on line 650 explicitly notes that the `all_entities` param was dropped and why. No silent signature change.

## Decision rationale

0 blockers, 0 majors, 0 unresolved minors. Both attempt-1 majors are genuinely fixed (mental inversion verifies). Five of six minors are resolved; the sixth has a defensible deferral to a downstream phase (D.1c) that owns the relevant plumbing. Test count grows by exactly 2 (the two pin tests claimed). Full suite stays green. → **APPROVED**.

## Next steps

Ready for human approval / merge. No follow-up required for D.1a. D.1b should expect to inherit `ExportArtifact.vault_dir` slot + the `NotImplementedError`-raising path as a body-only fill-in.
