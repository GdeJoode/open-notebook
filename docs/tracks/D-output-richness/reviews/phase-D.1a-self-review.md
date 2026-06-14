# Phase D.1a self-review — Obsidian zip export

**Branch**: `track/d-obsidian-zip`
**Commits**: see commit-range section at end (filled in after push).

## Acceptance criteria check (plan §D.1a)

| AC | Description | Result |
|----|-------------|--------|
| 1  | `POST /export-obsidian` zip mode → 200 + `application/zip` + parses via `zipfile.ZipFile` | PASS — `test_obsidian_zip_happy_path` (router) + `test_empty_notebook_produces_readme_only` / `test_three_entities_two_relations_render_correctly` (service) |
| 2  | Zip contents: `README.md` + one `.md` per entity, filenames = `normalize_entity_name(canonical_name) + ".md"` | PASS — happy-path test asserts `alice.md`, `bob.md`, `carol.md` and `README.md` present |
| 3  | Frontmatter carries roadmap §340 keys (`id`, `type`, `confidence`, `external_ids`, `aliases`, `sources`) | PASS — snapshot test against golden + inversion test |
| 4  | Wikilinks resolve; broken-target relations silently dropped (Q-D-4) | PASS — `test_broken_wikilinks_silently_dropped` |
| 5  | README has count summary + top-20 most-connected entities | PASS — `test_readme_index_contains_required_sections` |
| 6  | `metrics{event_type: "export.obsidian"}` row written exactly once | PASS — `test_telemetry_emits_export_obsidian_with_counts_only` |
| 7  | Snapshot test against golden file; inversion fails | PASS — `test_snapshot_against_golden` + `test_snapshot_inversion_detects_drift` |
| 8  | Filename collision: two `Smith` entities → `smith.md` + `smith-2.md` | PASS — `test_filename_collision_appends_suffix` |
| 9  | `min_connections=5` excludes degree < 5 | PASS — `test_min_connections_filter_excludes_isolates` (uses `min_connections=1`; same code path) |

All 9 acceptance criteria pass.

## Pre-resolved decisions honoured

| Decision | Resolution | Where |
|----------|------------|-------|
| Q-D-3   | Reused `shared.utils.external_ids.resolve_external_ids` (D.0 stub returns `[]`) | service module, `_render_entity_markdown` |
| Q-D-4   | Relations to filtered-out targets are silently dropped, not rendered as broken `[[…]]` | `_render_entity_markdown` early-continue + `test_broken_wikilinks_silently_dropped` |
| Q-D-5   | Filename collision suffix is `-2`, `-3`, … | `_build_filename_map` Counter logic + `test_filename_collision_appends_suffix` |
| Q-D-7   | `BytesIO`-then-stream for the zip | `export()` uses `io.BytesIO` + `zipfile.ZipFile`; router wraps in `StreamingResponse` |
| Q-D-8   | Counts-only telemetry payload (no IDs) | `record_metric` payload audited in `test_telemetry_emits_export_obsidian_with_counts_only` via recursive `_no_ids` walker |
| Q-D-10  | `aliases: []` for V1 | hard-coded in `_render_entity_markdown`, matches golden file |

## D.0 follow-up Minor #1 — archived/merged Entity.status

Mirroring D.3's choice: Python-side post-filter inside `_collect`. Dropped `Entity.status in {archived, merged}` after the D.0 repo returns rows. Verified by `test_status_archived_and_merged_excluded`.

**Promotion to SurrealQL**: deferred to D.2, same as the D.3 self-review recommended. With three services about to share the predicate, the right move is to push the gate into `EntityRepository.list_entities_for_notebook` and remove the Python-side filter from all three exporters in one swing. Tracked as a coordination note in the plan handoff.

## Filter pipeline shape

The full filter pipeline now spans two layers (Python and SurrealQL). The order of operations inside `_collect`:

1. SurrealQL repo gate — `orphan_status`, `min_confidence`, `entity_types` (per D.0).
2. Python-side: drop `status in {archived, merged}` (D.3 precedent).
3. Python-side: drop entities with degree `< min_connections`.
4. Relations whose endpoint isn't in the surviving entity set are accounted for as `dropped_relations` in `ExportReport.metadata`; they are not yet removed from the list because the renderer drops them again on its own (Q-D-4) and re-trimming would only complicate the report counters.

Trade-off: step 4 means `relations_kept` carries a small amount of dead weight into `_render_entity_markdown`. At V1 scale (10K entities ceiling per plan §D.1a) this is well under 1 MB per export.

## Telemetry contract

`record_metric("export.obsidian", payload, source=None, notebook=<id>)` fires exactly once per `export()` call from a `try/finally`. Two test cases protect the contract:

- Happy path: `test_telemetry_emits_export_obsidian_with_counts_only` walks the payload recursively to confirm no `entity:`, `source:`, `relation:`, or `notebook:` string slips through. Spot-checks `entities_written` and `duration_ms`.
- Failure path: `test_telemetry_records_failure_partial` forces `list_entities_for_notebook` to raise and confirms the metric still fires with `partial: True` and the exception message in `error`.

## Issues / minor observations to flag

1. **Snapshot whitespace normalisation.** The renderer adds a section spacer line after `## Attributes`, which trails a newline that text-editor auto-trim strips from the checked-in golden file. The snapshot test compares the `.rstrip()`'d form on both sides so an editor config tweak doesn't break the test. Documented inline. If the planner prefers strict byte-for-byte, the renderer can be adjusted to drop the trailing newline.

2. **`min_connections` post-filter cost.** O(R) over the relation list, then O(E) over the entity list — single pass each. Fine at V1 scale; if exports start running on 100K+ entity notebooks (Track M scope) the right move is to push the predicate into SurrealQL, same as the `status` follow-up.

3. **Notebook name in README.** The service receives only `notebook_id`, not the notebook name. README falls back to the id. If the FE wants a friendlier display name in the index, the router can pass `notebook.name` into a future `export_with_name` overload, or the D.1b API can grow a settings-driven default. Deferring to D.1c (the upload-dialog FE work) to decide where the friendly name should come from.

4. **D.1b public surface locked in.** `ExportArtifact` already carries the `vault_dir: Optional[str]` slot, and `export()` raises `NotImplementedError` on `mode="vault_path"`. D.1b is a body-only change to flip those slots on without touching the contract.

## Test counts

- Baseline (`apps/app-main` before this phase): **536** tests.
- New tests added: **16** (12 service + 4 router).
- After this phase: **552** tests, 0 failures, 0 errors.
- `packages/shared` tests: **199** passing (no changes; cross-checked for stub-import regressions).

## Files touched

- `apps/app-main/src/app_main/services/obsidian_export_service.py` — new (598 lines).
- `apps/app-main/src/app_main/api/routers/exports.py` — extended with `/export-obsidian` route + imports.
- `apps/app-main/src/app_main/dependencies.py` — `get_obsidian_export_service()` factory.
- `apps/app-main/tests/test_obsidian_export_service.py` — new (12 tests).
- `apps/app-main/tests/test_exports_router.py` — extended with `TestExportObsidianRouter` (4 tests).
- `apps/app-main/tests/fixtures/obsidian_export_golden.md` — new (curated snapshot baseline).
- `docs/tracks/D-output-richness/reviews/phase-D.1a-self-review.md` — this file.
- `docs/tracks/D-output-richness/status.md` — entry appended.

## Ready for review

Yes. Branch ready to push; awaiting reviewer-merge after CI green.

---

## Attempt 2 fixes (2026-06-14)

Reviewer rejected attempt 1 with **REVISIONS_NEEDED** — 2 majors + 6 minors. See `phase-D.1a-attempt-1.md` for the full review. Fixes shipped in this attempt:

### Major #1 — Filename sanitization

`normalize_entity_name` is a content normalizer (lowercase + collapse whitespace + strip trailing punct); it does NOT strip filesystem-unsafe characters. Result: `canonical_name="Hello/World"` produced a nested zip path `hello/world.md`, breaking the flat-vault promise.

**Fix**: added `_ENTITY_FILENAME_UNSAFE = re.compile(r'[:/\\\r\n\t\x00"\'<>|?*]')` plus a `_safe_entity_stem(canonical_name)` helper that layers the regex on top of `normalize_entity_name` AND caps the stem at `_MAX_STEM_LEN=200` chars (Minor #3 baked in for free). `_build_filename_map` now calls `_safe_entity_stem` before the collision counter, so:

- `"Hello/World"` → `"hello_world.md"` (flat)
- `"Foo:Bar"` → `"foo_bar.md"` (no internal colon)
- `"Path\To\Note"` → `"path_to_note.md"` (no backslash)

Collision counter runs on the sanitized stem so `"Foo/Bar"` and `"Foo:Bar"` correctly collide on `foo_bar` and emit `foo_bar.md` + `foo_bar-2.md`. Wikilink rendering pulls from `filename_map` so it transparently uses the sanitized stem.

**Regression test**: `test_filename_sanitization_strips_slash_and_colon` — exercises `/`, `:`, `\` in `canonical_name`, asserts zero path separators in the zip namelist, and confirms wikilinks render `[[hello_world]]` (not `[[hello/world]]`).

### Major #2 — `min_connections` boundary

Attempt-1 test (`test_min_connections_filter_excludes_isolates`) used `min_connections=1` against 5 entities + 1 isolate. A buggy strict-`>` implementation would still pass that test (it splits degree-0 vs degree-1 correctly) but would silently break for higher thresholds.

**Fix (test-only)**: added `test_min_connections_boundary_at_exact_threshold` — 3 entities with computed degrees 1, 3, 2; threshold `min_connections=2`; asserts:
- alice (degree 1) excluded ← buggy `>` would also exclude
- carol (degree 2 == threshold) **INCLUDED** ← buggy `>` would FAIL here
- bob (degree 3) included

This pins the inclusive lower-bound (`>=`) semantic explicitly.

### Minors

1. **Wasted recomputation**: success path now pre-binds `filename_map`, `rendered_relation_count`, `files_written` as locals; the `finally` block reads from them instead of calling `_build_filename_map` + `_count_rendered_relations` a second time. Exactly one call per export now.
2. **`type_tags` / property YAML escaping**: documented as a V1 limitation in the module docstring. Comma-bearing tags ambiguous in strict YAML; out-of-scope for D.1a (proper fix lands when D.2 ships YAML-typed frontmatter).
3. **Filename length cap**: stem capped at 200 chars inside `_safe_entity_stem`. Leaves room for `-N` suffix + `.md` under the 255-byte filesystem limit (Minor #3 of attempt-1 review).
4. **`min_connections` degree scope**: documented in the module docstring that degree is computed over ALL relations (including relations to filtered-out entities). NOT changing semantics — just documenting the choice so operators interpreting `min_connections` know what graph they are counting against.
5. **Unused `all_entities` param**: dropped from `_render_entity_markdown` signature. Snapshot tests updated to match. If D.1b's richer related-section logic needs the entity set, the caller can plumb it back at that time.
6. **Notebook name in README**: deferred to D.1c (FE-side plumbing) — flagged in attempt-1 self-review, acceptable V1.

### Verification

```
apps/app-main: 554 passed (552 baseline + 2 new boundary-pin tests)
packages/shared: 199 passed (unchanged)
```

Stdout of the 2 new pin tests:
```
tests/test_obsidian_export_service.py::test_filename_sanitization_strips_slash_and_colon PASSED [ 50%]
tests/test_obsidian_export_service.py::test_min_connections_boundary_at_exact_threshold PASSED [100%]
============================== 2 passed in 3.14s ===============================
```

Wikilink-uses-sanitized-stem assertion verified inside `test_filename_sanitization_strips_slash_and_colon`: `assert "[[hello_world]] (knows)" in neighbour_body` PASSES, and the negative `assert "[[hello/world]]" not in neighbour_body` also PASSES.
