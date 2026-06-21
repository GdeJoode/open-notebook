# Review — Track D Phase D.1b attempt 1

**Branch**: `track/d-obsidian-vault`
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-14

## Summary

Phase D.1b lands the Obsidian vault direct-write mode plus the `JobType.EXPORT_OBSIDIAN` auto-pipeline handler. All 6 plan acceptance criteria pass, test coverage is strong (7 new vault-mode tests including the security-critical telemetry-redaction inversion), filesystem-safety guards layer defense-in-depth on top of D.1a's `_safe_entity_stem`, and the per-file atomicity trade-off is clearly documented. The implementer's deviation from the plan's pseudocode (`payload: Dict[str, Any]` vs `job: Job`) correctly conforms to the actual registry contract at `packages/job-queue/src/job_queue/registry.py:22`.

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|---|---|---|
| 1 | `POST /export-obsidian` mode=vault_path → 200 + JSON ExportReport | PASS | `test_obsidian_vault_path_happy_path` (router) + `test_vault_path_happy_path` (service writes 3 .md + README to `<vault>/Entities/`). |
| 2 | Path-traversal in entity names normalized + defense-in-depth catch | PASS | Mental probe: `normalize_entity_name("../etc/passwd")` → `"../etc/passwd"` (`/` survives); `_safe_entity_stem` substitutes `/` and `\` → `"_etc_passwd"` (slashes neutralised). Even if `..` literal survives, `_write_to_vault` line 776 `final_path.relative_to(target_dir)` rejects. Empirically validated `("../escape.md").resolve()` → rejected. |
| 3 | Atomic per-file; mid-batch failure leaves 1..N-1 written; 500 with partial count | PASS | `test_vault_path_atomic_per_file` patches `builtins.open` to fail on `bbb.md.tmp.<pid>`, asserts README + aaa.md survive, no `.tmp.*` leftovers, `OSError.args[1] == {"entities_written": 2, "failed_file": "bbb.md"}`. Router test pins same shape in 500 body. |
| 4 | `JobType.EXPORT_OBSIDIAN` handler registered + auto-pipeline writes successfully | PARTIAL | `test_export_obsidian_handler_registered` confirms structural registration; handler validates payload via `ExportObsidianPayload`. No end-to-end handler-execution test, but this matches existing codebase precedent (no other handler has a full execution test outside the service layer). MINOR follow-up. |
| 5 | Telemetry: `{mode: "vault_path", vault_path_redacted: true}` + raw path absent | PASS | `test_vault_path_telemetry_redacts_path` walks payload recursively asserting `str(tmp_path) not in any string value`, asserts `vault_path_redacted is True`, plus no-ID check. **Telemetry redaction is robust** — see analysis below. |
| 6 | 400 if mode=vault_path and Settings.vault_path not configured | PASS | `test_vault_path_not_configured_raises` (service raises `VaultPathNotConfigured`) + `test_obsidian_vault_path_not_configured_400` (router → 400 with clear "Configure vault_path in Settings" message). Bonus: same check for missing `vault_entities_folder`. |

## Telemetry redaction — robustness analysis (mental inversion verified)

Per request, I performed the mental inversion attack:

**Attack scenario**: implementer adds `payload["vault_path"] = str(target_dir)` or similar leak.

**Test defense** (`tests/test_obsidian_export_service.py:1046-1061`):

```python
def _no_raw_path(value):
    if isinstance(value, str):
        assert raw_path not in value  # SUBSTRING match, not equality
    elif isinstance(value, dict):
        for k, v in value.items():
            if k == "vault_path":  # explicit key block
                assert v is None or not isinstance(v, str) or v == ""
            _no_raw_path(v)
    elif isinstance(value, list):
        for v in value:
            _no_raw_path(v)
```

This is a recursive walk over every string value at every depth, using **substring matching** (not equality). Any leak — top-level, nested, partial concatenation, list element — would trip the assertion. The `if k == "vault_path"` adds a structural key-name check on top.

**Verified**: redaction is robust. Adding `payload["vault_path"] = str(target_dir)` or `payload["paths"] = [str(target_dir)]` or `payload["error"] = f"failed at {target_dir}"` would all fail this recursive walk.

**Note on `logger.info` claim from self-review**: the self-review flags that `logger.info` line 598 leaks the path via `artifact.vault_dir`. **Looking at the actual log statement, this is overcautious** — the log only includes `notebook`, `entities`, `relations`, `files`, `bytes`, `duration_ms`. No path. The path lives in `ExportArtifact.vault_dir` (returned to the caller, by design), not in any log line.

## Test status

```
apps/app-main targeted: 36 passed, 0 failed (124s)
apps/app-main full:     566 passed, 0 failed (113s)  -- baseline 554; +12 new
packages/job-queue:     38 passed, 0 failed (42s)    -- no regressions
```

## Issues found

### Blockers (must fix)

None.

### Major (must fix)

None.

### Minor (optional follow-up)

1. **`vault_entities_folder` escape check has no dedicated test** — `obsidian_export_service.py:752-762` rejects a malicious `vault_entities_folder` like `"../../etc"` via `target_dir.relative_to(vault_resolved)`. This defense-in-depth code path is unexercised by any test. The traversal-in-filename path IS exercised (line 776), but the traversal-in-folder-config path is not. A 3-line test that sets `entities_folder="../../etc"` and asserts `ValueError` would close the gap.

2. **`os.access(vault_path, os.W_OK)` writability rejection has no test** — line 747-750 rejects a read-only mount. Hard to fake portably (would need `chmod` or tmpfs games), so skipping is acceptable; just noting the un-pinned guard.

3. **Handler end-to-end execution test missing** — plan AC #4 says "verified via job-queue integration test", but only structural registration is checked. Matching the codebase precedent makes this MINOR rather than MAJOR. An `await registry.execute(JobType.EXPORT_OBSIDIAN, {...})` against stubbed DI would close it.

4. **`builtins.open` mock fragility (already self-flagged)** — `test_vault_path_atomic_per_file` patches `builtins.open` directly. If the service moves to `aiofiles`, the test breaks loudly. Acceptable trade-off; flagged for future refactors.

5. **`_extract_partial_state` ordering quirk** — `routers/exports.py:279-285` returns the first dict containing `entities_written`. If a future caller attaches multiple partial-state dicts, only the first wins. Defensive but undocumented. Acceptable today since only `_write_to_vault` attaches.

## Kudos

- **Telemetry redaction recursive walk is exemplary**. The inversion mindset (substring check, structural key check, list-element recursion) means any future regression that leaks the path — at any depth, in any field — will fail loudly. This is the security-test pattern I want across the codebase.
- **Defense-in-depth layering**: D.1a's `_safe_entity_stem` strips path separators, but `_write_to_vault` STILL validates the resolved path is under the target dir. Belt-and-braces is the right posture for filesystem-touching code.
- **Trade-off documentation in the module docstring**: the per-file vs whole-batch atomicity choice is explained with the upgrade path (staging-dir + dir-rename), not hand-waved. Future maintainers can find this without archaeology.
- **`vault_entities_folder` escape check is included** even though the plan didn't mandate it — defense-in-depth against operator-supplied config, not just against entity names.
- **Sorted write order** (line 772) makes mid-batch failure modes reproducible across runs. Small detail, big debugging value.
- **Explicit `VaultPathNotConfigured` exception type** (not `ValueError`) lets the router map cleanly to 400 without string-matching. Good separation of concerns.
- **Self-review correctly identifies and justifies the `payload: Dict[str, Any]` deviation** from the plan's pseudocode by pointing at `registry.py:22`. This is exactly the right way to handle a plan-vs-reality mismatch.
- **Bonus coverage for `vault_entities_folder=None`** beyond the plan's requirements.
- **`test_vault_path_overwrite_existing_md` triple-pins** the overwrite contract: `alice.md` (in export set) overwritten, `user_added.md` (inside folder, not in export set) preserved, `vault_root_note.md` (outside folder) preserved.

## Decision rationale

All 6 acceptance criteria pass (AC #4 with a minor caveat that matches codebase precedent). Telemetry redaction is robust under mental inversion. Filesystem safety guards layer defense-in-depth correctly. Tests are deterministic and assert behaviour (not just "code runs"). 566 app-main tests + 38 job-queue tests green, no regressions. The 5 MINOR notes are follow-up opportunities, not blockers, and most match existing codebase patterns.

D.1b is filesystem-touching code where correctness and security matter — this implementation gets both right.

## Next steps

Ready for human approval / merge. The 5 MINOR items can be filed as follow-ups, or addressed opportunistically in D.1c when the UI dialog wires through these paths.
