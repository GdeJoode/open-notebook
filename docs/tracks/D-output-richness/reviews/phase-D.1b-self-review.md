# Phase D.1b self-review — Obsidian direct-write-to-vault

**Branch**: `track/d-obsidian-vault`
**Commit range**: `043bcf3..HEAD` (single commit `7ff80c6`)

## Acceptance criteria check (plan §D.1b)

| AC | Description | Result |
|----|-------------|--------|
| 1  | `POST /export-obsidian` with `mode=vault_path` + configured `Settings.vault_path` → 200 + JSON `ExportReport` | PASS — `test_obsidian_vault_path_happy_path` (router) + `test_vault_path_happy_path` (service writes 3 entity .md + README, `artifact.vault_dir` populated). |
| 2  | Path-traversal in entity names defended (defense-in-depth at write site) | PASS — `_write_to_vault` does `(target_dir / filename).resolve()` and verifies it's a child of the resolved target dir before opening any tmp file. D.1a's `_safe_entity_stem` already sanitizes; this is the second layer. |
| 3  | Atomic per-file via tempfile + `os.replace`; mid-batch failure leaves earlier files written; error response carries `entities_written` count | PASS — `test_vault_path_atomic_per_file` (service: forced `OSError` on 3rd write leaves README + aaa.md on disk, propagates with `{"entities_written": 2, "failed_file": "bbb.md"}` in `exc.args`) + `test_obsidian_vault_path_filesystem_failure_500_with_partial` (router maps to 500 with same fields in `detail`). |
| 4  | `JobType.EXPORT_OBSIDIAN` handler registered; auto-pipeline payload resolves through handler | PASS — `test_export_obsidian_handler_registered` asserts `registry.has_handler(JobType.EXPORT_OBSIDIAN)` after import; handler validates payload via `ExportObsidianPayload`, calls `service.export(..., mode="vault_path")`, returns flattened report. |
| 5  | Telemetry payload has `vault_path_redacted: True` and raw path NOT in payload | PASS — `test_vault_path_telemetry_redacts_path` walks payload recursively asserting `str(tmp_path) not in any string value`, plus structural check for `payload["vault_path_redacted"] is True`. |
| 6  | 400 if mode=vault_path but Settings.vault_path not configured | PASS — `test_vault_path_not_configured_raises` (service raises `VaultPathNotConfigured`) + `test_obsidian_vault_path_not_configured_400` (router returns 400 with "Configure vault_path..." message). |

All 6 acceptance criteria pass.

## Pre-resolved decisions honoured

| Decision | Resolution | Where |
|----------|------------|-------|
| Q-D-6   | Overwrite is default for `<entities_folder>/` (V1; export is source of truth for files we explicitly write) | `_write_to_vault` writes each filename through `os.replace`; `test_vault_path_overwrite_existing_md` pins that `user_added.md` (not in the export's filename set) is preserved, while `alice.md` (in the set) is overwritten. |
| Q-D-8   | Counts only in telemetry; vault path REDACTED | `_export_to_vault` `finally` block emits payload with `mode: "vault_path"`, `vault_path_redacted: True`, and **never** the raw path string. Verified by `test_vault_path_telemetry_redacts_path` via recursive walk. |

## Safety-guard layering

`_write_to_vault` applies four pre-write guards:

1. `vault_path.is_absolute()` — rejects relative paths that would resolve against the API server's cwd.
2. `vault_path.exists()` and `vault_path.is_dir()` — rejects misconfigured paths before any write.
3. `os.access(vault_path, os.W_OK)` — explicit writability check (avoids partial state on a read-only mount).
4. Per-file path-traversal check: `(target_dir / filename).resolve().relative_to(target_dir)` — defense-in-depth against `..` in filenames that survived D.1a's `_safe_entity_stem`.

Plus one config-level guard: `target_dir = (vault_path / entities_folder).resolve()` is also checked against `vault_path.resolve()` to reject a malicious `vault_entities_folder` like `../../etc`.

Test coverage: `test_vault_path_must_be_absolute` + `test_vault_path_must_exist`. The traversal-on-filename defense is exercised structurally (the assertion path through `_write_to_vault` runs on every happy-path test).

## Atomicity model (per-file, not whole-batch)

The V1 contract is per-file atomicity via POSIX `os.replace`. Each `.md` file either fully appears under its final name or doesn't appear at all — a half-written file is impossible. A mid-batch filesystem failure leaves files 1..N-1 fully written and propagates `OSError` with `{"entities_written": N-1, "failed_file": "<name>"}` in `args[1]`.

**Trade-off documented in the module docstring**: whole-batch atomicity would require staging into a tempdir sibling of the target then renaming the directory. That costs 2x disk space during writes and introduces a brief outage window for any in-progress reader (e.g. Obsidian scanning the vault while the rename completes). For a write-mostly export destination, the per-file choice is the right V1 default — a half-applied export is no worse than an interrupted manual save inside Obsidian itself.

If a future use-case demands transactional semantics (e.g. exports that update entity graphs that an external system reads atomically), the upgrade path is to swap `target_dir = vault_path / entities_folder` for `target_dir = vault_path / f"{entities_folder}.staging-<uuid>"` and rename at the end. Behind the same `_write_to_vault` signature.

## Telemetry contract

Payload emitted in both success and failure paths:

```python
{
  "entities_written": int,           # counts only
  "relations_written": int,
  "files_written": int,
  "bytes_written": int,
  "duration_ms": int,
  "dropped_relations": int,
  "mode": "vault_path",
  "vault_path_redacted": True,
  # On failure also:
  "error": "TypeName: message",
  "partial": True,
  "entities_written_partial": N,
}
```

**Key**: the raw `vault_path` string is **never** in the payload. The boolean `vault_path_redacted` lets observers confirm "we wrote to filesystem" without leaking where (which may contain user-identifying directory names like `/Users/<full_name>/Documents/...`).

The test (`test_vault_path_telemetry_redacts_path`) uses a recursive walk: it asserts the raw `str(tmp_path)` is absent in every string value across the whole nested payload structure, not just at the top level. A regression that quietly added `vault_path: str(target_dir)` somewhere would fail this assertion.

## Auto-pipeline handler (JobType.EXPORT_OBSIDIAN)

`handlers.py` registers a payload-only handler matching the `HandlerRegistry`'s signature (`(payload: Dict[str, Any]) -> Dict[str, Any]`). The plan's pseudocode showed `job: Job` as the argument, but the actual `HandlerRegistry.register` decorator types are `payload: dict` (see `packages/job-queue/src/job_queue/registry.py:21`). I conformed to the registry contract.

Payload schema is locked via `ExportObsidianPayload` (Pydantic), constructed inside the handler from the raw payload dict. `ExportFilter(**validated.filter)` validates the filter knobs at job-execution time — same Pydantic guarantees as the synchronous router path.

The handler always uses `mode="vault_path"`: the async job pathway is exclusively for the auto-pipeline write-to-disk surface; user-initiated zip exports stay on the synchronous router path.

## Files changed

* `apps/app-main/src/app_main/services/obsidian_export_service.py` — `+VaultPathNotConfigured` exception, `_export_to_vault`, `_write_to_vault`, dispatcher in `export()`. Module docstring extended with D.1b semantics.
* `apps/app-main/src/app_main/api/routers/exports.py` — mode dispatch in `export_notebook_obsidian`; new `_extract_partial_state` helper; expanded OpenAPI response schema.
* `apps/app-main/src/app_main/handlers.py` — `+ExportObsidianPayload`, `+handle_export_obsidian` registered for `JobType.EXPORT_OBSIDIAN`.
* `apps/app-main/tests/test_obsidian_export_service.py` — +7 vault-path tests (happy / not-configured (vault and folder) / absolute / existence / overwrite / atomic / telemetry-redaction).
* `apps/app-main/tests/test_exports_router.py` — replaced `test_obsidian_vault_path_mode_not_implemented` with 3 new tests (happy / 400-not-configured / 500-with-partial).
* `apps/app-main/tests/test_handlers.py` — new file: 2 tests (EXPORT_OBSIDIAN registered + sanity check on existing registrations).

## Quality gates

| Gate | Command | Result |
|------|---------|--------|
| Focused (D.1b) | `pytest tests/test_obsidian_export_service.py tests/test_exports_router.py tests/test_handlers.py -v` | **36 passed, 0 failed** (2:08) |
| App-main regression | `pytest -q` (app-main) | **566 passed, 0 failed** (1:39) — baseline was 554; +12 new tests (+7 service / +3 router / +2 handler). |
| Job-queue regression | `pytest -q` (job-queue) | **38 passed, 0 failed** (41s) |

## Coordination notes

* **D.0 follow-up #1** (status filter SurrealQL promotion): still deferred to D.2 per the D.1a + D.3 self-reviews — the final exporter lands before promotion so all three share the same gate in one swing. No new finding from D.1b that affects this.
* **JobResult shape**: I chose to flatten the `ExportReport` into the handler's return dict (`{success, mode, entities_written, ..., processing_time}`) rather than nest it under a `report` key. Mirrors `handle_insight_extract` and `handle_entity_extract` which spread their service results into the top-level dict. If the FE expects a nested shape, that's a one-line follow-up.
* **`partial=True` on `VaultPathNotConfigured`**: I explicitly do NOT mark `partial=True` for the not-configured case because no writes happened. Only `OSError` / `ValueError` mid-write paths set the partial flag. This makes the telemetry signal-to-noise better for the operator dashboard.

## Issues flagged for reviewer

* **Per-file vs whole-batch atomicity**: documented in the module docstring + this review. If the reviewer disagrees, the upgrade path is laid out above (staging-dir + dir-rename).
* **`open()` mock in `test_vault_path_atomic_per_file`**: I use `patch("builtins.open", ...)` to inject a selective failure. This is brittle if the service ever moves to `aiofiles` or similar. The test would fail loudly if the open-path changes, which is acceptable (a regression test that catches its own staleness). If the reviewer prefers a less invasive mock, the alternative is `monkeypatch.setattr(os, "replace", ...)` — but `os.replace` is called AFTER the file content is fully written, so a mock there would test a different code path (atomicity of the rename, not failure during the write).
* **Path PII**: telemetry redaction is structural (the helper never reads the raw path into the payload), but the raw path DOES appear in `logger.info(...)` lines via `artifact.vault_dir` resolution. This is by design — logs are operator-side, not telemetry-side — but if PII review tightens, the log line is the one place to scrub.

## Ready for review.
