# Track G — Agent Integration & Headless Mode — Retrospective

Shipped 2026-07-24/25 across eight PRs (#67–#74, which includes the #72 security
fix). Track G turns open-notebook into a headless agent platform: a versioned,
API-key-authed REST surface over the existing pipeline, plus an opt-in inbox
file-watcher.

## What shipped

| Phase | PR | What |
|---|---|---|
| G.1 | (stack) | `agent_keys` + `agent_audit_log` (migration 77), `X-API-Key` auth (`require_agent_key`, fail-closed, read<write<admin), per-IP throttle + per-key limit + audit middleware, `POST /extract-entities` |
| G.2 | #68 | `POST /generate-summary` over raw text (no-DB `summarize_text`) |
| G.3 | #69 | Ingest façade: `process-url` + `GET /jobs/{id}` (ownership-bound) + `GET /audit-log` |
| G.3b | #71 | Multipart `process-document` + `process-audio` (reuse `_create_source_impl`) |
| — | #72 | **Security**: upload filename path-traversal fix (surfaced by the G.3b review) |
| G.4 | #70 | "API Keys" settings tab (mint/list/revoke + show-once + audit drawer) |
| G.5 | #73 | Inbox `InboxWatcher` (opt-in, default OFF) |
| G.6 | #74 | File-watcher settings panel (read-only status + recent activity) |
| G.7 | — | OpenAPI spec-validity test + docs + this RETRO |

## Deliberate deviations from the plan

- **Façade-over-reuse, never a parallel pipeline.** Every `process-*` route builds
  a `SourceCreate` and calls the SAME `_create_source_impl` / `process_source`
  chain the UI uses; `jobs/{id}` is verbatim `get_command_status`. No new
  `JobType`/command was added (plan AC G.3-6). The upload endpoints reuse
  `enforce_upload_guards` (413/422) unchanged.
- **G.6 is read-only by design (deviates from AC2).** The watcher config is
  env-managed and read once at startup, so an editable/persisted toggle would
  persist a value that does NOT affect the running watcher until restart — a
  misleading no-op control. We ship an honest read-only status + activity panel
  and defer a genuinely runtime-reconfigurable watcher.
- **`process-document`/`process-audio` share one impl.** The `process_source`
  pipeline routes by file type, so the two routes are thin wrappers over one
  `_ingest_upload`; the split is documented intent, not divergent logic.
- **File-watcher ingest = create source + submit `process_source` with a staged
  file_path.** The watcher copies the drop into the canonical uploads folder
  (reusing the hardened `generate_unique_filename`) so the source's `file_path`
  is stable, then moves the inbox original to `_processed`/`_errors` after the
  job reaches a terminal state.

## Adversarial review earned its keep (the headline lesson)

Every substantial phase went through an adversarial-reviewer pass before merge,
and it repeatedly caught defects the author's own tests missed — several of them
security-critical. This is the strongest evidence yet for the
[[adversarial-review-standard]] as a hard merge gate.

- **G.3 needed 5 review rounds to converge.** An IDOR on `GET /jobs/{id}` (any
  read key could poll any job's result/error — jobs carry no owner column) and a
  broken deferred-job contract (`process-url` ran synchronously and returned a
  fake empty `job_id`) were both BLOCKERS the author's mocked tests sailed past.
  The SSRF guard then took THREE more rounds: numeric-encoding bypass
  (decimal/octal/hex `http://2130706433/` → loopback) → Unicode homoglyph-dot
  bypass (`http://169。254。169。254/` → metadata) → Unicode-16 outlined-digit
  divergence. The fix classifies the host the fetcher actually resolves (NFKC +
  UTS-46 dot separators + `inet_aton` + outlined-digit fold) before the
  private-IP check.
- **The G.3b review found a pre-existing HIGH.** A filename path-traversal in the
  shared `generate_unique_filename` (`../../etc/cron.d/x` escaped the uploads
  folder — arbitrary file write). Not authored by G.3b, but G.3b newly exposed it
  to write-scoped agent keys. Fixed repo-wide in #72 (basename + containment
  assert), which also hardened the UI upload route.
- **The G.5 review found a MAJOR exactly-once hole.** `_await_terminal` + `_move`
  ran outside the try/except, so a transient raise stranded the file
  re-ingestable → a duplicate source on the next startup scan. Fixed by widening
  the guard + making `_move` best-effort, with a real-Observer integration test.
- **A recurring blind spot:** the author's tests mocked the exact layer carrying
  the guarantee (the ingest reuse; the authz DB query). The fix each time was a
  live-DB or real-Observer integration test of the derivation, not another mock.

## Open follow-ups

Tracked in `status.md`: runtime-reconfigurable watcher; shared fetch-layer SSRF
(DNS-rebinding + redirect-to-private); `transformations` forwarding on the async
`process_source` payload; executing the live e2e specs against a running stack;
a bounded watcher recent-activity scan.
