# Phase I.H1 — adversarial review (attempt 1)

> Branch: `track/i-upload-guards` (HEAD 3ff343d), diff `main...HEAD` (base `main` = 21e0108)
> Reviewer: adversarial-reviewer agent
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.H1 (5 ACs)
> Date: 2026-06-19

## VERDICT: REVISIONS_NEEDED

The size guard (AC1) is genuinely sound — Starlette streams the multipart file into a
`SpooledTemporaryFile` (rolls to disk past 1 MB) and tracks `UploadFile.size`
incrementally, so the size check reads `.size` without pulling the body into RAM.
AC3/AC4 mechanics are correct. But there is a confirmed regression that 500s a
previously-working public endpoint, a real OOM hole in the page-count guard that
defeats the stated purpose of the phase, and test theater on the two highest-risk ACs.

## Findings

### 1. BLOCKER — `create_source_json` 500s on every request (regression)
`apps/app-main/src/app_main/api/routers/sources_upload.py:549`

`create_source_json` calls the *decorated* `create_source(request, response, form_data, …)`
positionally. The slowapi `async_wrapper` success path runs `_inject_headers(kwargs.get("response"), …)`;
because `response` was passed positionally, `kwargs.get("response")` is `None`, and
`_inject_headers(None, …)` raises `parameter 'response' must be an instance of
starlette.responses.Response`. The exception escapes as a generic 500 (not even the
router's JSON error handler). Empirically confirmed via traceback. On `main` this
endpoint worked (no decorator). Registered public route `POST /api/sources/json`, **zero
test coverage**.

*Fix direction*: extract a plain inner `_create_source_impl` that neither endpoint
decorates; both wrappers delegate to it. (Or pass `response=` as a keyword.)

### 2. BLOCKER — page-count guard reads the entire file into RAM, defeating the OOM goal
`apps/app-main/src/app_main/api/routers/sources_upload.py:134`

The point of I.H1 is preventing backend OOM on large PDFs. The size guard caps at
`MAX_FILE_SIZE_MB` (default **500 MB**). The page-count guard then does
`content = await upload_file.read()` — up to 500 MB into one `bytes` — and
`pdfium.PdfDocument(content)` holds another copy. A 499 MB PDF (under the size limit)
forces ~1 GB transient RAM before any page-count rejection. The guard runs *after* the
OOM is already possible.

*Fix direction*: feed pdfium a file path / readable stream (the body is already spooled to
disk) rather than a full in-memory `bytes`. At minimum read the page count from the spool
without rasterizing.

### 3. Major — AC3/AC5 tests don't exercise the production endpoint or limiter
`apps/app-main/tests/test_rate_limiter.py:34-58`

`_make_client` builds a throwaway app with a throwaway header-keyed `Limiter` and a trivial
`_upload`. The real `create_source` endpoint, real `get_remote_address`, and real `app.py`
wiring never run through the middleware. AC5-vs-production is a single identity assertion
(`production_limiter._key_func is get_remote_address`). This is why #1 slipped through: no
test drives a real decorated endpoint with `SlowAPIMiddleware` + `headers_enabled=True` on
the success path.

*Fix direction*: add a test mounting the real `sources_upload.router` with `SlowAPIMiddleware`,
asserting a 200 success carries rate-limit headers (catches #1) and a burst trips 429.

### 4. Major — upload-rewind correctness not asserted (self-flagged, unaddressed)
`apps/app-main/src/app_main/api/routers/sources_upload.py:135`

`await upload_file.seek(0)` after the page-count `read()` is correct, but
`test_valid_pdf_passes_guards` patches `save_uploaded_file`, so nothing verifies the bytes
survive. Drop the seek → valid PDFs persist as zero-byte files, suite stays green.

*Fix direction*: assert saved/forwarded byte length equals the original after guards run.

### 5. Minor — `create_source_json` double-invokes rate-limit accounting
`apps/app-main/src/app_main/api/routers/sources_upload.py:536,549`

Both endpoints carry `@limiter.limit`; the JSON path passes through two decorated wrappers.
`request.state._rate_limiting_complete` prevents a double count, but the coupling is fragile
(and is the mechanism behind #1). Decorating one shared impl removes both problems.

### 6. Minor — `MAX_FILE_SIZE_MB`/`MAX_PAGE_COUNT`/`RATE_LIMIT_RPM` module constants dead/misleading
`apps/app-main/src/app_main/config.py:71-73`

Module evaluates the constants at import, docstring says "prefer the getters", nothing uses
them; they snapshot env at import and won't reflect a test monkeypatch. Drop or mark clearly.

## High-risk spot assessment

1. **OOM-defeat**: PARTIAL FAIL → BLOCKER (#2). Size guard fine (reads `UploadFile.size`,
   spools to disk); page-count guard reads up to 500 MB into RAM + pdfium copy.
2. **AC3 Retry-After**: PASS (mechanics). `headers_enabled=True` + `_rate_limit_exceeded_handler`;
   test asserts the header. Caveat: against the stand-in app (#3).
3. **AC5 per-IP vs per-process**: PASS on keying; honestly under-claimed on multi-worker
   (in-memory store = per-process; true cross-worker needs `storage_uri`). Disclosed, not
   hidden. Acceptable as documented follow-up. Production per-IP assertion is just an
   identity check (#3).
4. **AC2 page-count**: PASS on robustness (non-PDF/corrupt/missing-pypdfium all skip without
   500), FAIL on perf — full in-memory `bytes` (#2).
5. **Guard ordering & cleanup**: MOSTLY PASS — guards run before disk write; cleanup on
   failures exists; but the `response` param breaks the JSON sibling (#1).
6. **Tests real, not theater**: MIXED — upload-guard tests honest (inverting the size check
   fails the test; asserts no DB call); rate-limiter tests run a replica (#3); rewind
   untested (#4); JSON endpoint untested (how #1 shipped green).

---

## Attempt 1 — revisions

All BLOCKER + Major findings resolved. Tests: **9 passed** (was 7) via the WSL `.venv`.

| # | Severity | Resolution | Commit |
|---|---|---|---|
| 1 | BLOCKER | Extracted undecorated `_create_source_impl`; both route wrappers delegate to it. JSON endpoint no longer invokes the decorated multipart handler → no more 500. | `fbec50d` |
| 2 | BLOCKER | Page-count guard now streams pypdfium2 from the spooled `upload_file.file` (block reads), never loading the body into RAM. Rewinds on every exit path. Docstring corrected. | `fbec50d` |
| 3 | Major | Added `test_json_endpoint_succeeds_through_real_wiring` — real router + `SlowAPIMiddleware` + production limiter; asserts `POST /sources/json` → 200 with rate-limit headers (drives the exact success-path injection that broke). | `ae8bc72` |
| 4 | Major | Added `test_guards_rewind_preserves_upload_bytes` — runs the real guard over a real PDF and asserts the upload bytes survive. Dropping the `seek(0)` now fails a test. | `ae8bc72` |
| 5 | Minor | Subsumed by #1 — the JSON path no longer double-passes through `@limiter.limit`. | `fbec50d` |
| 6 | Minor | Removed the dead module-level `MAX_FILE_SIZE_MB` / `MAX_PAGE_COUNT` / `RATE_LIMIT_RPM` constants; getters are the single source of truth. | `527e920` |

### Out of scope (left per surgical-change discipline — not introduced by this run)
- `ruff check` reports 2 pre-existing `I001` (import-sort) errors on function-local imports in
  `sources_upload.py` (`from surrealdb_service.connection import ...`, ~lines 386 & 617). These
  predate I.H1 and sit outside every changed region; not touched. Candidate for a separate
  lint-cleanup PR.

### AC5 (per-IP, multi-worker) — reviewer's accepted position
Keying is genuinely per-IP (`get_remote_address`); the in-memory store is per-process, so a true
single cross-worker budget needs a shared `storage_uri` (Redis). The reviewer accepted this as a
disclosed limitation rather than a blocker, consistent with the plan's generous defaults. Documented
in the self-review; deferred as a follow-up, not fixed here.
