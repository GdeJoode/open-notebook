# Phase I.H1 — self review

> Branch: `track/i-upload-guards`
> Commits: `605c32f` (config + dep) → `a12485b` (limiter wiring) →
> `82f65a2` (upload guards) → `e534047` (tests)
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.H1
> Reviewer cycle: ×1.5

## Plan-vs-reality corrections (file paths in the plan were partly stale)

The plan's "Files to modify" listed `sources_files.py` for the upload-guard
preflight. The upload POST endpoint is **not** there — it lives in
`apps/app-main/src/app_main/api/routers/sources_upload.py`
(`create_source`, the multipart `@router.post("/")`, plus its JSON sibling
`create_source_json`). `sources_files.py` is download/serving only and was
left untouched. The guard preflight was added to `sources_upload.py` in the
`if upload_file and source_data.type == "upload":` branch, before
`save_uploaded_file` writes to disk.

`RateLimitError` (`exceptions.py:54`) and its handler (`app.py`) existed but
were never raised anywhere — there was no in-process limiter to "replace",
only scaffolding. Per AC4 that handler is left intact; slowapi was added
**alongside** it as the actual enforcement layer.

A fifth file beyond the plan's list was created: `app_main/api/rate_limit.py`.
Rationale under AC5 / Inversion 3 below — it breaks a circular import.

## AC-by-AC

### AC1 — POST file > MAX_FILE_SIZE_MB → 413 referencing the limit

**Status**: PASS (covered by `test_oversize_file_rejected_with_413`).

`enforce_upload_guards` reads `UploadFile.size` (populated by Starlette from
the multipart part length, so no body read is needed) and raises
`HTTPException(413, ...)` when it exceeds `MAX_FILE_SIZE_MB * 1024 * 1024`.
The detail string names the limit and `MAX_FILE_SIZE_MB`; the test asserts
both `"limit"` and `"MAX_FILE_SIZE_MB"` appear, and that `source_svc.create`
is never called (rejection happens before any DB work).

### AC2 — POST PDF with pages > MAX_PAGE_COUNT → 422; count via pypdfium

**Status**: PASS (covered by `test_oversize_pages_rejected_with_422`).

For `.pdf` uploads the guard lazily imports `pypdfium2`, reads the body once,
`pdf = pdfium.PdfDocument(content)`, `len(pdf)`, and raises
`HTTPException(422, ...)` when over `MAX_PAGE_COUNT`. This mirrors the
`/page-count` endpoint's pdfium usage. The body is rewound
(`await upload_file.seek(0)`) so the subsequent `save_uploaded_file` re-reads
from offset 0. The test mocks `pypdfium2.PdfDocument` to return an
11-element object (`len` == 11) against a limit of 10 and asserts 422 +
`"MAX_PAGE_COUNT"` + `"11"` in the detail, and that no source was created.

`pypdfium2.PdfDocument` accepts a bytes buffer as well as a path, so the
guard can run before the file is written to disk (the OOM-safety goal: reject
before persisting). If pdfium is unavailable or the bytes are not a parseable
PDF, the guard logs and skips (size guard still applies) rather than failing
the upload — same tolerance as `/page-count`.

### AC3 — Burst > RATE_LIMIT_RPM from same IP → 429 with Retry-After

**Status**: PASS (covered by
`test_burst_from_one_ip_trips_429_with_retry_after`).

The `Limiter` is constructed with `headers_enabled=True`; the default
slowapi `_rate_limit_exceeded_handler` then injects `Retry-After` on the 429
(without `headers_enabled` it omits it — verified by reading
`slowapi/extension.py::_inject_headers`, which gates the header behind
`self._headers_enabled`). The test bursts 4 requests at a 3/minute limit
from one key: first 3 → 200, 4th → 429, and asserts `"Retry-After"` is
present on the 429 response headers.

### AC4 — RateLimitError handler still active (JSON, not stack trace)

**Status**: PASS (unchanged code).

The `@application.exception_handler(RateLimitError)` block in `app.py` is
untouched and still returns `JSONResponse(status_code=429, ...)`. slowapi's
`RateLimitExceeded` handler is registered separately and does not collide
(different exception type). Nothing in this phase raises `RateLimitError`, so
the two coexist: slowapi enforces, the legacy handler remains available for
any code path that chooses to raise `RateLimitError` explicitly.

### AC5 — Rate limit per-IP not per-process

**Status**: PARTIAL — per-IP keying is verified; true multi-worker
shared-state is NOT (honest limitation, see below).

The limiter's `key_func` is `get_remote_address`, i.e. buckets are keyed on
the client IP, not a process-global counter. Two coverage points:

- `test_second_ip_is_unaffected`: with a 2/minute limit, IP A is exhausted
  (3rd request → 429) while IP B still gets its full bucket — proving the
  bucket is per-key, not a single shared counter.
- `test_production_limiter_keyed_per_ip`: asserts
  `production_limiter._key_func is get_remote_address`, locking the per-IP
  contract so a future edit can't silently regress it to a per-process key.

**Honest limitation (do not over-claim)**: slowapi's *default storage* is
in-memory and therefore **per-process**. With multiple uvicorn workers, each
worker keeps its own in-memory buckets, so the effective global limit is
`RATE_LIMIT_RPM × num_workers` for a given IP, and an IP can be balanced
across workers. "Per-IP" (the *keying*) holds in every worker; "one global
budget per IP across all workers" does **not** without a shared store
(Redis/memcached via slowapi's `storage_uri`). The plan's AC5 parenthetical
"verified in multi-worker test" is not achievable with the default in-memory
backend, and I did not stand up a shared store in this phase. If a true
cross-worker budget is required, the follow-up is to pass
`storage_uri="redis://..."` (or memcached) to the `Limiter` in
`rate_limit.py`; the keying code does not change. I chose not to introduce a
Redis dependency that the plan did not call for.

## Mental inversion tests

### Inversion 1 — remove the size check

If I deleted the `size_bytes > max` branch, `test_oversize_file_rejected_with_413`
fails: the request would fall through to the patched/real save path and not
return 413. Caught.

### Inversion 2 — forget to rewind the upload after reading for page count

If I dropped `await upload_file.seek(0)` after `upload_file.read()` in the
page-count branch, the valid-PDF path would still pass the guard but
`save_uploaded_file` would persist an empty (already-consumed) file. This is
a real gap **not** directly caught by the current tests:
`test_valid_pdf_passes_guards` patches `save_uploaded_file`, so it never
asserts the bytes are intact after the guard. Honest coverage hole. Mitigation
considered: add an assertion reading the saved file's length. Deferred because
it would require either a real temp-dir write (slower, touches disk) or a
more elaborate mock capturing the seek position; the seek is a one-liner
adjacent to the read and is documented. Flagging for the reviewer.

### Inversion 3 — define `limiter` in `app.py` instead of `rate_limit.py`

This is what the first cut did, and it broke at import time across the whole
suite: `app.py` builds the application at module load (`app = create_app()`),
`create_app()` imports the routers, `sources_upload` imported
`from app_main.api.app import limiter`, and that re-entered the
mid-initialisation `app` module → `partially initialized module ... has no
attribute 'router'`. Moving `limiter` into a leaf module
(`app_main/api/rate_limit.py`) that neither imports `app` nor any router
breaks the cycle. Verified: `pytest --collect-only` now collects all 597
tests with zero import errors (it errored before the fix). If someone moves
`limiter` back into `app.py`, collection breaks again immediately — so this
is self-guarding.

### Inversion 4 — drop headers_enabled=True

If `headers_enabled` reverted to the default `False`, the 429 would still be
returned but **without** `Retry-After`, and
`test_burst_from_one_ip_trips_429_with_retry_after` would fail on the
`"Retry-After" in tripped.headers` assertion. Caught. (It would also stop
slowapi from trying to inject success headers, which is why the endpoints
gained a `response: Response` param — see Inversion 5.)

### Inversion 5 — remove the `response: Response` param from the endpoint

With `headers_enabled=True`, slowapi's `@limit` wrapper injects rate-limit
headers into the success response; for a non-`Response` return value (the
endpoints return a Pydantic `SourceResponse`) it looks for a `response`
kwarg to write into and raises
`"parameter `response` must be an instance of starlette.responses.Response"`
if absent. So the `response: Response` parameter is load-bearing, not
cosmetic. Not covered by a dedicated test, but any successful-path request
through a decorated endpoint with `headers_enabled=True` would crash without
it — the rate-limiter tests' 200 responses exercise exactly this path and
would fail if the param were removed from the test handler.

## Tests + results

```
tests/test_upload_guards.py::test_oversize_file_rejected_with_413 PASSED
tests/test_upload_guards.py::test_oversize_pages_rejected_with_422 PASSED
tests/test_upload_guards.py::test_valid_pdf_passes_guards         PASSED
tests/test_upload_guards.py::test_non_pdf_skips_page_guard        PASSED
tests/test_rate_limiter.py::test_burst_from_one_ip_trips_429_with_retry_after PASSED
tests/test_rate_limiter.py::test_second_ip_is_unaffected          PASSED
tests/test_rate_limiter.py::test_production_limiter_keyed_per_ip  PASSED
7 passed in 69.92s
```

Regression check (no regressions):

- `pytest --collect-only tests/` → **597 tests collected**, 0 import errors
  (this is the load-bearing regression signal — the circular-import fix is
  proven suite-wide).
- Representative subset
  (`test_health_router`, `test_notebooks_router`, `test_config_router`,
  `test_source_service`, `test_sources_processing`) → **36 passed**.

I did not run the *entire* 597-test suite to green in this environment: a
full run exceeded the practical time budget here (heavy ML imports +
SurrealDB-touching integration tests that need a live DB). The collect-only
pass plus the API-layer subset cover the code this phase touches.

## Test-design note (why the rate-limiter test keys on a header)

Starlette's `TestClient` pins `request.client.host` to a single value for
every request, which makes "two distinct IPs through `get_remote_address`"
awkward to express. The rate-limiter tests therefore build a *test* limiter
keyed on an `X-Test-IP` header to drive distinct keys deterministically; the
bucketing logic under test (one key trips, another doesn't; Retry-After on
429) is identical regardless of where the key string originates. The
**production** key func is separately pinned to `get_remote_address` by
`test_production_limiter_keyed_per_ip`, so the per-IP contract is asserted
against the real wiring, not the stand-in.

## Lint

`ruff check` on the two new test files → clean (after `--fix` reordered
imports per the project's isort config). The four pre-existing `I001`
import-order warnings in `app.py` and `sources_upload.py` are **pre-existing
on `main`** (verified by running ruff against the untouched main checkout) and
were left alone per the surgical-change rule — they are not introduced by
this phase.

## Sandbox limitations honestly recorded

- The worktree's own `.venv` would not populate under this Windows/WSL
  setup (uv install on the NTFS mount left site-packages effectively empty).
  Tests were run with the fully-provisioned **root** venv
  (`/mnt/e/repos/private/open-notebook/.venv`, 536 packages, slowapi
  installed via `uv pip install slowapi>=0.1.9`) plus
  `PYTHONPATH=<worktree>/apps/app-main/src` so the worktree's source is what
  executes. Verified the override resolves `app_main` to the worktree path.
- `slowapi` was installed into the root venv with `uv pip install`; the
  dependency is recorded in `apps/app-main/pyproject.toml` and `uv.lock`
  (`slowapi==0.1.10`, transitively `limits`, `deprecated`) so a clean
  `uv sync` will pick it up.
- **Multi-worker per-IP** (AC5 parenthetical) was NOT tested with real
  workers + shared store — see AC5 above. This is the one acceptance
  criterion I cannot fully verify as literally written in this environment,
  and the in-memory backend cannot satisfy a true cross-worker budget
  regardless of the test harness.

## What the reviewer should look at first

1. **AC5 multi-worker honesty**: confirm the in-memory / per-process
   storage tradeoff is acceptable, or require a `storage_uri` shared store.
2. **Inversion 2** (upload rewind not asserted) and **Inversion 5**
   (`response` param load-bearing) — the two coverage holes I'm flagging.
3. The new `rate_limit.py` leaf module (circular-import fix) — confirm
   you're happy with the extra module vs. some other cycle break.
4. Guard placement in `create_source` — it runs before `save_uploaded_file`,
   so oversized/over-paged uploads never hit disk.
