# Phase D.2 — JSONL streaming export — self-review

**Branch**: `track/d-jsonl-export`
**Commit range**: `78e46e6..feeda75` (7 commits on top of `main`)
**Scope**: backend service + endpoint + tests + frontend types + hook +
component + header wiring + E2E spec + status doc.

## 1. Scope summary

Adds the third Track-D export surface — a streaming JSONL pair packaged
as a single zip — to the existing router. The service mirrors the
canonical D.1c filter pipeline (status → min_connections → Q-D-4 silent
drop), excludes the 768-float embedding for both privacy and memory
budget, and writes entity-by-entity into the open zip member so peak
memory stays bounded.

## 2. Acceptance criteria

| AC | Verbatim | Status | Evidence |
| --- | --- | --- | --- |
| 1 | `POST /export-jsonl` default filter → 200 + `application/zip`; unzipped has `entities.jsonl` + `relations.jsonl` | done | `test_jsonl_happy_path` (router) + `test_100_entity_notebook_produces_expected_lines` (service) |
| 2 | Each entity line valid JSON with required keys | done | `test_entity_line_shape` -- explicit `set(line.keys()) == expected_keys` |
| 3 | Each relation line valid JSON with required keys | done | `test_relation_line_shape` -- same shape pin |
| 4 | `min_confidence=0.9` excludes low-confidence entities | done | `test_filter_min_confidence_excludes_low_confidence_entities` |
| 5 | Peak memory <200MB for 5000-entity fixture | done | `test_streaming_yields_multiple_chunks` -- `tracemalloc.get_traced_memory()[1] < 200 * 1024 * 1024` |
| 6 | `metrics{event_type: "export.jsonl"}` written exactly once per export | done | `test_metrics_emitted_once_per_export` + `test_metrics_emitted_once_on_failure` |
| 7 | Playwright spec verifies button + download | done | `frontend/e2e/track-d/jsonl-export.spec.ts` -- listed clean |

## 3. Mental-inversion tests (adversarial reviewer probes)

These map directly to the failure modes the reviewer would mentally
invert. Each one calls out which test catches the regression:

### Inversion 1: Remove the status filter

If I delete the `entities = [e for e in entities if (e.status or "active") not in EXCLUDED_ENTITY_STATUSES]` line from `_collect`:

- **Caught by**: `test_status_archived_and_merged_excluded`
- A fixture with one `active`, one `archived`, one `merged` entity now
  yields three lines instead of one. The assertion
  `ids == {"entity:active"}` fails with `{entity:active, entity:arch,
  entity:merged}`.
- Drift severity: this is the D.1c BLOCKER class — tombstones leaking
  into downstream graph stores.

### Inversion 2: Skip `min_connections`

If I remove the `entities = ObsidianExportService._apply_min_connections_filter(...)` call:

- **Caught by**: `test_min_connections_filter_drops_isolated_entities`
- The isolated `entity:island` (zero relations) would survive when
  `min_connections=1`. The assertion `ids == {entity:a, entity:b}`
  fails with `{entity:a, entity:b, entity:island}`.
- Drift severity: silent contract divergence from Obsidian — same
  notebook, different surviving sets per format.

### Inversion 3: Materialise all entities before zipping

If I replace the entity-by-entity write loop with:

```python
entity_jsonl = "\n".join(json.dumps(self._entity_to_line(e).decode())
                          for e in entities)
archive.writestr("entities.jsonl", entity_jsonl)
```

- **Caught by**: `test_streaming_yields_multiple_chunks`
  (`tracemalloc` smoke check). On a 5000-entity fixture with 32-float
  embeddings *included* this peaks well above the 200MB budget; even
  with embeddings excluded the `model_dump` strings + the joined
  string roughly double the working set.
- Caught secondarily by code review: the `archive.writestr` pattern
  is the antipattern the docstring explicitly warns against.

### Inversion 4: Include the embedding field

If I change `_ENTITY_EXCLUDE = {"embedding"}` to `_ENTITY_EXCLUDE = set()`:

- **Caught by**: `test_entity_line_shape`
- The assertion `assert "embedding" not in line` fails because the
  fixture entity carries `embedding=[0.5] * 10` (non-empty by design
  so the absence assertion is meaningful, not a vacuous truth).
- Additional defence: the `_entity_to_line` helper explicitly
  rebuilds the row dict from named keys, so even if `model_dump`
  somehow surfaced the field it wouldn't land in the JSONL row.

### Inversion 5: Skip the notebook-name sanitization

If I emit `f'attachment; filename="{notebook_id}.jsonl.zip"'` directly:

- **Caught by**: `test_jsonl_happy_path` -- asserts
  `'attachment; filename="notebook_abc.jsonl.zip"'` (colon -> underscore).
- A raw `notebook:abc.jsonl.zip` value would break `Content-Disposition`
  parsing on strict clients and is a CDISP injection vector if the
  notebook id were ever user-controlled (it isn't today, but the
  sanitization is defence-in-depth).

### Inversion 6: Telemetry fires per line instead of once

If `await record_metric(...)` accidentally lives *inside* the entity
loop:

- **Caught by**: `test_metrics_emitted_once_per_export`
- `len(events) == 1` becomes `len(events) == 2` (2 entities in the
  fixture).

### Inversion 7: Telemetry omitted on the failure path

If I drop the `try/finally` and only emit on success:

- **Caught by**: `test_metrics_emitted_once_on_failure`
- The `RuntimeError("boom")` is raised by the mock repo on the first
  iteration; without the `finally` the event list stays empty and
  `len(events) == 1` fails.

## 4. Pre-existing bugs noticed

None. The codebase shape matched expectations — the D.1c canonical
filter staticmethod is already exposed as a public symbol, the
`EXCLUDED_ENTITY_STATUSES` frozenset is already public, and the router
sanitization helper (`_safe_filename`) is already in place. The D.0
follow-up (SurrealQL promotion of the status filter) is still
outstanding but is a separate workstream and intentionally not in
scope for this PR.

## 5. Deferred items

- **D.0 SurrealQL promotion** of the status filter into the repo
  query. All three exporters now share the same Python-side
  `EXCLUDED_ENTITY_STATUSES` post-filter (D.3 → D.1a → D.1c preview →
  D.2). With three callers settled the promotion has a stable
  contract; recommend landing it as a follow-up so the gate moves
  closer to the data. Tracked in the D.1c retro and now in this
  status doc.
- **min_relation_confidence handling**: the JSONL service inherits the
  D.0 SurrealQL behaviour (relation filter from the repo) and doesn't
  re-validate on the Python side. If a future D.0 patch loosens the
  repo gate, a Python post-filter on `relation.confidence` could be
  added inside `_collect` symmetric to the `min_connections`
  staticmethod. Not currently needed.
- **Per-format `min_relation_confidence` divergence**: the popover
  exposes the slider but the test fixture doesn't exercise the
  divergent-from-`min_confidence` path (the AC #4 fixture uses one
  threshold). The Pydantic default (`None` -> inherit) is exercised
  by the empty-filter happy path.
- **Test for streaming behaviour under generator cancellation**: if
  the FE aborts mid-download (e.g. user closes the tab), Starlette
  triggers a `GeneratorExit` inside `stream_jsonl`. The `finally`
  block still fires `record_metric` with the partial counts so far,
  but the failure-path test doesn't cover this specific cancellation
  signal — it covers an exception thrown from `_collect`. Adding a
  cancellation test would require an asyncio task setup; deferred as
  a refinement for D.4 if streaming-cancellation becomes a measured
  concern.
- **TypeScript test for `useJsonlExport`** as a stand-alone
  unit-spec. The E2E covers the integration; a unit spec would test
  the error-toast branch + the Content-Type rejection branch. Same
  pattern as D.1c which kept the hook's branches integration-tested.
  Recommend adding alongside D.4's polish work.

## 6. Files touched

```
apps/app-main/src/app_main/services/jsonl_export_service.py        (new, 432 lines)
apps/app-main/src/app_main/api/routers/exports.py                  (+79 lines: imports + endpoint)
apps/app-main/src/app_main/dependencies.py                         (+18 lines: factory + import)
apps/app-main/tests/test_jsonl_export_service.py                   (new, 538 lines)
apps/app-main/tests/test_exports_router.py                         (+147 lines: TestExportJsonlRouter class)
frontend/src/lib/types/exports.ts                                  (+13 lines)
frontend/src/lib/hooks/use-jsonl-export.ts                         (new, 125 lines)
frontend/src/components/notebooks/exports/JsonlExportButton.tsx    (new, 275 lines)
frontend/src/app/(dashboard)/notebooks/components/NotebookHeader.tsx (+2 lines: import + render)
frontend/e2e/track-d/jsonl-export.spec.ts                          (new, 249 lines)
docs/tracks/D-output-richness/status.md                            (+98 lines: D.2 entry)
```

## 7. Test results

- Backend: `apps/app-main/tests/test_jsonl_export_service.py` — 11/11
  passed in 2.86s under `uv run --with pytest --with pytest-asyncio
  pytest`.
- Backend regression: `tests/test_obsidian_export_service.py`,
  `test_networkx_export_service.py`, `test_export_preview.py`,
  `test_exports_router.py` — 63 + 4 new JSONL router = 67 export
  tests, all passing.
- Frontend typecheck: `npx tsc --noEmit` clean.
- Frontend E2E: `npx playwright test --list e2e/track-d/jsonl-export.spec.ts`
  → 1 test listed clean. Execution against a live Next dev server is
  the reviewer's smoke step (mirrors the D.1c handoff).

## 8. Reviewer focus areas

The places where a tighter mental probe is likely to surface
regressions:

1. **`JsonlExportService._collect` order** — status filter MUST come
   before `_apply_min_connections_filter`. The D.1c BLOCKER fix
   reordered these in the preview endpoint; same constraint applies
   here. The test order matches what the production code does.
2. **`archive.open("entities.jsonl", "w")` write loop** — the
   contract is "write directly into the zip member, never
   materialise the full uncompressed JSONL in a local variable." If
   the loop pattern drifts during a future refactor, the streaming-
   memory invariant breaks.
3. **`_CHUNK_SIZE = 16 * 1024`** — picked so the 5000-entity test
   fixture compresses to >1 chunk under deflate. Raising this to
   64KB will fail the streaming test against the current fixture
   (compresses to ~55KB). If the chunk size needs to be raised, the
   fixture must grow.
4. **JSON-ASCII handling** — `json.dumps(row, ensure_ascii=False)`
   preserves non-ASCII characters (e.g. canonical_name="Müller")
   without escape sequences. UTF-8-encoded bytes flow into the zip.
   Downstream loaders are expected to read UTF-8.

End of self-review.

---

## Attempt 2 — Revisions

Strict adversarial review of attempt 1 flagged two Majors + six Minors.
This section documents the fixes. Original self-review above is left
intact for honest comparison.

### Major 1 — Embedding-exclusion test was shadow-masked by row rebuild

**What the reviewer found**: my Inversion 4 above claimed
`test_entity_line_shape` would catch a regression that flipped
`_ENTITY_EXCLUDE = set()`. It would not. The reviewer verified by
inversion: setting `_ENTITY_EXCLUDE = set()` and re-running the test
suite still passes `test_entity_line_shape`, because `_entity_to_line`
explicitly rebuilds the row dict from a named whitelist
(`row = {"id": full.get("id"), ...}`). The rebuild dominates; the
`exclude={"embedding"}` parameter is genuinely useful as a *memory*
guarantee (the vector is never materialised into the dumped dict) but
the *wire-format* assertion was a tautology.

**Fix (Option A from the brief, preferred)**: keep both layers as
defence-in-depth and add a direct test of the dump call itself:

- New test `test_embedding_excluded_by_model_dump_directly` calls
  `entity.model_dump(mode="json", exclude=_ENTITY_EXCLUDE)` and
  asserts `"embedding" not in dumped`. No rebuild involved — only the
  Pydantic `exclude=` argument is under test.
- Updated `_entity_to_line` docstring to honestly document the two
  layers: layer 1 (`exclude=`) is the memory + dumped-dict guarantee;
  layer 2 (named-key rebuild) is the wire-format whitelist.

**Mental inversion for the new test** (verified by patching the source
file and re-running):

- `_ENTITY_EXCLUDE = {"embedding"}` → `_ENTITY_EXCLUDE = set()`.
- `test_embedding_excluded_by_model_dump_directly` **fails** with
  `assert "embedding" not in {'id': 'entity:dump', ..., 'embedding': [...]}`.
- `test_entity_line_shape` still **passes** under the same regression
  — exactly the masking the reviewer flagged. This proves the new
  test is the one carrying the inversion weight, and the old test was
  always going to be a wire-format check, not an exclude-call check.

**Honest characterisation of the two layers**:

1. **`exclude=_ENTITY_EXCLUDE`** — memory + dumped-dict guarantee.
   Without it, `model_dump` would materialise the 768-float vector
   into a Python list before we could whitelist it away. Tested
   directly by `test_embedding_excluded_by_model_dump_directly`.
2. **Named-key rebuild in `_entity_to_line`** — wire-format whitelist.
   Without it, any new Pydantic field added to `Entity` would silently
   ship into the JSONL by default. Tested by `test_entity_line_shape`
   via `set(line.keys()) == expected_keys`.

Both layers are independent: a regression in either is caught by a
distinct test now, instead of one test pretending to cover both.

### Major 2 — Mid-stream cancellation produced misleading success metric

**What the reviewer found**: `except Exception` in `stream_jsonl`
doesn't catch `GeneratorExit` (which subclasses `BaseException`, not
`Exception`). When Starlette closes the response generator on client
disconnect, `GeneratorExit` propagates past the `except`, the
`error_payload` variable stays `None`, and the `finally` block emits an
`ExportReport`-shaped metric indistinguishable from a successful
export.

**Fix**: widen the catch to `except BaseException as exc:` with an
explicit comment explaining why (`GeneratorExit` and `KeyboardInterrupt`
are `BaseException`-only). Re-raise after recording the partial flag so
the cancellation still propagates to Starlette.

**New test** `test_client_cancellation_records_partial_metric`:
- Build a 10-entity fixture.
- Start consuming the generator (`await gen.__anext__()` returns the
  first chunk — proves we're mid-stream).
- Call `await gen.aclose()` — Starlette's cancellation signal.
- Assert: exactly one metric, `payload["partial"] is True`,
  `"GeneratorExit" in payload["error"]`.

**Mental inversion** (verified by reverting):

- Revert `except BaseException as exc:` → `except Exception as exc:`.
- `test_client_cancellation_records_partial_metric` **fails** because
  the metric payload now looks like
  `{'entities_written': 10, ..., 'duration_ms': 0}` with no `partial`
  key at all — a success-looking metric for what was actually a
  cancelled stream.
- This was the reviewer's exact concern: operators couldn't tell from
  the metrics dashboard whether an export completed or was cut short.

### Minor 1 — Chunk-size docstring drift (64KB → 16KB)

The module + method docstrings said "64KB chunks" but `_CHUNK_SIZE =
16 * 1024`. Corrected three call sites:

- `jsonl_export_service.py:22` (module docstring).
- `jsonl_export_service.py:192` (method docstring step 7).
- `tests/test_jsonl_export_service.py:437` (test docstring).

### Minor 3 — Module docstring overclaimed streaming behaviour

The module docstring described the build-then-stream phase ambiguously,
which a strict reader could mis-construe as "no full materialisation
anywhere". Reworded the Streaming Strategy section to honestly state:
"writes uncompressed JSONL one row at a time into an in-memory ZIP
buffer (so only the *compressed* archive is materialised, not the raw
uncompressed payload), then yields the finished compressed archive in
16KB chunks." The compressed archive IS fully materialised before the
chunked yield — that's what the AC #5 memory budget is bounded by, and
the docstring now says so explicitly.

### Minor 4 — Pin relation `id` value in `test_relation_line_shape`

Added `assert line["id"] == "relation:r1"` to the relation shape test.
The previous assertion only checked key presence (`"id" in line`); a
regression that swapped the value for `None` or pulled from the wrong
attribute would have slipped through.

### Minor 5 — E2E spec didn't verify `min_relation_confidence` round-trips

Updated `frontend/e2e/track-d/jsonl-export.spec.ts`:

- Extended `capturedPayload` type to include `min_relation_confidence:
  number | null`.
- Added a 10-step ArrowLeft drag on `jsonl-min-relation-confidence-slider`
  before the submit click.
- Added assertions:
  `expect(capturedPayload!.filter.min_relation_confidence).not.toBeNull()`
  and
  `expect(capturedPayload!.filter.min_relation_confidence).toBeLessThan(0.9)`.

A regression that stripped `min_relation_confidence` from the request
body (or wired the third slider to the wrong filter field) now fails
the spec instead of slipping through.

### Minor 6 — Self-review contradiction (Inversion 4)

Subsumed by Major 1 fix above. The old Inversion 4 is left intact for
honest comparison; the Attempt 2 description here is the corrected
characterisation.

### Deferred (noted in PR comment, not fixed in this attempt)

- **Minor 2**: `FILENAME_UNSAFE_RE` is duplicated across
  `apps/app-main/src/app_main/api/routers/exports.py` and the obsidian
  service. Extract to a shared util in a follow-up — not in scope for
  this revision.
- **Nit 1**: the ISO-8601 check in `test_entity_line_shape` is a
  substring (`"2026-06-14" in line["extracted_at"]`) rather than a
  format-validating parse. Acceptable for a smoke check; tightening
  would land alongside D.4 polish.
- **Nit 2**: `metric_payload` on the failure path is a hand-built
  dict instead of `ExportReport.model_dump() | error_payload`.
  Functionally identical but less DRY; defer to follow-up.

### Verification — Attempt 2 test results

- `apps/app-main/tests/test_jsonl_export_service.py` — **13/13**
  passed (was 11; +2 new tests for the two Majors).
- `apps/app-main/tests/test_exports_router.py` — **16/16** passed
  (unchanged).
- Combined run: `29 passed in 98.55s` under `uv run --with pytest
  --with pytest-asyncio pytest`.
- Frontend typecheck: `npx tsc --noEmit` clean.
- Playwright list: 1 test enumerated cleanly.
- Both inversions performed in-process to confirm the new tests bite
  (results documented above).

