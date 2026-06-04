# Review — Track A Phase A.1a attempt 1

**Branch**: `track/a-mineru-service`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-03

## Summary

Phase A.1a delivers a `services/mineru` FastAPI/CLI wrapper, a
`MineruHttpClient` symmetric with the existing `DoclingHttpClient`, and
the Q-A-5 `mineru_layout_parser` that reconstructs `ExtractedDocument`
with populated bboxes from MinerU's `content_list.json`. Every
implementer claim I could independently verify is accurate: 263/263
tests pass, coverage is 98% on the HTTP client and 93% on the layout
parser, `mineru` is correctly absent from `open_notebook.depends_on`,
and every element-type branch in the parser populates a `BoundingBox`
when the source bbox is well-formed. The work has documented
limitations (lossy table merged-cell handling, list-item collapsing,
no live `/process` smoke against a real PDF) but each is openly called
out in `status.md` and the spike doc, and none of them breaks the
phase's acceptance criteria. No blockers, no majors. A handful of
minor improvements worth tracking but not blocking.

## Independent verification of implementer claims

| Claim | Verdict | Evidence |
|-------|---------|----------|
| 263 tests pass, no regressions | **Accurate** | `uv run --project apps/app-main pytest apps/app-main/tests/` → `263 passed in 122.64s` locally. |
| 98% coverage on `mineru_http_client.py` | **Accurate** | `--cov` report: 82 stmts, 2 missing (lines 136-137, OSError on staged-unlink cleanup), 98%. |
| 93% coverage on `mineru_layout_parser.py` | **Accurate** | `--cov` report: 295 stmts, 21 missing (defensive branches), 93%. Combined 94% across both files. |
| `docker compose build mineru` succeeded + `/health` returned 200 | **Cannot re-verify** (build is ~48 min, ~12 GB image) but Dockerfile, api.py, and docker-compose entry are internally consistent and follow the docling pattern. No inspection finding that would have prevented build success. |
| Bbox parity — every element type yields BoundingBox | **Accurate** | Every `_build_*` builder in `mineru_layout_parser.py` consumes `base_kwargs["bbox"]` (or passes it via `**base_kwargs`). When MinerU emits a 4-tuple bbox the parser normalises it; when malformed, `_bbox_from_mineru` returns `None` cleanly (downstream PdfChunkViewer already handles missing bboxes). |
| `mineru` not in `open_notebook.depends_on` | **Accurate** | `docker-compose.yml` line 204-206: `depends_on: [surrealdb, docling]` — no mineru. |

## Acceptance criteria check

| # | Criterion | Status | Notes |
|---|-----------|--------|-------|
| 1 | `docker compose build mineru` completes and `GET /health` returns 200 within 60s | ✅ (claimed) | Implementer reports ~48 min cold build; `/health` returns in <5 s (no model-load on the health path). Dockerfile + compose entry inspection finds nothing that would prevent this. |
| 2 | `curl POST /process` returns valid response with `markdown_path` | 🟡 Deferred | Only mocked tests exercise this. Caveat is documented in `status.md` (first /process triggers ~5-8 GB model download). Acceptable for V1 per the brief's caveats list; A.1c integration testing or A.3 live smoke will close this gap. |
| 3 | `MineruHttpClient().process(Path)` returns `IngestionResult` with populated `document.full_markdown` | ✅ | Verified by `test_process_success_returns_ingestion_result_with_document` against synthetic-but-schema-accurate on-disk fixtures. |
| 4 | `pytest test_mineru_http_client.py` green; coverage ≥ 70% | ✅ | 12/12 passing; 98% coverage. Far exceeds the 70% bar. |
| 5 | `open_notebook` boots without mineru running | ✅ | `depends_on` confirmed minimal; status.md records a `docker compose stop mineru` while `open_notebook` was up. |
| 6 | (Q-A-5) `ExtractedElement` carries populated bbox for every element type | ✅ | Verified via `test_text_levels_map_to_title_heading_and_paragraph` (exact bbox values), `test_table_record_yields_extracted_table_with_rows_and_markdown`, `test_image_and_chart_records_yield_extracted_image_with_resolved_path`, `test_vlm_backend_uses_unit_normalisation` (factor-1 normalisation), and `test_page_furniture_types_classified`. |

## Test status

```
$ uv run --project apps/app-main pytest apps/app-main/tests/ --tb=line -q
263 passed, 3 warnings in 122.64s

$ uv run --project apps/app-main pytest apps/app-main/tests/test_mineru_http_client.py apps/app-main/tests/test_mineru_layout_parser.py --cov=app_main.services.mineru_http_client --cov=app_main.services.parsing.mineru_layout_parser
Name                                                                  Stmts   Miss  Cover
apps/app-main/src/app_main/services/mineru_http_client.py                82      2    98%
apps/app-main/src/app_main/services/parsing/mineru_layout_parser.py     295     21    93%
TOTAL                                                                   377     23    94%
28 passed in 3.27s
```

## Issues found

### 🔴 Blockers (must fix)

None.

### 🟡 Major (must fix)

None.

### 🔵 Minor (optional follow-ups)

1. **List-item collapsing diverges from spike-doc recommendation** — `apps/app-main/src/app_main/services/parsing/mineru_layout_parser.py:293-311`
   - The spike doc §6 / §7d states "flatten `list_items` into one element per item, sharing the parent bbox". The implementation joins all items into one `LIST_ITEM` element with `\n- ` prefixes. Raw items are preserved on `metadata["list_items"]`, so it's recoverable, and the implementer flagged this in `status.md` caveat #3, but the divergence from the design doc is worth a follow-up note.
   - Recommendation: either update the spike doc to reflect the implemented behaviour, or revisit when chunk-builder needs per-item granularity.

2. **`_HTMLTableParser._is_header_cell` is per-cell, not per-row** — `mineru_layout_parser.py:577`
   - `self._is_header_cell` is overwritten on every `td`/`th` start; the value used in `handle_endtag` reflects only the LAST cell of the row. Tables with mixed `<th>`/`<td>` cells in the same row (row headers, e.g. `<th>Row1</th><td>...</td>`) would be misclassified. MinerU output appears to use `<th>` only for the first row, so this works in practice, but the logic is fragile. Add a comment explaining the dependency on MinerU's output format, or track header-ness per row.

3. **`output_dir` parameter unused in `_build_table_element`** — `mineru_layout_parser.py:314-318`
   - Function signature accepts `output_dir: Path` keyword-only but never references it. Either drop the parameter or use it (e.g. to resolve `img_path` for the table's preview image when MinerU writes one). Currently `metadata["img_path"]` stores only the relative string.

4. **`image_paths` collects all files, not images** — `mineru_http_client.py:118-120`
   - `result.image_paths = sorted(images_dir.glob("*"))` enumerates everything under `images/` including any non-image artefacts MinerU might write. Compare docling client (`.glob("*.png")`). Tighten to `glob("*.jpg")` + `glob("*.png")` + relevant formats, or filter by suffix.

5. **Bbox negative-coordinate handling** — `mineru_layout_parser.py:464-466`
   - `width = max(0.0, x1 - x0)` clamps negative deltas to 0 (handles `[x1,y1,x0,y0]` reversal silently). A bbox with reversed coords ends up as a zero-area box at `(x0, y0)`. Consider logging once at WARNING level so we notice if MinerU emits these.

6. **Service exposes `/process` on host port 8104 with no authentication** — `services/mineru/api.py:153` + `docker-compose.yml:149-150`
   - Same security posture as docling. The `input_path` field accepts arbitrary host paths (mounted volumes) without sanitisation; a host-network attacker could exfiltrate any file under `/data/`. Pre-existing pattern (docling is identical), so no new attack surface, but worth treating as a known limitation. Track for the integration phase (A.3) — possibly bind to internal docker network only or add a shared-secret header.

7. **`_call_service` raises if `results[0].success` is False but the response shape supports multi-file batches** — `mineru_http_client.py:170-173`
   - The client always submits a single file, so this is correct in current usage. If anyone reuses the client for batch processing later they will silently lose all but the first result. Add a docstring noting "single-file usage only" or guard explicitly against `len(results) > 1`.

8. **`mineru[all]>=2.0` is unpinned to a major** — `services/mineru/requirements.txt:5`
   - MinerU 2.x is still evolving; an upstream breaking minor release (e.g. CLI flag rename, output-tree restructure) would surface as a runtime failure. Pin to `mineru[all]>=2.0,<3.0` or stricter to make image rebuilds reproducible.

9. **`MineruHttpClient` and `DoclingHttpClient` import `IngestionResult` from different paths** — cosmetic
   - mineru: `from ingestion.models import IngestionResult, SourceMetadata, SourceType`
   - docling: `from ingestion.models.source import IngestionResult, SourceMetadata, SourceType`
   - Both resolve to the same symbol via `__init__.py` re-exports. Pick one for consistency. Mineru's choice (the `__init__` re-export) is the conventional public API; docling could be updated to match in a future cleanup.

10. **No tests for malformed table HTML** — `test_mineru_layout_parser.py`
    - The HTML→rows parser has a try/except in `_parse_html_table` that returns `([], [])` on exception (lines 606-607), but no test exercises this branch. Same for tables with `<colspan>`/`<rowspan>` (lossy fallback at the markdown level). Both are documented caveats, but a fixture asserting current behaviour would lock in the V1 contract.

11. **`extract_annotations` and `read_handwriting` fields not mirrored** — `services/mineru/api.py:100-115`
    - The docling `ProcessRequest` accepts `extract_annotations` and `read_handwriting`; the MinerU request omits them deliberately (different parser capability set). Not a defect — but worth a one-line comment in `MineruHttpClient._call_service` explaining why we don't pass those, so a future maintainer doesn't try to "make it symmetric" by adding them.

## Decision rationale

Phase A.1a is a foundation phase whose acceptance criteria are
deliberately scoped to "the service builds, exists, responds, and
fits the docling-symmetric interface." All six criteria are met
(criterion 2's live `/process` smoke is acknowledged as deferred to
A.3, which is consistent with the brief). The Q-A-5 bbox parser is
implemented thoughtfully — every element-type branch carries through
the `BoundingBox`, the VLM/pipeline normalisation difference is
handled, page furniture is preserved with appropriate `ElementType`
mappings, and the empty-document case raises rather than silently
producing a degraded artefact.

Test quality is genuinely good: 28 tests across 12 + 16 cases that
exercise behaviour (titles vs headings, multi-page grouping, table
extraction, image-path resolution, VLM unit normalisation, malformed
JSON, missing files, HTTP 5xx, service-reported failure, layout-
parse failure) — not just "code runs without crashing". Mocking is
at the right boundary (`httpx.MockTransport` for the network,
synthetic on-disk fixtures for the layout parser).

The minor items above are real follow-ups but none would justify
blocking merge of A.1a. The `SourceExtractor` integration in A.1b
will exercise additional code paths and is the right time to revisit
some of these (especially #4 image-path filter and #2 table header
classification).

## Next steps

**APPROVED** for human approval / merge. Recommend the orchestrator:

1. Open the PR for `track/a-mineru-service` against `main`.
2. File the 11 minor items above as follow-up issues (or as a single
   "A.1a follow-ups" issue) so they're not lost. Items 1, 4, and 8
   are the highest signal.
3. Schedule the live `/process` smoke against a real PDF before A.1c
   merges — it's the only acceptance criterion not yet exercised by
   automation, and A.1c will lean on actual MinerU output for the
   auto-fallback path.
4. Proceed to plan/start Phase A.1b (`parser_engine` rename +
   migration + dispatcher) — A.1a's HTTP client is independently
   usable and unblocks downstream wiring.
