# app-main test fixtures

Tiny PDF fixtures (< 20 KB each) used by Track A's integration tests.
**Not** representative test corpora — the fixtures exist so deterministic
file paths can be referenced from `frontend/e2e/track-a/parser-engine-integration.spec.ts`
and the threshold-tuning baseline row in
`docs/tracks/A-mineru/threshold-tuning.md`.

## Files

| File | Shape | Used by |
|------|-------|---------|
| `synthetic_clean_text.pdf` | Single page with one heading + dense lipsum prose | Integration spec upload; tuning baseline ("clean text" row) |
| `synthetic_table_heavy.pdf` | Single page with an 8-row text-rendered table | Tuning baseline ("table-heavy" row) |
| `synthetic_image_only.pdf` | Single rasterised page (no text layer) | Tuning baseline ("scanned / image-only" row) |

## Regeneration

```bash
uv run --project apps/app-main python apps/app-main/tests/fixtures/generate.py
```

The generator uses only stdlib + Pillow (already a transitive dep of
docling); it deliberately does **not** add `reportlab` to the workspace
(see `docs/tracks/A-mineru/plan-A.3.md` §13). The output is bit-stable
on any platform with the same Pillow version — running the script just
refreshes the bytes.

## Why so small

A real PDF corpus for tuning lives at `docling_input/` (gitignored,
developer-local). These fixtures are intentionally minimal: they would
score badly on `heading_rate` and `text_density` and are **not** part
of the tuning decision. See
`docs/tracks/A-mineru/threshold-tuning.md` for the curated corpus
breakdown and the final threshold call.
