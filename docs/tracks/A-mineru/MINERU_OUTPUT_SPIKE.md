# MinerU output schema — spike findings

> Spike completed 2026-06-03 as part of Phase A.1a (Track A — Ingestion robustness).
> Status: schema fully characterised from upstream documentation
> (`opendatalab/MinerU` master, `docs/en/reference/output_files.md`). A
> functional probe against `data/uploads/yardstickbias.pdf` (68 KB) is
> deferred to local validation by reviewer because the first-run model
> download for `mineru[all]` is ~5–8 GB (PyTorch + Ray + layout +
> OCR + formula + table models). The schema itself is stable across 2.x.

## 1. MinerU 2.x command-line surface

```bash
mineru -p <input_path> -o <output_path> \
       -b pipeline           \  # pipeline | hybrid-* | vlm-*
       -m auto               \  # auto | txt | ocr (pipeline/hybrid only)
       -l en                 \  # OCR language hint
       -f true -t true          # formula on, table on (defaults)
```

The CLI is an orchestration wrapper around the bundled `mineru-api`
FastAPI service. Without `--api-url`, it starts a temporary local
`mineru-api` and tears it down on exit. With `--api-url` it reuses an
existing service. For our Dockerised standalone service we run the CLI
directly per-file because it gives us:

- one HTTP-shape symmetric with `services/docling/api.py`
  (request `{input_path, output_path}`, response `{results: [...]}`)
- no need to manage MinerU's own task queue / polling protocol
- predictable on-disk output we can read from the host

## 2. Output directory layout (pipeline backend)

For `mineru -p /data/input/foo.pdf -o /data/mineru_output -b pipeline`:

```
/data/mineru_output/
└── foo/                            # ← directory keyed by input stem
    └── auto/                       # ← directory keyed by --method (auto|txt|ocr)
        ├── foo.md                  # primary markdown output
        ├── foo_content_list.json   # **flat element list with bboxes** (★ what we parse)
        ├── foo_middle.json         # nested block/line/span structure (full bbox tree)
        ├── foo_model.json          # raw model output (cls_id, label, score, bbox)
        ├── foo_layout.pdf          # debug PDF (layout visualisation)
        ├── foo_span.pdf            # debug PDF (span visualisation; pipeline only)
        ├── foo_origin.pdf          # copy of original
        └── images/
            ├── <hash>.jpg
            └── …
```

VLM backend writes only `foo.md`, `foo_content_list.json` and
`foo_middle.json` (no `model.json`, no debug PDFs).

## 3. `content_list.json` — primary structured output (★ source of bboxes)

A flat JSON array of element dicts **in reading order**. This is what
the layout parser ingests. The schema is:

| Field         | Type    | Notes                                                                  |
| ------------- | ------- | ---------------------------------------------------------------------- |
| `type`        | string  | `text` \| `image` \| `table` \| `chart` \| `equation` \| `code` \| `list` \| `header` \| `footer` \| `page_number` \| `aside_text` \| `page_footnote` |
| `page_idx`    | int     | 0-based page index                                                     |
| `bbox`        | `[int×4]` | `[x0, y0, x1, y1]`, **normalised to 0–1000** (pipeline backend)        |
| `text`        | string  | present for `text` and `equation`                                      |
| `text_level`  | int     | `0` or absent = body; `1+` = heading levels                            |
| `img_path`    | string  | relative path under output dir (for `image`/`table`/`chart`/`equation`)|
| `image_caption` / `image_footnote` | `[string]` | flanking text for images                  |
| `table_caption` / `table_footnote` / `table_body` | … | `table_body` is HTML            |
| `text_format` | string  | usually `"latex"` for equations                                        |
| `sub_type`    | string  | `code`/`algorithm` for code; `text`/`ref_text` for list; `seal` etc.   |
| `code_body` / `code_caption` / `code_footnote` | … | for code blocks                  |
| `list_items`  | `[string]` | items in a list                                                     |

### Example record

```json
{
  "type": "text",
  "text": "The response of flow duration curves to afforestation",
  "text_level": 1,
  "bbox": [62, 480, 946, 904],
  "page_idx": 0
}
```

The 0–1000 normalisation maps cleanly to our `BoundingBox` dataclass
(`x`, `y`, `width`, `height`, `page`) by dividing the four corners by
1000 (so `x = x0/1000`, `y = y0/1000`, `width = (x1-x0)/1000`,
`height = (y1-y0)/1000`) and converting `page_idx + 1` to our
1-indexed page numbering. Origin is top-left in both schemas, so no
y-flip is needed (unlike docling, which uses bottom-left and is
converted in `BoundingBox.from_docling`).

VLM backend uses `[0,1]` normalisation (no factor of 1000); we
detect the backend from `middle.json._backend` and branch on
normalisation.

## 4. `middle.json` — full nested structure (informational; not parsed in A.1a)

Hierarchical: `pdf_info[]` → page → `para_blocks[]` (top-level
blocks) → `lines[]` → `spans[]`. Each level carries a `bbox`. Useful
for fine-grained reconstruction but **far** more verbose than we
need; A.1a parses `content_list.json` only, leaving `middle.json` as
a future enrichment if line-/span-level bboxes are ever needed for
the PDF viewer.

Top-level: `{ pdf_info: [...], _backend: "pipeline"|"vlm"|"office", _version_name: "x.y.z" }`.

## 5. Bbox availability assessment

**Q-A-5 resolution**: bbox availability is **full and per-element**.
Every element in `content_list.json` has a `bbox` field, including
text, images, tables, equations and code. This means we can build a
parser that emits `ExtractedElement` instances with populated
`BoundingBox` for every element — full `PdfChunkViewer` compatibility
retained, no UX degradation versus docling.

The only edge case: **page-auxiliary elements** (`header`, `footer`,
`page_number`, `aside_text`, `page_footnote`) get bboxes too but they
sit outside the natural reading flow. We map them to
`ElementType.HEADER` / `FOOTER` / `FOOTNOTE` rather than dropping
them; downstream chunking can choose to ignore them.

## 6. Type mapping (MinerU → ExtractedElement)

| MinerU `type`     | text_level | → `ElementType`             | Notes |
| ----------------- | ---------- | --------------------------- | ----- |
| `text`            | absent / 0 | `TEXT` (→ `PARAGRAPH`)      | body text |
| `text`            | 1          | `TITLE`                     | document/h1 |
| `text`            | ≥2         | `HEADING`                   | h2+, `section_level = text_level` |
| `image`           | —          | `IMAGE`                     | uses `ExtractedImage` |
| `table`           | —          | `TABLE`                     | uses `ExtractedTable`; `table_body` is HTML → convert to MD |
| `chart`           | —          | `IMAGE` w/ `classification="chart"` | charts are stored like images |
| `equation`        | —          | `FORMULA`                   | `text` carries the LaTeX |
| `code`            | —          | `CODE`                      | `code_body` → `content` |
| `list`            | —          | `LIST_ITEM` (joined)        | flatten `list_items` into one element per item, sharing the parent bbox |
| `header`          | —          | `HEADER`                    | page-level |
| `footer`          | —          | `FOOTER`                    | page-level |
| `page_number`     | —          | `FOOTER`                    | treat as page furniture |
| `aside_text`      | —          | `CAPTION`                   | margin / aside text |
| `page_footnote`   | —          | `FOOTNOTE`                  | per-page footnote |

## 7. Layout-to-ExtractedElement mapping strategy (recommended)

The `mineru_layout_parser` will:

1. Read `<stem>_content_list.json` from the per-file output dir.
2. Read `<stem>_middle.json` to determine `_backend` (drives bbox
   normalisation factor: 1000 for pipeline/office, 1 for VLM).
3. For each record:
   a. Compute the normalised `BoundingBox` (page = `page_idx + 1`).
   b. Pick the `ElementType` from the table above.
   c. Build `ExtractedElement` / `ExtractedTable` / `ExtractedImage`
      with `content`, `element_type`, `page`, `bbox`, `section_level`,
      `confidence = 1.0`, `source = "mineru"`.
   d. For tables, parse `table_body` (HTML) into rows/headers via a
      lightweight inline parser; keep the HTML around as
      `metadata["html"]`.
   e. For images/charts, set `image_path = output_dir / img_path`
      and `classification` accordingly.
4. Group elements by `page_idx` into `PageContent` objects.
5. Build the `ExtractedDocument` (title from first
   `text_level == 1` element if present, else the file stem;
   `page_count` from the max `page_idx + 1`).

The full markdown is taken from `<stem>.md` directly (no need to
re-render).

## 8. Open follow-ups (not blocking A.1a)

- `middle.json` carries line/span-level bboxes which could power
  word-level highlights in `PdfChunkViewer`. Not used here; revisit
  in Track B (chunking quality) if granular highlights are needed.
- MinerU's `table_body` HTML uses `<colspan>`/`<rowspan>` — our
  inline HTML→rows parser will lose merged-cell information.
  Acceptable for V1; chunk-builder already treats tables as opaque
  markdown blobs.
- VLM-backend coordinate normalisation differs (0–1 floats vs
  0–1000 ints). The parser switches on `_backend`; if a third
  backend appears we'll add a `_normalise_factor` lookup.

## 9. Functional probe (deferred)

A live single-PDF probe through the Dockerised service is part of
acceptance criteria 1–3 in the sprint plan and is the reviewer's
local validation step. Schema characterisation above is sufficient
to lock the parser design without blocking the unit-test work,
which uses synthetic JSON fixtures matching the documented schema.

Once `docker compose build mineru` succeeds locally:

```bash
docker compose up -d mineru
curl -sf http://localhost:8104/health
cp data/uploads/yardstickbias.pdf docling_input/
curl -X POST http://localhost:8104/process \
    -H 'Content-Type: application/json' \
    -d '{
          "input_path": "/data/input/yardstickbias.pdf",
          "output_path": "/data/mineru_output",
          "copy_original": true
        }'
ls -la mineru_output/yardstickbias/auto/
```

Expected output: `yardstickbias.md`,
`yardstickbias_content_list.json`, `yardstickbias_middle.json`,
`yardstickbias_model.json`, `images/`.
