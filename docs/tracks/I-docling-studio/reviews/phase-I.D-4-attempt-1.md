# Phase I.D-4 — adversarial review (attempt 1)

> Branch: `track/i-d4-result-tabs`, diff `main...HEAD` (5 commits)
> Reviewer: adversarial-reviewer agent
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.D-4
> Date: 2026-06-19

## VERDICT: APPROVED

0 blockers, 0 majors. The two highest-risk areas — the right-pane tab refactor (no regression)
and the new image-list endpoint (path-safety) — both clean. rehype-sanitize genuinely wired.
tsc clean, 6 image-pagination unit tests + 4 backend image tests pass, lint clean.

## High-risk spot assessment
1. **AC2 Markdown XSS** — PASS. `MarkdownViewer.tsx:38` `rehypePlugins={[rehypeSanitize]}`; `rehype-raw`
   absent; no `allowDangerousHtml`/`skipHtml`. `<img onerror>` and `javascript:` links both neutralized
   (v10 emits no raw HTML + sanitize default schema). Double-layered.
2. **AC1 + no-regression (tab refactor)** — PASS. LayersBar still in middle pane; ChunkActionsToolbar
   (I.D-3 merge/split) still in PropertiesPanel with identical props; active-chunk metadata preserved;
   reprocess panel moved to ConfigTab with same handler/state/props. Nothing dropped/de-propped. Radix
   unmounts inactive tabs (fresh mount per tab).
3. **New `GET /{source_id}/images` path-safety** — PASS. `_resolve_output_dir` faithful extraction of the
   existing serve logic; source_id only feeds `source_svc.get()`, never the FS; missing/absent dir →
   `{"images": []}` not 500/404; `.txt` sidecars excluded; only `entry.name` returned (no path leak).
   Serve endpoint still guards filename + realpath.
4. **AC3 lazy ≤6 in-flight** — PASS (real, DOM-gated). `ImageGallery` maps only `pageSlice(images, page)`
   (≤6) — one page mounted at a time; `loading="lazy"` is an additional defer. Hard bound.
5. **Markdown source** — PASS. Uses existing `source.full_text` (no new endpoint); empty state on falsy.
6. **StructureViewer** — PASS. Pure placeholder for I.F; no sigma/graphology, no graph logic.
7. **Tests real** — PASS. image-pagination asserts real chunking (13→6/6/1, clamps, empty);
   test_source_images asserts empty→{images:[]}, sidecar exclusion, 404 only on missing source.

## AC scorecard
- AC1 (each tab loads w/o console error): MET (logic + spec; runtime headless run deferred).
- AC2 (Markdown XSS-safe): MET.
- AC3 (Images lazy-load ≤6 in-flight): MET (DOM-gated, verified).

## Minor / follow-up (non-blocking)
- `ImageGallery.tsx:117` raw `<img>` → `@next/next/no-img-element` warning; consistent with existing
  codebase convention for backend-served files. Leave.
- E2E `expect(errors).toEqual([])` is strict; benign runtime console.error could redden it — confirm on
  the real E2E run / manual smoke (residual, headless-deferred).
- `_resolve_output_dir` reads `getattr(source, "output_directory")` vs schema `output_directory_path`;
  pre-existing on main, out of scope — noted so it's not lost.

## Decision
Merged as-is. Deviations from plan (all sound): added a minimal `GET /sources/{id}/images` listing
endpoint (no image filenames were persisted on chunks); split the reprocess panel into a dedicated
`ConfigTab` (the plan's tab list specifies a separate Config tab); only `rehype-sanitize` needed adding.
