# Track OKF — Retrospective (Open Knowledge Format export/import)

> Closing date: 2026-07-24
> Branches: `track/okf1-export`, `track/okf3-import`, `track/okf4-mcp`,
> `track/okf5-ui-docs`
> Final state: see [`status.md`](./status.md); spec pin in
> [`SPEC-v0.1.md`](./SPEC-v0.1.md).

This retrospective draws on the per-phase `status.md` entries. It informs
future interchange/adapter tracks; it is not the project's general
lessons-learned doc.

## Summary

Track OKF added a fourth export target — Google Cloud's **Open Knowledge
Format v0.1** — plus its inverse import, an MCP agent surface, and a
notebook-header export UI. It closed as a deliberately **small, reuse-heavy**
track: a conformance layer over the existing Track D projection and the
K.3/K.5 dedup stack, not a new projection engine.

What shipped, phase by phase:

- **OKF.1** — the bundle mapper (`okf_export_service.py`): the shared
  entity/relation/note projection → an OKF v0.1 Knowledge Bundle (one Markdown
  concept per entity + `index.md`, standard-markdown links, tree + manifest).
  Byte-stable output (sorted keys, caller-injected timestamp) and the
  omitted-field ledger baked into the `ExportReport`.
- **OKF.2** — `POST /notebooks/{id}/export/okf`: inline build+zip for normal
  notebooks (report in the `X-OKF-Export-Report` header), a job-queue seam
  (`JobType.EXPORT_OKF`, `202` + pollable `job_id`) for notebooks over the
  entity threshold.
- **OKF.3** — `okf_import_service.py`: parse an external bundle → entities /
  relations / notes / sources, deterministic `(name, type)` upsert dedup +
  injection-safe RELATE, idempotent under re-import, all provenance-tagged
  `okf-import`. Malformed *concepts* are skipped non-silently; a malformed
  *container* raises.
- **OKF.4** — MCP `export_okf` / `import_okf` tools (Track W conventions) plus
  the REST import endpoint that OKF.3 had deferred.
- **OKF.5** — the "Export OKF" dialog (zip-only, mirrors the Obsidian dialog),
  the omitted-field ledger surfaced in the UI, and this docs/close-out.

## Decisions

- **OKF-D1 (spec pinning).** OKF v0.1 is explicitly "a starting point, not a
  finished standard", so the mapping is isolated in one module and the spec is
  vendored/pinned in `SPEC-v0.1.md`. A spec bump is a localised change, not a
  silent break across the exporters.
- **OKF-D2 (avoid a second drift surface).** The OKF exporter reuses the Track
  D shared projection + `ExportFilter` + `ExportReport` rather than forking a
  parallel projection. It also reuses `ObsidianExportRequest` as its request
  contract (OKF only ever emits a zip, so `mode` is fixed to `"zip"`), so the
  `/export-preview` parity invariant holds for OKF for free — dialog counts and
  the actual bundle cannot drift.
- **OKF-D3/D4 (lossy = drop-and-report, never silent).** Embeddings, chunk
  provenance (Track X), verdict/contradiction edges (Track Z), and
  hybrid-search signal (Track W) have no OKF representation and are dropped —
  **not** stashed as non-portable `x-` frontmatter. `OKF_OMITTED_FIELDS` is the
  single source of truth, copied verbatim into every report and surfaced
  through the REST header, the MCP report, and the UI ledger.

## Honest scoping

Track OKF is **mostly a conformance layer over the Obsidian export**. The
genuinely new work was the SPEC-conformant frontmatter/link mapping + bundle
tree/manifest (OKF.1) and the import parse/dedup path (OKF.3); everything else
is thin wiring over the existing exports router, MCP patterns, and the D.1c
export dialog. The value is realised even as "one more interchange target
alongside Obsidian/TTL/JSONL/NetworkX" — the track deliberately did **not**
re-architect around OKF, and adoption of the (Vertex-anchored) format remains
an open bet.

## Known follow-ups

- **Async large-notebook artifact store.** ~~not built~~ **backend done
  (2026-07-24)**: the deferred handler persists the zip under
  `<vault_path>/okf_exports/` (keyed by notebook id via the shared
  `okf_artifact_path`), and `GET /notebooks/{id}/export/okf/download` serves it
  — `404` until the job completes, so a client polls to `200`. The `202` body
  now carries a `download_url`. Remaining: a **download-polling UI** for the
  deferred path (the dialog still relies on the "queued" toast).
- **MCP lazy-import layering exception.** `export_okf` / `import_okf` are the
  one deliberate exception to surrealdb-service's repo-direct, no-app-main rule:
  the OKF services are app-main orchestrators, so the tools reach them via a
  lazy import inside the tool body and degrade to `import_error` when app-main
  is absent. Documented in the MCP server module header; revisit if/when the
  OKF services move to a shared package.
- **Import UI.** ~~export only~~ **DONE (2026-07-24)**: `OkfImportDialog` +
  `use-okf-import` add an "Import OKF" entry to the notebook header — upload a
  `.zip`, toggle dedup, and see the created/matched counts + the skip /
  dangling-link ledger. Wraps the existing `POST /notebooks/{id}/import/okf`.
- **External-id round-trip.** An entity without a real external id exports a
  synthesised `urn:open-notebook:…`; on import it is kept as an `okf_resource`
  property but not re-promoted to `external_ids`, so a re-export regenerates a
  fresh urn (harmless, but not idempotent at the urn level).
