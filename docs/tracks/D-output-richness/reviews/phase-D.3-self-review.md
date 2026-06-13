# Phase D.3 self-review — NetworkX 7-format export

**Branch**: `track/d-networkx-export`
**Commits**: `d0f3d87..dd9ae33`

## Acceptance criteria check

| AC | Description | Result |
|----|-------------|--------|
| 1  | POST `/export-networkx` `{format: graphml}` → 200 + `application/xml` + parses via `nx.read_graphml(BytesIO(body))` | PASS — `test_graphml_happy_path` |
| 2  | Same happy-path test for every other format (GEXF, GML, JSON-tree, edge-list, adjacency-list, pickle) | PASS — 7 service tests + 3 router tests |
| 3  | Round-trip preserves nodes + edges + `type_tags` | PASS — `test_graphml_round_trip_preserves_canonical_name_and_counts` + `test_type_tags_flatten_and_unflatten_via_graphml` |
| 4  | Filter applied (same semantics as D.0) | PASS — request body carries `ExportFilter`, repo methods consume it as-is |
| 5  | Telemetry payload includes `format: "<format>"` | PASS — `test_telemetry_payload_has_counts_only` + `test_telemetry_fires_with_format_in_payload` |
| 6  | Playwright verifies 2 formats (GraphML + JSON-tree) | PASS — `frontend/e2e/track-d/networkx-export.spec.ts` |
| 7  | No new migrations, no behavioural change | PASS — no SurrealQL touched; D.0 read methods reused as-is |

## Round-trip outcomes per format

All 7 formats round-trip cleanly. Service tests use `nx.read_*` (or the matching `pickle.loads` / `nx.tree_graph` / `nx.node_link_graph`) on the bytes the writer produced and assert on node + edge counts:

| Format          | Wire size (3-node graph) | Round-trip reader     | Status |
|-----------------|--------------------------|-----------------------|--------|
| graphml         | ~1.5 KB                  | `nx.read_graphml`     | PASS   |
| gexf            | ~1.5 KB                  | `nx.read_gexf`        | PASS   |
| gml             | ~200 B                   | `nx.parse_gml`        | PASS   |
| json-tree       | varies                   | `nx.tree_graph`       | PASS   |
| edge-list       | small                    | `nx.read_edgelist`    | PASS   |
| adjacency-list  | small                    | `nx.read_adjlist`     | PASS   |
| pickle          | varies                   | `pickle.loads`        | PASS   |

Edge case worth recording: `nx.write_edgelist` of an empty graph legitimately produces 0 bytes (no edges to encode). The reader still returns an empty `DiGraph` from empty bytes, so the semantics are preserved. The empty-notebook parametrized test handles this special-case in its bytes assertion; documented inline in the test docstring.

## Q-D-8 telemetry compliance

`record_metric("export.networkx", payload, ...)` payload audited in `test_telemetry_payload_has_counts_only`: the JSON repr of the payload is searched for `"entity:"` and `"entity_id"` substrings; both must be absent. Confirmed.

The payload itself is the `ExportReport.model_dump(mode="json")` merged with `{"format": <fmt>}`. `ExportReport` was designed in D.0 to be counts-only by spec.

## D.0 follow-up Minor #1 — archived / merged Entity.status

Picked the **cheaper option**: a Python-side post-filter in `NetworkxExportService._build_graph` that drops `Entity.status in {archived, merged}` after the D.0 repository returns rows. Documented in the module docstring under "Archived/merged entities".

Rationale:
- The D.0 repository SurrealQL gates on `orphan_status`, a *different* axis (B.5b lifecycle vs B.5b reconcile). Adding a `status` predicate to the existing query would require a repo change that touches the D.0 contract.
- The post-filter costs one boolean per entity. At V1 scale (10K entities ceiling per the plan), that's negligible compared to the round-trip we just saved on the DB.
- The test `test_archived_and_merged_entities_are_filtered_out` verifies both:
  1. archived/merged nodes disappear from the export, and
  2. an edge pointing into a dropped node is silently dropped along with it (Q-D-4 semantics, applied to this branch).

Trade-off: D.1 and D.2 will need the same post-filter when they land, or D.0 should be patched to gate on `status` in SurrealQL. Recommend the latter once two services need it — DRY wins. Noted for the D.1/D.2 planner.

## Issues to flag

1. **NotebookHeader location surprise.** The plan asks for the menu to live "next to the existing TTL download (B.2b)". `NotebookHeader.tsx` lives at `frontend/src/app/(dashboard)/notebooks/components/NotebookHeader.tsx`, not `frontend/src/components/notebooks/NotebookHeader.tsx` as the plan states. The TTL download itself isn't in NotebookHeader — it lives in the Schema page (`schema/page.tsx`). I wired the new menu next to TtlDownloadButton in that same Schema page, which is the user-visible adjacent location even though the file path differs from the plan. Recommend the planner update the path reference in future plans.

2. **networkx>=3.0 was already a transitive dep** via `entity-filtering` (pulled in by app-main), so adding it explicitly to `apps/app-main/pyproject.toml` is documentation rather than a new dep. The explicit decl is still worth keeping so a direct `import networkx as nx` in app-main code is auditable without traversing the workspace graph.

3. **Frontend dev-server port.** Tests need a Next dev server running on the worktree's checkout. The default port 8502 is occupied by a stale server elsewhere on the host, so I ran tests against an explicit `PLAYWRIGHT_BASE_URL=http://localhost:8512`. Recommend other implementors follow the same pattern when their worktree port collides.

## Test counts

- `apps/app-main`: 508 → 533 (+25 from D.3: 19 service + 6 router)
- `packages/shared`: 199 → 199 (no change — D.3 doesn't touch shared models)
- `frontend e2e (track-d)`: 0 → 2 (GraphML + JSON-tree dropdown specs)

## Quality gates summary

| Gate                                                        | Result |
|-------------------------------------------------------------|--------|
| `pytest tests/test_networkx_export_service.py tests/test_exports_router.py -v` | 25 passed |
| `pytest -q apps/app-main` regression                        | 533 passed |
| `pytest -q packages/shared` regression                      | 199 passed |
| `npx tsc --noEmit`                                          | clean |
| `npm run lint`                                              | clean (no new warnings) |
| `npx playwright test e2e/track-d/networkx-export.spec.ts`   | 2 passed |
