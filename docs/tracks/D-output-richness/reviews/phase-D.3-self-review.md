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

---

## Attempt 2 fixes

Reviewer rejected attempt 1 with **REVISIONS_NEEDED** — 0 blockers + 2 majors + 6
minors. This section records the attempt-2 fixes.

### M1 (json-tree multi-rooted DAG silent truncation) — FIXED

Resolution: **code fix, not document-and-pin**. Pre-check the graph with
`nx.is_arborescence` and route anything that fails the structural test through
`nx.node_link_data` instead of `nx.tree_data`. The previous try/except shape
relied on `tree_data` raising on non-tree input; it doesn't on multi-rooted DAGs,
which is precisely the silent-loss bug the reviewer pinned.

Two new pin tests:

* `test_json_tree_multi_rooted_dag_falls_back_to_node_link` — 3 entities
  `a, b, c` with edges `a→c, b→c` (multi-rooted DAG). Asserts the envelope
  is `node_link_data` shape (keys `nodes` + `links` + `directed`), the
  round-tripped graph has 3 nodes + 2 edges, and the build summary
  matches.
* `test_json_tree_rooted_arborescence_uses_tree_data` — counterpart pin
  for a true rooted tree (`root → a → b`): envelope must have `id` +
  `children` keys (tree_data shape), no `nodes` key, and round-trips
  via `nx.tree_graph` with 3 nodes + 2 edges. Stops a future drift
  where someone accidentally routes everything through node-link.

### M2 (edge-list drops isolated nodes) — DOCUMENTED + PINNED

Resolution: **Option A — document + pin** (per reviewer's recommendation). Three
surfaces updated:

1. Service module docstring gains a `Format limitations` section calling
   out edge-list's isolate-drop behaviour, the json-tree fallback
   conditions, and the pickle code-execution caveat (minor 2 bundled).
2. `NetworkxExportMenu.tsx` `FormatOption` gains an optional `caveat`
   field. Edge-list shows `drops isolated entities` and pickle shows
   `trust only files you generated yourself` as italic amber-coloured
   text below the description, plus the same string is concatenated
   into the menu item's `aria-label` so screen readers announce it.
3. New pin test `test_edge_list_documented_behavior_drops_isolated_nodes`
   — 3 entities `a, b, c` with single edge `a→b`. The isolated `c`
   must NOT appear in the round-tripped graph; exactly 2 nodes + 1 edge
   survive. The service report still counts all 3 entities (no
   accountancy regression).

### Pin test stdout (the two new majors-fix tests)

```
$ uv run --extra dev pytest \
    apps/app-main/tests/test_networkx_export_service.py::test_json_tree_multi_rooted_dag_falls_back_to_node_link \
    apps/app-main/tests/test_networkx_export_service.py::test_edge_list_documented_behavior_drops_isolated_nodes \
    -v
============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.4.2, pluggy-1.6.0
configfile: pyproject.toml
plugins: anyio-4.11.0, Faker-37.11.0, langsmith-0.4.37, asyncio-1.2.0
asyncio: mode=Mode.AUTO ...
collecting ... collected 2 items

apps/app-main/tests/test_networkx_export_service.py::test_json_tree_multi_rooted_dag_falls_back_to_node_link PASSED [ 50%]
apps/app-main/tests/test_networkx_export_service.py::test_edge_list_documented_behavior_drops_isolated_nodes PASSED [100%]

============================== 2 passed in 4.92s ===============================
```

### Minors bundled in attempt 2

| # | Minor | Action |
|---|-------|--------|
| 1 | `type_tags` flattening edge cases | Documented in module docstring under `Attribute flattening edge cases` |
| 2 | Pickle security caveat | Added to module docstring AND UI dropdown caveat |
| 3 | `_FILENAME_UNSAFE_RE` strip set | Deferred (file as separate cleanup with B.2b) |
| 4 | Plan path discrepancy | Plan §D.3 updated to point at `schema/page.tsx` |
| 5 | D.0 SurrealQL filter promotion | Deferred (will happen when D.2 lands) |
| 6 | CORS `expose_headers` global | Deferred (follow-up infra PR) |

### Attempt 2 quality gates

| Gate                                                        | Result |
|-------------------------------------------------------------|--------|
| New pin tests (3: 2 majors + 1 counterpart)                | 3 passed |
| `pytest -q apps/app-main` regression                        | 536 passed (was 533 + 3 new) |
| `pytest -q packages/shared` regression                      | 199 passed (no change) |
| `npx tsc --noEmit`                                          | clean |
| `npm run lint` (NetworkxExportMenu only)                    | clean (no new warnings) |
| `npx playwright test e2e/track-d/networkx-export.spec.ts`   | 2 passed (against localhost:8513 — 8502/8503/8512 occupied by sibling worktrees) |

### Rebase

`git rebase origin/main` against the post-attempt-1-review tip (`15e824a docs(track-d): D.3 attempt-1 review`). Clean fast-forward of the 3 D.3 commits; no merge markers, no conflict, no manual intervention required.
