# Review — Track D Phase D.3 attempt 1

**Branch**: `track/d-networkx-export`
**Decision**: **REVISIONS_NEEDED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-13

## Summary

Implementation excellent — clean code, good docs, telemetry hygiene, accessible FE. 25 unit/router + 2 Playwright pass. **BUT two formats silently lose data on realistic notebooks** because test fixtures only exercise tiny fully-connected DAGs. AC 3 (50/30 round-trip preserves counts) violated for `edge-list` and at risk for `json-tree`.

## Acceptance criteria

5/7 PASS. AC 3 FAILS for edge-list, AT RISK for json-tree.

## Majors (2)

### M1: `json-tree` silently truncates multi-rooted DAGs

`networkx_export_service.py:393-417`. try/except catches NetworkXError/NotImplemented/TypeError. But `nx.tree_data` does NOT raise on multi-parent DAG (e.g. `a→c, b→c`): walks reachable nodes from chosen root, returns partial tree. With root=alphabetically-first canonical_name=`Alpha`, export omits `Beta` and `Beta→Gamma` edge.

**Verified in-process**: 3-entity / 2-edge multi-rooted DAG returns JSON envelope with 2 nodes + 1 edge after round-trip.

**Real notebooks rarely conform to rooted-tree shape** (entities have multiple parents/types). Users picking "JSON tree" get quietly truncated export, no error.

**Fix recommendation**: detect non-tree shape before `tree_data` (verify root reaches every node) OR unconditionally fall back to `node_link_data` for non-rooted-tree graphs. Add test pinning multi-rooted DAG: `a→c, b→c` must round-trip 3 nodes / 2 edges.

### M2: `edge-list` drops isolated nodes — AC 3 violated

`networkx_export_service.py:358-361`. `nx.read_edgelist` only materializes nodes appearing in edges. **On 50-entity / 30-relation graph (AC 3's exact shape), round-trip reconstructs 34 nodes, not 50** — 16 isolated entities silently disappear.

**Verified in-process**: 3-entity / 1-relation fixture round-trips to 2-node DiGraph; isolated node vanishes.

Existing `test_export_edge_list` only exercises fully-connected 3-node DAG → cannot detect loss. Empty-graph captures 0/0 special case but not realistic-with-isolates.

**Fix recommendation**: either (a) document limitation prominently (module docstring + UI tooltip — "edge-list discards isolated nodes; pick adjacency-list or graphml if isolates matter") + regression test pinning documented behavior, OR (b) change writer to emit isolated nodes (as comments or adjacency-list trailer). Option (a) sane — NetworkX-standard surface, behaves as documented at library level.

## Minors (6, non-blocking)

1. `type_tags` flattening not collision-safe: `["A,B"]` → `"A,B"` → `["A","B"]`. Document in flattening contract.
2. Pickle security caveat missing from docstring + tooltip (`pickle.loads` on untrusted = code-execution)
3. `_FILENAME_UNSAFE_RE` doesn't strip `<>|?*` (consistent with B.2b — file as future cleanup for both)
4. Plan path discrepancy: plan said NotebookHeader.tsx; actual location is schema/page.tsx. Update plan for D.1/D.2 reference.
5. D.0 Minor #1 fixed Python-side (defensible) — promote to SurrealQL when D.2 lands
6. CORS `expose_headers=["Content-Disposition"]` not set globally (B.2b shares this gap)

## Tests (independently verified)

- apps/app-main D.3: 25/25 passed
- apps/app-main regression: 533/533 (no regressions)
- packages/shared: 199 (no regression)
- frontend tsc + lint clean
- Playwright 2/2 against localhost:8512

## Kudos

- Documentation density excellent — module docstring covers flattening WHY, json-tree fallback rationale, telemetry constraint, archived/merged post-filter in one place
- `_GraphBuildSummary` dataclass cleanly separates accountancy from report-building
- 422/404 tested with `dependency_overrides` + `notebook_svc.get.assert_not_called()` belt-and-braces
- Telemetry "no IDs" via JSON-blob substring search catches accidental leaks
- Implementer flagged plan path mismatch + dep-decl rationale + dev-server port collision in self-review

## Decision rationale

Code quality, security posture, telemetry hygiene, accessibility all solid. **Two formats with round-trip data-loss issues are real bugs AC 3 explicitly forbids**. Both easy to fix (guard/fallback OR document+pin). Cannot ship silently — real notebooks routinely produce both shapes.

Both MAJOR (not BLOCKER) because OTHER five formats round-trip correctly and bugs are silent rather than crashing — but mislead users who pick affected formats. ≥1 major → REVISIONS_NEEDED.

## Next steps

1. Fix or document M1 (json-tree multi-rooted DAG truncation) + test pinning chosen behavior
2. Fix or document M2 (edge-list isolated-node drop) + test pinning chosen behavior
3. Optionally address minors 1-3 same revision
4. Re-submit for attempt 2
