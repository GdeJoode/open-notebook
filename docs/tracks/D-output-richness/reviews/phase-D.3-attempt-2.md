# Review — Track D Phase D.3 attempt 2

**Branch**: `track/d-networkx-export`
**Decision**: **APPROVED**
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-13

## Summary

Attempt 1's two majors both addressed cleanly. M1 fixed code-side (pre-check
+ lossless fallback); M2 documented + pinned in three surfaces. All 22 service
tests pass (was 19); regression 536/536; shared 199/199; tsc + lint clean;
Playwright 2/2. Mental inversion of both pins confirms they will fail if the
fixes regress. Ship.

## Acceptance criteria

7/7 PASS. M1 and M2 (json-tree + edge-list round-trip) now pin behavior
explicitly; AC 3 holds on the 3-entity / 2-edge multi-rooted DAG shape that
broke in attempt 1.

## Attempt-1 majors revisit

### M1 — json-tree multi-rooted DAG truncation: **FIXED (code)**

Read `_serialize_json_tree` at `networkx_export_service.py:402-456`:

- L424: empty-graph guard → `node_link_data` (avoids `tree_data` raise).
- L429: `nx.is_arborescence(graph)` **pre-check** — required structural
  test that catches what `tree_data`'s try/except missed.
- L429-444: TRUE branch picks deterministic root (min by canonical_name,
  ties broken by node id) → `nx.tree_data`.
- L450-456: FALSE branch logs at INFO, routes through `nx.node_link_data`
  (lossless for multi-rooted DAGs, cycles, disconnected components,
  multi-parent diamonds).

Pin test `test_json_tree_multi_rooted_dag_falls_back_to_node_link`
(`test_networkx_export_service.py:367-408`):

- Fixture: 3 entities `a, b, c` + edges `a→c, b→c` (multi-rooted DAG).
- Asserts envelope shape (`nodes` + `links` + `directed: true`) — NOT
  `tree_data`'s `id`/`children`.
- Asserts round-trip via `nx.node_link_graph` → 3 nodes + 2 edges (NOT 2
  + 1 from the silent walk).
- Asserts `report.entities_written == 3`, `report.relations_written == 2`.

Counterpart pin `test_json_tree_rooted_arborescence_uses_tree_data`
(`test_networkx_export_service.py:411-441`):

- Fixture: rooted tree `root → a → b`.
- Asserts envelope has `id` and lacks `nodes` (tree_data shape).
- Round-trips via `nx.tree_graph` → 3 nodes + 2 edges.

**Mental inversion verified**: I removed the `is_arborescence` check
in-process and re-ran tree_data on the M1 fixture — got 2 nodes / 1 edge
(silent truncation). The pin asserts 3 / 2, so removing the check breaks
CI. Symmetric: removing the TRUE branch (always node_link_data) breaks the
arborescence pin because the envelope no longer carries `id`. Pinned in
both directions.

### M2 — edge-list isolated-node drop: **DOCUMENTED + PINNED**

Three surfaces updated as promised:

1. **Module docstring** `networkx_export_service.py:52-66` `Format limitations`
   section names edge-list's isolate drop explicitly, points users at
   adjacency-list / graphml as alternatives, calls it "a NetworkX library
   convention, not an open-notebook limitation".

2. **UI caveat** `NetworkxExportMenu.tsx:104-108` adds
   `caveat: 'drops isolated entities'` on the edge-list `FormatOption`.
   L219-223 concatenates the caveat into `aria-label` for screen readers
   (`"${label} — ${description} ${caveat}"`); L237-244 renders an italic
   amber span (`text-amber-600 dark:text-amber-400`) below the description
   for sighted users with `data-testid` hook for future e2e specs.

3. **Pin test** `test_edge_list_documented_behavior_drops_isolated_nodes`
   (`test_networkx_export_service.py:444-490`):
   - 3 entities, single edge `a→b`; `c` is isolated.
   - Asserts service still counts 3 entities + 1 relation in report
     (accountancy unchanged).
   - Round-trips via `nx.read_edgelist` → 2 nodes + 1 edge.
   - Asserts `"entity:c" not in g.nodes` — the isolate vanished from the
     wire format.
   - Docstring cites NetworkX library behaviour as the cause.

**Mental inversion verified**: I confirmed in-process that
`nx.write_edgelist` on the fixture emits only `entity:a entity:b {}\n` and
`read_edgelist` reconstructs 2 nodes. If a future implementer swaps the
writer for one that preserves isolates (e.g. a custom adj-list-style
trailer), the round-trip would give 3 nodes and the pin would fail —
forcing an explicit decision to update docs + UI + test together.

## Minors

1. **`type_tags` flattening edge cases** — `networkx_export_service.py:68-76`
   documents `["A,B"]` ambiguity and `[]` → `[""]` edge case explicitly.
   Addressed.
2. **Pickle security caveat** — `networkx_export_service.py:64-66`
   docstring + `NetworkxExportMenu.tsx:121-124` UI caveat
   ("trust only files you generated yourself"). Addressed in both
   surfaces with consistent wording.
3. **`_FILENAME_UNSAFE_RE`** — explicitly deferred in self-review (track
   with B.2b). Acceptable.
4. **Plan path fix** — `plan.md:328` now points at
   `frontend/src/app/(dashboard)/notebooks/[id]/schema/page.tsx`. Diff
   from the prior `NotebookHeader.tsx` reference is explicit, calls out
   the surprise, and recommends D.1/D.2 follow the same location.
   Addressed.
5. **D.0 SurrealQL filter promotion** — deferred to D.2 landing. Acceptable.
6. **CORS `expose_headers`** — deferred to follow-up infra PR. Acceptable.

## Tests (independently verified)

- `apps/app-main/tests/test_networkx_export_service.py`: 22/22 passed
  (was 19 in attempt 1; +3 new: multi-rooted DAG pin, arborescence pin,
  edge-list isolate pin).
- `apps/app-main` regression: **536/536** passed (was 533; +3 from the
  new pin tests). No regressions.
- `packages/shared`: 199/199 passed (unchanged).
- `frontend tsc --noEmit`: clean.
- `frontend npm run lint`: no new warnings (4 pre-existing warnings in
  unrelated files: ExtractionTab.tsx + use-models.ts).
- Playwright `e2e/track-d/networkx-export.spec.ts`: 2/2 passed against
  a fresh Next dev server on `localhost:8515`. (The self-review's port
  collision note is real — ports 8502/8503/8512 are occupied by sibling
  worktrees on this host; reviewer used 8515.)

## Edge cases verified in-process

| Case | `is_arborescence` | Branch | Result |
|---|---|---|---|
| Empty graph | n/a (early return) | special-case `node_link_data` | `{}` envelope |
| Single node `only` | True | `tree_data` (trivial tree) | works |
| `a→b` | True | `tree_data` | works |
| `a→b→a` (cycle) | False | `node_link_data` | lossless |
| `a→c, b→c` (multi-root) | False | `node_link_data` | lossless |
| 2 isolates `x`, `y` (disconnected) | False | `node_link_data` | lossless |

Every shape behaves correctly. The self-review's claim that
"multi-rooted/cyclic/disconnected → `node_link_data`" matches code.

## Kudos

- The `is_arborescence` pre-check is the *right* fix — better than
  patching the symptom (e.g. counting nodes post-walk) because it
  detects the cause structurally before any data is touched.
- Counterpart pin test (rooted arborescence) is a particularly good
  defensive choice — stops a future implementer from "simplifying" by
  always routing through `node_link_data`. Pins behaviour in both
  directions.
- Module docstring `Format limitations` section reads like a release
  note for downstream consumers: each format that loses data names
  what + why + what to pick instead. Excellent user-facing doc density.
- UI caveat surfaces as italic amber text *AND* in `aria-label` —
  sighted and screen-reader users get parity. Most teams remember one
  surface, not both.
- INFO-level log on the json-tree fallback path is right: this is the
  common case for real notebook graphs (not an error), but having a
  log line means a user wondering "why did my JSON look different than
  I expected" can grep `journalctl` and see "graph is not an arborescence".
- Self-review is honest about the dev-server port collision and the
  rebase note ("clean fast-forward of the 3 D.3 commits"). The
  attempt-2 quality-gates table reproduces the exact commands and
  outcomes — reviewer could replay without spelunking.

## Decision rationale

Both attempt-1 majors fixed with the right granularity: M1 was a real
silent-loss bug → code fix; M2 was a library-level convention → docs +
pin (Option A, reviewer-recommended). All 6 minors either addressed or
explicitly deferred with a track. Mental inversion holds on both pins.
536/536 regression. 0 blockers + 0 majors → **APPROVED**.

## Next steps

1. Human approval / merge to main.
2. D.1/D.2 implementers: follow the schema-page wiring pattern (not
   NotebookHeader); promote `Entity.status` filter to SurrealQL per
   D.0 minor #1 when D.2 lands.
3. Track minors 3 + 6 (filename strip set, CORS `expose_headers`) as
   shared infra cleanups with B.2b.
