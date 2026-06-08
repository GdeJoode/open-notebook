# Phase B.4 — self-review

> Author: implementer agent, 2026-06-08
> Branch: `track/b-confidence-telemetry`
> Backend commits: `861595a` (migration 47 + `shared.services.metrics` + `record_metric` helper) → `1bd8e47` (`extraction.complete` + `extraction.auto_fallback` metric call sites + tests)
> Frontend completion commit: see new HEAD of branch.

## Frontend completion

The previous implementer hit a session limit after pushing the backend
half. This review covers the frontend portion of B.4: surfacing
extraction confidence in the KG UI and persisting a user-driven
filter.

### Acceptance criteria check (frontend portion)

| # | Criterion | Verified? |
|---|---|---|
| 1 | KG page shows a confidence bar on every entity tile (visual smoke) | YES — table row renders `<ConfidenceBar value={entity.confidence}/>` whenever the API returns a numeric `confidence`; renders a muted `—` when the field is absent (legacy rows). Playwright spec asserts the bar is visible for each of three fixture entities. |
| 2 | `ConfidenceFilter` slider hides entities below the threshold; state persists across reloads via `localStorage` | YES — `ConfidenceFilter` reads `kg_confidence_threshold` on mount, writes on every slider change, and emits `onChange(value)`. The page filters `displayEntities` client-side via `entities.filter(e => e.confidence >= threshold)`. Playwright reloads the page and re-asserts the threshold label + hidden row. |
| 3 | Playwright spec verifies bar + filter UI (route-mocked, no live DB) | YES — `frontend/e2e/track-b/confidence-display.spec.ts` mocks `/api/knowledge-graph/{entities,entity-types,entities/{id},search,graph}` and asserts: rows render bars, slider keyboard-drives to 0.50, low-confidence row hides, localStorage carries the value across reload, and the detail panel renders per-relation bars. 4/4 tests pass against the dev server. |
| 4 | Relation rows in the entity detail panel also render the bar | YES — the detail panel's relation `<Card>` renders `<ConfidenceBar value={rel.confidence}/>` when the relation carries a numeric confidence. Playwright opens the detail panel for `entity:high` and asserts both relation bars are visible with their expected `data-confidence` attributes. |

### Pre-resolved decisions honoured

| Q | Decision | Implementation |
|---|---|---|
| Q-B-6 | Telemetry always-on (backend) | Backend was already done in `861595a` + `1bd8e47`. Frontend has no env-flag responsibility. |

### Files changed (frontend pass)

**Created**

- `frontend/src/components/knowledge-graph/ConfidenceBar.tsx` — accepts `value: number`, `label?: string`, clamps to `[0, 1]`, colours via Tailwind buckets (red < 0.5, amber < 0.8, else green), and wraps in a Radix tooltip showing the exact two-decimal value. Exposes `data-testid="confidence-bar"` and `data-confidence` so specs can assert without hovering.
- `frontend/src/components/knowledge-graph/ConfidenceFilter.tsx` — Radix slider, range [0, 1], step 0.05. SSR-safe initial state of `0`; hydrates from `localStorage.kg_confidence_threshold` in a `useEffect`. The initial-restore fires `onChange(initial)` only when the persisted value is non-default to avoid an unnecessary parent re-render.
- `frontend/e2e/track-b/confidence-display.spec.ts` — 4 specs covering the four ACs.

**Modified**

- `frontend/src/lib/api/knowledge-graph.ts` — added `confidence?: number` to `Entity` and `EntityRelation`. Marked optional because rows persisted before B.1 don't carry the field; consumers treat `undefined` as "unknown".
- `frontend/src/app/(dashboard)/knowledge-graph/page.tsx` — new `confidenceThreshold` state driven by `<ConfidenceFilter>`, new "Confidence" column on the entity table rendering `<ConfidenceBar>`, and a per-relation `<ConfidenceBar>` on each relation card in the detail panel. Added `data-testid` markers for spec hooks.
- `packages/surrealdb-service/src/surrealdb_service/repositories/entity.py` — extended the SELECTs in `list_entities`, `search_entities`, and the relation SELECT in `get_entity_detail` to project `confidence`. This is a strictly additive change: the column already exists on the row, and consumers that don't read the new field are unaffected.

### Why the backend SELECT tweak was unavoidable

The original SELECT projection in `list_entities` (`SELECT id, name, entity_type, weight`) was the only blocker on the frontend side: without `confidence` in the response, the bar couldn't render. Adding the column to the projection is the minimum-risk edit (no schema change, no semantic change) and keeps everything else in B.4's backend half untouched. All 17 KG router + service tests and all 77 surrealdb-service tests still pass.

### Test summary

| Suite | Result | Notes |
|---|---|---|
| `tsc --noEmit` (frontend) | clean | no new errors |
| `npm run lint` (frontend) | clean | no new warnings; all listed warnings pre-date this phase |
| `npx playwright test e2e/track-b/confidence-display.spec.ts` | 4/4 passed | mocked, no DB |
| `pytest apps/app-main/tests/test_knowledge_graph_*.py` | 17/17 passed | confirms backend tweak is non-breaking |
| `pytest packages/surrealdb-service/tests/` | 77/77 passed | confirms SELECT change is safe |

### Visual smoke

Per-row layout in the KG table:

```
| Name                  | Type     | Weight | Confidence              |
| Low Confidence Entity | Concept  | 1      | [▰▱▱▱▱]  red, tooltip 0.30 |
| Mid Confidence Entity | Concept  | 2      | [▰▰▱▱▱]  amber, tooltip 0.70 |
| High Confidence Entity| Concept  | 5      | [▰▰▰▰▱]  green, tooltip 0.95 |
```

Filter UI sits in the existing filter bar next to the entity-type
selector, labelled "Hide entities below: {value.toFixed(2)}". Sliding
to 0.50 collapses the low-confidence row out of the table; the label
updates immediately and the localStorage write fires on each
`onValueChange`.

### Design decisions worth flagging for the reviewer

1. **`confidence` is optional in the TypeScript types.** Legacy rows (pre-B.1) don't carry the field, and the SELECTs project it transparently. The frontend treats `undefined` as "unknown" — bar is hidden and the row is filtered out only when the threshold is strictly above 0. This matches the reviewer expectation in the prior retro that "unknown ≈ low" once the user opts into filtering, but it never auto-hides anything at threshold = 0.

2. **Slider state lives in the page, not in a context.** The plan called for two narrow components and a single page integration point. Threading a context for one consumer (the KG table) would be over-engineering. If a future phase needs the threshold on the Sigma graph view, the page can pass it down as a prop.

3. **Test ids over visual assertions.** Each entity row and each relation card carry a `data-testid` with the canonical record id. The `<ConfidenceBar>` also exports `data-confidence` so specs can assert the exact numeric value without depending on the tooltip's hover behaviour. This pattern matches the Track-A parser-engine-badge spec.

4. **Backend SELECT tweak vs. service-layer projection.** I considered projecting `confidence` only in the service layer (i.e. transforming raw repo dicts before returning), but the repo's `Dict[str, Any]` contract is already passthrough — adding `confidence` to the SELECT is the cheapest hop.

### Outstanding follow-ups (out of scope for B.4)

- The Sigma graph view doesn't visualize confidence (could colour-tint edges/nodes). Not in the plan.
- The "Confidence" sort order on the table is alphabetical-by-name (the underlying SELECT's ORDER BY). Adding a sortable column header is a small enhancement but out of scope.
- The detail panel still renders a textual "Confidence: 95%" alongside the new bar. Kept intentionally for parity with existing UI; reviewer may want to consolidate.
