# Phase B.3a — Self-review (attempt 2)

**Date**: 2026-06-08
**Author**: implementer (track B, B.3a attempt 2)
**Branch**: `track/b-schema-tab-view`
**Reviewer feedback resolved**: `phase-B.3a-attempt-1.md` (1 blocker, 1 major, 5 minors)

---

## Scope

This phase ships the **view-only** Schema tab on the notebook detail page:

- A flat `role="listbox"` entity-type browser (base ontology + accepted extensions).
- A per-source coverage table populated from `GET /api/notebooks/{id}/pass1_results`.
- A pending-extensions panel with disabled `Accept` / `Reject` buttons (mutations land in B.3b).
- A TTL download button that drives `GET /api/notebooks/{id}/schema.ttl` through the bearer-token-aware `apiClient`.
- Two new backend endpoints: `GET /api/notebooks/{id}/schema` (JSON) and `GET /api/notebooks/{id}/pass1_results`.

Attempt-1 shipped all of the above but the Playwright spec failed 4/4 because the route mocks didn't match the actual request URLs.

---

## Attempt-2 changes

### Blocker — Playwright spec mock URLs

**Root cause**. `notebooksApi.get` interpolates the raw id into the URL (`/notebooks/${id}`), so the request URL carries a literal colon (`notebook:b3a-fix`). `notebookSchemaApi.get` and the TTL download both call `encodeURIComponent`, so their URLs contain `notebook%3Ab3a-fix`. Playwright glob route patterns are literal — `:` does NOT match `%3A`. The attempt-1 spec used `**/api/notebooks/${encodeURIComponent(NOTEBOOK_ID)}/...` everywhere, so the **notebook detail** mock never fired (request had a literal colon) → 404 → the page rendered "Notebook Not Found".

**Fix**. Switched all four `page.route` calls to RegExp patterns with an alternation that accepts both spellings:

```ts
const NB_ID_ENCODED_RE = '(?:notebook:b3a-fix|notebook%3Ab3a-fix)'

await page.route(
  new RegExp(`/api/notebooks/${NB_ID_ENCODED_RE}/schema(?:\\?.*)?$`),
  ...
)
```

Rejected the prompt's suggestion `**/api/notebooks/${NOTEBOOK_ID}` alone because it would still miss the schema/pass1/TTL mocks (those URLs are encoded). The regex approach is robust against either client deciding to flip its encoding strategy in the future.

**Verification**.

```
$ npx playwright test e2e/track-b/schema-tab-view.spec.ts --reporter=line
Running 4 tests using 3 workers
[1/4] schema page renders tree, coverage table, and pending extensions
[2/4] selecting a tree item updates the side panel via keyboard
[3/4] TTL download button triggers a browser download
[4/4] Schema link on NotebookHeader is highlighted as active
  4 passed (6.1s)
```

### Major — listbox vs tree

**Decision**. Took **option B** from the reviewer prompt: accepted the flat-listbox pivot, softened plan AC #2/#6, documented the design decision in the SchemaBrowser docstring.

**Reasoning**.

1. **Shallow hierarchy in practice**. The in-scope base ontologies (scholarly, general) use a single `parent_type` level at most — no nested type-of-type-of-type. A flat list with "extends X" on the side panel conveys the same info without making the user expand nodes.
2. **Listbox is the correct ARIA semantic for single-select**. `role="tree"` carries `aria-expanded` semantics that imply child nodes; using it for a flat list mis-signals expandability to screen readers.
3. **No B.3b dependency**. Rename/merge/split edit ops operate on individual types, not on tree edges — they consume the same flat `items[]` list.
4. **Reviewer explicitly preferred option B** in the prompt and ranked the case for it (lower complexity, no UX win from real tree).

**Implementation tightening**. Added `aria-activedescendant` on the `<ul role="listbox">` pointing to the selected option's `id` — required for AT consistency since some screen readers don't read `aria-selected` reliably on a `<button role="option">`. The id is derived via `useId` so multiple SchemaBrowser instances on a page wouldn't collide.

### Minor 1 — dead `notebookSchemaApi.getTtlUrl`

Deleted. Replaced with a `NOTE:` block explaining why future re-introduction needs a paired auth review (a raw-URL helper would silently side-step the `apiClient` bearer-token interceptor).

### Minor 2 — lifted repo factories to `app_main.dependencies`

`get_notebook_schema_repo` and `get_pass1_result_repo` moved from the schemas router into the central dependencies module. B.3b's edit-ops endpoints and B.3c's soft-nudge toggle will both import these — the central location matches the workspace convention (`get_<service>()` factories live in `dependencies.py`). Tests updated accordingly (`from app_main.dependencies import ...`).

### Minor 3 — `_normalise_extension` `"string"` default comment

Added an inline comment explaining that `"string"` is the safe TTL-compatible fallback (`xsd:string` round-trips losslessly) and that the frontend treats unknown data-types as opaque labels. B.3b's editor surfaces the value so users can correct mis-inference.

### Minor 4 — extension/base type-name collision

Deferred per reviewer's note ("B.3b scope").

### Minor 5 — focus ring on disabled-button wrapper

Added `focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2` to the `<span tabIndex={0}>` wrapping each disabled `Accept` / `Reject` button. Without it, Tab onto the wrapper looked like focus disappeared (the disabled button itself can't receive focus, so the span is the de-facto focus target).

---

## Quality gates

| Gate | Result |
|---|---|
| `npx playwright test e2e/track-b/schema-tab-view.spec.ts --reporter=line` | **4 passed (6.1s)** |
| `npx tsc --noEmit` | clean |
| `npm run lint` | clean for changed files (pre-existing warnings unchanged in other source files) |
| `pytest apps/app-main/tests/test_schemas_router.py -v` | **21 passed** |
| `pytest apps/app-main -q` (full suite) | **410 passed** (no regressions vs 410-baseline) |

The schemas router suite covers the new dependency-override import path (now from `app_main.dependencies` instead of the router module).

---

## Files touched in attempt 2

- `frontend/e2e/track-b/schema-tab-view.spec.ts` — RegExp route patterns (blocker fix).
- `frontend/src/lib/api/notebook-schema.ts` — deleted dead `getTtlUrl`, added explanatory NOTE.
- `frontend/src/components/notebooks/schema/SchemaBrowser.tsx` — docstring expanded with listbox-vs-tree decision; added `useId`-based `aria-activedescendant` + option ids.
- `frontend/src/components/notebooks/schema/PendingExtensionsPanel.tsx` — focus-visible ring on the disabled-button wrapper span.
- `apps/app-main/src/app_main/dependencies.py` — added `get_notebook_schema_repo`, `get_pass1_result_repo`.
- `apps/app-main/src/app_main/api/routers/schemas.py` — removed local factories, import from dependencies, inline comment on `data_type` default.
- `apps/app-main/tests/test_schemas_router.py` — repointed factory imports.
- `docs/tracks/B-kg-quality/plan.md` — softened AC #2 and #6 wording.
- `docs/tracks/B-kg-quality/status.md` — phase summary entry.

---

## Open questions / handoff notes

- **Q**: Should B.3b validate extension/base type-name collisions at the API boundary or rely on the SchemaPatch contract? **A**: B.3b's reviewer call. The flagged minor 4 from this review is still open.
- **The Playwright spec depends on `mockDashboardChrome` from track-A helpers** — keep that import in sync if track-A reorganises.
- **`aria-activedescendant`** points to the *selected* option (single-select listbox), not a separately-tracked "focused" option. We don't yet manage focus via JS keyboard handlers (we rely on native Tab between `<button>` rows). If B.3b/B.3c adds arrow-key navigation, that hook should also update `aria-activedescendant`.

Ready for adversarial review.
