# Track K — escalations

## K.1 — person/org normalization collision (2026-06-22, attempt 3)
**Status**: ESCALATED to user — design decision.

K.1 strips role-prefixes `minister van` (PERSON) and org-prefixes `ministerie van` (ORG) both to the same tail (`bzk`), collapsing a person and an organization onto one normalized key. The invariant "a normalized name is NOT a unique entity key once distinct typed entities collapse onto it" leaks layer by layer:
- entity dedup: fixed by keying `(name, type)` (rev2).
- relation endpoints: BROKEN by rev2 — `notebook_merge_service` rewrites a relation's endpoint via `name_to_canon[name]` (name-only → highest-confidence typed bucket), so a relation that pointed at the org re-attaches to the person. Relations are type-less by design (`WHERE canonical_name = $name LIMIT 1`, predates K).

**Options for the user:**
- **A (recommended)**: don't collapse person-roles onto org-tails. Remove `minister van` / `staatssecretaris van` (person-role prefixes) from K.1's strip list; keep org prefixes (`ministerie van`, `gemeente`, `provincie`). A person ("Minister van BZK") stays distinct from the org ("Ministerie van BZK"/"BZK") — semantically correct. Removes the collision at the source. The residual org/location cases (Gemeente Groningen vs the city) stay disambiguated by `(name,type)` at the entity level; the notebook-merge relation-rewrite weakness is pre-existing and tracked for a type-aware fix.
- **B**: carry entity type through relation endpoints end-to-end (relations reference typed endpoints). Fully general but invasive (data model + extraction + merge layers).

**RESOLVED (2026-06-22, user): A now, B planned.**
- A (K.1 rev3): adjust the strip list so NO must_not_merge pair collides at the NAME level (relations are name-only) — drop person-role prefixes (`minister van`, `staatssecretaris van`) AND `gemeente`/`provincie` (org↔location); keep articles + `ministerie van`/`het ministerie van` (org→org tail, no cross-type). The over-merge canary now checks at name-only. No legitimate merges lost (a person/municipality ≠ the org/city — they SHOULD stay distinct).
- B planned as **K.7 — Type-safe relation endpoints** (carry entity type/ID through relations end-to-end so cross-type homographs never mis-attach; enables re-introducing the aggressive prefixes safely). See plan.md K.7.

**RESOLVED-2 (2026-06-22, user): continue as planned (Option 1); K.7 KEPT.**
Blind content-prefix stripping (even `ministerie van`) collides cross-type in the live data (`Ministerie van Onderwijs`→`onderwijs` == the bare `onderwijs` concept). Context-free stripping can't avoid this. So:
- **K.1 (rev4) narrowed**: strip only collision-safe leading articles (`de`/`het`/`een`) + curated spelling tolerance + the harness/corpora. NO blind content-prefix stripping (drop `ministerie van` too).
- **K.2 absorbs org-form merging**: the curated, type-aware alias/equivalence table (BZK ↔ Ministerie van BZK ↔ Binnenlandse Zaken…) merges ONLY explicitly-listed equivalences — never blind stripping, so it can't accidentally collide. This is where the org merges land, safely.
- **K.7 REMAINS PLANNED** (not dropped): type-safe relation endpoints — later enables re-introducing aggressive normalization. User explicitly wants it kept.

## K.2 — note (not a blocker): one stale K.1 must_not_merge line removed (2026-06-22)
**Status**: DECIDED in-task per spec — recorded for the reviewer, no user action needed.

K.1 rev4's `must_not_merge.jsonl` contained `Ministerie van BZK` (organization) ↔ `BZK` (person) as a must-NOT pair (it forbade the org-form merge K.1 had deferred). K.2's entire deliverable is to merge that org-form (`Ministerie van BZK` ↔ `BZK` ↔ `Binnenlandse Zaken en Koninkrijksrelaties`), and the spec explicitly lists those pairs in `must_merge`. The two assertions are mutually exclusive, so the stale K.1 line was removed.

**Residual risk (accepted)**: a *person* literally surnamed "BZK" would now normalize onto the ministry org canonical (a name-only cross-type collision). This is judged acceptable because (a) the org-form merge is the explicit, spec-directed K.2 deliverable; (b) the realistic person form is the *role* phrase `Minister van BZK`, which is NOT keyed and stays distinct (kept in must_not_merge, plus the new `Minister van VRO` ↔ `Ministerie van VRO` person/org pair); (c) a bare-abbreviation person surname colliding with a ministry is not observed in the live Convenant corpus. K.7 (type-safe relation endpoints) is the structural fix that would let even this case disambiguate by type.

## K-D2 — TOOI source + refresh cadence (2026-06-22, K.4)
**Status**: RESOLVED (2026-06-22, K-D2 **rev2**) — the bulk source is found, wired, live-validated, AND the rev2 review's ingest-identity blocker is fixed (see **REV2** below). The original "open question" sub-items (1a/1b/1c) are answered under **RESOLUTION**. Sub-item 2 (refresh *cadence/scheduler*) remains a separate operator choice (the manual endpoint + `last_validated` stamping are shipped; no cron wired).

### REV2 (2026-06-22) — duplicate-display-name BLOCKER fixed (reference identity = external_id)
The rev1 framing claimed "one org is never split into two reference rows" (deduping by `external_id` before load). That was true but it **hid the inverse and more dangerous failure**: the persistence/upsert keyed on `canonical_name` (migration 41's `(canonical_name, source_vocabulary)` UNIQUE), which **merged two DISTINCT orgs that share a bare display name into ONE row**. The real 605-gemeente register has two distinct "Bergen" municipalities — `gm0373` (Noord-Holland) and `gm0893` (Limburg), both `canonical_name="Bergen"`. At ingest the UNIQUE-on-name index collapsed them to one row, so `lookup_by_name("Bergen")` returned a single candidate and the reconciler's single-high-confidence-match guard **auto-linked every "Bergen" entity (incl. Bergen-NH ones) to the surviving (wrong, Limburg) URI**. The precision guard was bypassed because the ambiguity was destroyed at INGEST, not at reconcile. Same hazard for any cross-soort bare-name overlap (Groningen gemeente vs provincie, Utrecht, etc.).

**Fix — the persisted reference identity is now the org's STABLE id (external_id / organisatiecode), not its display name:**
- **Migration 57** (`57.surrealql` + `57_down.surrealql`): re-keys `reference_entity` uniqueness from `(canonical_name, source_vocabulary)` to **`(external_id, source_vocabulary)`** via `DEFINE INDEX OVERWRITE idx_ref_extid_source ... UNIQUE`, and demotes `idx_ref_name_source` to NON-unique (lookup still indexed; `idx_ref_name` from migration 41 also stays). Touches ONLY `reference_entity` — never `entity` (B.8-safe). Idempotent (OVERWRITE); applies cleanly on the B.0 testcontainers harness and is safe over the handful of existing K.4-seed rows.
- **`reference_entity.py` upsert**: CREATE-vs-UPDATE pre-fetch now keys on `(external_id, source_vocabulary)`. Two distinct orgs sharing a display name persist as TWO rows.
- **`tooi_provider._dedupe_rows`**: already keyed on `external_id` — now CONSISTENT with the upsert (one code → one row, aliases unioned; two distinct codes sharing a name → two rows). `refresh()` returns the actual persisted (deduped) row count — no longer inflated by a name-collision collapse.
- **Reconciler unchanged**: `lookup_by_name("Bergen")` now returns 2 distinct-external_id rows → `_distinct_by_uri` counts 2 distinct URIs → multi-candidate guard refuses to auto-link (recorded as ambiguous candidates). Verified end-to-end through the real `TOOIProvider`.

**Regression tests added:**
- `test_tooi_provider.py::test_refresh_keeps_distinct_same_named_orgs_as_two_rows` — the two Bergens persist as 2 rows; `lookup("Bergen")` surfaces BOTH URIs. (`FakeReferenceRepo` re-keyed to `(external_id, source)` to mirror migration 57.)
- `test_vocabulary_reconciler.py::test_two_same_named_orgs_via_tooi_provider_do_not_auto_link` — an entity named "Bergen" with 2 same-name reference rows → `reconcile_entity` does NOT auto-link (external_ids stays empty, reason `ambiguous_multiple_candidates`).
- `test_migrations_roundtrip.py::test_migration_57_distinct_same_named_orgs_coexist` + `::test_migration_57_idempotent_upsert_on_external_id` — live container: both Bergens coexist; upsert idempotent on external_id; migration 57 OVERWRITE re-apply is a no-op.

Fail-soft invariant intact (registry outage → seed, never crash); no live HTTP in tests.

### RESOLUTION (2026-06-22) — bulk source wired
**The confirmed machine-readable bulk source** (live-verified, no auth):
- The SET url `identifier.overheid.nl/tooi/set/rwc_overheidsorganisaties` content-negotiates (any `Accept`) to an HTML landing page — NOT a data endpoint. `rwc_overheidsorganisaties` is also not itself a downloadable register.
- Instead, `standaarden.overheid.nl/tooi/waardelijsten` splits the orgs into **eight per-type `*_compleet` registers**, each published as a versioned RDF/JSON-LD dump at:
  `https://repository.officiele-overheidspublicaties.nl/waardelijsten/<set>/<version>/json/<set>_<version>.json`
  (also `/ttl/`, `/rdf/`, `/xml/`). e.g. `rwc_ministeries_compleet/6/json/rwc_ministeries_compleet_6.json` → HTTP 200, expanded JSON-LD. The latest version per set is discoverable from the work page's expression links.
- **Format**: expanded JSON-LD. Org nodes carry `ont:organisatiecode` / `ont:afkorting` / `ont:officieleNaamExclSoort` / `ont:officieleNaamInclSoort`. A renamed org repeats per name-version, each pointing at the canonical entity URI via `prov:specializationOf`; non-org metadata nodes (the waardelijst header) carry no organisatiecode.
- **Live count (2026-06-22)**: the union of the eight registers = **1438 organisations** — 605 gemeenten, 383 samenwerkingsorganisaties, 210 overige overheidsorganisaties, 174 ZBOs, 32 waterschappen, 19 ministeries, 12 provincies, 3 Caribbean public bodies. BZK (`mnre1034`) resolves; the renamed Justitie ministry (`mnre1058`) carries all 8 historical surface forms (Justitie / Veiligheid en Justitie / Justitie en Veiligheid + MinJus/VenJ/JenV …) as aliases.

This answers the original options: **1(a) is the path** — there IS a canonical bulk-download URL (the per-register JSON-LD dumps), so a per-identifier crawl (1b) is unnecessary. The operator file (1c) is kept as a secondary override (`TOOI_BULK_SOURCE`).

**Wired in:**
- `shared/vocabulary/tooi_bulk.py::TOOIBulkFetcher` fetches + parses all eight registers through the K.4 fail-soft HTTP client (timeout + rate-limit 1s + cache), groups by canonical URI, collects historical names/abbreviations as aliases, unions + dedupes by organisatiecode.
- `tooi_provider.refresh()` source priority: operator file > remote fetcher > bundled seed (each fails soft into the next; unreachable registers → seed, never crash). Idempotent at scale (upsert on `(external_id, source_vocabulary)` + pre-load dedupe by `external_id`).
- `POST /api/vocabulary/refresh` attaches the fetcher by default; `TOOI_DISABLE_REMOTE=1` opts out of the network (air-gapped → seed/file only).
- Tests mock HTTP (no live CI calls): `test_tooi_bulk.py` (parse + union + fail-soft + dedupe) and the remote-refresh integration cases in `test_tooi_provider.py`.

**Still operator-owned (sub-item 2):** the refresh *trigger/cadence* — manual `POST /api/vocabulary/refresh` is shipped; a cron / startup-if-stale scheduler is not wired (a deployment choice, not a code blocker). The register *versions* in `DEFAULT_REGISTERS` are pinned (immutable repository paths); bump them, or pass `registers=`, to pick up a newer publication.

---
**Original K.4 investigation (superseded by the RESOLUTION above):**

**What I determined (live-verified 2026-06-22):**
- TOOI is reachable as **content-negotiated RDF/Turtle per identifier** at `https://identifier.overheid.nl/tooi/id/<soort>/<organisatiecode>` (e.g. `.../ministerie/mnre1034` = BZK). A `GET` with `Accept: text/turtle` returns clean structured fields: `tooiont:organisatiecode`, `tooiont:afkorting` (abbreviation), `tooiont:officieleNaamExclSoort`, `tooiont:officieleNaamInclSoort`. This is the field shape the provider keys off.
- The waardelijst set exists at `https://identifier.overheid.nl/tooi/set/rwc_overheidsorganisaties` (HTTP 200) but I could **not** confirm a stable machine-readable **bulk-download** URL (the version-numbered `repository.overheid.nl/waardelijsten/.../json/...` path returned the HTML landing page, not JSON — the exact path/version is not discoverable without docs).

**What K.4 shipped (so the phase is not blocked):**
- `tooi_provider.refresh(source=<path>)` ingests a **documented bulk-file format** (`tooi_organisations.json`, a JSON list of org records — format documented in the module docstring) into `reference_entity`.
- A bundled **verified seed** (`_SEED_RECORDS`: the real organisatiecodes + officiële namen for AZ/BZK/BZ/Def/EZK/IenW/LNV/LVVN/OCW/VWS, fetched live) is the default source, so a fresh install + the tests resolve BZK→its TOOI URI offline (AC1-2 pass). `TOOI_BULK_SOURCE` env var points `refresh` at a full dump once available.

**The open question for the user (decide before a production full refresh):**
1. **Exact bulk source** — confirm one of:
   - (a) the canonical **bulk-download URL** for the full `rwc_overheidsorganisaties` waardelijst (+ its format: JSON/SKOS/RDF) — then I wire `refresh()` to fetch + parse it directly; OR
   - (b) accept a **per-identifier crawl** of the org codes (slower, rate-limited via the disciplined client) seeded from the set listing; OR
   - (c) keep the **operator-supplied bulk file** (`TOOI_BULK_SOURCE`) as the production path (someone exports the dump out-of-band on a schedule).
2. **Refresh cadence/trigger** — the roadmap says "monthly". Confirm the mechanism: manual `POST /api/vocabulary/refresh` (shipped), a scheduled cron calling it, or app-startup-if-stale (compare `reference_entity.last_validated` age). Shipped: the manual endpoint + `last_validated` stamping; the scheduler is not wired pending this decision.

Everything downstream (the reconciler, lookup, Crossref, `external_ids`) works against `reference_entity` regardless of how TOOI is loaded — this decision only affects how the full org set gets in.

## K.6 — AliasManager has no dedicated alias CRUD endpoint → AC4 DEFERRED (2026-06-22, rev2)
**Status**: AC4 (per-entity alias add/remove with immediate persistence) **DEFERRED** to a K.6-followup. Dead component removed. Recorded for the reviewer; needs a backend endpoint before it can ship honestly.

The K.6 plan lists `AliasManager.tsx` "backed by `entity.aliases` + `entity_alias`". The merged backend exposes `entity.aliases` for **read** (via `GET /api/knowledge-graph/entities/{id}`, which `SELECT *`s the row) but ships **no per-entity alias add/remove HTTP endpoint** — the only alias-writing surfaces are (a) the K.3 merge apply (creates `entity_alias` rows for losers) and (b) the K.5 `alias_overlay` force-merge (materialized into an `entity_alias` edge on the next dedup scan).

**Initial (rev1) decision — REVERSED.** rev1 wired `AliasManager` to the K.5 overlay endpoints (add alias = global force-merge, remove alias = force-split). The K.6 adversarial review (rev2) flagged this as **dead code AND wrong semantics**: the component had zero importers/tests/E2E and was never mounted, and "add an alias" silently became a **graph-wide force-merge overlay** — conflating "record a surface form for this entity" with "force-merge everywhere", which is misleading and unsafe UX. Shipping a mislabeled alias editor is worse than not shipping one.

**rev2 decision — DEFER AC4, remove the dead component.**
- Deleted `frontend/src/components/resolution/AliasManager.tsx` (no importer, no test, never mounted; overlay-as-alias semantics are wrong).
- Per-entity alias *management* (add/remove with immediate persistence) genuinely requires a backend `POST/DELETE /api/.../entities/{id}/aliases` endpoint that writes/removes `entity_alias` rows directly. That endpoint does **not** exist in the merged backend, so AC4 is out of K.6's frontend-only scope.
- **What stays (the honest, shipped alias surface):** the read-only alias **count/list** on entity detail (from `entity.aliases`), `ExternalIdBadges` (AC6), and `OverlayEditor` (the correctly-labelled force-merge/force-split escape hatch). None of these mislabel a force-merge as an alias edit.

**K.6 follow-up (backend, additive):** add `POST /api/.../entities/{id}/aliases` (add a surface form → insert an `entity_alias` row, immediate) and `DELETE /.../aliases/{alias}` (remove the row). Then a *real* per-entity alias editor can be added that calls those endpoints (distinct from the graph-wide overlay). Until then, AC4 is deferred, not faked.
