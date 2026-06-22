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
**Status**: RESOLVED (2026-06-22, K-D2 follow-up) — the bulk source is found, wired, and live-validated. The original "open question" sub-items (1a/1b/1c) are answered below under **RESOLUTION**. Sub-item 2 (refresh *cadence/scheduler*) remains a separate operator choice (the manual endpoint + `last_validated` stamping are shipped; no cron wired).

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
- `tooi_provider.refresh()` source priority: operator file > remote fetcher > bundled seed (each fails soft into the next; unreachable registers → seed, never crash). Idempotent at scale (upsert on `(canonical_name, source_vocabulary)` + pre-load dedupe by `external_id`).
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
