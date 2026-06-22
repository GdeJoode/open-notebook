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
