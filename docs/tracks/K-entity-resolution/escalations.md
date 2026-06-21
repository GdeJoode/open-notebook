# Track K — escalations

## K.1 — person/org normalization collision (2026-06-22, attempt 3)
**Status**: ESCALATED to user — design decision.

K.1 strips role-prefixes `minister van` (PERSON) and org-prefixes `ministerie van` (ORG) both to the same tail (`bzk`), collapsing a person and an organization onto one normalized key. The invariant "a normalized name is NOT a unique entity key once distinct typed entities collapse onto it" leaks layer by layer:
- entity dedup: fixed by keying `(name, type)` (rev2).
- relation endpoints: BROKEN by rev2 — `notebook_merge_service` rewrites a relation's endpoint via `name_to_canon[name]` (name-only → highest-confidence typed bucket), so a relation that pointed at the org re-attaches to the person. Relations are type-less by design (`WHERE canonical_name = $name LIMIT 1`, predates K).

**Options for the user:**
- **A (recommended)**: don't collapse person-roles onto org-tails. Remove `minister van` / `staatssecretaris van` (person-role prefixes) from K.1's strip list; keep org prefixes (`ministerie van`, `gemeente`, `provincie`). A person ("Minister van BZK") stays distinct from the org ("Ministerie van BZK"/"BZK") — semantically correct. Removes the collision at the source. The residual org/location cases (Gemeente Groningen vs the city) stay disambiguated by `(name,type)` at the entity level; the notebook-merge relation-rewrite weakness is pre-existing and tracked for a type-aware fix.
- **B**: carry entity type through relation endpoints end-to-end (relations reference typed endpoints). Fully general but invasive (data model + extraction + merge layers).
