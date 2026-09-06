# Phase PC.3 — attempts 1–5

- **Branch**: `feature/track-pc3-cross-document` (20 commits, merged as `6d81c453`)
- **Trajectory**: 3 blockers + 8 majors → 0 + 3 → 1 + 1 → 0 + 1 → 0 + 2
- **Outcome**: merged on the owner's decision; the last formal verdict was
  REVISIONS_NEEDED and its two majors were closed after it, unreviewed
- **Date**: 2026-09-05 / 2026-09-06

Five rounds, and what they were about is not what the phase was about. The phase
set out to enable cross-document resolution. It ended by proving the feature does
nothing, and the review rounds were almost entirely about the instruments used to
prove it.

## One defect, in eight costumes

Every blocker and most majors were the same thing: **checking something adjacent
to the claim.** The reviewer named it in round 1 and it kept appearing anyway.

| round | the claim | what was actually checked |
|---|---|---|
| 1 | "every writer derives `name_key` through the one rule" | whether the FILE mentions `normalize_entity_name` — deleting the column from the real CREATE left the import and the suite green |
| 1 | "the capped slice is ordered" | `"ORDER BY" in getsource(...)` — satisfied by the docstring's own prose |
| 1 | "stage 10's output has no reader" | `"kg_entity_id"` as a substring — matched the comment saying so, and the measurement script |
| 2 | "the guard covers the writers" | the params-dict KEY `{"name_key": …}` stood in for the SQL column |
| 2 | "475 of 543 active entities span >1 source" | 475 was the count over all 5,500 rows; the active rate is 46/531 |
| 2 | "seven of eight errors are relations" | one of the eight was a merge that never happened |
| 3 | "the lookup is keyed on identity" | `name_key` in the PROJECTION exempted a lookup keyed on the display column |
| 3 | "six defensible merges" | three were classified from a 40-character console line |
| 4 | "the file's ids are validated" | one file of the three the container ships |
| 5 | "the validator refuses" | that it was CALLED, and separately that a pattern EXISTS — never that calling it refuses |

The last one is the sharpest. Replacing `validate_record_id`'s body with
`return value`, leaving the pattern in the file and every call site intact, passed
27 tests and reopened every injection closed over two rounds. The adjacency had
moved from inside a test to *between two tests*.

## What made the guards work

Each became sound at the moment it stopped asking whether a token appears and
started asking what structurally happens:

- **is the column ASSIGNED** — read the `SET` clause, and `=` not `==`;
- **what is the lookup KEYED on** — read the `WHERE` clause, ignore the projection;
- **does calling it REFUSE** — call it, with the payload;
- **over which space** — the service directory, not the file the last finding was
  in; every write statement the repo contains, not the ones remembered.

That last line is the durable one. The track's existing rule is *derive the space,
do not sample it*. Five rounds show the rule has a second failure mode: deriving
correctly over a space drawn one boundary too small. A params key inside a call, a
name inside a module, a module inside a service, a test beside another test.

## The correction the phase owes its own plan

PC.3 was planned on "every document writes its entities fresh". That is not
something this corpus can confirm or deny — 8.7% of entities span more than one
source whether measured over the active graph or all rows, and ten of fourteen
documents are convenanten for different regions, so a low rate is what overlap
predicts. Whether `upsert_entity`'s lookup HITS is recorded nowhere.

An earlier draft of the report inverted that figure into 87.5% and used it to
overturn the plan's premise. Both the number and the inference were withdrawn.

## What review confirmed against running code

- migration 79's `?? "active"` coalesce, its computed `identity_key`, its refusal,
  and `79_down`, applied against a fresh container;
- migration 80's repair order — with the header corrected, because the mechanism
  it cited does not fire on this shape and the reviewer tested that;
- `("active", "reference")` as the repo's live set, from `audit_service`;
- `kg_entity_id` having no production consumer — the finding the whole
  recommendation rests on, verified independently;
- the staging audit: 24 rows with a post-2026-09-04 `updated_at`, exactly the 12
  documented merges and their winners, no relation rows touched.

## The security surface, and the reviewer's standing advice

The extraction service's injection surface entered scope on attempt 3, at the
owner's instruction, and produced a finding on every attempt afterwards: the
`where` family, then the sibling module, then the validator's own unproven
refusal plus the `UPSERT` statement form. Three families are now closed across two
modules behind a directory-scoped guard.

The reviewer's recommendation, recorded here because it was not acted on:

> PC.3's identity work has been merge-ready since attempt 3 and nothing in
> attempts 4 or 5 touched it. The extraction service's injection surface is a
> different piece of work and should be its own phase — it has produced a finding
> on every attempt since it entered scope, not because the fixes were wrong but
> because each was scoped to the boundary the last finding sat on.

He was explicit that this is not a quality problem in the work: the scope widened
one ring per round precisely because each correct finding was accepted and closed
properly. Still open there: `POST /extract`'s `file_path` with no root
confinement, and the app-wide auth posture — `notebooks`, `sources`,
`entity_resolution` and `knowledge_graph` use no auth dependency at all.

## Carried forward

- `upsert_entity`'s lookup has no status filter, the defect fixed in
  `find_by_type`. Found the hard way: a probe updated an archived row and it was
  then deleted — one unreferenced archived row lost, recorded rather than
  reconstructed from guesswork.
- 75 historical merges have no alias row; their `merged_into` proves the merge.
- `open_notebook` sits at migration 31, 49 behind. One integration test is red
  against it and now says why.
- 144 rows carry an empty `canonical_name`, all `reference`, all with
  `name_key = ''` — promoting a second one of a type to active now collides, and
  `set_status` swallows it.
- Three producers of entity embeddings use two different texts, compared by cosine.
- The orphan invariant cannot see a key inside `properties` or an exported class
  with no caller. Both orphans this phase found live there.
- `relation_source` / `relation_sources` and `incremental_report` — reassigned to
  PC.7, not closed here.
