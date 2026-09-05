# PC.3 — Look at the graph that is already there

*Plan. Measured 2026-09-05, against a repository at `bb715873` and the
`open_notebook/staging` database — which holds the corpus, contrary to what an
earlier draft of this plan said.*

## What the original plan said, and what measurement changes

The plan's finding holds: KG resolution is what matches a new mention against
existing entities, and it never runs. `KGResolutionConfig.enabled` is `False`, and
the `FilteringConfig` the app builds at `entity_extraction_service.py:1916` does
not set it. Three convenanten naming the same ministers produced 58 entities with
no consolidation.

**But the plan points at one lever and there are two, and the second is the one
that produces the case-variant rows.** `upsert_entity` looks up on
`(canonical_name, entity_type)` — the migration-39 UNIQUE index — and
`entity_persistence_service.py:634` sets `canonical_name=text`, the **raw surface
form**. So `Brede Welvaart` and `brede welvaart` are two rows whether or not stage
10 runs. Enabling resolution alone leaves every pair the resolver misses.

Three further corrections to what the phase can assume:

- **Tier 1 will never fire.** PC.2 set `register_aliases=False`, and
  `entity_alias` holds no rows, so alias lookup is dead until a curator makes an
  alias. Enabling KG resolution gives tiers 2 and 3 — fuzzy and semantic — and
  that is the coherent outcome of PC.2's decision, not a defect.
- **The cost is small but the cap is real.** Per entity: one
  `find_by_type(entity_type, limit=max_candidates)` plus in-memory Levenshtein and
  cosine over stored vectors. No API calls. But `max_candidates` bounds how much
  of the graph is even looked at, and on a corpus with hundreds of entities of one
  type that silently decides the answer — the same capped-sample problem
  `concept_alignment` documents for its own fetch.
- **The AC's numbers are NOT gone** — an earlier version of this plan said they
  were, on a zero-row reading taken while the SurrealDB container was up without
  its volume after a Docker restart. `open_notebook/staging` holds the corpus it
  always did: 14 sources, 3,824 chunks, 5,501 entities, 1,895 relations, 68 names
  containing "Regio Deal", ingested 20 June – 1 July.

  That changes the shape of the measurement for the better. **The graph as it
  stands IS the before-state** — built with stage 10 off, because stage 10 has
  never run. So the AC needs only the after: the same sources re-extracted with
  cross-document resolution on, into a SEPARATE database, leaving `staging`
  untouched as the control.

## Decided

**Cross-document resolution goes in the default path, with a measured cost.**
(user) — enable it in the config the app builds, and measure rows before/after,
wall time per document, and how often the candidate cap actually binds.

**The write key gains a separate normalised column.** (user) — `canonical_name`
stays the display form; a new `name_key` carries `normalize_entity_name(...)` and
takes the UNIQUE index. Case and spacing variants then never become two rows,
while the readable name survives for the curator and the exports.

`normalize_entity_name` is the right function and PC.2 says so explicitly: it is
*the identity rule*, distinct from `fold_for_comparison`, and it documents
collision-safety — *"Ministerie van Onderwijs and Onderwijs stay distinct"*. That
claim is load-bearing here and this phase must test it rather than inherit it.

## Steps

### 1. `name_key`, and a migration that refuses rather than merges

Migration 79 adds `name_key`, backfills it from the existing `canonical_name`, and
moves the UNIQUE index onto `(name_key, entity_type)`.

**The backfill will find collisions, and that is the point.** Two rows whose names
differ only in case fold to one key — which is exactly the duplication this phase
exists to remove — but *merging them is destructive and is a curator's decision*,
not a migration's. PC.2 shipped the door for it: `fold_equal`, same type, lands in
the auto-merge band. So the migration **detects collisions and refuses**, naming
the colliding pairs and pointing at the curator queue. That is PC.6's rule applied
to a migration: a step that cannot do its job says why rather than doing something
else.

**And it is not free on this database.** An earlier draft said `entity` holds
zero rows and the migration would apply unopposed. It holds 5,501, and running the
backfill against them is what found the design error in the index — see step 1's
outcome below. The refusal is not a hypothetical for a future operator; it fired
here first.

### 2. Turn stage 10 on where the app builds its config

`entity_extraction_service.py:1916`. One field, and the phase's whole risk surface.

### 3. Make the candidate cap visible

`max_candidates` silently bounds what the resolver considers. Report how often the
fetch returned exactly the cap — the shape `concept_alignment` already uses for
`capped_type_fetches` — so "it did not match" is distinguishable from "it never
looked".

### 4. Give `kg_resolution_report` its reader

PC.1b filed it as derived state with no consumer, and this phase's AC needs the
figure it holds. It reaches `filtering_stats` the way N.5's counters do, so the
"materially fewer rows" claim has a producer instead of a promise.

### 5. The inventory rows this phase owns

- `source_chunk_id` / `source_grounding` / `extraction_context` never persisted —
  they have live in-run consumers, and only the persist boundary drops them. They
  are the evidence a cross-document match needs to be explainable, which is
  exactly what this phase makes routine.
- `incremental_report`.
- `relation_source` / `relation_sources`, reassigned here by PC.2: relation
  provenance, not entity identity.

### 6. Measure it, against the corpus that is already there

`staging` IS the before-state: 14 sources, 3,824 chunks, 5,501 entities, built
with stage 10 off because stage 10 has never run. So the measurement is the after,
and it goes into a **separate database** — `staging` stays untouched as the
control rather than being overwritten by the thing it is the control for.

Record, for both: entity rows, distinct identity keys, wall time per source,
cap-hit count, and whether the ministers named across the three convenanten
resolve to one entity each. **This doubles as PC.6's first end-to-end test** —
nothing in that phase was verified against a corpus, and the reason given at the
time (that there was none) was wrong.

## Acceptance criteria

Restated. Not because the original figures are unreproducible — they are, and an
earlier draft wrongly said otherwise — but because the before-state already exists
and the comparison should use it rather than manufacture a second one:

1. Re-extracting the corpus with stage 10 on produces **materially fewer entity
   rows** than the same corpus without it, both figures named. The before-state
   already exists — `staging`, built with resolution off — so the after-run goes
   into a separate database and `staging` is the control rather than the casualty.
2. A person named in more than one of the three convenanten is **one entity**.
3. No two active entities differ only by case or spacing — enforced by the UNIQUE
   index, not by a sweep.
4. The candidate cap's bite is reported, so a miss is distinguishable from a
   look that never happened.
5. `normalize_entity_name`'s collision-safety claim is **tested**, not inherited:
   a case that would collapse two distinct entities fails.

## Risks

- **The backfill on a populated database.** Mitigated by refusing rather than
  merging, but it means an operator with duplicates must clear them through the
  curator queue before migrating. That is the right order and it should be said
  out loud in the migration header.
- **Resolution merging things that are not the same.** Tiers 2 and 3 are
  threshold-based, and this phase turns them on by default. The thresholds are
  the same ones PC.2's dedup door uses, and PC.2's measurements there — the
  `Regio Deal Groningen` / `Regio Deal Drenthe` pair at ≈0.83 — are the reference
  for what must NOT merge.
- **`normalize_entity_name` expands curated org aliases.** That is a stronger
  transform than the comparison fold, and putting it on the write key means it
  decides identity. Step 1's test is what makes that safe rather than assumed.

## Guard conventions this phase inherits

From PC.6, and they apply to every check written here:

1. Derive the space, do not sample it.
2. Verify a guard by putting it in the state it claims to prevent, and confirm it
   fails — *the* state, not one of that shape.
