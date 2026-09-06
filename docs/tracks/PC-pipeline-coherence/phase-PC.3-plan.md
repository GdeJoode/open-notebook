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
figures it holds — matched, new, by tier, and the cap counters from step 3. It
reaches `filtering_stats` the way N.5's counters do, so AC #2's count of what the
run consolidated has a producer instead of a promise.

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
cap-hit count, and **the state of the 20 active `government_organization` rows** —
which of AC #2's four groups collapsed, and a reason for each pair that did not.
That class is the measurement because it is where the variants this phase targets
actually live; the ministers the original plan named are not in the active graph at
all.

**This doubles as PC.6's first end-to-end test** — nothing in that phase was
verified against a corpus, and the reason given at the time (that there was none)
was wrong.

## Acceptance criteria

**Rewritten on measurement (2026-09-05), and the reason matters more than the
criteria.** The originals — "materially fewer than 117 rows" and "`M.C.G. Keijzer`
is one entity" — describe a graph this corpus does not have.

Two things measured on `staging` moved them:

* **Consolidation already happens.** `upsert_entity` looks up on
  `(canonical_name, entity_type)` and unions `source_documents`, so exact-name
  matches merge across documents today, when they hit.

  **46 of 531 active entities span more than one source — 8.7%, not the 87.5% an earlier draft of this document reported.** The figure 475 is the count over ALL 5,500 rows of every status (8.6% there too); presenting it against the 543 active ones inverted it. Corrected on re-measurement after review.
  
  And the inference built on it is withdrawn, not just the number. A spanning rate measures how much the SOURCES overlap, not whether consolidation works: ten of the fourteen documents are convenanten for different regions, so most topics genuinely appear once. Whether `upsert_entity`'s lookup HITS is not recorded anywhere, so neither figure settles the phase's original premise in either direction.
* **There are no active persons at all.** The active graph is 440 `topic`, 53
  `administrative_area`, 27 `programme`, 20 `government_organization`, 3
  `concept`. `M.C.G. Keijzer` exists as 3 `reference` and 2 `archived` rows and
  has never been active, so AC #2 as written could not be evaluated — not because
  the corpus is missing but because triage has retired that whole class.

Stating an unmeasurable criterion and reporting against it anyway is the failure
this track keeps finding in itself, so the criteria now name what this corpus can
actually answer.

1. **The identity key's effect, named — and it is small.** The backfill reports
   **12 collision groups over the 5,501 rows, each holding exactly 2 active rows**,
   so the identity key takes 543 active entities to 531. Eleven of the twelve are
   pure case variants of a `topic` (`brede welvaart`/`Brede welvaart`,
   `PROGRAMMALIJN 1: BETER LEREN`/`Programmalijn 1: Beter Leren`); exactly one is
   the article variant, `Regio`/`de Regio`, and it is an `administrative_area`.

   Naming that it is 12 and not hundreds is the point of the criterion. The plan
   was written expecting the key to be the main lever and it is not — 3,414
   distinct keys over 5,501 rows means the duplication that remains is
   overwhelmingly *not* case and spacing.

2. **The variant class the identity key does not touch — measured, not assumed.**
   Every one of the 20 active `government_organization` rows carries a DISTINCT
   `normalize_entity_name` key. **20 rows → 20 keys: the identity key changes
   nothing here at all.** The normaliser is doing real work on this class — it
   expands `IenW` to `infrastructuur en waterstaat`, `JenV` to `justitie en
   veiligheid`, and repairs the misspelled `Koninkrijkrelaties` to
   `koninkrijksrelaties` — and it still separates all twenty, because **13 of the
   20 names carry a parenthetical** and the parenthetical is part of the key:

   ```
   IenW                                    → infrastructuur en waterstaat
   IenW (Infrastructuur en Waterstaat)     → ienw (infrastructuur en waterstaat)
   Infrastructuur en Waterstaat (Ministerie) → infrastructuur en waterstaat (ministerie)
   ```

   Three surface forms of one ministry, three keys. `Ministerie van X` against
   `X (Ministerie)` against `ABBR (X)` is a structural difference, not a spelling
   one, so only stage 10's semantic tier can reach it.

   **The criterion is a named count against a named target.** Four groups here are
   the same organisation under different surface forms, and consolidating them
   takes 20 rows to 14:

   | group | rows | |
   |---|---|---|
   | BZK | `Binnenlandse Zaken en Koninkrijkrelaties (Ministerie)`, `BZK (Binnenlandse Zaken en Koninkrijkrelaties)` | 2 → 1 |
   | IenW | `IenW`, `IenW (Infrastructuur en Waterstaat)`, `Infrastructuur en Waterstaat (Ministerie)` | 3 → 1 |
   | VRO | `Volkshuisvesting en Ruimtelijke Ordening (Ministerie)`, `VRO (Volkshuisvesting en Ruimtelijke Ordening)`, `VRO (ministerie van Volkshuisvesting en Ruimtelijke Ordening)` | 3 → 1 |
   | HG/BZK | `Binnenlandse Zaken en Koninkrijkrelaties/Herstel Groningen`, `HG/BZK (Herstel Groningen/Binnenlandse Zaken en Koninkrijkrelaties)` | 2 → 1 |

   How many of the six the run removes is the number, and **every pair it leaves
   gets a named reason**. Six of six with reasons for none is not a better result
   than two of six with four reasons; the reasons are the deliverable.

3. **Nothing that must stay apart merges.** All six of these are live in this
   corpus, which makes them a regression test rather than a hypothetical. Each
   pairs with something in the table above and each must survive:

   * `De Staatssecretaris van Infrastructuur en Waterstaat (IenW)` — the office-of
     shape PC.2 cut 29 affixes to exclude, and it shares its parenthetical with the
     IenW group. An organ OF X is not X.
   * `Economische Zaken (Ministerie)` against `Ministerie van Economische Zaken en
     Klimaat` — EZ and EZK are different ministries however similar the strings.
   * `VRO (VROM, Volkshuisvesting, Ruimtelijke Ordening en Milieubeheer)` — VROM is
     the abolished predecessor, not a spelling of VRO.
   * `HG/BZK (…)` against BZK — a joint programme is not one of its participants.
   * `ministeries van het Rijk: Volkshuisvesting en Ruimtelijke Ordening` and
     `ministeries van het Rijk: Volkshuisvesting en Ruimtelijke Ordening,
     Binnenlandse Zaken en Koninkrijkrelaties/Herstel Groningen, Sociale Zaken en
     Werkgelegenheid, Justitie en Veiligheid` — a conjunction of four ministries
     captured as one entity. It must not merge into any of the four it names, and
     it must not merge into the shorter one either: a list of one and a list of
     four are not the same list.
   * `VRO (Ministerie van Volksgezondheid, Welzijn en Sport)` — see below.

4. **The candidate cap's bite is reported**, so a miss is distinguishable from a
   look that never happened.

5. **`normalize_entity_name`'s collision-safety claim is tested**, not inherited:
   a case that would collapse two distinct entities fails.

6. **The measurement runs against a separate database.** `staging` is the
   before-state — built with stage 10 off, because stage 10 has never run — so it
   is the control and stays untouched.

### Recorded, not fixed here

* **An extraction error, not a resolution one.** `VRO (Ministerie van
  Volksgezondheid, Welzijn en Sport)` is wrong on its face: VRO is
  Volkshuisvesting, not Volksgezondheid. Resolution must not "fix" it — merging it
  into either ministry propagates a false claim. It belongs to whatever phase owns
  extraction quality.
* **144 rows carry an empty `canonical_name`**, all `reference`. An entity with no
  name is meaningless and they all fold onto the empty key. That something writes
  them is a code defect, independent of what happens to these rows.

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
