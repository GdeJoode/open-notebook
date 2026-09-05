# Phase PC.3 — cross-document resolution

- **Branch**: `feature/track-pc3-cross-document`
- **Outcome**: the identity key ships; the headline feature does not
- **Date**: 2026-09-05

## What the phase set out to do, and what it found

PC.3 was planned on one premise: *cross-document resolution never runs, so every
document writes its entities fresh.* Stage 10 was off, nothing turned it on, and
three convenanten naming the same ministers produced 58 entities.

Both halves of that premise turned out to be wrong, in opposite directions.

**Consolidation already happened.** `upsert_entity` looks up on
`(canonical_name, entity_type)` and unions `source_documents`, so exact-name
matches merged across documents before this phase existed. **475 of 543 active
entities already spanned more than one source.** What did not consolidate was
VARIANTS — a far smaller population than the premise implied.

**And turning stage 10 on would not have fixed those.** Measured across the whole
active graph: 18 merges over 531 entities, of which roughly five are defensible.
Then the finding that settles it — `kg_entity_id`, `kg_match_type` and
`kg_similarity_score` have **no consumer anywhere in the production tree**.
`entity_persistence_service` holds zero occurrences of `kg_` and identifies the
entity by `name_key` exactly as it does with the stage off. Stage 10, enabled,
consolidates nothing.

So the phase's step 2 — "one field, and the phase's whole risk surface" — turned
on a stage whose primary output is dead. That is the failure PC.6 spent five
review rounds making unreachable, reintroduced by the phase after it.

## What ships

**The identity key, and it is the real deliverable.** `name_key` carries
`normalize_entity_name(canonical_name)`; migration 79 puts a UNIQUE index on
`(identity_key, entity_type)` where `identity_key` is the name key for ACTIVE
rows and the record id otherwise — so a retired row and a live row may share a
name without competing. Three production writers derive it through the one rule,
and a statement-level guard says so.

**Migration 79 refuses rather than merges.** The key cannot be computed in
SurrealQL, so the migration blocks until `scripts/backfill_name_key.py` has run,
and that tool reports colliding groups instead of picking winners. On `staging`
it reported 12 groups of 2 active rows; they were resolved through PC.2's own
curator path (`plan_merges` / `apply_merge`, which independently produced the same
12 groups), taking 543 active entities to 531.

**Migration 80 — an alias table nothing could write.** `entity_alias.match_type`
had two definitions: migration 78 declared it `option<string>` with
`IF NOT EXISTS`, which is a no-op wherever the field already exists, and `staging`
already had `TYPE string ASSERT $value INSIDE [...]` from no migration at all.
Measured, three of the four production writers could not write an alias on the
database that holds the data, two of them into a swallowed exception. All four now
succeed on both. `entity_alias` holds its first 12 rows.

**A candidate fetch that could not work.** `find_by_type` selected every row of a
type regardless of status. For `concept` the capped 100 contained **zero active
rows**, so a correct match was structurally impossible. It now filters to the live
set (`active`, `reference`) with a declared ordering.

**A refusal instead of a silent duplicate.** On a database below migration 79
`name_key` does not exist, so the lookup missed and every upsert fell through to
CREATE — a re-ingest would have doubled a document's entities. Found by running
against the config-default database, which sits at migration 31.
`_assert_identity_column` refuses once per process per database, naming both.

**Grounding survives persist**, merged by source id rather than overwritten, so a
curator can ask why two mentions are one entity and get an answer per source.

## What does not ship, and the condition for its return

Stage 10 goes back to off. Not because the stage is bad — because its answer has
nowhere to go, and because the answer is mostly the wrong SHAPE.

**Seven of the eight clear errors are correct RELATIONS recorded as identity.**
`coöperatief wonen` is narrower than `wonen`; `Versterken regionale samenwerking`
is an action on it; the Staatssecretaris is an organ of IenW; VROM is the
predecessor of VRO; `HG/BZK` is a joint construct; `ministeries van het Rijk: …`
is a list containing its members. Only the `VRO (Ministerie van VWS)` pairing is a
genuine extraction error.

The similarity signal is real. The destination is missing — and the repository had
already said so twice without acting: PC.2 filed the office-of class as "an
organ-of relation, a later phase", and stage 15's docstring states that
subsumption "is currently handled nowhere".

`test_kg_resolution_default.py` encodes the condition rather than a veto: the
config must STATE the choice explicitly either way, and when it is `True` the
guard demands that something outside the resolver actually reads `kg_entity_id`.

## What review should be hardest on

Three guards in this phase could not fail for their own case, and all three were
mine:

1. **The identity guard** asked whether a FILE mentions `normalize_entity_name`.
   Deleting `name_key = $name_key` from the real production CREATE left the import
   in place and the suite green. Verified by doing it; now statement-level.
2. **The candidate-fetch guard** asserted `"ORDER BY" in inspect.getsource(...)`,
   which the docstring's own prose satisfied with the clause deleted.
3. **The stage-10 destination check** searched for `kg_entity_id` as a substring
   and found two "readers": the comment explaining why the stage is off, and the
   measurement script. Both prose. Now AST-based, which cannot see comments, with
   docstrings dropped explicitly and a test that pins exactly that.

Three instances of one pattern — *a guard satisfied by commentary about the thing
it guards* — two of them after the pattern was named aloud in this same session.

## Carried forward, not fixed here

* **`upsert_entity`'s lookup has no status filter**, the same defect fixed in
  `find_by_type`. A new mention can update a row triage retired. Found the hard
  way: a probe in this session updated an archived row and I then deleted it —
  `entity:bf6v9jle08gg21evl0eq`, unreferenced, one archived row lost, recorded
  rather than reconstructed from guesswork.
* **75 historical merges have no alias row.** Their `merged_into` proves the
  merge; the alias was never written because of the migration-80 defect.
* **`open_notebook` sits at migration 31**, 49 behind. One integration test is red
  against it, and now says so in its error rather than reporting an index clash.
* **144 rows carry an empty `canonical_name`**, all `reference`.
* **Three producers of entity embeddings use two different texts**, compared by
  cosine.
* **The orphan invariant cannot see two shapes** — a key inside `properties`, and
  an exported class with no caller. Both orphans this phase found live there.

## Documents

* `phase-PC.3-plan.md` — the plan, with acceptance criteria rewritten on
  measurement after the originals proved unmeasurable (there are no active
  persons; the row-count target assumed no consolidation existed).
* `phase-PC.3-measurement.md` — sections A–F: the live graph, the incremental
  build, the threshold sweep, the TOOI authority, all five entity types, and the
  dead output.
* `pipeline-wiring.md` — the whole ingestion pipeline as it is actually wired,
  written after the phase kept finding the same class of defect one layer up.
* `scripts/pc3_resolution_measurement.py` — reproducible, read-only.
