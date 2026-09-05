# Phase PC.2 — attempts 1–3

- **Branch**: `feature/track-pc2-one-identity`
- **Reviewed commits**: `58eddd5d`…`c7e7ade6` (approved), plus `54173a7b` (the
  five approved-past minors, taken anyway)
- **Outcome**: REVISIONS_NEEDED → REVISIONS_NEEDED → **APPROVED**
- **Date**: 2026-09-04

The phase that gives the graph one comparison fold, one alias policy, and a
curator path for the pairs the similarity tiers structurally cannot see. Three
rounds, and what the rounds were actually about is not what the phase was about.

> **Correction (2026-09-05).** This document said the working corpus had been
> emptied and its figures could no longer be re-measured. That was wrong. The
> `open_notebook/staging` database holds the corpus it always did — 14 sources,
> 3,824 chunks, 5,501 entities, 68 of them naming a Regio Deal, ingested
> 20 June – 1 July. The zero-row reading behind the claim was taken in the window
> after Docker Desktop restarted and the SurrealDB container came back without
> its volume; a single measurement on a just-restarted stack was treated as a
> fact about the data. Every figure below is therefore re-measurable, and the
> graph as it stands is the before-state for PC.3's acceptance criterion —
> the corpus built WITHOUT cross-document resolution.
## The one pattern, three times

Every blocker and two of the three round-2 majors were the same defect wearing
different clothes: **a value produced for a surface that never consumes it.**

| round | instance |
|---|---|
| 1 (blocker) | `candidateTypeLabel` / `isCrossTypeCandidate` written, tested, imported by the card — called nowhere. A cross-type candidate rendered `Regio Deal ↔ Regio Deal · programme`, one click from a destructive apply with the only distinguishing fact hidden. |
| 2 (major) | `head_affix` returns the removed head run instead of a bool, with a docstring saying why — *"a card that says 'differs by the head run `gemeente`' is reviewable, one that says 'containment' is not"*. The caller discarded it. |
| 2 (major) | `CandidateMergeCard.test.ts` beside `CandidateMergeCard.test.tsx` made TypeScript drop the `.tsx` by extension-priority dedupe. `tsc --noEmit` reported clean while never looking at the file guarding the round-1 blocker — and it was hiding a real `TS2741`. |

PC.1b built an invariant against exactly this — *a producer must name its
consumer* — and it is Python-only. All three instances sat on the
Python/TypeScript boundary, where nothing enforces it. That is the durable
finding, and a cross-boundary guard is a track-level instrument rather than
something to bolt onto this phase. Round 3's escalation sweep confirmed no fourth
instance: 40 fields across every `entity-resolution` response model against every
identifier in `frontend/src`, **0 orphaned by name**.

## What review disproved, not just corrected

Two findings changed what I believed, rather than what I had typed.

**Migration 78 rested on a claim this repository had already disproved twice.** My
first version was a bare `DEFINE FIELD ... TYPE bool DEFAULT false`, with a header
asserting that the DEFAULT would make pre-existing rows read as unverified. A
SurrealDB DEFAULT applies only to newly created records; a row predating the
DEFINE keeps NONE, and a strict type then rejects the **whole record** on the next
UPDATE, because a SCHEMAFULL update re-validates every field. Migrations 61
(`entity`) and 64 (`source`) each fixed that reactively after it surfaced live,
and migration 65 turned the repair into an idempotent forward sweep — which runs
*before* 78 and so cannot cover a field 78 defines. I had read none of the three.

It would have broken four live paths: the K.2 alias transfer in two places, the
K.3 apply, and the vault round trip. The reviewer verified the fix independently
against a container, applying the migration through the runner's own derived SQL
rather than the hand-typed line — a gap neither of my two tests covered, now
closed by a third.

**An organ OF X is not X.** The curated affix list was assembled from shapes
present in the corpus, and frequency is the wrong evidence for a rule about
reference: Dutch government documents are full of bodies and offices, so the
governance affixes were the *most* common and the *least* correct. The rule
proposed `Burgemeester van Rotterdam` ~ `Rotterdam`, `Gemeenteraad van Amsterdam`
~ `Amsterdam`, `Raad van Toezicht` ~ `Toezicht` and more as merges — and
`test_org_affixes.py` **pinned one of them as correct behaviour**. A test written
from the same misunderstanding as the code cannot catch it. That is the argument
for adversarial review in one example.

Cut from 40 affixes to 11, and it costs the plan's own example: `Minister van
Binnenlandse Zaken en Koninkrijksrelaties` beside `Binnenlandse Zaken en
Koninkrijksrelaties` is an office-of pair and is no longer produced. That class is
real and belongs in a later phase as an **organ-of relation** — proposing a merge
asserts something stronger than the evidence supports.

Round 2 then found that `waterschap` had survived the cut under a sentence that
did not cover it, putting the phase's two new test files in direct contradiction
on the same distinction. Round 3 corrected `provincie`'s stated justification for
the same reason — it survives on the homonym argument, not the `gemeente` one.

## The reading that carries forward

Removing `waterschap` and re-adding it left the suite **green**, and I read that as
"the mutation survived". It had not: `test_short_forms_are_recognised` still
asserted the old behaviour, so the mutant was passing a test that agreed with it.

A mutant no guard catches and a mutant a guard *endorses* look identical from the
summary line, and they need opposite fixes — strengthen the guard, or move the
assertion. Deleting code without moving its test is the same defect as deleting a
producer without deleting its consumer, which is the defect this whole phase kept
producing.

## What the reviewer credited

The `_record` / `_strongest_band` fix — a 0.945 embedding was overwriting a 0.94
fuzzy and *demoting* the pair from auto to review, because two tiers with different
thresholds were being compared on raw score. Found by reasoning about scales, not
by a failing test.

`test_cross_type_never_auto_merges_even_at_maximum_score`, which drives the band at
score 1.0 rather than trusting a threshold to be high enough.

And, in round 3's words, the thing that is hardest to do under review pressure:
correcting the **argument** rather than only the code. `waterschap` removed with
the reason written down; the `stichting` residual risk named rather than hidden;
the guard's limits paragraph rewritten to state a measured claim instead of an
intended one; and the report's "three numbers that do not reconcile" section left
as a recorded discrepancy rather than a reconciliation nobody can check.

## Environment notes worth keeping

- **Docker was down for round 1 and up for rounds 2–3.** Round 1's skipped
  container results were genuine, not a misconfiguration, and nothing in either
  round needed retracting. Worth checking before treating a skip as a verdict.
- **The working database was emptied between rounds 1 and 2**, for real data. Every
  figure in the phase report predates that and cannot be re-run. The 82-candidate
  containment measurement also predates the affix cut, so what the 11-affix list
  produces on real data is **unmeasured** — the first run against real content is
  the measurement.

## Follow-ups filed, not fixed here

- `POST /apply` performs no band or type check — it applies whatever cluster the
  client echoes, so the router docstring's "only `auto_merge` candidates may be
  applied" is enforced only by the frontend. Pre-existing from K.5; PC.2 is what
  puts cross-type pairs into the review list. → **PC.5**
- `CandidatesResponse.auto_merge` and its counts are fetched by the resolution page
  and never rendered. A UI decision rather than forgotten wiring — but `fold_equal`
  is a new AUTO producer, so a same-type case-only duplicate now lands in a band no
  curator sees and `okf_import_service` applies unattended. → **PC.5**
- The office-of pair class needs a home as a relation. → later phase
- Three AST evasions remain, declared in the guard's own docstring: a pattern
  reached through a subscript, a tuple unpack, or another module's namespace.
