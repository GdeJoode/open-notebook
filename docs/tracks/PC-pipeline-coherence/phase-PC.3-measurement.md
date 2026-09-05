# PC.3 step 6 — what stage 10 actually does

*Measured 2026-09-05 against `open_notebook/staging` at migration 80. Reproduce
with `SURREAL_DATABASE=staging uv run python scripts/pc3_resolution_measurement.py
--sweep`. Read-only.*

## The measurement is not the one the plan described

The plan called for re-extracting the corpus into a separate database. Two facts
changed that.

**The cheap replay does not reach the question.** `run_filtering_only` re-runs
filtering over a stored `extraction_result` without re-extracting — ideal, except
that only **6 of the 14 sources still have one**, and they are the English-language
papers. Not a single convenant. Two of 531 active entities and **one of the twenty
`government_organization` rows** touch a replayable source, so a replay measures
nothing this phase asks about.

**The expensive run is not needed for AC #2/#3.** Stage 10's question is "given
this mention, what does it match in the graph?" That can be asked directly,
twenty times, in seconds, using the production resolver at the production
thresholds. It is not a substitute for an end-to-end run; it IS the measurement
the criteria ask for.

## A. Against the live graph

Every one of the twenty mentions matches its own row at fuzzy 1.000 — expected
and uninteresting. What it measures is the surroundings:

* **The cap does not bind.** 54 rows of this type against a cap of 100;
  `capped_fetches = 0`. AC #4 is answered: on this class a miss is a miss, not a
  look that never happened.
* **34 of those 54 candidates are `archived`.** `find_by_type` applies no status
  filter, so two thirds of what the resolver compares against are rows triage
  deliberately retired.

Declared limit: this shows an identical ACTIVE row wins the tie. It does **not**
show archived rows are harmless to a mention with no active twin, and this probe
cannot reach that.

## B. What a real run would consolidate

The twenty arrive one at a time in creation order, each asking whether it matches
what is already there. Production resolver, production thresholds; the repository
is a stand-in that serves the real rows with their real embeddings through
`find_by_type`'s own contract, and exists only to pose "a graph that does not yet
contain this row".

**20 rows → 15 entities. Of the five merges, one is right and four are wrong.**

| | merge | tier | score |
|---|---|---|---|
| ✅ | `VRO (Volkshuisvesting en Ruimtelijke Ordening)` → `VRO (ministerie van Volkshuisvesting en Ruimtelijke Ordening)` | semantic | 0.961 |
| ❌ | `VRO (VROM, Volkshuisvesting, Ruimtelijke Ordening en Milieubeheer)` → `VRO (ministerie van …)` | semantic | 0.937 |
| ❌ | `De Staatssecretaris van Infrastructuur en Waterstaat (IenW)` → `IenW (Infrastructuur en Waterstaat)` | semantic | 0.916 |
| ❌ | `VRO (Ministerie van Volksgezondheid, Welzijn en Sport)` → `Ministerie van Volksgezondheid, Welzijn en Sport` | fuzzy | 0.889 |
| ❌ | `ministeries van het Rijk: Volkshuisvesting en Ruimtelijke Ordening` → `Volkshuisvesting en Ruimtelijke Ordening (Ministerie)` | semantic | 0.930 |

Every one of the four is a case **AC #3 names in advance as must-not-merge**:
the office-of shape PC.2 cut 29 affixes to exclude; an abolished predecessor
ministry; the extraction error the plan explicitly forbade resolution from
"fixing"; and a conjunction of four ministries merged into one of its members.

Against AC #2's target — four groups, six consolidations, 20 → 14 — it achieves
**one of six**. The BZK pair, the whole IenW group and the HG/BZK pair are all
missed.

**Why it fails in both directions at once.** The pairs that should merge are
abbreviation against expansion: `IenW` / `Infrastructuur en Waterstaat
(Ministerie)`, `BZK` / `Binnenlandse Zaken en Koninkrijkrelaties (Ministerie)`.
Those are far apart in edit distance AND in a name-only embedding — a four-letter
token and a forty-character phrase. What the resolver *does* merge is names that
are structurally different but topically adjacent, and every Dutch ministry name
embeds close to every other. Both tiers are measuring an axis that is not
identity.

## C. There is no threshold that helps

| fuzzy | semantic | entities | correct | wrong |
|---|---|---|---|---|
| **0.85** | **0.90** | **15** | **1** | **4** | ← shipped |
| 0.85 | 0.95 | 18 | 0 | 2 |
| 0.90 | 0.90 | 15 | 1 | 4 |
| 0.90 | 0.95 | 19 | 0 | 1 |
| 0.92 | 0.95 | 19 | 0 | 1 |
| 0.90 | 0.97 | 20 | 0 | 0 |
| 0.95 | 0.97 | 20 | 0 | 0 |

**Correct never exceeds wrong at any setting**, and the correct merge dies before
the wrong ones do. Raise the semantic threshold to 0.95 and the count of correct
merges goes to zero while wrong merges remain.

The reason is worse than a badly chosen threshold, and it is the finding that
settles this. At 0.95, `VRO (VROM, …)` survives as its own entity — and
`VRO (Volkshuisvesting en Ruimtelijke Ordening)` then matches **it** at 0.965,
*higher* than the 0.961 at which it finds the correct ministry. The abolished
predecessor outranks the real one. No threshold separates them because the
**ranking itself is wrong**; tuning can only choose which wrong answer to accept.

This also makes the outcome non-monotone in the threshold: raising it turned a
correct merge into a wrong one, because removing a competitor changes who wins.
The graph therefore depends on document arrival order.

## What follows

**The recommendation is that stage 10 does NOT go on by default**, and it
contradicts the premise this phase was planned on. That premise — "cross-document
resolution goes in the default path, with a measured cost" — was a reasonable
decision made before anyone had measured it. The measured cost is four wrong
identity claims for one right one, three of which assert something false about
Dutch government (a state secretary is their ministry; VROM is VRO; a mention
mislabelled by extraction is a real ministry).

What the measurement does NOT say: that resolution is worthless. It says the two
tiers available cannot do this on names alone. The abbreviation↔expansion class is
exactly what the **curated org-alias table** already handles — `normalize_entity_name`
turns bare `IenW` into `infrastructuur en waterstaat` today — and what defeats it
is the parenthetical, which is a normalisation problem, not a similarity one. That
is a cheaper and more accurate lever than a threshold.

Three things this phase should carry forward rather than fix here:

* **`find_by_type` has no status filter**, so retired rows compete for identity.
  The same defect sits in `upsert_entity`'s lookup, which is how a probe in this
  session updated an archived row.
* **Three producers of entity embeddings, two texts.** `_embed_entities` uses
  `entity.text` and `backfill_entity_embeddings` uses `canonical_name` — both the
  bare name — while `semantic-intelligence/scripts/test_pipeline.py` embeds
  `f"{etype}: {name}. {desc}"`. Vectors built from different texts are compared
  against each other by cosine.
* **`KGResolutionConfig`'s docstring is now stale** on both of its factual claims:
  `enabled` is no longer False, and `entity_alias` no longer holds 0 rows.
