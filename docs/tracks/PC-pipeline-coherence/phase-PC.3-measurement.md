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

## D. The authority check — asked for, and missing from everything above

*Added after the question "where is the check against TOOI?", which the
measurement above did not consider. It changes the recommendation's reasoning.*

TOOI — the Dutch government's authoritative register — is **already in this
repository**, and it holds exactly the mapping stage 10 cannot compute. Per
organisation it carries the abbreviation, the official name without the
organisation type, and the official name with it:

```
tooiont:afkorting              "BZK"
tooiont:officieleNaamExclSoort "Binnenlandse Zaken en Koninkrijksrelaties"
tooiont:officieleNaamInclSoort "ministerie van Binnenlandse Zaken en Koninkrijksrelaties"
```

Three surface forms, one stable URI. That is the abbreviation↔expansion class
solved by LOOKUP rather than by similarity — the class section B shows both tiers
failing at.

**It runs nowhere.** `shared/vocabulary/tooi_provider.py` and
`entity_resolution/vocabulary_reconciler.py` exist and are exported; the only
entry points are three manual endpoints (`POST /vocabularies/refresh`,
`POST /validate`, `POST /vocabulary/refresh`). **`reconcile_entity` has no
production caller at all**, and `reference_entity` holds **0 rows** on `staging`
— the register has never been loaded. Nothing in extraction or filtering
references TOOI.

This is PC.1b's own defect one level up: a producer with no consumer, except the
orphan is a whole capability rather than a field. `tests/test_derived_state_has_
readers.py` cannot see it — it enumerates fields on `FilteredResult` and the
metadata keys the pipeline writes, and a service method that nobody calls is not
a field. The handoff inventory does not mention vocabulary, TOOI or the
reconciler anywhere.

### Would it have helped? Measured, and the answer names the real defect

Loading the bundled seed and posing the same twenty surface forms:

| lookup | resolved |
|---|---|
| exact, as the names stand | **1 / 20** |
| after stripping the decoration (`X (Ministerie)` → `X`, `ABBR (X)` → `ABBR`) | **7 / 20** |

**The same thing defeats all three mechanisms**, and this is the finding that
matters more than any of the individual numbers:

| mechanism | result on the 20 | what defeats it |
|---|---|---|
| `normalize_entity_name` (identity key) | 20 rows → 20 keys | the parenthetical |
| stage 10, fuzzy + semantic | 1 right, 4 wrong | the parenthetical |
| TOOI authority lookup | 1 / 20 | the parenthetical |

The problem was never *which matcher*. Nothing strips the decoration before
matching, so every mechanism is asked to compare `Infrastructuur en Waterstaat
(Ministerie)` against `Infrastructuur en Waterstaat` and each fails in its own
way. Strip it first and the cheapest, most accurate mechanism — an exact lookup
against a register maintained by the government itself — starts working.

The thirteen remaining misses are not a failure of the method. They split into
two clean groups:

* **Register coverage, not matching.** `JenV`, `VRO`, `SZW` and `EZ` are real
  ministries absent from the bundled 10-record seed (which holds AZ, BZK, BZ,
  Def, EZK, IenW, LNV, LVVN, OCW, VWS). Four of the twenty rows hinge on VRO
  alone. The full TOOI register has all of them; the seed is the offline floor,
  not the register.
* **Correctly unresolvable.** The two `ministeries van het Rijk: …` conjunctions,
  the two Herstel-Groningen joint constructs, and the `VRO (Ministerie van VWS)`
  extraction error. A register that returned a hit for any of these would be
  wrong, and stage 10 merged three of them.

### One warning, from the same measurement

Stripping the decoration is not free. The crude rule used here resolved
`De Staatssecretaris van Infrastructuur en Waterstaat (IenW)` to the IenW
ministry — by pulling `IenW` out of the parenthetical. That is the **same
over-merge** stage 10 made and the same one PC.2 spent a review round removing
from the affix list: an organ OF X is not X. Decoration-stripping needs PC.2's
discipline — the head run is evidence, and `(Ministerie)` as a type suffix is not
the same shape as `(IenW)` as an abbreviation gloss, which is not the same shape
as a `Staatssecretaris van …` office prefix.

### What this changes

The recommendation from section C stands — stage 10 does not go on by default —
but the reason is sharper. It is not that resolution is hopeless; it is that the
pipeline reaches for similarity FIRST while an authoritative register sits
unloaded in the same repository. The order should be inverted:

1. **Normalise the decoration**, with PC.2's care about what a head run means.
2. **Look up the authority.** Exact, explainable, and it carries a stable URI
   instead of a score.
3. **Similarity last, if at all** — and on this evidence, not on by default.

Concretely, before any of this is worth measuring end to end: load the full TOOI
register (the bulk fetcher exists), give `reconcile_entity` a caller, and extend
`tests/test_derived_state_has_readers.py` — or add a sibling — so that an
exported capability with no production caller fails the way an orphaned field
does.
