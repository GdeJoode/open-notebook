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

## E. The other 96% of the graph — and a candidate pool that could not work

*Section B measured ONE entity type of twenty rows and section C recommended a
global default from it. That is sampling, which is the failure this track keeps
finding in itself, so here is the same measurement on every active type.*

| type | active | → entities | merged | capped fetches |
|---|---|---|---|---|
| `topic` | 429 | 416 | 13 | **327** |
| `administrative_area` | 52 | 52 | 0 | 0 |
| `programme` | 27 | 27 | 0 | 0 |
| `government_organization` | 20 | 15 | 5 | 0 |
| `concept` | 3 | 3 | 0 | 0 |

Three of five types produce **no merges at all**. Across the whole active graph
stage 10 makes **18 merges out of 531 entities**, and the `topic` merges show the
same ratio as the ministries — roughly five defensible against eight that are not:

```
✅ Leven Lang Ontwikkelen (LLO)          -> Leven Lang Ontwikkelen        [semantic 0.960]
❌ coöperatief wonen                     -> wonen                         [semantic 0.936]
❌ Versterken regionale samenwerking     -> Regionale samenwerking        [semantic 0.939]
❌ Arbeidsmarkt en Economie              -> Economie, Onderwijs en Arb…   [semantic 0.940]
❌ transformatie van het landelijk gebied-> Transitie van het landelijk…  [fuzzy    0.868]
❌ Innovatie omgevingen                  -> Intensiveren innovaties in…   [semantic 0.934]
```

The dominant error has a name, and it is **the same one PC.2 identified for
organisations**: a qualified concept absorbed into its own head noun.
`coöperatief wonen` is not `wonen`; `Versterken regionale samenwerking` is not
`Regionale samenwerking`. An organ of X is not X — now appearing for topics.
And note the one clear success, `Leven Lang Ontwikkelen (LLO)`: it is the
parenthetical case again.

### The candidate pool could not have worked

`find_by_type` selected every row of a type regardless of status, and capped at
100. Measured on the live repository:

| type | live rows | ALL rows | the capped 100 contained |
|---|---|---|---|
| `topic` | 785 | 1408 | **31 active** |
| `concept` | 574 | 1892 | **0 active** |

For `concept` the resolver could not reach one of the three active entities. Every
candidate it was ever offered was archived or merged, so a correct match was
**structurally impossible**, and nothing anywhere said so.

This also means **section B was optimistic**: its stand-in repository served only
active rows, so it measured stage 10 with a better candidate pool than production
ever gave it. The real behaviour was worse than the numbers above.

Fixed — `("active", "reference")` being the live set that `audit_service` and
`deep_audit_service` already use, plus a declared ordering:

| type | the capped 100 contained, after |
|---|---|
| `topic` | 63 active (was 31) |
| `administrative_area` | all 52 (was 73 mixed) |
| `programme` | all 27 (was 89 mixed) |
| `government_organization` | all 20 (was 54 mixed) |
| `concept` | **still 0 active** |

`concept` remains broken: 571 of its 574 live rows are `reference`, and the three
active ones do not fit inside a cap of 100. Reported rather than hidden — that is
what `capped_fetches` is for — but it is a real remaining limitation, not a fix.

### What this does and does not change

It does **not** rescue stage 10's precision. The quality figures in sections B and
E were already measured against an active-only pool, so they are the *after* the
fix numbers. Five defensible merges out of eighteen, across 531 entities, is a
property of the two tiers, not of the pool.

It does change what an honest verdict rests on. Turning off a feature that had
never been given a fair candidate set would have been the wrong reason for the
right conclusion.

**The recommendation therefore stands, on the whole graph rather than one type:
stage 10 does not go on by default.** The order should be inverted — normalise
the decoration, look up the authority, similarity last if at all — for a reason
sections B, D and E now agree on from three directions: the one thing every
mechanism fails on is the decoration, and the one class stage 10 gets right in
both `topic` and `government_organization` is the parenthetical it happens to
score above threshold.

### A guard that could not fail, again

The first version of `test_candidate_fetch_excludes_retired.py` asserted
`"ORDER BY" in inspect.getsource(find_by_type)` — and the docstring above the
query explains the ordering in prose, so the assertion held with the clause
deleted. Verified by deleting it. The check now reads the function's string
literals with the docstring removed.

The behavioural half is weaker than it looks and says so: SurrealDB v2 returned id
order with and without the clause on 20 rows, so no fixture here can catch the
removal by behaviour. The claim in the repository docstring was corrected to what
was measured rather than what was intended.

## F. Stage 10's verdict reaches nothing

*The last check before recommending a default: what does persistence DO with a
match? The answer reframes every number above.*

The resolver writes four keys onto a matched mention
(`kg_resolver._enrich_entity`):

```python
props["kg_entity_id"]        = kg_entity_id
props["kg_match_type"]       = match_type
props["kg_similarity_score"] = round(similarity_score, 6)
props["is_new"]              = False
```

Searching every production module for consumers:

| key | read by |
|---|---|
| `kg_entity_id` | **nothing** |
| `kg_match_type` | **nothing** |
| `kg_similarity_score` | **nothing** |
| `is_new` | stage 15 only (`workflow.py:778`, concept alignment) |

`entity_persistence_service.py` contains **zero** occurrences of `kg_`. It
identifies the entity by `name_key` through `upsert_entity`, exactly as it does
with stage 10 off. And `match_candidates` — the curator queue — is fed by the LLM
matcher at stage 6b, not by stage 10, so a resolution match is not put in front of
a human either.

The properties bag IS persisted (`stored_props` copies it wholesale minus the
embedding), so the verdict is **written to the graph and never read**. Concretely:
`coöperatief wonen` gets its own row under its own `name_key`, carrying
`kg_entity_id → wonen` as a stored annotation that nothing acts on.

**So enabling stage 10 consolidates nothing.** Its only reachable effect is that
matched entities are not marked `is_new`, which reduces what stage 15 classifies.

### What that means for this phase, plainly

PC.3 step 2 — "turn stage 10 on where the app builds its config, one field, and
the phase's whole risk surface" — turned on a stage whose primary output is dead.
That is the failure PC.6 spent five review rounds making unreachable, introduced
by the phase that came after it: *flag on, zero effect*.

It also explains why PC.6's own coherence check did not catch it. `config_coherence`
reasons in the downstream direction — stage 15 needs the `is_new` that stage 10
produces — and that dependency is real and satisfied. Nobody checked the other
direction: whether stage 10's own headline output has a reader. Likewise
`tests/test_derived_state_has_readers.py` enumerates fields on `FilteredResult`
and pipeline metadata keys; `kg_entity_id` is a key inside an entity's
`properties`, so the invariant does not reach it — the same blind spot that hid
the unwired TOOI reconciler in section D.

### The measurement keeps its value, as a forecast

Sections B, C and E now read as the answer to a different question: *what would
happen if someone wired `kg_entity_id` into persistence?* Five defensible merges
against thirteen wrong ones across 531 entities, with no threshold that separates
them and a ranking that puts an abolished ministry above the real one. So the
answer to "should we wire it" is no — not in this form, and not before the
decoration is normalised and the authority is consulted.

### What DOES resolve across documents

`upsert_entity`'s lookup on `name_key` — PC.3 step 1. It is measured, it works,
**46 of 531 active entities span more than one source — 8.7%, not the 87.5% an earlier draft of this document reported.** The figure 475 is the count over ALL 5,500 rows of every status (8.6% there too); presenting it against the 543 active ones inverted it. Corrected on re-measurement after review.

And the inference built on it is withdrawn, not just the number. A spanning rate measures how much the SOURCES overlap, not whether consolidation works: ten of the fourteen documents are convenanten for different regions, so most topics genuinely appear once. Whether `upsert_entity`'s lookup HITS is not recorded anywhere, so neither figure settles the phase's original premise in either direction.

The identity key remains this phase's real deliverable, and it never depended on
stage 10 — but that rests on the 12 collision groups it collapses and on the
code, not on a spanning rate that answers a different question.

## G. Corrections after review

*Review checked the claims in sections A–F against the database rather than
against each other, and three did not survive. They are corrected here rather
than edited away, because the phase's argument was built on them.*

### G1. The cross-document spanning rate was inverted

`475 of 543` was reported as the share of ACTIVE entities spanning more than one
source. 475 is the count over **all 5,500 rows of every status**. Re-measured:

```
ALL statuses    5500 rows | span >1 source:  475 (8.6%)
active only      531 rows | span >1 source:   46 (8.7%)
```

8.7%, not 87.5%. **And the inference is withdrawn with the number.** A spanning
rate measures how much the SOURCES overlap — ten of the fourteen documents are
convenanten for different regions, so most topics genuinely appear once. Whether
`upsert_entity`'s lookup HITS is recorded nowhere, so neither figure settles the
phase's original premise in either direction. What survives is the code fact (the
lookup unions `source_documents` when it matches) and the 12 collision groups.

### G2. "Seven of the eight errors are relations recorded as identity" — withdrawn

Three things were wrong with it.

**`HG/BZK` was never merged.** Section B's own table lists the HG/BZK pair among
the groups stage 10 MISSES. Citing it as one of the errors counted a correct
non-merge as a defect.

**Three real errors are not subsumption at all** — they are topical adjacency
between siblings, where neither is broader than the other:

```
ketenontwikkeling en innovatie      -> Intensiveren innovaties in onderwijs…  [0.915]
Innovatie omgevingen                -> Intensiveren innovaties in onderwijs…  [0.934]
Arbeidsmarkt en Economie            -> Economie, Onderwijs en Arbeidsmarkt    [0.940]
```

**And the arithmetic did not reconcile across three documents** — 5/13, 6/12 and
5/13 for the same 18 merges, because "roughly" was doing work a count should have
done, and only 6 of the 13 `topic` merges were ever listed.

### G3. The complete list, so the classification is checkable

All 18 merges. `administrative_area` (52), `programme` (27) and `concept` (3)
produce none.

**`government_organization` — 5 merges of 20 rows**

| | merge | tier | score |
|---|---|---|---|
| ✅ | `VRO (Volkshuisvesting en Ruimtelijke Ordening)` → `VRO (ministerie van …)` | semantic | 0.961 |
| ❌ narrower/broader | `VRO (VROM, …)` → `VRO (ministerie van …)` | semantic | 0.937 |
| ❌ organ-of | `De Staatssecretaris van IenW` → `IenW (Infrastructuur en Waterstaat)` | semantic | 0.916 |
| ❌ extraction error | `VRO (Ministerie van VWS)` → `Ministerie van VWS` | fuzzy | 0.889 |
| ❌ member-of | `ministeries van het Rijk: VRO` → `Volkshuisvesting en Ruimtelijke Ordening (Ministerie)` | semantic | 0.930 |

**`topic` — 13 merges of 429 rows**

| | merge | tier | score |
|---|---|---|---|
| ✅ | `Leven Lang Ontwikkelen (LLO)` → `Leven Lang Ontwikkelen` | semantic | 0.960 |
| ✅ | `Programmalijn 3: Economie, Onderwijs en Arbe…` → same, truncated | fuzzy | 0.942 |
| ✅ | `toegankelijkheid naar opleidingen en bijscho…` → same, truncated | semantic | 0.937 |
| ✅ | `Behouden en aantrekken van talenten` → same, case variant | semantic | 0.940 |
| ✅ | `Sociaal-emotionele ontwikkeling` → `sociaal-emotionele ontwikkeling va…` | semantic | 0.967 |
| ❌ narrower/broader | `coöperatief wonen` → `wonen` | semantic | 0.936 |
| ❌ narrower/broader | `Versterken regionale samenwerking` → `Regionale samenwerking` | semantic | 0.939 |
| ❌ narrower/broader | `Krachtige Kernen: samenredzaamheid…` → `Programmalijn 1: Krachtige kernen:…` | semantic | 0.906 |
| ❌ sibling | `Arbeidsmarkt en Economie` → `Economie, Onderwijs en Arbeidsmarkt` | semantic | 0.940 |
| ❌ sibling | `ketenontwikkeling en innovatie` → `Intensiveren innovaties in onderwijs…` | semantic | 0.915 |
| ❌ sibling | `Innovatie omgevingen` → `Intensiveren innovaties in onderwijs…` | semantic | 0.934 |
| ❌ different words | `transformatie van het landelijk gebied` → `Transitie van het landelijk gebied` | fuzzy | 0.868 |
| ❌ probably same | `strategische netwerkorganisatie in Noord-Holland` → `sterke strategische netwerkorganisatie…` | fuzzy | 0.885 |

**The count, stated once: 18 merges, 6 defensible, 12 not.** Of the 12: four are
narrower-into-broader, three are siblings with no subsumption between them, one
is organ-of, one is member-of, one is an extraction error, one pairs two
different words, and one is arguable either way.

### G4. What the classification actually supports

Not "the errors are relations". The defensible claim is narrower and still
useful:

> **The largest single class of error is a qualified concept absorbed into its
> own head noun** — `coöperatief wonen` into `wonen`, `Versterken regionale
> samenwerking` into `Regionale samenwerking`, the Staatssecretaris into IenW.
> Five of the twelve are that shape, and it is the same shape PC.2 identified for
> organisations and removed 29 affixes to avoid.

The sibling cases are a different problem and a harder one: `Arbeidsmarkt en
Economie` and `Economie, Onderwijs en Arbeidsmarkt` are neither the same nor
one-inside-the-other, and no relation type in this graph expresses "overlapping
compound topic". Routing similarity to a relation output would handle the first
class and not the second.

So the recommendation is unchanged — stage 10 off until its output has a
destination — but the destination is smaller than "relations": it is subsumption
for the narrower-into-broader class, and a curator's judgement for the rest.
