# Phase PC.2 — One identity

- **Branch**: `feature/track-pc2-one-identity`
- **Commits**: `58eddd5d`, `05b8a364`, `729e3925`, `cb3532dc`, `215954a8`,
  `79e8deed`, `0f96e71b`, `4a9886e0`
- **Date**: 2026-09-03

The phase closes the four AC items in the plan: one normalisation, one alias
policy, the long-form/short-form pairs reachable by a curator, and a guard that
fails if a fifth copy of the fold appears.

## 1. One comparison fold

Four stages carried a byte-identical four-line transform —
`EntityDeduplicator._normalize_key`, `FuzzyResolver._normalize`,
`KGResolver._normalize`, `concept_alignment._normalize` — and a fifth call site
reached one of them through a private attribute across a package boundary
(`candidate_dedup_service` calling `self._fuzzy._normalize`).

Verified identical by **execution** over 18 adversarial inputs (NBSP, ideographic
space, full-width forms, ligatures, Turkish dotted I, empty, whitespace-only)
before consolidating, so the change is a refactor with no behavioural delta at any
of the four.

`shared.utils.text_folding.fold_for_comparison` is deliberately not named
`normalize_*`: `normalize_entity_name` lives one file away and does something
materially different (trailing-punctuation stripping, Dutch article stripping,
spelling canonicalisation, curated org-alias expansion). At a call site the two
would be indistinguishable by name, and picking the wrong one inside concept
alignment would compare post-alias-expansion strings — pre-merging exactly the
identities D-N4-9 says must not be merged without a decision.

`tests/test_one_comparison_fold.py` is an AST guard against a fifth copy. It scans
tracked **and untracked** files, recognises precompiled `re.compile(r"\s+")`
patterns, and keys its allow-list by `path::name` so a rename cannot slip past it.

One transform was deliberately **not** folded in: `EntityNormalizer._normalize_text`
trips a similar shape but is a configurable merging transform inside a pipeline
stage rather than a comparison key — it does not lowercase and it strips English
articles. Folding it would change what the Normalizer merges, on by default, in a
phase whose remit is a refactor.

## 2. Three generators the similarity tiers structurally cannot see

### Band selection per method, not per raw score

`_record` kept one `(score, method)` per candidate and overwrote on the higher
number. Fuzzy and embedding have different thresholds, so a 0.945 embedding
replaced a 0.94 fuzzy and **demoted the pair from `auto_merge` to `review`**.
Scores from different scales are not comparable; the band is. Every method now
keeps its own score and `_strongest_band` picks the strongest band.

### `fold_equal`

Names equal under the shared fold. Measured on the live graph: of **15 case-only
duplicate groups among active entities, 6 reached no curator at all** — a name
short enough or a vector far enough apart and the pair was invisible to both
tiers. Equality is not a similarity score, so it does not depend on either
threshold.

Same type → `auto_merge`. Different type → `review`, never auto.

### `containment`

`Gemeente Leudal` / `Leudal` sits at Levenshtein **0.53**. No threshold the dedup
config would accept reaches it, and one that could would also merge `Regio Deal
Groningen` with `Regio Deal Drenthe` (≈0.83) — which the config comment already
names as the tension it refuses to resolve by lowering the bar. Jaro-Winkler does
not help: it boosts common prefixes, and `Minister van …` differs at position 0.

The rule took **four** attempts. Three were measured on 5000 entity rows of the
working corpus; the fourth came from adversarial review.

| rule | outcome |
|---|---|
| unanchored (`_is_token_subsequence`, as `concept_alignment` had it) | pairs `Regio Deal` with **both** `Regio Deal Groningen` and `Regio Deal Drenthe` — manufactures exactly the merge the config refuses |
| head-anchored + free length guard (inner ≥ 2 tokens) | 315 candidates; a place name in tail position pairs `Het Hogeland` with seven organisations merely operating there |
| head-anchored + 40 curated head runs | 82 candidates, place-name noise gone — **but 9 of the 40 affixes name an organ OF the remainder** |
| head-anchored + 12 curated head runs | organ-of pairs gone |

**The fourth attempt is the one that matters, and review found it, not me.** The
40-affix list was assembled from shapes present in the corpus, and frequency is a
terrible guide here: the governance affixes are common precisely because Dutch
government documents are full of bodies and offices, and a body OF X is a
different entity from X. The list proposed all of these as merges:

```
'Raad van Toezicht'                        ~ 'Toezicht'
'Raad van Advies'                          ~ 'Advies'
'College van Beroep'                       ~ 'Beroep'
'Gemeenteraad van Amsterdam'               ~ 'Amsterdam'
'Burgemeester van Rotterdam'               ~ 'Rotterdam'
'Gedeputeerde Staten van Drenthe'          ~ 'Drenthe'
'Dagelijks Bestuur van Wetterskip Fryslân' ~ 'Wetterskip Fryslân'
'Ministerie van Onderwijs'                 ~ 'Onderwijs'
```

The first three are verbatim the objection `nl_normalization.strip_leading_noise`
states in writing — collapsing a named body onto a bare concept token another
entity owns. "It is a review proposal, a human decides" answers that one. It does
not answer the rest: the mayor is not the city, and asking a curator to arbitrate
between two identities the rule has declared to be one is a leading question in
front of a destructive button. Worse, `test_org_affixes.py` **pinned the Wetterskip
pair as correct behaviour**, so this was a design position, not an oversight.

What survives is 12 affixes: municipality / province / water board, legal forms
(`Stichting`, `Vereniging`, `Coöperatie`), and personal honorifics. All eleven
counterexamples are now rejected and pinned, plus a structural guard that no
curated run ends in a relational `van` — which is the shape every removed affix
had, and the addition most likely to be made again.

**One judgement is deliberately kept.** `Gemeente Leudal` and `Leudal` are not
literally the same thing either — a governing body and an area. In Dutch policy
prose they denote one actor and appear interchangeably in the same paragraph, and
this is the pair the PC.2 plan named as the case to solve. Kept as a judgement
about this corpus, said out loud in the module docstring rather than implied.

**What the correction costs.** The plan also named
`Minister van Binnenlandse Zaken en Koninkrijksrelaties` beside
`Binnenlandse Zaken en Koninkrijksrelaties`. That is an office-of pair, so it is
no longer produced. The class is real and worth surfacing — but as an
**organ-of relation**, not a merge, because proposing a merge asserts something
stronger than the evidence supports. Filed for a later phase.

Containment is always `review`, never `auto`: which of the two forms is canonical
is exactly what a reviewer is for.

## 3. A deliberate contract change, recorded rather than deleted

`test_cross_type_homograph_never_proposed` asserted that a same-name person and
organisation produce no candidate at all. Its **intent** — never auto-merge across
types — is preserved by an auto floor of 2.0 that no score can reach, and is now
pinned by a test that drives `_strongest_band` at score 1.0 rather than trusting a
threshold to be high enough.

Its **over-reach** is dropped: suppressing the candidate entirely assumes
`entity_type` is reliable. It is not. On the live graph, 4 of 539 distinct names
exist twice with byte-identical spelling under two different types, and under the
old rule no curator could ever see them.

## 4. Cross-type candidates reach a card that shows the difference

Cross-type pairs have byte-identical names on both sides, so a single-type card
renders `Regio Deal ↔ Regio Deal · programme` — nothing to decide on.
`entity_type_b` now travels service → `CandidateOut` → `MergeCandidate` → card.

Two labelling bugs fixed on the way, both about the same principle:

- `MergeCandidate.to_merge_cluster` already documents that `new_canonical` must
  be the **winner's** name, because K.3's apply repoints relations onto
  `winner_id`. The frontend's `candidateToApplyCluster` took `name_a`
  unconditionally, so any pair where b won on confidence renamed the survivor to
  the entity it had just absorbed. Pre-existing and independent of cross-type.
- The same argument for the type, which cross-type makes reachable. Both sides now
  resolve by winner.

`isCrossTypeCandidate` / `candidateTypeLabel` are pure functions rather than JSX,
because a mutation forcing cross-type off left the suite green.

**And that fix was not enough — review found the card still unwired.** The two
functions were written, unit-tested, and imported by `CandidateMergeCard.tsx`, and
called nowhere. A `git checkout --` during that same mutation reverted the card
edit along with the file I meant to restore, and restoring the other file did not
restore this one. `tsc` was clean, the suite was green, and a cross-type candidate
rendered exactly the string this section claimed to have fixed — one click from a
destructive apply with the only distinguishing fact hidden.

Extracting a decision into a tested pure function is not the same as making it.
`CandidateMergeCard.test.tsx` now **mounts the real component** under jsdom, which
this project supports through a per-file docblock — so the earlier claim that the
node-environment vitest could not render it was my assumption rather than a limit.
Re-planting the exact regression fails it.

## 5. One surface form, one canonical

`find_by_alias` did `WHERE alias_text = $t LIMIT 1` with no ordering. Nothing
constrains `alias_text` to be unique and `register_alias` writes a row per
resolution, so two rows can bind one text to two entities and the caller got
whichever the storage engine returned — the same input resolving differently
between two runs, with both answers looking equally confident.

Ordering is now part of the contract: `verified DESC, similarity_score DESC,
id ASC`. A human decision outranks a machine one — which is also the disagreement
this reader was on the wrong side of, since `vault_sync_service` exports only
`verified = true` aliases while this accepted any. Ranking rather than filtering,
because every writer today writes `verified = false` and a filter would make
tier-1 resolution inert.

### The schema drift the fix ran straight into

`entity_alias` is SCHEMAFULL. Migration 39 declares **five** fields;
`register_alias` sets six keys, **four of which no migration declared**. SurrealDB
drops the undeclared ones silently — no error, no log. So on any freshly migrated
database every alias loses `match_type`, `similarity_score`, `method` and
`verified`, and `vault_sync_service` can never export a single alias because
nothing can ever be `verified = true`.

Invisible against `staging`, where the table predates being schema-locked and all
nine columns exist. The ordering test **passes there and fails on a fresh
container**, where all three sort keys read NONE.

**The first version of migration 78 was a bare DEFINE, and review caught it.** A
SurrealDB column DEFAULT applies only to NEWLY created records; a row that
predates the DEFINE keeps NONE, and a strict type then rejects the **whole
record** on the next UPDATE, because a SCHEMAFULL update re-validates every field.
This repository had already measured and repaired that exact class twice —
migration 61 (`entity.manual_override`) and migration 64 (`source.private`) — and
migration 65 turned the repair into an idempotent forward sweep. 65 runs *before*
78, so it cannot cover a field 78 defines. The header of my first draft asserted
the opposite of what those three migrations document.

It would have broken four live paths: the K.2 duplicate-merge alias transfer in
`canonical_entities.py` and again in `services/extraction/api.py`, the K.3 apply
in `recanonicalization_service.py`, and the vault round trip. Migration 78 now
ends with `UPDATE entity_alias SET verified = verified ?? false`, following 61 and
64's form, and `verified` is the only field made strict — it carries a policy
(the ordering ranks by it, the vault export filters on it) and a tri-state boolean
makes both meaningless. The other three are legitimately absent for aliases
created outside the resolver, so `option<>` is the honest type.

The test that proves it forges the legacy row with the migration-64/65 technique
(OVERWRITE as `option<>` → create at NONE → re-DEFINE strict without backfilling),
asserts the K.2 transfer **fails** before the repair and **passes** after, and
greps the migration body so the fix cannot live only in the test. The previous
test covered a newly created row, which is exactly the case a DEFAULT does handle
— so it could not have seen this.

## 6. One alias policy

`KGResolver` auto-registered on a fuzzy match while concept alignment, in the same
pass, refused on the stated grounds that merging identities must be a deliberate
act (D-N4-9). Settled the way D-N4-9 settled it, for a reason `find_by_alias`
makes concrete: `use_alias_table` consults `entity_alias` at tier 1, so an alias
written from a fuzzy match becomes ground truth for the next resolution — a
machine guess hardening into identity with no human in the loop.

The default moves in **both** places, because a caller constructing `KGResolver`
directly bypasses `KGResolutionConfig` — which is how one flag comes to mean two
things.

This changes no behaviour today, which is the argument for doing it now:
`enabled` is False and `entity_alias` holds 0 rows on the working database, so
the writer has never run. It remains a default, not a removal.

## 7. The dead producer the door replaced

`lexical_alias_candidates` computed name-containment pairs into
`report["alias_candidates"]`, read by nothing since it was written. PC.1b's
inventory assigned the row to PC.2; PC.2 built the replacement.

Same population — candidates come from `find_by_type` on the entity repo, i.e.
graph entities. Weaker rule — unanchored containment, and
`test_alias_candidates_are_direction_agnostic` asserted the `Regio Deal` /
`Regio Deal Midden-Limburg` pairing as *correct behaviour*. No consumer, no card,
no apply path.

Removed: `AliasCandidate`, `lexical_alias_candidates`, the report key, the third
element of `_classify`'s return, and `min_inner_tokens` from both the aligner and
`ConceptAlignmentConfig`. Kept and rewritten: the D-N4-1 property that belongs to
*that* module — containment is not evidence of subsumption and may not steer the
alignment verdict — plus an assertion that the report key does not come back.

## Measured end to end — and what can no longer be checked

**The working database has since been emptied.** Every figure below was taken
against `staging` on 2026-09-03, before the corpus was cleared for real data.
They are a record of what was observed, not something a later reader can re-run,
and where they disagree with each other I have said so rather than picked one.

Against 543 active entities, global scope:

```
auto_merge: 20   review: 34
by method: embedding 34, fold_equal_cross_type 8, fold_equal 6, containment 4, fuzzy 2
```

Two caveats that earlier drafts of this report did not carry:

- The **82 containment candidates** were measured over 5000 **raw** rows,
  unbucketed. Production runs `_score_containment` inside the `entity_type`
  bucket loop, and the 4 above are what survives that plus the active filter.
  Different populations, and neither the count nor its precision transfers from
  one to the other. Both are stated; neither should be quoted alone.
- After the affix correction, that 82 is itself historical. The 12-affix list was
  never run against the corpus, because the corpus was gone before the correction
  was made. What the reduced list produces on real data is **unmeasured**, and
  the first run against real content should be treated as the measurement.

### Three numbers that do not reconcile

| claim | where |
|---|---|
| 4 of 539 distinct names exist twice under two types | a probe grouping **byte-identical** `name` strings |
| 8 cross-type candidates | the end-to-end run, grouping by the **comparison fold** |
| 15 case-only duplicate groups, of which 6 reached no curator | an earlier probe over active entities |

The first two most likely differ because one compares bytes and the other
compares folded strings, so `Regio Deal` / `regio deal` under two types counts in
the second and not the first — which also means the PC.4 evidence table below is
**fold-equal**, not byte-identical as an earlier draft said. And 6 `fold_equal` +
8 cross-type = 14 against 15 groups, with the remaining one unaccounted for;
"groups" and "pairs" are also not the same unit, since a group of three yields
three pairs.

I cannot verify any of this now. Recording the discrepancy is more useful to PC.4
than a confident reconciliation I cannot check.

## Evidence handed to PC.4

Eight names the graph held twice under two types, equal under the comparison fold:

| name | types |
|---|---|
| `Regio Deal` | programme ↔ topic |
| `Regio` | administrative_area ↔ programme |
| `Leerwerkvoorzieningen` | programme ↔ topic |
| `Nij Begun Academie` | topic ↔ programme |
| `Verlengde en verrijkte schooldag` | programme ↔ topic |
| `Actieagenda Sterk Bestuur` | programme ↔ topic |
| `Leerwerktrajecten` | programme ↔ topic |
| `Straatteams` | programme ↔ topic |

Six of eight are the same confusion — `programme` against `topic` — which is a
sharper signal than a scattering of unrelated disagreements. The corpus is gone,
so PC.4 should re-derive the list from the cross-type band on real data rather
than treat this table as current; what carries forward is the shape of the
finding, not the rows.

## A behaviour change in an unattended destructive path

`okf_import_service._run_dedup` applies **every** `auto_merge` candidate with
`apply_merge` — no human, no confirmation. `fold_equal` is a new AUTO producer at
score 1.0, so PC.2 widens what that path merges: a case-only duplicate of the
same type now merges unattended during an OKF import.

That is intended. Same type and equal under the fold is what a duplicate is, and
the fold strips no punctuation and folds no diacritics, so the rule is tight. But
it is a change to a destructive path and belongs in the record rather than in the
diff alone. `test_the_auto_merge_band_never_carries_a_cross_type_pair` asserts on
the list that path actually reads, not on `_strongest_band`, so a cross-type pair
cannot reach it.

Related and **not** fixed here: `POST /apply` performs no band or type check — it
applies whatever cluster the client echoes, so the router docstring's "only
`auto_merge` candidates may be applied" is enforced only by the frontend.
Pre-existing from K.5, surfaced by review, and filed for **PC.5** (curator
surface) rather than changed in a phase about identity.

## Mutation testing

Fifteen mutations. Thirteen caught on the first run; two survived, and both are
the reason a test exists.

| # | mutation | caught by |
|---|---|---|
| M1 | force-split veto disabled | 2 tests |
| M2 | cross-type allowed to auto-merge | 2 |
| M3 | containment allowed to auto-merge | 1 |
| M4 | head-anchoring dropped | **nothing** → see below |
| M5 | curated list dropped (any head run accepted) | 1 |
| M6 | band picked by raw score again | 1 |
| M7 | winner-name reverted to `name_a` | 1 |
| M8 | `crossType` forced false | **nothing** → extracted to a pure function, then 1 |
| M9 | migration 78 stops declaring `verified` | 2 |
| M10 | `ORDER BY` removed | 1 |
| M11 | `verified` removed from the sort key | 1 |
| M12 | config default back to `True` | 1 |
| M13 | constructor default back to `True` | 2 |
| M14 | the card's functions imported but not called (round 2) | 1 |
| M15 | the organ-of affixes re-added (round 2) | 8 |

**M4 is the useful one.** Deleting the head-anchoring check left all 28 tests
green, because the pair it was written for — `Regio Deal` /
`Regio Deal Groningen` — is rejected by the curated list anyway (`regio` is not
an affix). The check earns its place on a shape no test had: a curated head over
a *different* tail, where dropping it clips the head blind and pairs
`Gemeente Amsterdam` with `Utrecht`.

**M8 is the one I got wrong twice.** The rendering decision had no test that could
fail; I extracted it into a pure function, tested the function, and never changed
the caller. The mutation I ran afterwards tested the function I had just written,
not the behaviour it was for. M14 is that same mutation aimed at the right target.

## What review changed, and what that says

Adversarial review returned REVISIONS_NEEDED on attempt 1 with two blockers and
four majors. Both blockers were things this phase's own report asserted as done:

- **The card was never wired** (§4). Green suite, clean types, claim in writing.
- **Migration 78 was a bare DEFINE** (§5), with a header asserting the opposite of
  what migrations 61, 64 and 65 document in this same repository. I had read
  neither before writing it.

And the major that mattered most, **M1 — an organ OF X is not X** — was not a
missed edge case. The 40-affix list was built from corpus frequency, and the tests
*pinned one of the false pairs as correct*. Frequency is the wrong evidence for a
rule about reference, and a test written from the same misunderstanding as the
code cannot catch it. That is the argument for adversarial review in one example.

## Process note

Uncommitted work was lost to `git checkout --` during mutation reverts nine times
across this track — the ninth is B1 above, and it shipped. The rule is now: mutate
against a scratchpad copy and restore with `cp`, never `git checkout`. On the
eighth occurrence the test written *before* the mutation caught the loss
immediately, which is the argument for that ordering; on the ninth there was no
such test, which is how it reached the report as a completed item.
