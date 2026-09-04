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

Three rules were tried and **measured on 5000 live entity rows**:

| rule | candidates | failure mode |
|---|---|---|
| unanchored (`_is_token_subsequence`, as `concept_alignment` had it) | — | pairs `Regio Deal` with **both** `Regio Deal Groningen` and `Regio Deal Drenthe` — manufactures exactly the merge the config refuses |
| head-anchored + free length guard (inner ≥ 2 tokens) | 315 | a place name in tail position: `Het Hogeland` pairs with seven organisations merely operating there (`Mensenwerk Het Hogeland`, `Ondernemersplatform Het Hogeland`, …) |
| head-anchored + curated head run | **82** | place-name noise gone; every candidate explainable by naming the run removed |

The rule rests on two facts about Dutch naming: a qualifier that does **not**
change the referent sits at the head (`Gemeente X`, `Minister van X`, `de heer X`),
while a discriminator that **does** sits at the tail (`Regio Deal Groningen`). So
the shorter name must be a token *suffix* of the longer, and the removed run must
be a class or role noun rather than a proper name.

A curated list is therefore not a shortcut around a general rule; on this data it
**is** the rule, and it has a property no length threshold has — a curator can be
told *why*. `shared/utils/org_affixes.py` returns the removed run, not a boolean,
for exactly that reason.

**Why a merge rule may not do this and a review rule may.**
`nl_normalization.strip_leading_noise` refuses to strip these affixes, in writing,
because each can collapse a surface form onto a bare concept token another entity
owns (`Ministerie van Onderwijs` → `onderwijs`). That objection is against a
*normalisation*, which merges silently and irreversibly. It does not carry to a
*review proposal*, where both forms go in front of a human. `Onderwijs` /
`Ministerie van Onderwijs` is in the 82 and belongs there.

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
because a mutation forcing cross-type off left the suite green — the card mounts a
Radix AlertDialog the node-environment vitest cannot render.

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
`register_alias` inserts **nine**. SurrealDB drops the undeclared ones silently —
no error, no log. So on any freshly migrated database every alias loses
`match_type`, `similarity_score`, `method` and `verified`, and `vault_sync_service`
can never export a single alias because nothing can ever be `verified = true`.

Invisible against `staging`, where the table predates being schema-locked and all
nine columns exist. The ordering test **passes there and fails on a fresh
container**, where all three sort keys read NONE. Migration 78 declares the four,
`verified` with `DEFAULT false` so pre-existing rows read as unverified rather
than NONE — ranked below verified and excluded from the export, which is the safe
direction.

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

## Measured end to end

Against the working database (`staging`), 543 active entities:

```
auto_merge: 20   review: 34
by method: embedding 34, fold_equal_cross_type 8, fold_equal 6, containment 4, fuzzy 2
```

The 82 containment candidates quoted above were measured over 5000 **raw** rows
including merged and reference entities; the 4 here are what survives the active
filter on the global scope. Different populations, both reported.

## Evidence handed to PC.4

The cross-type candidates are the list PC.4's AC needs — labels holding two
canonical answers on real data, rather than a sweep over shipped ontologies:

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
sharper signal than a scattering of unrelated disagreements would be.

## Mutation testing

Thirteen mutations. Twelve caught on the first run; **one survived and is the
reason a test was added**.

| # | mutation | caught by |
|---|---|---|
| M1 | force-split veto disabled | 2 tests |
| M2 | cross-type allowed to auto-merge | 2 tests |
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

**M4 is the useful one.** Deleting the head-anchoring check left all 28 tests
green, because the pair it was written for — `Regio Deal` /
`Regio Deal Groningen` — is rejected by the curated list anyway (`regio` is not an
affix). The check earns its place on a shape no test had: a curated head over a
*different* tail, where dropping it clips the head blind and pairs
`Gemeente Amsterdam` with `Utrecht`. Both halves of the rule are now pinned
independently.

**M8** is the same lesson in the frontend: the rendering decision had no test that
could fail, because the card cannot be mounted in this environment. Extracting the
decision made it testable.

## Process note

Uncommitted work was lost to `git checkout --` during mutation reverts for the
seventh and eighth time this track. The rule is now: mutate against a scratchpad
copy and restore with `cp`, never `git checkout`. On the eighth occurrence the
test written *before* the mutation caught the loss immediately, which is the
argument for the ordering.
