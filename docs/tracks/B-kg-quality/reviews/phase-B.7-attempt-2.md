# Review — Track B Phase B.7 attempt 2

**Branch**: `track/b-integration-retro`
**Decision**: APPROVED
**Reviewer**: adversarial-reviewer agent
**Date**: 2026-06-13

## Summary

Attempt 2 closes the 1 major (phase/PR arithmetic off-by-one) and all
4 minors from attempt 1 with surgical doc-only edits across the same
five files. Every count statement now reconciles ("18 PRs across 17
production sub-phases + 1 integration/retro phase (B.7)"); the
CHANGELOG kwarg name now matches the actual Python signature; the
ARCHITECTURE deviating-path rows carry explicit workspace
parentheticals; the FEATURE_ROADMAP duplicate one-line summary is
gone; the RETRO attempt-count table carries per-row review-file
hyperlinks. No code changes. Self-review file lands the attempt-2
fix log with verification commands. **Track B is officially DONE.**

## Attempt 1 issue check

| # | Severity | Issue | Status |
|---|---|---|---|
| Major #1 | 🟡 | Phase/PR count off-by-one in RETRO + FEATURE_ROADMAP | ✅ RESOLVED |
| Minor #1 | 🔵 | CHANGELOG `age_threshold` vs actual `max_age_days` | ✅ RESOLVED |
| Minor #2 | 🔵 | RETRO attempt-count table had no review-file links | ✅ RESOLVED |
| Minor #3 | 🔵 | ARCHITECTURE `orphan_connector` row lacked workspace path | ✅ RESOLVED (+ symmetric application to `metrics`) |
| Minor #4 | 🔵 | FEATURE_ROADMAP banner duplicated its one-line summary | ✅ RESOLVED |

## Verification — Major #1 (phase/PR arithmetic)

Spot-checks of every count statement in the docs:

| Location | Old text | New text | Status |
|---|---|---|---|
| `RETRO.md:22-34` summary header | "after 17 sub-phases delivered across 17 PRs ... reviewer-rejection rate was roughly 50%" | "with **18 PRs across 17 production sub-phases + 1 integration/retro phase (B.7)** ... reviewer-rejection rate was roughly 47%" | ✅ |
| `RETRO.md:132` What-hurt CI noise | "over 17 PRs" | "over 18 PRs" | ✅ |
| `RETRO.md:138-145` Recommendation #1 | "8/17 phases" | "8/17 production sub-phases" | ✅ |
| `RETRO.md:217-219` table preamble | (none) | New blockquote: "Rejection rate calculated on 17 production sub-phases; B.7 itself is reviewed separately ... and not counted in the denominator below." | ✅ |
| `RETRO.md:241-247` pattern sentence | "9/17 phases passed first try" | "9/17 production sub-phases passed first try" | ✅ |
| `RETRO.md:288-297` closing | "17 PRs over ~7 calendar days. Nine PRs APPROVED first try" | "**18 PRs (17 production sub-phases + 1 integration/retro phase, B.7)** ... Nine production PRs APPROVED first try, eight needed an attempt-2 (B.7 itself needed an attempt-2 for arithmetic fixes); zero rollbacks shipped. The reviewer-rejection rate (~47% on the 17 production sub-phases)" | ✅ |
| `FEATURE_ROADMAP.md:128-135` Track-B banner | "All 17 sub-phases (...) merged across 17 PRs" | "All 18 PRs merged (17 production sub-phases — B.0, B.1a–B.1f, B.2a/B.2b, B.3a–B.3d, B.4, B.5a/B.5b, B.6 — plus B.7 integration/retro)" | ✅ |
| `status.md:3-7` head | "All 17 sub-phases merged" | "All 18 PRs merged (17 production sub-phases + 1 integration/retro phase, B.7)" | ✅ |

Global verification:

```
$ grep -nE '\b17 PRs\b|\b17 sub-phases\b' RETRO.md FEATURE_ROADMAP.md CHANGELOG.md ARCHITECTURE.md status.md
(returns 0 lines)
```

The denominator clarification (17 vs 18) is now explicit at every
site that quotes a rejection rate — the denominator is "production
sub-phases" (excluding B.7) at the rate sentences, and the numerator
is "PRs" (including B.7) at the PR-count sentences. The B.7-counted-
in-attempts caveat ("B.7 itself needed an attempt-2 for arithmetic
fixes") is added to the closing paragraph; that's the self-aware
touch that closes the loop.

## Verification — Minor #1 (CHANGELOG kwarg)

`CHANGELOG.md:29` now reads:

> The orphan prune-lifecycle archive rule defaults to `max_attempts=3`
> and `max_age_days=90` (kwarg names match
> `pipelines/entity-filtering/.../orphan_prune.py::archive_stale_orphans`).

Cross-checked against actual code:

| Claim | Code reality |
|---|---|
| Kwarg named `max_age_days` | `orphan_prune.py:446 max_age_days: int = DEFAULT_MAX_AGE_DAYS` ✓ |
| Default value `90` | `orphan_prune.py:86 DEFAULT_MAX_AGE_DAYS = 90` ✓ |
| Function name `archive_stale_orphans` | `orphan_prune.py:441 async def archive_stale_orphans(...)` ✓ |
| Module path `pipelines/entity-filtering/...orphan_prune.py` | `pipelines/entity-filtering/src/entity_filtering/resolution/orphan_prune.py` exists ✓ |

The added module-path citation improves greppability for operators —
exactly the rationale in the original minor.

## Verification — Minor #2 (RETRO hyperlinks)

Sixteen of the seventeen production sub-phase rows in the attempt-
count table now link to a specific review file. Spot-check (three
randomly chosen):

| Link | Verified |
|---|---|
| `./reviews/phase-B.3c-attempt-1.md` | exists; documents resume-sentinel + LLM prompt corruption ✓ |
| `./reviews/phase-B.5b-attempt-1.md` | exists; documents "no production caller invoking update path" ✓ |
| `./reviews/phase-B.6-attempt-1.md` | exists; documents idempotency mock divergence ✓ |

All 16 link targets verified to exist on disk. The implementer
documented the linking strategy explicitly in the self-review
(REVISIONS → `phase-B.N-attempt-1.md`; APPROVED → `phase-B.N-attempt-
1.md`; B.1a → self-review.md because no attempt-1 file exists).

The B.1b row (1 attempt, APPROVED) is unlinked — there is no
`phase-B.1b-attempt-1.md` adversarial-review file; only a self-review
exists. This is consistent with the implementer's documented linking
rule but stylistically inconsistent with the rest of the APPROVED
rows. See Minor #1 below for the discretionary callout.

## Verification — Minor #3 (ARCHITECTURE parentheticals)

`ARCHITECTURE.md:143-144` now reads:

> | `shared.services.metrics.record_metric` (in `packages/shared/`) | ... | B.4 |
> | `entity_filtering.resolution.orphan_connector` (in `pipelines/entity-filtering/`) | ... | B.5a / B.5b |

The symmetric treatment (both deviating rows get parentheticals; the
four `apps/`-resident rows stay bare under the "unless noted
otherwise" caveat) is the right call. A reader scanning the table can
now see at a glance which services live where without having to
cross-reference the workspace layout.

## Verification — Minor #4 (FEATURE_ROADMAP duplicate sentence)

Before: the banner included BOTH the opening prose ("Multi-schema KG
extraction, schema-edit UX, telemetry, orphan-lifecycle, cross-
notebook merge") AND a redundant "**One-line summary**: Track B
COMPLETE ... multi-schema KG extraction, schema-edit UX, telemetry,
orphan-lifecycle, cross-notebook merge" trailer (copy-paste residue).

After: the opening prose stands alone. The substantive content is
preserved; no information lost.

```
$ grep -n "One-line summary" docs/FEATURE_ROADMAP.md
(returns 0 lines)
```

## Verification — no code changes

```
$ git diff --stat main...origin/track/b-integration-retro -- '*.py' '*.ts' '*.tsx' '*.js' '*.jsx'
(returns 0 lines)
```

Full diff (`git diff --stat 7d8096c..c14892a`) confirms attempt-2
touched only 5 files: ARCHITECTURE.md (+4/-4 lines), CHANGELOG.md
(+1/-1), FEATURE_ROADMAP.md (+5/-8), RETRO.md (+62/-48 — mostly
table rewrites for the hyperlinks), status.md (+30/-1), plus the new
self-review file (+135). No business logic touched.

## Verification — self-review attempt-2 entry

`docs/tracks/B-kg-quality/reviews/phase-B.7-self-review.md` exists
(135 lines). Structure:

- Attempt 1 section (lines 7-29): self-claim table for ACs + reviewer
  decision summary.
- Attempt 2 fixes section (lines 31-113): per-issue breakdown —
  Major #1 (with site-by-site delta list and verification command),
  Minor #1 (with code-path cross-reference), Minor #2 (with linking-
  rule documentation), Minor #3 (with the symmetry decision call-
  out), Minor #4 (with verification command).
- Final verification section (lines 115-128): git-diff-stat snapshot.
- Ready-for-review marker (lines 130-135).

The self-review accurately reflects the attempt-2 work and would let
a third reviewer (or future-me) reconstruct the fix logic without
diff-spelunking.

## Issues found

### 🔴 Blockers (must fix)

(none)

### 🟡 Major (must fix)

(none)

### 🔵 Minor (optional, file as track-C+ follow-up if desired)

1. **B.1b row in RETRO attempt-count table is unlinked.** The B.1a
   row treats "no attempt-1 file exists" by linking the parenthetical
   "(See [self-review](./reviews/phase-B.1a-self-review.md).)" onto
   the row. B.1b has the same situation (only `phase-B.1b-self-
   review.md` exists; no `phase-B.1b-attempt-1.md`) but the row is
   left as a bare "APPROVED." with no link. For full table-wide
   linking symmetry this could read "APPROVED. (See
   [self-review](./reviews/phase-B.1b-self-review.md).)" Trivial
   stylistic nit — not worth re-opening the PR but worth noting for
   the Track-C+ RETRO templates.

## Kudos

The attempt-2 fixes are exactly what the doctor ordered:

- **The "B.7 itself needed an attempt-2 for arithmetic fixes" self-
  aware caveat in the RETRO closing paragraph** is the right level
  of meta-honesty. It would have been tempting to silently fix the
  arithmetic and leave the closing-paragraph attempt-count clean;
  instead the implementer surfaced their own attempt-2 in the public-
  facing closing. Future-me reading this in 2027 will see the cycle
  played itself in real time.
- **The clarifying blockquote above the attempt-count table** ("...not
  counted in the denominator below") makes the 17-vs-18 reasoning
  explicit at the point a reader is most likely to be confused. This
  is better than I would have asked for in the original review.
- **The CHANGELOG module-path citation** (`pipelines/entity-filtering/
  .../orphan_prune.py::archive_stale_orphans`) goes beyond the minor's
  literal ask of "rename the kwarg" by adding the greppable function
  name. Operators reading the CHANGELOG can now grep the repo for the
  function in question without having to guess the module path.
- **Symmetric ARCHITECTURE parentheticals** (both `metrics` and
  `orphan_connector` get path qualifiers, not just the
  `orphan_connector` row the review called out) — the implementer
  read the spirit of the minor (consistency) rather than just the
  letter (one specific row).
- **The attempt-2 status.md entry** documents not just the fixes but
  links to the self-review for the verification commands. This is the
  hand-off form a future implementer needs.

## Decision rationale

Every issue from attempt 1 closed. Zero new issues introduced. No
code changes (verified by diff-stat). Self-review file lands the fix
log with verification commands that would let a third reviewer
reproduce the checks. The single remaining minor (B.1b row link
symmetry) is one-bullet-point trivia, doesn't affect the doc's
substantive content, and would be a discretionary polish at best.

Under the user's "HIGH quality bar for docs" direction, this attempt
meets the bar.

## Next steps

APPROVED. **Track B is officially DONE.** Ready for human approval
and merge to main.

The RETRO is now ready to inform Tracks C-G's planning. Tracks C
implementers should read at minimum:

- RETRO §What worked / What hurt for testcontainers + adversarial-
  cycle ROI signals.
- RETRO Recommendation #1 (rejection rate ~47% → double your "code
  time" estimate to get "merged time").
- RETRO Recommendation #5 (B.0 testcontainers FIRST for SCHEMAFULL
  table work).
- RETRO Recommendation #6 (inversion test pattern for central
  regression tests).
- RETRO Recommendation #7 (telemetry-first; land `metrics` event
  types in the feature migration, not a follow-up).
