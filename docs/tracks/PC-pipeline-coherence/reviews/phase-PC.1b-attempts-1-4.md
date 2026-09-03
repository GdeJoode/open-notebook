# Phase PC.1b — attempts 1–3

- **Branch**: `feature/track-pc1b-derived-state`
- **Reviewed commits**: `12f0056f`, `cabcf624`, `70d840b3`
- **Date**: 2026-09-03

The phase that turns "a producer names its consumer" from a convention into an
enforced rule. The review's own summary is the fairest description of the outcome:
*the conclusion survives, the evidence as written does not support it as stated,
and the instrument built on it cannot enforce the rule it declares.*

## What was verified and held

`kg_resolution_report` and `validation_report` genuinely have zero readers —
checked across every file type, not only Python. No `getattr`, no dict access, no
serialisation path, no frontend, no MCP tool. The two deleted fields were equally
dead. **The deletions broke nothing and the central argument stands.**

## The three blockers

**B1 — the CI job was red on its first run.** The whole structural claim of this
phase is "a rule that runs nowhere is not a rule". The reviewer ran the workflow's
two commands in an isolated environment and collection died: `uv sync --extra dev`
at the root installs pydantic, loguru and python-dotenv, because
`[tool.uv.sources]` only says *where* workspace members come from *if* something
depends on them. `tests/` imports six packages that a bare sync does not install.
`db-integration.yml` survives the identical line only because it sets a
`working-directory`. Fixed with `--all-packages`, verified in a scratch env.

**B2 — the guard passed with genuinely dead fields, three ways.** This is the one
I had explicitly asked the reviewer to attack, and all three plants were green:

| planted | why it passed |
|---|---|
| a field named `metrics` | `_reader_count` matched `.field` anywhere; `confidence` has 27 "readers", `metrics` 9 |
| `class ResolvedResult(FilteredResult)` with a dead field | the scan filtered on a hard-coded pair of class names |
| one line: `result.zzz_orphan_writeonly_report = {"n": 1}` | the counter counted **writes as readers** |

The third is decisive and it exposes why the first run *appeared* to work: the
four orphans it found were found incidentally, because all four were written
through constructor kwargs, which carry no leading dot. Counting is now AST-based
over `ast.Attribute` in a **Load** context; inheritance is resolved transitively;
and ambiguous field names are refused outright with a message naming them, since
the guard cannot do type inference and should not pretend to.

**B3 — W3 was filed as WIRE and is not.** Its "waiting consumer" was recorded as
*PC.3's own AC*. A future phase's acceptance criterion is not a reader. By the
inventory's own cut rule that is an ACCEPT row, and it has moved there. The code
stays because PC.3 needs the instrument; the label is what was dishonest. **The
phase broke its own rule in its own new code**, which is the finding worth
carrying forward.

## The majors

- **W3b had no guard at all.** The reviewer reverted the fix and both suites
  stayed green — 1839 and 69, zero failures. My own five-mutant verification had
  simply omitted that boundary. Three tests now, mutation-verified.
- **D1 and D2 were listed as deleted and were not.** The table had no status
  column, so a reader took them as done. Both executed; the table now says which.
- **Four new guards live in package test directories**, which `unit-guards.yml`
  does not run — in the phase whose header names "it runs nowhere" as failure
  mode #1. Recorded rather than fixed: widening the job dies on
  `Plugin already registered under a different name` under `--import-mode=importlib`.
- **The documents overstate.** Two specifics, both mine, both in several files.

## Corrections to my own claims

**"4/4/2/2 readers" was not what I measured.** Those were grep *occurrence*
counts. Each payload field is read in exactly **one** production file. The
contrast is 1-versus-0, not 4-versus-0. It points the same way, and the argument
survives — but I stated a different metric from the one the guard uses, in four
documents, and called both "readers".

**"The underscore is the discard" was wrong.** I wrote that `workflow.py:157`'s
`merged, _decision = await run_multi_schema(...)` was where the soft-nudge
verdict was lost. It was not: `run_multi_schema` already writes the verdict into
`merged.metadata["soft_nudge"]`, and `_emit_soft_nudge` reads exactly that key.
The real defect was a **missing writer** to `notebook_event`. The state was
carried faithfully and nothing acted on it — which supports the thesis better
than my version did, and makes the misstatement gratuitous.

## Findings the reviewer contributed beyond the brief

- `soft_nudge_dismissed` is set to `True` by the API and **nothing ever sets it
  back to `False`**, though a comment claims the orchestrator re-arms it. With
  the new duplicate suppression, one click on "Don't show again" permanently
  silences the wire this phase just built. → PC.5.
- Duplicate suppression hides a worse-coverage document behind a better-coverage
  first event, and the payload keeps the first document's coverage. Latent today
  because the banner renders a static string and never reads the payload; live
  the moment anyone surfaces it. → recorded.
- Deleting the four uncollectable test files was right, but it cost the only
  coverage of `clean_thinking_content` and `parse_thinking_content` in
  `shared/utils/text.py`. Named rather than assumed.
- `test_an_entity_without_merge_tags_is_unchanged` compares `[]` against an
  absent key — both take the same `or []` branch, so it passes identically before
  and after W2.

## Numbers

- Invariant: 11 tests, of which **8 are controls**. The file-list control earned
  its keep on the first run by catching that the exclusion filter checked
  `"/tests/" in path` while the repo's own root suite is `tests/…`, with no
  leading slash.
- Suites after the fixes: app-main 1833 / 6 skipped, root `tests/` 69.


---

# Attempt 2 — REVISIONS_NEEDED (2 blockers, 3 majors)

Blockers 1 and 3 and majors 4, 6, 7 verified fixed. Blocker 2 was **partially**
fixed: the three plants from round 1 died, and three new attacks got through.

| attack | why it passed |
|---|---|
| dead fields named `error`, `score`, `state` | `_AMBIGUOUS_FIELD_NAMES` held 16 hand-picked names against 2733 distinct Load-context attribute names. It denied the two names round 1 happened to plant and missed the three most frequent Loads in the repo — `info` (141 files), `warning` (135), `error` (88), all loguru |
| a subclass declared in ANOTHER package | "inheritance is resolved transitively from the AST" was true only *within* `shared/models/extraction.py`, which is the one file the scan parsed |
| a field written and logged by its own producer | any file with a Load counted, including the writer's. The inventory records exactly this defect elsewhere: `aliases_registered` is initialised and logged, never incremented |

Plus: unparseable files were silently skipped (the reviewer's own broken plant
was dropped and produced a plausible orphan for the wrong reason), and the
correction I had just made to the reader-count claim was **wrong in the other
direction** — "every derived-state field is read in none" contradicted the table
two lines below it, which lists `concept_alignment_report` as having one.

## The observation that mattered more than the count

> Rounds 1 and 2 each closed the specific fields I planted rather than the
> property that let them through. […] A third round that adds `error`/`warning`/
> `info` to the list and parses a second file would be the same move again, and I
> would come back with `output`, `stat` and `page_count`.

That is correct, and it is the finding of this phase.

# Attempt 3 — the guard inverted

All three surviving attacks have one root cause: **the counter attributed a read
by bare attribute name, with no information about what object the attribute
belonged to.** `x.foo` cannot tell you the type of `x` without inference, so a
name-based sweep is either too loose or too strict, and no denylist fixes that.

So the requirement is inverted. The guard no longer searches the repository for
somebody who might read a field. **Every derived-state field must declare its
consumer**, and the guard verifies the declaration:

- `Reads(path)` — the named file must Load the attribute AND not be its only
  writer, which closes the self-read hole by construction rather than by
  denylist.
- `Owned(phase)` — no reader yet, a named phase owns it, and the claim must also
  appear in `handoff-inventory.md`. A promise made in one place is not a promise.

A field with no entry fails, **whatever it is called, wherever it is declared,
and whoever looks at it**. All three round-2 attacks now die for the same reason,
which is the property rather than the instances.

**What it explicitly cannot do**, now stated in the module docstring rather than
implied: it cannot prove the Load it finds refers to *this* model rather than a
same-named attribute elsewhere. It narrows the collision surface from 478 files
to one and makes the claim reviewable. That is as far as name-based checking
honestly goes, and saying so is cheaper than a fourth round.

Two further findings, both from writing the new version:

- The tree contains **two unrelated classes called `ExtractionResult`** — the
  pydantic model and a dataclass in `source_extractor.py` carrying `chunks`,
  `title`, `url`. Resolving inheritance by name alone dragged the second one in,
  and the cross-file scan found it on its first run. The root is now anchored to
  its declaring file; descendants are still matched by name, which is the
  residual imprecision and is documented.
- Parsing is no longer tolerant, and a control asserts every production file
  parses. This matters more than it looks: CI pins Python 3.11 while development
  runs 3.12, so a 3.12-only construct would parse locally and vanish in CI only —
  in the direction that hides readers.

Twelve tests, of which **eight are controls**. Two earned their keep immediately:
the file-list control caught that the exclusion filter tested `"/tests/" in path`
while the repo's own root suite is `tests/…`, and the cross-file scan caught the
duplicate class name.

---

# Attempt 4 — APPROVED

Every attack from rounds 1–3 replayed against the committed tree, plus three new
ones aimed at the round-4 additions. All eight die, and the three `Owned` variants
fail in one run with three distinct messages — the check distinguishes *no phase
named* from *phase named, no evidence*.

| attack | fails on |
|---|---|
| dead fields named `error` / `score` / `state` (Load-read in 88 / 8 / 3 files) | `test_every_derived_state_field_is_declared` |
| a subclass declared in another package | same |
| a field written and logged by its own producer | same |
| an unparseable production file | `test_every_production_file_parses` |
| a bare-string declaration | `test_every_declaration_has_a_verified_shape` |
| `Owned("")` / `Owned("PC.9 …")` / a real phase with no row | `test_every_owned_declaration_appears_in_the_inventory` |

## The residual hole, accepted deliberately

A dead field named identically to an existing inventory row (`metrics`,
`alias_candidates`) satisfies the row check. The reviewer judged this the intended
reviewability boundary rather than a defect, and the reasoning is recorded in the
inventory: the collision set is now the ~15 rows the file contains rather than any
substring in it, each already naming a phase that owns that subject; it requires a
visible `Owned(...)` entry rather than a value that looks like ordinary usage; and
closing it needs semantic knowledge of what a row is about, which is the same wall
as type inference. Stated in the document rather than left for someone to find.

## What this phase is actually about

Three findings earned a place beyond the code.

**Closing planted instances is not closing the property, and the tell is that each
fix names a field rather than a rule.** Rounds 1 and 2 added denylist entries and
parsed one more file; round 3 inverted the question; round 4 closed the
declaration's own escape hatches. Only the last two transfer to anything else. The
reviewer's round-2 sentence is the one to keep: *"a third round that adds
`error`/`warning`/`info` to the list and parses a second file would be the same
move again, and I would come back with `output`, `stat` and `page_count`."*

**A guard's escape hatch is most dangerous when it looks like ordinary usage.** The
bare-string declaration was a blocker specifically because `Dict[str, str]` is what
that same table looked like one commit earlier — so the unchecked path was not
exotic, it was what anyone copying from history would write.

**A correction that leaves the corrected text in place is not a correction**, and
reporting it as done is worse than not doing it. One sentence about reader counts
took three rounds because each round appended its correction below the error
instead of replacing it, and in round 3 I reported it fixed when one of the two
documents was untouched. That converted a review round into a round that verified
nothing.

## A pattern with enough instances to be evidence

Three times in this phase a new control caught, on its **first execution**, the
thing it was written against:

- the file-list control found that the exclusion filter tested `"/tests/" in path`
  while the repo's own root suite is `tests/…`, so root test files were being
  scanned as production code;
- the cross-file class scan found two unrelated classes named `ExtractionResult`;
- the stricter `Owned` row check failed on the phase's own two inventory rows,
  which were vaguer than the declaration claimed.

## Verification at approval

```
apps/app-main               1833 passed,  6 skipped
shared + entity-filtering   1313 passed,  4 skipped
tests/ (local)                75 passed
tests/ (isolated, uv sync --all-packages --extra dev)   75 passed
```
