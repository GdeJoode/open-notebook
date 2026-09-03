# Phase PC.1b — attempt 1 — VERDICT: REVISIONS_NEEDED (3 blockers, 5 majors)

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
