# Phase PC.1 — attempts 1–5 — VERDICT: APPROVED (attempt 5)

- **Branch**: `feature/track-pc1-curator-queue` — merged as `0786e0a6`
- **Commits**: `83befe00` → `79541156` → `c34ce930` → `78399e22` → `6538806b` →
  `22a7e440` → `15994a5c` → `c88c79c6` → `74b6b39d` → `dc4a9e88` → `303575f1`
- **Date**: 2026-09-03

The curator queue's missing writer. One blocker per round for three rounds, one
major in the fourth, and an escalation to the user in between. Roughly twenty
mutants at the end, all dead.

## What shipped

1. **The three writers** — `ensure_row`, `merge_pending_extensions` and
   `set_coverage_pct` on `NotebookSchemaRepository`. Dedup is case-insensitive
   against pending, accepted AND `excluded_types`; a name that cannot survive
   the accept/reject route as a path segment is refused rather than queued
   unactionable.
2. **The row is created BEFORE schema detection.** That ordering is the fix, not
   a detail — see G0b below.
3. **The row declares no base ontology.** The schema is established per
   document; the notebook's own history is consulted only when that fails.
4. **The applicability sample spans the document** — 40 windows sharing a 40000
   character budget — instead of its first chunk.
5. **Coverage** is the mean over sources of each source's best schema within its
   newest RUN, using a `run_id` the orchestrator now stamps into the FLEXIBLE
   `pass1_metadata`.

## G0b — the finding that could not have been read out of the code

The first implementation was fully tested, fully green and **completely inert on
real data**. It created the row inside `_record_pass1_outcome`, which sits after
`_run_multi_schema`'s early return to the legacy path — and that early return is
exactly what happens when no schema row exists. The cycle:

> no row → nothing forced → detection finds nothing → legacy path → Pass 1 never
> runs → no proposals → nothing creates the row

Live at the time: 17 `pass1_results` rows carrying 111 proposals across 79
distinct type names, and **zero** `notebook_schema` rows. Every writer,
including this phase's own, correctly reported "no row" and did nothing.

**Binding for the track**: a fix for a "nothing ever writes X" finding must be
proven on data where X is genuinely absent, not only on a fixture that says so.

## The blockers

**B1 (attempt 1) — both new repository methods could be reduced to no-ops with
every runnable test green.** The roundtrip tests were docker-gated. 24
docker-free tests now exercise the real repository with only its two DB calls
mocked; eight of ten mutants died immediately, and the two that did not became
MAJOR 1 of round 2.

**B2 (attempt 2) — the row was created after detection.** See G0b. Moving the
call before `_apply_notebook_schema_default` means the first document a notebook
ever sees gets a Pass-1 pass instead of the legacy path.

**B3 (attempt 3) — `config.ontology_name` was persisted as the notebook's base
ontology.** That parameter defaults to `"general"` and nothing in the product
sets it. Three consequences, none intended by a phase scoped to "give the queue
a writer": it overrode the `"scholarly"` every read path falls back to;
`_apply_notebook_schema_default` then forced `"general"` into every later run's
applied set at 0.85, measurably re-typing entities (`Person`: concept → person),
which is PC.4's decision to make; and the schema TTL download would 500, because
`general.yaml` uses a shape `load_yaml_ontology` cannot parse.

Not hypothetical: the live row already carried `base_ontology: general`, written
by this phase's own verification run, and that notebook's TTL export already
raised `AttributeError`. Repaired in place, queue of 79 and coverage 0.508
intact.

**Binding for the track**: per-request state must not set per-notebook state.
The two look alike at a call site and diverge everywhere else.

**B4 (attempt 4, escalated) — the replacement constant was wrong too.**
`DEFAULT_BASE_ONTOLOGY` is `"scholarly"`, chosen for the TTL-export READ path.
Forced into extraction it would have run Dutch Regio-Deal convenanten against
`Article`, `Author` and `PreprintServer`. Measured: `scholarly` is detected for
**zero** of the fourteen live sources, and all 17 `pass1_results` rows ran
against `policy_themes`. The reviewer escalated rather than deciding.

## The user's decision, and what it exposed

> "Dat schema moet per document worden vastgesteld. Als het echt niet lukt dan
> fallback op meest voorkomende in notebook."

Removing the forced base to let detection decide revealed that **detection did
not work**. Applicability was scored against the first chunk capped at 2000
characters, described in the code as "a cheap signal". In this corpus the first
chunk of a parsed PDF is a title fragment with a **median length of 66
characters** — the scorer was matching a document's vocabulary against its cover
page.

| sample | documents detecting a schema |
|---|---|
| first chunk, 2000 chars (production) | **2 of 14** |
| 40 windows spread across the document | **13 of 14** |

The forced base had been compensating for a blind detector. A document that
detects nothing takes the legacy single-schema path where Pass 1 never runs —
so the queue this phase exists to fill was empty for reasons that had nothing to
do with the queue.

Both sample constants are swept, not chosen. 13/14 is the ceiling; 40000/40 was
taken over 20000/60 because its windows are three times wider and the scorer
matches multi-word phrases as substrings, so every boundary can cut one in half.

## MAJOR (round 4) — the head bias came back inside its own fix

The budget was spent front to back rather than shared, and that break fires in
ascending index order. On long-chunk documents the spread stopped partway
through: **17 of 40 windows at 1500-character chunks, the document scored on its
first 41%** — the same cover-page bias, an order of magnitude larger.

The two tests around it could not see it. One used 9-character chunks where the
budget never binds; the other used 5000-character chunks but asserted only the
LENGTH of the result. Between them they touched the failing input and the
failing property and never at the same time.

## Guards that could not fail — three found, in three different shapes

1. **A control-character rule pinned by a string with no control character.**
   The test literal was `"Bell\\x07"` — six source characters — refused by the
   BACKSLASH rule. Deleting the control-character rule left the file green.
2. **A rejection test containing no rejection**, and then a rejection through
   `reject_pending_extension`, which has no production caller (the router uses
   `SchemaEditService.reject_extension`, keyed on `type_name`). Its promise to
   fail "the day a rejection leaves a trace" held only if PC.5 wrote that trace
   through the dead method. There are now two guards, one per layer.
3. **A correct unit with no seam behind it.** `_applicability_sample` was tested
   directly; nothing pinned that `_run_multi_schema` calls it. Reverting the call
   site left every test green.

**Binding for the track**: a correct unit with no seam behind it is not a guard.

## Corrections made to my own reported numbers

- I told the reviewer my measurement showed 14/14 documents detecting nothing,
  contradicting their 8/14. Mine was wrong: the query compared record ids as
  strings, so I was scoring against an empty string. The 2/14 in this report is
  measured with real text.
- I claimed "the applied set is unchanged from before PC.1". It was not; the
  reviewer measured the change and was right.
- A mutant I reported as killed was not faithful to the rule it claimed to
  restore. Restoring the exact previous implementation was what proved the
  coverage test.

## Carried out of PC.1

- A durable "no" for rejected proposals, and the `_ensure_schema_row` rename →
  **PC.5**.
- The read-modify-write race, now reachable because PC.1 is the first production
  caller → **PC.6**.
- Ollama truncating prompts at 4096 tokens while `num_ctx` is never sent →
  **PC.6** (review finding R4b).
- 11 of 14 sources move from the legacy single-schema path to multi-schema:
  more LLM calls per document and a different result shape. The point of the
  phase, but a real change to every ingest.
