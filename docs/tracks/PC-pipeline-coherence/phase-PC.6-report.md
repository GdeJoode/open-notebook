# Phase PC.6 — Configuration that expresses one intent

- **Branch**: `feature/track-pc6-config-coherence`
- **Date**: 2026-09-05

The acceptance criterion is negative: **enabling a feature either works or says
why it cannot, and "flag on, zero effect, one warning" must be unreachable.** So
most of this phase is about making things refuse.

## What the phase found that the plan did not

### Two model-configuration systems with opposite privacy defaults

| | `model_routing.yaml` | `default_models` + `route_resolver` |
|---|---|---|
| used by | extraction service, pipeline steps | app-main (judge, chat, agent extraction) |
| keyed by | step × privacy level | `LLMTask` × `PrivacyMode` |
| privacy default | `internal` — local only | **`CLOUD`** — prefer cloud, fall back local |
| state today | every routed local model installed (3 distinct, 9 routes) | **zero rows** |

Each default is defensible alone. Together they mean the same document is treated
as local by one path and cloud-eligible by the other, which is a privacy question
rather than a routing preference.

**Decided (user): keep both, one authoritative per domain, with a bridge check.**
Merging them touches both code paths, the `/api/models/defaults` surface and the
J.4 telemetry, for no behaviour anyone asked for. What was missing was never one
system — it was a check that the two do not contradict each other.

`default_models` holding zero rows is a **legitimate** "not configured yet", not a
defect: models for summaries and transformations arrive later. It is a finding
only when something enabled depends on it.

### Startup seeded model rows for a provider the config had retired

`seed_nim_routes` runs on every boot, and it is where the single NVIDIA `model`
row came from — a configuration contradicting itself once per startup, for a
vendor nothing is allowed to call. Now skipped while the provider is declared
unavailable.

### Validation that reports success without validating

The sharpest instance, and the inventory understated it. PC.1b filed
`validation_report` as "stage 11 is inert". Measured against the real filter,
`OntologyConstraintFilter(ontology=None)` does not merely skip:

```
DEBUG  No ontology provided, skipping validation
report {"total_entities": 1, "valid_entities": 1, "invalid_entities": 0, …}
```

A run with validation on and no ontology is **indistinguishable, in its own
output, from a run that validated everything successfully**. Zero effect is the
defect this phase targets; a success report for a check that never happened is the
same defect with evidence attached.

## What was built

`packages/shared/src/shared/config_coherence.py`. Findings carry a severity and a
**mandatory remedy** — a finding that names a problem without a fix is the log
line this phase exists to abolish. `BLOCK` means an enabled feature cannot do its
job and startup refuses; `WARN` is logged.

**Severity follows reachability.** A `public` route to a retired provider is a
WARN because `default_privacy: internal` never selects it; flip the default and
the same route is a BLOCK. That is what makes keeping the retired routes coherent
rather than sentimental.

Against the shipped configuration: **5 WARN, 0 BLOCK** — it starts. Enable
concept alignment without KG resolution and it does not:

```
[BLOCK] alignment-without-resolution: concept alignment is enabled but KG
        resolution is not; alignment classifies the entities KG resolution marks
        `is_new`, so with resolution off nothing is marked and nothing is
        classified
     → enable kg_resolution, or disable concept alignment
       (ENABLE_CONCEPT_ALIGNMENT=false)
```

NVIDIA is **declared unavailable rather than deleted** (user's decision):
`available: false` plus a reason, its `public` routes kept as the record of what
the cloud path was, and `get_model_config` raising `ProviderUnavailableError` at
**resolution** — the last point where the reason is still known. One step later
the failure is an HTTP error from a vendor, which cannot say the route was retired
on purpose. An explicit `model_override` naming a retired provider still resolves:
the gate is against a route reaching one *silently*, not against a caller who has
decided.

## Two dead knobs, removed from all three places at once

`OLLAMA_MODEL` (from `EXTRACTION_MODEL`) and `OLLAMA_NUM_CTX` (from
`EXTRACTION_NUM_CTX`) were each assigned once in `services/extraction/api.py` and
used nowhere. Extraction resolves through `model_routing.call_llm`.

Worse than dead code: a knob set in `docker-compose.yml` and read into a constant
reads as a working control, so the compose file documented a configuration the
service did not have. **It cost a fully reverted branch** — `EXTRACTION_MODEL` was
repointed, the container restarted, and the resolver went on returning
`llama3.1:8b-instruct-q4_0`.

Removed from the code, the Dockerfile `ENV` defaults and the compose entries
together, so no half-trail remains, each replaced by a line naming the real lever.
Verified after rebuild: both absent from the container env, service healthy,
resolver reporting `ollama / llama3.1:8b-instruct-q4_0 / num_ctx 8192`.

The guard is the **class** — though round 1 shipped it scanning only `api.py`
while a live instance sat one file over (`entity_validator.py` bound `OLLAMA_URL`
and never read it), and it missed six spellings including `os.environ.get`. It now
scans every module in the service, handles all six, and is joined by
`tests/test_compose_env_is_consumed.py` for the half it structurally cannot see:
a name that appears in `docker-compose.yml` and nowhere else.

## Two claims in the plan that measurement disproved

Corrected in `plan.md` rather than left to be inherited as fact. **The first of
these took two attempts, and the first attempt is not reproduced here** — a
correction that leaves the corrected text standing is not a correction; see the
round-2 section below for what the measurement actually shows.

**The Ollama `num_ctx` paragraph (R4b)** claimed esperanto never sends `num_ctx`
and that every long prompt is therefore truncated tail-first. Two of its three
parts are wrong and one is right, and the settled version is in **Round 2** below.
What holds: truncation discards the **head**, not the tail — verified with markers
at both ends — and extraction is not truncating, with a ~4,530-token worst case
against the 8,192 its caller does send.

**No router fallback resolves to `gemma2`.** Both are `llama3.1:8b*`
(`model_routing.py:184`, `route_resolver._DEFAULT_PROVIDER_MODEL_NAME`). An
earlier draft of this correction over-reached to "`gemma` appears nowhere in the
repository", which is false — it is in two docs and a test fixture, and
`ai-models.md:441` listing `gemma3` for transformations is the likely origin of
the original claim.

Every routed local model is installed, including `granite3.2-vision`. That is
**three distinct models across nine ollama routes** — the "eleven" in an earlier
draft was the number of tags pulled in Ollama, a different quantity. An earlier
probe of mine reported `granite3.2-vision` missing because the model list had been
truncated by `head`; caught by re-checking before reporting.

## Three inventory rows reassigned to PC.5, with the measurement

- **`metrics` rows nothing reads** — only `routing.served` is read
  (`model_routes.py`); `export.jsonl` and `export.obsidian` are written and read
  nowhere.
- **Alignment report keys dropped at the `filtering_stats` copy.**
- **`_save_result` storing pre-filter entities beside post-filter stats.**

All three are about what a completed run *reports to a human*, which is PC.5's
remit, not configuration coherence. Reassigning rows out of the phase that owned
them needs its reason on the record: deciding them here, without the surface that
would show whether the decision was right, is exactly how `FilteredResult`
acquired four fields nobody reads.

**Round 1 of this reassignment did not reach PC.5**, which review established by
reading the diff rather than arguing: only PC.6's section had been edited, so the
rows sat in a table with a forwarding address while PC.5's scope and AC covered
none of them. All three are now in PC.5's bullet list with its AC widened —
together with review's pushback that the `metrics` row has nothing to do with a
curator door and needs no surface to decide.

## Round 2 — what review disproved

Attempt 1 returned REVISIONS_NEEDED with three blockers and nine majors, and each
blocker contradicted a claim above.

**The phase's sharpest new BLOCK was itself unreachable.** `collect_findings`'s
single production caller never passed `ontology_validation_enabled` or
`ontology_supplied`, so the finding was structurally impossible and the inventory's
"**Made unreachable**" was true of the guard, not the behaviour. Built,
unit-tested, never called — the same shape as the M6 mutation this report credits
the phase with catching, one layer further out. The check now lives in
`FilteringWorkflow.__init__`, the only place holding both facts.

**`available: false` did nothing on the second of the two systems the phase exists
to bridge.** `RouteResolver` filtered on API-key presence alone, so NVIDIA —
retired in the YAML, key still in `.env`, which is exactly what a retired vendor
leaves behind — stayed at the HEAD of the CLOUD chain for chat, the judge and
agent extraction. Declaration on, zero effect on half the system: the acceptance
criterion verbatim, failed by the phase's own mechanism.

**A partially-pulled Ollama hard-refused startup** for steps an operator may never
use, and worst in the common case — no Ollama at all degraded to one WARN while
Ollama with *some* models refused. `model-not-installed` is now BLOCK only for
`REQUIRED_STEPS`.

### The class was wider than the phase surveyed

The AC says "unreachable", and attempt 1 had closed the one case the plan named
plus a judge case. Six more, measured on the shipped defaults:

| flag | why it did nothing |
|---|---|
| `semantic.entity_linking_enabled` | `linking_provider` defaults to `"none"`, so stage 8 never runs |
| `outlier_detection_enabled` | outlier classification belongs to the graph analyser, built only when centrality is on |
| `llm_verification.*` | a whole sub-config, five flags, **zero** readers |
| `kg_resolution.match_strategy` | never passed to `KGResolver` |
| `treekg_enabled` / `raptor_enabled` | no consumer |
| `EdgePredictionConfig` | `EdgePredictor()` built with no config, so every weight was ignored while the flag was honoured |

The first two refuse. Four are deleted — each existed only in `config.py` and in a
test asserting its default, which is what let them survive: a test pinning dead
code rather than guarding behaviour. `EdgePredictionConfig` is wired instead, and
that also makes the documented defaults the ones actually used, since the
constructor's fallbacks differed.

### And the correction in this report was itself wrong

The `num_ctx` paragraph said both callers send it. Measured through the real
esperanto factory: `num_ctx` appears **zero times** in the package and is filtered
by `get_completion_kwargs` before the Ollama provider is reached. So the two
callers differ — `_call_ollama` sends it, esperanto cannot — which means the M.4
guard in `llm_call.py` was inert and every app-main LLM call runs at Ollama's 4096
default. `check_ollama_context` reports a model row promising more than its model
bakes; the remedy is a `PARAMETER num_ctx` variant, which is *not* inert on that
path, contrary to what the first correction claimed.

## The habit behind the recurrence — named by review, and the fix

Three rounds each found a guard of this phase that could not fail for the case it
was written for. That is one habit, not five bugs, and review stated it precisely:

> the generalisation was asserted in prose and sampled in a test

* the remedy guard's docstring said "it holds for findings added later"; the test
  ran one `collect_findings` call;
* the AST guard's docstring said "this guard is the class"; the code matched one
  spelling in one file;
* the compose guard's docstring said "a name that appears there and nowhere else";
  the scan matched bare tokens and read its own sources;
* the context check's comment said the packer's promise is verified; nothing
  asserted the call site.

And the *survey* had the same shape: `check_feature_dependencies` enumerates the
flag pairs someone thought of. Round 2 closed six; two more turned up in the
dataclasses; round 3 closed those; a ninth turned up in the stage graph —
`semantic_blocking` builds a UMAP/HDBSCAN blocker whose only use sits behind the
LLM matcher's gate.

**What fixed each one was the same move**: stop exhibiting members of the space
and derive it. The remedy guard became sound by walking `Finding(...)`
constructions by AST; the compose guard by matching read *contexts* and excluding
its own sources; the AST guard when `_flatten` derived module scope from the tree
rather than from `tree.body`.

So the ninth case is fixed with a pair, and the habit is fixed with an
enumeration. `pipelines/entity-filtering/tests/test_stage_graph_enumeration.py`
reads `FilteringWorkflow.__init__` by AST, finds every `self._X` assigned under a
condition, and asserts each is read elsewhere in the class. It detects 11 stages,
and mutation — renaming a stage's readers OUTSIDE `__init__`, leaving construction
intact — confirms it flags all four historical findings:

```
readers of _semantic_blocker removed -> ['_semantic_blocker']
readers of _entity_linker removed    -> ['_entity_linker']
readers of _graph_analyzer removed   -> ['_graph_analyzer']
readers of _llm_matcher removed      -> ['_llm_matcher']
```

What that enumeration actually asserts is narrower than the paragraph above it
first claimed, and the claim is retracted in the file itself: it holds that
nothing is constructed the class cannot reach. Replayed against `main` it finds
none of the four historical findings, because in each the attribute IS read — the
defect was that the reader sits on an unreachable path, or that the construction
yields None. The guard still earns its place; the sentence claiming it would have
pre-empted four review rounds did not. Excluding `__init__` from the
reader scan is load-bearing: a stage's own construction reads the attribute it
just set, so counting those would make every stage its own consumer, which is
PC.1b's four-round trap in a new costume.

**The transferable rule has two halves, and the second is the one that caught the
fourth instance.** Both belong in the track's standing conventions rather than in
one phase's report.

> **1. Derive the space, do not sample it.** Do not write "this guard is the
> class" until the guard derives the class from the code — by AST, by reflection,
> by walking the config — rather than by exhibiting members of it.
>
> **2. Verify a guard by putting it in the state it claims to prevent, and
> confirm it fails.** Not a state of that shape — *the* state. If the guard is
> retroactive, replay the commit. If it is a pattern, feed it the literal source
> line. If it is a wiring assertion, delete the call.

The first half governs a guard's shape; it says nothing about how you check a
guard once built, and all four failures were verification failures rather than
design failures. In every case the guard was tested against a state that was
**already true for another reason**:

* the remedy check enumerated one input's output instead of the constructor space;
* the compose scan matched tokens instead of read contexts;
* the context check had no assertion on its call site;
* the constant pattern was verified against an aggregate that three unrelated
  files already satisfied.

**Two instances of that happened in this phase after the failure mode was named
out loud**, which is the evidence that half one is necessary and not sufficient.
The stage-graph enumeration's own docstring claimed it "would have found all four
in one pass"; replaying the detector against `main`, where all four defects were
live, finds **zero** — the defect was never "no reader", and one of the four is
not a `self._X` at all. The mutation that appeared to prove the claim deleted a
stage's readers, i.e. tested the shape the guard checks rather than the state it
claimed to catch. One command settles it: `git show main:workflow.py`.

The same round produced a compose-guard pattern that matched nothing in the
repository, under a test asserting `"GROBID_URL" in _referenced_names(...)` — true
already, because three unrelated files spell the name literally. The test that
settles that one is `re.search(pattern, '_GROBID_URL_ENV = "GROBID_URL"')`.

## Mutation testing

Eight mutations. Six caught on the first run; **two survived, and both are this
track's recurring defect rather than edge cases.**

| # | mutation | caught by |
|---|---|---|
| M1 | the alignment/resolution check removed | 2 |
| M2 | its BLOCK downgraded to WARN | 2 |
| M3 | `raise_if_blocking` neutered | 1 |
| M4 | an unreachable runtime treated as an empty model set | 1 |
| M5 | severity ignoring reachability | 1 |
| M6 | the startup call deleted | **nothing**, then 1 |
| M6b | the call kept but wrapped in a best-effort `try/except` | 1 |
| M7 | the provider-availability gate removed | **nothing**, then 1 |
| M8 | a dead env constant re-planted | 1 |
| M13 | the context check silenced (round 2) | 1 |
| M14 | a missing bake treated as unlimited (round 2) | 1 |
| M15 | `REQUIRED_STEPS` emptied (round 2) | 1 |
| M16 | required-step gating ignored (round 2) | 1 |
| M17 | remedy blanked on a path no test exercised (round 2) | 1 |

**M6 — the wiring, not the function.** Deleting
`await _check_configuration_coherence()` from the lifespan left all three startup
tests green, because every one of them called the function directly. Built,
tested, never called: the same shape as PC.2's blocker, one layer down. The
replacement test drives the real `lifespan` with a sentinel and asserts both that
the call site exists **and** that a refusal propagates — M6b is why the second
half is not redundant, since the best-effort `try/except` that wraps the seeds
directly below would silence every BLOCK while leaving the call in place.

**M7 — the availability gate had no test at all.** I verified
`ProviderUnavailableError` by hand in a shell and never wrote one.

## Live verification

- The startup check run against the real configuration: 5 WARN, 0 BLOCK, startup
  proceeds.
- The extraction container rebuilt without the dead `ENV`s, healthy, resolver
  confirmed from inside the container.
- All eleven routed local models confirmed present against `/api/tags`.

The `staging` database is empty (cleared for real data), so no end-to-end
extraction was run. Everything above is either a live probe against the running
containers or a unit-level measurement; nothing rests on a corpus.
