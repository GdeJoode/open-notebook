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
| state today | all 11 routed local models installed | **zero rows** |

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

The guard is the **class**: an AST test failing when any module constant in
`api.py` is assigned from the environment and never read, with a walker control
and a mutant control.

## Two claims in the plan that measurement disproved

Corrected in `plan.md` rather than left to be inherited as fact.

**The Ollama `num_ctx` paragraph (R4b).** It said esperanto never sends `num_ctx`,
so every long prompt is truncated and the JSON instructions in the tail are cut.
Measured — during an adversarial review that reverted a branch built on it:

- `num_ctx` **is** sent unconditionally by both callers, and a request value
  *overrides* a Modelfile `PARAMETER`, so baking context into a model variant is
  inert on this path;
- truncation discards the **head**, not the tail — instructions at the end
  survive, document content is what is lost;
- extraction is not truncating at all: ~4,530-token worst case against 8,192 sent.

The genuine finding underneath is narrower and stays with PC.6: nothing checks
that a step's `num_ctx` is large enough for the prompts that step builds.

**The `gemma2` fallback.** `gemma` appears nowhere in the repository. All eleven
routed local models are installed, including `granite3.2-vision` — which an
earlier probe of mine reported as missing because the model list was truncated by
`head`, caught by re-checking before reporting.

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
