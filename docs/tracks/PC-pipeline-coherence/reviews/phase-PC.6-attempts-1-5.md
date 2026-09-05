# Phase PC.6 — attempts 1–5

- **Branch**: `feature/track-pc6-config-coherence`
- **Outcome**: REVISIONS_NEEDED ×3 → spot-check (2 rejections) → **technical sign-off**
- **Trajectory**: 3 blockers + 9 majors → 2 + 5 → 0 + 1 → 0 + 0 → 0 + 0
- **Date**: 2026-09-05

The phase's acceptance criterion is negative — *enabling a feature either works or
says why it cannot, and "flag on, zero effect, one warning" must be unreachable* —
and five rounds were mostly about discovering that my own instruments had that
same defect.

> **Correction (2026-09-05).** This document said the working corpus had been
> emptied and its figures could no longer be re-measured. That was wrong. The
> `open_notebook/staging` database holds the corpus it always did — 14 sources,
> 3,824 chunks, 5,501 entities, 68 of them naming a Regio Deal, ingested
> 20 June – 1 July. The zero-row reading behind the claim was taken in the window
> after Docker Desktop restarted and the SurrealDB container came back without
> its volume; a single measurement on a just-restarted stack was treated as a
> fact about the data. Every figure below is therefore re-measurable, and the
> graph as it stands is the before-state for PC.3's acceptance criterion —
> the corpus built WITHOUT cross-document resolution.
## What review disproved, round by round

Each blocker contradicted something the phase report asserted as done.

**Round 1.** The phase's sharpest new BLOCK, `validation-without-ontology`, was
itself unreachable: `collect_findings`' single production caller never passed the
arguments that could produce it. The inventory said "**made unreachable**" of the
*behaviour*; it was true only of the guard. Separately, `available: false` — the
whole mechanism for retiring NVIDIA — had zero effect on the second of the two
configuration systems the phase exists to bridge, so a retired vendor whose key
was still in `.env` stayed at the head of the CLOUD chain. And a partially-pulled
Ollama hard-refused startup for steps an operator may never use, worst in the
common case: no Ollama at all degraded to one WARN.

**Round 2.** The refusal moved into `FilteringWorkflow.__init__` landed inside a
pre-existing `except Exception` at the main production call site, and
`ConfigurationError` subclasses `RuntimeError`. The outcome was **worse than the
defect it replaced**: the entire filtering stage dropped — dedup, fuzzy,
embedding, edge prediction, alignment — while the extraction completed and
reported success, reachable by flipping an env flag on a running server. The
unreachability had not moved; it had been re-created one frame up.

The seventh flag-on-zero-effect case was also found, and it was the only one **on
by default**: `orphan_connector.enabled=True` with three collaborators no
production call site passes, skipped on every run behind a warning an earlier
review had added as an improvement.

**Round 3.** A ninth case, `semantic_blocking` building a UMAP/HDBSCAN blocker
whose only use sits behind the LLM matcher's gate.

**Rounds 4–5.** Two of my own fixes rejected — see below.

## The correction I owed the user

My own correction of the plan's `num_ctx` paragraph was wrong on its load-bearing
half. Measured through the real esperanto factory:

```
in   config={"num_ctx": 16384, "temperature": 0.1, "max_tokens": 8}
out  {"options": {"num_predict": 8, "temperature": 0.1, "top_p": 0.9}}
```

`num_ctx` appears **zero times** in the package. So the two callers differ —
`_call_ollama` sends it, esperanto cannot — the M.4 guard in `llm_call.py` was
inert, every app-main LLM call runs at Ollama's 4096 default, and the model-variant
approach is *not* inert on that path. A branch had been reverted the day before on
a premise true for extraction and false for app-main.

## The habit, and why naming it was not enough

Four guards built in this phase could not fail for the case they were written for:
the remedy check, the compose scan, the context-check wiring, and the constant
pattern. Review's diagnosis:

> the generalisation was asserted in prose and sampled in a test

The docstring claimed the space; the test exhibited one point in it. The *survey*
had the same shape — `check_feature_dependencies` enumerated the flag pairs
someone had thought of, which is why there was always another case.

What fixed each was the same move: stop exhibiting members, derive the space. The
remedy guard by walking `Finding(...)` constructions by AST; the compose guard by
matching read *contexts*; the AST guard when `_flatten` derived module scope from
the tree rather than `tree.body`.

**But two of the four happened after the failure mode was named out loud**, one of
them inside the commit that named it. That is the evidence for a second half:

> **1. Derive the space, do not sample it.** Do not write "this guard is the
> class" until the guard derives the class from the code.
>
> **2. Verify a guard by putting it in the state it claims to prevent, and
> confirm it fails.** Not a state of that shape — *the* state. Replay the commit;
> feed the pattern the literal source line; delete the call.

Half one governs a guard's shape and says nothing about checking one once built,
and all four failures were verification failures: each tested against a state
already true for another reason.

The two rejections in rounds 4–5 are the clearest illustration. The stage-graph
enumeration's docstring claimed it "would have found all four in one pass";
replayed against `main`, where all four defects were live, it finds **zero** — the
attribute is read in every case, and one of the four is not a `self._X` at all. My
mutation deleted a stage's readers, testing the shape the guard checks rather than
the state it claimed to catch. One command settles it. And the compose guard's new
constant pattern matched nothing in the repository while its test passed, because
`"GROBID_URL" in _referenced_names(...)` was already true through three unrelated
files.

## What holds, verified against running code

Review confirmed each of these against the live system rather than the report:

- the refusal reaches both production call sites and propagates to a failed,
  resumable job carrying the finding's message;
- retirement holds across both configuration systems — a retired provider is
  dropped from the CLOUD chain even with its key present;
- the surfaces say why: `POST /extract` answers **503** with the reason, and
  `/routing` marks retired routes `[UNAVAILABLE]`;
- the dead knobs are gone from code, Dockerfile and compose together, verified in
  the rebuilt container;
- against the shipped configuration: **5 WARN, 0 BLOCK** — it starts.

## Carried forward

- **Nothing was verified end-to-end against a corpus.** `staging` was emptied for
  real data, so every claim rests on live probes or unit-level measurement. The
  first real extraction after this merges is the first real test of it.
- Minor 9 — 11 unread config fields in `pipelines/summarization`, one in
  `ontology-manager` — recorded in the inventory as out of scope. Whoever picks it
  up should use the enumeration approach from the start rather than pair-listing.
- Residual AST-guard limits (`from os import environ as e`, `os.environ.copy()`,
  module-level `for` targets, augmented assignment) are declared in the guard's
  own docstring rather than claimed.

## Note on authority

The reviewer signed off **technically** and said plainly that a reviewer's verdict
is not the repository owner's consent: *"I sign off technically, and a human
authorises the merge."* That is the correct division and is recorded here because
the temptation to read a green review as permission is exactly what five rounds of
this phase argue against.
