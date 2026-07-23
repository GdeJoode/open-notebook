# Design note — teacher-bootstrap for schemas, prompts and seed labels

> **Status**: idea captured 2026-07-24, **not scheduled**. Companion to
> [`design-thematic-classification.md`](./design-thematic-classification.md) (the
> cascade this bootstraps) and [`purview-lessons.md`](./purview-lessons.md).
>
> Origin: a by-the-way question — *can we use NotebookLM as a "teacher" for
> open-notebook's schemas and prompts?* Captured here deliberately, because the
> question itself was lost once already (a bare `/btw` submits no arguments, so the
> question never reached the assistant). Design notes survive; chat does not.

## The problem it addresses

Entity typing is the standing quality gap: **27% `other` + 44% generic
`concept`/`topic`**. The causes are known — free-form LLM typing, and a cold-start
where there are no labeled examples to do anything cheaper or more stable. The
thematic-classification cascade solves the *runtime* side (embeddings + a light
classifier, LLM only on the margin), but its best tier needs data:

> "Mature / enough labels → **Trained classifier** (Purview-style, ≥~50 pos/theme)"

Getting those ~50 positives per theme is exactly the cold-start problem. A **teacher**
is how you skip it.

## The idea

Use a strong model as an offline **teacher** over our own corpus to produce
high-quality reference output, then use that output three ways — all **offline /
bootstrap**, never in the runtime path.

### Role 1 — Seed labels for the classifier cascade ⭐ strongest fit
Teacher labels N documents per theme → those become the labeled examples that move
the cascade from *zero-shot* → *kNN* → *trained classifier*. Directly unblocks the
tier that `design-thematic-classification.md` says is "fastest + most stable once
data exists". This is the concrete "train a model" path for typing.

### Role 2 — Schema induction (Track B / Q8)
Teacher proposes entity types, relations and policy themes over the Regio Deal
corpus → **curated by a human** into the small managed label ontology → mapped
through the Track L canonical bridge. Q8 already made schemas a reviewable
artifact with view+edit, so there is a natural landing place; this fills it with a
data-driven starting point instead of hand-authored guesses.

### Role 3 — Prompt distillation as an eval set (Track M)
Teacher produces the ideal structured extraction for a handful of documents → commit
that as a **gold/eval set** → iterate the open-notebook extraction prompt until the
student (local or NIM) reproduces it. Converts prompt work from taste into a
**measurable gap metric**.

## Which teacher — NotebookLM is the wrong tool for the pipeline

The question named NotebookLM. Verified 2026-07-24:

- **No public consumer API.** You cannot pull notebooks or automate it.
- An **enterprise API** exists (Google Cloud; notebook CRUD, sources, queries) but is
  enterprise-only, not developer sign-up. A consumer API is "in the works".
- Rebranded to **Gemini Notebook** (July 2026).
- Unofficial reverse-engineered Python clients exist — **ToS risk, not a foundation**.

**Therefore**: NotebookLM/Gemini Notebook is usable as a *manual, one-off bootstrap*
(open it, read its grounded output, curate by hand). For anything repeatable, use an
API-accessible teacher — we already selected **NIM `qwen2.5-72b-instruct` +
`guided_json`** for extraction (see `docs/extraction-model-and-prompt-decisions.md`),
and the Gemini API is the closer analogue of NotebookLM's underlying model.

## Honest caveats

- **Teacher output is not ground truth.** It needs the same single-high-confidence
  precision guard and human review as K.4/K.5 — a confident wrong label propagates
  further than a missing one.
- **Terms of service.** Using another product's outputs to train a model can conflict
  with its terms. Manual inspiration + an eval set is materially safer than automated
  distillation at scale. Check before building a pipeline on it.
- **Offline only.** Never call a teacher per-document at runtime; that reintroduces
  exactly the cost/latency problem the cascade exists to remove.
- **Bounded value.** This buys a cold-start, not a permanent dependency: once real
  labels accumulate from actual use + review, the teacher is no longer needed.

## Cheapest first step (if this gets scheduled)

1. Run 3–5 corpus documents through NotebookLM/Gemini Notebook **manually**.
2. Extract a candidate theme/entity ontology + a gold extraction for those documents.
3. Commit both as an eval fixture.
4. Use it to (a) tighten the extraction prompt against a measurable gap, and (b) seed
   the thematic classifier past its cold start.

If that shows value, promote the teacher to an API model (NIM/Gemini) and scale role 1.

## Open questions

- [ ] Schedule as its own track, or fold into Track L (typing) / Track R (cascade)?
- [ ] Which teacher for the scaled version — NIM `qwen2.5-72b` (already chosen for
      extraction) or Gemini API?
- [ ] How many seed labels per theme are actually needed here (the ~50 figure is
      Purview's, not measured on our corpus)?
