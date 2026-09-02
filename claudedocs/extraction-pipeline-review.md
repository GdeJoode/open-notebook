# Extraction pipeline review — document to graph

**Scope**: everything from an uploaded document to a persisted entity or relation.
Before the LLM (parse, chunk, schema selection, anchors), the LLM itself (Pass 1,
Pass 2, merge), and after it (fifteen filtering stages, the canonical bridge, the
writes, and the queues that hang off them). Read-side concerns — search, MCP,
note auto-linking — are deliberately out of scope.

**Method**: measured, not remembered. Where a claim could be checked by running
code, it was. The attrition numbers come from a live end-to-end run against the
real database with a real Regio-Deal convenant; the structural findings come from
the code, with file and line. Where something needs a live run I could not do,
the report says so instead of guessing.

**Not done here**: nothing was fixed. A reviewer who repairs on the way reports on
their own work.

**Correction, 2026-09-02.** The measured runs in sections 3 and 3b ran against a
DIFFERENT SurrealDB instance than the project's own: before Docker Desktop's WSL
integration was enabled, `localhost:8000` reached an empty instance with a single
stale model row. After the container stack was restarted the same address reaches
the project's database (2 notebooks, 14 sources, 5302 entities, 17 configured
models). Nothing in those runs touched the project's data.

What that changes: **finding R4 is withdrawn** (see below). What it does not
change: every code-level finding, which was established by reading the source, and
the attrition measurements, which are about pipeline behaviour and if anything are
cleaner for having started from an empty graph. Findings verified against the
project's own database afterwards are marked **[verified live]**.

---

## 1. What the chain actually is

See `claudedocs/extraction-pipeline.excalidraw` for the picture. In words, five
blocks:

**Before the LLM.** A document is parsed (docling over HTTP, MinerU, or
markitdown as a local fallback), split into one chunk per document element, and
the notebook's applied vocabulary is assembled: `detect_applicable_schemas` scores
every ontology against the document and keeps the top three, the notebook's own
`base_ontology` plus its affinity bundle is forced on top, and the notebook's
accepted schema edits are projected onto deep copies of the result (N.4d.3).
Candidate anchors (N.1, spaCy) are mined from the chunk text and threaded into the
prompt.

**The LLM.** Pass 1 validates the document against the schema and proposes
extensions into `pending_extensions`. Pass 2 extracts typed entities and relations
per chunk. Two deterministic passes bracket it: Hearst patterns mine `is_a` pairs
from the same text (N.2), and the not-a-concept gate drops page furniture before
anything is appended (N.3). A multi-schema merge then reconciles the per-schema
passes into `primary_type` + `type_tags`.

**Filtering.** Fifteen stages, of which the shipped default config enables about
half: noise filter, normalise, reclassify, string/fuzzy/embedding/LLM dedup,
KG resolution against the existing graph, then the optional ontology constraint
filter, centrality analysis, edge prediction, orphan connector, and concept
alignment.

**Into the graph.** The canonical bridge maps the rich ontology label to the
coarse `entity_type` enum, entities are upserted, relations are upserted on
`(in, out, relation_type)` with `source_documents` unioned across documents, and
matched surface forms are registered as aliases.

**After the graph.** Triage assigns a status and may queue an entity for review.
Concept alignment records ontology gaps, which at frequency 5 become schema
proposals.

---

## 2. Who decides what

| Decision | Who | Reversible | Visible to the user |
|---|---|---|---|
| Which ontologies apply to a document | system (content score + notebook config) | n/a per run | no |
| Which types Pass 1 proposes | system → `pending_extensions` | yes | yes, Schema tab |
| Accept / reject a proposed type | **user** | yes | yes |
| Rename / merge / split / delete a type | **user** | yes (recorded, not applied to YAML) | yes |
| Re-parent a type (N.4d.3) | **user**, on an advisory placement | yes (last-wins) | yes |
| Whether extraction pauses for review | user, via `review_required` (default **off**) | yes | yes |
| Which entities are page furniture (N.3) | system, LLM judge on the middle band | **no** | counters only |
| Which entities are noise (stage 1) | system | no | `removed_entities` |
| Which entities are the same (stages 4–6b) | system | no, merged in place | merge groups |
| Whether a new mention is an existing graph entity (stage 10) | system | no | `match_candidates` |
| **Registering an alias** | **system** at stage 10, **never** at stage 15 | no | no |
| Whether a concept is NOVEL | system, LLM judge on the middle band | n/a | report only |
| Recording an ontology gap | system, gated on the reason code | no | **nowhere** |
| Creating a schema proposal | **system, unasked, at frequency 5** | n/a | **nowhere** |
| Triage status / queueing | system, with a user override endpoint | yes | yes |

The single most important line in that table: **by default, nothing waits for the
user.** `review_required` is off, so Pass-1 proposals accumulate while extraction
proceeds. Every other gate is either fully automatic or advisory.

---

## 3. Where things fall away — the live run

One real Regio-Deal convenant (`Convenant_Oost-Groningen.pdf`), parsed by docling,
8 chunks, extracted by `qwen2.5:14b` through the production service against the
live `staging` database, with `ENABLE_CONCEPT_ALIGNMENT=true`. 147 seconds.
Raw data: `claudedocs/pipeline-review-run.json`.

| stage | in | out | dropped |
|---|---|---|---|
| Pass 2 (LLM) | 8 chunks | 16 entities, 12 relations | — |
| 1 noise filter | 16 / 12 | 16 / 12 | **0** |
| 2 normalize | 16 | 16 | 0 |
| 3 reclassify | 16 | 16 | 0 |
| 4 string dedup | 16 | 16 | **0** |
| 5 fuzzy dedup | 16 | 16 | **0** |
| 6 embedding dedup | 16 | 16 | **0** |
| 10 KG resolution | — | — | **stage disabled** |
| 11 ontology filter / 12 centrality | — | — | **stages disabled** |
| 15 concept alignment | 16 | 0 classified | **inert** |
| persisted | | 16 entities, 12 relations | |

**Nothing was filtered, nothing was merged, nothing was dropped.** Sixteen entities
left the LLM and sixteen entities entered the graph. Under the configuration the
application builds by default, the fifteen-stage pipeline is a pass-through.

That is not a bug in any stage. Each stage did what its config said. It is the
composition that is the finding: the stages that would have caught something are
the ones the default config leaves off.

### What landed in the graph (single document)

```
K.H. Ollongren            concept    Person        M.C.G. Keijzer          concept   Person
drs. K.H. Ollongren       person     Ambtenaar     mr. drs. M.C.G. Keijzer person    Ambtenaar
R.W. Knops                concept    Person        Carola Schouten         person    Ambtenaar
drs. R.W. Knops           person     Ambtenaar     Regio Deal Oost-Gron.   programme Deal
```

Three people are in the graph **twice**, and each pair is split across two
different canonical types. Relations: `IS_BESTUURDER_VAN` (7) and `RELATES_TO` (5)
— a predicate the model invented and a generic fallback; no `is_a` at all.

## 3b. The corpus run — six documents into one notebook

A single document cannot show whether the pipeline recognises something it has
already seen. So the same harness was run over six documents in ONE notebook:
three Regio-Deal convenanten (which share ministers and ministries), two Dutch
policy documents of a different genre, and one English academic paper. 10 chunks
each, ~10 minutes total. Raw data: `claudedocs/pipeline-review-corpus.json`.

| document | entities | relations | graph after |
|---|---|---|---|
| Convenant Oost-Groningen | 17 | 2 | 17 / 3 |
| Convenant Zuid-Oost-Drenthe II | 22 | 13 | 35 / 13 |
| Convenant Zuidwest-Friesland | 19 | 0 | 54 / 13 |
| Bijlage 3 Achterhoek NPVR | 0 (harness, see below) | 0 | 54 / 13 |
| Van meer waarde (brede welvaart) | 25 | 10 | 78 / 25 |
| fiscal_equalisation | 24 | 7 | 101 / 33 |

Two documents were added afterwards into the SAME notebook: `Bennett_test.pdf`
(an English book's front matter, 10 chunks → **0 entities**) and the Achterhoek
bijlage re-run with the length threshold relaxed (10 chunks → 17 entities, 6
relations). **Final graph: 117 entities, 39 relations, 0 gaps, across 8 documents
and 70 chunks.**

**A limitation of this table, stated so it is not misread.** It starts AFTER the
LLM. Track N.3's not-a-concept gate runs inside Pass 2, before the
`ExtractionResult` is built, so anything it removed is already gone by the first
row here. "Nothing was dropped" is true of the fifteen filtering stages and says
nothing about N.3 — and it cannot be recovered after the fact, because
`not_a_concept_removed` and `abstained_chunks` are discarded by the multi-schema
merger (R3). N.5a fixes that.

**The graph grows linearly.** 124 entities were extracted across the eight
documents; 117 rows exist. Seven collapsed, every one of them inside a single
document's batch. Three convenanten
naming the same ministers produced 58 entities with essentially no consolidation.

### The duplicates this exposes

Exact duplicates after normalisation — the same string, twice in one graph:

```
M.C.G. Keijzer   person / Ambtenaar        M.C.G. Keijzer   concept / Person
Brede Welvaart   topic / BeleidsThema      brede welvaart   topic / BeleidsThema
Het Rijk         (x2)
```

`Brede Welvaart` and `brede welvaart` differ only in capitalisation. Both stages
that would catch this — string dedup (stage 4, which lowercases) and KG resolution
against the existing graph (stage 10) — either run only within one document's
batch or are disabled by default. Nothing looks at the graph that is already there.

Twenty-five long-form/short-form pairs sit unmerged, including
`Binnenlandse Zaken en Koninkrijksrelaties` beside
`Minister van Binnenlandse Zaken en Koninkrijksrelaties`, and
`Infrastructuur en Waterstaat` beside
`Staatssecretaris van Infrastructuur en Waterstaat`. These are exactly the pairs
D-N4-9 said the fuzzy tier structurally misses and that alias candidates were kept
alive to catch — and those candidates go nowhere (G2).

### Typing is unstable across documents

```
Provincie Drenthe      programme            RegioDeal    tags=[RegioDeal, Deal]
Provincie Overijssel   administrative_area  Provincie    tags=[Provincie, AdministrativeArea]
```

One province is a `RegioDeal`; the other is a `Provincie`. The model mislabelled
the first, and nothing downstream disagreed: the ontology constraint filter is off,
so the canonical bridge faithfully mapped a wrong rich label to a wrong canonical.

Across all 117 entities the buckets are `concept` 35, `topic` 28,
`government_organization` 15, `person` 14, `other` 10, `administrative_area` 6,
`programme` 4, `organization` 3, `location` 2. **Thirty-eight percent landed in
`concept` or `other`** — the two buckets that carry no meaning.

Sentence fragments also survived as entities:
`het verhogen van het bruto binnenlands product van de regio` (typed
`BeleidsPijler`) and `bereid maken voor bijstand of ondersteuning`. The N.3
not-a-concept gate is aimed at page furniture, not at clauses.

### One document produced nothing, for a reason worth separating

`Bijlage 3 Achterhoek NPVR` yielded 0 chunks and therefore 0 entities. That is a
limitation of this harness, not a pipeline defect, and the distinction matters:
docling line-fragments that document into 100 text elements of which **none reach
40 characters**, and its substance sits in 4 tables. The harness reads only
`doc.texts` and skips elements under 40 characters; the production chunker
(`chunking.chunk_builder.from_document`) emits a chunk per table and per element
regardless of length. Re-run with the threshold at 0 the same document produced **17 entities and 6
relations** from 10 fragment-chunks, so the content is extractable. What the pair
of runs shows is a real layout sensitivity: a document docling line-fragments
reaches Pass 2 as ~30-character pieces, and what comes out depends entirely on
how they are packed.

### A document that produced nothing, and a record that cannot say why

`Bennett_test.pdf` gave Pass 2 **ten genuine chunks and zero entities**. That may
well be correct — the first ten elements of a 1980 book are its title page and
colophon, which is exactly the page furniture the N.3 gate exists to suppress.
But the stored record cannot distinguish "the model found nothing" from "the gate
removed everything", because `abstained_chunks` and `not_a_concept_removed` are
discarded by the multi-schema merger (R3). The metrics that would explain the one
run that needs explaining are the ones that were thrown away.

Pass 1 meanwhile worked on that document: `best_coverage` 0.82, three proposed
extensions (`Method`, `GrantFundingSource`, …). Which leads to the finding below.

### Inconsistencies — two parts disagreeing

**I1. One label, two canonical types, decided by a content score.**
`resolve_ontology_type("Person", …)` returns `person` under `general`, and
`concept` under `deals` and `government`. Which applies is decided by
`detect_applicable_schemas` scoring the *document*. This run applied
`policy_themes` + `deals`, so every entity the model labelled `Person` landed in
`concept` — the junk bucket — while every entity it labelled `Ambtenaar` landed
in `person`. Same document, same kind of thing, two buckets, and the more generic
label gets the worse answer.
*Evidence*: the live graph above; `canonical_bridge.resolve_ontology_type`.

**I2. Aliases are automatic in stage 10 and forbidden in stage 15.**
`KGResolver` registers an alias on a fuzzy match with `register_aliases=True` by
default (`kg_resolver.py:301`). Concept alignment refuses to, on the explicit
grounds that merging two identities "must be an explicit decision, not a side
effect of classification" (D-N4-9). Both run in the same pass over the same
entities. One decision, two opposite policies.

**I3. Subsumption is mined at instance level and decided at type level.**
N.2 mines `is_a` pairs from chunk text and seeds them as relations; D-N4-12
concluded subsumption belongs at the TYPE boundary and retired the instance-level
tier. Both are shipped. And `is_a` is declared in no ontology, so it survives only
because `OntologyValidator` downgrades an unknown predicate to a WARNING outside
strict mode (`validator.py:230`) and because stage 11 is off by default. Turning
on `strict_mode` — a reasonable quality decision — silently deletes every mined
hierarchy edge. No test pins this.

### Gaps — nobody decides

**G0b. And the row that HOLDS the queue is never created either — the cycle is
closed. [verified live]** Found only by running a real extraction against the
project's own database, which produced 323 entities while leaving
`notebook_schema` at zero rows and `pass1_results` unchanged: Pass 1 had not run
at all.

```
no schema row -> no base ontology forced -> content scoring clears the floor
for nothing -> _run_multi_schema RETURNS EARLY to the legacy single-schema
path -> Pass 1 never runs -> no proposals -> nothing creates the row
```

Nothing in production writes a `notebook_schema` row. The router's
`_ensure_schema_row` builds one in memory and returns it unpersisted, while
its own docstring claims "we materialise the row eagerly so the toggle persists
across restarts"; the only writers are the three toggle endpoints, so the row
exists only if a user happens to flip a switch. The comment on the early return
already calls this "the common case: a notebook with no configured schema" — what
it does not say is that the case is self-sustaining.

Measured after fixing it (PC.1): the row is created, `coverage_pct` moves from
0.0 to 0.508, and the 111 stranded proposals collapse to 79 distinct queued types.

**G0. The curator queue has no writer, so the whole review surface is unreachable.
[verified live]**
This is the largest finding in the review, and it only became visible by running
several documents.

Pass 1 works and is persisted. On the project's own corpus: **17 `pass1_results`
rows carrying 111 proposals across 79 distinct type names** — `Coalitie`,
`Coalition`, `CoalitieType`, `Acteur`, `BestuursActeur`, `CollectiveActor`,
exactly the near-duplicates one curator decision resolves — none of which had ever
reached a screen. On the review's own eight-document run, `pass1_results` holds
**14 rows**, each with a coverage figure (0.42–0.85) and one to three proposed types
(`Method`, `GrantFundingSource`, …). But `notebook_schema.pending_extensions` is
**empty**, and `NotebookSchemaRepository.add_pending_extension` has **no
production caller** — the only callers in the repository are in its own roundtrip
test.

Everything downstream operates on `pending_extensions`: `accept_extension`,
`reject_extension`, the `PendingExtensionsPanel`, and — through the accept step —
the entire N.4d chapter: the type placement, the judge, and the re-parent. Four
sub-phases of work sit behind a queue that nothing fills.

`notebook_schema.coverage_pct` is dead in the same way: the router reads it for
the Schema tab and it is documented as driving the B.3c soft-nudge, but nothing
in the extraction path writes it. After eight documents and 14 Pass-1 measurements
it is still `0.0`. The orchestrator computes `best_coverage` and puts it in the
extraction metadata instead.


**G1. The gap loop writes to a room with no door.**
`ontology_gap` and `schema_proposal` rows are written by
`OntologyEvolutionAgent`, whose only production caller is the recorder wired in
N.4d.4. There is **no API route, no MCP tool, no CLI, and no frontend** that reads
either table — verified by searching every `.py`, `.ts` and `.tsx` outside tests.
Meanwhile `auto_propose=True` with `frequency_threshold=5` means the system writes
schema proposals unasked. The write path exists; the read path does not.

**G2. Alias candidates go into a report the application discards.**
D-N4-9 kept the lexical signal alive as review candidates precisely because
`KGResolver`'s fuzzy tier structurally misses long-form/short-form pairs (a large
length delta tanks Levenshtein). The live run demonstrates the miss —
`M.C.G. Keijzer` vs `mr. drs. M.C.G. Keijzer` — and the candidates that would have
caught it are placed on `concept_alignment_report`, which `run_extraction` reads
for counters and then drops. Nothing surfaces them.

**G3. Nothing waits for the user by default.**
`review_required` is off by default, so Pass-1 proposals accumulate while
extraction proceeds. Every other decision in the chain is automatic or advisory.
That may well be the right default — but it means the human-in-the-loop story is
opt-in, and the queues that exist (`match_candidates`, alias candidates, gaps) are
mostly unread.

### Risks — works today, on an unstated assumption

**R1. Enabling concept alignment does nothing in the default configuration.**
The stage is gated on `ENABLE_CONCEPT_ALIGNMENT`, but it classifies only entities
KG resolution marked `is_new` — and the default `FilteringConfig` the application
builds does not enable KG resolution. The live run had the flag ON, the recorder
wired (`gap_recorder_wired: true`), and classified **zero** entities. The workflow
does warn, which is why this is a risk and not an inconsistency; but a feature
reachable only by also changing a second, unrelated default is effectively off.

**R2. Identity is defined four times.**
`EntityDeduplicator._normalize_key`, `FuzzyResolver._normalize`,
`KGResolver._normalize` and `concept_alignment._normalize` are the same six lines,
copied. `_levenshtein_similarity` is duplicated too. Consistent today by
coincidence of maintenance, not by construction; any one can drift invisibly.

**R3. Track N.3's observability is discarded on the production path — and it is
missed exactly where it is needed.**
`_merge_results` builds a fresh `ExtractionResult` with new metadata, so
`not_a_concept_removed`, `not_a_concept_judged` and `abstained_chunks` — the
over-generation metrics N.3 exists to produce — survive only on the single-schema
path. The stored `extraction_result.metadata` from the live run contains none of
them.

**R4. ~~The model route falls back to a model that is not installed.~~
WITHDRAWN.** This described the empty instance the review had unknowingly
connected to, which held one stale model row and no `default_models`. The
project's own database has 17 configured models and a populated `default_models`
whose `default_extraction_model` resolves. The claim was wrong for this
environment and is retracted rather than softened.

**R4b. Ollama silently truncates every prompt over its runtime default.
[verified live]** Replacing R4, and a larger problem. `llama3.1:8b` has a native
context of 131072 tokens, but Ollama's runtime default here is 4096 and the model
carries no `num_ctx`. Esperanto's Ollama provider forwards only `temperature`,
`top_p` and `max_tokens`; it never sends `num_ctx`, so the `context_window`
column on a model row cannot influence it. Measured on a 5507-token prompt:
`prompt_eval_count` 4096 against the stock model, 7220 against a variant created
with `num_ctx=16384`. Nothing errors — the model answers a prompt whose tail was
cut off, and the JSON instructions live in that tail, which is what produces the
`Failed to parse LLM response` errors in the extraction logs.

Note the ORDER of any fix: filling in `context_window` makes the M.3 packer build
LARGER prompts, so doing that before raising Ollama's context makes the truncation
worse, not better.

**R5. Stage ORDER is load-bearing and only partly guarded.**
The N.4d.0 review already established by mutation that the workflow tests do not
catch a misplaced producer of the shape that matters. Stage 14 documents why it
must follow stage 11; stage 15's position is documented as "kept to avoid churn,
NOT because it is guarded".

---

## 5. What this means for N.5

N.5 is currently "regression gate + docs, landing residuals R2–R5 and C2–C5".
The review says that is too small. Three things are worth more than a regression
gate over a pipeline that currently filters nothing:

0. **Fill the curator queue, or admit the loop is not wired.** Nothing writes
   `pending_extensions` or `coverage_pct`, so the accept/reject surface and every
   phase built on it (N.4d.1–N.4d.3) cannot be reached from a real run. This
   outranks everything else in the list, because the other work is downstream of
   it.
1. **Make the default configuration coherent.** Decide what the shipped pipeline
   should actually do — today it is an LLM-to-graph pass-through with three
   quality stages off — and make the flags express one intent instead of five
   independent ones. This is the finding with the largest practical effect.
2. **Close the identity story.** One normalisation, one alias policy, and a
   surface for the long-form/short-form pairs that fuzzy matching cannot catch.
   The live run produced three duplicate people out of sixteen entities.
3. **Give the curator loop a door, or stop writing through it.** Either surface
   gaps and proposals, or turn `auto_propose` off until something reads them.

The existing residuals (R2–R5, C2–C5) remain valid but are small relative to
these. My recommendation is to re-plan N.5 around the three above and carry the
residuals into it as line items.
