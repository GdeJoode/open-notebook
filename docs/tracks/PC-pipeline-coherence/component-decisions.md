# From unstructured text to a graph you can defend

*A decision document, not a survey. Written 2026-09-06 after PC.3, on the
evidence in `phase-PC.3-measurement.md` and `pipeline-wiring.md`. Every "on/off"
and every consumer count is derived from the code.*

## The two questions

PC.3 measured one stage and the answer generalised. Stage 10 made 18 merges over
531 entities, 4 of them defensible, and the errors were not random: a qualified
concept absorbed into its own head noun, an office into its ministry, an abolished
predecessor into its successor, a list into one of its members. **Every one of
those is a true statement about a relationship, recorded as an identity.**

Similarity measures relatedness. It therefore cannot distinguish "the same thing"
from "a related thing", and this graph is mostly related things. That is not a
tuning problem — at no threshold does correct exceed wrong, and at 0.95 the
resolver ranks an abolished ministry ABOVE the correct one.

So a component earns its place by answering two questions:

> **1. Is the answer CHECKABLE, or is it a score?**
> A checkable answer — an exact key, a register lookup, a document boundary — may
> write the graph. A score may not; it goes to the queue a human reads.
>
> **2. What does it PRODUCE that was not there before?**
> Most stages subtract (noise, duplicates) or annotate. Very few create.

The second question is the one that surprised on inspection. Of seventeen
filtering stages plus the boundary services, **four create anything**: extraction
makes the mention, edge prediction makes relations, `upsert_entity` assigns
identity, and grounding records provenance. Everything else removes or labels.

## The components, on both axes

| component | produces | checkable or score | where the output belongs | state |
|---|---|---|---|---|
| **extraction** (LLM) | the mention: text, type, chunk, context | neither — it is the source | the mention stream | on |
| noise / normalize / reclassify (1–3) | nothing; removes | checkable | — | on |
| string dedup (4) | collapses restatements *within one document* | checkable | the mention stream | on |
| fuzzy + embedding dedup (5, 6) | collapses near-restatements within one document | **score** | writes directly today | on |
| LLM matcher (6b) | merge proposals + `match_candidates` | score | **the curator queue** — correct | off |
| semantic enrichment (7, 8, 9) | annotations | score | nothing reads them | off |
| **KG resolution (10)** | `kg_entity_id`, `kg_match_type`, score | **score** | **nothing reads it** | off (PC.3) |
| incremental clustering (10b) | cluster ids + report | score | no reader | off |
| ontology constraint (11) + centrality (12) | `validation_report` | checkable | no reader | off |
| **edge prediction (13)** | **relations** | **score** | written to the graph, **unmarked** | **on** |
| orphan connector (14) | relations | score | — | off |
| concept alignment (15) | RELATED_TO / NOVEL | score | emits no relations; needs 10's `is_new` | off |
| **`upsert_entity`** | **identity** | **checkable** (exact key on `normalize_entity_name`) | the graph | on |
| **grounding** | **provenance per source** | checkable | the graph | on (PC.3) |
| PC.2 curator queue | a human decision | checkable *after* the human | the graph | fed by 6b, which is off |
| **TOOI / Crossref authority** | **reference + a stable URI** | **checkable** | the graph | **exists, no caller, never loaded** |

## Three things that table makes obvious

**1. The only component that decides identity is the last one anyone would look
at.** `upsert_entity`'s `WHERE name_key = $k AND entity_type = $t`. It is
checkable, it works, and it is the phase's real deliverable. Everything upstream
that touches identity is either scoped to a single document, or produces a score,
or is off.

**2. Edge prediction is on, creates relations, and stamps no provenance.**
`edge_predictor` sets no `extraction_method`; predicted edges are appended to
`filtered.relations` and persisted identically. Measured on `staging`: all 1,892
relations read `extraction_method = 'llm'`. **A machine-inferred edge is
indistinguishable from one the model read in the text.** That is the same class of
defect as stage 10 — a score writing the graph — except here it succeeded in
writing, which is worse. It has never been measured.

**3. The authority is the only checkable answer available for named things, and
it is unwired.** TOOI holds abbreviation, official name without the type, and
official name with it, under one URI — exactly the pairs no matcher could compute.
`reconcile_entity` has no production caller; `reference_entity` holds zero rows.
Crossref sits beside it for works.

But authorities cover *named things*. **429 of 531 active entities are `topic`,
and there is no register of topics.** So authority-first is right and reaches a
minority by count. For topics the answerable question is not "which registered
thing is this" but "how does this relate to what the graph already holds" — which
is what stage 15 was built for, and which its own docstring says is *"currently
handled nowhere"*.

## The order that follows

Today: extract → clean → dedup within document → *guess against the graph* →
assign identity by exact key. The identity rule runs **last**, after two stages
have already guessed with weaker rules, and the decoration that defeats all three
mechanisms is never removed.

Proposed, and each step is a component that answers a checkable question or hands
off to one that does:

1. **Mention** — text, type, chunk, context. Unchanged; it is the only source.
2. **Normalise the surface, including the decoration.** One function, not the four
   that disagree today. This is the single highest-value change in the list:
   `20 rows → 20 keys` for the ministries becomes `7/20` resolvable by lookup once
   `X (Ministerie)` and `ABBR (X)` are handled. With PC.2's discipline — an
   `(IenW)` gloss, a `(Ministerie)` type suffix and a `Staatssecretaris van …`
   prefix are three different shapes and only two may be stripped.
3. **Resolve reference against an authority** where one exists. Exact,
   explainable, and it returns a URI instead of a number.
4. **Assign identity** by exact key for everything with no authority.
5. **Relate, don't merge.** Similarity output becomes a *relation proposal*, never
   an identity. Subsumption handles the largest error class PC.3 measured; the
   rest goes to (6).
6. **The curator queue** — which is a real destination: a human reads it daily or
   weekly (owner, 2026-09-05). That is what makes routing scores there honest
   rather than a way of discarding them.
7. **Provenance at every step**, including which rule made the call. Edge
   prediction is the current counter-example.

## The decisions this asks for

**D1 — Do we normalise decoration?** *Recommend yes, first.* It is the one change
that helps every mechanism at once, and PC.3 measured all three failing on the
same feature. Risk: over-stripping, which PC.2 spent a review round on; the
mitigation is its affix discipline and its curator queue.

**D2 — Do we load TOOI and give `reconcile_entity` a caller?** *Recommend yes,
after D1.* Cheap, checkable, and it produces a stable URI — the strongest form of
structure available here. Bounded: it reaches named things only.

**D3 — What happens to topics?** *Open, and the largest question.* No authority
exists. The candidates are the project's own ontology (stage 15's original job,
emitting subsumption relations rather than merges) or leaving topics
per-document. This is the decision that determines whether the graph is a
knowledge graph or a per-document index.

**D4 — Does similarity keep writing anywhere?** *Recommend no.* Stages 5 and 6
still write directly, within a document. They should propose like everything else
once the queue is on the critical path.

**D5 — What happens to the eight dormant stages?** *Recommend deciding per stage,
not fixing them.* Four (7, 8, 9, 14) have no output consumer at all. A stage with
no reachable configuration is documentation pretending to be code.

**D6 — Edge prediction: measure or disable?** *Recommend measure first, this
week.* It is on, it creates, and nobody has ever looked. If its precision
resembles stage 10's, it has been writing unmarked machine guesses into the graph
for the life of the corpus.

## What must not be repeated

PC.3 spent five review rounds mostly on its own instruments. The two rules that
came out of it apply to this work before any of it starts:

1. **Derive the space, do not sample it** — and check that the space is not one
   boundary too small. A params key inside a call, a name inside a module, a
   module inside a service, a test beside another test.
2. **Verify a guard by putting it in the state it claims to prevent.** Not a state
   of that shape — *the* state.

And the one this phase added, which has no test: **a claim in prose is not checked
by anything.** Three times PC.3 stated a number larger than the data supported,
and each time the only thing that caught it was publishing the raw values so
someone else could re-derive them.
