# Coherence Review: Entity/Relation Extraction (KG) vs. Embeddings vs. Cluster Summaries

**Date:** 2026-06-25
**Scope:** Read-only architectural review. No code changed.
**Question:** For the stated goal — *connect documents/sources by content; fine-grained detail not needed* — is the heavyweight entity/relation KG extraction the right mechanism, or are per-source embeddings + cross-source (cluster) summaries the better, lower-noise, cheaper path? Where does each belong?
**Evidence base:** Code (file:line) + live SurrealDB counts (`ns/db = open_notebook/staging`, queried 2026-06-25).

---

## Executive summary (honest verdict, 5 lines)

1. **The KG is the wrong primary tool for content-based source linking, and the live data proves it is noisy.** 4,870 entities; only **423 active (8.7%)**, **3,313 archived (68%)**; `other` is the 3rd-largest type (739); the "active" graph is **81% `topic`** (344/423). Triage exists precisely because extraction over-produces.
2. **The relation layer is thin and incoherent for linking:** 1,466 edges over 37 predicate types, but **40% are two Dutch predicates** (`IS_PIJLER_VAN` 317, `LEIDT_TOT` 274), with EN/NL duplicates and typos (`ACEPTS`). This is a vocabulary, not a clean relational schema.
3. **The embedding layer that *should* carry source linking is half-built and currently dormant:** there are **10,501 chunks but 0 chunk embeddings** in staging, **no source-level embedding field at all**, and **0 note/insight embeddings**. Vector search exists as a function but has nothing to search.
4. **Cluster/cross-source summaries — the user's preferred mechanism — do not exist.** RAPTOR runs *within a single source*; there are **no notebook-level summaries**, summaries are **not embedded**, and **summaries are never read downstream** (display-only). 1 summary persisted total.
5. **Verdict:** Make **source-level embeddings + embedded notebook/cluster summaries** the content-linking backbone. Keep the KG, but **scope it down** to where it earns its keep (typed entity queries, Obsidian/NetworkX export, dedup/resolution) — not as the linking engine. The detail-rich relation extraction is largely noise for the linking goal.

---

## Area 1 — Entity/Relation extraction → KG coherence

### Pipeline (as built)
- Extraction orchestrator: `apps/app-main/src/app_main/services/entity_extraction_service.py:982` (`run_extraction`), multi-schema two-pass path at `:802` (`_run_multi_schema`) — Pass-1 schema detection (no LLM) `:880`, Pass-2 typed extraction `:970`.
- Persistence: `apps/app-main/src/app_main/services/entity_persistence_service.py` — entity upsert `:571`, relation RELATE `:305` (`_upsert_relation`), canonical type list `_ALLOWED_ENTITY_TYPES` `:44`, 4-tier type resolution `_resolve_entity_type` `:111`.
- Triage (Track Q): `apps/app-main/src/app_main/services/triage/triage_pipeline.py:135`; status constants `STATUS_ACTIVE`/`STATUS_REFERENCE` in `status_assignment_service.py:34`; assignment is a pure function of `tier × degree × manual_override` (`:115`).

### Where the noise comes from (evidence)
Live type distribution (4,870 entities total):

| entity_type | n | note |
|---|---:|---|
| concept | 1,676 | unbounded, generic |
| topic | 1,222 | unbounded, generic |
| other | **739** | explicit "no type" bucket — pure noise signal |
| organization | 446 | |
| person | 315 | |
| location | 138 | |
| programme | 86 | |
| government_organization | 54 | |
| (rest) | <50 each | administrative_area, product, event, technology, legislation, creative_work, dataset |

- **`concept` + `topic` + `other` = 3,637 / 4,870 = 75%** of all entities are generic/untyped buckets. This is the Track L "rich typing needs an ontology-following model" problem made visible: the local extraction model produces mostly generic labels (consistent with the project memory note on `qwen14b too slow / local llama = generic`).
- **Triage is mostly a discard valve.** Status distribution: `active 423`, `reference 1,055`, `merged 79`, **`archived 3,313`**. 68% of everything extracted is archived. Triage (Track Q) being necessary at all is an admission that extraction yield is low-signal; the magnitude (68% archived) quantifies it.
- **The surviving "active" graph is not a rich typed KG — it is a topic list.** Active-only type distribution: `topic 344`, `administrative_area 35`, `programme 24`, `government_organization 17`, `concept 3`. So **81% of active entities are `topic`** — exactly the coarse, content-thematic signal that an embedding already captures more cheaply.

### Relation layer is thin and incoherent
- 1,466 relations / 37 distinct `relation_type` values. **Concentration:** `IS_PIJLER_VAN` 317 + `LEIDT_TOT` 274 = **591 (40%)**; the long tail is mostly singletons (`ACEPTS`=1 [sic, typo], `HAS_DATE`=1, `DEVELOPS`=1…).
- **Bilingual duplication** is uncontrolled: `FINANCED_BY` vs `FINANCIEERD_DOOR`, `COLLABORATES_WITH` vs `BIJDRAGT_AAN`/`COLLABORATES_WITHIN`. The predicate vocabulary is open-ended free text (relation_type is a string, `entity_persistence_service.py:305`), so it accumulates near-synonyms and misspellings. This is noise for any query or linking use.
- Active-subgraph connectivity: only **456 active→active edges**, and **only 152 of 423 active entities (36%) have any active-active edge.** ~64% of the curated graph is effectively isolated nodes. For *source linking*, edges this sparse and this concentrated on two NL predicates carry little cross-source signal.

### Is the KG actually consumed downstream?
Yes — but **not for content-based source linking.** Consumers found:
- **Graph viz UI** (Sigma.js): `api/routers/knowledge_graph.py:71` (`/graph`), service `knowledge_graph_service.py:113` (`get_graph_data`). Visualization, not linking.
- **Entity resolution / dedup UI**: `api/routers/entity_resolution.py:114`.
- **Exports** (this is where it earns its keep): Obsidian `api/routers/exports.py:194` + `obsidian_export_service.py` (status-filtered), NetworkX `exports.py:100`, JSONL.
- **Vault sync**: `vault_sync_service.py:160`. **Source-detail entity count**: `sources_crud.py:246`.

**Critically:** no consumer uses entity/relation tables to compute *source-to-source similarity or links*. The KG is built, curated, visualized, and exported — but it is **not** the mechanism that connects sources by content. For the user's stated goal, it is built-and-adjacent, not built-and-used.

---

## Area 2 — Per-source embeddings

### Granularity that exists (code)
Four embedding fields are defined, all `array<float>`:
- **Chunk-level** → table `source_embedding` (`migrations/1.surrealql:16-20`, chunk link added `migrations/27.surrealql:16`); generated `pipelines/embeddings/src/embeddings/service.py:113` (`_embed_chunks`), stored `repositories/source.py:253` (`add_embedding`).
- **Insight-level** → `source_insight.embedding` (`migrations/1.surrealql:22-26`), `service.py:221` (`embed_insight`).
- **Note-level** → `note.embedding` (`migrations/1.surrealql:39`), `service.py:188`.
- **Entity-level** (Track P) → `entity.embedding` (`migrations/39.surrealql:30`, FLEXIBLE), backfilled by `scripts/backfill_entity_embeddings.py` over `canonical_name` (`:99`).

### Dimension pin (I.G)
The 768-dim pin is enforced *architecturally*, not in SQL: embeddings must use the single DB-configured model and must **not** be routed through the cloud LLM chain. Canary test `apps/app-main/tests/test_embedding_local_guardrail.py` asserts `LLMTask` has no EMBEDDING member and `EmbeddingService` imports no `model_routing` (`route_resolver.py:24` comment). Resolution via `dependencies.py:636` (`get_embedding_service`). There is **no dimension ASSERT** in `migrations/3.surrealql`; correctness relies on a single pinned model.

### How embeddings are used
- Vector search function `fn::vector_search` (`migrations/3.surrealql:10-73`) does `vector::similarity::cosine(embedding,$query)` over **chunk**, **insight**, and **note** embeddings. Python wrapper `pipelines/retrieval/src/retrieval/service.py:89`; hybrid text+vector merge `repositories/search.py:98`. Entity embeddings feed dedup (`embedding_deduplicator.py`).

### The gap that matters for source linking
- **There is NO source-level content embedding.** The `Source` model (`packages/shared/src/shared/models/source.py:107-177`) and `source` table (`migrations/1.surrealql:1-14`) have **no `embedding` field.** To compare two whole sources you must aggregate their chunk vectors — and nothing in the codebase does this.
- **Live state: the embedding layer is dormant.** Staging has **10,501 chunks but 0 rows in `source_embedding`**, **0 `note`**, **0 `source_insight`**. So `fn::vector_search` currently returns nothing — vector retrieval is *implemented but inert* on this dataset. Meanwhile 1,083 of 4,870 entities *do* carry embeddings (Track P backfill ran partially; 3,787 still empty).

**Interpretation:** the project has invested embedding effort at the *entity* grain (for dedup) while the grain that the user actually needs for source linking — **source-level content vectors** — does not exist, and even the **chunk** grain that would let you synthesize one is currently unpopulated. The embedding strategy is mis-aimed relative to the linking goal.

---

## Area 3 — Cluster / cross-source summaries

### What exists
- **RAPTOR is real but single-source.** `pipelines/summarization/src/summarization/raptor/strategy.py` — GMM+PCA clustering `:116-166`, but it clusters the **chunks of one source** (`summarize()` operates on the passed `chunks` list `:227`; workflow `workflow.py:107` takes a single `chunks` list; orchestrator `source_summarization_orchestrator.py:32` runs per-source). `max_layers=5` (`config.py:38`). Implemented strategies: `naive`, `treekg`, `raptor` (`summarization_service.py:15`); 8+ others are stubs that raise `NotImplementedError` (`workflow.py:47-105`).
- **Summaries are stored but not embedded.** `summary` table is schemaless (`repositories/summary.py:62`), record carries `source_id, strategy, document_summary, summary_nodes[...]` (`summarization_service.py:217`). `SummaryNode` (`models/result.py:55`) has **no embedding field**; embeddings are used transiently during RAPTOR clustering and discarded.

### What does NOT exist (clear absences)
- **No notebook-level / cross-source summary.** Every summary is keyed to one `source_id`; the `notebook` table (`migrations/1.surrealql:44-52`) has no summary field; no code gathers chunks/summaries across the sources of a notebook. The user's "overarching summaries across sources (cluster summaries)" concept is **not implemented.**
- **No embedded summaries** → cannot be used for retrieval or linking.
- **Summaries are display-only.** `api/routers/summaries.py` is CRUD; no retrieval/RAG/linking endpoint reads summary text; chat retrieves chunks, not summaries. Live: **1 summary persisted** in the whole DB.

**Interpretation:** the exact mechanism the user proposes for cheap, low-noise source linking (cross-source cluster summaries, themselves embedded) is the one capability the system is missing. RAPTOR's clustering machinery is present and could be repointed from "within a source" to "across a notebook's source-summaries," which is most of the work.

---

## Synthesis — what should carry the load

For the goal *connect sources by content; detail not needed*, rank by signal/noise and cost:

| Mechanism | Fit for source linking | Noise | Cost | Live readiness |
|---|---|---|---|---|
| **Source-level content embedding** | **Best** — direct source↔source cosine | Low | 1 vector/source | **Missing** (no field; chunks exist to build it) |
| **Embedded cluster/notebook summary** | **Best for "overarching"** — 1 vector/theme | Low | RAPTOR-class compute, amortized | **Missing** (RAPTOR is single-source) |
| Chunk embeddings (aggregate) | Good fallback for source vector | Low-med | exists at scale (10.5k) | present schema, **0 populated** |
| Entity KG (active topics) | Weak proxy (81% `topic`) | Med | high (extraction+filter+triage) | over-built |
| Relation graph | Poor (sparse, 40% two NL predicates) | High | highest | thin |

**Where the KG genuinely earns its keep (do not delete):**
- **Typed entity queries** (find all `government_organization`/`programme`) — embeddings can't answer "give me the orgs."
- **Obsidian / NetworkX / JSONL export** (`exports.py:100,194`) — these *are* the product surface that consumes the KG.
- **Entity resolution / dedup** (`entity_resolution.py`) and vault sync — identity management.
- The KG's value is **structured recall and export**, not **content-based source linking.** Stop asking it to do the latter.

**What is noise vs signal:**
- *Noise:* the `other` bucket (739), most singleton relation predicates, EN/NL predicate duplicates, and the 68% archived tail. The fine-grained relation extraction is the most expensive and least-used-for-linking output.
- *Signal:* the coarse `topic`/`programme`/`government_organization` typing, and (potentially) embeddings — which the system under-populates.

---

## Recommendation (staged, concrete)

**Stage 0 — turn the dormant layer on (prerequisite, low effort).**
Backfill `source_embedding` for the 10,501 existing chunks (`embeddings/service.py:_embed_chunks` already exists; the pipeline just hasn't run on this data). Without this, every vector-based option below is inert. Verify `fn::vector_search` returns hits afterward.

**Stage 1 — add a source-level content embedding (the missing primary mechanism).**
- Add `embedding: option<array<float>>` to the `source` table + `Source` model.
- Populate by **aggregating chunk vectors** (mean/centroid is the cheapest defensible choice; or embed the source's `document_summary` once Stage 3 exists). Reuse the pinned 768-dim model (`get_embedding_service`) — do not introduce a new model (respect the I.G pin).
- New endpoint: `GET /sources/{id}/related` = cosine kNN over `source.embedding`. **This directly answers "connect sources by content"** at ~1 vector/source, no LLM at query time.

**Stage 2 — de-emphasize relation extraction for linking; keep entities lean.**
- Do **not** invest further in richer relation extraction *for the linking goal*; its output is sparse/noisy (evidence above). Keep relation extraction only where export/graph-viz consumes it.
- Normalize the relation predicate vocabulary (map NL↔EN synonyms, fix `ACEPTS`) **only if** the graph-viz/export surface needs it — otherwise leave as-is; it isn't on the linking path.
- Consider suppressing the `other` type at persistence (or routing it straight to `reference`) to cut the 739-entity noise bucket.

**Stage 3 — build embedded cross-source (cluster) summaries (the user's preferred path).**
- Repoint RAPTOR from "chunks of one source" to "summary/centroid per source across a notebook": cluster the per-source vectors (or per-source summaries) and summarize each cluster → **notebook-level cluster summaries.** The GMM+PCA machinery already exists (`raptor/strategy.py:116`); the change is the *input scope* (notebook's sources) and a new `notebook_summary`/`cluster_summary` table keyed by `notebook_id`.
- **Embed those summaries** (add an embedding field; this is the one thing RAPTOR currently throws away). Then: cluster summaries are searchable, avoid recompute, and link sources by shared theme — exactly the stated design.

**Stage 4 — let embeddings drive triage/typing instead of the LLM doing it cold.**
- Since 81% of active entities are `topic`, a `topic` is essentially a cluster label. Source/cluster embeddings can *generate* topics cheaply, reducing reliance on the noisy LLM typing pass. This shrinks the KG to its high-value core (named entities: person/org/programme/legislation) and offloads "concept/topic" to the embedding/summary layer.

**Trade-offs / honest caveats:**
- A mean-pooled source vector loses multi-topic nuance (a source spanning 3 themes blurs). For the *detail-not-needed* goal this is acceptable; if it isn't, fall back to chunk-kNN aggregation or the cluster-summary vectors from Stage 3.
- Cross-source RAPTOR adds LLM cost at ingest time, but it's amortized and replaces per-query work — net cheaper than maintaining the full relation pipeline for linking.
- Keep the KG; this is a *re-scoping*, not a teardown. Its export/resolution value is real and load-bearing elsewhere.

---

## Open questions for the user

1. **Linking grain:** do you want source↔source links (Stage 1) or theme/cluster↔source links (Stage 3), or both? They answer "connect by content" differently (pairwise vs. thematic).
2. **Multi-topic sources:** is a single mean-pooled source vector good enough, or do your sources span enough distinct themes that you need chunk-kNN or cluster-summary vectors?
3. **KG future:** are Obsidian/NetworkX export and entity resolution the *intended* end-products of the KG (i.e., keep it for those), or was the KG always meant to be the linking engine? This determines how hard to shrink it.
4. **Relation vocabulary:** is anyone querying relations (the 40%-NL predicate set) today, or is the graph viz purely exploratory? If unused, predicate cleanup is wasted effort.
5. **Why is the embedding layer empty in staging?** Is `source_embedding` unpopulated because ingestion was reset, or because the embedding step isn't wired into the current ingest path? (Stage 0 hinges on this.)
6. **Topic offloading:** are you willing to let embeddings/cluster-summaries *replace* LLM `topic`/`concept` extraction, accepting fewer but cleaner KG entities?
