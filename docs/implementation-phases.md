# Implementation Phases — Updated Status

Last updated: 2026-04-01

## Completed

All 23 original code review items are done. Plus:

- Entity extraction with dual extractors (LLM + LangExtract)
- Entity filtering pipeline wired to extraction + KG persistence
- Naive summarization strategy
- Edge predictor (co-occurrence + embedding + Adamic-Adar)
- Entity embedding generation before dedup
- SemanticBlocker (UMAP + HDBSCAN)
- LLMMatcher (Ollama, Qwen3.5:35b-a3b, NL abbreviation dictionary)
- Bidirectional Obsidian vault sync (backend + frontend)
- Enhanced EntityGraphView (ForceAtlas2, click-to-select, type filter)
- VaultSync UI component (import/export review tabs)

---

## Phase 5: Frontend Wiring Gaps

These backend features are built but the frontend doesn't expose them yet.

### 5.1 Filtering UI in EntitiesTab
- **What**: Add dedup option checkboxes (string, fuzzy, embedding, LLM, edge prediction) and "Run Filtering" button to the EntitiesTab extraction panel
- **Backend**: `POST /sources/{id}/run-filtering` ready, `sourcesApi.runFiltering()` client ready
- **Frontend**: `EntitiesTab.tsx` needs collapsible "Filtering Options" section + stats display
- **Effort**: ~2 hours

### 5.2 Vault settings in SettingsForm
- **What**: Add vault_path, vault_entities_folder, vault_sync_on_startup fields to the Settings form
- **Backend**: Settings model has the fields, API accepts them
- **Frontend**: `SettingsForm.tsx` needs a "Vault Integration" section with inputs
- **Effort**: ~1 hour

### 5.3 KG page provenance + merge history
- **What**: Show source provenance (which documents) and merge history in entity detail panel
- **Backend**: `EntityPersistenceService` stores `merged_from` + `source_ids` in entity properties
- **Frontend**: KG page entity detail panel needs provenance section
- **Effort**: ~2 hours

---

## Phase 6: Test Coverage

### 6.1 VaultSyncService tests
- Test scan_vault_entities with mock vault folder
- Test apply_imports (accept/reject)
- Test queue_export and write_exports
- Test get_merged_dictionary
- **Effort**: ~2 hours

### 6.2 EntityPersistenceService tests
- Test entity upsert (new + existing)
- Test relation creation
- Test merge group provenance
- **Effort**: ~1 hour

### 6.3 SemanticBlocker + LLMMatcher tests
- SemanticBlocker: test with/without HDBSCAN deps, cluster formation
- LLMMatcher: test abbreviation fast-path, mock Ollama responses
- **Effort**: ~2 hours

---

## Phase 7: Summarization Strategies

13 strategies raise `NotImplementedError`. Priority order:

### 7.1 TreeKG (high value — uses document structure)
- Section-based summarization following document hierarchy
- **File**: `pipelines/summarization/src/summarization/treekg/strategy.py`

### 7.2 MapReduce (medium value — parallel batch processing)
- Map: summarize each chunk independently
- Reduce: combine summaries in tree structure
- **File**: `pipelines/summarization/src/summarization/map_reduce/strategy.py`

### 7.3 Refine (medium value — iterative improvement)
- Process chunks sequentially, refining the summary with each chunk
- **File**: `pipelines/summarization/src/summarization/refine/strategy.py`

### 7.4 Others (lower priority)
- Hybrid, WalkingTree, ExtractiveAbstractive, Skeleton, LinkedEntity, DPR_ABS
- Enhancement layers: ChainOfDensity, SelfCorrection, GistTokens

---

## Phase 8: Empty Pipelines

### 8.1 Retrieval Pipeline (blocks canvas + chat apps)
- **Current**: Empty scaffold at `pipelines/retrieval/`
- **Needed**: Vector search, reranking, hybrid retrieval
- **Note**: `SearchRepository` already has SurrealDB queries. This pipeline wraps them with proper abstraction.
- **Effort**: ~4-6 hours

### 8.2 Enrichment Pipeline
- **Current**: Empty scaffold at `pipelines/enrichment/`
- **Dependencies declared**: scholarly, habanero (CrossRef/ORCID)
- **Needed**: Academic metadata lookup, citation enrichment, verification
- **Effort**: ~4-6 hours

---

## Phase 9: Standalone Apps

### 9.1 Chat App
- **Current**: Empty scaffold at `apps/chat/`
- **Note**: Full chat implementation exists in app-main (LangGraph-based). This would extract it into a standalone FastAPI app.
- **Depends on**: Retrieval pipeline
- **Effort**: ~6-8 hours (mostly refactoring)

### 9.2 Canvas App
- **Current**: Empty scaffold at `apps/canvas/`
- **Purpose**: Visual note creation and knowledge mapping
- **Depends on**: Retrieval pipeline
- **Effort**: ~8-12 hours (new feature)

---

## Phase 10: Semantic Dedup Enhancements

From the analysis in `docs/semantic-dedup-analysis.md`:

### 10.1 EDC (Extract-Define-Canonicalize)
- LLM generates natural language definitions per entity
- Match by definition embedding similarity
- High quality for Dutch policy abbreviations but slow

### 10.2 Wikidata/REL Entity Linking
- Link entities to Wikidata QIDs
- REL (Radboud Entity Linker) for Dutch-specific linking
- DBpedia Spotlight already integrated; add Wikidata as complementary

### 10.3 Contrastive Learning
- Fine-tune embedding model on labeled entity pairs
- Bootstrap labels from LLM matcher output
- Long-term quality optimization

### 10.4 Cascaded LLM (local → API fallback)
- Local model handles ~80% of matches
- Uncertain cases escalated to Claude/GPT API
- Already possible via Esperanto multi-provider support

---

## Recommended Next Actions (priority order)

1. **Phase 5** (frontend wiring) — Quick wins, everything is built on backend
2. **Phase 6** (tests) — Safety net before further development
3. **Phase 7.1** (TreeKG strategy) — High-value summarization
4. **Phase 8.1** (retrieval pipeline) — Unblocks chat + canvas apps
5. **Phase 7.2-7.3** (MapReduce + Refine) — More summarization options
6. **Phase 8.2** (enrichment) — Academic metadata
7. **Phase 9** (standalone apps) — Depends on retrieval
8. **Phase 10** (dedup enhancements) — Incremental quality improvements
