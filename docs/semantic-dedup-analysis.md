# Semantic Entity & Relation Deduplication Pipeline — Analyse

## Stap 1: Inventarisatie Bestaande Codebase

### Huidige architectuur

```
Extraction (ontology-extraction)
  → ExtractionResult (entities, relations)
    → FilteringWorkflow (entity-filtering, 13 stages)
      → FilteredResult (deduped entities, predicted edges)
        → EntityPersistenceService → SurrealDB (entity, relation tables)
          → Knowledge Graph frontend (Sigma.js + ForceAtlas2)
```

### Wat al gebouwd en werkend is

| Component | Status | Bestanden |
|-----------|--------|-----------|
| String dedup (normalisatie) | ✅ Productie-klaar | `entity_deduplicator.py` |
| Fuzzy matching (Levenshtein, Jaro-Winkler, Soundex/Metaphone) | ✅ Productie-klaar | `fuzzy_resolver.py` |
| Embedding dedup (FAISS k-NN + numpy fallback) | ✅ Geïmplementeerd | `embedding_deduplicator.py` |
| KG resolutie (3-tier cascade: alias → fuzzy → semantic) | ✅ Geïmplementeerd | `kg_resolver.py` |
| Entity linking (DBpedia Spotlight) | ✅ Geïmplementeerd | `entity_linker.py` |
| Edge prediction (co-occurrence + embedding + Adamic-Adar) | ✅ Geïmplementeerd | `edge_predictor.py` |
| Embedding infrastructuur (Esperanto, Ollama support) | ✅ Productie-klaar | `embeddings/providers/` |
| UnionFind (transitieve merging) | ✅ Productie-klaar | `union_find.py` |
| Configuratiesysteem (alle stages togglebaar) | ✅ Compleet | `config.py` (8 geneste dataclasses) |

### Wat ONTBREEKT voor semantische dedup

1. **Entity-level embedding generatie** — Entiteiten komen uit extractie zonder embeddings. De embedding pipeline werkt op chunks, niet op losse entiteiten. Er is een stap nodig die entity-teksten embed vóór dedup.
2. **HDBSCAN/clustering als blocking** — Alleen FAISS k-NN pairwise beschikbaar. Geen cluster-gebaseerde blocking om het n²-probleem te reduceren.
3. **LLM verificatie stage** — Config bestaat (`LLMVerificationConfig`) maar geen implementatie in de workflow.
4. **Abbreviation dictionary** — Geen expliciet woordenboek voor Nederlandse beleidsafkortingen (BZK, EZK, NPVR, etc.).

### Beschikbare dependencies

| Package | Status | Gebruikt door |
|---------|--------|---------------|
| numpy | ✅ Geïnstalleerd | Embedding dedup, similarity |
| faiss-cpu | ⚙️ Optioneel (gedeclareerd) | FAISS k-NN dedup |
| jellyfish | ⚙️ Optioneel (gedeclareerd) | Fuzzy matching phonetic |
| networkx | ✅ Geïnstalleerd | Graph analyse, centrality |
| scikit-learn | ✅ Geïnstalleerd | Clustering (RAPTOR) |
| sentence-transformers | ⚙️ Optioneel (gedeclareerd) | Lokale embedding modellen |
| httpx | ✅ Geïnstalleerd | Entity linking HTTP calls |
| hdbscan | ❌ Niet beschikbaar | Nodig voor blocking |
| umap-learn | ❌ Niet beschikbaar | Nodig voor dim-reductie |
| rapidfuzz | ❌ Niet beschikbaar | Snelle string similarity |

### Ollama integratie

- Esperanto verbindt via `http://localhost:11434` (default)
- Ondersteunt zowel language models als embedding models
- Geconfigureerde modellen: `qwen2.5:latest`, `qwen2.5:14b`
- Embedding model: niet vooraf geconfigureerd — runtime instelbaar

---

## Stap 2: Beoordeling van de 10 Methoden

### VRAM-budget (RTX 5070 Ti, 16GB)

| Component | VRAM | Wanneer geladen |
|-----------|------|-----------------|
| Ollama LLM (7-8B, Q4_K_M) | ~4.5-5.5 GB | Matching fase |
| Embedding model (multilingual-e5-large) | ~1.1 GB | Blocking fase |
| CUDA overhead + KV cache | ~1-2 GB | Altijd |
| **Beschikbaar voor concurrent** | **~8-10 GB** | LLM + embedding tegelijk |

### Methode-evaluatie samenvatting

| # | Methode | VRAM | Doorlooptijd (1k entities) | Kwaliteit NL beleid | Complexiteit | Zelfstandig bruikbaar? |
|---|---------|------|---------------------------|---------------------|--------------|----------------------|
| 1 | Embedding + Cosine | ~1.1 GB | ~15 sec | Matig | Laag | Nee (alleen blocking) |
| 2 | Semantic Blocking (HDBSCAN) | ~1.1 GB | ~20 sec | Matig | Laag | Nee (alleen blocking) |
| 3 | LLM-as-Matcher | ~6 GB | ~15-30 min | Hoog | Middel | Ja (met blocking) |
| 4 | Extract-Define-Canonicalize | ~6 GB | ~30-50 min | Zeer hoog | Middel-Hoog | Ja (met blocking) |
| 5 | Schema Clustering | ~5 GB | ~2-5 min | Matig | Laag-Middel | Nee (complementair) |
| 6 | Cascaded LLM (lokaal → API) | ~5+API | ~25 min | Zeer hoog | Middel | Ja |
| 7 | GNN Alignment | ~1 GB | ~5 min | Laag-Matig | Hoog | Nee |
| 8 | Contrastive Learning | ~2-4 GB | ~15 sec (inference) | Potentieel hoog | Hoog (training) | Nee (data nodig) |
| 9 | Entity Linking KB | ~2-4 GB | ~1-3 min | Matig-Hoog | Middel | Nee (coverage gaps) |
| 10 | Hybrid Pipeline | ~6-7 GB | ~15-35 min | Hoogst | Hoog | Ja (aanbevolen) |

### Gedetailleerde beoordeling per methode

#### Methode 1: Embedding + Cosine Similarity
- **Past in VRAM**: Ja, ~1.1 GB voor multilingual-e5-large
- **Past in codebase**: Ja — `EmbeddingDeduplicator` bestaat al met FAISS k-NN
- **Complexiteit**: Laag — activeer `embedding_dedup.enabled = True`
- **NL beleidsdocumenten**: Matig. Dichte embeddings plaatsen "BZK" niet altijd dicht genoeg bij "Ministerie van Binnenlandse Zaken en Koninkrijksrelaties". Goed als blocking-stap, onvoldoende alleen.

#### Methode 2: Semantic Blocking (HDBSCAN)
- **Past in VRAM**: Ja, clustering draait op CPU
- **Past in codebase**: Nee — HDBSCAN en UMAP moeten worden toegevoegd
- **Complexiteit**: Laag — drop-in preprocessing stap
- **NL beleidsdocumenten**: Uitstekend als blocking-stap. HDBSCAN met `min_cluster_size=2` vindt ook paren. Belangrijk: UMAP naar ~15-25 dimensies vóór HDBSCAN vanwege curse of dimensionality.

#### Methode 3: LLM-as-Matcher
- **Past in VRAM**: Ja, ~6 GB totaal (embedding + LLM concurrent)
- **Past in codebase**: Gedeeltelijk — `LLMVerificationConfig` bestaat, implementatie ontbreekt
- **Complexiteit**: Middel — prompt templates, gestructureerde output, batching
- **NL beleidsdocumenten**: Hoog. LLM kan redeneren over afkortingen, context-afhankelijke entiteiten. Kwaliteit hangt af van few-shot voorbeelden met NL beleidsafkortingen. **Bottleneck: ~1-2 vergelijkingen/seconde op 7B model.**

#### Methode 4: Extract-Define-Canonicalize (EDC)
- **Past in VRAM**: Ja, zelfde als methode 3
- **Past in codebase**: Nee — drie nieuwe stages nodig
- **Complexiteit**: Middel-Hoog — definitie-generatie, caching, kwaliteitsvalidatie
- **NL beleidsdocumenten**: Zeer hoog. Definities disambigueren afkortingen expliciet. Vergelijken van definities robuuster dan korte entiteitsnamen. **Maar: ~30-50 minuten voor 1000 entiteiten.**

#### Methode 5: Schema Clustering
- **Past in VRAM**: Ja
- **Past in codebase**: Ja — ontology-manager bestaat, entity types in SurrealDB
- **Complexiteit**: Laag-Middel
- **NL beleidsdocumenten**: Matig. Helpt bij systematische variaties (alle "Ministerie" types standaardiseren) maar mist instance-level ambiguïteit. **Complementair, geen vervanging.**

#### Methode 6: Cascaded LLM Pipeline
- **Past in VRAM**: Ja voor lokale tier; API-tier gebruikt geen lokale VRAM
- **Past in codebase**: Ja — Esperanto ondersteunt al OpenAI/Anthropic/Google API's
- **Complexiteit**: Middel — confidence scoring, routing logica
- **NL beleidsdocumenten**: Zeer hoog. Lokaal model handelt ~80% af; API voor ambigue gevallen. **Kosten: ~€0.10-€0.30 per 1000 entiteiten (GPT-4o-mini).**

#### Methode 7: GNN Alignment
- **Past in VRAM**: Ja (~1 GB)
- **Past in codebase**: Nee — PyTorch Geometric/DGL niet geïnstalleerd
- **Complexiteit**: Hoog — graph constructie, GNN training, tuning
- **NL beleidsdocumenten**: Laag-Matig. GNN alignment werkt best bij twee gescheiden grafen. Voor within-corpus dedup voegt het weinig toe t.o.v. eenvoudiger methoden.

#### Methode 8: Contrastive Learning
- **Past in VRAM**: Ja (~2-4 GB voor fine-tuning)
- **Past in codebase**: Nee — training pipeline nodig
- **Complexiteit**: Hoog (training) — gelabelde paren nodig
- **NL beleidsdocumenten**: Potentieel hoog na training. Maar: eerst gelabelde data verzamelen. **Overweeg als optimalisatie nadat grondwaarheid is verzameld via LLM pipeline.**

#### Methode 9: Entity Linking naar Externe KB
- **Past in VRAM**: Ja (~2-4 GB lokaal, of API)
- **Past in codebase**: Ja — `DBpediaSpotlightLinker` bestaat al. **REL (Radboud Entity Linker)** is specifiek voor Nederlands ontwikkeld.
- **Complexiteit**: Middel
- **NL beleidsdocumenten**: Matig-Hoog voor bekende organisaties (ministeries → Wikidata), laag voor domeinspecifieke entiteiten (obscure beleidsprogramma's, commissienamen).

#### Methode 10: Hybrid Pipeline
- **Past in VRAM**: Ja (~6-7 GB totaal)
- **Past in codebase**: Gedeeltelijk — veel bouwstenen bestaan al
- **Complexiteit**: Hoog initieel, maar modulair
- **NL beleidsdocumenten**: Hoogst. Elke stage compenseert zwaktes van andere:
  - Blocking vangt semantisch vergelijkbare entiteiten
  - String similarity vangt afkortingspatronen
  - LLM matching handelt ambigue gevallen
  - Transitieve closure propageert matches

---

## Stap 3: Implementatieadvies

### Rangschikking (haalbaarheid × waarde)

1. **Methode 10: Hybrid Pipeline** — Hoogste kwaliteit, bouwt voort op bestaande code
2. **Methode 2: HDBSCAN Blocking** — Essentiële component, laag risico
3. **Methode 3: LLM-as-Matcher** — Kernmatcher, hoge kwaliteit voor NL
4. **Methode 6: Cascaded LLM** — Pragmatisch, beste kwaliteit met API fallback
5. **Methode 1: Embedding + Cosine** — Al geïmplementeerd, activeren
6. **Methode 4: EDC** — Zeer hoge kwaliteit maar langzaam
7. **Methode 9: Entity Linking** — Complementair signaal
8. **Methode 5: Schema Clustering** — Nuttig voor ontologie-opschoning
9. **Methode 8: Contrastive Learning** — Toekomstige optimalisatie
10. **Methode 7: GNN Alignment** — Overkill voor dit gebruik

### Gefaseerd implementatieplan

#### Fase 1: Quick Wins (direct integreerbaar, ~1 dag)

**Activeer bestaande embedding dedup + entity embedding generatie**

Wat:
- Voeg entity-tekst embedding generatie toe aan de extractie-flow
- Activeer `EmbeddingDeduplicator` in de `FilteringConfig`
- Activeer `FuzzyResolver` met Jaro-Winkler voor NL

Bestanden:
- `apps/app-main/src/app_main/services/entity_extraction_service.py` — embed entities na extractie
- `pipelines/entity-filtering/src/entity_filtering/config.py` — defaults aanpassen

Benodigde packages: `faiss-cpu` (al gedeclareerd als optional dep)

Ollama model: een embedding model nodig, bijv. via `ollama pull nomic-embed-text` of sentence-transformers `multilingual-e5-large`

#### Fase 2: Semantic Blocking + LLM Matching (~3-5 dagen)

**HDBSCAN blocking + LLM-as-Matcher implementatie**

Wat:
- UMAP + HDBSCAN clustering als blocking-stap vóór pairwise vergelijking
- LLM-gebaseerde matcher via Ollama voor kandidaatparen
- NL beleidsafkortingen dictionary als few-shot voorbeelden

Nieuwe bestanden:
- `pipelines/entity-filtering/src/entity_filtering/deduplication/semantic_blocker.py`
- `pipelines/entity-filtering/src/entity_filtering/resolution/llm_matcher.py`
- `pipelines/entity-filtering/src/entity_filtering/data/nl_abbreviations.json`

Benodigde packages: `hdbscan`, `umap-learn`, `rapidfuzz`

Ollama model: `ollama pull qwen2.5:7b-instruct` (4.5 GB, goede NL ondersteuning)

#### Fase 3: Verrijking + Cascade (~2-3 dagen)

**EDC definities + Wikidata linking + API fallback**

Wat:
- LLM genereert definities per entiteit, match via definitie-embeddings
- Wikidata/REL integratie voor bekende organisaties
- Onzekere gevallen doorsturen naar API-model (Claude/GPT)

Nieuwe bestanden:
- `pipelines/entity-filtering/src/entity_filtering/resolution/definition_matcher.py`
- `pipelines/entity-filtering/src/entity_filtering/resolution/wikidata_linker.py`

Benodigde packages: `qwikidata` of `SPARQLWrapper`

---

### Aanbevolen embedding modellen

| Model | Grootte | VRAM | NL kwaliteit | Aanbeveling |
|-------|---------|------|--------------|-------------|
| `multilingual-e5-large` | 560M | ~1.1 GB | Zeer goed | **Primair model** |
| `bge-m3` (BAAI) | 568M | ~1.1 GB | Zeer goed | Alternatief (dense+sparse) |
| `paraphrase-multilingual-MiniLM-L12-v2` | 118M | ~0.25 GB | Goed | Lightweight fallback |

### Aanbevolen Ollama modellen voor matching

| Model | Quant | VRAM | NL kwaliteit | Concurrent met embedding? |
|-------|-------|------|--------------|--------------------------|
| `qwen2.5:7b-instruct` | Q4_K_M | ~4.5 GB | Goed | ✅ Ja (~6 GB totaal) |
| `llama3.1:8b-instruct` | Q4_K_M | ~5.0 GB | Goed | ✅ Ja (~6.5 GB totaal) |
| `mistral-nemo:12b` | Q4_K_M | ~7.0 GB | Zeer goed | ⚠️ Krap (~8.5 GB totaal) |

---

### Risico's en beperkingen voor NL beleidsjargon

1. **Afkortingsdekking** — Een 7B LLM kent niet alle NL overheidsafkortingen. **Mitigatie**: onderhoud een expliciet afkortingswoordenboek als few-shot voorbeelden in de prompt.

2. **Doorlooptijd** — Bij 15-30 minuten per 1000 entiteiten kost 10.000+ entiteiten uren. **Mitigatie**: agressieve blocking reduceert kandidaatparen; batch LLM calls; async processing.

3. **Transitieve closure fouten** — Als A=B en B=C maar A≠C, creëert transitieve closure een valse merge. **Mitigatie**: gebruik graph-gebaseerde clustering met edge-weight drempels i.p.v. naïeve transitieve closure. Dit is al gebouwd via `UnionFind`.

4. **Context-afhankelijke entiteiten** — "De minister" verwijst in document X naar persoon A, in document Y naar persoon B. **Mitigatie**: include bron-context in de LLM matching prompt.

5. **Modelversionering** — Ollama modelversies veranderen. **Mitigatie**: pin specifieke modelversies in pipeline config.

---

## Stap 4: Prototype-ontwerp

Het prototype implementeert de Hybrid Pipeline (Methode 10) met de volgende stages:

```
1. Entity Embedding Generatie (Esperanto/Ollama)
   ↓
2. UMAP Dimensiereductie (384→20 dimensies)
   ↓
3. HDBSCAN Semantic Blocking (clusters vormen)
   ↓
4. Binnen elk blok: feature-extractie
   - Cosine similarity (embedding)
   - String similarity (Jaro-Winkler via rapidfuzz)
   - Afkortingscheck (dictionary lookup)
   ↓
5. LLM Matching (Ollama, qwen2.5:7b)
   - Alleen voor kandidaatparen met gecombineerde score > drempel
   - Few-shot prompt met NL beleidsvoorbeelden
   ↓
6. Graph-gebaseerde clustering (UnionFind met drempel)
   ↓
7. Canonicalisatie (langste/formele vorm als canonical)
   ↓
8. SurrealDB persistentie (merge-beslissingen, canonical labels, provenance)
```

### Bestanden aan te maken

```
pipelines/entity-filtering/src/entity_filtering/
├── deduplication/
│   └── semantic_blocker.py          ← NIEUW: UMAP + HDBSCAN blocking
├── resolution/
│   └── llm_matcher.py               ← NIEUW: LLM-gebaseerde matching via Ollama
├── data/
│   └── nl_policy_abbreviations.json  ← NIEUW: Afkortingswoordenboek
└── config.py                         ← AANPASSEN: SemanticBlockingConfig toevoegen
```

### Nieuwe dependencies

```toml
# In pipelines/entity-filtering/pyproject.toml
[project.optional-dependencies]
semantic = [
    "hdbscan>=0.8.33",
    "umap-learn>=0.5.4",
    "rapidfuzz>=3.0.0",
]
```
