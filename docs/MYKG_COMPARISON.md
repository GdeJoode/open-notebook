# myKG vs open-notebook — vergelijking + porting kandidaten

> **Bron**: lokale clone van `GdeJoode/mykg` (fork van `SenolIsci/mykg`) op
> `/mnt/e/repos/public/mykg/`, getoetst tegen open-notebook na de
> Phase 0-7 refactor (zie `REFACTOR_PLAN.md`).
> Datum: 2026-06-03.

## 1. Fundamenteel verschil in vorm

| Aspect | **myKG** | **open-notebook** |
|---|---|---|
| Vorm | Single Python CLI (`pip install mykg`) | Full-stack web app (6 docker containers) |
| Gebruik | Batch one-shot extraction → static output | Ongoing research workflow met UI |
| Persistence | Filesystem (session dirs) | SurrealDB (multi-notebook, chat history) |
| LOC | ~27k (incl. 48 testfiles) | ~80k+ (apps + packages + pipelines + frontend) |
| Maturiteit | v0.2.17 beta, 1 dev | v2.0.0, gerefactord, multi-service |
| Licentie | MIT | MIT |

Dit is geen apples-to-apples vergelijking — het zijn twee verschillende
soorten producten. myKG is een transformatie-tool; open-notebook is een
werkomgeving.

## 2. Featurematrix

| Capability | myKG | open-notebook |
|---|---|---|
| **Ingestion** | | |
| Markdown | ✅ passthrough | ✅ |
| PDF | ✅ via **MinerU** (zwaar, GPU) | ✅ via **docling** (lichter, GPU optioneel via service) |
| DOCX/PPTX/Images | ✅ MinerU | ✅ docling |
| HTML | ✅ markdownify | ✅ BeautifulSoup |
| **Audio (mp3/m4a)** | ❌ | ✅ WhisperX |
| **Extraction** | | |
| Ontology-guided | ✅ formele RDFS/OWL induction | ✅ YAML-gebaseerde ontology |
| **Two-pass (induce schema → extract)** | ✅ expliciet Pass 1 / Pass 2 | ❌ minder gescheiden |
| **TTL/RDFS export** (Protégé-edibel) | ✅ + rdflib validatie | ⚠️ recent toegevoegd in `rdf_owl_shacl.py` (basis) |
| **SKOS thesaurus support** | ✅ exact/closeMatch | ❌ |
| **Confidence scoring per node/edge** | ✅ systematisch 0.0-1.0 | ⚠️ wel in entity-filtering, niet overal |
| **Orphan-connection pass** | ✅ aparte stap (heuristic + LLM) | ❌ niet als losse fase |
| **Cross-session graph merge** | ✅ dedicated merger module | ❌ |
| **Append/incremental** | ✅ `--append` op bestaande session | ✅ jobs queue handelt dit |
| **Outputs** | | |
| JSONL (Neo4j/RAG ready) | ✅ | ❌ |
| Turtle RDF | ✅ verifiable + Protégé | ⚠️ via nieuwe `rdf_owl_shacl` |
| NetworkX (7 formaten) | ✅ | ⚠️ partial |
| **Obsidian vault** (wikilinked .md) | ✅ | ❌ |
| Interactive HTML graph | ✅ ingebouwd | ✅ via frontend |
| **Persistent DB (queries)** | ❌ filesystem | ✅ SurrealDB (SQL+graph) |
| **Real-time chat over KG** | ⚠️ via Claude Code skill | ✅ ingebouwd (RAG) |
| **LLM providers** | | |
| Anthropic API | ✅ | ✅ via esperanto/llm-manager |
| OpenAI | ✅ | ✅ |
| Ollama (lokaal) | ✅ | ✅ |
| OpenRouter | ✅ | ⚠️ via esperanto |
| `claude` CLI | ✅ subprocess adapter | ⚠️ niet expliciet |
| **Claude Code agent mode** | ✅ inbox/outbox filesystem | ❌ |
| **Architectuur** | | |
| Setup | `pip install mykg` | `docker compose up` (6 containers) |
| Workflow | session-based runs | persistent notebooks |
| Resumable pipeline | ✅ elke stap herstart | ✅ via job queue |
| Frontend UI | ❌ CLI + HTML output | ✅ Next.js |
| Multi-user / notebooks | ❌ | ✅ |

## 3. Antwoord op de centrale vraag

### Kan myKG out-of-the-box dienen als alternatief voor open-notebook?

**Alleen als de use-case fundamenteel verandert.**

- ✅ **Ja, kies myKG** wanneer de workflow is: `documents → run extraction → static graph + Obsidian vault → query via Claude Code`. Eleganter, lagere operationele complexiteit, geen draaiende stack.
- ❌ **Nee, kies myKG niet** wanneer je nodig hebt: persistente notebook-workflow met chat, audio-ingestie, web-UI voor ongoing onderzoek, Zotero-koppeling, of de gerefactorde architectuur die er nu staat.

### Beter: open-notebook upgraden met myKG features

**Ja, dit is de slimme route.** myKG heeft een aantal ideeën die porteerbaar
zijn naar open-notebook en de KG-kwaliteit een sprong vooruit helpen.

## 4. Porting kandidaten (geprioriteerd)

| Prio | Feature uit myKG | Waar in open-notebook | Effort | Status myKG-code |
|---|---|---|---|---|
| 🔴 P0 | **Two-pass extractie** (Pass 1 induce schema → Pass 2 extract instances) | uitbreiden van `pipelines/ontology-extraction` | medium | `src/mykg/pass1.py` + `pass2.py` |
| 🔴 P0 | **TTL/RDFS schema-export met Protégé-validatie** | finishen van `packages/ontology-manager/rdf_owl_shacl.py` | klein | `src/mykg/ttl_validator.py` direct porteerbaar |
| 🟡 P1 | **Obsidian vault export** | nieuwe `pipelines/obsidian-export/` of feature in app-main | medium | `src/mykg/exporter.py` |
| 🟡 P1 | **MinerU als optioneel PDF parser-alternatief** | nieuwe `services/mineru/` naast docling | klein-medium | `src/mykg/uv_venv.py` (ephemeral venv pattern) |
| 🟡 P1 | **Confidence scoring overal** (nu alleen entity-filtering) | uitrollen naar entity-extraction + retrieval | medium | scattered across `pass2.py` / `assembler.py` |
| 🟡 P1 | **Orphan-connection pass** als expliciete stap | uitbreiden van `pipelines/entity-filtering/resolution/` | klein | `src/mykg/orphan_connector.py` |
| 🟢 P2 | **SKOS thesaurus support** | uitbreiden ontology-manager | medium | `src/mykg/thesaurus.py` |
| 🟢 P2 | **Cross-session merge** | nieuwe `merge` API in app-main | medium | `src/mykg/merger.py` + `merge_orchestrator.py` |
| 🟢 P2 | **Claude Code agent adapter** (inbox/outbox) | herbruikbaar voor langlopende LLM workflows | klein | `src/mykg/llm/agent_adapter.py` |

## 5. MinerU als parser-alternatief — diepere kijk

MinerU is een academic-grade PDF/document parser die complementair kan zijn
aan het bestaande docling-service in open-notebook. De manier waarop myKG het
integreert is bovendien een patroon dat we kunnen overnemen.

### Wat MinerU goed doet (vs docling)

- **Wetenschappelijke PDFs**: betere herkenning van formules, tabellen
  met complexe layouts, en figuren met bijschriften.
- **Layout-bewust**: behoudt reading order in multi-column documenten beter.
- **Bredere formaatdekking**: PDF, DOCX, DOC, PPTX, PNG, JPG — alles via
  hetzelfde pipeline. Bij docling moet je per format-suffix iets anders
  configureren.
- **Native markdown output**: docling produceert eigen JSON-schema dat we
  daarna naar chunks moeten mappen; MinerU produceert direct gestructureerde
  markdown.

### Hoe myKG het integreert (overneembare pattern)

myKG installeert MinerU **niet** in z'n eigen interpreter. In plaats daarvan:

1. **Ephemeral `uv`-managed venv** per parse-call (Python 3.12 gepind via
   `uv venv --python 3.12`). De venv leeft in een TemporaryDirectory en wordt
   verwijderd na afloop, ook bij failures.
2. `uv pip install -U mineru[all]` binnen die venv (first-run download
   PyTorch + Ray, ~30 min — daarom de 1800s install_timeout).
3. MinerU draait als subprocess; output (.md) wordt teruggelezen door myKG.
4. Configuratie via `mykg_config.yaml`:

   ```yaml
   preprocess:
     enabled: true
     subdir: _preprocessed
     extra_args: []                # bv. ["--backend", "pipeline"]
     timeout_seconds: 1800
     uv_path: uv
     uv_python_version: "3.12"
     mineru_spec: mineru[all]
     install_timeout_seconds: 1800
     extensions: [.pdf, .docx, .doc, .pptx, .png, .jpg, .jpeg]
   ```

Bron: `src/mykg/uv_venv.py::ephemeral_mineru_venv`.

### Mogelijke integratie in open-notebook

Drie smaken, oplopend in invasiviteit:

1. **Standalone service**, parallel aan docling:
   `services/mineru/` met eigen Dockerfile (CUDA-base zoals whisperx).
   Routing op file-extension of explicit user-keuze. Past binnen het bestaande
   pattern (`USE_DOCLING_SERVICE=true` env var → vergelijkbaar
   `USE_MINERU_SERVICE` per source).
2. **Optionele backend binnen ingestion-pipeline**:
   `pipelines/ingestion/.../mineru_client.py` naast de bestaande
   `docling_http_client.py`. Dispatch in `_process_file`.
3. **Per-source override** via ContentSettings:
   `parser_engine: "docling" | "mineru"` veld in user-instellingen + UI.

Aanbeveling: **(1) standalone service** is conform de architectuur en
isoleert de zware MinerU install (PyTorch + Ray) van de andere services.

## 6. Concreet plan voor de top-3

### P0a — Two-pass extractie

1. Lees `src/mykg/pass1.py` en `pass2.py` in mykg.
2. Maak in `pipelines/ontology-extraction/`:
   - `pass1_schema_induction.py` — neemt corpus, returneert geinduceerde schema (RDFS).
   - `pass2_entity_extraction.py` — neemt schema + één document, returneert typed entities/relations.
3. Wire in workflow: schema-induction is een aparte stap die je optioneel met
   `--review` flag kunt pauzeren voor menselijke validatie.
4. Tests: port `tests/test_pass1.py`, `tests/test_pass2.py` patterns.

### P0b — TTL/RDFS export met validatie

1. Lees `src/mykg/ttl_validator.py` (rdflib validation).
2. Vervang/aanvul `packages/ontology-manager/.../rdf_owl_shacl.py` met de
   validatie-logica (let op: de bestaande rdflib-bug in dat bestand moet
   sowieso opgelost — zie REFACTOR_PLAN.md follow-up).
3. Export endpoint in app-main router: `GET /api/notebooks/{id}/schema.ttl`.

### P1a — MinerU service

1. Maak `services/mineru/Dockerfile` (CUDA-base, `mineru[all]` install).
2. HTTP API à la `services/docling/api.py` (parse endpoint).
3. `pipelines/ingestion/.../mineru_http_client.py` als client.
4. Dispatch in `SourceExtractor._process_file` op basis van content-settings of file-type.

---

## 7. Eindconclusie

Hou open-notebook als hoofdproduct, maar lees `src/mykg/pass1.py`, `pass2.py`,
`ttl_validator.py`, `schema_merge.py`, `exporter.py` en `uv_venv.py` — daar zit
een schema-discipline, output-rijkdom, en isolatie-pattern (ephemeral venvs
voor heavy deps zoals MinerU) die je direct kunt overnemen voor 2-4 weken werk
en je KG-kwaliteit een sprong vooruit helpt.

myKG-code is MIT-licensed, dus directe portering is wettelijk toegestaan —
geef SenolIsci attribution in de commit-message of file-header.
