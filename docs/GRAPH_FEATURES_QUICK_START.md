# SurrealDB Graph Features - Quick Start Guide

**Goal**: Add citation networks, entity extraction, and knowledge graph capabilities to Open Notebook in 6-8 weeks.

---

## 🚀 Week 1-2: Citation Networks (Start Here!)

### What You'll Build
- Sources can cite other sources (like academic papers)
- Track citation chains (papers citing papers that cite your paper)
- Find common citations between sources
- Visualize citation networks

### Quick Implementation Steps

**1. Add Citation Schema** (5 minutes)
```bash
# Run this migration
surreal sql --endpoint ws://localhost:8000 --username root --password root

DEFINE TABLE cites SCHEMAFULL;
DEFINE FIELD in ON TABLE cites TYPE record<source>;
DEFINE FIELD out ON TABLE cites TYPE record<source>;
DEFINE FIELD citation_type ON TABLE cites TYPE string;
DEFINE FIELD citation_context ON TABLE cites TYPE string;
DEFINE FIELD confidence ON TABLE cites TYPE float;
DEFINE FIELD created ON TABLE cites TYPE datetime;
DEFINE INDEX cites_in_out ON TABLE cites COLUMNS in, out;
```

**2. Create Citation Model** (copy from guide)
- File: `open_notebook/domain/citation.py`
- Functions: `create_citation()`, `get_citations_for_source()`, `get_citation_chain()`

**3. Add API Endpoint** (copy from guide)
- File: `api/routers/sources.py`
- Endpoint: `POST /sources/{source_id}/extract-citations`

**4. Add Frontend Component** (copy from guide)
- File: `frontend/src/components/source/CitationNetwork.tsx`
- Shows: "Cites" and "Cited By" lists with extraction button

**Test It**:
```bash
# Extract citations from a paper
curl -X POST http://localhost:5055/api/sources/{source_id}/extract-citations

# View citations
curl http://localhost:5055/api/sources/{source_id}/citations
```

---

## 🧠 Week 3-4: Entity Extraction

### What You'll Build
- Extract entities (people, organizations, concepts) from documents
- Track entity mentions across sources
- Link entities with relationships (works_at, located_in, etc.)
- Search entities semantically

### Quick Implementation Steps

**1. Add Entity Schema** (5 minutes)
```sql
DEFINE TABLE entity SCHEMAFULL;
DEFINE FIELD name ON TABLE entity TYPE string;
DEFINE FIELD entity_type ON TABLE entity TYPE string;
DEFINE FIELD description ON TABLE entity TYPE string;
DEFINE FIELD metadata ON TABLE entity TYPE object;
DEFINE FIELD embedding ON TABLE entity TYPE array;
DEFINE INDEX entity_name_type ON TABLE entity COLUMNS name, entity_type UNIQUE;

DEFINE TABLE mentions SCHEMAFULL;
DEFINE FIELD in ON TABLE mentions TYPE record<source>;
DEFINE FIELD out ON TABLE mentions TYPE record<entity>;
DEFINE FIELD mention_count ON TABLE mentions TYPE int DEFAULT 1;
DEFINE FIELD contexts ON TABLE mentions TYPE array<string>;

DEFINE TABLE related_to SCHEMAFULL;
DEFINE FIELD in ON TABLE related_to TYPE record<entity>;
DEFINE FIELD out ON TABLE related_to TYPE record<entity>;
DEFINE FIELD relationship_type ON TABLE related_to TYPE string;
DEFINE FIELD confidence ON TABLE related_to TYPE float;
```

**2. Create Entity Model** (copy from guide)
- File: `open_notebook/domain/entity.py`
- Class: `Entity` with methods for mentions and relationships

**3. Create Entity Extractor** (copy from guide)
- File: `open_notebook/utils/entity_extractor.py`
- Function: `process_source_entities()` - uses LLM to extract entities

**4. Add API Endpoints** (copy from guide)
- File: `api/routers/entities.py`
- Endpoint: `POST /entities/sources/{source_id}/extract`

**Test It**:
```bash
# Extract entities from a document
curl -X POST http://localhost:5055/api/entities/sources/{source_id}/extract

# View entities in source
curl http://localhost:5055/api/entities/sources/{source_id}

# View sources mentioning an entity
curl http://localhost:5055/api/entities/{entity_id}/sources
```

---

## 📈 Week 5-6: Recommendations

### What You'll Build
- Recommend sources based on citation networks
- Recommend sources based on shared entities
- Recommend sources based on related entities
- Personalized recommendations for users

### Quick Implementation Steps

**1. Add User Tracking** (5 minutes)
```sql
DEFINE TABLE reads SCHEMAFULL;
DEFINE FIELD in ON TABLE reads TYPE record<user>;
DEFINE FIELD out ON TABLE reads TYPE record<source>;
DEFINE FIELD timestamp ON TABLE reads TYPE datetime;

DEFINE TABLE interested_in SCHEMAFULL;
DEFINE FIELD in ON TABLE interested_in TYPE record<user>;
DEFINE FIELD out ON TABLE interested_in TYPE record<entity>;
DEFINE FIELD interest_score ON TABLE interested_in TYPE float DEFAULT 1.0;
```

**2. Create Recommender** (copy from guide)
- File: `open_notebook/utils/recommender.py`
- Functions: `recommend_sources_by_citation()`, `recommend_sources_by_entities()`

**3. Add API Endpoints** (copy from guide)
- File: `api/routers/recommendations.py`

**Test It**:
```bash
# Get recommendations for a source
curl http://localhost:5055/api/recommendations/sources/{source_id}/by-citation
curl http://localhost:5055/api/recommendations/sources/{source_id}/by-entities
```

---

## 🎨 Week 7-8: Graph Visualization

### What You'll Build
- Interactive force-directed graph visualization
- Shows citations, entities, and relationships
- Click nodes to navigate
- Zoom, pan, fit-to-view controls

### Quick Implementation Steps

**1. Install Frontend Dependency**
```bash
cd frontend
npm install react-force-graph-2d
```

**2. Add Graph Viewer Component** (copy from guide)
- File: `frontend/src/components/graph/KnowledgeGraphViewer.tsx`
- Interactive visualization with zoom/pan controls

**3. Add to Source Detail Page**
```typescript
import { KnowledgeGraphViewer } from '@/components/graph/KnowledgeGraphViewer';

// In component
<KnowledgeGraphViewer sourceId={sourceId} />
```

**Test It**: Visit a source detail page and see the interactive graph!

---

## 🔑 Key SurrealDB Graph Features You'll Use

### 1. RELATE Statement (Create Relationships)
```sql
-- Create citation relationship
RELATE source:paper1->cites->source:paper2
CONTENT {
  citation_context: "Referenced in section 3.2",
  confidence: 0.95
};
```

### 2. Graph Traversal (Follow Relationships)
```sql
-- Get papers cited by this paper
SELECT ->cites->source.* FROM source:paper1;

-- Get papers citing this paper
SELECT <-cites<-source.* FROM source:paper1;

-- 2-hop: Papers citing papers that cite this paper
SELECT <-cites<-source->cites->source.* FROM source:paper1;
```

### 3. Relationship Filtering
```sql
-- Find papers citing this paper with high confidence
SELECT <-cites<-source.* FROM source:paper1
WHERE <-cites.confidence > 0.8;
```

### 4. Aggregations on Relationships
```sql
-- Count citations per source
SELECT out.title, COUNT() as citation_count
FROM cites
GROUP BY out
ORDER BY citation_count DESC;
```

### 5. Complex Graph Queries
```sql
-- Find sources mentioning the same entities
LET $source_entities = (
  SELECT out FROM mentions WHERE in = source:doc1
);

SELECT in as similar_source, COUNT() as common_entities
FROM mentions
WHERE out IN $source_entities
AND in != source:doc1
GROUP BY in
ORDER BY common_entities DESC;
```

---

## 📊 Expected Results by Phase

### After Phase 1 (Citations)
- Users can see what sources cite each document
- Users can see what sources are cited by each document
- Citation extraction button on source detail pages
- Citation network visualization (basic)

### After Phase 2 (Entities)
- Entities automatically extracted from documents
- Entity detail pages showing all mentions
- Entity relationship graphs
- Search entities semantically

### After Phase 3 (Recommendations)
- "You might also like" based on citations
- "Related sources" based on shared entities
- Personalized recommendations on homepage
- Smart entity-based exploration

### After Phase 4 (Visualization)
- Interactive knowledge graph on every source
- Visual exploration of entity relationships
- Click-to-navigate graph nodes
- Beautiful, informative visualizations

---

## 🎯 Success Metrics

**Technical**:
- Citation extraction accuracy: >80%
- Entity extraction accuracy: >75%
- Graph query performance: <500ms for 2-hop traversals
- Recommendation relevance: User click-through >20%

**User Impact**:
- Users discover 3x more related sources
- Time to find relevant information reduced by 40%
- User engagement with knowledge base increased by 50%
- Feature becomes #1 differentiator vs competitors

---

## 🚨 Important Notes

### Performance
- Entity extraction is CPU-intensive; consider background queue
- Graph traversals >3 hops can be slow; add caching
- Limit graph visualization to <100 nodes for smooth UX

### LLM Usage
- Citation extraction: ~2-5K tokens per document
- Entity extraction: ~3-8K tokens per document
- Relationship extraction: ~2-5K tokens per entity set
- Consider cost: ~$0.01-0.05 per document with GPT-4

### Data Quality
- LLM extraction confidence scores help filter low-quality results
- Manual review tools for entity/citation corrections recommended
- Fuzzy matching needed for entity deduplication

---

## 🔧 Troubleshooting

### Common Issues

**Q: Entity extraction creates duplicates**
A: Improve entity matching in `match_citation_to_existing_source()` - add fuzzy string matching.

**Q: Graph queries timeout**
A: Add query timeouts, limit depth, ensure indexes exist on relationship tables.

**Q: Frontend graph too cluttered**
A: Implement filtering (top-N connections only), add UI controls for node types.

**Q: LLM returns invalid JSON**
A: Add response cleaning in `extract_entities_from_text()` - strip markdown code blocks.

### Debug Queries

```sql
-- Check citation count
SELECT COUNT() FROM cites;

-- Most cited sources
SELECT out.title, COUNT() as citations
FROM cites
GROUP BY out
ORDER BY citations DESC
LIMIT 10
FETCH out;

-- Most mentioned entities
SELECT out.name, SUM(mention_count) as total
FROM mentions
GROUP BY out
ORDER BY total DESC
LIMIT 10
FETCH out;

-- Entity relationship types
SELECT relationship_type, COUNT() as count
FROM related_to
GROUP BY relationship_type;
```

---

## 📚 Next Steps

1. **Read Full Guide**: `docs/GRAPH_FEATURES_IMPLEMENTATION_GUIDE.md` (110 pages)
2. **Start with Phase 1**: Citation networks (simplest, immediate value)
3. **Test Thoroughly**: Use sample documents for each phase
4. **Iterate**: Gather user feedback, refine extraction accuracy
5. **Scale**: Add background processing, caching, optimization

---

## 🎉 Why This Is Powerful

**Before**: Documents in isolation, manual exploration

**After**: Connected knowledge graph with:
- Automatic relationship discovery
- Smart recommendations
- Visual exploration
- Semantic search across entities
- Citation chain analysis

**Your Competitive Advantage**: RAGFlow and SurfSense don't have this level of graph integration. You're using SurrealDB's multi-model capabilities to their full potential—document storage + vector search + graph database—all in one, with zero infrastructure complexity.

**Get Started**: Copy Phase 1 code from the full guide and add your first citation network this week! 🚀
