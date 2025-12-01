# SurrealDB Graph Features Implementation Guide

**Status**: Implementation-ready guide for adding graph capabilities to Open Notebook
**Estimated Total Effort**: 6-8 weeks
**Priority**: High (unlocks knowledge graph features without infrastructure complexity)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Phase 1: Citation Networks (Week 1-2)](#phase-1-citation-networks)
3. [Phase 2: Entity Extraction & Relationships (Week 3-4)](#phase-2-entity-extraction--relationships)
4. [Phase 3: Knowledge Graph Recommendations (Week 5-6)](#phase-3-knowledge-graph-recommendations)
5. [Phase 4: Graph Visualization (Week 7-8)](#phase-4-graph-visualization)
6. [Database Schema](#database-schema)
7. [Testing Strategy](#testing-strategy)

---

## Architecture Overview

### Current State
```
Notebook ──reference──> Source
Notebook ──artifact──> Note
ChatSession ──refers_to──> Notebook/Source
```

### Target State (After Implementation)
```
Notebook ──reference──> Source ──cites──> Source
                          │
                          ├──mentions──> Entity ──related_to──> Entity
                          │                │
                          │                ├──works_at──> Entity (org)
                          │                └──located_in──> Entity (place)
                          │
User ────read──────────> Source
  │                        │
  └──interested_in──> Entity
                        │
Chunk ──contains──> Entity
```

### Design Principles

1. **Non-Breaking**: All changes are additive, existing functionality unchanged
2. **Async-First**: All graph operations use async/await
3. **Idempotent**: Operations can be safely retried
4. **Performant**: Use SurrealDB's graph traversal (not joins)
5. **Type-Safe**: Leverage Pydantic models throughout

---

## Phase 1: Citation Networks (Week 1-2)

### Overview
Track citation relationships between sources (papers cite other papers, articles reference other articles).

### 1.1 Database Schema

**New Tables**:
```sql
-- Citation relationship (edge table)
DEFINE TABLE cites SCHEMAFULL;
DEFINE FIELD in ON TABLE cites TYPE record<source>;
DEFINE FIELD out ON TABLE cites TYPE record<source>;
DEFINE FIELD citation_type ON TABLE cites TYPE string;  -- 'direct', 'indirect', 'related'
DEFINE FIELD citation_context ON TABLE cites TYPE string;  -- Quote or context where cited
DEFINE FIELD confidence ON TABLE cites TYPE float;  -- 0.0-1.0
DEFINE FIELD created ON TABLE cites TYPE datetime;

-- Index for fast lookups
DEFINE INDEX cites_in_out ON TABLE cites COLUMNS in, out;
```

### 1.2 Domain Model

Create new file: `open_notebook/domain/citation.py`

```python
from typing import ClassVar, List, Optional
from pydantic import Field
from surrealdb import RecordID

from open_notebook.domain.base import ObjectModel
from open_notebook.database.repository import ensure_record_id, repo_query


class Citation(ObjectModel):
    """Represents a citation relationship between two sources."""
    table_name: ClassVar[str] = "cites"

    # in/out are automatically created by RELATE
    citation_type: str = Field(
        default="direct",
        description="Type: direct, indirect, related"
    )
    citation_context: Optional[str] = Field(
        None,
        description="Quote or context where source is cited"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for citation extraction"
    )


async def create_citation(
    citing_source_id: str,
    cited_source_id: str,
    citation_type: str = "direct",
    citation_context: Optional[str] = None,
    confidence: float = 1.0,
) -> Citation:
    """Create a citation relationship between two sources."""
    from open_notebook.database.repository import repo_relate

    result = await repo_relate(
        source=citing_source_id,
        relationship="cites",
        target=cited_source_id,
        data={
            "citation_type": citation_type,
            "citation_context": citation_context,
            "confidence": confidence,
        },
    )
    return Citation(**result[0]) if result else None


async def get_citations_for_source(source_id: str) -> List[dict]:
    """Get all sources that this source cites (outgoing citations)."""
    result = await repo_query(
        """
        SELECT
            out as cited_source,
            citation_type,
            citation_context,
            confidence
        FROM cites
        WHERE in = $source_id
        FETCH cited_source
        """,
        {"source_id": ensure_record_id(source_id)},
    )
    return result


async def get_citing_sources(source_id: str) -> List[dict]:
    """Get all sources that cite this source (incoming citations)."""
    result = await repo_query(
        """
        SELECT
            in as citing_source,
            citation_type,
            citation_context,
            confidence
        FROM cites
        WHERE out = $source_id
        FETCH citing_source
        """,
        {"source_id": ensure_record_id(source_id)},
    )
    return result


async def get_citation_chain(source_id: str, depth: int = 2) -> List[dict]:
    """
    Get citation chains (papers citing papers that cite this paper).
    Depth 1 = direct citations, 2 = citations of citations, etc.
    """
    if depth < 1:
        raise ValueError("Depth must be at least 1")

    # Build dynamic query based on depth
    arrow_chain = "->cites->source" * depth

    result = await repo_query(
        f"""
        SELECT {arrow_chain}.* as sources
        FROM $source_id
        """,
        {"source_id": ensure_record_id(source_id)},
    )
    return result


async def find_common_citations(source_id1: str, source_id2: str) -> List[dict]:
    """Find sources cited by both source1 and source2."""
    result = await repo_query(
        """
        SELECT out as common_citation
        FROM cites
        WHERE in = $source1
        AND out IN (
            SELECT out FROM cites WHERE in = $source2
        )
        FETCH common_citation
        """,
        {
            "source1": ensure_record_id(source_id1),
            "source2": ensure_record_id(source_id2),
        },
    )
    return result
```

### 1.3 Citation Extraction Service

Create: `open_notebook/utils/citation_extractor.py`

```python
from typing import List, Dict, Optional
from loguru import logger
import re

from open_notebook.domain.models import model_manager
from open_notebook.domain.notebook import Source
from open_notebook.domain.citation import create_citation


async def extract_citations_from_source(source: Source) -> List[Dict]:
    """
    Extract citations from a source document using LLM.
    Returns list of dicts with: {title, authors, year, context}
    """
    if not source.full_text:
        logger.warning(f"No text to extract citations from for source {source.id}")
        return []

    model = await model_manager.get_model("LONG_CONTEXT_LLM")
    if not model:
        logger.error("No LLM configured for citation extraction")
        return []

    prompt = f"""Extract all citations from the following document.
For each citation, provide:
- Title (if available)
- Authors (if available)
- Year (if available)
- Context (the sentence or paragraph where it's cited)

Document:
{source.full_text[:10000]}  # Truncate for token limits

Return as JSON array: [{{"title": "...", "authors": "...", "year": "...", "context": "..."}}]
"""

    try:
        response = await model.ainvoke(prompt)
        # Parse response (assuming JSON format)
        import json
        citations = json.loads(response)
        return citations
    except Exception as e:
        logger.error(f"Failed to extract citations: {e}")
        return []


async def match_citation_to_existing_source(
    citation: Dict,
    notebook_id: Optional[str] = None
) -> Optional[str]:
    """
    Match a citation to an existing source in the database.
    Uses fuzzy title matching and metadata comparison.
    """
    from open_notebook.database.repository import repo_query

    title = citation.get("title")
    if not title:
        return None

    # Search for sources with similar titles
    query = """
        SELECT id, title, full_text
        FROM source
        WHERE title @@ $title
        LIMIT 5
    """

    results = await repo_query(query, {"title": title})

    if not results:
        return None

    # Use LLM to confirm match (fuzzy matching)
    model = await model_manager.get_model("FAST_LLM")
    for result in results:
        prompt = f"""Are these the same paper/article?

Citation: {citation}
Source: {result.get('title')}

Answer with just 'yes' or 'no'.
"""
        response = await model.ainvoke(prompt)
        if "yes" in response.lower():
            return result["id"]

    return None


async def create_citation_network_for_source(
    source_id: str,
    notebook_id: Optional[str] = None
) -> int:
    """
    Extract citations from a source and create graph relationships.
    Returns count of citations created.
    """
    from open_notebook.domain.base import ObjectModel

    source = await ObjectModel.get(source_id)

    # Extract citations using LLM
    citations = await extract_citations_from_source(source)
    logger.info(f"Extracted {len(citations)} citations from source {source_id}")

    created_count = 0
    for citation in citations:
        # Try to match to existing source
        cited_source_id = await match_citation_to_existing_source(
            citation, notebook_id
        )

        if cited_source_id:
            # Create citation relationship
            await create_citation(
                citing_source_id=source_id,
                cited_source_id=cited_source_id,
                citation_type="direct",
                citation_context=citation.get("context"),
                confidence=0.8,  # LLM-based extraction confidence
            )
            created_count += 1
            logger.info(f"Created citation: {source_id} -> {cited_source_id}")

    return created_count
```

### 1.4 API Endpoints

Add to: `api/routers/sources.py`

```python
from open_notebook.domain.citation import (
    get_citations_for_source,
    get_citing_sources,
    get_citation_chain,
    find_common_citations,
)
from open_notebook.utils.citation_extractor import (
    create_citation_network_for_source
)


@router.post("/sources/{source_id}/extract-citations")
async def extract_citations(source_id: str, notebook_id: Optional[str] = None):
    """Extract citations from source and create citation network."""
    try:
        count = await create_citation_network_for_source(source_id, notebook_id)
        return {"status": "success", "citations_created": count}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/{source_id}/citations")
async def get_source_citations(source_id: str):
    """Get all sources cited by this source."""
    try:
        citations = await get_citations_for_source(source_id)
        return {"citations": citations}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/{source_id}/cited-by")
async def get_sources_citing(source_id: str):
    """Get all sources that cite this source."""
    try:
        citing = await get_citing_sources(source_id)
        return {"cited_by": citing}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/{source_id}/citation-chain")
async def get_source_citation_chain(source_id: str, depth: int = 2):
    """Get multi-hop citation chain."""
    try:
        chain = await get_citation_chain(source_id, depth)
        return {"chain": chain}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
```

### 1.5 Database Migration

Create: `migrations/add_citation_network.surql`

```sql
-- Citation relationship table
DEFINE TABLE cites SCHEMAFULL;
DEFINE FIELD in ON TABLE cites TYPE record<source>;
DEFINE FIELD out ON TABLE cites TYPE record<source>;
DEFINE FIELD citation_type ON TABLE cites TYPE string;
DEFINE FIELD citation_context ON TABLE cites TYPE string;
DEFINE FIELD confidence ON TABLE cites TYPE float;
DEFINE FIELD created ON TABLE cites TYPE datetime;

-- Indexes
DEFINE INDEX cites_in_out ON TABLE cites COLUMNS in, out;
DEFINE INDEX cites_out ON TABLE cites COLUMNS out;

-- Permissions (optional - adjust based on your auth setup)
DEFINE FIELD created ON TABLE cites VALUE time::now() READONLY;
```

Run migration:
```bash
# Add to your migration runner
python -c "
from open_notebook.database.repository import repo_query
import asyncio

async def run_migration():
    with open('migrations/add_citation_network.surql', 'r') as f:
        migration_sql = f.read()
    await repo_query(migration_sql)
    print('Citation network migration complete')

asyncio.run(run_migration())
"
```

### 1.6 Frontend Components

Create: `frontend/src/components/source/CitationNetwork.tsx`

```typescript
import React, { useEffect, useState } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { ArrowRight, ArrowLeft, Network } from 'lucide-react';

interface Citation {
  cited_source: {
    id: string;
    title: string;
  };
  citation_type: string;
  citation_context?: string;
  confidence: number;
}

export function CitationNetwork({ sourceId }: { sourceId: string }) {
  const [citations, setCitations] = useState<Citation[]>([]);
  const [citedBy, setCitedBy] = useState<Citation[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCitations = async () => {
      setLoading(true);
      try {
        const [citationsRes, citedByRes] = await Promise.all([
          fetch(`/api/sources/${sourceId}/citations`),
          fetch(`/api/sources/${sourceId}/cited-by`),
        ]);

        const citationsData = await citationsRes.json();
        const citedByData = await citedByRes.json();

        setCitations(citationsData.citations || []);
        setCitedBy(citedByData.cited_by || []);
      } catch (error) {
        console.error('Failed to fetch citations:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchCitations();
  }, [sourceId]);

  const extractCitations = async () => {
    try {
      const response = await fetch(`/api/sources/${sourceId}/extract-citations`, {
        method: 'POST',
      });
      const data = await response.json();
      alert(`Extracted ${data.citations_created} citations`);
      // Refresh citations
      window.location.reload();
    } catch (error) {
      console.error('Failed to extract citations:', error);
    }
  };

  if (loading) {
    return <div>Loading citation network...</div>;
  }

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle className="flex items-center gap-2">
              <Network className="h-5 w-5" />
              Citation Network
            </CardTitle>
            <Button onClick={extractCitations} variant="outline" size="sm">
              Extract Citations
            </Button>
          </div>
        </CardHeader>
        <CardContent>
          {/* Sources this document cites */}
          <div className="mb-6">
            <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
              <ArrowRight className="h-4 w-4" />
              Cites ({citations.length})
            </h3>
            {citations.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No citations found. Click "Extract Citations" to analyze this document.
              </p>
            ) : (
              <ul className="space-y-2">
                {citations.map((citation, idx) => (
                  <li key={idx} className="border-l-2 pl-3 py-1">
                    <a
                      href={`/sources/${citation.cited_source.id}`}
                      className="text-sm font-medium hover:underline"
                    >
                      {citation.cited_source.title}
                    </a>
                    {citation.citation_context && (
                      <p className="text-xs text-muted-foreground mt-1">
                        "{citation.citation_context.substring(0, 100)}..."
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>

          {/* Sources that cite this document */}
          <div>
            <h3 className="text-sm font-semibold mb-2 flex items-center gap-2">
              <ArrowLeft className="h-4 w-4" />
              Cited By ({citedBy.length})
            </h3>
            {citedBy.length === 0 ? (
              <p className="text-sm text-muted-foreground">
                No sources cite this document yet.
              </p>
            ) : (
              <ul className="space-y-2">
                {citedBy.map((citation, idx) => (
                  <li key={idx} className="border-l-2 pl-3 py-1">
                    <a
                      href={`/sources/${citation.citing_source.id}`}
                      className="text-sm font-medium hover:underline"
                    >
                      {citation.citing_source.title}
                    </a>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
```

Add to source detail page: `frontend/src/app/(dashboard)/sources/[id]/page.tsx`

```typescript
import { CitationNetwork } from '@/components/source/CitationNetwork';

// Inside the component
<CitationNetwork sourceId={sourceId} />
```

---

## Phase 2: Entity Extraction & Relationships (Week 3-4)

### Overview
Extract entities (people, organizations, locations, concepts) from documents and create knowledge graph relationships.

### 2.1 Database Schema

Create: `migrations/add_entity_graph.surql`

```sql
-- Entity table
DEFINE TABLE entity SCHEMAFULL;
DEFINE FIELD name ON TABLE entity TYPE string;
DEFINE FIELD entity_type ON TABLE entity TYPE string;  -- person, organization, location, concept
DEFINE FIELD description ON TABLE entity TYPE string;
DEFINE FIELD metadata ON TABLE entity TYPE object;  -- flexible JSON for entity-specific data
DEFINE FIELD embedding ON TABLE entity TYPE array;  -- vector for similarity search
DEFINE FIELD created ON TABLE entity TYPE datetime;
DEFINE FIELD updated ON TABLE entity TYPE datetime;

-- Indexes
DEFINE INDEX entity_name ON TABLE entity COLUMNS name;
DEFINE INDEX entity_type ON TABLE entity COLUMNS entity_type;
DEFINE INDEX entity_name_type ON TABLE entity COLUMNS name, entity_type UNIQUE;

-- Relationship: Source mentions Entity
DEFINE TABLE mentions SCHEMAFULL;
DEFINE FIELD in ON TABLE mentions TYPE record<source>;
DEFINE FIELD out ON TABLE mentions TYPE record<entity>;
DEFINE FIELD mention_count ON TABLE mentions TYPE int DEFAULT 1;
DEFINE FIELD contexts ON TABLE mentions TYPE array<string>;  -- Where entity was mentioned
DEFINE FIELD created ON TABLE mentions TYPE datetime;

-- Relationship: Chunk contains Entity
DEFINE TABLE contains_entity SCHEMAFULL;
DEFINE FIELD in ON TABLE contains_entity TYPE record<chunk>;
DEFINE FIELD out ON TABLE contains_entity TYPE record<entity>;
DEFINE FIELD positions ON TABLE contains_entity TYPE array;  -- bounding boxes where entity appears
DEFINE FIELD created ON TABLE contains_entity TYPE datetime;

-- Relationship: Entity relationships
DEFINE TABLE related_to SCHEMAFULL;
DEFINE FIELD in ON TABLE related_to TYPE record<entity>;
DEFINE FIELD out ON TABLE related_to TYPE record<entity>;
DEFINE FIELD relationship_type ON TABLE related_to TYPE string;  -- works_at, located_in, founded_by, etc.
DEFINE FIELD confidence ON TABLE related_to TYPE float;
DEFINE FIELD created ON TABLE related_to TYPE datetime;

-- Specific relationship types (optional - provides stronger typing)
DEFINE TABLE works_at SCHEMAFULL;
DEFINE FIELD in ON TABLE works_at TYPE record<entity>;  -- person
DEFINE FIELD out ON TABLE works_at TYPE record<entity>;  -- organization
DEFINE FIELD role ON TABLE works_at TYPE string;
DEFINE FIELD start_date ON TABLE works_at TYPE string;
DEFINE FIELD end_date ON TABLE works_at TYPE string;

DEFINE TABLE located_in SCHEMAFULL;
DEFINE FIELD in ON TABLE located_in TYPE record<entity>;  -- organization or person
DEFINE FIELD out ON TABLE located_in TYPE record<entity>;  -- location

-- User interest tracking
DEFINE TABLE interested_in SCHEMAFULL;
DEFINE FIELD in ON TABLE interested_in TYPE record<user>;
DEFINE FIELD out ON TABLE interested_in TYPE record<entity>;
DEFINE FIELD interest_score ON TABLE interested_in TYPE float DEFAULT 1.0;
DEFINE FIELD last_interaction ON TABLE interested_in TYPE datetime;
```

### 2.2 Domain Models

Create: `open_notebook/domain/entity.py`

```python
from typing import ClassVar, List, Optional, Dict, Any, Literal
from pydantic import Field
from loguru import logger

from open_notebook.domain.base import ObjectModel
from open_notebook.database.repository import ensure_record_id, repo_query, repo_relate


EntityType = Literal["person", "organization", "location", "concept", "event"]


class Entity(ObjectModel):
    """Represents a named entity extracted from documents."""
    table_name: ClassVar[str] = "entity"

    name: str = Field(description="Entity name")
    entity_type: EntityType = Field(description="Type of entity")
    description: Optional[str] = Field(None, description="Entity description")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional entity metadata"
    )
    embedding: Optional[List[float]] = Field(None, description="Entity embedding for search")

    async def get_sources(self) -> List[Dict]:
        """Get all sources that mention this entity."""
        result = await repo_query(
            """
            SELECT
                in as source,
                mention_count,
                contexts
            FROM mentions
            WHERE out = $entity_id
            FETCH source
            ORDER BY mention_count DESC
            """,
            {"entity_id": ensure_record_id(self.id)},
        )
        return result

    async def get_related_entities(
        self,
        relationship_type: Optional[str] = None
    ) -> List[Dict]:
        """Get entities related to this entity."""
        if relationship_type:
            query = """
                SELECT
                    out as related_entity,
                    relationship_type,
                    confidence
                FROM related_to
                WHERE in = $entity_id AND relationship_type = $rel_type
                FETCH related_entity
            """
            params = {
                "entity_id": ensure_record_id(self.id),
                "rel_type": relationship_type
            }
        else:
            query = """
                SELECT
                    out as related_entity,
                    relationship_type,
                    confidence
                FROM related_to
                WHERE in = $entity_id
                FETCH related_entity
            """
            params = {"entity_id": ensure_record_id(self.id)}

        result = await repo_query(query, params)
        return result

    async def add_mention(
        self,
        source_id: str,
        context: str,
        increment: bool = True
    ):
        """Add or update a mention relationship between source and entity."""
        # Check if relationship already exists
        existing = await repo_query(
            """
            SELECT * FROM mentions
            WHERE in = $source_id AND out = $entity_id
            """,
            {
                "source_id": ensure_record_id(source_id),
                "entity_id": ensure_record_id(self.id),
            },
        )

        if existing:
            # Update existing mention
            if increment:
                await repo_query(
                    """
                    UPDATE mentions
                    SET
                        mention_count = mention_count + 1,
                        contexts = array::append(contexts, $context)
                    WHERE in = $source_id AND out = $entity_id
                    """,
                    {
                        "source_id": ensure_record_id(source_id),
                        "entity_id": ensure_record_id(self.id),
                        "context": context,
                    },
                )
        else:
            # Create new mention
            await repo_relate(
                source=source_id,
                relationship="mentions",
                target=self.id,
                data={
                    "mention_count": 1,
                    "contexts": [context],
                },
            )

    async def relate_to_entity(
        self,
        target_entity_id: str,
        relationship_type: str,
        confidence: float = 1.0,
    ):
        """Create a relationship between this entity and another entity."""
        await repo_relate(
            source=self.id,
            relationship="related_to",
            target=target_entity_id,
            data={
                "relationship_type": relationship_type,
                "confidence": confidence,
            },
        )


async def get_or_create_entity(
    name: str,
    entity_type: EntityType,
    description: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> Entity:
    """Get existing entity or create new one."""
    # Check if entity exists
    existing = await repo_query(
        """
        SELECT * FROM entity
        WHERE name = $name AND entity_type = $entity_type
        LIMIT 1
        """,
        {"name": name, "entity_type": entity_type},
    )

    if existing:
        return Entity(**existing[0])

    # Create new entity
    entity = Entity(
        name=name,
        entity_type=entity_type,
        description=description,
        metadata=metadata or {},
    )
    await entity.save()
    return entity


async def find_entities_by_source(source_id: str) -> List[Entity]:
    """Get all entities mentioned in a source."""
    result = await repo_query(
        """
        SELECT out as entity
        FROM mentions
        WHERE in = $source_id
        FETCH entity
        ORDER BY mention_count DESC
        """,
        {"source_id": ensure_record_id(source_id)},
    )
    return [Entity(**r["entity"]) for r in result] if result else []


async def search_entities(
    query: str,
    entity_type: Optional[EntityType] = None,
    limit: int = 10,
) -> List[Entity]:
    """Search entities by name or embedding similarity."""
    from open_notebook.domain.models import model_manager

    # Get embedding for query
    EMBEDDING_MODEL = await model_manager.get_embedding_model()
    if not EMBEDDING_MODEL:
        logger.warning("No embedding model available for entity search")
        # Fallback to text search
        if entity_type:
            query_str = """
                SELECT * FROM entity
                WHERE name @@ $query AND entity_type = $entity_type
                LIMIT $limit
            """
            params = {"query": query, "entity_type": entity_type, "limit": limit}
        else:
            query_str = """
                SELECT * FROM entity
                WHERE name @@ $query
                LIMIT $limit
            """
            params = {"query": query, "limit": limit}

        result = await repo_query(query_str, params)
        return [Entity(**r) for r in result] if result else []

    # Vector search
    embedding = (await EMBEDDING_MODEL.aembed([query]))[0]

    if entity_type:
        query_str = """
            SELECT *, vector::similarity(embedding, $embedding) as score
            FROM entity
            WHERE entity_type = $entity_type
            AND score > 0.5
            ORDER BY score DESC
            LIMIT $limit
        """
        params = {
            "embedding": embedding,
            "entity_type": entity_type,
            "limit": limit
        }
    else:
        query_str = """
            SELECT *, vector::similarity(embedding, $embedding) as score
            FROM entity
            WHERE score > 0.5
            ORDER BY score DESC
            LIMIT $limit
        """
        params = {"embedding": embedding, "limit": limit}

    result = await repo_query(query_str, params)
    return [Entity(**r) for r in result] if result else []
```

### 2.3 Entity Extraction Service

Create: `open_notebook/utils/entity_extractor.py`

```python
from typing import List, Dict, Tuple
from loguru import logger
import json

from open_notebook.domain.models import model_manager
from open_notebook.domain.notebook import Source, Chunk
from open_notebook.domain.entity import (
    Entity,
    EntityType,
    get_or_create_entity,
)


async def extract_entities_from_text(text: str) -> List[Dict]:
    """
    Extract named entities from text using LLM.
    Returns list of dicts with: {name, type, description, confidence}
    """
    model = await model_manager.get_model("LONG_CONTEXT_LLM")
    if not model:
        logger.error("No LLM configured for entity extraction")
        return []

    prompt = f"""Extract all named entities from the following text.
For each entity, provide:
- name: The entity name
- type: One of [person, organization, location, concept, event]
- description: Brief description (1 sentence)
- confidence: Your confidence score (0.0-1.0)

Text:
{text[:8000]}  # Truncate for token limits

Return as JSON array: [{{"name": "...", "type": "...", "description": "...", "confidence": 0.9}}]
Only return the JSON, no other text.
"""

    try:
        response = await model.ainvoke(prompt)
        # Clean response (remove markdown code blocks if present)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        entities = json.loads(response)
        return entities
    except Exception as e:
        logger.error(f"Failed to extract entities: {e}")
        logger.debug(f"Response was: {response}")
        return []


async def extract_entity_relationships(
    entities: List[Entity],
    text: str
) -> List[Dict]:
    """
    Extract relationships between entities using LLM.
    Returns list of dicts with: {entity1, entity2, relationship_type, confidence}
    """
    if len(entities) < 2:
        return []

    model = await model_manager.get_model("FAST_LLM")
    if not model:
        return []

    entity_names = [e.name for e in entities]

    prompt = f"""Given these entities: {', '.join(entity_names)}

And this text:
{text[:5000]}

Identify relationships between these entities.
For each relationship, provide:
- entity1: First entity name
- entity2: Second entity name
- relationship_type: Type of relationship (works_at, located_in, founded_by, collaborated_with, etc.)
- confidence: Your confidence score (0.0-1.0)

Return as JSON array: [{{"entity1": "...", "entity2": "...", "relationship_type": "...", "confidence": 0.8}}]
Only return the JSON, no other text.
"""

    try:
        response = await model.ainvoke(prompt)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.startswith("```"):
            response = response[3:]
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        relationships = json.loads(response)
        return relationships
    except Exception as e:
        logger.error(f"Failed to extract relationships: {e}")
        return []


async def process_source_entities(source_id: str) -> Tuple[int, int]:
    """
    Extract entities from a source and create knowledge graph.
    Returns (entities_created, relationships_created).
    """
    from open_notebook.domain.base import ObjectModel

    source = await ObjectModel.get(source_id)

    if not source.full_text:
        logger.warning(f"No text to extract entities from for source {source_id}")
        return 0, 0

    # Extract entities
    entity_dicts = await extract_entities_from_text(source.full_text)
    logger.info(f"Extracted {len(entity_dicts)} entities from source {source_id}")

    # Create or update entities
    entities = []
    for entity_dict in entity_dicts:
        entity = await get_or_create_entity(
            name=entity_dict["name"],
            entity_type=entity_dict["type"],
            description=entity_dict.get("description"),
            metadata={"confidence": entity_dict.get("confidence", 1.0)},
        )

        # Add mention relationship
        # Extract context (find where entity is mentioned in text)
        # Simple approach: find first occurrence
        import re
        pattern = re.escape(entity.name)
        match = re.search(pattern, source.full_text, re.IGNORECASE)
        if match:
            start = max(0, match.start() - 100)
            end = min(len(source.full_text), match.end() + 100)
            context = source.full_text[start:end]
        else:
            context = entity.name

        await entity.add_mention(source_id, context)
        entities.append(entity)

    # Extract relationships between entities
    relationships = await extract_entity_relationships(entities, source.full_text)
    logger.info(f"Extracted {len(relationships)} relationships")

    # Create relationship edges
    entity_map = {e.name: e for e in entities}
    relationships_created = 0

    for rel in relationships:
        entity1 = entity_map.get(rel["entity1"])
        entity2 = entity_map.get(rel["entity2"])

        if entity1 and entity2:
            await entity1.relate_to_entity(
                target_entity_id=entity2.id,
                relationship_type=rel["relationship_type"],
                confidence=rel.get("confidence", 0.8),
            )
            relationships_created += 1

    return len(entities), relationships_created
```

### 2.4 API Endpoints

Add to: `api/routers/entities.py` (new file)

```python
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from loguru import logger

from open_notebook.domain.entity import (
    Entity,
    EntityType,
    find_entities_by_source,
    search_entities,
)
from open_notebook.utils.entity_extractor import process_source_entities


router = APIRouter(prefix="/entities", tags=["entities"])


@router.post("/sources/{source_id}/extract")
async def extract_source_entities(source_id: str):
    """Extract entities from a source and build knowledge graph."""
    try:
        entities_created, relationships_created = await process_source_entities(source_id)
        return {
            "status": "success",
            "entities_created": entities_created,
            "relationships_created": relationships_created,
        }
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/{source_id}")
async def get_source_entities(source_id: str):
    """Get all entities mentioned in a source."""
    try:
        entities = await find_entities_by_source(source_id)
        return {
            "entities": [e.model_dump() for e in entities]
        }
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{entity_id}")
async def get_entity(entity_id: str):
    """Get entity details."""
    try:
        from open_notebook.domain.base import ObjectModel
        entity = await ObjectModel.get(entity_id)
        return entity.model_dump()
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=404, detail="Entity not found")


@router.get("/{entity_id}/sources")
async def get_entity_sources(entity_id: str):
    """Get all sources that mention this entity."""
    try:
        from open_notebook.domain.base import ObjectModel
        entity = await ObjectModel.get(entity_id)
        sources = await entity.get_sources()
        return {"sources": sources}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{entity_id}/related")
async def get_related_entities(
    entity_id: str,
    relationship_type: Optional[str] = None
):
    """Get entities related to this entity."""
    try:
        from open_notebook.domain.base import ObjectModel
        entity = await ObjectModel.get(entity_id)
        related = await entity.get_related_entities(relationship_type)
        return {"related_entities": related}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_entities_endpoint(
    query: str,
    entity_type: Optional[EntityType] = None,
    limit: int = 10,
):
    """Search for entities by name or semantic similarity."""
    try:
        entities = await search_entities(query, entity_type, limit)
        return {
            "entities": [e.model_dump() for e in entities]
        }
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
```

Register router in `api/main.py`:
```python
from api.routers import entities
app.include_router(entities.router)
```

---

## Phase 3: Knowledge Graph Recommendations (Week 5-6)

### 3.1 Recommendation Engine

Create: `open_notebook/utils/recommender.py`

```python
from typing import List, Dict, Optional
from loguru import logger

from open_notebook.database.repository import repo_query, ensure_record_id
from open_notebook.domain.notebook import Source
from open_notebook.domain.entity import Entity


async def recommend_sources_by_citation(
    source_id: str,
    limit: int = 5
) -> List[Dict]:
    """
    Recommend sources based on citation network.
    Suggests: Papers cited by papers that cite this paper.
    """
    result = await repo_query(
        """
        -- Find sources that cite this source
        LET $citing_sources = (
            SELECT in as source FROM cites WHERE out = $source_id
        );

        -- Find what those sources cite
        SELECT out as recommended_source, COUNT() as score
        FROM cites
        WHERE in IN $citing_sources
        AND out != $source_id  -- Exclude the original source
        GROUP BY out
        ORDER BY score DESC
        LIMIT $limit
        FETCH recommended_source
        """,
        {
            "source_id": ensure_record_id(source_id),
            "limit": limit,
        },
    )
    return result


async def recommend_sources_by_entities(
    source_id: str,
    limit: int = 5
) -> List[Dict]:
    """
    Recommend sources based on shared entities.
    Suggests: Sources that mention the same entities.
    """
    result = await repo_query(
        """
        -- Get entities mentioned in this source
        LET $source_entities = (
            SELECT out as entity FROM mentions WHERE in = $source_id
        );

        -- Find other sources mentioning these entities
        SELECT in as recommended_source, COUNT() as common_entities
        FROM mentions
        WHERE out IN $source_entities
        AND in != $source_id
        GROUP BY in
        ORDER BY common_entities DESC
        LIMIT $limit
        FETCH recommended_source
        """,
        {
            "source_id": ensure_record_id(source_id),
            "limit": limit,
        },
    )
    return result


async def recommend_sources_by_entity_relationships(
    entity_id: str,
    limit: int = 5
) -> List[Dict]:
    """
    Recommend sources based on related entities.
    Example: User interested in "John Doe" → recommend sources about people he works with.
    """
    result = await repo_query(
        """
        -- Get related entities
        LET $related_entities = (
            SELECT out as entity FROM related_to WHERE in = $entity_id
        );

        -- Find sources mentioning related entities
        SELECT in as recommended_source, out as entity, mention_count
        FROM mentions
        WHERE out IN $related_entities
        ORDER BY mention_count DESC
        LIMIT $limit
        FETCH recommended_source, entity
        """,
        {
            "entity_id": ensure_record_id(entity_id),
            "limit": limit,
        },
    )
    return result


async def get_user_interests(user_id: str) -> List[Entity]:
    """Get entities a user is interested in based on their activity."""
    # This requires user activity tracking
    # For now, return entities from sources they've read
    result = await repo_query(
        """
        -- Get sources user has interacted with (e.g., via chat sessions)
        LET $user_sources = (
            SELECT out as source
            FROM reads  -- Assuming you add user read tracking
            WHERE in = $user_id
        );

        -- Get entities from those sources
        SELECT out as entity, SUM(mention_count) as total_mentions
        FROM mentions
        WHERE in IN $user_sources
        GROUP BY out
        ORDER BY total_mentions DESC
        LIMIT 10
        FETCH entity
        """,
        {"user_id": ensure_record_id(user_id)},
    )
    return [Entity(**r["entity"]) for r in result] if result else []


async def recommend_for_user(
    user_id: str,
    limit: int = 10
) -> List[Dict]:
    """
    Personalized recommendations based on user's interests.
    Combines multiple recommendation strategies.
    """
    # Get user interests
    interests = await get_user_interests(user_id)

    if not interests:
        logger.warning(f"No interests found for user {user_id}")
        return []

    # Get recommendations for each interest
    all_recommendations = []
    for entity in interests[:3]:  # Top 3 interests
        recs = await recommend_sources_by_entity_relationships(entity.id, limit=limit)
        all_recommendations.extend(recs)

    # Deduplicate and sort by relevance
    seen = set()
    unique_recs = []
    for rec in all_recommendations:
        source_id = rec["recommended_source"]["id"]
        if source_id not in seen:
            seen.add(source_id)
            unique_recs.append(rec)

    return unique_recs[:limit]
```

### 3.2 User Activity Tracking

Add to: `open_notebook/domain/user.py` (new file)

```python
from typing import ClassVar
from open_notebook.domain.base import ObjectModel
from open_notebook.database.repository import repo_relate, repo_query, ensure_record_id


class User(ObjectModel):
    """User model for activity tracking."""
    table_name: ClassVar[str] = "user"
    email: str
    name: str

    async def track_read(self, source_id: str):
        """Track that user read a source."""
        await repo_relate(
            source=self.id,
            relationship="reads",
            target=source_id,
            data={"timestamp": "time::now()"},
        )

    async def track_interest(self, entity_id: str, score: float = 1.0):
        """Track user interest in an entity."""
        # Check if interest already exists
        existing = await repo_query(
            """
            SELECT * FROM interested_in
            WHERE in = $user_id AND out = $entity_id
            """,
            {
                "user_id": ensure_record_id(self.id),
                "entity_id": ensure_record_id(entity_id),
            },
        )

        if existing:
            # Update interest score
            await repo_query(
                """
                UPDATE interested_in
                SET
                    interest_score = interest_score + $score,
                    last_interaction = time::now()
                WHERE in = $user_id AND out = $entity_id
                """,
                {
                    "user_id": ensure_record_id(self.id),
                    "entity_id": ensure_record_id(entity_id),
                    "score": score,
                },
            )
        else:
            # Create new interest
            await repo_relate(
                source=self.id,
                relationship="interested_in",
                target=entity_id,
                data={
                    "interest_score": score,
                    "last_interaction": "time::now()",
                },
            )
```

### 3.3 API Endpoints

Add to: `api/routers/recommendations.py` (new file)

```python
from fastapi import APIRouter, HTTPException
from typing import Optional
from loguru import logger

from open_notebook.utils.recommender import (
    recommend_sources_by_citation,
    recommend_sources_by_entities,
    recommend_sources_by_entity_relationships,
    recommend_for_user,
)


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/sources/{source_id}/by-citation")
async def get_citation_recommendations(source_id: str, limit: int = 5):
    """Get source recommendations based on citation network."""
    try:
        recommendations = await recommend_sources_by_citation(source_id, limit)
        return {"recommendations": recommendations}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sources/{source_id}/by-entities")
async def get_entity_recommendations(source_id: str, limit: int = 5):
    """Get source recommendations based on shared entities."""
    try:
        recommendations = await recommend_sources_by_entities(source_id, limit)
        return {"recommendations": recommendations}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/entities/{entity_id}/sources")
async def get_entity_source_recommendations(entity_id: str, limit: int = 5):
    """Get source recommendations based on related entities."""
    try:
        recommendations = await recommend_sources_by_entity_relationships(entity_id, limit)
        return {"recommendations": recommendations}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/users/{user_id}")
async def get_user_recommendations(user_id: str, limit: int = 10):
    """Get personalized recommendations for a user."""
    try:
        recommendations = await recommend_for_user(user_id, limit)
        return {"recommendations": recommendations}
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))
```

---

## Phase 4: Graph Visualization (Week 7-8)

### 4.1 Frontend Graph Visualization

Create: `frontend/src/components/graph/KnowledgeGraphViewer.tsx`

```typescript
import React, { useEffect, useState, useCallback } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Network, ZoomIn, ZoomOut, Maximize2 } from 'lucide-react';

interface GraphNode {
  id: string;
  name: string;
  type: 'source' | 'entity' | 'citation';
  val?: number;  // Node size
  color?: string;
}

interface GraphLink {
  source: string;
  target: string;
  type: string;
  label?: string;
}

interface GraphData {
  nodes: GraphNode[];
  links: GraphLink[];
}

export function KnowledgeGraphViewer({
  sourceId,
  entityId
}: {
  sourceId?: string;
  entityId?: string;
}) {
  const [graphData, setGraphData] = useState<GraphData>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const fgRef = React.useRef<any>();

  useEffect(() => {
    const fetchGraphData = async () => {
      setLoading(true);
      try {
        let data: GraphData = { nodes: [], links: [] };

        if (sourceId) {
          // Fetch citation network + entities for a source
          const [citations, citedBy, entities] = await Promise.all([
            fetch(`/api/sources/${sourceId}/citations`).then(r => r.json()),
            fetch(`/api/sources/${sourceId}/cited-by`).then(r => r.json()),
            fetch(`/api/entities/sources/${sourceId}`).then(r => r.json()),
          ]);

          // Add source node
          data.nodes.push({
            id: sourceId,
            name: 'This Source',
            type: 'source',
            val: 20,
            color: '#3b82f6',
          });

          // Add citation nodes and links
          citations.citations?.forEach((citation: any) => {
            const citedSource = citation.cited_source;
            data.nodes.push({
              id: citedSource.id,
              name: citedSource.title,
              type: 'citation',
              val: 10,
              color: '#10b981',
            });
            data.links.push({
              source: sourceId,
              target: citedSource.id,
              type: 'cites',
              label: 'cites',
            });
          });

          // Add cited-by nodes and links
          citedBy.cited_by?.forEach((citation: any) => {
            const citingSource = citation.citing_source;
            data.nodes.push({
              id: citingSource.id,
              name: citingSource.title,
              type: 'citation',
              val: 10,
              color: '#f59e0b',
            });
            data.links.push({
              source: citingSource.id,
              target: sourceId,
              type: 'cites',
              label: 'cites',
            });
          });

          // Add entity nodes and links
          entities.entities?.forEach((entity: any) => {
            data.nodes.push({
              id: entity.id,
              name: entity.name,
              type: 'entity',
              val: 8,
              color: '#8b5cf6',
            });
            data.links.push({
              source: sourceId,
              target: entity.id,
              type: 'mentions',
              label: 'mentions',
            });
          });
        } else if (entityId) {
          // Fetch entity graph (sources + related entities)
          const [sources, related] = await Promise.all([
            fetch(`/api/entities/${entityId}/sources`).then(r => r.json()),
            fetch(`/api/entities/${entityId}/related`).then(r => r.json()),
          ]);

          // Add entity node
          data.nodes.push({
            id: entityId,
            name: 'This Entity',
            type: 'entity',
            val: 20,
            color: '#8b5cf6',
          });

          // Add source nodes
          sources.sources?.forEach((item: any) => {
            const source = item.source;
            data.nodes.push({
              id: source.id,
              name: source.title,
              type: 'source',
              val: 10,
              color: '#3b82f6',
            });
            data.links.push({
              source: source.id,
              target: entityId,
              type: 'mentions',
              label: `mentions (${item.mention_count}x)`,
            });
          });

          // Add related entity nodes
          related.related_entities?.forEach((item: any) => {
            const relatedEntity = item.related_entity;
            data.nodes.push({
              id: relatedEntity.id,
              name: relatedEntity.name,
              type: 'entity',
              val: 10,
              color: '#8b5cf6',
            });
            data.links.push({
              source: entityId,
              target: relatedEntity.id,
              type: item.relationship_type,
              label: item.relationship_type,
            });
          });
        }

        setGraphData(data);
      } catch (error) {
        console.error('Failed to fetch graph data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchGraphData();
  }, [sourceId, entityId]);

  const handleNodeClick = useCallback((node: GraphNode) => {
    if (node.type === 'source') {
      window.location.href = `/sources/${node.id}`;
    } else if (node.type === 'entity') {
      window.location.href = `/entities/${node.id}`;
    }
  }, []);

  const handleZoomIn = () => {
    if (fgRef.current) {
      fgRef.current.zoom(fgRef.current.zoom() * 1.2);
    }
  };

  const handleZoomOut = () => {
    if (fgRef.current) {
      fgRef.current.zoom(fgRef.current.zoom() / 1.2);
    }
  };

  const handleFitView = () => {
    if (fgRef.current) {
      fgRef.current.zoomToFit(400);
    }
  };

  if (loading) {
    return <div>Loading knowledge graph...</div>;
  }

  if (graphData.nodes.length === 0) {
    return (
      <Card>
        <CardContent className="p-6">
          <p className="text-muted-foreground">
            No graph data available. Extract citations and entities first.
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Network className="h-5 w-5" />
            Knowledge Graph
          </CardTitle>
          <div className="flex gap-2">
            <Button onClick={handleZoomIn} variant="outline" size="sm">
              <ZoomIn className="h-4 w-4" />
            </Button>
            <Button onClick={handleZoomOut} variant="outline" size="sm">
              <ZoomOut className="h-4 w-4" />
            </Button>
            <Button onClick={handleFitView} variant="outline" size="sm">
              <Maximize2 className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <div style={{ height: '600px', border: '1px solid #e5e7eb', borderRadius: '8px' }}>
          <ForceGraph2D
            ref={fgRef}
            graphData={graphData}
            nodeLabel="name"
            nodeColor="color"
            nodeVal="val"
            linkLabel="label"
            linkDirectionalArrowLength={6}
            linkDirectionalArrowRelPos={1}
            linkCurvature={0.2}
            onNodeClick={handleNodeClick}
            nodeCanvasObject={(node: any, ctx, globalScale) => {
              const label = node.name;
              const fontSize = 12 / globalScale;
              ctx.font = `${fontSize}px Sans-Serif`;
              ctx.textAlign = 'center';
              ctx.textBaseline = 'middle';
              ctx.fillStyle = node.color;

              // Draw node circle
              ctx.beginPath();
              ctx.arc(node.x, node.y, node.val, 0, 2 * Math.PI, false);
              ctx.fill();

              // Draw label
              ctx.fillStyle = '#000';
              ctx.fillText(label, node.x, node.y + node.val + fontSize);
            }}
          />
        </div>
        <div className="mt-4 flex gap-4 text-sm">
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-blue-500"></div>
            <span>Sources</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-purple-500"></div>
            <span>Entities</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="h-3 w-3 rounded-full bg-green-500"></div>
            <span>Citations</span>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
```

Install dependency:
```bash
cd frontend
npm install react-force-graph-2d
```

---

## Database Schema Summary

### Tables

```
source (existing)          - Documents
chunk (existing)           - Document segments
entity (new)               - Named entities
user (existing/new)        - Users

### Relationships (Edges)

cites (new)                - source → source (citations)
mentions (new)             - source → entity (entity occurrences)
contains_entity (new)      - chunk → entity (chunk-level mentions)
related_to (new)           - entity → entity (relationships)
works_at (new)             - entity(person) → entity(org)
located_in (new)           - entity → entity(location)
interested_in (new)        - user → entity (user interests)
reads (new)                - user → source (read tracking)
```

---

## Testing Strategy

### Unit Tests

Create: `tests/test_graph_features.py`

```python
import pytest
from open_notebook.domain.citation import (
    create_citation,
    get_citations_for_source,
    get_citation_chain,
)
from open_notebook.domain.entity import (
    Entity,
    get_or_create_entity,
)
from open_notebook.domain.notebook import Source


@pytest.mark.asyncio
async def test_create_citation():
    # Create test sources
    source1 = Source(title="Paper 1", full_text="Content 1")
    await source1.save()

    source2 = Source(title="Paper 2", full_text="Content 2")
    await source2.save()

    # Create citation
    citation = await create_citation(
        citing_source_id=source1.id,
        cited_source_id=source2.id,
        citation_type="direct",
        confidence=1.0,
    )

    assert citation is not None
    assert citation.citation_type == "direct"

    # Verify citation exists
    citations = await get_citations_for_source(source1.id)
    assert len(citations) == 1
    assert citations[0]["cited_source"]["id"] == source2.id


@pytest.mark.asyncio
async def test_entity_creation():
    # Create entity
    entity = await get_or_create_entity(
        name="John Doe",
        entity_type="person",
        description="Test person",
    )

    assert entity is not None
    assert entity.name == "John Doe"
    assert entity.entity_type == "person"

    # Test idempotency
    entity2 = await get_or_create_entity(
        name="John Doe",
        entity_type="person",
    )

    assert entity.id == entity2.id


@pytest.mark.asyncio
async def test_entity_mentions():
    # Create source and entity
    source = Source(title="Test Source", full_text="About John Doe")
    await source.save()

    entity = await get_or_create_entity(
        name="John Doe",
        entity_type="person",
    )

    # Add mention
    await entity.add_mention(source.id, "About John Doe")

    # Verify mention
    sources = await entity.get_sources()
    assert len(sources) == 1
    assert sources[0]["source"]["id"] == source.id
    assert sources[0]["mention_count"] == 1


@pytest.mark.asyncio
async def test_entity_relationships():
    # Create entities
    person = await get_or_create_entity("John Doe", "person")
    company = await get_or_create_entity("Acme Corp", "organization")

    # Create relationship
    await person.relate_to_entity(
        target_entity_id=company.id,
        relationship_type="works_at",
        confidence=1.0,
    )

    # Verify relationship
    related = await person.get_related_entities("works_at")
    assert len(related) == 1
    assert related[0]["related_entity"]["id"] == company.id
```

### Integration Tests

```python
@pytest.mark.asyncio
async def test_full_citation_network():
    """Test complete citation network flow."""
    # Create 3 sources: A cites B, B cites C
    source_a = Source(title="Paper A", full_text="...")
    await source_a.save()

    source_b = Source(title="Paper B", full_text="...")
    await source_b.save()

    source_c = Source(title="Paper C", full_text="...")
    await source_c.save()

    # Create citations
    await create_citation(source_a.id, source_b.id)
    await create_citation(source_b.id, source_c.id)

    # Test 2-hop citation chain
    chain = await get_citation_chain(source_a.id, depth=2)
    assert source_c.id in [s["id"] for s in chain[0]["sources"]]


@pytest.mark.asyncio
async def test_full_entity_extraction():
    """Test complete entity extraction flow."""
    from open_notebook.utils.entity_extractor import process_source_entities

    source = Source(
        title="Test Paper",
        full_text="John Doe works at Acme Corp in San Francisco. He collaborated with Jane Smith."
    )
    await source.save()

    # Extract entities
    entities_count, relationships_count = await process_source_entities(source.id)

    assert entities_count > 0
    assert relationships_count > 0
```

---

## Deployment Checklist

### Pre-Deployment

- [ ] Run all database migrations
- [ ] Update environment variables (if any new ones)
- [ ] Run test suite: `pytest tests/test_graph_features.py`
- [ ] Build frontend: `cd frontend && npm run build`
- [ ] Update Docker images

### Migration Steps

1. **Backup Database**
   ```bash
   # Backup SurrealDB data
   docker cp open-notebook-surreal:/data ./surreal_backup_$(date +%Y%m%d)
   ```

2. **Run Migrations**
   ```bash
   python scripts/run_migrations.py
   ```

3. **Verify Schema**
   ```bash
   # Connect to SurrealDB and verify tables
   surreal sql --endpoint ws://localhost:8000 --username root --password root
   INFO FOR TABLE cites;
   INFO FOR TABLE entity;
   INFO FOR TABLE mentions;
   ```

### Post-Deployment

- [ ] Test citation extraction on sample source
- [ ] Test entity extraction on sample source
- [ ] Verify graph visualization renders correctly
- [ ] Monitor performance (query execution times)
- [ ] Check logs for errors

---

## Performance Optimization

### Indexing Strategy

```sql
-- Citation lookups
DEFINE INDEX cites_in ON TABLE cites COLUMNS in;
DEFINE INDEX cites_out ON TABLE cites COLUMNS out;

-- Entity lookups
DEFINE INDEX entity_name_type ON TABLE entity COLUMNS name, entity_type;

-- Mention lookups
DEFINE INDEX mentions_in ON TABLE mentions COLUMNS in;
DEFINE INDEX mentions_out ON TABLE mentions COLUMNS out;
```

### Query Optimization Tips

1. **Use FETCH sparingly**: Only fetch related records when needed
2. **Limit result sets**: Always use LIMIT in graph traversals
3. **Cache frequently accessed data**: Store entity embeddings for faster similarity search
4. **Batch operations**: Use parallel processing for entity extraction
5. **Monitor query performance**: Log slow queries (>1s) for optimization

### Scaling Considerations

- **Document Processing**: Entity extraction is CPU-intensive; consider background queue (Celery)
- **Graph Traversals**: Deep traversals (3+ hops) can be slow; cache results
- **Embeddings**: Entity embeddings add storage; consider dimensionality reduction
- **UI Performance**: Limit graph visualization to <100 nodes for smooth rendering

---

## Future Enhancements

### Phase 5: Advanced Features (Future)

1. **Temporal Knowledge Graph**: Track entity relationships over time
2. **Multi-language Support**: Entity extraction for non-English documents
3. **Graph Algorithms**: PageRank, centrality, community detection
4. **Graph RAG**: Use graph structure to improve retrieval context
5. **Entity Disambiguation**: Resolve ambiguous entity references
6. **Automatic Relationship Inference**: ML-based relationship prediction
7. **Graph Export**: Export to Neo4j/GraphML for advanced analytics

---

## Support & Troubleshooting

### Common Issues

**Issue**: Entity extraction fails with JSON parse error
**Solution**: LLM response may not be valid JSON. Add response cleaning logic (strip markdown, validate before parsing).

**Issue**: Graph traversal queries timeout
**Solution**: Add query timeouts, limit traversal depth, add indexes on relationship tables.

**Issue**: Frontend graph too cluttered
**Solution**: Implement graph filtering (show only top-N connections), add zoom/pan controls.

**Issue**: Duplicate entities created
**Solution**: Improve entity matching logic, add fuzzy string matching (levenshtein distance).

### Debug Queries

```sql
-- Count citations
SELECT COUNT() FROM cites;

-- Count entities by type
SELECT entity_type, COUNT() as count FROM entity GROUP BY entity_type;

-- Find most mentioned entities
SELECT out.name as entity, SUM(mention_count) as total_mentions
FROM mentions
GROUP BY out
ORDER BY total_mentions DESC
LIMIT 10
FETCH out;

-- Find most cited sources
SELECT out.title as source, COUNT() as citation_count
FROM cites
GROUP BY out
ORDER BY citation_count DESC
LIMIT 10
FETCH out;
```

---

## Conclusion

This implementation guide provides a complete roadmap for adding knowledge graph capabilities to Open Notebook using SurrealDB's native graph features. The phased approach ensures:

1. **Non-breaking changes**: Existing functionality remains intact
2. **Incremental value**: Each phase delivers tangible features
3. **Production-ready**: Includes testing, performance optimization, deployment
4. **Extensible**: Foundation for future advanced features

**Total Estimated Effort**: 6-8 weeks
**Expected Outcome**: Production-grade knowledge graph with citation networks, entity extraction, and recommendations—all without the complexity of a separate graph database.
