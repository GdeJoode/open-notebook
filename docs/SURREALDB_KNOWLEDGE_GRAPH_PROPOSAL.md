# SurrealDB Knowledge Graph Proposal

## Executive Summary

Dit document beschrijft een voorstel voor het implementeren van een Knowledge Graph (KG) bovenop SurrealDB om diverse brontypen te integreren: academische papers, beleidsstukken, beleidsadviezen, social media (LinkedIn), en andere content. Het benut SurrealDB's native graph-capabilities, vector search, en multi-model architectuur.

---

## Deel 1: Wat moet er in de Knowledge Graph?

### 1.1 Core Entity Types

```
┌─────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE GRAPH ENTITIES                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  📄 CONTENT ENTITIES           👤 ACTOR ENTITIES                │
│  ├─ AcademicPaper              ├─ Person                        │
│  ├─ PolicyDocument             ├─ Organization                  │
│  ├─ PolicyAdvice               ├─ Institution                   │
│  ├─ SocialMediaPost            └─ GovernmentBody                │
│  ├─ NewsArticle                                                 │
│  ├─ Report                     📍 CONTEXTUAL ENTITIES           │
│  ├─ LegalDocument              ├─ Topic                         │
│  └─ Presentation               ├─ Theme                         │
│                                ├─ Concept                       │
│  🔗 META ENTITIES              ├─ Keyword                       │
│  ├─ Citation                   ├─ GeographicRegion              │
│  ├─ Claim                      ├─ TimePeriod                    │
│  ├─ Evidence                   └─ PolicyDomain                  │
│  └─ Argument                                                    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Content Entity Details

#### AcademicPaper
```yaml
fields:
  - id: record_id
  - title: string
  - abstract: text
  - full_text: text (optional)
  - doi: string (unique)
  - arxiv_id: string
  - publication_date: datetime
  - journal: string
  - volume: int
  - issue: int
  - pages: string
  - peer_reviewed: bool
  - open_access: bool
  - citation_count: int
  - methodology: enum[quantitative, qualitative, mixed, meta_analysis, review]
  - study_type: enum[empirical, theoretical, case_study, survey, experiment]
  - embedding: vector<float, 1024>
  - metadata: object

relationships:
  - authored_by -> Person[]
  - affiliated_with -> Institution[]
  - cites -> AcademicPaper[]
  - cited_by <- AcademicPaper[]
  - discusses -> Topic[]
  - supports -> Claim[]
  - contradicts -> Claim[]
  - extends -> AcademicPaper
  - replicates -> AcademicPaper
```

#### PolicyDocument
```yaml
fields:
  - id: record_id
  - title: string
  - document_type: enum[wet, amvb, beleidsregel, circulaire, richtlijn, verdrag, verordening]
  - jurisdiction: enum[gemeente, provincie, nationaal, eu, internationaal]
  - issuing_body: string
  - document_number: string (e.g., "Kamerstuk 35925-XV")
  - publication_date: datetime
  - effective_date: datetime
  - expiry_date: datetime (optional)
  - status: enum[concept, gepubliceerd, geamendeerd, ingetrokken]
  - full_text: text
  - summary: text
  - policy_domain: string[]  # e.g., ["zorg", "onderwijs", "klimaat"]
  - embedding: vector<float, 1024>

relationships:
  - issued_by -> GovernmentBody
  - supersedes -> PolicyDocument
  - superseded_by <- PolicyDocument
  - implements -> PolicyDocument  # EU directive -> NL wet
  - amends -> PolicyDocument
  - relates_to -> PolicyDocument[]
  - addresses -> Topic[]
  - referenced_in -> PolicyAdvice[]
```

#### PolicyAdvice (Beleidsadvies)
```yaml
fields:
  - id: record_id
  - title: string
  - advice_type: enum[advies, rapport, evaluatie, quickscan, verkenning]
  - advisory_body: string  # e.g., "WRR", "Raad van State", "CPB", "SCP"
  - recipient: string  # e.g., "Tweede Kamer", "Minister van..."
  - request_date: datetime
  - publication_date: datetime
  - document_number: string
  - summary: text
  - full_text: text
  - recommendations: text[]
  - key_findings: text[]
  - methodology_description: text
  - embedding: vector<float, 1024>

relationships:
  - authored_by -> Organization
  - requested_by -> GovernmentBody
  - addresses_policy -> PolicyDocument[]
  - supports -> Claim[]
  - references -> Source[]  # academic papers, other documents
  - leads_to -> PolicyDocument[]  # policy outcomes
```

#### SocialMediaPost
```yaml
fields:
  - id: record_id
  - platform: enum[linkedin, twitter, mastodon, bluesky]
  - post_id: string (platform-specific)
  - author_handle: string
  - author_name: string
  - content: text
  - post_date: datetime
  - engagement:
      likes: int
      comments: int
      shares: int
      views: int
  - hashtags: string[]
  - mentioned_handles: string[]
  - media_urls: string[]
  - is_thread: bool
  - thread_position: int
  - language: string
  - sentiment: float  # -1 to 1
  - embedding: vector<float, 1024>

relationships:
  - posted_by -> Person
  - replies_to -> SocialMediaPost
  - quotes -> SocialMediaPost
  - mentions -> Person[]
  - discusses -> Topic[]
  - references -> Source[]  # links to papers, articles
  - part_of_thread -> SocialMediaPost  # thread head
```

### 1.3 Actor Entities

#### Person
```yaml
fields:
  - id: record_id
  - name: string
  - aliases: string[]
  - orcid: string
  - linkedin_url: string
  - twitter_handle: string
  - email: string
  - bio: text
  - expertise_areas: string[]
  - h_index: int
  - current_position: string
  - embedding: vector<float, 1024>  # based on bio + expertise

relationships:
  - affiliated_with -> Organization[]
  - authored -> Source[]
  - coauthored_with -> Person[]
  - cited_by -> Person[]  # who cites their work
  - cites -> Person[]  # who they cite
  - collaborates_with -> Person[]
  - advised_by -> Person
  - advises -> Person[]
  - same_as -> Person  # deduplication link
```

#### Organization
```yaml
fields:
  - id: record_id
  - name: string
  - aliases: string[]
  - type: enum[university, research_institute, think_tank, advisory_body,
               government, ngo, company, international_org]
  - country: string
  - city: string
  - website: string
  - description: text
  - founded_year: int
  - ror_id: string  # Research Organization Registry
  - embedding: vector<float, 1024>

relationships:
  - employs -> Person[]
  - part_of -> Organization  # parent org
  - collaborates_with -> Organization[]
  - funded_by -> Organization[]
  - publishes -> Source[]
  - located_in -> GeographicRegion
```

### 1.4 Contextual Entities

#### Topic
```yaml
fields:
  - id: record_id
  - name: string
  - description: text
  - level: enum[broad, specific, narrow]
  - domain: string  # e.g., "healthcare", "education"
  - wikidata_id: string
  - embedding: vector<float, 1024>

relationships:
  - broader_than -> Topic[]
  - narrower_than -> Topic[]
  - related_to -> Topic[]
  - same_as -> Concept  # link to external ontology
```

#### Claim
```yaml
fields:
  - id: record_id
  - statement: text
  - claim_type: enum[factual, causal, normative, predictive]
  - confidence: float  # 0-1
  - verification_status: enum[unverified, supported, contested, refuted]
  - first_appearance: datetime
  - embedding: vector<float, 1024>

relationships:
  - made_in -> Source[]
  - supported_by -> Evidence[]
  - contradicted_by -> Evidence[]
  - related_to -> Claim[]
  - about -> Topic[]
```

#### Evidence
```yaml
fields:
  - id: record_id
  - description: text
  - evidence_type: enum[statistical, qualitative, experimental, observational,
                        expert_opinion, case_study]
  - strength: enum[weak, moderate, strong]
  - methodology: text
  - sample_size: int
  - confidence_interval: string
  - embedding: vector<float, 1024>

relationships:
  - from_source -> Source
  - supports -> Claim[]
  - contradicts -> Claim[]
  - replicated_by -> Evidence[]
```

---

## Deel 2: SurrealDB Features te Benutten

### 2.1 Native Graph Capabilities

#### Record Links (Direct References)
```surql
-- Direct embedding of related records
CREATE source:paper1 SET
    title = "AI in Healthcare",
    authors = [person:author1, person:author2],
    topics = [topic:ai, topic:healthcare];

-- Access without joins
SELECT title, authors.name, topics.name FROM source:paper1;
```

#### Graph Relations (RELATE)
```surql
-- Create typed relationships with properties
RELATE person:author1->authored->source:paper1
    SET role = "lead_author", contribution_pct = 60;

RELATE source:paper1->cites->source:paper2
    SET citation_context = "methodology comparison",
        section = "literature_review",
        sentiment = "positive";

-- Policy chain tracking
RELATE policy:directive_eu->implements->policy:wet_nl
    SET implementation_date = d"2024-01-01",
        compliance_status = "partial";
```

#### Graph Traversal
```surql
-- Find all papers that cite papers written by a specific author
SELECT
    <-cites<-source AS citing_papers
FROM source
WHERE ->authored->person.name = "Dr. Smith";

-- Citation network depth
SELECT
    id,
    title,
    ->cites->source->cites->source.title AS second_order_citations
FROM source:paper1;

-- Policy impact chain
SELECT
    id,
    title,
    ->leads_to->policy.title AS resulting_policies,
    ->leads_to->policy->implements->policy.title AS eu_directives
FROM policy_advice;
```

### 2.2 Vector Search & Embeddings

#### Vector Index Definition
```surql
-- Define vector indexes for semantic search
DEFINE INDEX idx_source_embedding ON source
    FIELDS embedding
    MTREE DIMENSION 1024
    DIST COSINE;

DEFINE INDEX idx_claim_embedding ON claim
    FIELDS embedding
    MTREE DIMENSION 1024
    DIST COSINE;

DEFINE INDEX idx_topic_embedding ON topic
    FIELDS embedding
    MTREE DIMENSION 1024
    DIST COSINE;
```

#### Semantic Search Queries
```surql
-- Find semantically similar sources
LET $query_embedding = <embedding from user query>;

SELECT id, title, abstract,
    vector::similarity::cosine(embedding, $query_embedding) AS similarity
FROM source
WHERE vector::similarity::cosine(embedding, $query_embedding) > 0.7
ORDER BY similarity DESC
LIMIT 20;

-- Hybrid search: semantic + metadata filters
SELECT id, title,
    vector::similarity::cosine(embedding, $query_embedding) AS similarity
FROM source
WHERE
    vector::similarity::cosine(embedding, $query_embedding) > 0.6
    AND publication_date > d"2020-01-01"
    AND source_type IN ["academic_paper", "policy_advice"]
    AND ->discusses->topic.name CONTAINS "climate"
ORDER BY similarity DESC;
```

### 2.3 Multi-Model Queries (Document + Graph + Vector)

```surql
-- Complex analytical query combining all models
SELECT
    s.id,
    s.title,
    s.abstract,
    vector::similarity::cosine(s.embedding, $query_embedding) AS relevance,

    -- Graph traversals
    count(->cites) AS outgoing_citations,
    count(<-cites) AS incoming_citations,
    ->authored_by->person.name AS authors,
    ->discusses->topic.name AS topics,

    -- Related claims and evidence
    ->supports->claim AS supported_claims,
    ->supports->claim<-contradicts<-source.title AS contradicting_sources,

    -- Policy connections
    <-references<-policy_advice.title AS policy_relevance

FROM source AS s
WHERE
    vector::similarity::cosine(s.embedding, $query_embedding) > 0.5
ORDER BY
    relevance * 0.5 +
    (count(<-cites) / 100) * 0.3 +
    (count(<-references<-policy_advice)) * 0.2 DESC
LIMIT 50;
```

### 2.4 Computed Fields & Analytics

```surql
-- Define computed fields for analytics
DEFINE FIELD citation_score ON source VALUE
    count(<-cites) * 1.0 +
    count(<-cites<-cites) * 0.5;

DEFINE FIELD influence_score ON person VALUE
    count(->authored->source<-cites) +
    count(->advised->policy_advice->leads_to->policy) * 5;

DEFINE FIELD policy_impact ON source VALUE
    count(<-references<-policy_advice) +
    count(<-references<-policy_advice->leads_to->policy) * 2;
```

### 2.5 Live Queries (Real-time Updates)

```surql
-- Subscribe to new sources about specific topics
LIVE SELECT * FROM source
WHERE ->discusses->topic.name CONTAINS "AI regulation"
    AND publication_date > time::now() - 7d;

-- Monitor claim status changes
LIVE SELECT * FROM claim
WHERE verification_status CHANGED;
```

### 2.6 Transactions & Batch Operations

```surql
-- Atomic graph updates
BEGIN TRANSACTION;

LET $paper = CREATE source SET
    title = "New Research",
    source_type = "academic_paper";

RELATE person:author1->authored->$paper SET role = "lead";
RELATE $paper->discusses->topic:ai;
RELATE $paper->cites->source:paper1;
RELATE $paper->supports->claim:claim1 SET strength = "strong";

COMMIT TRANSACTION;
```

### 2.7 Full-Text Search

```surql
-- Define full-text search analyzer
DEFINE ANALYZER dutch_analyzer
    TOKENIZERS blank, class
    FILTERS lowercase, snowball(nld);

DEFINE INDEX idx_source_fulltext ON source
    FIELDS title, abstract, full_text
    SEARCH ANALYZER dutch_analyzer;

-- Combined full-text and semantic search
SELECT id, title,
    search::score(1) AS text_score,
    vector::similarity::cosine(embedding, $query_embedding) AS semantic_score
FROM source
WHERE
    (title @1@ "klimaatbeleid" OR abstract @1@ "klimaatbeleid")
    OR vector::similarity::cosine(embedding, $query_embedding) > 0.6
ORDER BY (text_score + semantic_score) DESC;
```

---

## Deel 3: Implementation Options

### Option A: Minimale Uitbreiding (MVP)

**Scope**: Extend existing Source/Chunk model with basic relationships

```
┌────────────────────────────────────────────────────────┐
│                    OPTION A: MVP                        │
├────────────────────────────────────────────────────────┤
│                                                         │
│   Source (extended)                                     │
│   ├─ source_type: enum                                 │
│   ├─ metadata: object (type-specific)                  │
│   └─ embedding: vector                                 │
│                                                         │
│   New Relations:                                        │
│   ├─ source->cites->source                             │
│   ├─ source->mentions->entity                          │
│   └─ entity->appears_in->source                        │
│                                                         │
│   Entity (new, simple)                                  │
│   ├─ name, type, aliases                               │
│   └─ embedding                                         │
│                                                         │
│   Effort: 2-3 weken                                     │
│   Complexity: Low                                       │
│   Value: Basic entity linking + citation network       │
│                                                         │
└────────────────────────────────────────────────────────┘
```

**Implementation**:
1. Add `source_type` field to Source
2. Create Entity table with NER extraction
3. Add `cites` and `mentions` relationships
4. Implement basic graph queries

### Option B: Typed Source Hierarchy

**Scope**: Separate tables per source type with inheritance

```
┌────────────────────────────────────────────────────────┐
│              OPTION B: TYPED HIERARCHY                  │
├────────────────────────────────────────────────────────┤
│                                                         │
│   Source (base)                                         │
│   ├─ AcademicPaper                                     │
│   │   └─ doi, journal, peer_reviewed, methodology     │
│   ├─ PolicyDocument                                    │
│   │   └─ jurisdiction, status, document_number        │
│   ├─ PolicyAdvice                                      │
│   │   └─ advisory_body, recommendations[]             │
│   ├─ SocialMediaPost                                   │
│   │   └─ platform, engagement, hashtags               │
│   └─ NewsArticle                                       │
│       └─ outlet, byline, section                       │
│                                                         │
│   Full Entity Model:                                    │
│   ├─ Person, Organization, Topic                       │
│   └─ Claim, Evidence                                   │
│                                                         │
│   Rich Relationships:                                   │
│   ├─ authored_by, affiliated_with                      │
│   ├─ cites, supports, contradicts                      │
│   └─ implements, supersedes, leads_to                  │
│                                                         │
│   Effort: 6-8 weken                                     │
│   Complexity: Medium                                    │
│   Value: Full provenance + policy tracking             │
│                                                         │
└────────────────────────────────────────────────────────┘
```

**Implementation**:
1. Create separate tables with shared base fields
2. Implement type-specific processors
3. Build entity extraction pipeline (NER + linking)
4. Add Claim/Evidence extraction
5. Implement relationship inference

### Option C: Full Knowledge Graph with Ontology

**Scope**: Complete ontology-driven KG with reasoning

```
┌────────────────────────────────────────────────────────┐
│           OPTION C: FULL ONTOLOGY-DRIVEN KG            │
├────────────────────────────────────────────────────────┤
│                                                         │
│   External Ontology Integration:                        │
│   ├─ Wikidata linking                                  │
│   ├─ SKOS topic hierarchy                              │
│   ├─ Dublin Core metadata                              │
│   └─ Schema.org entities                               │
│                                                         │
│   Advanced Features:                                    │
│   ├─ Claim verification pipeline                       │
│   ├─ Argument mining                                   │
│   ├─ Contradiction detection                           │
│   ├─ Evidence synthesis                                │
│   └─ Policy impact tracing                             │
│                                                         │
│   Inference Engine:                                     │
│   ├─ Transitive closure (A cites B cites C)           │
│   ├─ Author similarity networks                        │
│   ├─ Topic co-occurrence                               │
│   └─ Temporal analysis                                 │
│                                                         │
│   Effort: 3-4 maanden                                   │
│   Complexity: High                                      │
│   Value: Research-grade knowledge management           │
│                                                         │
└────────────────────────────────────────────────────────┘
```

---

## Deel 4: Recommended Approach - Phased Implementation

### Phase 1: Foundation (Week 1-3)

**Goal**: Basic KG infrastructure with source typing

```python
# New models
class SourceType(str, Enum):
    ACADEMIC_PAPER = "academic_paper"
    POLICY_DOCUMENT = "policy_document"
    POLICY_ADVICE = "policy_advice"
    SOCIAL_MEDIA = "social_media"
    NEWS_ARTICLE = "news_article"
    LEGAL_DOCUMENT = "legal_document"
    REPORT = "report"
    OTHER = "other"

# Extended Source model
class Source(ObjectModel):
    source_type: SourceType
    type_metadata: Dict[str, Any]  # Type-specific fields
    external_ids: Dict[str, str]  # doi, arxiv, etc.
```

**Deliverables**:
- [ ] Source type classification
- [ ] Type-specific metadata schema
- [ ] Basic source->cites->source relationship
- [ ] Migration script for existing sources

### Phase 2: Entity Extraction (Week 4-6)

**Goal**: Named Entity Recognition + Entity Linking

```python
class Entity(ObjectModel):
    table_name = "entity"
    name: str
    entity_type: Literal["person", "organization", "topic", "location"]
    aliases: List[str]
    external_ids: Dict[str, str]  # wikidata, orcid, ror
    embedding: List[float]

# Relationship: source->mentions->entity
```

**Deliverables**:
- [ ] NER pipeline integration (spaCy/GLiNER)
- [ ] Entity deduplication logic
- [ ] Entity linking to external KBs
- [ ] mentions relationship with context

### Phase 3: Citation & Reference Network (Week 7-9)

**Goal**: Full citation graph with context

```surql
-- Citation relationship with metadata
RELATE source:paper1->cites->source:paper2 SET
    citation_context = "...",
    section = "methodology",
    sentiment = "supportive",
    extracted_at = time::now();
```

**Deliverables**:
- [ ] Citation extraction from PDFs
- [ ] DOI/arXiv resolution
- [ ] Citation context extraction
- [ ] Network visualization API

### Phase 4: Claims & Evidence (Week 10-12)

**Goal**: Track factual claims and supporting evidence

```python
class Claim(ObjectModel):
    statement: str
    claim_type: Literal["factual", "causal", "normative"]
    verification_status: str
    sources: List[str]  # Record links

class Evidence(ObjectModel):
    description: str
    evidence_type: str
    source_id: str
    supports_claims: List[str]
```

**Deliverables**:
- [ ] Claim extraction pipeline
- [ ] Evidence linking
- [ ] Claim verification workflow
- [ ] Contradiction detection (basic)

### Phase 5: Advanced Queries & Analytics (Week 13-16)

**Goal**: Powerful search and analysis capabilities

**Deliverables**:
- [ ] Multi-hop graph queries
- [ ] Hybrid search (vector + graph + fulltext)
- [ ] Author influence metrics
- [ ] Topic evolution tracking
- [ ] Policy impact analysis

---

## Deel 5: Source-Specific Considerations

### 5.1 Academic Papers

**Acquisition**:
- Semantic Scholar API
- OpenAlex API
- CrossRef for metadata
- arXiv bulk access
- PubMed/PMC

**Key Extractions**:
- Authors + affiliations (ORCID linking)
- References/citations
- Methodology classification
- Key findings/claims
- Dataset references

### 5.2 Beleidsstukken (Dutch Policy Documents)

**Acquisition**:
- officielebekendmakingen.nl API
- Overheid.nl zoekdienst
- EUR-Lex (EU)
- Parlementaire documenten

**Key Extractions**:
- Document type classification
- Jurisdiction/scope
- Effective dates
- Amendment chains
- Referenced legislation
- Implementation requirements

### 5.3 Beleidsadviezen

**Sources**:
- WRR (Wetenschappelijke Raad voor het Regeringsbeleid)
- SCP (Sociaal en Cultureel Planbureau)
- CPB (Centraal Planbureau)
- PBL (Planbureau voor de Leefomgeving)
- Raad van State
- Onderwijsraad
- Etc.

**Key Extractions**:
- Recommendations
- Key findings
- Referenced sources
- Resulting policy actions
- Timeline of advice → policy

### 5.4 Social Media (LinkedIn Focus)

**Acquisition**:
- LinkedIn API (limited)
- Manual/curated collection
- RSS feeds from thought leaders
- Apify/scraping (ToS compliance!)

**Key Extractions**:
- Author expertise matching
- Cited sources/links
- Engagement metrics
- Thread reconstruction
- Sentiment analysis
- Hashtag/topic mapping

**Privacy Considerations**:
- Store only public posts
- Respect rate limits
- Anonymization options
- GDPR compliance

### 5.5 News Articles

**Sources**:
- NOS, RTL Nieuws
- NRC, Volkskrant, Trouw
- Specialized: Binnenlands Bestuur, Zorgvisie
- International: Reuters, BBC

**Key Extractions**:
- Quoted experts
- Referenced studies
- Event detection
- Stance/sentiment
- Source credibility

---

## Deel 6: Technical Architecture

### 6.1 Data Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [Source] ──► [Classifier] ──► [Type-Specific Processor]        │
│                                       │                          │
│                    ┌──────────────────┼──────────────────┐       │
│                    ▼                  ▼                  ▼       │
│              [PDF Parser]      [API Fetcher]      [Scraper]     │
│                    │                  │                  │       │
│                    └──────────────────┼──────────────────┘       │
│                                       ▼                          │
│                              [Content Extractor]                 │
│                                       │                          │
│                    ┌──────────────────┼──────────────────┐       │
│                    ▼                  ▼                  ▼       │
│              [NER Pipeline]   [Citation Parser]  [Claim Extractor]│
│                    │                  │                  │       │
│                    └──────────────────┼──────────────────┘       │
│                                       ▼                          │
│                              [Entity Linker]                     │
│                                       │                          │
│                                       ▼                          │
│                              [Embedding Generator]               │
│                                       │                          │
│                                       ▼                          │
│                              [SurrealDB Writer]                  │
│                                       │                          │
│                    ┌──────────────────┼──────────────────┐       │
│                    ▼                  ▼                  ▼       │
│               [Sources]         [Entities]        [Relations]    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 6.2 Query Interface

```python
class KnowledgeGraphQuery:
    """Unified interface for KG queries"""

    async def semantic_search(
        self,
        query: str,
        source_types: List[SourceType] = None,
        date_range: Tuple[datetime, datetime] = None,
        topics: List[str] = None,
        limit: int = 20
    ) -> List[Source]:
        """Semantic search with filters"""

    async def find_related(
        self,
        source_id: str,
        relationship_types: List[str] = None,
        max_hops: int = 2,
        limit: int = 50
    ) -> GraphResult:
        """Graph traversal from source"""

    async def trace_claim(
        self,
        claim_id: str
    ) -> ClaimTrace:
        """Trace claim through evidence to sources"""

    async def expert_network(
        self,
        topic: str,
        min_publications: int = 3
    ) -> AuthorNetwork:
        """Find experts on topic with collaboration network"""

    async def policy_lineage(
        self,
        policy_id: str
    ) -> PolicyLineage:
        """Trace policy from advice through implementation"""
```

### 6.3 SurrealDB Schema

```surql
-- Core tables
DEFINE TABLE source SCHEMAFULL;
DEFINE FIELD source_type ON source TYPE string;
DEFINE FIELD title ON source TYPE string;
DEFINE FIELD content ON source TYPE string;
DEFINE FIELD type_metadata ON source TYPE object;
DEFINE FIELD external_ids ON source TYPE object;
DEFINE FIELD embedding ON source TYPE array<float>;
DEFINE FIELD created ON source TYPE datetime DEFAULT time::now();
DEFINE FIELD updated ON source TYPE datetime DEFAULT time::now();

DEFINE TABLE entity SCHEMAFULL;
DEFINE FIELD name ON entity TYPE string;
DEFINE FIELD entity_type ON entity TYPE string;
DEFINE FIELD aliases ON entity TYPE array<string>;
DEFINE FIELD external_ids ON entity TYPE object;
DEFINE FIELD embedding ON entity TYPE array<float>;

DEFINE TABLE claim SCHEMAFULL;
DEFINE FIELD statement ON claim TYPE string;
DEFINE FIELD claim_type ON claim TYPE string;
DEFINE FIELD verification_status ON claim TYPE string;
DEFINE FIELD embedding ON claim TYPE array<float>;

-- Relationship tables (for RELATE)
DEFINE TABLE cites SCHEMAFULL;
DEFINE FIELD in ON cites TYPE record<source>;
DEFINE FIELD out ON cites TYPE record<source>;
DEFINE FIELD citation_context ON cites TYPE string;
DEFINE FIELD section ON cites TYPE string;

DEFINE TABLE mentions SCHEMAFULL;
DEFINE FIELD in ON mentions TYPE record<source>;
DEFINE FIELD out ON mentions TYPE record<entity>;
DEFINE FIELD context ON mentions TYPE string;
DEFINE FIELD confidence ON mentions TYPE float;

DEFINE TABLE supports SCHEMAFULL;
DEFINE FIELD in ON supports TYPE record<source>;
DEFINE FIELD out ON supports TYPE record<claim>;
DEFINE FIELD evidence_type ON supports TYPE string;
DEFINE FIELD strength ON supports TYPE string;

-- Indexes
DEFINE INDEX idx_source_embedding ON source FIELDS embedding MTREE DIMENSION 1024 DIST COSINE;
DEFINE INDEX idx_entity_embedding ON entity FIELDS embedding MTREE DIMENSION 1024 DIST COSINE;
DEFINE INDEX idx_source_type ON source FIELDS source_type;
DEFINE INDEX idx_entity_type ON entity FIELDS entity_type;
DEFINE INDEX idx_source_fulltext ON source FIELDS title, content SEARCH ANALYZER ascii;
```

---

## Deel 7: Comparison Matrix

| Feature | Option A (MVP) | Option B (Typed) | Option C (Full) |
|---------|----------------|------------------|-----------------|
| **Effort** | 2-3 weken | 6-8 weken | 3-4 maanden |
| **Source types** | Generic | Specialized | Ontology-linked |
| **Entity extraction** | Basic NER | NER + linking | Full KB linking |
| **Citation network** | Simple | With context | Multi-hop analysis |
| **Claims/Evidence** | ❌ | Basic | Full pipeline |
| **Policy tracking** | ❌ | Basic | Complete lineage |
| **Contradiction detection** | ❌ | ❌ | ✅ |
| **Semantic search** | ✅ | ✅ | ✅ + reasoning |
| **Graph queries** | Basic | Rich | Full traversal |
| **Maintenance** | Low | Medium | High |

---

## Deel 8: Recommended Starting Point

**Recommendation**: Start with **Option B (Typed Hierarchy)** with phased delivery

**Rationale**:
1. Option A is too limited for meaningful policy/research tracking
2. Option C is overengineered for initial needs
3. Option B provides:
   - Clear source differentiation
   - Rich enough relationships for useful queries
   - Foundation for future expansion
   - Manageable complexity

**First 4 Weeks Priority**:
1. Source type classification + metadata schema
2. Basic entity extraction (Person, Organization)
3. Citation relationship (source→cites→source)
4. Simple mentions relationship

**Success Metrics**:
- Can answer: "Which academic papers informed this policy advice?"
- Can answer: "Who are the key experts on topic X?"
- Can answer: "What evidence supports/contradicts claim Y?"
- Can visualize: citation networks and author collaborations

---

## Appendix A: Example Queries

### Find experts on a topic
```surql
SELECT
    person.name,
    count(->authored->source) AS publications,
    count(->authored->source<-cites) AS citations,
    person.expertise_areas
FROM person
WHERE
    ->authored->source->discusses->topic.name CONTAINS "klimaatbeleid"
    AND count(->authored->source) >= 3
ORDER BY citations DESC
LIMIT 10;
```

### Trace policy from advice to legislation
```surql
SELECT
    advice.title AS advice_title,
    advice.advisory_body,
    advice.publication_date,
    ->leads_to->policy.title AS resulting_policy,
    ->leads_to->policy.effective_date,
    ->leads_to->policy->implements->policy.title AS eu_directive
FROM policy_advice AS advice
WHERE advice.policy_domain CONTAINS "klimaat"
ORDER BY advice.publication_date DESC;
```

### Find contradicting evidence
```surql
SELECT
    c.statement AS claim,
    ->supports<-source.title AS supporting_sources,
    ->contradicts<-source.title AS contradicting_sources,
    count(->supports) AS support_count,
    count(->contradicts) AS contradict_count
FROM claim AS c
WHERE
    count(->supports) > 0
    AND count(->contradicts) > 0
ORDER BY (support_count + contradict_count) DESC;
```

### LinkedIn expert activity on topic
```surql
SELECT
    p.author_name,
    p.author_handle,
    count() AS post_count,
    math::sum(p.engagement.likes) AS total_likes,
    array::distinct(p.hashtags) AS used_hashtags
FROM source AS p
WHERE
    p.source_type = "social_media"
    AND p.platform = "linkedin"
    AND p.->discusses->topic.name CONTAINS "AI"
    AND p.post_date > d"2024-01-01"
GROUP BY p.author_handle
ORDER BY total_likes DESC;
```
