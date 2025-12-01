# Knowledge Graph Implementation Plan (Option C + HippoRAG)

## Overview

This document describes the full implementation plan for the Knowledge Graph system using SurrealDB as the storage layer and an external graph analysis module (NetworkX, swappable to igraph) for advanced algorithms like Personalized PageRank (PPR).

**Decision**: Implement Option C (Full Ontology-Driven KG) enhanced with HippoRAG techniques.

---

## Part 1: Architecture

### 1.1 System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         KNOWLEDGE GRAPH SYSTEM                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                        INGESTION LAYER                                  ││
│  │                                                                         ││
│  │  Source Input → Classifier → Type-Specific Processor                   ││
│  │                                    │                                    ││
│  │                     ┌──────────────┼──────────────┐                    ││
│  │                     ▼              ▼              ▼                    ││
│  │              [PDF Parser]   [API Fetcher]   [Scraper]                  ││
│  │                     │              │              │                    ││
│  │                     └──────────────┼──────────────┘                    ││
│  │                                    ▼                                    ││
│  │  ┌─────────────────────────────────────────────────────────────────┐  ││
│  │  │                 OpenIE Extraction Pipeline                       │  ││
│  │  │  • LLM-based NER (few-shot prompts)                             │  ││
│  │  │  • Triple extraction (subject-predicate-object)                 │  ││
│  │  │  • Claim/evidence extraction                                     │  ││
│  │  └─────────────────────────────────────────────────────────────────┘  ││
│  │                                    │                                    ││
│  │  ┌─────────────────────────────────────────────────────────────────┐  ││
│  │  │                 Entity Linking Pipeline                          │  ││
│  │  │  • KNN-based deduplication (embedding similarity > 0.8)         │  ││
│  │  │  • Synonymy edge creation (same_as relations)                   │  ││
│  │  │  • External KB linking (Wikidata, ORCID, ROR)                   │  ││
│  │  └─────────────────────────────────────────────────────────────────┘  ││
│  │                                    │                                    ││
│  │  ┌─────────────────────────────────────────────────────────────────┐  ││
│  │  │              Three-Tier Embedding Generation                     │  ││
│  │  │  • Passage embeddings (instruction: query_to_passage)           │  ││
│  │  │  • Entity embeddings (direct encoding)                          │  ││
│  │  │  • Fact embeddings (instruction: query_to_fact) on edges        │  ││
│  │  └─────────────────────────────────────────────────────────────────┘  ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                        STORAGE LAYER (SurrealDB)                        ││
│  │                                                                         ││
│  │  Nodes:                          Relations:                             ││
│  │  ├─ source (+ embedding)         ├─ cites (+ fact_embedding)           ││
│  │  ├─ entity (+ embedding)         ├─ mentions (+ context, confidence)   ││
│  │  ├─ claim (+ embedding)          ├─ supports / contradicts             ││
│  │  ├─ evidence                     ├─ same_as (synonymy, similarity)     ││
│  │  ├─ person                       ├─ authored_by                        ││
│  │  ├─ organization                 ├─ affiliated_with                    ││
│  │  └─ topic                        └─ discusses                          ││
│  │                                                                         ││
│  │  Indexes:                                                               ││
│  │  ├─ Vector (MTREE COSINE) on all embeddings                            ││
│  │  ├─ Full-text (dutch + english analyzers)                              ││
│  │  └─ Standard indexes on type fields                                    ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                   GRAPH ANALYSIS LAYER (External)                       ││
│  │                                                                         ││
│  │  ┌─────────────────────────────────────────────────────────────────┐  ││
│  │  │              GraphAnalyzer (Abstraction Layer)                   │  ││
│  │  │                                                                  │  ││
│  │  │  Interface:                                                      │  ││
│  │  │  • personalized_pagerank(reset_prob, damping)                   │  ││
│  │  │  • get_centrality(method: betweenness|eigenvector|degree)       │  ││
│  │  │  • detect_communities(algorithm: louvain|label_propagation)     │  ││
│  │  │  • find_shortest_path(source, target)                           │  ││
│  │  │  • get_neighbors(node, hops)                                    │  ││
│  │  │  • compute_influence_score(node)                                │  ││
│  │  └─────────────────────────────────────────────────────────────────┘  ││
│  │                           │                                            ││
│  │              ┌────────────┴────────────┐                              ││
│  │              ▼                         ▼                              ││
│  │  ┌─────────────────────┐   ┌─────────────────────┐                   ││
│  │  │  NetworkXBackend    │   │  IGraphBackend      │                   ││
│  │  │  (Current)          │   │  (Future)           │                   ││
│  │  │                     │   │                     │                   ││
│  │  │  • Pure Python      │   │  • C-based (fast)   │                   ││
│  │  │  • 500+ algorithms  │   │  • 10-100x faster   │                   ││
│  │  │  • Easy debugging   │   │  • Memory efficient │                   ││
│  │  └─────────────────────┘   └─────────────────────┘                   ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                      │                                      │
│                                      ▼                                      │
│  ┌────────────────────────────────────────────────────────────────────────┐│
│  │                        RETRIEVAL LAYER                                  ││
│  │                                                                         ││
│  │  Query → [Dual Embedding] → Fact Scoring → LLM Reranking               ││
│  │                                   │                                     ││
│  │                          Entity Extraction                              ││
│  │                                   │                                     ││
│  │                  ┌────────────────┴────────────────┐                   ││
│  │                  ▼                                 ▼                   ││
│  │         PPR Graph Search                   Dense Fallback              ││
│  │         (if entities found)                (if no entities)            ││
│  │                  │                                 │                   ││
│  │                  └────────────────┬────────────────┘                   ││
│  │                                   ▼                                     ││
│  │                          Ranked Passages                                ││
│  └────────────────────────────────────────────────────────────────────────┘│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Responsibilities

| Component | Responsibility | Technology |
|-----------|----------------|------------|
| **Ingestion Layer** | Parse sources, extract entities/relations, generate embeddings | Python, LLM APIs |
| **Storage Layer** | Persist nodes, edges, embeddings; provide basic queries | SurrealDB |
| **Graph Analysis Layer** | PPR, centrality, community detection, path analysis | NetworkX (→ igraph) |
| **Retrieval Layer** | Combine vector search + graph algorithms for ranking | Python orchestration |

---

## Part 2: SurrealDB Schema (Extended)

### 2.1 Core Tables

```surql
-- ============================================================================
-- SOURCE TABLE (extended with HippoRAG concepts)
-- ============================================================================
DEFINE TABLE source SCHEMAFULL;

-- Core fields
DEFINE FIELD source_type ON source TYPE string
    ASSERT $value IN ["academic_paper", "policy_document", "policy_advice",
                      "social_media", "news_article", "legal_document",
                      "report", "presentation", "other"];
DEFINE FIELD title ON source TYPE string;
DEFINE FIELD content ON source TYPE string;
DEFINE FIELD summary ON source TYPE option<string>;

-- Type-specific metadata (flexible object)
DEFINE FIELD type_metadata ON source TYPE object;
-- Examples:
-- academic_paper: {doi, arxiv_id, journal, volume, issue, peer_reviewed, methodology}
-- policy_document: {jurisdiction, document_number, status, effective_date}
-- social_media: {platform, post_id, engagement: {likes, shares, comments}}

-- External identifiers
DEFINE FIELD external_ids ON source TYPE object;
-- {doi: "...", arxiv: "...", wikidata: "...", url: "..."}

-- Embeddings (HippoRAG: passage-level)
DEFINE FIELD embedding ON source TYPE array<float>;

-- Timestamps
DEFINE FIELD created ON source TYPE datetime DEFAULT time::now();
DEFINE FIELD updated ON source TYPE datetime DEFAULT time::now();

-- Computed scores (cached from graph analysis)
DEFINE FIELD cached_scores ON source TYPE option<object>;
-- {pagerank: float, centrality: float, influence: float, last_computed: datetime}


-- ============================================================================
-- ENTITY TABLE (HippoRAG: entity nodes with embeddings)
-- ============================================================================
DEFINE TABLE entity SCHEMAFULL;

DEFINE FIELD name ON entity TYPE string;
DEFINE FIELD entity_type ON entity TYPE string
    ASSERT $value IN ["person", "organization", "topic", "location",
                      "concept", "event", "product", "other"];
DEFINE FIELD aliases ON entity TYPE array<string> DEFAULT [];
DEFINE FIELD description ON entity TYPE option<string>;

-- External KB links
DEFINE FIELD external_ids ON entity TYPE object DEFAULT {};
-- {wikidata: "Q...", orcid: "...", ror: "...", linkedin: "..."}

-- Entity embedding (for KNN deduplication)
DEFINE FIELD embedding ON entity TYPE array<float>;

-- Hash ID for deduplication (HippoRAG style)
DEFINE FIELD hash_id ON entity TYPE string;

DEFINE FIELD created ON entity TYPE datetime DEFAULT time::now();


-- ============================================================================
-- CLAIM TABLE (for fact tracking)
-- ============================================================================
DEFINE TABLE claim SCHEMAFULL;

DEFINE FIELD statement ON claim TYPE string;
DEFINE FIELD claim_type ON claim TYPE string
    ASSERT $value IN ["factual", "causal", "normative", "predictive"];
DEFINE FIELD verification_status ON claim TYPE string DEFAULT "unverified"
    ASSERT $value IN ["unverified", "supported", "contested", "refuted"];
DEFINE FIELD confidence ON claim TYPE option<float>;
DEFINE FIELD first_appearance ON claim TYPE option<datetime>;

DEFINE FIELD embedding ON claim TYPE array<float>;
DEFINE FIELD created ON claim TYPE datetime DEFAULT time::now();


-- ============================================================================
-- EVIDENCE TABLE
-- ============================================================================
DEFINE TABLE evidence SCHEMAFULL;

DEFINE FIELD description ON evidence TYPE string;
DEFINE FIELD evidence_type ON evidence TYPE string
    ASSERT $value IN ["statistical", "qualitative", "experimental",
                      "observational", "expert_opinion", "case_study"];
DEFINE FIELD strength ON evidence TYPE string DEFAULT "moderate"
    ASSERT $value IN ["weak", "moderate", "strong"];
DEFINE FIELD methodology ON evidence TYPE option<string>;
DEFINE FIELD sample_size ON evidence TYPE option<int>;

DEFINE FIELD embedding ON evidence TYPE array<float>;
DEFINE FIELD source_id ON evidence TYPE record<source>;
DEFINE FIELD created ON evidence TYPE datetime DEFAULT time::now();


-- ============================================================================
-- PERSON TABLE
-- ============================================================================
DEFINE TABLE person SCHEMAFULL;

DEFINE FIELD name ON person TYPE string;
DEFINE FIELD aliases ON person TYPE array<string> DEFAULT [];
DEFINE FIELD orcid ON person TYPE option<string>;
DEFINE FIELD email ON person TYPE option<string>;
DEFINE FIELD bio ON person TYPE option<string>;
DEFINE FIELD expertise_areas ON person TYPE array<string> DEFAULT [];
DEFINE FIELD h_index ON person TYPE option<int>;
DEFINE FIELD current_position ON person TYPE option<string>;

DEFINE FIELD external_ids ON person TYPE object DEFAULT {};
-- {linkedin: "...", twitter: "...", google_scholar: "..."}

DEFINE FIELD embedding ON person TYPE array<float>;
DEFINE FIELD created ON person TYPE datetime DEFAULT time::now();


-- ============================================================================
-- ORGANIZATION TABLE
-- ============================================================================
DEFINE TABLE organization SCHEMAFULL;

DEFINE FIELD name ON organization TYPE string;
DEFINE FIELD aliases ON organization TYPE array<string> DEFAULT [];
DEFINE FIELD org_type ON organization TYPE string
    ASSERT $value IN ["university", "research_institute", "think_tank",
                      "advisory_body", "government", "ngo", "company",
                      "international_org", "other"];
DEFINE FIELD country ON organization TYPE option<string>;
DEFINE FIELD city ON organization TYPE option<string>;
DEFINE FIELD website ON organization TYPE option<string>;
DEFINE FIELD description ON organization TYPE option<string>;
DEFINE FIELD ror_id ON organization TYPE option<string>;

DEFINE FIELD embedding ON organization TYPE array<float>;
DEFINE FIELD created ON organization TYPE datetime DEFAULT time::now();


-- ============================================================================
-- TOPIC TABLE (hierarchical)
-- ============================================================================
DEFINE TABLE topic SCHEMAFULL;

DEFINE FIELD name ON topic TYPE string;
DEFINE FIELD description ON topic TYPE option<string>;
DEFINE FIELD level ON topic TYPE string DEFAULT "specific"
    ASSERT $value IN ["broad", "specific", "narrow"];
DEFINE FIELD domain ON topic TYPE option<string>;
DEFINE FIELD wikidata_id ON topic TYPE option<string>;

DEFINE FIELD embedding ON topic TYPE array<float>;
DEFINE FIELD created ON topic TYPE datetime DEFAULT time::now();
```

### 2.2 Relationship Tables

```surql
-- ============================================================================
-- CITES (source -> source) with fact embedding
-- ============================================================================
DEFINE TABLE cites SCHEMAFULL TYPE RELATION FROM source TO source;

DEFINE FIELD citation_context ON cites TYPE option<string>;
DEFINE FIELD section ON cites TYPE option<string>;  -- "methodology", "literature_review", etc.
DEFINE FIELD sentiment ON cites TYPE option<string>
    ASSERT $value IN ["supportive", "critical", "neutral", none];

-- HippoRAG: fact embedding for the citation relationship
DEFINE FIELD fact_text ON cites TYPE option<string>;  -- "Paper A cites Paper B for methodology"
DEFINE FIELD fact_embedding ON cites TYPE option<array<float>>;

DEFINE FIELD created ON cites TYPE datetime DEFAULT time::now();


-- ============================================================================
-- MENTIONS (source -> entity) with context
-- ============================================================================
DEFINE TABLE mentions SCHEMAFULL TYPE RELATION FROM source TO entity;

DEFINE FIELD context ON mentions TYPE option<string>;  -- surrounding text
DEFINE FIELD confidence ON mentions TYPE float DEFAULT 1.0;
DEFINE FIELD position ON mentions TYPE option<object>;  -- {start: int, end: int, chunk_id: string}
DEFINE FIELD extraction_method ON mentions TYPE string DEFAULT "ner";

DEFINE FIELD created ON mentions TYPE datetime DEFAULT time::now();


-- ============================================================================
-- SUPPORTS / CONTRADICTS (source/evidence -> claim)
-- ============================================================================
DEFINE TABLE supports SCHEMAFULL TYPE RELATION FROM source TO claim;
DEFINE FIELD evidence_type ON supports TYPE option<string>;
DEFINE FIELD strength ON supports TYPE string DEFAULT "moderate";
DEFINE FIELD quote ON supports TYPE option<string>;
DEFINE FIELD created ON supports TYPE datetime DEFAULT time::now();

DEFINE TABLE contradicts SCHEMAFULL TYPE RELATION FROM source TO claim;
DEFINE FIELD evidence_type ON contradicts TYPE option<string>;
DEFINE FIELD strength ON contradicts TYPE string DEFAULT "moderate";
DEFINE FIELD quote ON contradicts TYPE option<string>;
DEFINE FIELD created ON contradicts TYPE datetime DEFAULT time::now();


-- ============================================================================
-- SAME_AS (entity -> entity) for synonymy/deduplication
-- ============================================================================
DEFINE TABLE same_as SCHEMAFULL TYPE RELATION FROM entity TO entity;

DEFINE FIELD similarity ON same_as TYPE float;  -- cosine similarity score
DEFINE FIELD method ON same_as TYPE string DEFAULT "embedding_knn";
DEFINE FIELD verified ON same_as TYPE bool DEFAULT false;  -- human-verified?
DEFINE FIELD created ON same_as TYPE datetime DEFAULT time::now();


-- ============================================================================
-- AUTHORED_BY (source -> person)
-- ============================================================================
DEFINE TABLE authored_by SCHEMAFULL TYPE RELATION FROM source TO person;

DEFINE FIELD role ON authored_by TYPE string DEFAULT "author"
    ASSERT $value IN ["author", "lead_author", "corresponding", "contributor"];
DEFINE FIELD position ON authored_by TYPE option<int>;  -- author order
DEFINE FIELD contribution ON authored_by TYPE option<string>;
DEFINE FIELD created ON authored_by TYPE datetime DEFAULT time::now();


-- ============================================================================
-- AFFILIATED_WITH (person -> organization)
-- ============================================================================
DEFINE TABLE affiliated_with SCHEMAFULL TYPE RELATION FROM person TO organization;

DEFINE FIELD role ON affiliated_with TYPE option<string>;
DEFINE FIELD department ON affiliated_with TYPE option<string>;
DEFINE FIELD start_date ON affiliated_with TYPE option<datetime>;
DEFINE FIELD end_date ON affiliated_with TYPE option<datetime>;
DEFINE FIELD is_current ON affiliated_with TYPE bool DEFAULT true;
DEFINE FIELD created ON affiliated_with TYPE datetime DEFAULT time::now();


-- ============================================================================
-- DISCUSSES (source -> topic)
-- ============================================================================
DEFINE TABLE discusses SCHEMAFULL TYPE RELATION FROM source TO topic;

DEFINE FIELD relevance ON discusses TYPE float DEFAULT 1.0;
DEFINE FIELD is_primary ON discusses TYPE bool DEFAULT false;
DEFINE FIELD created ON discusses TYPE datetime DEFAULT time::now();


-- ============================================================================
-- TOPIC HIERARCHY (topic -> topic)
-- ============================================================================
DEFINE TABLE broader_than SCHEMAFULL TYPE RELATION FROM topic TO topic;
DEFINE FIELD created ON broader_than TYPE datetime DEFAULT time::now();

DEFINE TABLE related_to SCHEMAFULL TYPE RELATION FROM topic TO topic;
DEFINE FIELD strength ON related_to TYPE float DEFAULT 1.0;
DEFINE FIELD created ON related_to TYPE datetime DEFAULT time::now();


-- ============================================================================
-- POLICY-SPECIFIC RELATIONS
-- ============================================================================
DEFINE TABLE implements SCHEMAFULL TYPE RELATION FROM source TO source;
-- e.g., NL law implements EU directive
DEFINE FIELD implementation_date ON implements TYPE option<datetime>;
DEFINE FIELD compliance_status ON implements TYPE option<string>;
DEFINE FIELD created ON implements TYPE datetime DEFAULT time::now();

DEFINE TABLE supersedes SCHEMAFULL TYPE RELATION FROM source TO source;
DEFINE FIELD effective_date ON supersedes TYPE option<datetime>;
DEFINE FIELD created ON supersedes TYPE datetime DEFAULT time::now();

DEFINE TABLE leads_to SCHEMAFULL TYPE RELATION FROM source TO source;
-- e.g., policy advice leads to policy document
DEFINE FIELD influence_type ON leads_to TYPE option<string>;
DEFINE FIELD created ON leads_to TYPE datetime DEFAULT time::now();
```

### 2.3 Indexes

```surql
-- ============================================================================
-- VECTOR INDEXES (for semantic search)
-- ============================================================================
DEFINE INDEX idx_source_embedding ON source
    FIELDS embedding MTREE DIMENSION 1024 DIST COSINE;

DEFINE INDEX idx_entity_embedding ON entity
    FIELDS embedding MTREE DIMENSION 1024 DIST COSINE;

DEFINE INDEX idx_claim_embedding ON claim
    FIELDS embedding MTREE DIMENSION 1024 DIST COSINE;

DEFINE INDEX idx_person_embedding ON person
    FIELDS embedding MTREE DIMENSION 1024 DIST COSINE;

DEFINE INDEX idx_topic_embedding ON topic
    FIELDS embedding MTREE DIMENSION 1024 DIST COSINE;

-- Fact embeddings on edges
DEFINE INDEX idx_cites_fact_embedding ON cites
    FIELDS fact_embedding MTREE DIMENSION 1024 DIST COSINE;


-- ============================================================================
-- FULL-TEXT INDEXES
-- ============================================================================
DEFINE ANALYZER dutch_analyzer
    TOKENIZERS blank, class
    FILTERS lowercase, snowball(nld);

DEFINE ANALYZER english_analyzer
    TOKENIZERS blank, class
    FILTERS lowercase, snowball(eng);

DEFINE INDEX idx_source_fulltext_nl ON source
    FIELDS title, content SEARCH ANALYZER dutch_analyzer;

DEFINE INDEX idx_source_fulltext_en ON source
    FIELDS title, content SEARCH ANALYZER english_analyzer;

DEFINE INDEX idx_claim_fulltext ON claim
    FIELDS statement SEARCH ANALYZER english_analyzer;


-- ============================================================================
-- STANDARD INDEXES
-- ============================================================================
DEFINE INDEX idx_source_type ON source FIELDS source_type;
DEFINE INDEX idx_entity_type ON entity FIELDS entity_type;
DEFINE INDEX idx_entity_hash ON entity FIELDS hash_id UNIQUE;
DEFINE INDEX idx_person_orcid ON person FIELDS orcid;
DEFINE INDEX idx_organization_ror ON organization FIELDS ror_id;
DEFINE INDEX idx_source_external_doi ON source FIELDS external_ids.doi;
```

---

## Part 3: Graph Analyzer Module

### 3.1 Abstraction Layer Design

```python
# open_notebook/graphs/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


class CentralityMethod(str, Enum):
    DEGREE = "degree"
    BETWEENNESS = "betweenness"
    EIGENVECTOR = "eigenvector"
    PAGERANK = "pagerank"
    CLOSENESS = "closeness"


class CommunityAlgorithm(str, Enum):
    LOUVAIN = "louvain"
    LABEL_PROPAGATION = "label_propagation"
    GREEDY_MODULARITY = "greedy_modularity"


@dataclass
class NodeInfo:
    """Information about a node in the graph."""
    id: str
    node_type: str  # "source", "entity", "person", etc.
    attributes: Dict[str, Any]


@dataclass
class EdgeInfo:
    """Information about an edge in the graph."""
    source_id: str
    target_id: str
    edge_type: str  # "cites", "mentions", "same_as", etc.
    weight: float
    attributes: Dict[str, Any]


@dataclass
class PPRResult:
    """Result of Personalized PageRank computation."""
    node_ids: List[str]
    scores: np.ndarray

    def top_k(self, k: int) -> List[Tuple[str, float]]:
        """Return top-k nodes by score."""
        indices = np.argsort(self.scores)[::-1][:k]
        return [(self.node_ids[i], self.scores[i]) for i in indices]


@dataclass
class CommunityResult:
    """Result of community detection."""
    node_to_community: Dict[str, int]
    communities: Dict[int, List[str]]
    modularity: float


class GraphBackend(ABC):
    """
    Abstract base class for graph analysis backends.

    Implementations: NetworkXBackend, IGraphBackend (future)
    """

    @abstractmethod
    def load_from_edges(
        self,
        edges: List[EdgeInfo],
        nodes: Optional[List[NodeInfo]] = None
    ) -> None:
        """Load graph from edge list."""
        pass

    @abstractmethod
    def add_node(self, node: NodeInfo) -> None:
        """Add a single node."""
        pass

    @abstractmethod
    def add_edge(self, edge: EdgeInfo) -> None:
        """Add a single edge."""
        pass

    @abstractmethod
    def remove_node(self, node_id: str) -> None:
        """Remove a node and its edges."""
        pass

    @abstractmethod
    def get_node_count(self) -> int:
        """Return number of nodes."""
        pass

    @abstractmethod
    def get_edge_count(self) -> int:
        """Return number of edges."""
        pass

    @abstractmethod
    def personalized_pagerank(
        self,
        reset_prob: Dict[str, float],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> PPRResult:
        """
        Compute Personalized PageRank.

        Args:
            reset_prob: Dict mapping node_id to reset probability (teleportation vector)
            damping: Probability of following edges (vs teleporting)
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            PPRResult with node scores
        """
        pass

    @abstractmethod
    def get_centrality(
        self,
        method: CentralityMethod,
        **kwargs
    ) -> Dict[str, float]:
        """Compute centrality scores for all nodes."""
        pass

    @abstractmethod
    def detect_communities(
        self,
        algorithm: CommunityAlgorithm,
        **kwargs
    ) -> CommunityResult:
        """Detect communities in the graph."""
        pass

    @abstractmethod
    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        weight: Optional[str] = None
    ) -> Optional[List[str]]:
        """Find shortest path between two nodes."""
        pass

    @abstractmethod
    def get_neighbors(
        self,
        node_id: str,
        hops: int = 1,
        edge_types: Optional[List[str]] = None
    ) -> List[str]:
        """Get neighbors within n hops."""
        pass

    @abstractmethod
    def get_subgraph(
        self,
        node_ids: List[str],
        include_edges_between: bool = True
    ) -> "GraphBackend":
        """Extract subgraph containing specified nodes."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        pass

    @abstractmethod
    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load graph from dictionary."""
        pass
```

### 3.2 NetworkX Implementation

```python
# open_notebook/graphs/networkx_backend.py

import networkx as nx
import numpy as np
from typing import Any, Dict, List, Optional

from open_notebook.graphs.base import (
    GraphBackend,
    NodeInfo,
    EdgeInfo,
    PPRResult,
    CommunityResult,
    CentralityMethod,
    CommunityAlgorithm,
)


class NetworkXBackend(GraphBackend):
    """NetworkX implementation of GraphBackend."""

    def __init__(self):
        self._graph: nx.DiGraph = nx.DiGraph()
        self._node_types: Dict[str, str] = {}  # node_id -> type
        self._edge_types: Dict[tuple, str] = {}  # (src, tgt) -> type

    def load_from_edges(
        self,
        edges: List[EdgeInfo],
        nodes: Optional[List[NodeInfo]] = None
    ) -> None:
        """Load graph from edge list."""
        self._graph = nx.DiGraph()
        self._node_types = {}
        self._edge_types = {}

        # Add nodes first if provided
        if nodes:
            for node in nodes:
                self.add_node(node)

        # Add edges
        for edge in edges:
            self.add_edge(edge)

    def add_node(self, node: NodeInfo) -> None:
        """Add a single node."""
        self._graph.add_node(node.id, **node.attributes)
        self._node_types[node.id] = node.node_type

    def add_edge(self, edge: EdgeInfo) -> None:
        """Add a single edge."""
        # Ensure nodes exist
        if edge.source_id not in self._graph:
            self._graph.add_node(edge.source_id)
        if edge.target_id not in self._graph:
            self._graph.add_node(edge.target_id)

        self._graph.add_edge(
            edge.source_id,
            edge.target_id,
            weight=edge.weight,
            **edge.attributes
        )
        self._edge_types[(edge.source_id, edge.target_id)] = edge.edge_type

    def remove_node(self, node_id: str) -> None:
        """Remove a node and its edges."""
        if node_id in self._graph:
            self._graph.remove_node(node_id)
            self._node_types.pop(node_id, None)
            # Clean up edge types
            self._edge_types = {
                k: v for k, v in self._edge_types.items()
                if node_id not in k
            }

    def get_node_count(self) -> int:
        return self._graph.number_of_nodes()

    def get_edge_count(self) -> int:
        return self._graph.number_of_edges()

    def personalized_pagerank(
        self,
        reset_prob: Dict[str, float],
        damping: float = 0.85,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> PPRResult:
        """
        Compute Personalized PageRank using NetworkX.

        HippoRAG uses damping=0.5 by default for more aggressive teleportation.
        """
        if not reset_prob:
            # Standard PageRank if no personalization
            scores = nx.pagerank(
                self._graph,
                alpha=damping,
                max_iter=max_iter,
                tol=tol
            )
        else:
            # Normalize reset probabilities
            total = sum(reset_prob.values())
            if total > 0:
                personalization = {k: v/total for k, v in reset_prob.items()}
            else:
                personalization = None

            scores = nx.pagerank(
                self._graph,
                alpha=damping,
                personalization=personalization,
                max_iter=max_iter,
                tol=tol
            )

        node_ids = list(scores.keys())
        score_array = np.array([scores[n] for n in node_ids])

        return PPRResult(node_ids=node_ids, scores=score_array)

    def get_centrality(
        self,
        method: CentralityMethod,
        **kwargs
    ) -> Dict[str, float]:
        """Compute centrality scores."""
        if method == CentralityMethod.DEGREE:
            # Normalize by n-1
            return dict(nx.degree_centrality(self._graph))

        elif method == CentralityMethod.BETWEENNESS:
            return dict(nx.betweenness_centrality(
                self._graph,
                weight=kwargs.get("weight", "weight"),
                normalized=True
            ))

        elif method == CentralityMethod.EIGENVECTOR:
            try:
                return dict(nx.eigenvector_centrality(
                    self._graph,
                    max_iter=kwargs.get("max_iter", 100),
                    weight=kwargs.get("weight", "weight")
                ))
            except nx.PowerIterationFailedConvergence:
                # Fallback to numpy-based computation
                return dict(nx.eigenvector_centrality_numpy(
                    self._graph,
                    weight=kwargs.get("weight", "weight")
                ))

        elif method == CentralityMethod.PAGERANK:
            return dict(nx.pagerank(
                self._graph,
                alpha=kwargs.get("alpha", 0.85),
                weight=kwargs.get("weight", "weight")
            ))

        elif method == CentralityMethod.CLOSENESS:
            return dict(nx.closeness_centrality(self._graph))

        else:
            raise ValueError(f"Unknown centrality method: {method}")

    def detect_communities(
        self,
        algorithm: CommunityAlgorithm,
        **kwargs
    ) -> CommunityResult:
        """Detect communities using specified algorithm."""
        # Convert to undirected for community detection
        undirected = self._graph.to_undirected()

        if algorithm == CommunityAlgorithm.LOUVAIN:
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(
                    undirected,
                    weight=kwargs.get("weight", "weight"),
                    resolution=kwargs.get("resolution", 1.0)
                )
                modularity = community_louvain.modularity(partition, undirected)
            except ImportError:
                # Fallback to greedy modularity
                return self.detect_communities(
                    CommunityAlgorithm.GREEDY_MODULARITY,
                    **kwargs
                )

        elif algorithm == CommunityAlgorithm.LABEL_PROPAGATION:
            communities = list(nx.community.label_propagation_communities(undirected))
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
            modularity = nx.community.modularity(undirected, communities)

        elif algorithm == CommunityAlgorithm.GREEDY_MODULARITY:
            communities = list(nx.community.greedy_modularity_communities(
                undirected,
                weight=kwargs.get("weight", "weight")
            ))
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
            modularity = nx.community.modularity(undirected, communities)

        else:
            raise ValueError(f"Unknown community algorithm: {algorithm}")

        # Build communities dict
        communities_dict: Dict[int, List[str]] = {}
        for node, comm_id in partition.items():
            if comm_id not in communities_dict:
                communities_dict[comm_id] = []
            communities_dict[comm_id].append(node)

        return CommunityResult(
            node_to_community=partition,
            communities=communities_dict,
            modularity=modularity
        )

    def shortest_path(
        self,
        source_id: str,
        target_id: str,
        weight: Optional[str] = None
    ) -> Optional[List[str]]:
        """Find shortest path."""
        try:
            return nx.shortest_path(
                self._graph,
                source=source_id,
                target=target_id,
                weight=weight
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def get_neighbors(
        self,
        node_id: str,
        hops: int = 1,
        edge_types: Optional[List[str]] = None
    ) -> List[str]:
        """Get neighbors within n hops."""
        if node_id not in self._graph:
            return []

        visited = {node_id}
        current_level = {node_id}

        for _ in range(hops):
            next_level = set()
            for node in current_level:
                # Outgoing edges
                for neighbor in self._graph.successors(node):
                    if neighbor not in visited:
                        # Filter by edge type if specified
                        if edge_types:
                            edge_type = self._edge_types.get((node, neighbor))
                            if edge_type not in edge_types:
                                continue
                        next_level.add(neighbor)

                # Incoming edges
                for neighbor in self._graph.predecessors(node):
                    if neighbor not in visited:
                        if edge_types:
                            edge_type = self._edge_types.get((neighbor, node))
                            if edge_type not in edge_types:
                                continue
                        next_level.add(neighbor)

            visited.update(next_level)
            current_level = next_level

        visited.discard(node_id)  # Remove starting node
        return list(visited)

    def get_subgraph(
        self,
        node_ids: List[str],
        include_edges_between: bool = True
    ) -> "NetworkXBackend":
        """Extract subgraph."""
        subgraph = NetworkXBackend()

        node_set = set(node_ids)

        # Add nodes
        for node_id in node_ids:
            if node_id in self._graph:
                subgraph._graph.add_node(
                    node_id,
                    **self._graph.nodes[node_id]
                )
                if node_id in self._node_types:
                    subgraph._node_types[node_id] = self._node_types[node_id]

        # Add edges
        if include_edges_between:
            for src, tgt, data in self._graph.edges(data=True):
                if src in node_set and tgt in node_set:
                    subgraph._graph.add_edge(src, tgt, **data)
                    if (src, tgt) in self._edge_types:
                        subgraph._edge_types[(src, tgt)] = self._edge_types[(src, tgt)]

        return subgraph

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "nodes": [
                {
                    "id": n,
                    "type": self._node_types.get(n, "unknown"),
                    "attributes": dict(self._graph.nodes[n])
                }
                for n in self._graph.nodes()
            ],
            "edges": [
                {
                    "source": src,
                    "target": tgt,
                    "type": self._edge_types.get((src, tgt), "unknown"),
                    "weight": data.get("weight", 1.0),
                    "attributes": {k: v for k, v in data.items() if k != "weight"}
                }
                for src, tgt, data in self._graph.edges(data=True)
            ]
        }

    def from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        self._graph = nx.DiGraph()
        self._node_types = {}
        self._edge_types = {}

        for node_data in data.get("nodes", []):
            self._graph.add_node(node_data["id"], **node_data.get("attributes", {}))
            self._node_types[node_data["id"]] = node_data.get("type", "unknown")

        for edge_data in data.get("edges", []):
            self._graph.add_edge(
                edge_data["source"],
                edge_data["target"],
                weight=edge_data.get("weight", 1.0),
                **edge_data.get("attributes", {})
            )
            self._edge_types[(edge_data["source"], edge_data["target"])] = \
                edge_data.get("type", "unknown")
```

### 3.3 Graph Analyzer (High-Level Interface)

```python
# open_notebook/graphs/analyzer.py

from typing import Any, Dict, List, Optional, Tuple, Type
import numpy as np
from loguru import logger

from open_notebook.graphs.base import (
    GraphBackend,
    NodeInfo,
    EdgeInfo,
    PPRResult,
    CommunityResult,
    CentralityMethod,
    CommunityAlgorithm,
)
from open_notebook.graphs.networkx_backend import NetworkXBackend
from open_notebook.database.repository import repo_query


class GraphAnalyzer:
    """
    High-level interface for graph analysis operations.

    Manages:
    - Loading graph data from SurrealDB
    - Caching and synchronization
    - Algorithm execution via backend
    - Result caching back to SurrealDB
    """

    def __init__(
        self,
        backend_class: Type[GraphBackend] = NetworkXBackend,
        cache_ttl_seconds: int = 3600
    ):
        self._backend: GraphBackend = backend_class()
        self._cache_ttl = cache_ttl_seconds
        self._last_sync: Optional[float] = None
        self._node_id_to_surreal_id: Dict[str, str] = {}

    @property
    def backend(self) -> GraphBackend:
        """Access underlying backend for advanced operations."""
        return self._backend

    async def load_full_graph(self, force_reload: bool = False) -> None:
        """
        Load complete graph from SurrealDB.

        For large graphs, consider using load_subgraph() instead.
        """
        import time

        if not force_reload and self._last_sync:
            if time.time() - self._last_sync < self._cache_ttl:
                logger.debug("Using cached graph")
                return

        logger.info("Loading full graph from SurrealDB...")

        nodes = []
        edges = []

        # Load source nodes
        sources = await repo_query("SELECT id, source_type FROM source")
        for s in sources:
            nodes.append(NodeInfo(
                id=s["id"],
                node_type="source",
                attributes={"source_type": s.get("source_type")}
            ))

        # Load entity nodes
        entities = await repo_query("SELECT id, entity_type, name FROM entity")
        for e in entities:
            nodes.append(NodeInfo(
                id=e["id"],
                node_type="entity",
                attributes={
                    "entity_type": e.get("entity_type"),
                    "name": e.get("name")
                }
            ))

        # Load person nodes
        persons = await repo_query("SELECT id, name FROM person")
        for p in persons:
            nodes.append(NodeInfo(
                id=p["id"],
                node_type="person",
                attributes={"name": p.get("name")}
            ))

        # Load edges: cites
        cites = await repo_query("SELECT in, out FROM cites")
        for c in cites:
            edges.append(EdgeInfo(
                source_id=c["in"],
                target_id=c["out"],
                edge_type="cites",
                weight=1.0,
                attributes={}
            ))

        # Load edges: mentions
        mentions = await repo_query("SELECT in, out, confidence FROM mentions")
        for m in mentions:
            edges.append(EdgeInfo(
                source_id=m["in"],
                target_id=m["out"],
                edge_type="mentions",
                weight=m.get("confidence", 1.0),
                attributes={}
            ))

        # Load edges: same_as (synonymy)
        same_as = await repo_query("SELECT in, out, similarity FROM same_as")
        for s in same_as:
            edges.append(EdgeInfo(
                source_id=s["in"],
                target_id=s["out"],
                edge_type="same_as",
                weight=s.get("similarity", 0.8),
                attributes={}
            ))

        # Load edges: authored_by
        authored = await repo_query("SELECT in, out FROM authored_by")
        for a in authored:
            edges.append(EdgeInfo(
                source_id=a["in"],
                target_id=a["out"],
                edge_type="authored_by",
                weight=1.0,
                attributes={}
            ))

        # Load edges: supports/contradicts
        supports = await repo_query("SELECT in, out, strength FROM supports")
        for s in supports:
            weight = {"weak": 0.3, "moderate": 0.6, "strong": 1.0}.get(
                s.get("strength", "moderate"), 0.6
            )
            edges.append(EdgeInfo(
                source_id=s["in"],
                target_id=s["out"],
                edge_type="supports",
                weight=weight,
                attributes={}
            ))

        contradicts = await repo_query("SELECT in, out, strength FROM contradicts")
        for c in contradicts:
            weight = {"weak": 0.3, "moderate": 0.6, "strong": 1.0}.get(
                c.get("strength", "moderate"), 0.6
            )
            edges.append(EdgeInfo(
                source_id=c["in"],
                target_id=c["out"],
                edge_type="contradicts",
                weight=weight,
                attributes={}
            ))

        # Load into backend
        self._backend.load_from_edges(edges, nodes)
        self._last_sync = time.time()

        logger.info(
            f"Graph loaded: {self._backend.get_node_count()} nodes, "
            f"{self._backend.get_edge_count()} edges"
        )

    async def load_subgraph(
        self,
        seed_ids: List[str],
        hops: int = 2,
        edge_types: Optional[List[str]] = None
    ) -> None:
        """
        Load a subgraph starting from seed nodes.

        More efficient for large graphs when you only need local analysis.
        """
        # Build query to fetch nodes within n hops
        edge_type_filter = ""
        if edge_types:
            edge_type_filter = f"AND type IN {edge_types}"

        # This is a simplified version - real implementation would use
        # recursive graph traversal in SurrealQL
        query = f"""
        SELECT * FROM (
            SELECT id, 'source' AS node_type FROM source WHERE id IN $seed_ids
            UNION ALL
            SELECT id, 'entity' AS node_type FROM entity WHERE id IN $seed_ids
        )
        """

        # For now, load full graph and extract subgraph
        await self.load_full_graph()

        # Get neighbors from loaded graph
        all_nodes = set(seed_ids)
        for seed in seed_ids:
            neighbors = self._backend.get_neighbors(seed, hops=hops, edge_types=edge_types)
            all_nodes.update(neighbors)

        # Extract subgraph
        self._backend = self._backend.get_subgraph(list(all_nodes))

        logger.info(f"Subgraph loaded: {self._backend.get_node_count()} nodes")

    async def hipporag_retrieve(
        self,
        query_embedding: np.ndarray,
        fact_scores: Dict[str, float],
        top_k: int = 20,
        damping: float = 0.5,
        passage_weight: float = 0.05
    ) -> List[Tuple[str, float]]:
        """
        HippoRAG-style retrieval using PPR.

        Args:
            query_embedding: Query vector for dense fallback
            fact_scores: Dict of entity_id -> relevance score from fact matching
            top_k: Number of results to return
            damping: PPR damping factor (HippoRAG uses 0.5)
            passage_weight: Weight multiplier for passage nodes

        Returns:
            List of (source_id, score) tuples
        """
        await self.load_full_graph()

        if not fact_scores:
            # Fallback to dense retrieval
            logger.info("No fact scores, falling back to dense retrieval")
            return await self._dense_fallback(query_embedding, top_k)

        # Build reset probability vector
        reset_prob: Dict[str, float] = {}

        # Add entity weights from fact scores
        for entity_id, score in fact_scores.items():
            if entity_id in self._backend._graph:  # type: ignore
                reset_prob[entity_id] = score

        # Add passage weights (lower weight as per HippoRAG)
        sources = await repo_query("SELECT id FROM source")
        for s in sources:
            source_id = s["id"]
            if source_id in self._backend._graph:  # type: ignore
                # Could also incorporate dense similarity here
                reset_prob[source_id] = passage_weight

        # Run PPR
        ppr_result = self._backend.personalized_pagerank(
            reset_prob=reset_prob,
            damping=damping
        )

        # Filter to only source nodes and return top-k
        source_scores = []
        for node_id, score in zip(ppr_result.node_ids, ppr_result.scores):
            if node_id.startswith("source:"):
                source_scores.append((node_id, score))

        source_scores.sort(key=lambda x: x[1], reverse=True)
        return source_scores[:top_k]

    async def _dense_fallback(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Dense retrieval fallback when graph retrieval fails."""
        # Use SurrealDB vector search
        results = await repo_query(
            """
            SELECT id, vector::similarity::cosine(embedding, $query_emb) AS score
            FROM source
            WHERE embedding != NONE
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"query_emb": query_embedding.tolist(), "limit": top_k}
        )
        return [(r["id"], r["score"]) for r in results]

    async def compute_centrality(
        self,
        method: CentralityMethod = CentralityMethod.PAGERANK,
        node_type: Optional[str] = None,
        cache_results: bool = True
    ) -> Dict[str, float]:
        """
        Compute centrality scores.

        Args:
            method: Centrality algorithm to use
            node_type: Filter to specific node type (e.g., "source", "person")
            cache_results: Whether to cache results back to SurrealDB
        """
        await self.load_full_graph()

        scores = self._backend.get_centrality(method)

        # Filter by node type if specified
        if node_type:
            scores = {
                k: v for k, v in scores.items()
                if self._backend._node_types.get(k) == node_type  # type: ignore
            }

        # Cache to SurrealDB
        if cache_results:
            await self._cache_scores_to_surreal(scores, f"centrality_{method.value}")

        return scores

    async def detect_communities(
        self,
        algorithm: CommunityAlgorithm = CommunityAlgorithm.LOUVAIN,
        cache_results: bool = True
    ) -> CommunityResult:
        """Detect communities in the graph."""
        await self.load_full_graph()

        result = self._backend.detect_communities(algorithm)

        logger.info(
            f"Found {len(result.communities)} communities "
            f"(modularity: {result.modularity:.3f})"
        )

        return result

    async def find_experts(
        self,
        topic_id: str,
        min_publications: int = 2,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find experts on a topic using graph analysis.

        Combines:
        - Publication count on topic
        - Citation centrality
        - Co-authorship network position
        """
        await self.load_full_graph()

        # Get persons who authored sources discussing the topic
        query = """
        SELECT
            person.id,
            person.name,
            count(->authored<-source) AS publications,
            count(->authored<-source<-cites) AS citations
        FROM person
        WHERE
            ->authored<-source->discusses->topic.id = $topic_id
        GROUP BY person.id
        HAVING publications >= $min_pubs
        ORDER BY citations DESC
        LIMIT $limit
        """

        results = await repo_query(query, {
            "topic_id": topic_id,
            "min_pubs": min_publications,
            "limit": top_k
        })

        # Enhance with centrality scores
        centrality = await self.compute_centrality(
            CentralityMethod.EIGENVECTOR,
            node_type="person",
            cache_results=False
        )

        for r in results:
            r["centrality"] = centrality.get(r["id"], 0.0)

        # Re-sort by combined score
        results.sort(
            key=lambda x: x["citations"] * 0.6 + x["centrality"] * 1000 * 0.4,
            reverse=True
        )

        return results

    async def trace_claim(
        self,
        claim_id: str
    ) -> Dict[str, Any]:
        """
        Trace a claim through its evidence to sources.

        Returns supporting and contradicting evidence chains.
        """
        query = """
        SELECT
            claim.id,
            claim.statement,
            claim.verification_status,
            (SELECT in AS source, strength, quote FROM supports WHERE out = $claim_id) AS supporting,
            (SELECT in AS source, strength, quote FROM contradicts WHERE out = $claim_id) AS contradicting
        FROM claim
        WHERE id = $claim_id
        """

        results = await repo_query(query, {"claim_id": claim_id})

        if not results:
            return {}

        result = results[0]

        # Get source details
        supporting_sources = []
        for s in result.get("supporting", []):
            source = await repo_query(
                "SELECT id, title, source_type FROM source WHERE id = $id",
                {"id": s["source"]}
            )
            if source:
                supporting_sources.append({
                    **source[0],
                    "strength": s["strength"],
                    "quote": s.get("quote")
                })

        contradicting_sources = []
        for c in result.get("contradicting", []):
            source = await repo_query(
                "SELECT id, title, source_type FROM source WHERE id = $id",
                {"id": c["source"]}
            )
            if source:
                contradicting_sources.append({
                    **source[0],
                    "strength": c["strength"],
                    "quote": c.get("quote")
                })

        return {
            "claim": {
                "id": result["id"],
                "statement": result["statement"],
                "status": result["verification_status"]
            },
            "supporting": supporting_sources,
            "contradicting": contradicting_sources,
            "support_count": len(supporting_sources),
            "contradict_count": len(contradicting_sources)
        }

    async def _cache_scores_to_surreal(
        self,
        scores: Dict[str, float],
        score_type: str
    ) -> None:
        """Cache computed scores back to SurrealDB."""
        import time

        for node_id, score in scores.items():
            table = node_id.split(":")[0] if ":" in node_id else None
            if not table:
                continue

            await repo_query(
                f"""
                UPDATE {node_id} SET cached_scores.{score_type} = $score,
                                     cached_scores.last_computed = $time
                """,
                {"score": score, "time": time.time()}
            )

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        return {
            "node_count": self._backend.get_node_count(),
            "edge_count": self._backend.get_edge_count(),
            "last_sync": self._last_sync,
            "backend": type(self._backend).__name__
        }
```

---

## Part 4: Retrieval Pipeline

### 4.1 HippoRAG-Style Retrieval

```python
# open_notebook/graphs/retrieval.py

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from loguru import logger

from open_notebook.graphs.analyzer import GraphAnalyzer
from open_notebook.database.repository import repo_query


class KnowledgeGraphRetriever:
    """
    HippoRAG-inspired retrieval pipeline.

    Pipeline:
    1. Query → Dual embeddings (fact + passage)
    2. Fact scoring via vector similarity
    3. LLM-based fact reranking (optional)
    4. Entity extraction from top facts
    5. PPR graph search OR dense fallback
    6. Return ranked passages
    """

    def __init__(
        self,
        graph_analyzer: GraphAnalyzer,
        embedding_model: Any,  # Your embedding model
        reranker_llm: Optional[Any] = None  # Optional LLM for fact reranking
    ):
        self.graph = graph_analyzer
        self.embed_model = embedding_model
        self.reranker = reranker_llm

        # Configuration
        self.fact_top_k = 100  # Candidates for reranking
        self.fact_after_rerank = 5  # Facts after reranking
        self.retrieval_top_k = 20  # Final documents
        self.similarity_threshold = 0.6
        self.ppr_damping = 0.5  # HippoRAG default

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval method.

        Returns list of sources with scores and metadata.
        """
        top_k = top_k or self.retrieval_top_k

        # Step 1: Generate query embeddings
        query_fact_emb = await self._embed_query(query, mode="fact")
        query_passage_emb = await self._embed_query(query, mode="passage")

        # Step 2: Score facts (relations/claims)
        fact_scores = await self._score_facts(query_fact_emb)

        # Step 3: Rerank facts (optional)
        if self.reranker and fact_scores:
            fact_scores = await self._rerank_facts(query, fact_scores)

        # Step 4: Extract entities from top facts
        entity_scores = await self._extract_entity_scores(fact_scores)

        # Step 5: Graph-based or dense retrieval
        if entity_scores:
            logger.info(f"Using PPR retrieval with {len(entity_scores)} seed entities")
            results = await self.graph.hipporag_retrieve(
                query_embedding=query_passage_emb,
                fact_scores=entity_scores,
                top_k=top_k,
                damping=self.ppr_damping
            )
        else:
            logger.info("Falling back to dense retrieval")
            results = await self._dense_retrieve(query_passage_emb, top_k)

        # Step 6: Fetch full source data
        return await self._fetch_source_details(results)

    async def _embed_query(self, query: str, mode: str = "passage") -> np.ndarray:
        """Generate query embedding with instruction."""
        if mode == "fact":
            instruction = "Represent this query for finding relevant facts and relations:"
        else:
            instruction = "Represent this query for finding relevant documents:"

        # Use your embedding model
        embedding = await self.embed_model.aembed([f"{instruction} {query}"])
        return np.array(embedding[0])

    async def _score_facts(
        self,
        query_embedding: np.ndarray
    ) -> List[Tuple[str, float, Dict]]:
        """
        Score facts using vector similarity.

        Facts are stored as:
        - claim embeddings
        - cites.fact_embedding (relation embeddings)
        """
        results = []

        # Score claims
        claims = await repo_query(
            """
            SELECT id, statement,
                   vector::similarity::cosine(embedding, $query_emb) AS score
            FROM claim
            WHERE embedding != NONE
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"query_emb": query_embedding.tolist(), "limit": self.fact_top_k}
        )

        for c in claims:
            if c["score"] >= self.similarity_threshold:
                results.append((c["id"], c["score"], {"type": "claim", "text": c["statement"]}))

        # Score citation relations with fact embeddings
        cites = await repo_query(
            """
            SELECT id, in, out, fact_text,
                   vector::similarity::cosine(fact_embedding, $query_emb) AS score
            FROM cites
            WHERE fact_embedding != NONE
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"query_emb": query_embedding.tolist(), "limit": self.fact_top_k}
        )

        for c in cites:
            if c["score"] >= self.similarity_threshold:
                results.append((c["id"], c["score"], {
                    "type": "citation",
                    "source": c["in"],
                    "target": c["out"],
                    "text": c.get("fact_text")
                }))

        # Sort by score
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:self.fact_top_k]

    async def _rerank_facts(
        self,
        query: str,
        facts: List[Tuple[str, float, Dict]]
    ) -> List[Tuple[str, float, Dict]]:
        """LLM-based fact reranking."""
        if not facts:
            return []

        # Build prompt
        fact_texts = [f["text"] or f["type"] for _, _, f in facts[:20]]  # Top 20 for LLM

        prompt = f"""Given the question: "{query}"

Select the most relevant facts that help answer this question.
Return the indices (0-based) of relevant facts, separated by commas.

Facts:
{chr(10).join(f"{i}. {text}" for i, text in enumerate(fact_texts))}

Relevant fact indices:"""

        try:
            response = await self.reranker.agenerate([prompt])
            selected_indices = [
                int(i.strip())
                for i in response.generations[0][0].text.split(",")
                if i.strip().isdigit()
            ]

            reranked = [facts[i] for i in selected_indices if i < len(facts)]
            return reranked[:self.fact_after_rerank]

        except Exception as e:
            logger.warning(f"Fact reranking failed: {e}, using top-k")
            return facts[:self.fact_after_rerank]

    async def _extract_entity_scores(
        self,
        facts: List[Tuple[str, float, Dict]]
    ) -> Dict[str, float]:
        """Extract entities from facts and compute scores."""
        entity_scores: Dict[str, float] = {}
        entity_counts: Dict[str, int] = {}

        for fact_id, score, metadata in facts:
            # Get entities related to this fact
            if metadata["type"] == "claim":
                # Get entities mentioned in sources supporting this claim
                entities = await repo_query(
                    """
                    SELECT out AS entity_id FROM mentions
                    WHERE in IN (SELECT in FROM supports WHERE out = $claim_id)
                    """,
                    {"claim_id": fact_id}
                )
            elif metadata["type"] == "citation":
                # Get entities from citing and cited sources
                entities = await repo_query(
                    """
                    SELECT out AS entity_id FROM mentions
                    WHERE in = $source OR in = $target
                    """,
                    {"source": metadata["source"], "target": metadata["target"]}
                )
            else:
                entities = []

            # Accumulate scores
            for e in entities:
                eid = e["entity_id"]
                if eid not in entity_scores:
                    entity_scores[eid] = 0.0
                    entity_counts[eid] = 0
                entity_scores[eid] += score
                entity_counts[eid] += 1

        # Normalize by occurrence count (HippoRAG style)
        for eid in entity_scores:
            if entity_counts[eid] > 0:
                entity_scores[eid] /= entity_counts[eid]

        return entity_scores

    async def _dense_retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: int
    ) -> List[Tuple[str, float]]:
        """Fallback dense retrieval."""
        results = await repo_query(
            """
            SELECT id, vector::similarity::cosine(embedding, $query_emb) AS score
            FROM source
            WHERE embedding != NONE
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"query_emb": query_embedding.tolist(), "limit": top_k}
        )
        return [(r["id"], r["score"]) for r in results]

    async def _fetch_source_details(
        self,
        results: List[Tuple[str, float]]
    ) -> List[Dict[str, Any]]:
        """Fetch full source details for results."""
        detailed = []

        for source_id, score in results:
            source = await repo_query(
                """
                SELECT id, title, source_type, summary, content,
                       ->authored_by->person.name AS authors,
                       ->discusses->topic.name AS topics
                FROM source
                WHERE id = $id
                """,
                {"id": source_id}
            )

            if source:
                detailed.append({
                    **source[0],
                    "retrieval_score": score
                })

        return detailed
```

---

## Part 5: Step-by-Step Implementation Checklist

### Phase 1: Foundation (Week 1-2) ✅ COMPLETED

> **Status**: Completed in commit `5ba3b13` on branch `feature/knowledge-graph`

#### 1.1 Schema Setup
- [x] Create SurrealDB migration script with all table definitions → `migrations/11.surrealql`
- [x] Define all indexes (vector, full-text, standard)
- [x] Create rollback migration → `migrations/11_down.surrealql`
- [ ] Test schema with sample data
- [ ] Create schema validation tests

#### 1.2 Base Models
- [x] Extend `ObjectModel` with `source_type` field
- [x] Create `Entity` model class → `open_notebook/domain/knowledge_graph.py`
- [x] Create `Claim` model class
- [x] Create `Evidence` model class
- [x] Create `Person` model class
- [x] Create `Organization` model class
- [x] Create `Topic` model class
- [x] Add relationship helper methods to models
- [x] Add HippoRAG-style entity hash computation

#### 1.3 Graph Analyzer Foundation
- [x] Create `open_notebook/graph_analysis/` directory
- [x] Implement `base.py` with abstract `GraphBackend`
- [x] Implement `networkx_backend.py`
- [ ] Write unit tests for NetworkX backend
- [x] Create `analyzer.py` with `GraphAnalyzer` class
- [x] Implement HippoRAG-style PPR retrieval
- [x] Add `networkx>=3.0` and `python-louvain>=0.16` dependencies
- [ ] Test graph loading from SurrealDB

### Phase 2: Entity Extraction (Week 3-4)

#### 2.1 OpenIE Pipeline
- [ ] Create `open_notebook/processors/openie.py`
- [ ] Implement NER extraction with LLM (few-shot prompts)
- [ ] Implement triple extraction (subject-predicate-object)
- [ ] Add entity hash ID generation
- [ ] Create extraction tests with sample documents

#### 2.2 Entity Linking
- [ ] Implement KNN-based entity deduplication
- [ ] Create `same_as` relationship creation logic
- [ ] Add external KB linking stubs (Wikidata, ORCID)
- [ ] Test deduplication with similar entities

#### 2.3 Embedding Generation
- [ ] Add three-tier embedding generation to ingestion
- [ ] Implement passage embeddings (sources)
- [ ] Implement entity embeddings
- [ ] Implement fact embeddings (on edges)
- [ ] Verify vector index functionality

### Phase 3: Graph Analysis (Week 5-6)

#### 3.1 Core Algorithms
- [ ] Implement and test PPR with custom reset probabilities
- [ ] Implement centrality calculations
- [ ] Implement community detection
- [ ] Add path finding utilities
- [ ] Create algorithm benchmarks

#### 3.2 SurrealDB Sync
- [ ] Implement efficient graph loading
- [ ] Add incremental graph updates
- [ ] Implement subgraph loading for large graphs
- [ ] Add score caching back to SurrealDB
- [ ] Test sync performance

#### 3.3 Analysis Methods
- [ ] Implement `find_experts()` method
- [ ] Implement `trace_claim()` method
- [ ] Add influence score computation
- [ ] Create analysis result models

### Phase 4: Retrieval Pipeline (Week 7-8)

#### 4.1 Fact Scoring
- [ ] Implement fact embedding search
- [ ] Add claim-based fact scoring
- [ ] Add citation-based fact scoring
- [ ] Test fact relevance ranking

#### 4.2 Reranking (Optional)
- [ ] Implement LLM-based fact reranker
- [ ] Add reranking configuration options
- [ ] Benchmark reranking impact

#### 4.3 HippoRAG Retrieval
- [ ] Implement entity extraction from facts
- [ ] Implement PPR-based passage ranking
- [ ] Implement dense fallback
- [ ] Create end-to-end retrieval tests
- [ ] Benchmark retrieval quality

### Phase 5: Integration & API (Week 9-10)

> **Note**: See `SURREALDB_KNOWLEDGE_GRAPH_PROPOSAL.md` **Deel 9** for detailed API endpoint specifications.

#### 5.1 API Endpoints (`api/routers/knowledge_graph.py`)
- [ ] Create router with `/knowledge-graph` prefix
- [ ] **Entity endpoints**:
  - [ ] `POST /entities/search` - Search entities by name/type
  - [ ] `GET /entities/{entity_id}` - Get entity with relationships
  - [ ] `GET /entities/{entity_id}/sources` - Get entity's source documents
  - [ ] `GET /entities/{entity_id}/similar` - Find similar entities
- [ ] **Claim endpoints**:
  - [ ] `GET /claims/{claim_id}` - Get claim with provenance
  - [ ] `GET /claims/{claim_id}/trace` - Trace claim to sources
  - [ ] `POST /claims/search` - Search claims
- [ ] **Expert endpoints**:
  - [ ] `GET /persons/{person_id}` - Get person profile
  - [ ] `GET /persons/{person_id}/publications` - Get publications
  - [ ] `GET /experts` - Find domain experts
- [ ] **Topic endpoints**:
  - [ ] `GET /topics` - List detected topics
  - [ ] `GET /topics/{topic_id}` - Get topic details
  - [ ] `GET /topics/{topic_id}/sources` - Sources about topic
- [ ] **Graph analysis endpoints**:
  - [ ] `GET /graph/stats` - Get graph statistics
  - [ ] `GET /graph/centrality` - Get node centrality scores
  - [ ] `GET /graph/communities` - Detect communities
  - [ ] `GET /graph/citation-network/{source_id}` - Citation network view
- [ ] **Retrieval endpoint**:
  - [ ] `POST /retrieve` - HippoRAG-style retrieval
- [ ] **Visualization endpoints**:
  - [ ] `GET /visualization/subgraph` - Get visualization data
  - [ ] `GET /visualization/topic-map` - Get topic map data
- [ ] **Source router extensions** (`api/routers/sources.py`):
  - [ ] `GET /sources/{id}/entities` - Entities from source
  - [ ] `GET /sources/{id}/claims` - Claims from source
  - [ ] `GET /sources/{id}/citations` - Citation relationships
  - [ ] `GET /sources/{id}/related` - Graph-based related sources

#### 5.2 Ingestion Integration
- [ ] Integrate OpenIE into source processing pipeline
- [ ] Add entity extraction to document ingestion
- [ ] Implement batch processing for existing sources
- [ ] Add progress tracking for bulk operations

#### 5.3 Testing & Documentation
- [ ] Create integration tests for all API endpoints
- [ ] Write OpenAPI documentation
- [ ] Create usage examples
- [ ] Performance benchmarking
- [ ] Document configuration options

### Phase 5.5: Frontend Implementation (Week 10-11)

> **Note**: See `SURREALDB_KNOWLEDGE_GRAPH_PROPOSAL.md` **Deel 10** for detailed UI component specifications and wireframes.

#### 5.5.1 Dependencies & Setup
- [ ] Add frontend dependencies:
  - [ ] `react-force-graph-2d` (or `react-force-graph-3d`)
  - [ ] `vis-network` (alternative)
  - [ ] `d3` for custom visualizations
- [ ] Create shared TypeScript interfaces in `lib/types/knowledge-graph.ts`

#### 5.5.2 Core Components (`components/knowledge-graph/`)
- [ ] `GraphVisualization.tsx` - Force-directed graph component
  - [ ] Node rendering with type-based styling
  - [ ] Edge rendering with relationship labels
  - [ ] Click, hover, and selection handlers
  - [ ] Layout options (force, hierarchical, radial)
- [ ] `EntityCard.tsx` - Entity display card
- [ ] `ClaimCard.tsx` - Claim with provenance display
- [ ] `ExpertCard.tsx` - Expert profile card
- [ ] `TopicBadge.tsx` - Topic tag component
- [ ] `GraphStats.tsx` - Graph statistics panel

#### 5.5.3 Pages (`app/(dashboard)/`)
- [ ] **Knowledge Graph Explorer** (`knowledge-graph/page.tsx`)
  - [ ] Search panel with filters
  - [ ] Interactive graph visualization
  - [ ] Node detail sidebar
  - [ ] Export functionality
- [ ] **Expert Finder** (`experts/page.tsx`)
  - [ ] Topic selection interface
  - [ ] Expert ranking display
  - [ ] Profile cards with metrics
- [ ] **Claim Tracker** (`claims/page.tsx`)
  - [ ] Claim search and filtering
  - [ ] Evidence chain visualization
  - [ ] Verification status indicators

#### 5.5.4 Source Detail Integration
- [ ] Add Knowledge Graph tab to source detail page
  - [ ] Entity list from source
  - [ ] Claims extracted from source
  - [ ] Related sources via graph
  - [ ] Mini graph visualization
- [ ] Add citation network view
- [ ] Add entity highlighting in source content

#### 5.5.5 Navigation Integration
- [ ] Add "Knowledge Graph" to sidebar navigation
- [ ] Add "Experts" to sidebar navigation
- [ ] Add "Claims" to sidebar navigation
- [ ] Update breadcrumbs for new pages

#### 5.5.6 API Integration (`lib/api/knowledge-graph.ts`)
- [ ] Create API client functions for all endpoints
- [ ] Add React Query hooks for data fetching
- [ ] Implement caching strategies
- [ ] Add error handling and loading states

### Phase 6: Advanced Features (Week 12-13)

#### 6.1 Claim Verification
- [ ] Implement claim extraction from sources
- [ ] Add evidence linking
- [ ] Implement contradiction detection (basic)
- [ ] Create verification status update logic

#### 6.2 Policy Tracking
- [ ] Implement `implements` relationship tracking
- [ ] Implement `supersedes` relationship tracking
- [ ] Implement `leads_to` relationship tracking
- [ ] Create policy lineage queries

#### 6.3 Optimization
- [ ] Profile and optimize hot paths
- [ ] Add caching layers
- [ ] Implement background graph updates
- [ ] Consider igraph migration for large graphs

---

## Part 6: Configuration

### 6.1 Environment Variables

```bash
# Graph Analysis
GRAPH_BACKEND=networkx  # or "igraph" when implemented
GRAPH_CACHE_TTL=3600    # seconds

# Retrieval
RETRIEVAL_TOP_K=20
FACT_SIMILARITY_THRESHOLD=0.6
PPR_DAMPING=0.5
ENABLE_FACT_RERANKING=false

# Entity Linking
ENTITY_SIMILARITY_THRESHOLD=0.8
ENTITY_KNN_K=100

# Embeddings
EMBEDDING_DIMENSION=1024
```

### 6.2 Python Configuration

```python
# open_notebook/config.py (additions)

from pydantic import BaseSettings

class GraphConfig(BaseSettings):
    backend: str = "networkx"
    cache_ttl: int = 3600

    class Config:
        env_prefix = "GRAPH_"

class RetrievalConfig(BaseSettings):
    top_k: int = 20
    fact_threshold: float = 0.6
    ppr_damping: float = 0.5
    enable_reranking: bool = False

    class Config:
        env_prefix = "RETRIEVAL_"

class EntityConfig(BaseSettings):
    similarity_threshold: float = 0.8
    knn_k: int = 100

    class Config:
        env_prefix = "ENTITY_"
```

---

## Part 7: Future Considerations

### 7.1 igraph Migration Path

When graph size exceeds ~50K nodes or performance becomes an issue:

1. Implement `IGraphBackend` following same interface
2. Add backend factory: `create_backend(name: str) -> GraphBackend`
3. Update configuration to allow backend selection
4. Migrate: change `GRAPH_BACKEND=igraph`
5. No other code changes required

### 7.2 Distributed Graph Processing

For very large graphs (>1M nodes):

- Consider Neo4j or TigerGraph for native graph storage
- Use SurrealDB for document storage, external graph DB for analysis
- Implement graph streaming for out-of-memory processing

### 7.3 Real-time Updates

For live knowledge graphs:

- Implement SurrealDB LIVE queries for graph updates
- Add incremental graph update methods
- Consider event-driven architecture for entity/relation changes

---

## Appendix: Dependencies

```toml
# pyproject.toml additions

[project.dependencies]
networkx = ">=3.0"
numpy = ">=1.24"
scipy = ">=1.10"  # for sparse matrix operations

[project.optional-dependencies]
graph-louvain = ["python-louvain>=0.16"]  # for Louvain community detection
graph-igraph = ["igraph>=0.11"]  # future: faster backend
```
