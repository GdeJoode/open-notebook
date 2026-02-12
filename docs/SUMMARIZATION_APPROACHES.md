# Summarization Approaches — Inventarisatie

**Status**: Implemented (Naive, TreeKG, RAPTOR) + 11 Future Approaches (stubs)

---

## 1. Implemented Strategies

### 1.1 Naive LLM (Baseline)

**How it works**: Concatenates all chunk texts and sends them to the LLM in a single pass. When the combined text exceeds a configurable limit, it splits into overlapping windows, summarizes each window, then optionally combines partial summaries in a final pass.

**Strengths**: Simple, fast, no preprocessing required, works with any text.

**Weaknesses**: Loses long-range context in chunk-and-combine mode. Quality depends heavily on the window size and the LLM's context window.

**When to use**: Quick summaries, short documents, or when structural metadata is unavailable.

### 1.2 TreeKG (Structure-Aware)

**How it works**: Leverages the document's table-of-contents hierarchy (chapters → sections → subsections) to build a tree of summary nodes. Summarizes bottom-up: leaf sections with content are summarized first, then parent sections synthesize from their children's summaries.

**Reference**: "Tree-KG: An Expandable Knowledge Graph Construction Framework" (ACL 2025)

**Strengths**: Preserves document structure, respects authorial intent, produces natural section-level summaries.

**Weaknesses**: Requires hierarchical metadata (`section_path`). Documents without TOC structure cannot benefit.

**When to use**: Academic papers, reports, books — any document with clear hierarchical structure.

### 1.3 RAPTOR (Semantic Clustering)

**How it works**: Recursive Abstractive Processing for Tree-Organized Retrieval. Clusters chunk embeddings using GMM soft clustering, summarizes each cluster, embeds the summaries, then repeats recursively. Produces a tree of increasingly abstract summary nodes.

**Reference**: Sarthi et al. (2024), "RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval"

**Strengths**: Discovers semantic themes regardless of document order. Multi-granularity retrieval (specific → thematic). Works with flat, unstructured text.

**Weaknesses**: Requires pre-computed embeddings. Clustering quality depends on embedding quality. More compute-intensive than naive.

**When to use**: Long documents, unstructured text, or when semantic theme discovery is more important than structural fidelity.

---

## 2. Future Approaches — Standalone Strategies

### 2.1 Map-Reduce Summarization

**Concept**: Classic parallel summarization — "map" each chunk through an LLM independently, then "reduce" the partial summaries into a final one. Can be multi-stage (reduce of reduces).

**Advantages over current**:
- Embarrassingly parallel map phase
- Scales to very large documents
- Simple to implement

**Disadvantages**:
- No cross-chunk context during map phase
- Quality ceiling lower than RAPTOR for thematic coherence
- "Lost in the Middle" effect during reduce

**Implementation complexity**: Low. Could be added as a 4th strategy with minimal new code.

### 2.2 Refine / Iterative Summarization

**Concept**: Process chunks sequentially, refining a running summary. Each step sees the previous summary + next chunk, producing an updated summary.

**Advantages**:
- Maintains running context
- Good for narrative/chronological documents
- Low memory footprint

**Disadvantages**:
- Inherently sequential (no parallelism)
- Early chunks disproportionately influence final summary
- "Lost in the Middle" — middle chunks get compressed most

**Implementation complexity**: Low-medium. Requires careful prompt engineering for the refine step.

### 2.3 Walking Tree Summarization

**Concept**: A linear tree approach that first segments the document by topic boundaries (using embedding similarity or LLM-detected shifts), then builds a tree where each branch is a coherent topic segment. Unlike RAPTOR's bottom-up clustering, this walks the document linearly.

**Advantages**:
- Preserves reading order / narrative flow
- Detects natural topic boundaries
- More interpretable tree structure than RAPTOR

**Disadvantages**:
- Sensitive to topic threshold parameter
- Linear scan means no parallelism during segmentation
- May over-segment or under-segment depending on content

**Implementation complexity**: Medium. Requires topic segmentation algorithm + tree building.

### 2.4 Extractive-Abstractive Summarization

**Concept**: Two-phase approach — first extract the most important sentences using TextRank or similar graph-based ranking, then rewrite the extracted sentences into a fluent abstractive summary via LLM.

**Advantages**:
- Extraction phase is fast and deterministic
- Extracted sentences anchor the abstractive phase (less hallucination)
- Configurable extraction ratio

**Disadvantages**:
- Quality ceiling depends on extraction quality
- TextRank may miss implicit themes not represented in individual sentences
- Two-phase adds latency

**Implementation complexity**: Medium. Requires TextRank/graph extractor + LLM rewrite step.

### 2.5 Skeleton-of-Thought Summarization

**Concept**: First generate a skeleton (outline of key points), then expand each skeleton point into a paragraph. The skeleton acts as a plan that prevents drift and ensures comprehensive coverage.

**Reference**: Ning et al. (2023), "Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding"

**Advantages**:
- Skeleton ensures structural completeness
- Point-wise expansion is parallelizable
- Natural outline structure for downstream use

**Disadvantages**:
- Skeleton quality is critical — bad skeleton → bad summary
- Two LLM calls minimum (skeleton + expansion)
- May produce overly formulaic summaries

**Implementation complexity**: Medium. Skeleton generation prompt + parallel expansion.

---

## 3. Future Approaches — Enhancement Layers

Enhancement layers wrap any strategy's output to improve quality. They can be composed.

### 3.1 Chain-of-Density (CoD) Enhancer

**Concept**: Iteratively densify a summary over 5 rounds. Each round adds missing salient entities while keeping the word count roughly constant. Produces increasingly information-dense summaries.

**Reference**: Adams et al. (2023), "From Sparse to Dense: GPT-4 Summarization with Chain of Density Prompting"

**Advantages**:
- Measurably increases information density
- Systematic approach to compression
- Works with any base strategy's output

**Disadvantages**:
- 5 sequential LLM calls per summary
- May over-compress nuanced content
- Final density level may be too terse for some use cases

**Implementation complexity**: Low. Iterative prompt with entity tracking.

### 3.2 Self-Correction Enhancer

**Concept**: A two-agent system — the Summarizer produces a summary, then a Critic agent evaluates it for factual consistency, completeness, and coherence, providing corrections. Can iterate multiple rounds.

**Advantages**:
- Catches hallucinations and factual errors
- Improves completeness through systematic review
- Critic can use different model for diversity

**Disadvantages**:
- At least doubles compute cost
- Critic may introduce its own errors
- Multiple rounds increase latency linearly

**Implementation complexity**: Medium. Requires critic prompt engineering + iterative correction loop.

### 3.3 Gist Tokens Enhancer

**Concept**: Compress a summary into a compact "gist token" representation that captures the essential information in fewer tokens. Useful for downstream retrieval and as compressed context for further LLM processing.

**Reference**: Mu et al. (2023), "Learning to Compress Prompts with Gist Tokens"

**Advantages**:
- Dramatically reduces token count for storage/retrieval
- Compressed representation usable as context prefix
- Enables efficient multi-document processing

**Disadvantages**:
- Requires specialized compression model or prompting
- Lossy compression — some information lost
- Not human-readable

**Implementation complexity**: Medium-high. Requires compression strategy (prompt-based or model-based).

---

## 4. Future Approaches — Cross-Cutting Strategies

### 4.1 Hybrid RAPTOR + TreeKG

**Concept**: Use TreeKG's structural hierarchy for the first pass, then apply RAPTOR's clustering within each section to discover sub-themes that cross subsection boundaries.

**Advantages**:
- Best of both worlds: structure + semantic discovery
- Section-scoped clustering reduces noise
- Produces structurally coherent + thematically rich summaries

**Disadvantages**:
- Most complex implementation
- Requires both structural metadata AND embeddings
- Potentially over-engineers short documents

**Implementation complexity**: Medium-high. Requires orchestrating both strategies with a merge step.

### 4.2 Dense Passage Retrieval + Abstractive (DPR-Abs)

**Concept**: Instead of summarizing all chunks, use dense retrieval to select the most relevant chunks for a given query or purpose, then summarize only those. Query-focused summarization.

**Advantages**:
- Produces query-relevant summaries
- Efficient — only processes a subset of chunks
- Combines well with RAG pipelines

**Disadvantages**:
- Requires a query/purpose at summarization time
- Not suitable for general-purpose document summaries
- Misses information not relevant to the query

**Implementation complexity**: Medium. Needs retrieval infrastructure (vector store) + summarization.

### 4.3 Linked Entity Summarization

**Concept**: Produces entity-anchored mini-summaries. For each significant entity mentioned in the document, generates a focused summary of how that entity is discussed, its relationships, and key facts. Creates an entity-indexed summary map.

**Advantages**:
- Queryable by entity (person, organization, concept)
- Preserves entity-specific context often lost in global summaries
- Natural integration with knowledge graph pipelines

**Disadvantages**:
- Requires entity extraction (NER) as preprocessing
- Not a replacement for global document summaries
- Entity boundary decisions affect quality

**Implementation complexity**: Medium. Requires NER + entity-scoped summarization + aggregation.

---

## 5. Combination Recipes

Strategies and enhancers can be composed to create specialized pipelines:

### Recipe 1: "Dense Factual"

**Pipeline**: Extractive-Abstractive → Chain-of-Density → Self-Correction

**How it works**: Extract key sentences, rewrite into a fluent summary, densify iteratively, then verify facts.

**Best for**: Legal documents, technical specifications, compliance reports — where factual accuracy and density are paramount.

### Recipe 2: "RAG Optimizer"

**Pipeline**: Skeleton-of-Thought → Linked Entity → Gist Tokens

**How it works**: Generate structured skeleton, anchor entities to skeleton points, compress for efficient retrieval.

**Best for**: RAG metadata layer, search index enrichment, multi-document collections.

### Recipe 3: "Academic Paper"

**Pipeline**: TreeKG → Chain-of-Density → Self-Correction

**How it works**: Preserve paper structure, densify section summaries, fact-check against source sections.

**Best for**: Scientific papers, research reports, structured academic documents.

### Recipe 4: "Triple Output"

**Pipeline**: Any strategy → Chain-of-Density → Linked Entity → Self-Correction

**How it works**: Produces three outputs: a dense summary, an entity map, and a fact audit trail.

**Best for**: Comprehensive document processing pipeline requiring multiple output types.

---

## 6. Strategy Selection Matrix

| Criterion | Naive | TreeKG | RAPTOR | Map-Reduce | Refine | Walking Tree | Extr-Abs | Skeleton | Hybrid | DPR-Abs | Linked Entity |
|-----------|-------|--------|--------|------------|--------|--------------|----------|----------|--------|---------|---------------|
| **Structure required** | No | Yes | No | No | No | No | No | No | Yes | No | No |
| **Embeddings required** | No | No | Yes | No | No | Optional | No | No | Yes | Yes | No |
| **Query required** | No | No | No | No | No | No | No | No | No | Yes | No |
| **NER required** | No | No | No | No | No | No | No | No | No | No | Yes |
| **Parallelizable** | Partial | No | Yes | Yes | No | No | Partial | Yes | Partial | Yes | Yes |
| **Quality ceiling** | Low | High | High | Medium | Medium | Medium | Medium-High | Medium | Highest | High* | Medium |
| **Compute cost** | Low | Medium | High | Medium | Low | Medium | Medium | Medium | High | Medium | Medium |
| **Best document type** | Short/any | Structured | Long/flat | Very long | Narrative | Topic-rich | Factual | Expository | Academic | Query-focused | Entity-rich |
| **Status** | Done | Done | Done | Stub | Stub | Stub | Stub | Stub | Stub | Stub | Stub |

*Quality depends on query relevance.

### Enhancement Layer Compatibility

| Enhancer | Works with | Adds | Cost |
|----------|-----------|------|------|
| Chain-of-Density | Any strategy | Information density | 5x LLM calls |
| Self-Correction | Any strategy | Factual accuracy | 2-4x LLM calls |
| Gist Tokens | Any strategy | Compressed representation | 1x compression |

---

## 7. "Lost in the Middle" Considerations

The "Lost in the Middle" phenomenon (Liu et al., 2023) shows that LLMs attend disproportionately to the beginning and end of their context window, underweighting middle content. This affects summarization in several ways:

1. **Naive strategy**: Most vulnerable. Long concatenated texts lose middle content.
2. **Map-Reduce**: Map phase avoids it (each chunk is small), but reduce phase is vulnerable if many partials are concatenated.
3. **Refine**: Middle chunks get compressed as the running summary grows.
4. **TreeKG**: Naturally mitigates by scoping summaries to sections (smaller context per call).
5. **RAPTOR**: Mitigates via clustering — each cluster is a focused subset, not a long sequence.
6. **Walking Tree**: Mitigates through topic segmentation — each segment is bounded.
7. **Extractive-Abstractive**: Extraction phase is immune; abstractive phase sees only extracted sentences.
8. **Skeleton**: Skeleton phase sees full doc but expansion is per-point (small context).

**Mitigation strategies**:
- Keep individual LLM calls short (< 4K tokens of source text)
- Use RAPTOR or TreeKG for long documents
- For Map-Reduce, use multi-stage reduce with small batches
- For Refine, periodically re-summarize from scratch rather than endlessly refining
- Chain-of-Density inherently mitigates by forcing re-reads of the source

---

## 8. Recommendations

1. **Default**: Use **Naive** for documents under 5 pages
2. **Structured documents**: Use **TreeKG** when section headers are available
3. **Long unstructured documents**: Use **RAPTOR** for the best quality/compute trade-off
4. **Very large documents**: Implement **Map-Reduce** next (low effort, good scaling story)
5. **Factual documents**: Combine any strategy with **Chain-of-Density + Self-Correction**
6. **RAG pipelines**: Combine **Skeleton-of-Thought** with **Linked Entity** for searchable outputs
7. **Academic papers**: Use **TreeKG + Chain-of-Density** for structure-preserving dense summaries
8. **Research priority**: Investigate **Hybrid RAPTOR+TreeKG** for the highest quality ceiling
