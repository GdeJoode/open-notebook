"""
Semantic Intelligence Layer for Open Notebook.

Loosely coupled modules that extend the knowledge graph with:
- Graph algorithms (PageRank, centrality, communities)
- Ontology management (RDF/OWL, SHACL validation, SKOS)
- Reasoning and provenance tracking
- Decision intelligence
- Entity and relation deduplication

All modules communicate exclusively via SurrealDB and the shared filesystem.
No direct Python imports between modules except for config.py and provenance
utility classes from reasoning.py.

Each module is independently runnable:
    python -m semantic_layer.graph_algorithms
    python -m semantic_layer.reasoning
    python -m semantic_layer.decision_tracker
    python -m semantic_layer.deduplication
    python -m semantic_layer.ontology_manager
"""
