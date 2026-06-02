"""
Semantic intelligence layer for Open Notebook.

Provides graph algorithms, reasoning, decision tracking, and the shared
SurrealDB/Ollama plumbing for the semantic-layer subsystem.

This package holds the **net new** capabilities. Two related modules
that extend existing pipelines were moved alongside their natural homes
during the Phase 4 refactor:

- Canonical-entity deduplication lives in
  ``entity_filtering.deduplication.canonical_entities`` (was
  ``semantic_layer.deduplication``).
- RDF/OWL/SHACL ontology export lives in
  ``ontology_manager.rdf_owl_shacl`` (was
  ``semantic_layer.ontology_manager``).

Both of those import :mod:`semantic_intelligence.config` for SurrealDB
plumbing.

Each module is independently runnable as a standalone script:

    python -m semantic_intelligence.graph_algorithms
    python -m semantic_intelligence.reasoning
    python -m semantic_intelligence.decision_tracker
    python -m ontology_manager.rdf_owl_shacl
    python -m entity_filtering.deduplication.canonical_entities

End-to-end test pipeline:

    python -m semantic_intelligence.scripts.test_pipeline /data/output/<doc_dir>

Note: ``config.py`` currently duplicates SurrealDB connection logic from
``surrealdb_service.connection``. Unifying these is a known follow-up.
"""
