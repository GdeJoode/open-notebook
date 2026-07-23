"""Reference extraction (Track V.1–V.3) + external work-resolution (Track V.4).

Track V turns a source's persisted document structure into ``ParsedReference``
units (the Track V → U.3 boundary, defined in
:mod:`shared.retrieval.cites_matching`). This subpackage holds both halves:

* **Producer (V.1–V.3)** — :func:`extract_references` chains region location
  (:func:`locate_reference_region`), segmentation (:func:`segment_region`) and
  per-entry parsing (:func:`parse_reference`) to yield a source's
  ``List[ParsedReference]`` from its chunks. Pure/offline — no DB, no LLM.
* **Resolver (V.4)** — U.3 matches those references against the IN-corpus
  sources; this half resolves the EXTERNAL ones — mapping a ``ParsedReference``
  to a canonical :class:`ResolvedWork` in an external authority.

The public surface is the :class:`ResolverCascade`: it routes a reference by its
*shape* (DOI, arXiv id, Dutch Kamerstuk, econ working paper, or bare title)
across the provider resolvers and stops at the first hit that clears the shared
precision guard (unresolved is always preferred over a wrong match). Every
provider ends in ``Resolver`` (repo ``*Resolver`` convention) and is DB-free and
unit-testable without a live network.

This is pure resolution — no DB, no ``cites`` materialization; a later phase feeds
a :class:`ResolvedWork` into the graph.
"""

from shared.references.arxiv_resolver import ArxivResolver
from shared.references.crossref_resolver import CrossrefResolver
from shared.references.datacite_resolver import DataCiteResolver
from shared.references.enrichment import ReferenceEnricher
from shared.references.openalex_resolver import OpenAlexResolver
from shared.references.overheid_resolver import OverheidResolver
from shared.references.reference_extractor import extract_references
from shared.references.reference_parser import parse_reference
from shared.references.region_locator import (
    LocatedRegion,
    ReferenceChunk,
    locate_reference_region,
)
from shared.references.repec_resolver import RePEcResolver
from shared.references.segmenter import AmbiguityResolver, segment_region
from shared.references.work_resolver import (
    DOI_MATCH_CONFIDENCE,
    MIN_MATCH_CONFIDENCE,
    MIN_TITLE_OVERLAP,
    ResolvedWork,
    ResolverCascade,
    WorkEnricher,
    WorkResolver,
    extract_arxiv_id,
    looks_like_econ_wp,
    looks_like_nl_policy,
    passes_title_author_guard,
    title_author_confidence,
    title_overlap,
)

__all__ = [
    # Producer (V.1–V.3)
    "ReferenceChunk",
    "LocatedRegion",
    "locate_reference_region",
    "segment_region",
    "AmbiguityResolver",
    "parse_reference",
    "extract_references",
    # Core
    "ResolvedWork",
    "WorkResolver",
    "WorkEnricher",
    "ResolverCascade",
    # Providers (active)
    "OpenAlexResolver",
    "CrossrefResolver",
    "DataCiteResolver",
    "ArxivResolver",
    "OverheidResolver",
    # Provider (gated)
    "RePEcResolver",
    # Enrichment (opt-in)
    "ReferenceEnricher",
    # Helpers / shape detection
    "title_overlap",
    "title_author_confidence",
    "passes_title_author_guard",
    "extract_arxiv_id",
    "looks_like_nl_policy",
    "looks_like_econ_wp",
    # Thresholds
    "DOI_MATCH_CONFIDENCE",
    "MIN_TITLE_OVERLAP",
    "MIN_MATCH_CONFIDENCE",
]
