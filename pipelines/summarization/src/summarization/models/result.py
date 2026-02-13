"""
Unified output models for the summarization pipeline.

All three strategies (RAPTOR, TreeKG, Naive) produce SummarizationResult
objects containing SummaryNode trees with consistent metadata.
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SummarizationStrategy(str, Enum):
    """Available summarization strategies."""

    # Implemented
    RAPTOR = "raptor"
    TREEKG = "treekg"
    NAIVE = "naive"
    # Standalone strategies (stubs)
    MAP_REDUCE = "map_reduce"
    REFINE = "refine"
    WALKING_TREE = "walking_tree"
    EXTRACTIVE_ABSTRACTIVE = "extractive_abstractive"
    SKELETON = "skeleton_of_thought"
    # Cross-cutting (stubs)
    HYBRID = "hybrid"
    DPR_ABS = "dpr_abs"
    LINKED_ENTITY = "linked_entity"
    # Enhancement layers (stubs — can also be selected as primary)
    CHAIN_OF_DENSITY = "chain_of_density"
    SELF_CORRECTION = "self_correction"
    GIST_TOKENS = "gist_tokens"


class ChunkInput(BaseModel):
    """Input chunk for summarization strategies.

    Carries the text plus optional embeddings and structural metadata
    that specific strategies may use.
    """

    text: str
    chunk_id: Optional[str] = None
    order: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    # TreeKG hierarchy fields
    section_path: Optional[List[str]] = None
    section_id: Optional[str] = None
    section_level: Optional[int] = None


class SummaryNode(BaseModel):
    """A single node in the summary tree.

    Represents either a leaf summary (from a single cluster or section)
    or an intermediate/root summary produced by hierarchical aggregation.
    """

    text: str
    layer: int = 0
    source_chunk_ids: List[str] = Field(default_factory=list)
    children_ids: List[str] = Field(default_factory=list)
    node_id: Optional[str] = None
    section_title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SummarizationResult(BaseModel):
    """Unified result from any summarization strategy.

    Contains the overall document summary, individual summary nodes,
    and metadata about the summarization process.
    """

    strategy: SummarizationStrategy
    document_summary: Optional[str] = None
    summary_nodes: List[SummaryNode] = Field(default_factory=list)
    num_layers: int = 0
    num_input_chunks: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def node_count(self) -> int:
        """Total number of summary nodes."""
        return len(self.summary_nodes)

    @property
    def has_hierarchy(self) -> bool:
        """Whether the result contains multi-layer summaries."""
        return self.num_layers > 1
