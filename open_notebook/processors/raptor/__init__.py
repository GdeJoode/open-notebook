"""
RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

This module implements hierarchical document summarization for improved RAG retrieval.
Based on the paper: https://arxiv.org/abs/2401.18059

Usage:
    from open_notebook.processors.raptor import RaptorProcessor, RaptorConfig

    config = RaptorConfig(max_layers=5)
    processor = RaptorProcessor(config)
    tree = await processor.process_source(source_id)
"""

from .config import RaptorConfig
from .processor import RaptorProcessor, process_source_with_raptor
from .tree_builder import RaptorTree, RaptorNode

__all__ = [
    "RaptorConfig",
    "RaptorProcessor",
    "RaptorTree",
    "RaptorNode",
    "process_source_with_raptor",
]
