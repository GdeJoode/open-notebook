"""
RAPTOR Processor

Main entry point for RAPTOR processing, integrating with open-notebook's
source pipeline and database.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from open_notebook.domain.notebook import Chunk, Source
from open_notebook.database.repository import repo_query, ensure_record_id

from .config import RaptorConfig
from .tree_builder import RaptorTreeBuilder, RaptorTree


class RaptorProcessor:
    """
    Processes source documents to create RAPTOR hierarchical summaries.

    Integrates with existing chunk storage. Summary nodes are stored as
    Chunk records with is_raptor_node=True and layer > 0.
    """

    def __init__(self, config: Optional[RaptorConfig] = None):
        self.config = config or RaptorConfig()
        self.tree_builder = RaptorTreeBuilder(config)

    async def process_source(
        self,
        source_id: str,
        existing_chunks: Optional[List[Dict[str, Any]]] = None,
        force_rebuild: bool = False,
    ) -> RaptorTree:
        """
        Build RAPTOR tree for a source and save summary chunks.

        Args:
            source_id: Source document ID
            existing_chunks: Optional pre-loaded chunks (with embeddings)
            force_rebuild: If True, delete existing RAPTOR nodes first

        Returns:
            RaptorTree with references to saved chunks
        """
        # Check if RAPTOR nodes already exist
        if not force_rebuild:
            existing_count = await self._count_existing_raptor_nodes(source_id)
            if existing_count > 0:
                logger.info(
                    f"Source {source_id} already has {existing_count} RAPTOR nodes. "
                    "Use force_rebuild=True to regenerate."
                )
                return await self._load_existing_tree(source_id)

        # Load chunks if not provided
        if existing_chunks is None:
            chunks = await self._load_source_chunks(source_id)
        else:
            chunks = existing_chunks

        if not chunks:
            logger.warning(f"No chunks found for source {source_id}")
            return RaptorTree(source_id=source_id)

        # Build RAPTOR tree
        tree = await self.tree_builder.build_tree(chunks)
        tree.source_id = source_id

        if not tree.nodes:
            logger.info(f"No RAPTOR nodes created for source {source_id}")
            return tree

        # Delete existing RAPTOR nodes if rebuilding
        if force_rebuild:
            await self._delete_existing_raptor_nodes(source_id)

        # Save summary nodes as chunks
        await self._save_tree_as_chunks(source_id, tree)

        return tree

    async def _count_existing_raptor_nodes(self, source_id: str) -> int:
        """Count existing RAPTOR nodes for a source."""
        result = await repo_query(
            """
            SELECT count() as cnt FROM chunk
            WHERE source = $source_id AND is_raptor_node = true
            GROUP ALL
            """,
            {"source_id": ensure_record_id(source_id)}
        )
        if result and len(result) > 0:
            return result[0].get("cnt", 0)
        return 0

    async def _delete_existing_raptor_nodes(self, source_id: str) -> int:
        """Delete existing RAPTOR nodes for a source."""
        result = await repo_query(
            "DELETE chunk WHERE source = $source_id AND is_raptor_node = true",
            {"source_id": ensure_record_id(source_id)}
        )
        deleted = len(result) if result else 0
        if deleted > 0:
            logger.info(f"Deleted {deleted} existing RAPTOR nodes for source {source_id}")
        return deleted

    async def _load_source_chunks(self, source_id: str) -> List[Dict[str, Any]]:
        """Load existing chunks for a source (layer 0 only)."""
        result = await repo_query(
            """
            SELECT * FROM chunk
            WHERE source = $source_id AND (layer = 0 OR layer IS NONE) AND (is_raptor_node = false OR is_raptor_node IS NONE)
            ORDER BY order
            """,
            {"source_id": ensure_record_id(source_id)}
        )

        if not result:
            return []

        return [
            {
                "id": str(r.get("id", "")),
                "text": r.get("text", ""),
                "order": r.get("order", 0),
                "embedding": r.get("embedding"),  # May be None
            }
            for r in result
        ]

    async def _load_existing_tree(self, source_id: str) -> RaptorTree:
        """Load existing RAPTOR tree from database."""
        result = await repo_query(
            """
            SELECT * FROM chunk
            WHERE source = $source_id AND is_raptor_node = true
            ORDER BY layer, order
            """,
            {"source_id": ensure_record_id(source_id)}
        )

        tree = RaptorTree(source_id=source_id)

        if not result:
            return tree

        from .tree_builder import RaptorNode

        max_layer = 0
        for r in result:
            layer = r.get("layer", 1)
            max_layer = max(max_layer, layer)

            node = RaptorNode(
                text=r.get("text", ""),
                layer=layer,
                embedding=r.get("embedding", []),
                children_ids=r.get("parent_chunk_ids", []),
                chunk_id=str(r.get("id", "")),
                metadata=r.get("metadata", {}),
            )
            tree.nodes.append(node)

        tree.num_layers = max_layer
        return tree

    async def _save_tree_as_chunks(
        self,
        source_id: str,
        tree: RaptorTree
    ) -> List[str]:
        """
        Save RAPTOR summary nodes as chunks in database.

        Returns:
            List of created chunk IDs
        """
        created_ids = []

        # Get max order from existing chunks to avoid collision
        max_order_result = await repo_query(
            """
            SELECT math::max(order) as max_order FROM chunk
            WHERE source = $source_id
            GROUP ALL
            """,
            {"source_id": ensure_record_id(source_id)}
        )

        if max_order_result and len(max_order_result) > 0:
            base_order = (max_order_result[0].get("max_order") or 0) + 1000
        else:
            base_order = 1000

        # Save each summary node
        for i, node in enumerate(tree.nodes):
            chunk = Chunk(
                source=source_id,
                text=node.text,
                chunk_order=base_order + (node.layer * 100) + i,
                physical_page=0,  # Summary nodes don't have page info
                element_type="raptor_summary",
                layer=node.layer,
                parent_chunk_ids=node.children_ids,
                is_raptor_node=True,
                positions=[],  # No spatial data for summaries
                metadata={
                    "raptor_layer": node.layer,
                    "children_count": len(node.children_ids),
                    **(node.metadata or {}),
                }
            )
            await chunk.save()
            node.chunk_id = chunk.id
            created_ids.append(chunk.id)

        logger.info(
            f"Saved {len(created_ids)} RAPTOR chunks for source {source_id} "
            f"(layers: {tree.num_layers})"
        )
        return created_ids


async def process_source_with_raptor(
    source_id: str,
    config: Optional[RaptorConfig] = None,
    force_rebuild: bool = False,
) -> RaptorTree:
    """
    Convenience function to process a source with RAPTOR.

    Args:
        source_id: Source to process
        config: Optional RAPTOR configuration
        force_rebuild: If True, regenerate even if RAPTOR nodes exist

    Returns:
        RaptorTree structure
    """
    processor = RaptorProcessor(config)
    return await processor.process_source(source_id, force_rebuild=force_rebuild)


async def delete_raptor_nodes(source_id: str) -> int:
    """
    Delete all RAPTOR nodes for a source.

    Args:
        source_id: Source to clean

    Returns:
        Number of deleted nodes
    """
    processor = RaptorProcessor()
    return await processor._delete_existing_raptor_nodes(source_id)
