"""
Graph-based entity analysis.

Builds a directed graph from entities and relations using NetworkX,
computes centrality scores (PageRank, betweenness), and optionally
filters entities with low centrality.
"""

from typing import Any, Dict, List, Tuple

import networkx as nx
from loguru import logger


class GraphAnalyzer:
    """Analyze and filter entities using graph centrality metrics.

    Builds an nx.DiGraph from entity relations, computes centrality
    scores, and enriches entities with those scores. Optionally removes
    low-centrality entities. When outlier classification is enabled,
    entities are classified by combining extraction centrality with
    KG resolution status (``is_new`` from KG resolver).

    Args:
        min_score: Minimum centrality score to retain an entity.
            Set to 0.0 to keep all entities (enrichment only).
        algorithm: Centrality algorithm -- "pagerank" or "betweenness".
        classify_outliers: Classify entities by extraction centrality
            and KG match status into categories (important_new,
            potential_noise, anomaly, established).
        outlier_centrality_low: Centrality score below which an entity
            is considered "low" for outlier classification.
    """

    def __init__(
        self,
        min_score: float = 0.01,
        algorithm: str = "pagerank",
        classify_outliers: bool = False,
        outlier_centrality_low: float = 0.05,
    ) -> None:
        self._min_score = min_score
        self._algorithm = algorithm
        self._classify_outliers = classify_outliers
        self._outlier_centrality_low = outlier_centrality_low

    def analyze(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Compute centrality and optionally filter low-scoring entities.

        Args:
            entities: Entity dicts with "text" and "label" keys.
            relations: Relation dicts with "source_entity", "target_entity",
                "relation_type" keys.

        Returns:
            Tuple of (filtered_entities, analysis_report).

            Each entity gets a "centrality_score" added to properties.

            analysis_report contains:
            - algorithm: str
            - node_count: int
            - edge_count: int
            - removed_count: int
            - min_score_threshold: float
            - centrality_scores: dict[str, float] (entity_text -> score)
        """
        report: Dict[str, Any] = {
            "algorithm": self._algorithm,
            "node_count": 0,
            "edge_count": 0,
            "removed_count": 0,
            "min_score_threshold": self._min_score,
            "centrality_scores": {},
        }

        if not entities:
            return entities, report

        # Build the graph
        graph = self._build_graph(entities, relations)
        report["node_count"] = graph.number_of_nodes()
        report["edge_count"] = graph.number_of_edges()

        # Compute centrality
        scores = self._compute_centrality(graph)
        report["centrality_scores"] = scores

        # Enrich entities with scores and optionally filter
        result = []
        entity_texts_in_graph = set(scores.keys())

        for entity in entities:
            text = entity.get("text", "")
            score = scores.get(text, 0.0)
            props = entity.setdefault("properties", {})
            props["centrality_score"] = round(score, 6)

            if self._min_score > 0.0 and score < self._min_score:
                # Only filter entities that ARE in the graph but have low score
                # Entities not in the graph (isolated) get score 0, keep them
                # unless they were actually added as nodes
                if text in entity_texts_in_graph:
                    report["removed_count"] += 1
                    continue

            result.append(entity)

        if self._classify_outliers:
            outlier_summary = self._classify_entities(result, scores)
            report["outlier_summary"] = outlier_summary

        logger.debug(
            "GraphAnalyzer ({}): {} nodes, {} edges, {} removed below {:.4f}",
            self._algorithm,
            report["node_count"],
            report["edge_count"],
            report["removed_count"],
            self._min_score,
        )

        return result, report

    def _build_graph(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> nx.DiGraph:
        """Build a directed graph from entities and relations."""
        graph = nx.DiGraph()

        # Add entity nodes
        for entity in entities:
            text = entity.get("text", "")
            if text:
                graph.add_node(text, label=entity.get("label", "MISC"))

        # Add relation edges
        for rel in relations:
            source = rel.get("source_entity", "")
            target = rel.get("target_entity", "")
            rel_type = rel.get("relation_type", "")
            if source and target:
                graph.add_edge(source, target, relation_type=rel_type)

        return graph

    def _compute_centrality(self, graph: nx.DiGraph) -> Dict[str, float]:
        """Compute centrality scores using the configured algorithm."""
        if graph.number_of_nodes() == 0:
            return {}

        if self._algorithm == "betweenness":
            return nx.betweenness_centrality(graph)

        # Default: pagerank
        # pagerank needs at least one node; handles disconnected graphs
        try:
            return nx.pagerank(graph)
        except nx.PowerIterationFailedConvergence:
            logger.warning("PageRank did not converge, falling back to betweenness")
            return nx.betweenness_centrality(graph)

    def _classify_entities(
        self,
        entities: List[Dict[str, Any]],
        scores: Dict[str, float],
    ) -> Dict[str, Any]:
        """Classify entities by combining extraction centrality with KG status.

        Categories:
        - ``important_new``: is_new=True, centrality >= threshold
        - ``potential_noise``: is_new=True, centrality < threshold
        - ``anomaly``: is_new=False, centrality < threshold
        - ``established``: is_new=False, centrality >= threshold
        - (unclassified): no ``is_new`` property (KG resolution didn't run)

        Sets ``properties["kg_classification"]`` on each classified entity.

        Returns:
            Summary dict with counts and entity text lists per category.
        """
        summary: Dict[str, Any] = {
            "important_new_count": 0,
            "potential_noise_count": 0,
            "anomaly_count": 0,
            "established_count": 0,
            "unclassified_count": 0,
            "important_new_entities": [],
            "potential_noise_entities": [],
            "anomaly_entities": [],
        }

        for entity in entities:
            props = entity.get("properties", {})
            is_new = props.get("is_new")

            # Skip entities without KG resolution data
            if is_new is None:
                summary["unclassified_count"] += 1
                continue

            text = entity.get("text", "")
            centrality = scores.get(text, 0.0)
            high_centrality = centrality >= self._outlier_centrality_low

            if is_new and high_centrality:
                classification = "important_new"
                summary["important_new_count"] += 1
                summary["important_new_entities"].append(text)
            elif is_new and not high_centrality:
                classification = "potential_noise"
                summary["potential_noise_count"] += 1
                summary["potential_noise_entities"].append(text)
            elif not is_new and not high_centrality:
                classification = "anomaly"
                summary["anomaly_count"] += 1
                summary["anomaly_entities"].append(text)
            else:
                classification = "established"
                summary["established_count"] += 1

            entity.setdefault("properties", {})[
                "kg_classification"
            ] = classification

        return summary


class MergedGraphAnalyzer:
    """Stub for future merged extraction+KG graph analysis.

    Planned enhancement: builds a combined graph from the extraction
    subgraph + a KG neighbourhood (fetched via entity_repo), computes
    centrality on the merged graph, giving contextual importance scores
    that account for the broader KG structure.
    """

    def __init__(
        self,
        entity_repo: Any = None,
        algorithm: str = "pagerank",
        kg_neighborhood_hops: int = 1,
        kg_max_neighbors: int = 50,
    ) -> None:
        self._entity_repo = entity_repo
        self._algorithm = algorithm
        self._kg_neighborhood_hops = kg_neighborhood_hops
        self._kg_max_neighbors = kg_max_neighbors

    async def analyze(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Analyze entities using a merged extraction+KG graph.

        Raises:
            NotImplementedError: Always. This is a planned enhancement.
        """
        raise NotImplementedError(
            "MergedGraphAnalyzer is a planned enhancement. "
            "Use GraphAnalyzer for extraction-only analysis."
        )
