"""
Graph Traversal Algorithms for Graph RAG enhancement.

This module provides sophisticated algorithms for hierarchical traversal and
component discovery within code structure graphs. Supports multiple traversal
strategies and relationship-aware navigation.
"""

import asyncio
import logging
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .structure_relationship_builder import GraphEdge, GraphNode, StructureGraph


class TraversalStrategy(Enum):
    """Enumeration of graph traversal strategies."""

    DEPTH_FIRST = "depth_first"
    BREADTH_FIRST = "breadth_first"
    BEST_FIRST = "best_first"  # Using node weights/importance
    RELATIONSHIP_WEIGHTED = "relationship_weighted"  # Prioritize certain relationship types
    SEMANTIC_SIMILARITY = "semantic_similarity"  # Use semantic weights


class RelationshipFilter(Enum):
    """Enumeration of relationship filters for traversal."""

    ALL = "all"
    HIERARCHICAL_ONLY = "hierarchical_only"  # parent_child only
    DEPENDENCIES_ONLY = "dependencies_only"  # dependency relationships
    IMPLEMENTATIONS_ONLY = "implementations_only"  # interface implementations
    SIBLINGS_ONLY = "siblings_only"  # sibling relationships
    FUNCTION_CALLS_ONLY = "function_calls_only"  # function call relationships
    NO_FUNCTION_CALLS = "no_function_calls"  # exclude function call relationships
    CUSTOM = "custom"  # Custom filter function


@dataclass
class TraversalOptions:
    """Configuration options for graph traversal."""

    strategy: TraversalStrategy = TraversalStrategy.DEPTH_FIRST
    relationship_filter: RelationshipFilter = RelationshipFilter.ALL
    max_depth: int = 5
    max_nodes: int = 100
    include_reverse_relationships: bool = False
    weight_threshold: float = 0.0
    confidence_threshold: float = 0.5
    custom_filter: Callable[[GraphEdge], bool] | None = None
    custom_scorer: Callable[[GraphNode], float] | None = None


@dataclass
class ComponentCluster:
    """Represents a cluster of related components."""

    central_node: GraphNode
    related_nodes: list[GraphNode]
    cluster_score: float
    relationship_types: set[str]
    max_distance: int
    total_nodes: int


@dataclass
class TraversalPath:
    """Represents a path through the graph."""

    path_nodes: list[GraphNode]
    path_breadcrumbs: list[str]
    path_edges: list[GraphEdge]
    total_weight: float
    path_length: int
    relationship_diversity: float


class GraphTraversalAlgorithms:
    """
    Advanced algorithms for traversing and analyzing code structure graphs.

    Provides sophisticated navigation capabilities including weighted traversal,
    relationship-aware pathfinding, and component clustering algorithms.
    """

    def __init__(self):
        """Initialize the traversal algorithms service."""
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self._traversal_stats = {
            "total_traversals": 0,
            "avg_nodes_visited": 0.0,
            "avg_execution_time_ms": 0.0,
            "strategy_usage": defaultdict(int),
        }

    async def advanced_traversal(
        self, graph: StructureGraph, start_breadcrumb: str, options: TraversalOptions
    ) -> tuple[list[GraphNode], list[str], dict[str, Any]]:
        """
        Perform advanced graph traversal with configurable strategy and filters.

        Args:
            graph: Structure graph to traverse
            start_breadcrumb: Starting breadcrumb for traversal
            options: Traversal configuration options

        Returns:
            Tuple of (visited_nodes, breadcrumb_path, metadata)
        """
        import time

        start_time = time.time()

        try:
            # Validate starting node
            if start_breadcrumb not in graph.nodes:
                return [], [], {"error": "start_breadcrumb_not_found"}

            # Select traversal strategy
            if options.strategy == TraversalStrategy.DEPTH_FIRST:
                visited_nodes, path = await self._depth_first_traversal(graph, start_breadcrumb, options)
            elif options.strategy == TraversalStrategy.BREADTH_FIRST:
                visited_nodes, path = await self._breadth_first_traversal(graph, start_breadcrumb, options)
            elif options.strategy == TraversalStrategy.BEST_FIRST:
                visited_nodes, path = await self._best_first_traversal(graph, start_breadcrumb, options)
            elif options.strategy == TraversalStrategy.RELATIONSHIP_WEIGHTED:
                visited_nodes, path = await self._relationship_weighted_traversal(graph, start_breadcrumb, options)
            elif options.strategy == TraversalStrategy.SEMANTIC_SIMILARITY:
                visited_nodes, path = await self._semantic_similarity_traversal(graph, start_breadcrumb, options)
            else:
                raise ValueError(f"Unknown traversal strategy: {options.strategy}")

            # Update statistics
            execution_time = (time.time() - start_time) * 1000
            self._update_traversal_stats(options.strategy, len(visited_nodes), execution_time)

            metadata = {
                "strategy_used": options.strategy.value,
                "nodes_visited": len(visited_nodes),
                "path_length": len(path),
                "execution_time_ms": execution_time,
                "max_depth_reached": len(path) >= options.max_depth,
                "max_nodes_reached": len(visited_nodes) >= options.max_nodes,
            }

            return visited_nodes, path, metadata

        except Exception as e:
            self.logger.error(f"Error in advanced traversal: {e}")
            raise

    async def find_component_clusters(
        self, graph: StructureGraph, cluster_threshold: float = 0.7, min_cluster_size: int = 3, max_clusters: int = 10
    ) -> list[ComponentCluster]:
        """
        Identify clusters of highly related components.

        Args:
            graph: Structure graph to analyze
            cluster_threshold: Minimum relationship strength for clustering
            min_cluster_size: Minimum number of nodes in a cluster
            max_clusters: Maximum number of clusters to return

        Returns:
            List of identified component clusters
        """
        try:
            clusters = []
            visited_nodes = set()

            # Sort nodes by semantic weight (highest first)
            sorted_nodes = sorted(graph.nodes.values(), key=lambda n: n.semantic_weight, reverse=True)

            for central_node in sorted_nodes:
                if central_node.breadcrumb in visited_nodes or len(clusters) >= max_clusters:
                    continue

                # Find strongly connected components around this node
                cluster_nodes = await self._find_strong_cluster(graph, central_node.breadcrumb, cluster_threshold)

                if len(cluster_nodes) >= min_cluster_size:
                    # Calculate cluster properties
                    cluster_score = self._calculate_cluster_score(cluster_nodes)
                    relationship_types = self._get_cluster_relationship_types(graph, cluster_nodes)
                    max_distance = self._calculate_max_distance(cluster_nodes)

                    cluster = ComponentCluster(
                        central_node=central_node,
                        related_nodes=cluster_nodes,
                        cluster_score=cluster_score,
                        relationship_types=relationship_types,
                        max_distance=max_distance,
                        total_nodes=len(cluster_nodes),
                    )
                    clusters.append(cluster)

                    # Mark nodes as visited
                    for node in cluster_nodes:
                        visited_nodes.add(node.breadcrumb)

            # Sort clusters by score
            clusters.sort(key=lambda c: c.cluster_score, reverse=True)

            self.logger.debug(f"Found {len(clusters)} component clusters")
            return clusters

        except Exception as e:
            self.logger.error(f"Error finding component clusters: {e}")
            return []

    async def find_optimal_paths(
        self, graph: StructureGraph, start_breadcrumb: str, end_breadcrumb: str, max_paths: int = 5
    ) -> list[TraversalPath]:
        """
        Find multiple optimal paths between two components.

        Args:
            graph: Structure graph to search
            start_breadcrumb: Starting component
            end_breadcrumb: Target component
            max_paths: Maximum number of paths to return

        Returns:
            List of optimal paths sorted by total weight
        """
        try:
            if start_breadcrumb not in graph.nodes or end_breadcrumb not in graph.nodes:
                return []

            # Use modified Dijkstra's algorithm to find multiple paths
            paths = await self._find_k_shortest_paths(graph, start_breadcrumb, end_breadcrumb, max_paths)

            # Convert to TraversalPath objects
            traversal_paths = []
            for path_breadcrumbs, total_weight in paths:
                path_nodes = [graph.nodes[b] for b in path_breadcrumbs if b in graph.nodes]
                path_edges = self._get_path_edges(graph, path_breadcrumbs)
                relationship_diversity = self._calculate_relationship_diversity(path_edges)

                traversal_path = TraversalPath(
                    path_nodes=path_nodes,
                    path_breadcrumbs=path_breadcrumbs,
                    path_edges=path_edges,
                    total_weight=total_weight,
                    path_length=len(path_breadcrumbs),
                    relationship_diversity=relationship_diversity,
                )
                traversal_paths.append(traversal_path)

            return traversal_paths

        except Exception as e:
            self.logger.error(f"Error finding optimal paths: {e}")
            return []

    async def analyze_connectivity(self, graph: StructureGraph, target_breadcrumb: str) -> dict[str, Any]:
        """
        Analyze the connectivity patterns of a specific component.

        Args:
            graph: Structure graph to analyze
            target_breadcrumb: Component to analyze

        Returns:
            Dictionary with connectivity analysis results
        """
        try:
            if target_breadcrumb not in graph.nodes:
                return {"error": "breadcrumb_not_found"}

            target_node = graph.nodes[target_breadcrumb]

            # Calculate various connectivity metrics
            incoming_edges = [e for e in graph.edges if e.target_breadcrumb == target_breadcrumb]
            outgoing_edges = [e for e in graph.edges if e.source_breadcrumb == target_breadcrumb]

            # Group by relationship type
            incoming_by_type = defaultdict(list)
            outgoing_by_type = defaultdict(list)

            for edge in incoming_edges:
                incoming_by_type[edge.relationship_type].append(edge)

            for edge in outgoing_edges:
                outgoing_by_type[edge.relationship_type].append(edge)

            # Calculate centrality metrics
            degree_centrality = len(incoming_edges) + len(outgoing_edges)
            weighted_centrality = sum(e.weight for e in incoming_edges + outgoing_edges)

            # Find components at different distances
            reachable_at_distance = await self._find_reachable_at_distance(graph, target_breadcrumb, max_distance=3)

            analysis = {
                "node_info": {
                    "breadcrumb": target_breadcrumb,
                    "name": target_node.name,
                    "type": target_node.chunk_type.value,
                    "depth": target_node.depth,
                    "semantic_weight": target_node.semantic_weight,
                },
                "connectivity_metrics": {
                    "degree_centrality": degree_centrality,
                    "weighted_centrality": weighted_centrality,
                    "incoming_connections": len(incoming_edges),
                    "outgoing_connections": len(outgoing_edges),
                },
                "relationship_breakdown": {
                    "incoming": {rt: len(edges) for rt, edges in incoming_by_type.items()},
                    "outgoing": {rt: len(edges) for rt, edges in outgoing_by_type.items()},
                },
                "reachability": reachable_at_distance,
                "influence_score": self._calculate_influence_score(target_node, incoming_edges, outgoing_edges),
            }

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing connectivity: {e}")
            return {"error": str(e)}

    # =================== Private Traversal Methods ===================

    async def _depth_first_traversal(
        self, graph: StructureGraph, start_breadcrumb: str, options: TraversalOptions
    ) -> tuple[list[GraphNode], list[str]]:
        """Perform depth-first traversal."""
        visited = set()
        visited_nodes = []
        path = []

        async def dfs(breadcrumb: str, depth: int):
            if depth > options.max_depth or breadcrumb in visited or len(visited_nodes) >= options.max_nodes:
                return

            visited.add(breadcrumb)
            if breadcrumb in graph.nodes:
                visited_nodes.append(graph.nodes[breadcrumb])
                path.append(breadcrumb)

            # Get filtered edges
            filtered_edges = self._filter_edges(graph.edges, breadcrumb, options, outgoing=True)

            # Sort edges by weight (descending)
            filtered_edges.sort(key=lambda e: e.weight, reverse=True)

            for edge in filtered_edges:
                if edge.confidence >= options.confidence_threshold:
                    await dfs(edge.target_breadcrumb, depth + 1)

        await dfs(start_breadcrumb, 0)
        return visited_nodes, path

    async def _breadth_first_traversal(
        self, graph: StructureGraph, start_breadcrumb: str, options: TraversalOptions
    ) -> tuple[list[GraphNode], list[str]]:
        """Perform breadth-first traversal."""
        visited = set()
        visited_nodes = []
        path = []
        queue = deque([(start_breadcrumb, 0)])

        while queue and len(visited_nodes) < options.max_nodes:
            breadcrumb, depth = queue.popleft()

            if breadcrumb in visited or depth > options.max_depth:
                continue

            visited.add(breadcrumb)
            if breadcrumb in graph.nodes:
                visited_nodes.append(graph.nodes[breadcrumb])
                path.append(breadcrumb)

            # Get filtered edges
            filtered_edges = self._filter_edges(graph.edges, breadcrumb, options, outgoing=True)

            for edge in filtered_edges:
                if edge.confidence >= options.confidence_threshold and edge.target_breadcrumb not in visited:
                    queue.append((edge.target_breadcrumb, depth + 1))

        return visited_nodes, path

    async def _best_first_traversal(
        self, graph: StructureGraph, start_breadcrumb: str, options: TraversalOptions
    ) -> tuple[list[GraphNode], list[str]]:
        """Perform best-first traversal using node importance scores."""
        import heapq

        visited = set()
        visited_nodes = []
        path = []

        # Priority queue: (negative_score, breadcrumb, depth)
        priority_queue = [(-self._get_node_score(graph.nodes[start_breadcrumb], options), start_breadcrumb, 0)]

        while priority_queue and len(visited_nodes) < options.max_nodes:
            neg_score, breadcrumb, depth = heapq.heappop(priority_queue)

            if breadcrumb in visited or depth > options.max_depth:
                continue

            visited.add(breadcrumb)
            if breadcrumb in graph.nodes:
                visited_nodes.append(graph.nodes[breadcrumb])
                path.append(breadcrumb)

            # Get filtered edges
            filtered_edges = self._filter_edges(graph.edges, breadcrumb, options, outgoing=True)

            for edge in filtered_edges:
                if (
                    edge.confidence >= options.confidence_threshold
                    and edge.target_breadcrumb not in visited
                    and edge.target_breadcrumb in graph.nodes
                ):
                    target_node = graph.nodes[edge.target_breadcrumb]
                    score = self._get_node_score(target_node, options)
                    heapq.heappush(priority_queue, (-score, edge.target_breadcrumb, depth + 1))

        return visited_nodes, path

    async def _relationship_weighted_traversal(
        self, graph: StructureGraph, start_breadcrumb: str, options: TraversalOptions
    ) -> tuple[list[GraphNode], list[str]]:
        """Perform traversal weighted by relationship types."""
        # Define relationship weights
        relationship_weights = {
            "parent_child": 1.0,
            "dependency": 0.8,
            "implementation": 0.9,
            "sibling": 0.6,
            "function_call": 0.7,  # Function calls are important but not as critical as hierarchical relationships
        }

        visited = set()
        visited_nodes = []
        path = []

        async def weighted_dfs(breadcrumb: str, depth: int):
            if depth > options.max_depth or breadcrumb in visited or len(visited_nodes) >= options.max_nodes:
                return

            visited.add(breadcrumb)
            if breadcrumb in graph.nodes:
                visited_nodes.append(graph.nodes[breadcrumb])
                path.append(breadcrumb)

            # Get filtered edges and apply relationship weights
            filtered_edges = self._filter_edges(graph.edges, breadcrumb, options, outgoing=True)

            # Calculate weighted scores for edges
            weighted_edges = []
            for edge in filtered_edges:
                rel_weight = relationship_weights.get(edge.relationship_type, 0.5)
                total_weight = edge.weight * rel_weight * edge.confidence
                weighted_edges.append((total_weight, edge))

            # Sort by weighted score (descending)
            weighted_edges.sort(key=lambda x: x[0], reverse=True)

            for weight, edge in weighted_edges:
                if weight >= options.weight_threshold:
                    await weighted_dfs(edge.target_breadcrumb, depth + 1)

        await weighted_dfs(start_breadcrumb, 0)
        return visited_nodes, path

    async def _semantic_similarity_traversal(
        self, graph: StructureGraph, start_breadcrumb: str, options: TraversalOptions
    ) -> tuple[list[GraphNode], list[str]]:
        """Perform traversal based on semantic similarity weights."""
        # This is a placeholder for semantic similarity-based traversal
        # In a real implementation, this would use embedding similarity
        return await self._best_first_traversal(graph, start_breadcrumb, options)

    # =================== Helper Methods ===================

    def _filter_edges(
        self, edges: list[GraphEdge], source_breadcrumb: str, options: TraversalOptions, outgoing: bool = True
    ) -> list[GraphEdge]:
        """Filter edges based on traversal options."""
        # Get edges from/to the source node
        if outgoing:
            filtered_edges = [e for e in edges if e.source_breadcrumb == source_breadcrumb]
        else:
            filtered_edges = [e for e in edges if e.target_breadcrumb == source_breadcrumb]

        # Apply relationship filter
        if options.relationship_filter == RelationshipFilter.HIERARCHICAL_ONLY:
            filtered_edges = [e for e in filtered_edges if e.relationship_type == "parent_child"]
        elif options.relationship_filter == RelationshipFilter.DEPENDENCIES_ONLY:
            filtered_edges = [e for e in filtered_edges if e.relationship_type == "dependency"]
        elif options.relationship_filter == RelationshipFilter.IMPLEMENTATIONS_ONLY:
            filtered_edges = [e for e in filtered_edges if e.relationship_type == "implementation"]
        elif options.relationship_filter == RelationshipFilter.SIBLINGS_ONLY:
            filtered_edges = [e for e in filtered_edges if e.relationship_type == "sibling"]
        elif options.relationship_filter == RelationshipFilter.FUNCTION_CALLS_ONLY:
            filtered_edges = [e for e in filtered_edges if e.relationship_type == "function_call"]
        elif options.relationship_filter == RelationshipFilter.NO_FUNCTION_CALLS:
            filtered_edges = [e for e in filtered_edges if e.relationship_type != "function_call"]
        elif options.relationship_filter == RelationshipFilter.CUSTOM and options.custom_filter:
            filtered_edges = [e for e in filtered_edges if options.custom_filter(e)]

        return filtered_edges

    def _get_node_score(self, node: GraphNode, options: TraversalOptions) -> float:
        """Calculate importance score for a node."""
        if options.custom_scorer:
            return options.custom_scorer(node)

        # Default scoring based on semantic weight and chunk type
        base_score = node.semantic_weight

        # Boost important chunk types
        type_multipliers = {"function": 1.2, "class": 1.3, "interface": 1.1, "method": 1.0, "constant": 0.8}

        multiplier = type_multipliers.get(node.chunk_type.value, 1.0)
        return base_score * multiplier

    async def _find_strong_cluster(self, graph: StructureGraph, central_breadcrumb: str, threshold: float) -> list[GraphNode]:
        """Find strongly connected components around a central node."""
        cluster_nodes = []
        visited = set()

        # BFS to find strongly connected nodes
        queue = deque([(central_breadcrumb, 1.0)])  # (breadcrumb, connection_strength)

        while queue:
            breadcrumb, strength = queue.popleft()

            if breadcrumb in visited or strength < threshold:
                continue

            visited.add(breadcrumb)
            if breadcrumb in graph.nodes:
                cluster_nodes.append(graph.nodes[breadcrumb])

            # Find connected nodes
            for edge in graph.edges:
                if edge.source_breadcrumb == breadcrumb:
                    new_strength = strength * edge.weight * edge.confidence
                    if new_strength >= threshold:
                        queue.append((edge.target_breadcrumb, new_strength))

        return cluster_nodes

    def _calculate_cluster_score(self, cluster_nodes: list[GraphNode]) -> float:
        """Calculate overall score for a cluster."""
        if not cluster_nodes:
            return 0.0

        # Average semantic weight
        avg_weight = sum(node.semantic_weight for node in cluster_nodes) / len(cluster_nodes)

        # Size bonus (logarithmic)
        import math

        size_bonus = math.log(len(cluster_nodes) + 1)

        return avg_weight * size_bonus

    def _get_cluster_relationship_types(self, graph: StructureGraph, cluster_nodes: list[GraphNode]) -> set[str]:
        """Get unique relationship types within a cluster."""
        cluster_breadcrumbs = {node.breadcrumb for node in cluster_nodes}
        relationship_types = set()

        for edge in graph.edges:
            if edge.source_breadcrumb in cluster_breadcrumbs and edge.target_breadcrumb in cluster_breadcrumbs:
                relationship_types.add(edge.relationship_type)

        return relationship_types

    async def analyze_function_call_patterns(self, graph: StructureGraph, target_breadcrumb: str = None) -> dict[str, Any]:
        """
        Analyze function call patterns in the graph.

        Args:
            graph: Structure graph to analyze
            target_breadcrumb: Optional specific component to analyze

        Returns:
            Dictionary with function call pattern analysis
        """
        try:
            function_call_edges = [e for e in graph.edges if e.relationship_type == "function_call"]

            if not function_call_edges:
                return {"total_function_calls": 0, "has_function_calls": False, "message": "No function call relationships found in graph"}

            # Overall statistics
            total_calls = len(function_call_edges)
            unique_callers = len(set(e.source_breadcrumb for e in function_call_edges))
            unique_callees = len(set(e.target_breadcrumb for e in function_call_edges))

            # Call type analysis
            call_type_breakdown = defaultdict(int)
            async_calls = 0

            for edge in function_call_edges:
                call_type = edge.get_call_type()
                if call_type:
                    call_type_breakdown[call_type] += 1
                if edge.is_async_call():
                    async_calls += 1

            # Confidence and weight analysis
            avg_confidence = sum(e.confidence for e in function_call_edges) / total_calls
            avg_weight = sum(e.weight for e in function_call_edges) / total_calls

            # Most called functions
            callee_counts = defaultdict(int)
            for edge in function_call_edges:
                callee_counts[edge.target_breadcrumb] += 1

            most_called = sorted(callee_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            # Caller analysis
            caller_counts = defaultdict(int)
            for edge in function_call_edges:
                caller_counts[edge.source_breadcrumb] += 1

            most_active_callers = sorted(caller_counts.items(), key=lambda x: x[1], reverse=True)[:10]

            analysis = {
                "total_function_calls": total_calls,
                "has_function_calls": True,
                "unique_callers": unique_callers,
                "unique_callees": unique_callees,
                "call_type_breakdown": dict(call_type_breakdown),
                "async_call_percentage": (async_calls / total_calls) * 100 if total_calls > 0 else 0,
                "average_confidence": avg_confidence,
                "average_weight": avg_weight,
                "most_called_functions": [{"breadcrumb": breadcrumb, "call_count": count} for breadcrumb, count in most_called],
                "most_active_callers": [{"breadcrumb": breadcrumb, "calls_made": count} for breadcrumb, count in most_active_callers],
            }

            # Specific target analysis if requested
            if target_breadcrumb and target_breadcrumb in graph.nodes:
                target_incoming = [e for e in function_call_edges if e.target_breadcrumb == target_breadcrumb]
                target_outgoing = [e for e in function_call_edges if e.source_breadcrumb == target_breadcrumb]

                analysis["target_analysis"] = {
                    "breadcrumb": target_breadcrumb,
                    "calls_received": len(target_incoming),
                    "calls_made": len(target_outgoing),
                    "callers": [e.source_breadcrumb for e in target_incoming],
                    "callees": [e.target_breadcrumb for e in target_outgoing],
                }

            return analysis

        except Exception as e:
            self.logger.error(f"Error analyzing function call patterns: {e}")
            return {"total_function_calls": 0, "has_function_calls": False, "error": str(e)}

    def _calculate_max_distance(self, cluster_nodes: list[GraphNode]) -> int:
        """Calculate maximum distance between nodes in cluster."""
        if len(cluster_nodes) <= 1:
            return 0

        depths = [node.depth for node in cluster_nodes]
        return max(depths) - min(depths)

    async def _find_k_shortest_paths(self, graph: StructureGraph, start: str, end: str, k: int) -> list[tuple[list[str], float]]:
        """Find k shortest paths between two nodes."""
        # Simplified implementation - in practice would use Yen's algorithm
        paths = []

        # Find shortest path first
        shortest_path = await self._dijkstra_shortest_path(graph, start, end)
        if shortest_path:
            paths.append(shortest_path)

        return paths

    async def _dijkstra_shortest_path(self, graph: StructureGraph, start: str, end: str) -> tuple[list[str], float] | None:
        """Find shortest path using Dijkstra's algorithm."""
        import heapq

        distances = {start: 0.0}
        previous = {}
        visited = set()
        queue = [(0.0, start)]

        while queue:
            current_dist, current = heapq.heappop(queue)

            if current in visited:
                continue

            visited.add(current)

            if current == end:
                # Reconstruct path
                path = []
                node = end
                while node is not None:
                    path.append(node)
                    node = previous.get(node)
                path.reverse()
                return path, current_dist

            # Check neighbors
            for edge in graph.edges:
                if edge.source_breadcrumb == current:
                    neighbor = edge.target_breadcrumb
                    distance = current_dist + (1.0 / edge.weight)  # Invert weight for distance

                    if neighbor not in distances or distance < distances[neighbor]:
                        distances[neighbor] = distance
                        previous[neighbor] = current
                        heapq.heappush(queue, (distance, neighbor))

        return None

    def _get_path_edges(self, graph: StructureGraph, path_breadcrumbs: list[str]) -> list[GraphEdge]:
        """Get edges for a given path."""
        path_edges = []

        for i in range(len(path_breadcrumbs) - 1):
            source = path_breadcrumbs[i]
            target = path_breadcrumbs[i + 1]

            # Find edge between these nodes
            for edge in graph.edges:
                if edge.source_breadcrumb == source and edge.target_breadcrumb == target:
                    path_edges.append(edge)
                    break

        return path_edges

    def _calculate_relationship_diversity(self, path_edges: list[GraphEdge]) -> float:
        """Calculate diversity of relationship types in a path."""
        if not path_edges:
            return 0.0

        unique_types = {edge.relationship_type for edge in path_edges}
        return len(unique_types) / len(path_edges)

    async def _find_reachable_at_distance(self, graph: StructureGraph, start_breadcrumb: str, max_distance: int) -> dict[int, list[str]]:
        """Find nodes reachable at each distance."""
        reachable = defaultdict(list)
        visited = {start_breadcrumb: 0}
        queue = deque([(start_breadcrumb, 0)])

        while queue:
            breadcrumb, distance = queue.popleft()

            if distance > max_distance:
                continue

            reachable[distance].append(breadcrumb)

            # Find neighbors
            for edge in graph.edges:
                if edge.source_breadcrumb == breadcrumb:
                    neighbor = edge.target_breadcrumb
                    new_distance = distance + 1

                    if neighbor not in visited or new_distance < visited[neighbor]:
                        visited[neighbor] = new_distance
                        queue.append((neighbor, new_distance))

        return dict(reachable)

    def _calculate_influence_score(self, node: GraphNode, incoming_edges: list[GraphEdge], outgoing_edges: list[GraphEdge]) -> float:
        """Calculate influence score for a node."""
        # Combine semantic weight, degree centrality, and relationship weights
        base_score = node.semantic_weight

        # Incoming influence (how many depend on this)
        incoming_weight = sum(edge.weight * edge.confidence for edge in incoming_edges)

        # Outgoing influence (how much this depends on others)
        outgoing_weight = sum(edge.weight * edge.confidence for edge in outgoing_edges)

        # Higher incoming weight means more influence
        influence_score = base_score + (incoming_weight * 0.7) + (outgoing_weight * 0.3)

        return influence_score

    def _update_traversal_stats(self, strategy: TraversalStrategy, nodes_visited: int, execution_time: float):
        """Update traversal statistics."""
        self._traversal_stats["total_traversals"] += 1
        self._traversal_stats["strategy_usage"][strategy.value] += 1

        # Update running averages
        total = self._traversal_stats["total_traversals"]
        current_avg_nodes = self._traversal_stats["avg_nodes_visited"]
        current_avg_time = self._traversal_stats["avg_execution_time_ms"]

        self._traversal_stats["avg_nodes_visited"] = (current_avg_nodes * (total - 1) + nodes_visited) / total
        self._traversal_stats["avg_execution_time_ms"] = (current_avg_time * (total - 1) + execution_time) / total

    def get_traversal_statistics(self) -> dict[str, Any]:
        """Get traversal performance statistics."""
        return self._traversal_stats.copy()

    def reset_statistics(self):
        """Reset traversal statistics."""
        self._traversal_stats = {
            "total_traversals": 0,
            "avg_nodes_visited": 0.0,
            "avg_execution_time_ms": 0.0,
            "strategy_usage": defaultdict(int),
        }
