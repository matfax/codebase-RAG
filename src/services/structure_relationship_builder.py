"""
Structure Relationship Builder for Graph RAG enhancement.

This service builds code structure relationship graphs from enhanced CodeChunk objects,
creating hierarchical relationships and dependency mappings for Graph RAG functionality.
Leverages Wave 1's breadcrumb and parent_name metadata to construct accurate graphs.
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional

from ..models.code_chunk import ChunkType, CodeChunk
from .structure_analyzer_service import StructureAnalyzerService


# Import graph data structures from graph_rag_service to avoid circular imports
# We'll define them here and import in graph_rag_service instead
@dataclass
class GraphNode:
    """Represents a node in the code structure graph."""

    chunk_id: str
    breadcrumb: str
    name: str
    chunk_type: ChunkType
    file_path: str
    parent_breadcrumb: str | None = None
    children_breadcrumbs: list[str] = None
    depth: int = 0
    semantic_weight: float = 1.0

    def __post_init__(self):
        if self.children_breadcrumbs is None:
            self.children_breadcrumbs = []


@dataclass
class GraphEdge:
    """Represents a relationship edge between graph nodes."""

    source_breadcrumb: str
    target_breadcrumb: str
    relationship_type: str  # "parent_child", "dependency", "implementation", "interface", "sibling"
    weight: float = 1.0
    confidence: float = 1.0  # Confidence in this relationship
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class StructureGraph:
    """Represents the complete code structure relationship graph."""

    nodes: dict[str, GraphNode]  # breadcrumb -> GraphNode
    edges: list[GraphEdge]
    project_name: str
    root_nodes: list[str] = None  # Top-level breadcrumbs
    build_timestamp: float = 0.0
    total_chunks_processed: int = 0

    def __post_init__(self):
        if self.root_nodes is None:
            self.root_nodes = []
        if self.build_timestamp == 0.0:
            self.build_timestamp = time.time()


@dataclass
class RelationshipStats:
    """Statistics for relationship building process."""

    total_chunks: int = 0
    nodes_created: int = 0
    edges_created: int = 0
    parent_child_relationships: int = 0
    dependency_relationships: int = 0
    interface_relationships: int = 0
    sibling_relationships: int = 0
    orphaned_nodes: int = 0
    max_depth: int = 0
    build_time_ms: float = 0.0
    language_breakdown: dict[str, int] = None

    def __post_init__(self):
        if self.language_breakdown is None:
            self.language_breakdown = {}


class StructureRelationshipBuilder:
    """
    Service for building code structure relationship graphs.

    This service analyzes CodeChunk objects with enhanced breadcrumb and parent_name
    metadata to construct hierarchical relationship graphs suitable for Graph RAG
    navigation and analysis.
    """

    def __init__(self, qdrant_service, structure_analyzer: StructureAnalyzerService):
        """
        Initialize the relationship builder.

        Args:
            qdrant_service: Qdrant service for data retrieval (will be used in task 2.5)
            structure_analyzer: Structure analyzer service for breadcrumb analysis
        """
        self.logger = logging.getLogger(__name__)
        self.qdrant_service = qdrant_service
        self.structure_analyzer = structure_analyzer

        # Configuration for relationship building
        self.max_dependency_depth = 5  # Maximum depth for dependency analysis
        self.confidence_threshold = 0.7  # Minimum confidence for relationships

        # Statistics tracking
        self._build_stats = RelationshipStats()

        self.logger.info("StructureRelationshipBuilder initialized")

    async def build_relationship_graph(self, chunks: list[CodeChunk], project_name: str) -> StructureGraph:
        """
        Build a complete structure relationship graph from CodeChunk objects.

        Args:
            chunks: List of enhanced CodeChunk objects with breadcrumb metadata
            project_name: Name of the project being analyzed

        Returns:
            StructureGraph representing the complete code structure relationships
        """
        start_time = time.time()

        try:
            self.logger.info(f"Building relationship graph for {len(chunks)} chunks in project: {project_name}")

            # Reset build statistics
            self._build_stats = RelationshipStats(total_chunks=len(chunks))

            # Phase 1: Create nodes from chunks
            nodes = await self._create_nodes_from_chunks(chunks)
            self._build_stats.nodes_created = len(nodes)

            # Phase 2: Build hierarchical relationships using breadcrumbs
            edges = await self._build_hierarchical_relationships(nodes)

            # Phase 3: Identify dependency relationships
            dependency_edges = await self._build_dependency_relationships(chunks, nodes)
            edges.extend(dependency_edges)

            # Phase 4: Identify interface/implementation relationships
            interface_edges = await self._build_interface_relationships(chunks, nodes)
            edges.extend(interface_edges)

            # Phase 5: Identify sibling relationships
            sibling_edges = await self._build_sibling_relationships(nodes)
            edges.extend(sibling_edges)

            # Phase 6: Calculate graph properties
            root_nodes = await self._identify_root_nodes(nodes, edges)
            await self._calculate_node_depths(nodes, edges, root_nodes)

            # Update statistics
            self._build_stats.edges_created = len(edges)
            self._build_stats.max_depth = max(node.depth for node in nodes.values()) if nodes else 0
            self._build_stats.build_time_ms = (time.time() - start_time) * 1000

            # Create the final graph
            graph = StructureGraph(
                nodes=nodes, edges=edges, project_name=project_name, root_nodes=root_nodes, total_chunks_processed=len(chunks)
            )

            self.logger.info(
                f"Built relationship graph: {len(nodes)} nodes, {len(edges)} edges, "
                f"max depth {self._build_stats.max_depth} in {self._build_stats.build_time_ms:.2f}ms"
            )

            return graph

        except Exception as e:
            self.logger.error(f"Error building relationship graph: {e}")
            raise

    async def _create_nodes_from_chunks(self, chunks: list[CodeChunk]) -> dict[str, GraphNode]:
        """
        Create graph nodes from CodeChunk objects.

        Args:
            chunks: List of CodeChunk objects to convert

        Returns:
            Dictionary mapping breadcrumb to GraphNode
        """
        nodes = {}

        for chunk in chunks:
            # Skip chunks without breadcrumbs (these are typically raw code or whole files)
            if not chunk.breadcrumb:
                self.logger.debug(f"Skipping chunk without breadcrumb: {chunk.name} in {chunk.file_path}")
                continue

            # Track language breakdown
            language = chunk.language.lower()
            if language not in self._build_stats.language_breakdown:
                self._build_stats.language_breakdown[language] = 0
            self._build_stats.language_breakdown[language] += 1

            # Create node from chunk
            node = GraphNode(
                chunk_id=chunk.chunk_id,
                breadcrumb=chunk.breadcrumb,
                name=chunk.name or "unnamed",
                chunk_type=chunk.chunk_type,
                file_path=chunk.file_path,
                parent_breadcrumb=self._extract_parent_breadcrumb(chunk.breadcrumb),
                semantic_weight=self._calculate_semantic_weight(chunk),
            )

            nodes[chunk.breadcrumb] = node

        self.logger.debug(f"Created {len(nodes)} nodes from chunks")
        return nodes

    async def _build_hierarchical_relationships(self, nodes: dict[str, GraphNode]) -> list[GraphEdge]:
        """
        Build parent-child relationships based on breadcrumb hierarchy.

        Args:
            nodes: Dictionary of graph nodes

        Returns:
            List of hierarchical relationship edges
        """
        edges = []

        for breadcrumb, node in nodes.items():
            if node.parent_breadcrumb and node.parent_breadcrumb in nodes:
                # Create parent-child edge
                edge = GraphEdge(
                    source_breadcrumb=node.parent_breadcrumb,
                    target_breadcrumb=breadcrumb,
                    relationship_type="parent_child",
                    weight=1.0,
                    confidence=1.0,  # High confidence for breadcrumb-based relationships
                    metadata={
                        "hierarchy_level": node.breadcrumb.count(".") + node.breadcrumb.count("::"),
                        "parent_type": nodes[node.parent_breadcrumb].chunk_type.value,
                        "child_type": node.chunk_type.value,
                    },
                )
                edges.append(edge)

                # Update parent's children list
                parent_node = nodes[node.parent_breadcrumb]
                if breadcrumb not in parent_node.children_breadcrumbs:
                    parent_node.children_breadcrumbs.append(breadcrumb)

                self._build_stats.parent_child_relationships += 1

        self.logger.debug(f"Built {self._build_stats.parent_child_relationships} hierarchical relationships")
        return edges

    async def _build_dependency_relationships(self, chunks: list[CodeChunk], nodes: dict[str, GraphNode]) -> list[GraphEdge]:
        """
        Build dependency relationships based on imports and function calls.

        Args:
            chunks: Original chunks with dependency information
            nodes: Graph nodes dictionary

        Returns:
            List of dependency relationship edges
        """
        edges = []

        # Note: Direct breadcrumb lookup used instead of chunk_id mapping

        for chunk in chunks:
            if not chunk.breadcrumb or chunk.breadcrumb not in nodes:
                continue

            # Analyze imports_used field for dependencies
            if chunk.imports_used:
                for import_name in chunk.imports_used:
                    target_breadcrumb = self._resolve_import_to_breadcrumb(import_name, nodes)
                    if target_breadcrumb and target_breadcrumb != chunk.breadcrumb:
                        edge = GraphEdge(
                            source_breadcrumb=chunk.breadcrumb,
                            target_breadcrumb=target_breadcrumb,
                            relationship_type="dependency",
                            weight=0.8,
                            confidence=0.9,
                            metadata={"dependency_type": "import", "import_name": import_name},
                        )
                        edges.append(edge)
                        self._build_stats.dependency_relationships += 1

            # Analyze dependencies field for additional relationships
            if chunk.dependencies:
                for dep_name in chunk.dependencies:
                    target_breadcrumb = self._resolve_dependency_to_breadcrumb(dep_name, nodes)
                    if target_breadcrumb and target_breadcrumb != chunk.breadcrumb:
                        edge = GraphEdge(
                            source_breadcrumb=chunk.breadcrumb,
                            target_breadcrumb=target_breadcrumb,
                            relationship_type="dependency",
                            weight=0.7,
                            confidence=0.8,
                            metadata={"dependency_type": "reference", "reference_name": dep_name},
                        )
                        edges.append(edge)
                        self._build_stats.dependency_relationships += 1

        self.logger.debug(f"Built {self._build_stats.dependency_relationships} dependency relationships")
        return edges

    async def _build_interface_relationships(self, chunks: list[CodeChunk], nodes: dict[str, GraphNode]) -> list[GraphEdge]:
        """
        Build interface/implementation relationships.

        Args:
            chunks: Original chunks to analyze
            nodes: Graph nodes dictionary

        Returns:
            List of interface relationship edges
        """
        edges = []

        # Group chunks by type to identify interfaces and implementations
        interfaces = {breadcrumb: node for breadcrumb, node in nodes.items() if node.chunk_type == ChunkType.INTERFACE}

        classes = {breadcrumb: node for breadcrumb, node in nodes.items() if node.chunk_type == ChunkType.CLASS}

        # Find implementation relationships by analyzing signatures and names
        for class_breadcrumb, class_node in classes.items():
            # Find corresponding chunk for detailed analysis
            class_chunk = next((c for c in chunks if c.breadcrumb == class_breadcrumb), None)
            if not class_chunk:
                continue

            # Look for interface implementations in the signature or dependencies
            for interface_breadcrumb, interface_node in interfaces.items():
                if self._is_implementation_relationship(class_chunk, interface_node):
                    edge = GraphEdge(
                        source_breadcrumb=class_breadcrumb,
                        target_breadcrumb=interface_breadcrumb,
                        relationship_type="implementation",
                        weight=0.9,
                        confidence=0.85,
                        metadata={"interface_name": interface_node.name, "class_name": class_node.name},
                    )
                    edges.append(edge)
                    self._build_stats.interface_relationships += 1

        self.logger.debug(f"Built {self._build_stats.interface_relationships} interface relationships")
        return edges

    async def _build_sibling_relationships(self, nodes: dict[str, GraphNode]) -> list[GraphEdge]:
        """
        Build sibling relationships between nodes with the same parent.

        Args:
            nodes: Graph nodes dictionary

        Returns:
            List of sibling relationship edges
        """
        edges = []

        # Group nodes by parent to identify siblings
        parent_to_children = defaultdict(list)
        for breadcrumb, node in nodes.items():
            if node.parent_breadcrumb:
                parent_to_children[node.parent_breadcrumb].append(breadcrumb)

        # Create sibling edges for nodes with the same parent
        for parent_breadcrumb, children in parent_to_children.items():
            if len(children) > 1:
                # Create bidirectional sibling relationships
                for i, child1 in enumerate(children):
                    for child2 in children[i + 1 :]:
                        # Create edge from child1 to child2
                        edge1 = GraphEdge(
                            source_breadcrumb=child1,
                            target_breadcrumb=child2,
                            relationship_type="sibling",
                            weight=0.5,
                            confidence=1.0,
                            metadata={"parent_breadcrumb": parent_breadcrumb, "sibling_count": len(children)},
                        )

                        # Create edge from child2 to child1
                        edge2 = GraphEdge(
                            source_breadcrumb=child2,
                            target_breadcrumb=child1,
                            relationship_type="sibling",
                            weight=0.5,
                            confidence=1.0,
                            metadata={"parent_breadcrumb": parent_breadcrumb, "sibling_count": len(children)},
                        )

                        edges.extend([edge1, edge2])
                        self._build_stats.sibling_relationships += 2

        self.logger.debug(f"Built {self._build_stats.sibling_relationships} sibling relationships")
        return edges

    async def _identify_root_nodes(self, nodes: dict[str, GraphNode], edges: list[GraphEdge]) -> list[str]:
        """
        Identify root nodes (nodes with no parents).

        Args:
            nodes: Graph nodes dictionary
            edges: List of all edges

        Returns:
            List of root node breadcrumbs
        """
        # Find nodes that are not targets of any parent-child relationship
        child_nodes = set()
        for edge in edges:
            if edge.relationship_type == "parent_child":
                child_nodes.add(edge.target_breadcrumb)

        root_nodes = []
        for breadcrumb in nodes.keys():
            if breadcrumb not in child_nodes:
                root_nodes.append(breadcrumb)

        # If no clear roots found, use nodes with shortest breadcrumbs
        if not root_nodes:
            min_depth = min(breadcrumb.count(".") + breadcrumb.count("::") for breadcrumb in nodes.keys())
            root_nodes = [breadcrumb for breadcrumb in nodes.keys() if breadcrumb.count(".") + breadcrumb.count("::") == min_depth]

        self.logger.debug(f"Identified {len(root_nodes)} root nodes")
        return root_nodes

    async def _calculate_node_depths(self, nodes: dict[str, GraphNode], edges: list[GraphEdge], root_nodes: list[str]):
        """
        Calculate depth for each node in the hierarchy.

        Args:
            nodes: Graph nodes dictionary (modified in place)
            edges: List of all edges
            root_nodes: List of root node breadcrumbs
        """
        # Build adjacency list for parent-child relationships
        children_map = defaultdict(list)
        for edge in edges:
            if edge.relationship_type == "parent_child":
                children_map[edge.source_breadcrumb].append(edge.target_breadcrumb)

        # BFS to calculate depths
        visited = set()
        queue = [(breadcrumb, 0) for breadcrumb in root_nodes]

        for breadcrumb in root_nodes:
            if breadcrumb in nodes:
                nodes[breadcrumb].depth = 0
                visited.add(breadcrumb)

        while queue:
            current_breadcrumb, depth = queue.pop(0)

            for child_breadcrumb in children_map[current_breadcrumb]:
                if child_breadcrumb not in visited and child_breadcrumb in nodes:
                    nodes[child_breadcrumb].depth = depth + 1
                    visited.add(child_breadcrumb)
                    queue.append((child_breadcrumb, depth + 1))

        # Handle orphaned nodes
        orphan_count = 0
        for breadcrumb, node in nodes.items():
            if breadcrumb not in visited:
                # Assign depth based on breadcrumb structure as fallback
                node.depth = breadcrumb.count(".") + breadcrumb.count("::")
                orphan_count += 1

        self._build_stats.orphaned_nodes = orphan_count
        if orphan_count > 0:
            self.logger.debug(f"Found {orphan_count} orphaned nodes, assigned depths based on breadcrumb structure")

    # =================== Helper Methods ===================

    def _extract_parent_breadcrumb(self, breadcrumb: str) -> str | None:
        """Extract parent breadcrumb from a child breadcrumb."""
        if not breadcrumb:
            return None

        # Handle both dot notation and double colon notation
        if "::" in breadcrumb:
            parts = breadcrumb.split("::")
            if len(parts) > 1:
                return "::".join(parts[:-1])
        else:
            parts = breadcrumb.split(".")
            if len(parts) > 1:
                return ".".join(parts[:-1])

        return None

    def _calculate_semantic_weight(self, chunk: CodeChunk) -> float:
        """
        Calculate semantic weight for a chunk based on its properties.

        Args:
            chunk: CodeChunk to analyze

        Returns:
            Semantic weight value (0.1 to 1.0)
        """
        weight = 0.5  # Base weight

        # Increase weight for functions and classes (more important)
        if chunk.chunk_type in [ChunkType.FUNCTION, ChunkType.CLASS, ChunkType.INTERFACE]:
            weight += 0.3

        # Increase weight for documented chunks
        if chunk.docstring:
            weight += 0.1

        # Increase weight based on complexity
        if chunk.complexity_score:
            weight += min(chunk.complexity_score, 0.1)

        # Decrease weight for chunks with syntax errors
        if chunk.has_syntax_errors:
            weight -= 0.2

        return max(0.1, min(1.0, weight))

    def _resolve_import_to_breadcrumb(self, import_name: str, nodes: dict[str, GraphNode]) -> str | None:
        """
        Resolve an import name to a breadcrumb in the graph.

        Args:
            import_name: Name of the imported entity
            nodes: Available graph nodes

        Returns:
            Matching breadcrumb or None
        """
        # Direct name match
        for breadcrumb, node in nodes.items():
            if node.name == import_name:
                return breadcrumb

        # Partial breadcrumb match
        for breadcrumb in nodes.keys():
            if import_name in breadcrumb:
                return breadcrumb

        return None

    def _resolve_dependency_to_breadcrumb(self, dep_name: str, nodes: dict[str, GraphNode]) -> str | None:
        """
        Resolve a dependency name to a breadcrumb in the graph.

        Args:
            dep_name: Name of the dependency
            nodes: Available graph nodes

        Returns:
            Matching breadcrumb or None
        """
        # Similar to import resolution but with lower confidence
        return self._resolve_import_to_breadcrumb(dep_name, nodes)

    def _is_implementation_relationship(self, class_chunk: CodeChunk, interface_node: GraphNode) -> bool:
        """
        Determine if a class implements an interface.

        Args:
            class_chunk: Class chunk to check
            interface_node: Interface node to check against

        Returns:
            True if implementation relationship exists
        """
        if not class_chunk.signature:
            return False

        # Check if interface name appears in class signature
        interface_name = interface_node.name
        return interface_name in class_chunk.signature

    def get_build_statistics(self) -> RelationshipStats:
        """Get the latest build statistics."""
        return self._build_stats
