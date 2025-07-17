"""
Implementation Chain Service for Graph RAG enhancement.

This service provides implementation chain tracking functionality, enabling developers
to trace complete execution flows from entry points to detailed implementations,
understanding how functionality is organized and connected across the codebase.

Built on top of Wave 2's Graph RAG infrastructure and traversal algorithms.
"""

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..models.code_chunk import ChunkType, CodeChunk
from .graph_rag_service import GraphRAGService
from .hybrid_search_service import HybridSearchParameters, HybridSearchService, HybridSearchStrategy
from .structure_relationship_builder import GraphNode, StructureGraph


class ChainType(Enum):
    """Types of implementation chains that can be tracked."""

    EXECUTION_FLOW = "execution_flow"  # Function calls and execution paths
    DATA_FLOW = "data_flow"  # Data transformation and processing chains
    DEPENDENCY_CHAIN = "dependency_chain"  # Component dependency chains
    INHERITANCE_CHAIN = "inheritance_chain"  # Class inheritance hierarchies
    INTERFACE_IMPLEMENTATION = "interface_implementation"  # Interface to implementation mappings
    SERVICE_LAYER_CHAIN = "service_layer_chain"  # Service layer interaction chains
    API_ENDPOINT_CHAIN = "api_endpoint_chain"  # API endpoint to implementation chains
    EVENT_HANDLING_CHAIN = "event_handling_chain"  # Event handling and processing chains
    CONFIGURATION_CHAIN = "configuration_chain"  # Configuration to usage chains
    TEST_COVERAGE_CHAIN = "test_coverage_chain"  # Test to implementation coverage chains


class ChainDirection(Enum):
    """Direction for chain traversal."""

    FORWARD = "forward"  # From entry point to implementation details
    BACKWARD = "backward"  # From implementation details to entry points
    BIDIRECTIONAL = "bidirectional"  # Both directions


@dataclass
class ChainLink:
    """Represents a single link in an implementation chain."""

    source_component: GraphNode
    target_component: GraphNode
    relationship_type: str  # Type of relationship (calls, inherits, implements, etc.)
    link_strength: float  # Strength of the relationship (0.0-1.0)

    # Context information
    interaction_type: str  # How they interact (method_call, data_access, etc.)
    parameters_passed: list[str] = None  # Parameters or data passed
    return_values: list[str] = None  # Return values or outputs

    # Evidence for the link
    evidence_source: str = ""  # Where this relationship was detected
    confidence: float = 1.0  # Confidence in this relationship

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.parameters_passed is None:
            self.parameters_passed = []
        if self.return_values is None:
            self.return_values = []


@dataclass
class ImplementationChain:
    """Represents a complete implementation chain."""

    chain_id: str
    chain_type: ChainType
    entry_point: GraphNode
    terminal_points: list[GraphNode]  # End points of the chain

    # Chain structure
    links: list[ChainLink]
    depth: int  # Maximum depth of the chain
    branch_count: int  # Number of branches in the chain

    # Chain metrics
    complexity_score: float  # How complex this chain is
    completeness_score: float  # How complete the chain tracking is
    reliability_score: float  # How reliable the chain relationships are

    # Context information
    project_name: str
    scope_breadcrumb: str = ""  # Common breadcrumb scope for the chain
    functional_purpose: str = ""  # What this chain accomplishes

    # Chain statistics
    total_components: int = 0
    components_by_type: dict[ChunkType, int] = None
    avg_link_strength: float = 0.0

    def __post_init__(self):
        """Initialize default values and compute statistics."""
        if self.components_by_type is None:
            self.components_by_type = {}

        # Compute statistics from links
        if self.links:
            # Count total unique components
            all_components = set()
            total_strength = 0.0

            for link in self.links:
                all_components.add(link.source_component.chunk_id)
                all_components.add(link.target_component.chunk_id)
                total_strength += link.link_strength

            self.total_components = len(all_components)
            self.avg_link_strength = total_strength / len(self.links)

            # Count components by type
            for link in self.links:
                for component in [link.source_component, link.target_component]:
                    chunk_type = component.chunk_type
                    self.components_by_type[chunk_type] = self.components_by_type.get(chunk_type, 0) + 1


@dataclass
class ChainAnalysisResult:
    """Result of implementation chain analysis."""

    project_name: str
    analysis_scope: str
    chains_found: list[ImplementationChain]

    # Analysis statistics
    total_entry_points_analyzed: int
    total_components_in_chains: int
    analysis_time_ms: float
    coverage_percentage: float  # Percentage of components covered by chains

    # Chain distribution
    chains_by_type: dict[ChainType, int] = None
    chains_by_complexity: dict[str, int] = None  # Simple/Medium/Complex
    chains_by_depth: dict[int, int] = None  # Chain depth distribution

    # Quality metrics
    avg_chain_completeness: float = 0.0
    avg_chain_reliability: float = 0.0
    chain_connectivity_score: float = 0.0  # How well chains connect to each other

    def __post_init__(self):
        """Initialize default values and compute statistics."""
        if self.chains_by_type is None:
            self.chains_by_type = {}
        if self.chains_by_complexity is None:
            self.chains_by_complexity = {}
        if self.chains_by_depth is None:
            self.chains_by_depth = {}

        # Compute statistics from found chains
        if self.chains_found:
            # Count chains by type
            for chain in self.chains_found:
                self.chains_by_type[chain.chain_type] = self.chains_by_type.get(chain.chain_type, 0) + 1

                # Categorize by complexity
                if chain.complexity_score >= 0.7:
                    complexity_category = "complex"
                elif chain.complexity_score >= 0.4:
                    complexity_category = "medium"
                else:
                    complexity_category = "simple"
                self.chains_by_complexity[complexity_category] = self.chains_by_complexity.get(complexity_category, 0) + 1

                # Count by depth
                self.chains_by_depth[chain.depth] = self.chains_by_depth.get(chain.depth, 0) + 1

            # Calculate average metrics
            self.avg_chain_completeness = sum(c.completeness_score for c in self.chains_found) / len(self.chains_found)
            self.avg_chain_reliability = sum(c.reliability_score for c in self.chains_found) / len(self.chains_found)


class ImplementationChainService:
    """
    Service for tracking and analyzing implementation chains in codebases.

    This service traces complete execution flows from entry points to implementation
    details, helping developers understand how functionality is organized and
    connected across the codebase.
    """

    def __init__(
        self,
        graph_rag_service: GraphRAGService,
        hybrid_search_service: HybridSearchService,
    ):
        """Initialize the implementation chain service.

        Args:
            graph_rag_service: Service for graph operations and structural analysis
            hybrid_search_service: Service for hybrid searching capabilities
        """
        self.graph_rag_service = graph_rag_service
        self.hybrid_search_service = hybrid_search_service
        self.logger = logging.getLogger(__name__)

        # Cache for chain analysis results
        self._chain_cache = {}

    async def trace_implementation_chain(
        self,
        entry_point_breadcrumb: str,
        project_name: str,
        chain_type: ChainType = ChainType.EXECUTION_FLOW,
        direction: ChainDirection = ChainDirection.FORWARD,
        max_depth: int = 10,
        min_link_strength: float = 0.3,
    ) -> ImplementationChain:
        """
        Trace a complete implementation chain from an entry point.

        Args:
            entry_point_breadcrumb: Breadcrumb of the entry point component
            project_name: Name of the project to analyze
            chain_type: Type of chain to trace
            direction: Direction for chain traversal
            max_depth: Maximum depth to traverse
            min_link_strength: Minimum link strength to include

        Returns:
            ImplementationChain with the traced chain
        """
        try:
            self.logger.info(f"Tracing {chain_type.value} chain from: {entry_point_breadcrumb}")

            # Get the entry point component
            project_graph = await self.graph_rag_service.get_project_structure_graph(project_name)

            if not project_graph:
                self.logger.warning(f"No structure graph found for project: {project_name}")
                return self._create_empty_chain(entry_point_breadcrumb, project_name, chain_type)

            entry_point = self._find_component_by_breadcrumb(project_graph, entry_point_breadcrumb)

            if not entry_point:
                self.logger.warning(f"Entry point not found: {entry_point_breadcrumb}")
                return self._create_empty_chain(entry_point_breadcrumb, project_name, chain_type)

            # Trace the chain based on type and direction
            links = []
            visited_components = set()
            terminal_points = []

            if direction in [ChainDirection.FORWARD, ChainDirection.BIDIRECTIONAL]:
                forward_links, forward_terminals = await self._trace_forward_chain(
                    entry_point, project_graph, chain_type, max_depth, min_link_strength, visited_components
                )
                links.extend(forward_links)
                terminal_points.extend(forward_terminals)

            if direction in [ChainDirection.BACKWARD, ChainDirection.BIDIRECTIONAL]:
                backward_links, backward_terminals = await self._trace_backward_chain(
                    entry_point, project_graph, chain_type, max_depth, min_link_strength, visited_components
                )
                links.extend(backward_links)
                terminal_points.extend(backward_terminals)

            # Create the implementation chain
            chain = ImplementationChain(
                chain_id=f"{project_name}_{entry_point_breadcrumb}_{chain_type.value}",
                chain_type=chain_type,
                entry_point=entry_point,
                terminal_points=terminal_points,
                links=links,
                depth=self._calculate_chain_depth(links, entry_point),
                branch_count=self._calculate_branch_count(links, entry_point),
                complexity_score=self._calculate_chain_complexity(links, terminal_points),
                completeness_score=self._calculate_chain_completeness(links, entry_point, chain_type),
                reliability_score=self._calculate_chain_reliability(links),
                project_name=project_name,
                scope_breadcrumb=self._determine_chain_scope(links),
                functional_purpose=self._infer_functional_purpose(entry_point, links, chain_type),
            )

            self.logger.info(f"Traced chain with {len(links)} links, depth {chain.depth}, " f"completeness {chain.completeness_score:.2f}")

            return chain

        except Exception as e:
            self.logger.error(f"Error tracing implementation chain: {e}")
            return self._create_empty_chain(entry_point_breadcrumb, project_name, chain_type)

    async def analyze_project_chains(
        self,
        project_name: str,
        scope_breadcrumb: str | None = None,
        chain_types: list[ChainType] | None = None,
        max_depth: int = 8,
    ) -> ChainAnalysisResult:
        """
        Analyze implementation chains across a project or scope.

        Args:
            project_name: Name of the project to analyze
            scope_breadcrumb: Optional breadcrumb to limit analysis scope
            chain_types: Types of chains to analyze (all types if None)
            max_depth: Maximum depth for chain tracing

        Returns:
            ChainAnalysisResult with analysis of all chains
        """
        start_time = time.time()

        try:
            self.logger.info(f"Analyzing implementation chains for project: {project_name}")

            # Default to analyzing all chain types
            if chain_types is None:
                chain_types = [
                    ChainType.EXECUTION_FLOW,
                    ChainType.DEPENDENCY_CHAIN,
                    ChainType.INHERITANCE_CHAIN,
                    ChainType.SERVICE_LAYER_CHAIN,
                ]

            # Get project structure
            project_graph = await self.graph_rag_service.get_project_structure_graph(project_name)

            if not project_graph:
                self.logger.warning(f"No structure graph found for project: {project_name}")
                return ChainAnalysisResult(
                    project_name=project_name,
                    analysis_scope=scope_breadcrumb or "project-wide",
                    chains_found=[],
                    total_entry_points_analyzed=0,
                    total_components_in_chains=0,
                    analysis_time_ms=(time.time() - start_time) * 1000,
                    coverage_percentage=0.0,
                )

            # Find entry points for chain analysis
            entry_points = self._identify_entry_points(project_graph, scope_breadcrumb)

            # Trace chains from each entry point
            all_chains = []

            for entry_point in entry_points:
                for chain_type in chain_types:
                    try:
                        chain = await self.trace_implementation_chain(
                            entry_point.breadcrumb,
                            project_name,
                            chain_type,
                            ChainDirection.FORWARD,
                            max_depth,
                        )

                        if chain.links:  # Only include chains with actual links
                            all_chains.append(chain)

                    except Exception as e:
                        self.logger.error(f"Error tracing chain from {entry_point.breadcrumb}: {e}")
                        continue

            # Calculate coverage and connectivity
            coverage_percentage = self._calculate_chain_coverage(project_graph, all_chains, scope_breadcrumb)
            connectivity_score = self._calculate_chain_connectivity(all_chains)

            analysis_time_ms = (time.time() - start_time) * 1000

            result = ChainAnalysisResult(
                project_name=project_name,
                analysis_scope=scope_breadcrumb or "project-wide",
                chains_found=all_chains,
                total_entry_points_analyzed=len(entry_points),
                total_components_in_chains=sum(chain.total_components for chain in all_chains),
                analysis_time_ms=analysis_time_ms,
                coverage_percentage=coverage_percentage,
                chain_connectivity_score=connectivity_score,
            )

            self.logger.info(
                f"Chain analysis completed in {analysis_time_ms:.2f}ms. "
                f"Found {len(all_chains)} chains with {coverage_percentage:.1f}% coverage."
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in project chain analysis: {e}")
            analysis_time_ms = (time.time() - start_time) * 1000
            return ChainAnalysisResult(
                project_name=project_name,
                analysis_scope=scope_breadcrumb or "project-wide",
                chains_found=[],
                total_entry_points_analyzed=0,
                total_components_in_chains=0,
                analysis_time_ms=analysis_time_ms,
                coverage_percentage=0.0,
            )

    async def find_similar_implementation_patterns(
        self,
        reference_chain: ImplementationChain,
        target_projects: list[str],
        similarity_threshold: float = 0.6,
    ) -> list[ImplementationChain]:
        """
        Find similar implementation patterns across projects.

        Args:
            reference_chain: The chain to find similar patterns for
            target_projects: Projects to search in
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of similar implementation chains
        """
        try:
            self.logger.info(f"Finding similar implementation patterns to {reference_chain.chain_id}")

            similar_chains = []

            for project_name in target_projects:
                # Analyze chains in target project
                project_analysis = await self.analyze_project_chains(project_name, chain_types=[reference_chain.chain_type])

                # Find similar chains
                for chain in project_analysis.chains_found:
                    similarity = self._calculate_chain_similarity(reference_chain, chain)

                    if similarity >= similarity_threshold:
                        # Add similarity information
                        chain.functional_purpose += f" (similarity: {similarity:.2f})"
                        similar_chains.append(chain)

            # Sort by similarity (highest first)
            similar_chains.sort(key=lambda c: float(c.functional_purpose.split("similarity: ")[-1].split(")")[0]), reverse=True)

            return similar_chains

        except Exception as e:
            self.logger.error(f"Error finding similar implementation patterns: {e}")
            return []

    async def trace_data_flow_chain(
        self,
        data_source_breadcrumb: str,
        project_name: str,
        max_depth: int = 8,
    ) -> ImplementationChain:
        """
        Trace data flow through the codebase from a source.

        Args:
            data_source_breadcrumb: Breadcrumb of the data source
            project_name: Name of the project to analyze
            max_depth: Maximum depth to trace

        Returns:
            ImplementationChain representing the data flow
        """
        return await self.trace_implementation_chain(
            data_source_breadcrumb,
            project_name,
            ChainType.DATA_FLOW,
            ChainDirection.FORWARD,
            max_depth,
        )

    async def trace_api_endpoint_implementation(
        self,
        endpoint_breadcrumb: str,
        project_name: str,
        max_depth: int = 10,
    ) -> ImplementationChain:
        """
        Trace API endpoint implementation from endpoint to business logic.

        Args:
            endpoint_breadcrumb: Breadcrumb of the API endpoint
            project_name: Name of the project to analyze
            max_depth: Maximum depth to trace

        Returns:
            ImplementationChain representing the API implementation
        """
        return await self.trace_implementation_chain(
            endpoint_breadcrumb,
            project_name,
            ChainType.API_ENDPOINT_CHAIN,
            ChainDirection.FORWARD,
            max_depth,
        )

    def _find_component_by_breadcrumb(self, graph: StructureGraph, breadcrumb: str) -> GraphNode | None:
        """Find a component in the graph by its breadcrumb."""
        for node in graph.nodes:
            if node.breadcrumb == breadcrumb:
                return node
        return None

    async def _trace_forward_chain(
        self,
        entry_point: GraphNode,
        graph: StructureGraph,
        chain_type: ChainType,
        max_depth: int,
        min_link_strength: float,
        visited: set,
    ) -> tuple[list[ChainLink], list[GraphNode]]:
        """Trace forward chain from entry point."""
        links = []
        terminals = []

        try:
            # Use BFS to trace forward relationships
            queue = deque([(entry_point, 0)])  # (node, depth)
            visited.add(entry_point.chunk_id)

            while queue and len(links) < 100:  # Limit to prevent infinite loops
                current_node, depth = queue.popleft()

                if depth >= max_depth:
                    terminals.append(current_node)
                    continue

                # Find forward relationships based on chain type
                forward_relationships = await self._find_forward_relationships(current_node, graph, chain_type)

                has_children = False
                for target_node, relationship_type, strength in forward_relationships:
                    if target_node.chunk_id not in visited and strength >= min_link_strength:
                        # Create chain link
                        link = ChainLink(
                            source_component=current_node,
                            target_component=target_node,
                            relationship_type=relationship_type,
                            link_strength=strength,
                            interaction_type=self._determine_interaction_type(relationship_type),
                            evidence_source="graph_traversal",
                            confidence=strength,
                        )

                        links.append(link)
                        visited.add(target_node.chunk_id)
                        queue.append((target_node, depth + 1))
                        has_children = True

                # If no forward relationships found, this is a terminal
                if not has_children and depth > 0:
                    terminals.append(current_node)

            return links, terminals

        except Exception as e:
            self.logger.error(f"Error tracing forward chain: {e}")
            return [], []

    async def _trace_backward_chain(
        self,
        entry_point: GraphNode,
        graph: StructureGraph,
        chain_type: ChainType,
        max_depth: int,
        min_link_strength: float,
        visited: set,
    ) -> tuple[list[ChainLink], list[GraphNode]]:
        """Trace backward chain to entry point."""
        links = []
        terminals = []

        try:
            # Use BFS to trace backward relationships
            queue = deque([(entry_point, 0)])  # (node, depth)

            while queue and len(links) < 100:  # Limit to prevent infinite loops
                current_node, depth = queue.popleft()

                if depth >= max_depth:
                    terminals.append(current_node)
                    continue

                # Find backward relationships based on chain type
                backward_relationships = await self._find_backward_relationships(current_node, graph, chain_type)

                has_parents = False
                for source_node, relationship_type, strength in backward_relationships:
                    if source_node.chunk_id not in visited and strength >= min_link_strength:
                        # Create chain link (source -> current)
                        link = ChainLink(
                            source_component=source_node,
                            target_component=current_node,
                            relationship_type=relationship_type,
                            link_strength=strength,
                            interaction_type=self._determine_interaction_type(relationship_type),
                            evidence_source="graph_traversal",
                            confidence=strength,
                        )

                        links.append(link)
                        visited.add(source_node.chunk_id)
                        queue.append((source_node, depth + 1))
                        has_parents = True

                # If no backward relationships found, this is a terminal
                if not has_parents and depth > 0:
                    terminals.append(current_node)

            return links, terminals

        except Exception as e:
            self.logger.error(f"Error tracing backward chain: {e}")
            return [], []

    async def _find_forward_relationships(
        self,
        node: GraphNode,
        graph: StructureGraph,
        chain_type: ChainType,
    ) -> list[tuple[GraphNode, str, float]]:
        """Find forward relationships for a node based on chain type."""
        relationships = []

        try:
            # Look for relationships in the graph
            for edge in graph.edges:
                if edge.source_breadcrumb == node.breadcrumb:
                    # Find target node
                    target_node = None
                    for graph_node in graph.nodes:
                        if graph_node.breadcrumb == edge.target_breadcrumb:
                            target_node = graph_node
                            break

                    if target_node:
                        # Filter relationships based on chain type
                        if self._is_relevant_relationship(edge.relationship_type, chain_type):
                            strength = edge.weight * edge.confidence
                            relationships.append((target_node, edge.relationship_type, strength))

            # Sort by strength (highest first)
            relationships.sort(key=lambda r: r[2], reverse=True)

            return relationships[:10]  # Limit to top 10 relationships

        except Exception as e:
            self.logger.error(f"Error finding forward relationships: {e}")
            return []

    async def _find_backward_relationships(
        self,
        node: GraphNode,
        graph: StructureGraph,
        chain_type: ChainType,
    ) -> list[tuple[GraphNode, str, float]]:
        """Find backward relationships for a node based on chain type."""
        relationships = []

        try:
            # Look for relationships where this node is the target
            for edge in graph.edges:
                if edge.target_breadcrumb == node.breadcrumb:
                    # Find source node
                    source_node = None
                    for graph_node in graph.nodes:
                        if graph_node.breadcrumb == edge.source_breadcrumb:
                            source_node = graph_node
                            break

                    if source_node:
                        # Filter relationships based on chain type
                        if self._is_relevant_relationship(edge.relationship_type, chain_type):
                            strength = edge.weight * edge.confidence
                            relationships.append((source_node, edge.relationship_type, strength))

            # Sort by strength (highest first)
            relationships.sort(key=lambda r: r[2], reverse=True)

            return relationships[:10]  # Limit to top 10 relationships

        except Exception as e:
            self.logger.error(f"Error finding backward relationships: {e}")
            return []

    def _is_relevant_relationship(self, relationship_type: str, chain_type: ChainType) -> bool:
        """Check if a relationship type is relevant for the chain type."""
        relevance_map = {
            ChainType.EXECUTION_FLOW: ["parent_child", "dependency", "calls"],
            ChainType.DATA_FLOW: ["dependency", "data_flow", "transforms"],
            ChainType.DEPENDENCY_CHAIN: ["dependency", "uses", "imports"],
            ChainType.INHERITANCE_CHAIN: ["parent_child", "inherits", "extends"],
            ChainType.INTERFACE_IMPLEMENTATION: ["implementation", "implements", "realizes"],
            ChainType.SERVICE_LAYER_CHAIN: ["dependency", "uses", "calls"],
            ChainType.API_ENDPOINT_CHAIN: ["parent_child", "dependency", "handles"],
            ChainType.EVENT_HANDLING_CHAIN: ["dependency", "listens", "handles"],
            ChainType.CONFIGURATION_CHAIN: ["dependency", "configures", "uses"],
            ChainType.TEST_COVERAGE_CHAIN: ["tests", "covers", "verifies"],
        }

        relevant_types = relevance_map.get(chain_type, ["parent_child", "dependency"])
        return any(rel_type in relationship_type.lower() for rel_type in relevant_types)

    def _determine_interaction_type(self, relationship_type: str) -> str:
        """Determine interaction type from relationship type."""
        interaction_map = {
            "parent_child": "structural_containment",
            "dependency": "functional_dependency",
            "implementation": "interface_realization",
            "calls": "method_invocation",
            "uses": "component_usage",
            "inherits": "inheritance_relationship",
            "implements": "interface_implementation",
            "configures": "configuration_setting",
            "tests": "test_verification",
        }

        for key, interaction in interaction_map.items():
            if key in relationship_type.lower():
                return interaction

        return "unknown_interaction"

    def _identify_entry_points(
        self,
        graph: StructureGraph,
        scope_breadcrumb: str | None,
    ) -> list[GraphNode]:
        """Identify potential entry points for chain analysis."""
        entry_points = []

        try:
            # Filter nodes by scope if specified
            nodes = graph.nodes
            if scope_breadcrumb:
                nodes = [n for n in nodes if n.breadcrumb and n.breadcrumb.startswith(scope_breadcrumb)]

            # Look for entry point characteristics
            for node in nodes:
                is_entry_point = False

                # Public methods/functions
                if node.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD]:
                    # Check if it's a public interface
                    if not node.name or not node.name.startswith("_"):  # Not private
                        is_entry_point = True

                # API endpoints (functions/methods with specific naming patterns)
                if node.name:
                    api_patterns = ["api", "endpoint", "route", "handler", "controller"]
                    if any(pattern in node.name.lower() for pattern in api_patterns):
                        is_entry_point = True

                # Class constructors
                if node.chunk_type == ChunkType.CLASS:
                    is_entry_point = True

                # Main functions
                if node.name and node.name.lower() in ["main", "__main__", "run", "start"]:
                    is_entry_point = True

                # Service layer components
                if node.name and "service" in node.name.lower():
                    is_entry_point = True

                if is_entry_point:
                    entry_points.append(node)

            # Limit to reasonable number of entry points
            return entry_points[:50]

        except Exception as e:
            self.logger.error(f"Error identifying entry points: {e}")
            return []

    def _calculate_chain_depth(self, links: list[ChainLink], entry_point: GraphNode) -> int:
        """Calculate the maximum depth of the chain."""
        if not links:
            return 0

        try:
            # Build adjacency list
            graph = defaultdict(list)
            for link in links:
                graph[link.source_component.chunk_id].append(link.target_component.chunk_id)

            # BFS to find maximum depth
            max_depth = 0
            queue = deque([(entry_point.chunk_id, 0)])
            visited = set()

            while queue:
                node_id, depth = queue.popleft()
                if node_id in visited:
                    continue

                visited.add(node_id)
                max_depth = max(max_depth, depth)

                for neighbor in graph[node_id]:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

            return max_depth

        except Exception as e:
            self.logger.error(f"Error calculating chain depth: {e}")
            return len(links)  # Fallback to link count

    def _calculate_branch_count(self, links: list[ChainLink], entry_point: GraphNode) -> int:
        """Calculate the number of branches in the chain."""
        if not links:
            return 0

        try:
            # Count outgoing edges for each node
            outgoing_counts = defaultdict(int)
            for link in links:
                outgoing_counts[link.source_component.chunk_id] += 1

            # Count nodes with more than one outgoing edge (branches)
            branches = sum(1 for count in outgoing_counts.values() if count > 1)
            return branches

        except Exception as e:
            self.logger.error(f"Error calculating branch count: {e}")
            return 0

    def _calculate_chain_complexity(self, links: list[ChainLink], terminals: list[GraphNode]) -> float:
        """Calculate complexity score for the chain."""
        if not links:
            return 0.0

        try:
            # Factors for complexity
            link_count_factor = min(1.0, len(links) / 20.0)  # Normalize to 20 links
            terminal_count_factor = min(1.0, len(terminals) / 10.0)  # Normalize to 10 terminals

            # Calculate link strength variance (higher variance = more complexity)
            strengths = [link.link_strength for link in links]
            avg_strength = sum(strengths) / len(strengths)
            strength_variance = sum((s - avg_strength) ** 2 for s in strengths) / len(strengths)
            variance_factor = min(1.0, strength_variance)

            complexity = link_count_factor * 0.4 + terminal_count_factor * 0.3 + variance_factor * 0.3
            return complexity

        except Exception as e:
            self.logger.error(f"Error calculating chain complexity: {e}")
            return 0.5

    def _calculate_chain_completeness(
        self,
        links: list[ChainLink],
        entry_point: GraphNode,
        chain_type: ChainType,
    ) -> float:
        """Calculate completeness score for the chain."""
        if not links:
            return 0.0

        try:
            # Factors for completeness
            completeness_factors = []

            # Factor 1: Link confidence average
            avg_confidence = sum(link.confidence for link in links) / len(links)
            completeness_factors.append(avg_confidence)

            # Factor 2: Evidence quality
            evidence_quality = sum(1.0 for link in links if link.evidence_source) / len(links)
            completeness_factors.append(evidence_quality)

            # Factor 3: Chain connectivity (no isolated components)
            all_components = set()
            connected_components = set()

            for link in links:
                all_components.add(link.source_component.chunk_id)
                all_components.add(link.target_component.chunk_id)
                connected_components.add(link.source_component.chunk_id)
                connected_components.add(link.target_component.chunk_id)

            connectivity_ratio = len(connected_components) / len(all_components) if all_components else 0.0
            completeness_factors.append(connectivity_ratio)

            return sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating chain completeness: {e}")
            return 0.5

    def _calculate_chain_reliability(self, links: list[ChainLink]) -> float:
        """Calculate reliability score for the chain."""
        if not links:
            return 0.0

        try:
            # Average link strength and confidence
            avg_strength = sum(link.link_strength for link in links) / len(links)
            avg_confidence = sum(link.confidence for link in links) / len(links)

            # Reliability is combination of strength and confidence
            reliability = avg_strength * 0.6 + avg_confidence * 0.4
            return reliability

        except Exception as e:
            self.logger.error(f"Error calculating chain reliability: {e}")
            return 0.5

    def _determine_chain_scope(self, links: list[ChainLink]) -> str:
        """Determine the common breadcrumb scope for the chain."""
        if not links:
            return ""

        try:
            # Collect all breadcrumbs
            breadcrumbs = []
            for link in links:
                if link.source_component.breadcrumb:
                    breadcrumbs.append(link.source_component.breadcrumb)
                if link.target_component.breadcrumb:
                    breadcrumbs.append(link.target_component.breadcrumb)

            if not breadcrumbs:
                return ""

            # Find common prefix
            breadcrumb_parts = [bc.split(".") for bc in breadcrumbs]
            min_length = min(len(parts) for parts in breadcrumb_parts)

            common_prefix = []
            for i in range(min_length):
                parts_at_i = [parts[i] for parts in breadcrumb_parts]
                if all(part == parts_at_i[0] for part in parts_at_i):
                    common_prefix.append(parts_at_i[0])
                else:
                    break

            return ".".join(common_prefix)

        except Exception as e:
            self.logger.error(f"Error determining chain scope: {e}")
            return ""

    def _infer_functional_purpose(
        self,
        entry_point: GraphNode,
        links: list[ChainLink],
        chain_type: ChainType,
    ) -> str:
        """Infer the functional purpose of the chain."""
        try:
            purpose_parts = []

            # Add chain type context
            purpose_parts.append(chain_type.value.replace("_", " "))

            # Add entry point context
            if entry_point.name:
                purpose_parts.append(f"from {entry_point.name}")

            # Add complexity context
            if len(links) > 10:
                purpose_parts.append("(complex implementation)")
            elif len(links) > 5:
                purpose_parts.append("(moderate implementation)")
            else:
                purpose_parts.append("(simple implementation)")

            return " ".join(purpose_parts)

        except Exception as e:
            self.logger.error(f"Error inferring functional purpose: {e}")
            return f"{chain_type.value} chain"

    def _calculate_chain_coverage(
        self,
        graph: StructureGraph,
        chains: list[ImplementationChain],
        scope_breadcrumb: str | None,
    ) -> float:
        """Calculate what percentage of components are covered by chains."""
        try:
            # Get all components in scope
            all_components = set()
            for node in graph.nodes:
                if not scope_breadcrumb or (node.breadcrumb and node.breadcrumb.startswith(scope_breadcrumb)):
                    all_components.add(node.chunk_id)

            if not all_components:
                return 0.0

            # Get components covered by chains
            covered_components = set()
            for chain in chains:
                for link in chain.links:
                    covered_components.add(link.source_component.chunk_id)
                    covered_components.add(link.target_component.chunk_id)

            coverage = len(covered_components & all_components) / len(all_components)
            return coverage * 100.0  # Return as percentage

        except Exception as e:
            self.logger.error(f"Error calculating chain coverage: {e}")
            return 0.0

    def _calculate_chain_connectivity(self, chains: list[ImplementationChain]) -> float:
        """Calculate connectivity score between chains."""
        try:
            if len(chains) < 2:
                return 1.0  # Perfect connectivity for single chain

            # Find overlapping components between chains
            chain_components = []
            for chain in chains:
                components = set()
                for link in chain.links:
                    components.add(link.source_component.chunk_id)
                    components.add(link.target_component.chunk_id)
                chain_components.append(components)

            # Calculate pairwise overlaps
            total_comparisons = 0
            total_overlap_score = 0.0

            for i in range(len(chain_components)):
                for j in range(i + 1, len(chain_components)):
                    components1 = chain_components[i]
                    components2 = chain_components[j]

                    overlap = len(components1 & components2)
                    union = len(components1 | components2)

                    if union > 0:
                        overlap_score = overlap / union
                        total_overlap_score += overlap_score

                    total_comparisons += 1

            return total_overlap_score / total_comparisons if total_comparisons > 0 else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating chain connectivity: {e}")
            return 0.0

    def _calculate_chain_similarity(self, chain1: ImplementationChain, chain2: ImplementationChain) -> float:
        """Calculate similarity between two implementation chains."""
        try:
            if chain1.chain_type != chain2.chain_type:
                return 0.0

            similarity_factors = []

            # Factor 1: Structural similarity (depth and branch count)
            depth_similarity = 1.0 - abs(chain1.depth - chain2.depth) / max(chain1.depth, chain2.depth, 1)
            branch_similarity = 1.0 - abs(chain1.branch_count - chain2.branch_count) / max(chain1.branch_count, chain2.branch_count, 1)
            structural_similarity = (depth_similarity + branch_similarity) / 2
            similarity_factors.append(structural_similarity)

            # Factor 2: Component type similarity
            types1 = set(chain1.components_by_type.keys())
            types2 = set(chain2.components_by_type.keys())
            type_overlap = len(types1 & types2) / len(types1 | types2) if (types1 | types2) else 0.0
            similarity_factors.append(type_overlap)

            # Factor 3: Quality similarity
            quality_similarity = 1.0 - abs(chain1.pattern_quality - chain2.pattern_quality) if hasattr(chain1, "pattern_quality") else 0.8
            similarity_factors.append(quality_similarity)

            # Factor 4: Complexity similarity
            complexity_similarity = 1.0 - abs(chain1.complexity_score - chain2.complexity_score)
            similarity_factors.append(complexity_similarity)

            return sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating chain similarity: {e}")
            return 0.0

    def _create_empty_chain(
        self,
        entry_point_breadcrumb: str,
        project_name: str,
        chain_type: ChainType,
    ) -> ImplementationChain:
        """Create an empty implementation chain for error cases."""
        return ImplementationChain(
            chain_id=f"{project_name}_{entry_point_breadcrumb}_{chain_type.value}_empty",
            chain_type=chain_type,
            entry_point=GraphNode(
                chunk_id="unknown",
                breadcrumb=entry_point_breadcrumb,
                name="Unknown",
                chunk_type=ChunkType.FUNCTION,
                file_path="",
            ),
            terminal_points=[],
            links=[],
            depth=0,
            branch_count=0,
            complexity_score=0.0,
            completeness_score=0.0,
            reliability_score=0.0,
            project_name=project_name,
        )


# Factory function for dependency injection
_implementation_chain_service_instance = None


def get_implementation_chain_service(
    graph_rag_service: GraphRAGService = None,
    hybrid_search_service: HybridSearchService = None,
) -> ImplementationChainService:
    """
    Get or create an ImplementationChainService instance.

    Args:
        graph_rag_service: Graph RAG service instance (optional, will be created if not provided)
        hybrid_search_service: Hybrid search service instance (optional, will be created if not provided)

    Returns:
        ImplementationChainService instance
    """
    global _implementation_chain_service_instance

    if _implementation_chain_service_instance is None:
        if graph_rag_service is None:
            from .graph_rag_service import get_graph_rag_service

            graph_rag_service = get_graph_rag_service()

        if hybrid_search_service is None:
            from .hybrid_search_service import get_hybrid_search_service

            hybrid_search_service = get_hybrid_search_service()

        _implementation_chain_service_instance = ImplementationChainService(
            graph_rag_service=graph_rag_service,
            hybrid_search_service=hybrid_search_service,
        )

    return _implementation_chain_service_instance
