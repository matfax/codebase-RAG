"""
Path-Based Indexer for Wave 2.0 Task 2.2 - Relational Path Extraction Algorithms.

This service implements sophisticated algorithms to extract various types of relational
paths from code graphs, including execution paths, data flow paths, and dependency paths.
It leverages the Wave 1.0 foundation and integrates with the relational path data models.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models.code_chunk import ChunkType, CodeChunk
from ..models.relational_path import (
    AnyPath,
    DataFlowPath,
    DependencyPath,
    ExecutionPath,
    PathConfidence,
    PathDirection,
    PathEdge,
    PathExtractionResult,
    PathNode,
    PathType,
    RelationalPathCollection,
)
from .graph_rag_service import GraphRAGService
from .lightweight_graph_service import LightweightGraphService, NodeMetadata
from .structure_relationship_builder import GraphEdge, GraphNode, StructureGraph


@dataclass
class ExtractionOptions:
    """Configuration options for path extraction."""
    
    # General extraction settings
    max_path_length: int = 50               # Maximum nodes per path
    max_paths_per_type: int = 100           # Maximum paths to extract per type
    min_confidence_threshold: float = 0.3   # Minimum confidence for path inclusion
    
    # Path type selection
    enable_execution_paths: bool = True     # Extract execution paths
    enable_data_flow_paths: bool = True     # Extract data flow paths
    enable_dependency_paths: bool = True    # Extract dependency paths
    
    # Filtering options
    filter_trivial_paths: bool = True       # Skip single-node paths
    filter_duplicate_paths: bool = True     # Remove duplicate path patterns
    prioritize_critical_paths: bool = True  # Prioritize high-importance paths
    
    # Performance settings
    max_extraction_time_ms: float = 30000   # 30 second timeout
    enable_parallel_extraction: bool = True # Extract different path types in parallel
    cache_intermediate_results: bool = True # Cache path components
    
    # Quality settings
    require_high_confidence_nodes: bool = False  # Only include high-confidence nodes
    validate_path_connectivity: bool = True      # Ensure paths are properly connected
    include_context_information: bool = True     # Add context to path nodes


@dataclass
class ExtractionContext:
    """Context information for path extraction process."""
    
    project_name: str
    start_time: float = field(default_factory=time.time)
    processed_nodes: Set[str] = field(default_factory=set)
    processed_edges: Set[Tuple[str, str]] = field(default_factory=set)
    path_cache: Dict[str, List[AnyPath]] = field(default_factory=dict)
    extraction_warnings: List[str] = field(default_factory=list)
    statistics: Dict[str, int] = field(default_factory=lambda: {
        "nodes_processed": 0,
        "edges_processed": 0,
        "paths_extracted": 0,
        "paths_filtered": 0
    })


class PathBasedIndexer:
    """
    Advanced path extraction service that implements sophisticated algorithms
    to extract various types of relational paths from code structure graphs.
    
    This service builds on Wave 1.0's foundation and implements the PathRAG
    methodology for enhanced code understanding and retrieval.
    """
    
    def __init__(
        self,
        lightweight_graph_service: LightweightGraphService,
        graph_rag_service: Optional[GraphRAGService] = None
    ):
        """
        Initialize the path-based indexer.
        
        Args:
            lightweight_graph_service: Wave 1.0 lightweight graph service
            graph_rag_service: Optional Graph RAG service for advanced operations
        """
        self.logger = logging.getLogger(__name__)
        self.lightweight_graph = lightweight_graph_service
        self.graph_rag_service = graph_rag_service
        
        # Path extraction algorithms
        self._execution_extractor = ExecutionPathExtractor()
        self._data_flow_extractor = DataFlowPathExtractor()
        self._dependency_extractor = DependencyPathExtractor()
        
        # Performance monitoring
        self._extraction_stats = {
            "total_extractions": 0,
            "successful_extractions": 0,
            "failed_extractions": 0,
            "average_extraction_time_ms": 0.0,
            "total_paths_extracted": 0
        }
    
    async def extract_relational_paths(
        self,
        project_name: str,
        entry_points: Optional[List[str]] = None,
        options: Optional[ExtractionOptions] = None
    ) -> PathExtractionResult:
        """
        Extract comprehensive relational paths from a project's code graph.
        
        Args:
            project_name: Name of the project to extract paths from
            entry_points: Optional list of entry point breadcrumbs to focus on
            options: Extraction configuration options
            
        Returns:
            PathExtractionResult containing extracted paths and metadata
        """
        start_time = time.time()
        options = options or ExtractionOptions()
        
        try:
            self.logger.info(f"Starting path extraction for project: {project_name}")
            
            # Initialize extraction context
            context = ExtractionContext(project_name=project_name)
            
            # Get project structure graph
            structure_graph = await self._get_project_structure_graph(project_name)
            if not structure_graph:
                return self._create_empty_result(project_name, "No structure graph available")
            
            # Determine extraction scope
            target_nodes = await self._determine_extraction_scope(
                structure_graph, entry_points, options
            )
            
            self.logger.info(f"Extracting paths from {len(target_nodes)} nodes")
            
            # Extract different types of paths
            extraction_tasks = []
            
            if options.enable_execution_paths:
                extraction_tasks.append(
                    self._extract_execution_paths(structure_graph, target_nodes, options, context)
                )
            
            if options.enable_data_flow_paths:
                extraction_tasks.append(
                    self._extract_data_flow_paths(structure_graph, target_nodes, options, context)
                )
            
            if options.enable_dependency_paths:
                extraction_tasks.append(
                    self._extract_dependency_paths(structure_graph, target_nodes, options, context)
                )
            
            # Execute extractions (parallel if enabled)
            if options.enable_parallel_extraction and len(extraction_tasks) > 1:
                path_results = await asyncio.gather(*extraction_tasks, return_exceptions=True)
            else:
                path_results = []
                for task in extraction_tasks:
                    result = await task
                    path_results.append(result)
            
            # Process results and handle any exceptions
            execution_paths, data_flow_paths, dependency_paths = [], [], []
            
            for i, result in enumerate(path_results):
                if isinstance(result, Exception):
                    context.extraction_warnings.append(f"Task {i} failed: {str(result)}")
                    continue
                
                if i == 0 and options.enable_execution_paths:
                    execution_paths = result
                elif i == 1 and options.enable_data_flow_paths:
                    data_flow_paths = result
                elif i == 2 and options.enable_dependency_paths:
                    dependency_paths = result
            
            # Create path collection
            collection = RelationalPathCollection(
                collection_id=f"{project_name}_{uuid.uuid4().hex[:8]}",
                collection_name=f"Path Collection for {project_name}",
                execution_paths=execution_paths,
                data_flow_paths=data_flow_paths,
                dependency_paths=dependency_paths,
                primary_entry_points=entry_points or [],
                coverage_score=self._calculate_coverage_score(target_nodes, execution_paths + data_flow_paths + dependency_paths),
                coherence_score=self._calculate_coherence_score(execution_paths + data_flow_paths + dependency_paths),
                completeness_score=min(1.0, len(execution_paths + data_flow_paths + dependency_paths) / max(1, len(target_nodes) * 0.5))
            )
            
            # Create extraction result
            processing_time_ms = (time.time() - start_time) * 1000
            result = PathExtractionResult(
                path_collection=collection,
                processing_time_ms=processing_time_ms,
                source_chunks_count=len(target_nodes),
                success_rate=1.0 - (len(context.extraction_warnings) / max(1, len(extraction_tasks))),
                paths_with_high_confidence=sum(1 for path in collection.execution_paths + collection.data_flow_paths + collection.dependency_paths
                                               if self._is_high_confidence_path(path)),
                paths_requiring_review=len(context.extraction_warnings),
                extraction_warnings=context.extraction_warnings,
                extraction_stats=dict(context.statistics)
            )
            
            self._update_performance_stats(processing_time_ms, result.path_collection.get_total_path_count(), True)
            
            self.logger.info(
                f"Path extraction completed: {result.path_collection.get_total_path_count()} paths "
                f"in {processing_time_ms:.2f}ms"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Path extraction failed for {project_name}: {str(e)}")
            self._update_performance_stats((time.time() - start_time) * 1000, 0, False)
            return self._create_empty_result(project_name, f"Extraction failed: {str(e)}")
    
    async def _get_project_structure_graph(self, project_name: str) -> Optional[StructureGraph]:
        """Get or build the structure graph for a project."""
        try:
            # Try to get graph from lightweight graph service
            if hasattr(self.lightweight_graph, 'get_project_graph'):
                return await self.lightweight_graph.get_project_graph(project_name)
            
            # Fallback to building graph from chunks if Graph RAG service is available
            if self.graph_rag_service:
                return await self.graph_rag_service.build_structure_graph(project_name)
            
            self.logger.warning(f"No graph service available for project: {project_name}")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get structure graph for {project_name}: {str(e)}")
            return None
    
    async def _determine_extraction_scope(
        self,
        graph: StructureGraph,
        entry_points: Optional[List[str]],
        options: ExtractionOptions
    ) -> List[GraphNode]:
        """Determine which nodes to include in path extraction."""
        if entry_points:
            # Focus on entry points and their neighborhoods
            target_nodes = []
            for entry_point in entry_points:
                if entry_point in graph.nodes:
                    target_nodes.append(graph.nodes[entry_point])
                    # Add immediate neighbors
                    for edge in graph.edges:
                        if edge.source_breadcrumb == entry_point and edge.target_breadcrumb in graph.nodes:
                            target_nodes.append(graph.nodes[edge.target_breadcrumb])
                        elif edge.target_breadcrumb == entry_point and edge.source_breadcrumb in graph.nodes:
                            target_nodes.append(graph.nodes[edge.source_breadcrumb])
            
            return list(set(target_nodes))  # Remove duplicates
        else:
            # Use all nodes, but prioritize important ones
            all_nodes = list(graph.nodes.values())
            
            if options.prioritize_critical_paths:
                # Sort by semantic weight (importance)
                all_nodes.sort(key=lambda n: n.semantic_weight, reverse=True)
                # Limit to manageable number
                return all_nodes[:min(len(all_nodes), options.max_paths_per_type * 2)]
            
            return all_nodes
    
    async def _extract_execution_paths(
        self,
        graph: StructureGraph,
        target_nodes: List[GraphNode],
        options: ExtractionOptions,
        context: ExtractionContext
    ) -> List[ExecutionPath]:
        """Extract execution paths from the structure graph."""
        self.logger.debug("Extracting execution paths")
        
        execution_paths = []
        function_nodes = [n for n in target_nodes if n.chunk_type in {ChunkType.FUNCTION, ChunkType.METHOD}]
        
        for node in function_nodes[:options.max_paths_per_type]:
            try:
                paths = await self._execution_extractor.extract_from_node(node, graph, options, context)
                execution_paths.extend(paths)
                context.statistics["paths_extracted"] += len(paths)
                
            except Exception as e:
                context.extraction_warnings.append(f"Failed to extract execution path from {node.breadcrumb}: {str(e)}")
        
        # Apply filtering
        if options.filter_duplicate_paths:
            execution_paths = self._filter_duplicate_execution_paths(execution_paths)
        
        if options.filter_trivial_paths:
            execution_paths = [p for p in execution_paths if len(p.nodes) > 1]
        
        context.statistics["paths_filtered"] += len([p for p in execution_paths if len(p.nodes) <= 1])
        
        return execution_paths[:options.max_paths_per_type]
    
    async def _extract_data_flow_paths(
        self,
        graph: StructureGraph,
        target_nodes: List[GraphNode],
        options: ExtractionOptions,
        context: ExtractionContext
    ) -> List[DataFlowPath]:
        """Extract data flow paths from the structure graph."""
        self.logger.debug("Extracting data flow paths")
        
        data_flow_paths = []
        variable_nodes = [n for n in target_nodes if n.chunk_type in {ChunkType.VARIABLE, ChunkType.PROPERTY}]
        
        for node in variable_nodes[:options.max_paths_per_type]:
            try:
                paths = await self._data_flow_extractor.extract_from_node(node, graph, options, context)
                data_flow_paths.extend(paths)
                context.statistics["paths_extracted"] += len(paths)
                
            except Exception as e:
                context.extraction_warnings.append(f"Failed to extract data flow path from {node.breadcrumb}: {str(e)}")
        
        # Apply filtering
        if options.filter_duplicate_paths:
            data_flow_paths = self._filter_duplicate_data_flow_paths(data_flow_paths)
        
        if options.filter_trivial_paths:
            data_flow_paths = [p for p in data_flow_paths if len(p.transformations) > 0]
        
        return data_flow_paths[:options.max_paths_per_type]
    
    async def _extract_dependency_paths(
        self,
        graph: StructureGraph,
        target_nodes: List[GraphNode],
        options: ExtractionOptions,
        context: ExtractionContext
    ) -> List[DependencyPath]:
        """Extract dependency paths from the structure graph."""
        self.logger.debug("Extracting dependency paths")
        
        dependency_paths = []
        import_nodes = [n for n in target_nodes if n.chunk_type in {ChunkType.IMPORT, ChunkType.MODULE_DOCSTRING}]
        
        for node in import_nodes[:options.max_paths_per_type]:
            try:
                paths = await self._dependency_extractor.extract_from_node(node, graph, options, context)
                dependency_paths.extend(paths)
                context.statistics["paths_extracted"] += len(paths)
                
            except Exception as e:
                context.extraction_warnings.append(f"Failed to extract dependency path from {node.breadcrumb}: {str(e)}")
        
        # Apply filtering
        if options.filter_duplicate_paths:
            dependency_paths = self._filter_duplicate_dependency_paths(dependency_paths)
        
        return dependency_paths[:options.max_paths_per_type]
    
    def _filter_duplicate_execution_paths(self, paths: List[ExecutionPath]) -> List[ExecutionPath]:
        """Remove duplicate execution paths based on node sequences."""
        seen_signatures = set()
        filtered_paths = []
        
        for path in paths:
            # Create signature from breadcrumb sequence
            signature = tuple(node.breadcrumb for node in path.nodes)
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                filtered_paths.append(path)
        
        return filtered_paths
    
    def _filter_duplicate_data_flow_paths(self, paths: List[DataFlowPath]) -> List[DataFlowPath]:
        """Remove duplicate data flow paths based on transformation sequences."""
        seen_signatures = set()
        filtered_paths = []
        
        for path in paths:
            # Create signature from data source and transformation sequence
            signature = (path.data_source, tuple(path.transformations))
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                filtered_paths.append(path)
        
        return filtered_paths
    
    def _filter_duplicate_dependency_paths(self, paths: List[DependencyPath]) -> List[DependencyPath]:
        """Remove duplicate dependency paths based on module sequences."""
        seen_signatures = set()
        filtered_paths = []
        
        for path in paths:
            # Create signature from required modules
            signature = tuple(sorted(path.required_modules))
            if signature not in seen_signatures:
                seen_signatures.add(signature)
                filtered_paths.append(path)
        
        return filtered_paths
    
    def _calculate_coverage_score(self, target_nodes: List[GraphNode], paths: List[AnyPath]) -> float:
        """Calculate how well the extracted paths cover the target nodes."""
        if not target_nodes:
            return 0.0
        
        covered_breadcrumbs = set()
        for path in paths:
            for node in path.nodes:
                covered_breadcrumbs.add(node.breadcrumb)
        
        target_breadcrumbs = {node.breadcrumb for node in target_nodes}
        coverage = len(covered_breadcrumbs.intersection(target_breadcrumbs)) / len(target_breadcrumbs)
        
        return min(1.0, coverage)
    
    def _calculate_coherence_score(self, paths: List[AnyPath]) -> float:
        """Calculate coherence score based on path connectivity and relationships."""
        if not paths:
            return 0.0
        
        # Simple coherence calculation based on shared nodes/breadcrumbs
        all_breadcrumbs = []
        for path in paths:
            for node in path.nodes:
                all_breadcrumbs.append(node.breadcrumb)
        
        unique_breadcrumbs = set(all_breadcrumbs)
        if not unique_breadcrumbs:
            return 0.0
        
        # Higher coherence if paths share more nodes
        shared_ratio = (len(all_breadcrumbs) - len(unique_breadcrumbs)) / len(all_breadcrumbs)
        return min(1.0, shared_ratio + 0.5)  # Ensure minimum coherence
    
    def _is_high_confidence_path(self, path: AnyPath) -> bool:
        """Check if a path has high confidence based on its edges."""
        if not path.edges:
            return False
        
        high_confidence_edges = sum(
            1 for edge in path.edges 
            if edge.confidence in {PathConfidence.HIGH, PathConfidence.VERY_HIGH}
        )
        
        return high_confidence_edges / len(path.edges) > 0.7
    
    def _create_empty_result(self, project_name: str, error_message: str) -> PathExtractionResult:
        """Create an empty extraction result for error cases."""
        collection = RelationalPathCollection(
            collection_id=f"{project_name}_empty",
            collection_name=f"Empty Collection for {project_name}"
        )
        
        return PathExtractionResult(
            path_collection=collection,
            processing_time_ms=0.0,
            source_chunks_count=0,
            success_rate=0.0,
            extraction_warnings=[error_message]
        )
    
    def _update_performance_stats(self, processing_time_ms: float, paths_extracted: int, success: bool):
        """Update internal performance statistics."""
        self._extraction_stats["total_extractions"] += 1
        
        if success:
            self._extraction_stats["successful_extractions"] += 1
        else:
            self._extraction_stats["failed_extractions"] += 1
        
        # Update average extraction time
        current_avg = self._extraction_stats["average_extraction_time_ms"]
        total_extractions = self._extraction_stats["total_extractions"]
        self._extraction_stats["average_extraction_time_ms"] = (
            (current_avg * (total_extractions - 1) + processing_time_ms) / total_extractions
        )
        
        self._extraction_stats["total_paths_extracted"] += paths_extracted
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics."""
        return dict(self._extraction_stats)


class ExecutionPathExtractor:
    """Specialized extractor for execution paths (function call chains)."""
    
    async def extract_from_node(
        self,
        start_node: GraphNode,
        graph: StructureGraph,
        options: ExtractionOptions,
        context: ExtractionContext
    ) -> List[ExecutionPath]:
        """Extract execution paths starting from a given node."""
        paths = []
        
        # Find function call edges from this node
        call_edges = [
            edge for edge in graph.edges
            if edge.source_breadcrumb == start_node.breadcrumb and edge.is_function_call()
        ]
        
        for edge in call_edges:
            if edge.target_breadcrumb not in graph.nodes:
                continue
                
            path = await self._build_execution_path(start_node, edge, graph, options, context)
            if path:
                paths.append(path)
        
        return paths
    
    async def _build_execution_path(
        self,
        start_node: GraphNode,
        initial_edge: GraphEdge,
        graph: StructureGraph,
        options: ExtractionOptions,
        context: ExtractionContext
    ) -> Optional[ExecutionPath]:
        """Build a single execution path following function calls."""
        try:
            path_nodes = []
            path_edges = []
            visited = set()
            
            # Add starting node
            start_path_node = self._convert_to_path_node(start_node, "source")
            path_nodes.append(start_path_node)
            visited.add(start_node.breadcrumb)
            
            # Follow the execution chain
            current_edge = initial_edge
            depth = 0
            
            while current_edge and depth < options.max_path_length:
                # Add edge to path
                path_edge = self._convert_to_path_edge(current_edge)
                path_edges.append(path_edge)
                
                # Add target node
                target_node = graph.nodes.get(current_edge.target_breadcrumb)
                if not target_node or target_node.breadcrumb in visited:
                    break
                
                role = "target" if depth == options.max_path_length - 1 else "intermediate"
                target_path_node = self._convert_to_path_node(target_node, role)
                path_nodes.append(target_path_node)
                visited.add(target_node.breadcrumb)
                
                # Find next function call edge
                next_edges = [
                    edge for edge in graph.edges
                    if (edge.source_breadcrumb == target_node.breadcrumb and 
                        edge.is_function_call() and 
                        edge.target_breadcrumb not in visited)
                ]
                
                current_edge = next_edges[0] if next_edges else None
                depth += 1
            
            # Create execution path
            if len(path_nodes) > 1:  # Only create path if it has multiple nodes
                execution_path = ExecutionPath(
                    path_id=f"exec_{uuid.uuid4().hex[:8]}",
                    nodes=path_nodes,
                    edges=path_edges,
                    entry_points=[start_path_node.node_id],
                    exit_points=[path_nodes[-1].node_id],
                    max_depth=depth,
                    complexity_score=min(1.0, depth * 0.1 + len(path_edges) * 0.05),
                    criticality_score=start_path_node.importance_score
                )
                
                return execution_path
            
            return None
            
        except Exception as e:
            context.extraction_warnings.append(f"Failed to build execution path: {str(e)}")
            return None
    
    def _convert_to_path_node(self, graph_node: GraphNode, role: str) -> PathNode:
        """Convert a GraphNode to a PathNode."""
        return PathNode(
            node_id=f"node_{uuid.uuid4().hex[:8]}",
            breadcrumb=graph_node.breadcrumb,
            name=graph_node.name,
            chunk_type=graph_node.chunk_type,
            file_path=graph_node.file_path,
            line_start=0,  # Would need to get from original chunk
            line_end=0,    # Would need to get from original chunk
            role_in_path=role,
            importance_score=graph_node.semantic_weight
        )
    
    def _convert_to_path_edge(self, graph_edge: GraphEdge) -> PathEdge:
        """Convert a GraphEdge to a PathEdge."""
        # Map confidence from graph edge weight
        if graph_edge.confidence >= 0.8:
            confidence = PathConfidence.VERY_HIGH
        elif graph_edge.confidence >= 0.6:
            confidence = PathConfidence.HIGH
        elif graph_edge.confidence >= 0.4:
            confidence = PathConfidence.MEDIUM
        elif graph_edge.confidence >= 0.2:
            confidence = PathConfidence.LOW
        else:
            confidence = PathConfidence.VERY_LOW
        
        return PathEdge(
            source_node_id=graph_edge.source_breadcrumb,
            target_node_id=graph_edge.target_breadcrumb,
            relationship_type=graph_edge.relationship_type,
            weight=graph_edge.weight,
            confidence=confidence,
            call_expression=graph_edge.get_call_expression(),
            line_number=graph_edge.get_line_number()
        )


class DataFlowPathExtractor:
    """Specialized extractor for data flow paths (variable lifecycle)."""
    
    async def extract_from_node(
        self,
        start_node: GraphNode,
        graph: StructureGraph,
        options: ExtractionOptions,
        context: ExtractionContext
    ) -> List[DataFlowPath]:
        """Extract data flow paths starting from a variable node."""
        paths = []
        
        # For now, create a simple data flow path from variable usage
        # In a more sophisticated implementation, this would trace actual data transformations
        
        try:
            path_nodes = [PathNode(
                node_id=f"data_node_{uuid.uuid4().hex[:8]}",
                breadcrumb=start_node.breadcrumb,
                name=start_node.name,
                chunk_type=start_node.chunk_type,
                file_path=start_node.file_path,
                line_start=0,
                line_end=0,
                role_in_path="source",
                importance_score=start_node.semantic_weight
            )]
            
            data_flow_path = DataFlowPath(
                path_id=f"data_{uuid.uuid4().hex[:8]}",
                nodes=path_nodes,
                data_source=start_node.name,
                data_types=["unknown"],
                creation_point=start_node.breadcrumb,
                data_quality_score=start_node.semantic_weight
            )
            
            paths.append(data_flow_path)
            
        except Exception as e:
            context.extraction_warnings.append(f"Failed to extract data flow path: {str(e)}")
        
        return paths


class DependencyPathExtractor:
    """Specialized extractor for dependency paths (import relationships)."""
    
    async def extract_from_node(
        self,
        start_node: GraphNode,
        graph: StructureGraph,
        options: ExtractionOptions,
        context: ExtractionContext
    ) -> List[DependencyPath]:
        """Extract dependency paths starting from an import node."""
        paths = []
        
        try:
            # Find dependency edges from this node
            dep_edges = [
                edge for edge in graph.edges
                if edge.source_breadcrumb == start_node.breadcrumb and edge.is_dependency()
            ]
            
            if dep_edges:
                # Create dependency path
                path_nodes = [PathNode(
                    node_id=f"dep_node_{uuid.uuid4().hex[:8]}",
                    breadcrumb=start_node.breadcrumb,
                    name=start_node.name,
                    chunk_type=start_node.chunk_type,
                    file_path=start_node.file_path,
                    line_start=0,
                    line_end=0,
                    role_in_path="source",
                    importance_score=start_node.semantic_weight
                )]
                
                # Add target nodes
                for edge in dep_edges:
                    if edge.target_breadcrumb in graph.nodes:
                        target_node = graph.nodes[edge.target_breadcrumb]
                        path_nodes.append(PathNode(
                            node_id=f"dep_node_{uuid.uuid4().hex[:8]}",
                            breadcrumb=target_node.breadcrumb,
                            name=target_node.name,
                            chunk_type=target_node.chunk_type,
                            file_path=target_node.file_path,
                            line_start=0,
                            line_end=0,
                            role_in_path="target",
                            importance_score=target_node.semantic_weight
                        ))
                
                dependency_path = DependencyPath(
                    path_id=f"dep_{uuid.uuid4().hex[:8]}",
                    nodes=path_nodes,
                    dependency_type="import",
                    required_modules=[start_node.name],
                    stability_score=start_node.semantic_weight,
                    coupling_strength=min(1.0, len(dep_edges) * 0.2)
                )
                
                paths.append(dependency_path)
        
        except Exception as e:
            context.extraction_warnings.append(f"Failed to extract dependency path: {str(e)}")
        
        return paths


# Service factory function
def create_path_based_indexer(
    lightweight_graph_service: LightweightGraphService,
    graph_rag_service: Optional[GraphRAGService] = None
) -> PathBasedIndexer:
    """
    Factory function to create a PathBasedIndexer instance.
    
    Args:
        lightweight_graph_service: Wave 1.0 lightweight graph service
        graph_rag_service: Optional Graph RAG service for advanced operations
        
    Returns:
        Configured PathBasedIndexer instance
    """
    return PathBasedIndexer(lightweight_graph_service, graph_rag_service)