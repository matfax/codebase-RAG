"""
Relational Path Data Models for Wave 2.0 Path-Based Indexing System.

This module defines comprehensive data models for various types of relational paths
extracted from code graphs, including execution paths, data flow paths, and dependency paths.
These models support the PathRAG methodology for enhanced code understanding and retrieval.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from .code_chunk import ChunkType


class PathType(Enum):
    """Types of relational paths in code structure."""

    # Execution-related paths
    EXECUTION_PATH = "execution_path"  # Function call chains and execution flows
    CONTROL_FLOW = "control_flow"  # Conditional and loop flow paths
    ASYNC_EXECUTION = "async_execution"  # Asynchronous execution paths

    # Data-related paths
    DATA_FLOW = "data_flow"  # Variable lifecycle and data transformations
    DATA_DEPENDENCY = "data_dependency"  # Data usage and modification dependencies
    STATE_TRANSITION = "state_transition"  # Object state change paths

    # Structural-related paths
    DEPENDENCY_PATH = "dependency_path"  # Import relationships and module dependencies
    INHERITANCE_PATH = "inheritance_path"  # Class inheritance hierarchies
    COMPOSITION_PATH = "composition_path"  # Object composition relationships

    # API-related paths
    API_USAGE = "api_usage"  # External API usage patterns
    INTERFACE_PATH = "interface_path"  # Interface implementation paths

    # Pattern-related paths
    DESIGN_PATTERN = "design_pattern"  # Design pattern implementation paths
    ARCHITECTURAL_PATH = "architectural_path"  # High-level architectural relationships


class PathDirection(Enum):
    """Direction of path traversal."""

    FORWARD = "forward"  # Source to target direction
    BACKWARD = "backward"  # Target to source direction
    BIDIRECTIONAL = "bidirectional"  # Both directions


class PathConfidence(Enum):
    """Confidence levels for path relationships."""

    VERY_HIGH = "very_high"  # 0.9-1.0: Direct, explicit relationships
    HIGH = "high"  # 0.7-0.9: Clear, well-defined relationships
    MEDIUM = "medium"  # 0.5-0.7: Inferred relationships with good evidence
    LOW = "low"  # 0.3-0.5: Weak or speculative relationships
    VERY_LOW = "very_low"  # 0.0-0.3: Highly uncertain relationships


@dataclass
class PathNode:
    """
    Represents a single node in a relational path.

    Each node corresponds to a code element (function, class, variable, etc.)
    and contains metadata about its role in the path.
    """

    # Core identification
    node_id: str  # Unique identifier for this node
    breadcrumb: str  # Hierarchical path identifier
    name: str  # Node name (function name, class name, etc.)
    chunk_type: ChunkType  # Type of code construct
    file_path: str  # Source file path

    # Position information
    line_start: int  # Starting line number
    line_end: int  # Ending line number

    # Path-specific metadata
    role_in_path: str  # Role: "source", "intermediate", "target", "branch_point"
    importance_score: float = 0.0  # Importance within this specific path (0.0-1.0)
    complexity_contribution: float = 0.0  # Contribution to path complexity (0.0-1.0)

    # Context information
    local_context: str | None = None  # Surrounding code context
    semantic_summary: str | None = None  # Brief semantic description

    # Timing and access patterns
    execution_frequency: float = 0.0  # How often this node is executed (estimated)
    access_patterns: list[str] = field(default_factory=list)  # Access patterns for data nodes

    def __post_init__(self):
        """Validate and normalize node data."""
        # Ensure scores are within valid ranges
        self.importance_score = max(0.0, min(1.0, self.importance_score))
        self.complexity_contribution = max(0.0, min(1.0, self.complexity_contribution))
        self.execution_frequency = max(0.0, self.execution_frequency)

        # Validate role
        valid_roles = {"source", "intermediate", "target", "branch_point", "merge_point"}
        if self.role_in_path not in valid_roles:
            self.role_in_path = "intermediate"

    def is_critical_node(self) -> bool:
        """Check if this node is critical to the path."""
        return self.importance_score > 0.7 or self.role_in_path in {"source", "target", "branch_point"}

    def get_execution_weight(self) -> float:
        """Calculate execution weight based on frequency and importance."""
        return (self.execution_frequency * 0.6) + (self.importance_score * 0.4)


@dataclass
class PathEdge:
    """
    Represents a relationship edge between two nodes in a path.

    Contains detailed information about the nature of the relationship
    and its strength/confidence.
    """

    # Core relationship
    source_node_id: str  # Source node identifier
    target_node_id: str  # Target node identifier
    relationship_type: str  # Type of relationship (call, data_flow, dependency, etc.)

    # Relationship metadata
    weight: float = 1.0  # Strength of relationship (0.0-1.0)
    confidence: PathConfidence = PathConfidence.MEDIUM  # Confidence in this relationship
    direction: PathDirection = PathDirection.FORWARD  # Direction of relationship

    # Context information
    edge_context: str | None = None  # Code context where relationship occurs
    call_expression: str | None = None  # For function calls, the actual call expression
    data_transformation: str | None = None  # For data flow, the transformation applied

    # Metadata
    line_number: int | None = None  # Line where relationship occurs
    is_conditional: bool = False  # Whether relationship is conditional
    condition_expression: str | None = None  # Condition for conditional relationships

    # Performance characteristics
    execution_cost: float = 0.0  # Estimated execution cost
    is_bottleneck: bool = False  # Whether this edge is a performance bottleneck

    def __post_init__(self):
        """Validate edge data."""
        self.weight = max(0.0, min(1.0, self.weight))
        self.execution_cost = max(0.0, self.execution_cost)

    def get_confidence_score(self) -> float:
        """Get numeric confidence score."""
        confidence_map = {
            PathConfidence.VERY_HIGH: 0.95,
            PathConfidence.HIGH: 0.8,
            PathConfidence.MEDIUM: 0.65,
            PathConfidence.LOW: 0.4,
            PathConfidence.VERY_LOW: 0.15,
        }
        return confidence_map.get(self.confidence, 0.5)

    def is_reliable(self) -> bool:
        """Check if this edge represents a reliable relationship."""
        return self.confidence in {PathConfidence.HIGH, PathConfidence.VERY_HIGH} and self.weight > 0.5


@dataclass
class ExecutionPath:
    """
    Represents an execution path through code, capturing function call chains
    and control flow relationships.
    """

    # Core identification
    path_id: str  # Unique path identifier
    path_type: PathType = PathType.EXECUTION_PATH

    # Path structure
    nodes: list[PathNode] = field(default_factory=list)
    edges: list[PathEdge] = field(default_factory=list)

    # Path characteristics
    entry_points: list[str] = field(default_factory=list)  # Node IDs of entry points
    exit_points: list[str] = field(default_factory=list)  # Node IDs of exit points
    branch_points: list[str] = field(default_factory=list)  # Node IDs where path branches

    # Execution metadata
    is_async: bool = False  # Whether path contains async operations
    has_loops: bool = False  # Whether path contains loops
    has_exceptions: bool = False  # Whether path has exception handling
    max_depth: int = 0  # Maximum call depth

    # Performance characteristics
    estimated_execution_time: float = 0.0  # Estimated execution time (ms)
    complexity_score: float = 0.0  # Overall path complexity (0.0-1.0)
    criticality_score: float = 0.0  # Business/functional criticality (0.0-1.0)

    # Context and documentation
    semantic_description: str | None = None  # Human-readable path description
    use_cases: list[str] = field(default_factory=list)  # Common use cases
    related_patterns: list[str] = field(default_factory=list)  # Related design patterns

    def __post_init__(self):
        """Validate and compute derived properties."""
        self.complexity_score = max(0.0, min(1.0, self.complexity_score))
        self.criticality_score = max(0.0, min(1.0, self.criticality_score))
        self.estimated_execution_time = max(0.0, self.estimated_execution_time)

        # Update max_depth based on nodes
        if self.nodes:
            # Estimate depth from breadcrumb depth
            depths = [len(node.breadcrumb.split(".")) for node in self.nodes if node.breadcrumb]
            self.max_depth = max(depths) if depths else 0

    def get_path_length(self) -> int:
        """Get the number of nodes in the path."""
        return len(self.nodes)

    def get_critical_nodes(self) -> list[PathNode]:
        """Get nodes that are critical to this execution path."""
        return [node for node in self.nodes if node.is_critical_node()]

    def get_bottleneck_edges(self) -> list[PathEdge]:
        """Get edges that represent performance bottlenecks."""
        return [edge for edge in self.edges if edge.is_bottleneck]

    def estimate_complexity(self) -> float:
        """Estimate path complexity based on structure and characteristics."""
        base_complexity = len(self.nodes) * 0.1

        # Add complexity for special characteristics
        if self.has_loops:
            base_complexity += 0.3
        if self.has_exceptions:
            base_complexity += 0.2
        if self.is_async:
            base_complexity += 0.2

        # Add complexity from branch points
        base_complexity += len(self.branch_points) * 0.15

        # Normalize to 0.0-1.0 range
        return min(1.0, base_complexity)


@dataclass
class DataFlowPath:
    """
    Represents data flow relationships, tracking how data moves and transforms
    through the codebase.
    """

    # Core identification
    path_id: str  # Unique path identifier
    path_type: PathType = PathType.DATA_FLOW

    # Path structure
    nodes: list[PathNode] = field(default_factory=list)
    edges: list[PathEdge] = field(default_factory=list)

    # Data flow characteristics
    data_source: str | None = None  # Original data source (variable, parameter, etc.)
    data_destinations: list[str] = field(default_factory=list)  # Final data destinations
    transformations: list[str] = field(default_factory=list)  # Data transformations applied

    # Data properties
    data_types: list[str] = field(default_factory=list)  # Data types involved
    is_mutable: bool = True  # Whether data can be modified
    has_side_effects: bool = False  # Whether data flow has side effects

    # Lifecycle information
    creation_point: str | None = None  # Where data is created
    modification_points: list[str] = field(default_factory=list)  # Where data is modified
    access_points: list[str] = field(default_factory=list)  # Where data is accessed
    destruction_point: str | None = None  # Where data is destroyed/goes out of scope

    # Quality characteristics
    data_quality_score: float = 1.0  # Estimated data quality (0.0-1.0)
    consistency_score: float = 1.0  # Data consistency across transformations (0.0-1.0)

    def __post_init__(self):
        """Validate data flow properties."""
        self.data_quality_score = max(0.0, min(1.0, self.data_quality_score))
        self.consistency_score = max(0.0, min(1.0, self.consistency_score))

    def get_transformation_count(self) -> int:
        """Get the number of transformations in this data flow."""
        return len(self.transformations)

    def is_complex_flow(self) -> bool:
        """Check if this is a complex data flow."""
        return len(self.transformations) > 3 or len(self.modification_points) > 5 or self.has_side_effects

    def get_data_lifecycle_stages(self) -> list[str]:
        """Get the stages of data lifecycle represented in this path."""
        stages = []
        if self.creation_point:
            stages.append("creation")
        if self.modification_points:
            stages.append("modification")
        if self.access_points:
            stages.append("access")
        if self.destruction_point:
            stages.append("destruction")
        return stages


@dataclass
class DependencyPath:
    """
    Represents dependency relationships between code components,
    including import dependencies and module relationships.
    """

    # Core identification
    path_id: str  # Unique path identifier
    path_type: PathType = PathType.DEPENDENCY_PATH

    # Path structure
    nodes: list[PathNode] = field(default_factory=list)
    edges: list[PathEdge] = field(default_factory=list)

    # Dependency characteristics
    dependency_type: str = "import"  # Type: import, inheritance, composition, etc.
    is_circular: bool = False  # Whether dependency creates a cycle
    is_external: bool = False  # Whether depends on external libraries

    # Dependency metadata
    required_modules: list[str] = field(default_factory=list)  # Required modules/packages
    optional_modules: list[str] = field(default_factory=list)  # Optional dependencies
    version_constraints: dict[str, str] = field(default_factory=dict)  # Version requirements

    # Impact analysis
    stability_score: float = 1.0  # Stability of dependencies (0.0-1.0)
    coupling_strength: float = 0.0  # Strength of coupling (0.0-1.0)
    impact_radius: int = 0  # Number of components affected by changes

    def __post_init__(self):
        """Validate dependency properties."""
        self.stability_score = max(0.0, min(1.0, self.stability_score))
        self.coupling_strength = max(0.0, min(1.0, self.coupling_strength))
        self.impact_radius = max(0, self.impact_radius)

    def is_high_risk(self) -> bool:
        """Check if this dependency represents high risk."""
        return self.is_circular or self.stability_score < 0.5 or self.coupling_strength > 0.7

    def get_external_dependencies(self) -> list[str]:
        """Get list of external dependencies."""
        if self.is_external:
            return self.required_modules + self.optional_modules
        return []


@dataclass
class RelationalPathCollection:
    """
    Collection of related paths that form a cohesive unit for analysis.

    This represents a cluster of paths that are semantically or structurally
    related and should be analyzed together.
    """

    # Core identification
    collection_id: str  # Unique collection identifier
    collection_name: str  # Human-readable name

    # Path collections
    execution_paths: list[ExecutionPath] = field(default_factory=list)
    data_flow_paths: list[DataFlowPath] = field(default_factory=list)
    dependency_paths: list[DependencyPath] = field(default_factory=list)

    # Collection metadata
    primary_entry_points: list[str] = field(default_factory=list)  # Main entry points
    covered_files: set[str] = field(default_factory=set)  # Files covered by paths
    architectural_patterns: list[str] = field(default_factory=list)  # Detected patterns

    # Quality metrics
    coverage_score: float = 0.0  # How well paths cover the target area (0.0-1.0)
    coherence_score: float = 0.0  # How coherent the path collection is (0.0-1.0)
    completeness_score: float = 0.0  # How complete the analysis is (0.0-1.0)

    # Indexing metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    index_version: str = "2.0"  # Version of indexing system used

    def __post_init__(self):
        """Initialize and validate collection properties."""
        self.coverage_score = max(0.0, min(1.0, self.coverage_score))
        self.coherence_score = max(0.0, min(1.0, self.coherence_score))
        self.completeness_score = max(0.0, min(1.0, self.completeness_score))

        # Update covered files from paths
        for path in self.execution_paths + self.data_flow_paths + self.dependency_paths:
            for node in path.nodes:
                self.covered_files.add(node.file_path)

    def get_total_path_count(self) -> int:
        """Get total number of paths in collection."""
        return len(self.execution_paths) + len(self.data_flow_paths) + len(self.dependency_paths)

    def get_high_value_paths(self, threshold: float = 0.7) -> list[ExecutionPath | DataFlowPath | DependencyPath]:
        """Get paths with high importance/criticality scores."""
        high_value = []

        for path in self.execution_paths:
            if path.criticality_score >= threshold:
                high_value.append(path)

        for path in self.data_flow_paths:
            if path.data_quality_score >= threshold:
                high_value.append(path)

        for path in self.dependency_paths:
            if path.stability_score >= threshold:
                high_value.append(path)

        return high_value

    def get_complexity_distribution(self) -> dict[str, float]:
        """Get distribution of complexity scores across paths."""
        complexities = []

        for path in self.execution_paths:
            complexities.append(path.complexity_score)

        if not complexities:
            return {"mean": 0.0, "min": 0.0, "max": 0.0}

        return {"mean": sum(complexities) / len(complexities), "min": min(complexities), "max": max(complexities)}

    def to_dict(self) -> dict[str, Any]:
        """Convert collection to dictionary for serialization."""
        return {
            "collection_id": self.collection_id,
            "collection_name": self.collection_name,
            "execution_paths_count": len(self.execution_paths),
            "data_flow_paths_count": len(self.data_flow_paths),
            "dependency_paths_count": len(self.dependency_paths),
            "total_paths": self.get_total_path_count(),
            "primary_entry_points": self.primary_entry_points,
            "covered_files": list(self.covered_files),
            "architectural_patterns": self.architectural_patterns,
            "coverage_score": self.coverage_score,
            "coherence_score": self.coherence_score,
            "completeness_score": self.completeness_score,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "index_version": self.index_version,
            "complexity_distribution": self.get_complexity_distribution(),
        }


@dataclass
class PathExtractionResult:
    """
    Result of path extraction operation, containing extracted paths
    and metadata about the extraction process.
    """

    # Extraction results
    path_collection: RelationalPathCollection
    extraction_stats: dict[str, Any] = field(default_factory=dict)

    # Processing metadata
    processing_time_ms: float = 0.0  # Time taken for extraction
    source_chunks_count: int = 0  # Number of source chunks processed
    success_rate: float = 1.0  # Success rate of extraction (0.0-1.0)

    # Quality indicators
    paths_with_high_confidence: int = 0  # Number of high-confidence paths
    paths_requiring_review: int = 0  # Number of paths requiring manual review
    extraction_warnings: list[str] = field(default_factory=list)  # Warnings during extraction

    def __post_init__(self):
        """Validate extraction result."""
        self.processing_time_ms = max(0.0, self.processing_time_ms)
        self.source_chunks_count = max(0, self.source_chunks_count)
        self.success_rate = max(0.0, min(1.0, self.success_rate))
        self.paths_with_high_confidence = max(0, self.paths_with_high_confidence)
        self.paths_requiring_review = max(0, self.paths_requiring_review)

    def is_successful(self) -> bool:
        """Check if extraction was successful."""
        return self.success_rate > 0.8 and len(self.extraction_warnings) < 5

    def get_extraction_efficiency(self) -> float:
        """Calculate extraction efficiency (paths per second)."""
        if self.processing_time_ms <= 0:
            return 0.0

        total_paths = self.path_collection.get_total_path_count()
        return (total_paths * 1000.0) / self.processing_time_ms  # paths per second


# Type aliases for convenience
AnyPath = ExecutionPath | DataFlowPath | DependencyPath
PathId = str
NodeId = str
