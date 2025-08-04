"""
Unit tests for relational path data models.

This module tests the comprehensive path data models including ExecutionPath,
DataFlowPath, DependencyPath, and RelationalPathCollection.
"""

from datetime import datetime

import pytest

from .code_chunk import ChunkType
from .relational_path import (
    AnyPath,
    DataFlowPath,
    DependencyPath,
    ExecutionPath,
    NodeId,
    PathConfidence,
    PathDirection,
    PathEdge,
    PathExtractionResult,
    PathId,
    PathNode,
    PathType,
    RelationalPathCollection,
)


class TestPathNode:
    """Test PathNode data model."""

    def test_path_node_creation(self):
        """Test basic PathNode creation."""
        node = PathNode(
            node_id="node_1",
            breadcrumb="module.class.method",
            name="method",
            chunk_type=ChunkType.METHOD,
            file_path="/test/file.py",
            line_start=10,
            line_end=20,
            role_in_path="source",
            importance_score=0.8,
        )

        assert node.node_id == "node_1"
        assert node.breadcrumb == "module.class.method"
        assert node.name == "method"
        assert node.chunk_type == ChunkType.METHOD
        assert node.role_in_path == "source"
        assert node.importance_score == 0.8
        assert node.is_critical_node()  # High importance and source role

    def test_path_node_validation(self):
        """Test PathNode validation and normalization."""
        node = PathNode(
            node_id="node_1",
            breadcrumb="test.function",
            name="function",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            line_start=1,
            line_end=10,
            role_in_path="invalid_role",  # Invalid role
            importance_score=1.5,  # Out of range
            complexity_contribution=-0.5,  # Out of range
        )

        assert node.role_in_path == "intermediate"  # Normalized to valid role
        assert node.importance_score == 1.0  # Clamped to valid range
        assert node.complexity_contribution == 0.0  # Clamped to valid range

    def test_path_node_critical_detection(self):
        """Test critical node detection."""
        # High importance node
        high_importance = PathNode(
            node_id="node_1",
            breadcrumb="test",
            name="test",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            line_start=1,
            line_end=10,
            role_in_path="intermediate",
            importance_score=0.8,
        )
        assert high_importance.is_critical_node()

        # Critical role node
        critical_role = PathNode(
            node_id="node_2",
            breadcrumb="test",
            name="test",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            line_start=1,
            line_end=10,
            role_in_path="branch_point",
            importance_score=0.3,
        )
        assert critical_role.is_critical_node()

        # Non-critical node
        normal_node = PathNode(
            node_id="node_3",
            breadcrumb="test",
            name="test",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            line_start=1,
            line_end=10,
            role_in_path="intermediate",
            importance_score=0.3,
        )
        assert not normal_node.is_critical_node()

    def test_execution_weight_calculation(self):
        """Test execution weight calculation."""
        node = PathNode(
            node_id="node_1",
            breadcrumb="test",
            name="test",
            chunk_type=ChunkType.FUNCTION,
            file_path="/test.py",
            line_start=1,
            line_end=10,
            role_in_path="intermediate",
            execution_frequency=0.8,
            importance_score=0.6,
        )

        # Weight = (frequency * 0.6) + (importance * 0.4) = (0.8 * 0.6) + (0.6 * 0.4) = 0.48 + 0.24 = 0.72
        expected_weight = 0.72
        assert abs(node.get_execution_weight() - expected_weight) < 0.01


class TestPathEdge:
    """Test PathEdge data model."""

    def test_path_edge_creation(self):
        """Test basic PathEdge creation."""
        edge = PathEdge(
            source_node_id="node_1",
            target_node_id="node_2",
            relationship_type="function_call",
            weight=0.8,
            confidence=PathConfidence.HIGH,
            direction=PathDirection.FORWARD,
        )

        assert edge.source_node_id == "node_1"
        assert edge.target_node_id == "node_2"
        assert edge.relationship_type == "function_call"
        assert edge.weight == 0.8
        assert edge.confidence == PathConfidence.HIGH

    def test_path_edge_validation(self):
        """Test PathEdge validation."""
        edge = PathEdge(
            source_node_id="node_1",
            target_node_id="node_2",
            relationship_type="call",  # Valid relationship type
            weight=1.5,  # Out of range
            confidence=PathConfidence.HIGH,
        )

        assert edge.weight == 1.0  # Clamped to valid range

    def test_confidence_score_mapping(self):
        """Test confidence score mapping."""
        edge = PathEdge(source_node_id="node_1", target_node_id="node_2", relationship_type="call", confidence=PathConfidence.VERY_HIGH)

        assert edge.get_confidence_score() == 0.95

        edge.confidence = PathConfidence.LOW
        assert edge.get_confidence_score() == 0.4

    def test_reliability_check(self):
        """Test reliability check."""
        # Reliable edge
        reliable_edge = PathEdge(
            source_node_id="node_1", target_node_id="node_2", relationship_type="call", weight=0.8, confidence=PathConfidence.HIGH
        )
        assert reliable_edge.is_reliable()

        # Unreliable edge (low confidence)
        unreliable_edge = PathEdge(
            source_node_id="node_1", target_node_id="node_2", relationship_type="call", weight=0.8, confidence=PathConfidence.LOW
        )
        assert not unreliable_edge.is_reliable()

        # Unreliable edge (low weight)
        low_weight_edge = PathEdge(
            source_node_id="node_1", target_node_id="node_2", relationship_type="call", weight=0.3, confidence=PathConfidence.HIGH
        )
        assert not low_weight_edge.is_reliable()


class TestExecutionPath:
    """Test ExecutionPath data model."""

    def create_sample_execution_path(self) -> ExecutionPath:
        """Create a sample execution path for testing."""
        nodes = [
            PathNode("node_1", "main", "main", ChunkType.FUNCTION, "/main.py", 1, 10, "source"),
            PathNode("node_2", "helper", "helper", ChunkType.FUNCTION, "/helper.py", 5, 15, "intermediate"),
            PathNode("node_3", "utility", "utility", ChunkType.FUNCTION, "/util.py", 20, 30, "target"),
        ]

        edges = [
            PathEdge("node_1", "node_2", "function_call", 0.9, PathConfidence.HIGH),
            PathEdge("node_2", "node_3", "function_call", 0.8, PathConfidence.MEDIUM),
        ]

        return ExecutionPath(
            path_id="exec_path_1",
            nodes=nodes,
            edges=edges,
            entry_points=["node_1"],
            exit_points=["node_3"],
            has_loops=True,
            is_async=True,
            complexity_score=0.7,
        )

    def test_execution_path_creation(self):
        """Test ExecutionPath creation."""
        path = self.create_sample_execution_path()

        assert path.path_id == "exec_path_1"
        assert path.path_type == PathType.EXECUTION_PATH
        assert len(path.nodes) == 3
        assert len(path.edges) == 2
        assert path.has_loops
        assert path.is_async
        assert path.complexity_score == 0.7

    def test_path_length_calculation(self):
        """Test path length calculation."""
        path = self.create_sample_execution_path()
        assert path.get_path_length() == 3

    def test_critical_nodes_detection(self):
        """Test critical nodes detection."""
        path = self.create_sample_execution_path()

        # Make node_2 critical by increasing importance
        path.nodes[1].importance_score = 0.8

        critical_nodes = path.get_critical_nodes()
        assert len(critical_nodes) >= 2  # At least source and high-importance nodes

        # Check that critical nodes include source and target
        critical_roles = {node.role_in_path for node in critical_nodes}
        assert "source" in critical_roles

    def test_complexity_estimation(self):
        """Test complexity estimation."""
        path = ExecutionPath(
            path_id="complex_path",
            nodes=[
                PathNode(f"node_{i}", f"func_{i}", f"func_{i}", ChunkType.FUNCTION, f"/file_{i}.py", i, i + 10, "intermediate")
                for i in range(10)
            ],  # 10 nodes
            has_loops=True,
            has_exceptions=True,
            is_async=True,
            branch_points=["node_3", "node_7"],  # 2 branch points
        )

        estimated_complexity = path.estimate_complexity()

        # Base: 10 * 0.1 = 1.0
        # Loops: +0.3, Exceptions: +0.2, Async: +0.2
        # Branch points: 2 * 0.15 = 0.3
        # Total: 1.0 + 0.3 + 0.2 + 0.2 + 0.3 = 2.0, clamped to 1.0
        assert estimated_complexity == 1.0

    def test_bottleneck_detection(self):
        """Test bottleneck edge detection."""
        path = self.create_sample_execution_path()

        # Mark one edge as bottleneck
        path.edges[0].is_bottleneck = True

        bottlenecks = path.get_bottleneck_edges()
        assert len(bottlenecks) == 1
        assert bottlenecks[0].source_node_id == "node_1"


class TestDataFlowPath:
    """Test DataFlowPath data model."""

    def create_sample_data_flow_path(self) -> DataFlowPath:
        """Create a sample data flow path for testing."""
        nodes = [
            PathNode("var_creation", "create_data", "create_data", ChunkType.FUNCTION, "/data.py", 1, 5, "source"),
            PathNode("var_transform", "transform_data", "transform_data", ChunkType.FUNCTION, "/transform.py", 10, 20, "intermediate"),
            PathNode("var_usage", "use_data", "use_data", ChunkType.FUNCTION, "/usage.py", 30, 40, "target"),
        ]

        edges = [
            PathEdge("var_creation", "var_transform", "data_flow", 0.9, PathConfidence.HIGH),
            PathEdge("var_transform", "var_usage", "data_flow", 0.8, PathConfidence.MEDIUM),
        ]

        return DataFlowPath(
            path_id="data_flow_1",
            nodes=nodes,
            edges=edges,
            data_source="user_input",
            data_destinations=["database", "cache"],
            transformations=["validation", "normalization", "enrichment"],
            data_types=["string", "dict"],
            has_side_effects=True,
            creation_point="var_creation",
            modification_points=["var_transform"],
            access_points=["var_usage"],
        )

    def test_data_flow_path_creation(self):
        """Test DataFlowPath creation."""
        path = self.create_sample_data_flow_path()

        assert path.path_id == "data_flow_1"
        assert path.path_type == PathType.DATA_FLOW
        assert path.data_source == "user_input"
        assert len(path.data_destinations) == 2
        assert len(path.transformations) == 3
        assert path.has_side_effects

    def test_transformation_count(self):
        """Test transformation count calculation."""
        path = self.create_sample_data_flow_path()
        assert path.get_transformation_count() == 3

    def test_complex_flow_detection(self):
        """Test complex flow detection."""
        # Simple flow
        simple_path = DataFlowPath(
            path_id="simple_flow", transformations=["validation"], modification_points=["point1"], has_side_effects=False
        )
        assert not simple_path.is_complex_flow()

        # Complex flow (many transformations)
        complex_path = DataFlowPath(
            path_id="complex_flow",
            transformations=["t1", "t2", "t3", "t4"],  # > 3 transformations
            modification_points=["point1"],
            has_side_effects=False,
        )
        assert complex_path.is_complex_flow()

        # Complex flow (many modifications)
        many_mods_path = DataFlowPath(
            path_id="many_mods_flow",
            transformations=["t1"],
            modification_points=["p1", "p2", "p3", "p4", "p5", "p6"],  # > 5 modifications
            has_side_effects=False,
        )
        assert many_mods_path.is_complex_flow()

        # Complex flow (side effects)
        side_effects_path = DataFlowPath(
            path_id="side_effects_flow", transformations=["t1"], modification_points=["point1"], has_side_effects=True
        )
        assert side_effects_path.is_complex_flow()

    def test_lifecycle_stages(self):
        """Test data lifecycle stages detection."""
        path = self.create_sample_data_flow_path()
        stages = path.get_data_lifecycle_stages()

        expected_stages = ["creation", "modification", "access"]
        assert set(stages) == set(expected_stages)


class TestDependencyPath:
    """Test DependencyPath data model."""

    def create_sample_dependency_path(self) -> DependencyPath:
        """Create a sample dependency path for testing."""
        nodes = [
            PathNode("module_a", "module_a", "module_a", ChunkType.MODULE_DOCSTRING, "/a.py", 1, 100, "source"),
            PathNode("module_b", "module_b", "module_b", ChunkType.MODULE_DOCSTRING, "/b.py", 1, 150, "target"),
        ]

        edges = [PathEdge("module_a", "module_b", "import_dependency", 0.9, PathConfidence.VERY_HIGH)]

        return DependencyPath(
            path_id="dep_path_1",
            nodes=nodes,
            edges=edges,
            dependency_type="import",
            is_circular=False,
            is_external=True,
            required_modules=["requests", "numpy"],
            optional_modules=["matplotlib"],
            stability_score=0.8,
            coupling_strength=0.6,
            impact_radius=5,
        )

    def test_dependency_path_creation(self):
        """Test DependencyPath creation."""
        path = self.create_sample_dependency_path()

        assert path.path_id == "dep_path_1"
        assert path.path_type == PathType.DEPENDENCY_PATH
        assert path.dependency_type == "import"
        assert not path.is_circular
        assert path.is_external
        assert len(path.required_modules) == 2
        assert len(path.optional_modules) == 1

    def test_high_risk_detection(self):
        """Test high risk dependency detection."""
        # Low risk dependency
        low_risk = DependencyPath(path_id="low_risk", is_circular=False, stability_score=0.8, coupling_strength=0.3)
        assert not low_risk.is_high_risk()

        # High risk - circular
        circular_risk = DependencyPath(path_id="circular_risk", is_circular=True, stability_score=0.8, coupling_strength=0.3)
        assert circular_risk.is_high_risk()

        # High risk - low stability
        unstable_risk = DependencyPath(path_id="unstable_risk", is_circular=False, stability_score=0.3, coupling_strength=0.3)
        assert unstable_risk.is_high_risk()

        # High risk - high coupling
        coupled_risk = DependencyPath(path_id="coupled_risk", is_circular=False, stability_score=0.8, coupling_strength=0.8)
        assert coupled_risk.is_high_risk()

    def test_external_dependencies(self):
        """Test external dependencies extraction."""
        path = self.create_sample_dependency_path()
        external_deps = path.get_external_dependencies()

        expected_deps = ["requests", "numpy", "matplotlib"]
        assert set(external_deps) == set(expected_deps)

        # Non-external dependency
        internal_path = DependencyPath(path_id="internal", is_external=False, required_modules=["internal_module"])
        assert internal_path.get_external_dependencies() == []


class TestRelationalPathCollection:
    """Test RelationalPathCollection data model."""

    def create_sample_collection(self) -> RelationalPathCollection:
        """Create a sample path collection for testing."""
        exec_path = ExecutionPath(
            path_id="exec_1",
            nodes=[PathNode("n1", "func1", "func1", ChunkType.FUNCTION, "/file1.py", 1, 10, "source")],
            criticality_score=0.8,
        )

        data_path = DataFlowPath(
            path_id="data_1",
            nodes=[PathNode("n2", "var1", "var1", ChunkType.VARIABLE, "/file2.py", 5, 15, "source")],
            data_quality_score=0.9,
        )

        dep_path = DependencyPath(
            path_id="dep_1", nodes=[PathNode("n3", "mod1", "mod1", ChunkType.IMPORT, "/file3.py", 1, 2, "source")], stability_score=0.7
        )

        return RelationalPathCollection(
            collection_id="collection_1",
            collection_name="Sample Collection",
            execution_paths=[exec_path],
            data_flow_paths=[data_path],
            dependency_paths=[dep_path],
            primary_entry_points=["func1"],
            architectural_patterns=["MVC", "Observer"],
            coverage_score=0.85,
            coherence_score=0.75,
            completeness_score=0.90,
        )

    def test_collection_creation(self):
        """Test RelationalPathCollection creation."""
        collection = self.create_sample_collection()

        assert collection.collection_id == "collection_1"
        assert collection.collection_name == "Sample Collection"
        assert len(collection.execution_paths) == 1
        assert len(collection.data_flow_paths) == 1
        assert len(collection.dependency_paths) == 1
        assert len(collection.covered_files) == 3  # Auto-populated from paths

    def test_total_path_count(self):
        """Test total path count calculation."""
        collection = self.create_sample_collection()
        assert collection.get_total_path_count() == 3

    def test_high_value_paths(self):
        """Test high value paths extraction."""
        collection = self.create_sample_collection()
        high_value = collection.get_high_value_paths(threshold=0.7)

        # Should include all paths as they all have scores >= 0.7
        assert len(high_value) == 3

        # Test with higher threshold
        very_high_value = collection.get_high_value_paths(threshold=0.85)
        assert len(very_high_value) == 1  # Only data path with 0.9 quality score

    def test_complexity_distribution(self):
        """Test complexity distribution calculation."""
        # Create collection with varied complexity scores
        exec_paths = [
            ExecutionPath(path_id="e1", complexity_score=0.3),
            ExecutionPath(path_id="e2", complexity_score=0.7),
            ExecutionPath(path_id="e3", complexity_score=0.9),
        ]

        collection = RelationalPathCollection(collection_id="test", collection_name="Test", execution_paths=exec_paths)

        distribution = collection.get_complexity_distribution()

        assert abs(distribution["mean"] - 0.63333) < 0.001  # (0.3 + 0.7 + 0.9) / 3
        assert distribution["min"] == 0.3
        assert distribution["max"] == 0.9

    def test_to_dict_serialization(self):
        """Test collection serialization to dictionary."""
        collection = self.create_sample_collection()
        data = collection.to_dict()

        assert data["collection_id"] == "collection_1"
        assert data["execution_paths_count"] == 1
        assert data["data_flow_paths_count"] == 1
        assert data["dependency_paths_count"] == 1
        assert data["total_paths"] == 3
        assert len(data["covered_files"]) == 3
        assert "complexity_distribution" in data


class TestPathExtractionResult:
    """Test PathExtractionResult data model."""

    def test_extraction_result_creation(self):
        """Test PathExtractionResult creation."""
        collection = RelationalPathCollection(collection_id="test", collection_name="Test Collection")

        result = PathExtractionResult(
            path_collection=collection,
            processing_time_ms=1500.0,
            source_chunks_count=100,
            success_rate=0.95,
            paths_with_high_confidence=25,
            paths_requiring_review=5,
            extraction_warnings=["Warning 1", "Warning 2"],
        )

        assert result.path_collection == collection
        assert result.processing_time_ms == 1500.0
        assert result.source_chunks_count == 100
        assert result.success_rate == 0.95
        assert result.paths_with_high_confidence == 25
        assert result.paths_requiring_review == 5
        assert len(result.extraction_warnings) == 2

    def test_success_detection(self):
        """Test extraction success detection."""
        collection = RelationalPathCollection(collection_id="test", collection_name="Test")

        # Successful extraction
        successful_result = PathExtractionResult(path_collection=collection, success_rate=0.9, extraction_warnings=["Warning 1"])
        assert successful_result.is_successful()

        # Failed extraction - low success rate
        failed_result = PathExtractionResult(path_collection=collection, success_rate=0.7, extraction_warnings=["Warning 1"])
        assert not failed_result.is_successful()

        # Failed extraction - too many warnings
        many_warnings_result = PathExtractionResult(
            path_collection=collection, success_rate=0.9, extraction_warnings=[f"Warning {i}" for i in range(10)]
        )
        assert not many_warnings_result.is_successful()

    def test_extraction_efficiency(self):
        """Test extraction efficiency calculation."""
        # Create collection with multiple paths
        exec_paths = [ExecutionPath(path_id=f"e{i}") for i in range(5)]
        data_paths = [DataFlowPath(path_id=f"d{i}") for i in range(3)]

        collection = RelationalPathCollection(
            collection_id="test", collection_name="Test", execution_paths=exec_paths, data_flow_paths=data_paths
        )

        result = PathExtractionResult(path_collection=collection, processing_time_ms=2000.0)  # 2 seconds

        efficiency = result.get_extraction_efficiency()
        # 8 paths in 2 seconds = 4 paths per second
        assert abs(efficiency - 4.0) < 0.01

        # Test zero time handling
        zero_time_result = PathExtractionResult(path_collection=collection, processing_time_ms=0.0)
        assert zero_time_result.get_extraction_efficiency() == 0.0


class TestTypeAliases:
    """Test type aliases and utility types."""

    def test_any_path_alias(self):
        """Test AnyPath type alias usage."""
        exec_path = ExecutionPath(path_id="exec")
        data_path = DataFlowPath(path_id="data")
        dep_path = DependencyPath(path_id="dep")

        paths: list[AnyPath] = [exec_path, data_path, dep_path]
        assert len(paths) == 3

        # Test that all paths have common interface
        for path in paths:
            assert hasattr(path, "path_id")
            assert hasattr(path, "path_type")
            assert hasattr(path, "nodes")
            assert hasattr(path, "edges")

    def test_path_id_node_id_aliases(self):
        """Test PathId and NodeId type aliases."""
        path_id: PathId = "test_path_123"
        node_id: NodeId = "test_node_456"

        assert isinstance(path_id, str)
        assert isinstance(node_id, str)
        assert path_id == "test_path_123"
        assert node_id == "test_node_456"


if __name__ == "__main__":
    pytest.main([__file__])
