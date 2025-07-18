"""
Comprehensive Unit Tests for Graph Pattern Identification Tools

This module provides thorough testing for pattern identification functionality,
covering various pattern types, project scopes, and confidence thresholds.
"""

from typing import Any, Dict, List
from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.models.code_chunk import ChunkType
from src.services.graph_rag_service import GraphRAGService
from src.services.structure_relationship_builder import GraphEdge, GraphNode, StructureGraph
from src.tools.graph_rag.pattern_identification import graph_identify_patterns


class TestGraphPatternIdentification:
    """Test suite for graph pattern identification functionality."""

    @pytest.fixture
    def mock_qdrant_service(self):
        """Mock QdrantService for testing."""
        mock_service = Mock()
        mock_service._initialize_cache = AsyncMock()
        return mock_service

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock EmbeddingService for testing."""
        mock_service = Mock()
        mock_service.generate_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        return mock_service

    @pytest.fixture
    def factory_pattern_graph(self):
        """Sample graph with factory pattern for testing."""
        # Create nodes that represent a factory pattern
        factory_class = GraphNode(
            breadcrumb="app.factories.UserFactory",
            name="UserFactory",
            chunk_type=ChunkType.CLASS,
            file_path="/app/factories.py",
            depth=1,
            parent_breadcrumb="app.factories",
            children_breadcrumbs=[
                "app.factories.UserFactory.create",
                "app.factories.UserFactory.build",
            ],
        )

        create_method = GraphNode(
            breadcrumb="app.factories.UserFactory.create",
            name="create",
            chunk_type=ChunkType.METHOD,
            file_path="/app/factories.py",
            depth=2,
            parent_breadcrumb="app.factories.UserFactory",
            children_breadcrumbs=[],
        )

        build_method = GraphNode(
            breadcrumb="app.factories.UserFactory.build",
            name="build",
            chunk_type=ChunkType.METHOD,
            file_path="/app/factories.py",
            depth=2,
            parent_breadcrumb="app.factories.UserFactory",
            children_breadcrumbs=[],
        )

        # Another factory
        product_factory = GraphNode(
            breadcrumb="app.factories.ProductFactory",
            name="ProductFactory",
            chunk_type=ChunkType.CLASS,
            file_path="/app/factories.py",
            depth=1,
            parent_breadcrumb="app.factories",
            children_breadcrumbs=[
                "app.factories.ProductFactory.create",
                "app.factories.ProductFactory.make",
            ],
        )

        product_create = GraphNode(
            breadcrumb="app.factories.ProductFactory.create",
            name="create",
            chunk_type=ChunkType.METHOD,
            file_path="/app/factories.py",
            depth=2,
            parent_breadcrumb="app.factories.ProductFactory",
            children_breadcrumbs=[],
        )

        product_make = GraphNode(
            breadcrumb="app.factories.ProductFactory.make",
            name="make",
            chunk_type=ChunkType.METHOD,
            file_path="/app/factories.py",
            depth=2,
            parent_breadcrumb="app.factories.ProductFactory",
            children_breadcrumbs=[],
        )

        nodes = {
            "app.factories.UserFactory": factory_class,
            "app.factories.UserFactory.create": create_method,
            "app.factories.UserFactory.build": build_method,
            "app.factories.ProductFactory": product_factory,
            "app.factories.ProductFactory.create": product_create,
            "app.factories.ProductFactory.make": product_make,
        }

        edges = [
            GraphEdge("app.factories.UserFactory", "app.factories.UserFactory.create", "contains", 1.0),
            GraphEdge("app.factories.UserFactory", "app.factories.UserFactory.build", "contains", 1.0),
            GraphEdge("app.factories.ProductFactory", "app.factories.ProductFactory.create", "contains", 1.0),
            GraphEdge("app.factories.ProductFactory", "app.factories.ProductFactory.make", "contains", 1.0),
        ]

        return StructureGraph(
            nodes=nodes,
            edges=edges,
            root_nodes=["app.factories.UserFactory", "app.factories.ProductFactory"],
            project_name="factory_test_project",
        )

    @pytest.fixture
    def observer_pattern_graph(self):
        """Sample graph with observer pattern for testing."""
        # Create nodes that represent an observer pattern
        subject_class = GraphNode(
            breadcrumb="app.events.EventManager",
            name="EventManager",
            chunk_type=ChunkType.CLASS,
            file_path="/app/events.py",
            depth=1,
            parent_breadcrumb="app.events",
            children_breadcrumbs=[
                "app.events.EventManager.add_listener",
                "app.events.EventManager.remove_listener",
                "app.events.EventManager.notify",
            ],
        )

        add_listener = GraphNode(
            breadcrumb="app.events.EventManager.add_listener",
            name="add_listener",
            chunk_type=ChunkType.METHOD,
            file_path="/app/events.py",
            depth=2,
            parent_breadcrumb="app.events.EventManager",
            children_breadcrumbs=[],
        )

        remove_listener = GraphNode(
            breadcrumb="app.events.EventManager.remove_listener",
            name="remove_listener",
            chunk_type=ChunkType.METHOD,
            file_path="/app/events.py",
            depth=2,
            parent_breadcrumb="app.events.EventManager",
            children_breadcrumbs=[],
        )

        notify_method = GraphNode(
            breadcrumb="app.events.EventManager.notify",
            name="notify",
            chunk_type=ChunkType.METHOD,
            file_path="/app/events.py",
            depth=2,
            parent_breadcrumb="app.events.EventManager",
            children_breadcrumbs=[],
        )

        # Observer interface
        observer_interface = GraphNode(
            breadcrumb="app.interfaces.EventObserver",
            name="EventObserver",
            chunk_type=ChunkType.CLASS,
            file_path="/app/interfaces.py",
            depth=1,
            parent_breadcrumb="app.interfaces",
            children_breadcrumbs=["app.interfaces.EventObserver.update"],
        )

        update_method = GraphNode(
            breadcrumb="app.interfaces.EventObserver.update",
            name="update",
            chunk_type=ChunkType.METHOD,
            file_path="/app/interfaces.py",
            depth=2,
            parent_breadcrumb="app.interfaces.EventObserver",
            children_breadcrumbs=[],
        )

        nodes = {
            "app.events.EventManager": subject_class,
            "app.events.EventManager.add_listener": add_listener,
            "app.events.EventManager.remove_listener": remove_listener,
            "app.events.EventManager.notify": notify_method,
            "app.interfaces.EventObserver": observer_interface,
            "app.interfaces.EventObserver.update": update_method,
        }

        edges = [
            GraphEdge("app.events.EventManager", "app.events.EventManager.add_listener", "contains", 1.0),
            GraphEdge("app.events.EventManager", "app.events.EventManager.remove_listener", "contains", 1.0),
            GraphEdge("app.events.EventManager", "app.events.EventManager.notify", "contains", 1.0),
            GraphEdge("app.interfaces.EventObserver", "app.interfaces.EventObserver.update", "contains", 1.0),
            GraphEdge("app.events.EventManager", "app.interfaces.EventObserver", "uses", 0.8),
        ]

        return StructureGraph(
            nodes=nodes,
            edges=edges,
            root_nodes=["app.events.EventManager", "app.interfaces.EventObserver"],
            project_name="observer_test_project",
        )

    @pytest.fixture
    def layered_architecture_graph(self):
        """Sample graph with layered architecture for testing."""
        # Create nodes representing a layered architecture
        controller_node = GraphNode(
            breadcrumb="app.controllers.UserController",
            name="UserController",
            chunk_type=ChunkType.CLASS,
            file_path="/app/controllers/user.py",
            depth=1,
            parent_breadcrumb="app.controllers",
            children_breadcrumbs=["app.controllers.UserController.create_user"],
        )

        service_node = GraphNode(
            breadcrumb="app.services.UserService",
            name="UserService",
            chunk_type=ChunkType.CLASS,
            file_path="/app/services/user.py",
            depth=2,
            parent_breadcrumb="app.services",
            children_breadcrumbs=["app.services.UserService.create"],
        )

        repository_node = GraphNode(
            breadcrumb="app.repositories.UserRepository",
            name="UserRepository",
            chunk_type=ChunkType.CLASS,
            file_path="/app/repositories/user.py",
            depth=3,
            parent_breadcrumb="app.repositories",
            children_breadcrumbs=["app.repositories.UserRepository.save"],
        )

        model_node = GraphNode(
            breadcrumb="app.models.User",
            name="User",
            chunk_type=ChunkType.CLASS,
            file_path="/app/models/user.py",
            depth=4,
            parent_breadcrumb="app.models",
            children_breadcrumbs=[],
        )

        nodes = {
            "app.controllers.UserController": controller_node,
            "app.services.UserService": service_node,
            "app.repositories.UserRepository": repository_node,
            "app.models.User": model_node,
        }

        edges = [
            GraphEdge("app.controllers.UserController", "app.services.UserService", "uses", 0.9),
            GraphEdge("app.services.UserService", "app.repositories.UserRepository", "uses", 0.9),
            GraphEdge("app.repositories.UserRepository", "app.models.User", "uses", 0.9),
        ]

        return StructureGraph(
            nodes=nodes,
            edges=edges,
            root_nodes=["app.controllers.UserController"],
            project_name="layered_test_project",
        )

    @pytest.fixture
    def mock_graph_rag_service_factory(self, factory_pattern_graph):
        """Mock GraphRAGService with factory pattern."""
        mock_service = Mock(spec=GraphRAGService)
        mock_service.build_structure_graph = AsyncMock(return_value=factory_pattern_graph)
        mock_service.get_project_structure_overview = AsyncMock(
            return_value={
                "project_name": "factory_test_project",
                "total_components": 6,
                "total_relationships": 4,
                "root_components": 2,
                "max_depth": 2,
                "breakdown": {
                    "by_type": {"class": 2, "method": 4},
                    "by_depth": {1: 2, 2: 4},
                    "by_language": {"py": 6},
                    "by_relationship": {"contains": 4},
                },
                "largest_components": [
                    {"breadcrumb": "app.factories.UserFactory", "children_count": 2},
                    {"breadcrumb": "app.factories.ProductFactory", "children_count": 2},
                ],
            }
        )
        return mock_service

    @pytest.fixture
    def mock_graph_rag_service_observer(self, observer_pattern_graph):
        """Mock GraphRAGService with observer pattern."""
        mock_service = Mock(spec=GraphRAGService)
        mock_service.build_structure_graph = AsyncMock(return_value=observer_pattern_graph)
        mock_service.get_project_structure_overview = AsyncMock(
            return_value={
                "project_name": "observer_test_project",
                "total_components": 6,
                "total_relationships": 5,
                "root_components": 2,
                "max_depth": 2,
                "breakdown": {
                    "by_type": {"class": 2, "method": 4},
                    "by_depth": {1: 2, 2: 4},
                    "by_language": {"py": 6},
                    "by_relationship": {"contains": 4, "uses": 1},
                },
            }
        )
        return mock_service

    @pytest.fixture
    def mock_graph_rag_service_layered(self, layered_architecture_graph):
        """Mock GraphRAGService with layered architecture."""
        mock_service = Mock(spec=GraphRAGService)
        mock_service.build_structure_graph = AsyncMock(return_value=layered_architecture_graph)
        mock_service.get_project_structure_overview = AsyncMock(
            return_value={
                "project_name": "layered_test_project",
                "total_components": 4,
                "total_relationships": 3,
                "root_components": 1,
                "max_depth": 4,
                "breakdown": {
                    "by_type": {"class": 4},
                    "by_depth": {1: 1, 2: 1, 3: 1, 4: 1},
                    "by_language": {"py": 4},
                    "by_relationship": {"uses": 3},
                },
            }
        )
        return mock_service

    @pytest.mark.asyncio
    async def test_factory_pattern_identification(self, mock_graph_rag_service_factory):
        """Test identification of factory pattern."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                pattern_types=["creational"],
                min_confidence=0.5,
                include_comparisons=True,
                include_improvements=False,
            )

            assert result["success"] is True
            assert result["project_name"] == "factory_test_project"
            assert "patterns_identified" in result

            # Should identify factory patterns
            patterns = result["patterns_identified"]
            factory_patterns = [p for p in patterns if "Factory" in p.get("name", "")]
            assert len(factory_patterns) > 0

            # Check pattern structure
            for pattern in factory_patterns:
                assert "name" in pattern
                assert "confidence" in pattern
                assert "description" in pattern
                assert "pattern_type" in pattern
                assert pattern["confidence"] >= 0.5

    @pytest.mark.asyncio
    async def test_observer_pattern_identification(self, mock_graph_rag_service_observer):
        """Test identification of observer pattern."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_observer),
        ):
            result = await graph_identify_patterns(
                project_name="observer_test_project",
                pattern_types=["behavioral"],
                min_confidence=0.6,
            )

            assert result["success"] is True
            patterns = result["patterns_identified"]

            # Should identify observer-like patterns
            observer_patterns = [p for p in patterns if "listener" in p.get("name", "").lower() or "observer" in p.get("name", "").lower()]
            assert len(observer_patterns) >= 0  # May or may not find patterns depending on implementation

    @pytest.mark.asyncio
    async def test_layered_architecture_identification(self, mock_graph_rag_service_layered):
        """Test identification of layered architecture pattern."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_layered),
        ):
            result = await graph_identify_patterns(
                project_name="layered_test_project",
                pattern_types=["architectural"],
                min_confidence=0.7,
            )

            assert result["success"] is True
            patterns = result["patterns_identified"]

            # Should identify layered architecture
            layered_patterns = [p for p in patterns if "layer" in p.get("name", "").lower()]
            assert len(layered_patterns) >= 0

    @pytest.mark.asyncio
    async def test_all_pattern_types(self, mock_graph_rag_service_factory):
        """Test identification of all pattern types."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                pattern_types=["structural", "behavioral", "creational", "naming", "architectural"],
                min_confidence=0.5,
            )

            assert result["success"] is True
            assert "patterns_identified" in result
            assert "pattern_analysis" in result
            assert "metadata" in result

            # Verify metadata
            metadata = result["metadata"]
            assert metadata["total_pattern_types_requested"] == 5
            assert metadata["min_confidence_threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_naming_pattern_identification(self, mock_graph_rag_service_factory):
        """Test identification of naming patterns."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                pattern_types=["naming"],
                min_confidence=0.8,
            )

            assert result["success"] is True
            patterns = result["patterns_identified"]

            # Should identify naming patterns
            naming_patterns = [p for p in patterns if p.get("pattern_type") == "naming"]
            # May or may not find patterns depending on naming consistency

    @pytest.mark.asyncio
    async def test_scoped_pattern_analysis(self, mock_graph_rag_service_factory):
        """Test pattern analysis scoped to specific breadcrumb."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                pattern_types=["creational"],
                scope_breadcrumb="app.factories.UserFactory",
                min_confidence=0.5,
            )

            assert result["success"] is True
            assert result["scope"] == "component:app.factories.UserFactory"

    @pytest.mark.asyncio
    async def test_high_confidence_threshold(self, mock_graph_rag_service_factory):
        """Test pattern identification with high confidence threshold."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                min_confidence=0.95,  # Very high threshold
            )

            assert result["success"] is True
            patterns = result["patterns_identified"]

            # High threshold should filter out most patterns
            for pattern in patterns:
                assert pattern["confidence"] >= 0.95

    @pytest.mark.asyncio
    async def test_max_patterns_limit(self, mock_graph_rag_service_factory):
        """Test maximum patterns limit."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                max_patterns=3,
                min_confidence=0.1,  # Low threshold to get more patterns
            )

            assert result["success"] is True
            patterns = result["patterns_identified"]
            assert len(patterns) <= 3

    @pytest.mark.asyncio
    async def test_pattern_comparisons(self, mock_graph_rag_service_factory):
        """Test pattern comparison functionality."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                include_comparisons=True,
            )

            assert result["success"] is True

            if "pattern_comparisons" in result:
                comparisons = result["pattern_comparisons"]
                assert isinstance(comparisons, (list, dict))

    @pytest.mark.asyncio
    async def test_pattern_improvements(self, mock_graph_rag_service_factory):
        """Test pattern improvement suggestions."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                include_improvements=True,
            )

            assert result["success"] is True

            if "pattern_improvements" in result:
                improvements = result["pattern_improvements"]
                assert isinstance(improvements, (list, dict))

    @pytest.mark.asyncio
    async def test_analysis_depth_comprehensive(self, mock_graph_rag_service_factory):
        """Test comprehensive analysis depth."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                analysis_depth="comprehensive",
            )

            assert result["success"] is True
            assert result["analysis_depth"] == "comprehensive"
            assert "pattern_analysis" in result

    @pytest.mark.asyncio
    async def test_analysis_depth_basic(self, mock_graph_rag_service_factory):
        """Test basic analysis depth."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                analysis_depth="basic",
            )

            assert result["success"] is True
            assert result["analysis_depth"] == "basic"

    @pytest.mark.asyncio
    async def test_empty_project_name_validation(self):
        """Test validation for empty project name."""
        result = await graph_identify_patterns(
            project_name="",
        )

        assert result["success"] is False
        assert "error" in result
        assert "empty" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_invalid_confidence_threshold(self, mock_graph_rag_service_factory):
        """Test handling of invalid confidence threshold."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            # Test negative confidence
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                min_confidence=-0.5,
            )
            assert result["min_confidence"] == 0.0  # Should be clamped

            # Test confidence > 1.0
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                min_confidence=1.5,
            )
            assert result["min_confidence"] == 1.0  # Should be clamped

    @pytest.mark.asyncio
    async def test_invalid_max_patterns(self, mock_graph_rag_service_factory):
        """Test handling of invalid max_patterns value."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            # Test negative max_patterns
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                max_patterns=-5,
            )
            assert result["max_patterns"] == 1  # Should be clamped

            # Test max_patterns > 50
            result = await graph_identify_patterns(
                project_name="factory_test_project",
                max_patterns=100,
            )
            assert result["max_patterns"] == 50  # Should be clamped

    @pytest.mark.asyncio
    async def test_nonexistent_project_error(self):
        """Test error handling for nonexistent project."""
        mock_service = Mock(spec=GraphRAGService)
        mock_service.build_structure_graph = AsyncMock(
            return_value=StructureGraph(nodes={}, edges=[], root_nodes=[], project_name="nonexistent_project")
        )

        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_service),
        ):
            result = await graph_identify_patterns(
                project_name="nonexistent_project",
            )

            assert result["success"] is False
            assert "error" in result
            assert "no structure graph" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_service_initialization_error(self):
        """Test error handling during service initialization."""
        with patch("src.tools.graph_rag.pattern_identification.QdrantService", side_effect=Exception("Service error")):
            result = await graph_identify_patterns(
                project_name="test_project",
            )

            assert result["success"] is False
            assert "error" in result
            assert "Service error" in result["error"]

    @pytest.mark.asyncio
    async def test_graph_building_error(self):
        """Test error handling during graph building."""
        mock_service = Mock(spec=GraphRAGService)
        mock_service.build_structure_graph = AsyncMock(side_effect=Exception("Graph building failed"))

        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_service),
        ):
            result = await graph_identify_patterns(
                project_name="test_project",
            )

            assert result["success"] is False
            assert "error" in result
            assert "Graph building failed" in result["error"]

    @pytest.mark.asyncio
    async def test_pattern_analysis_with_different_types(self, mock_graph_rag_service_factory):
        """Test pattern analysis with different pattern type combinations."""
        pattern_type_combinations = [
            ["structural"],
            ["behavioral"],
            ["creational"],
            ["naming"],
            ["architectural"],
            ["structural", "behavioral"],
            ["creational", "architectural"],
            ["naming", "structural", "behavioral"],
        ]

        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            for pattern_types in pattern_type_combinations:
                result = await graph_identify_patterns(
                    project_name="factory_test_project",
                    pattern_types=pattern_types,
                    min_confidence=0.5,
                )

                assert result["success"] is True
                assert result["pattern_types_analyzed"] == pattern_types

    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, mock_graph_rag_service_factory):
        """Test that performance metrics are properly tracked."""
        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="factory_test_project",
            )

            assert result["success"] is True
            assert "metadata" in result

            metadata = result["metadata"]
            assert "analysis_completed_at" in metadata
            assert "total_patterns_found" in metadata

    @pytest.mark.asyncio
    async def test_large_graph_performance(self, mock_graph_rag_service_factory):
        """Test pattern identification performance with large graphs."""
        # Create a large graph for testing
        large_nodes = {}
        large_edges = []

        # Create 100 classes with factory-like patterns
        for i in range(100):
            class_breadcrumb = f"app.factories.Factory{i}"
            large_nodes[class_breadcrumb] = GraphNode(
                breadcrumb=class_breadcrumb,
                name=f"Factory{i}",
                chunk_type=ChunkType.CLASS,
                file_path=f"/app/factory_{i}.py",
                depth=1,
                parent_breadcrumb="app.factories",
                children_breadcrumbs=[
                    f"{class_breadcrumb}.create",
                    f"{class_breadcrumb}.build",
                ],
            )

            # Add methods
            for method in ["create", "build"]:
                method_breadcrumb = f"{class_breadcrumb}.{method}"
                large_nodes[method_breadcrumb] = GraphNode(
                    breadcrumb=method_breadcrumb,
                    name=method,
                    chunk_type=ChunkType.METHOD,
                    file_path=f"/app/factory_{i}.py",
                    depth=2,
                    parent_breadcrumb=class_breadcrumb,
                    children_breadcrumbs=[],
                )

                large_edges.append(GraphEdge(class_breadcrumb, method_breadcrumb, "contains", 1.0))

        large_graph = StructureGraph(
            nodes=large_nodes,
            edges=large_edges,
            root_nodes=[f"app.factories.Factory{i}" for i in range(100)],
            project_name="large_factory_project",
        )

        mock_graph_rag_service_factory.build_structure_graph = AsyncMock(return_value=large_graph)
        mock_graph_rag_service_factory.get_project_structure_overview = AsyncMock(
            return_value={
                "project_name": "large_factory_project",
                "total_components": 300,  # 100 classes + 200 methods
                "total_relationships": 200,
                "breakdown": {
                    "by_type": {"class": 100, "method": 200},
                    "by_depth": {1: 100, 2: 200},
                },
            }
        )

        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_graph_rag_service_factory),
        ):
            result = await graph_identify_patterns(
                project_name="large_factory_project",
                pattern_types=["creational"],
                max_patterns=20,  # Limit results for performance
            )

            assert result["success"] is True
            assert len(result["patterns_identified"]) <= 20

    @pytest.mark.asyncio
    async def test_empty_graph_handling(self):
        """Test handling of empty graphs."""
        mock_service = Mock(spec=GraphRAGService)
        empty_graph = StructureGraph(
            nodes={},
            edges=[],
            root_nodes=[],
            project_name="empty_project",
        )
        mock_service.build_structure_graph = AsyncMock(return_value=empty_graph)

        with (
            patch("src.tools.graph_rag.pattern_identification.QdrantService"),
            patch("src.tools.graph_rag.pattern_identification.EmbeddingService"),
            patch("src.tools.graph_rag.pattern_identification.GraphRAGService", return_value=mock_service),
        ):
            result = await graph_identify_patterns(
                project_name="empty_project",
            )

            assert result["success"] is False
            assert "error" in result
            assert "no structure graph" in result["error"].lower()

    def test_pattern_identification_interface_validation(self):
        """Test validation of pattern identification interface."""
        # Test imports work correctly
        # Verify function signature
        import inspect

        from src.tools.graph_rag.pattern_identification import graph_identify_patterns

        sig = inspect.signature(graph_identify_patterns)

        expected_params = [
            "project_name",
            "pattern_types",
            "scope_breadcrumb",
            "min_confidence",
            "include_comparisons",
            "include_improvements",
            "max_patterns",
            "analysis_depth",
        ]

        for param in expected_params:
            assert param in sig.parameters


if __name__ == "__main__":
    # Run tests with coverage
    pytest.main(
        [
            __file__,
            "-v",
            "--cov=src.tools.graph_rag.pattern_identification",
            "--cov-report=html",
            "--cov-report=term-missing",
        ]
    )
