"""Tests for enhanced search tool functionality with target_projects parameter."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from src.tools.indexing.multi_modal_search_tools import multi_modal_search

# Import the search function from the correct module
from src.tools.indexing.search_tools import search_sync


class TestAdvancedSearchTool:
    """Test suite for enhanced search tool with target_projects functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.test_query = "authentication function"
        self.mock_collections = [
            "project_service1_code",
            "project_service1_config",
            "project_service1_documentation",
            "project_service2_code",
            "project_service2_config",
            "dir_frontend_code",
            "dir_frontend_documentation",
            "global_code",
        ]

        self.mock_search_results = [
            {
                "score": 0.95,
                "collection": "project_service1_code",
                "file_path": "/path/to/auth.py",
                "content": "def authenticate_user(username, password):",
                "chunk_type": "function",
                "language": "python",
                "project": "service1",
            },
            {
                "score": 0.87,
                "collection": "project_service2_code",
                "file_path": "/path/to/auth_service.py",
                "content": "class AuthenticationService:",
                "chunk_type": "class",
                "language": "python",
                "project": "service2",
            },
        ]

    @patch("src.tools.indexing.search_tools.get_qdrant_client")
    @patch("src.tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_target_projects_parameter_validation(self, mock_embeddings, mock_qdrant):
        """Test validation of target_projects parameter."""

        # Test invalid target_projects types
        result = search_sync(
            query=self.test_query,
            target_projects="invalid_type",  # Should be list
        )
        assert "error" in result
        assert "target_projects must be a list" in result["error"]

        # Test non-string elements in target_projects
        result = search_sync(
            query=self.test_query,
            target_projects=["valid", 123, "also_valid"],  # 123 is not string
        )
        assert "error" in result
        assert "All project names in target_projects must be strings" in result["error"]

        # Test empty target_projects list
        result = search_sync(
            query=self.test_query,
            target_projects=[],  # Empty list
        )
        assert "error" in result
        assert "target_projects cannot be empty if specified" in result["error"]

    @patch("src.tools.indexing.search_tools.get_qdrant_client")
    @patch("src.tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_target_projects_collection_filtering(self, mock_embeddings, mock_qdrant):
        """Test that target_projects correctly filters collections."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name=name) for name in self.mock_collections]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        mock_client.search.return_value = []

        # Test searching specific projects
        result = search_sync(query=self.test_query, target_projects=["service1", "service2"])

        # Should not have error
        assert "error" not in result

        # Should include target_projects in response
        assert result["target_projects"] == ["service1", "service2"]
        assert "specific projects: service1, service2" in result["search_scope"]

    @patch("src.tools.indexing.search_tools.get_qdrant_client")
    @patch("src.tools.indexing.search_tools.get_embeddings_manager_instance")
    @patch("src.tools.project.project_utils.get_available_project_names")
    def test_target_projects_not_found_error(self, mock_get_projects, mock_embeddings, mock_qdrant):
        """Test error handling when target projects are not found."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name=name) for name in self.mock_collections]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        mock_get_projects.return_value = ["service1", "service2", "frontend"]

        # Test searching for non-existent projects
        result = search_sync(query=self.test_query, target_projects=["nonexistent1", "nonexistent2"])

        # Should return error with available projects
        assert "error" in result
        assert "No indexed collections found for projects" in result["error"]
        assert result["available_projects"] == ["service1", "service2", "frontend"]

    @patch("src.tools.indexing.search_tools.get_qdrant_client")
    @patch("src.tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_target_projects_search_execution(self, mock_embeddings, mock_qdrant):
        """Test actual search execution with target_projects."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name=name) for name in self.mock_collections]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        # Mock search results
        mock_search_result_objects = []
        for result_data in self.mock_search_results:
            mock_result = Mock()
            mock_result.score = result_data["score"]
            mock_result.payload = {
                "file_path": result_data["file_path"],
                "content": result_data["content"],
                "chunk_type": result_data["chunk_type"],
                "language": result_data["language"],
                "project": result_data["project"],
                "line_start": 1,
                "line_end": 5,
            }
            mock_search_result_objects.append(mock_result)

        mock_client.search.return_value = mock_search_result_objects

        # Execute search with target projects
        result = search_sync(query=self.test_query, target_projects=["service1", "service2"])

        # Verify results
        assert "error" not in result
        assert result["total"] == len(self.mock_search_results)
        assert result["target_projects"] == ["service1", "service2"]
        assert len(result["results"]) == len(self.mock_search_results)

        # Verify collections searched
        expected_collections = [
            "project_service1_code",
            "project_service1_config",
            "project_service1_documentation",
            "project_service2_code",
            "project_service2_config",
        ]
        searched_collections = result["collections_searched"]
        for expected_collection in expected_collections:
            assert expected_collection in searched_collections

        # Should not include other projects
        assert "dir_frontend_code" not in searched_collections
        assert "global_code" not in searched_collections

    @patch("src.tools.indexing.search_tools.get_qdrant_client")
    @patch("src.tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_target_projects_with_hyphens_and_spaces(self, mock_embeddings, mock_qdrant):
        """Test project name normalization for names with hyphens and spaces."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        # Mock collections with normalized names
        normalized_collections = [
            "project_my_service_code",
            "project_my_service_config",
            "project_another_app_code",
        ]

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name=name) for name in normalized_collections]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        mock_client.search.return_value = []

        # Test with project names containing hyphens and spaces
        result = search_sync(
            query=self.test_query,
            target_projects=["my-service", "Another App"],  # Will be normalized
        )

        # Should not have error (collections should be found after normalization)
        assert "error" not in result
        assert result["target_projects"] == ["my-service", "Another App"]

    @patch("src.tools.indexing.search_tools.get_qdrant_client")
    @patch("src.tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_target_projects_none_uses_default_behavior(self, mock_embeddings, mock_qdrant):
        """Test that target_projects=None uses default search behavior."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name=name) for name in self.mock_collections]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        mock_client.search.return_value = []

        with patch("tools.indexing.search_tools.get_current_project") as mock_get_project:
            mock_get_project.return_value = {
                "name": "test_project",
                "collection_prefix": "project_test_project",
            }

            # Test with target_projects=None (default behavior)
            result = search_sync(query=self.test_query, target_projects=None)

            # Should use current project behavior
            assert "error" not in result
            assert result["target_projects"] is None
            assert result["search_scope"] == "current project"


class TestSearchResultEnhancements:
    """Test enhanced search results with project information."""

    @patch("src.tools.indexing.search_tools.get_qdrant_client")
    @patch("src.tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_search_result_includes_project_metadata(self, mock_embeddings, mock_qdrant):
        """Test that search results include enhanced project metadata."""

        # Setup mocks
        mock_client = Mock()
        mock_qdrant.return_value = mock_client

        mock_collections_response = Mock()
        mock_collections_response.collections = [Mock(name="project_test_code")]
        mock_client.get_collections.return_value = mock_collections_response

        mock_embedding_manager = Mock()
        mock_embeddings.return_value = mock_embedding_manager
        mock_embedding_manager.generate_embeddings.return_value = [0.1] * 768

        # Mock search result with enhanced metadata
        mock_result = Mock()
        mock_result.score = 0.95
        mock_result.payload = {
            "file_path": "/path/to/test.py",
            "content": "def test_function():",
            "project": "test_project",
            "chunk_type": "function",
            "name": "test_function",
            "signature": "def test_function():",
            "docstring": "Test function documentation",
            "breadcrumb": "test_module.test_function",
            "line_start": 10,
            "line_end": 15,
        }

        mock_client.search.return_value = [mock_result]

        with patch("tools.indexing.search_tools.get_current_project") as mock_get_project:
            mock_get_project.return_value = {
                "name": "test_project",
                "collection_prefix": "project_test",
            }

            result = search_sync(query="test function")

            # Verify enhanced metadata is preserved
            assert len(result["results"]) == 1
            result_item = result["results"][0]

            assert result_item["project"] == "test_project"
            assert result_item["chunk_type"] == "function"
            assert result_item["name"] == "test_function"
            assert result_item["signature"] == "def test_function():"
            assert result_item["docstring"] == "Test function documentation"
            assert result_item["breadcrumb"] == "test_module.test_function"
            assert result_item["line_start"] == 10
            assert result_item["line_end"] == 15


@pytest.mark.asyncio
class TestMultiModalSearchOutput:
    """Test suite for the minimal_output parameter in multi_modal_search."""

    @pytest.fixture
    def mock_retrieval_result(self):
        """Provides a mock retrieval result object with detailed information."""
        mock_result = Mock()
        mock_result.mode_used = "hybrid"
        mock_result.total_results = 1
        mock_result.total_execution_time_ms = 123.45
        mock_result.query_analysis_time_ms = 12.3
        mock_result.retrieval_time_ms = 100.1
        mock_result.post_processing_time_ms = 11.05
        mock_result.average_confidence = 0.95
        mock_result.result_diversity_score = 0.8
        mock_result.fallback_used = False
        mock_result.cache_hit = False
        mock_result.error_message = None
        mock_result.results = [
            {
                "file_path": "/test/service.py",
                "content": "def some_function(): pass",
                "breadcrumb": "service.some_function",
                "chunk_type": "function",
                "language": "python",
                "line_start": 10,
                "line_end": 11,
                "local_score": 0.9,
                "global_score": 0.8,
                "combined_score": 0.85,
                "confidence_level": "high",
            }
        ]
        return mock_result

    @patch("src.tools.indexing.multi_modal_search_tools.get_available_project_names")
    @patch("src.tools.indexing.multi_modal_search_tools.get_multi_modal_retrieval_strategy")
    async def test_minimal_output_is_default(self, mock_get_strategy, mock_get_projects, mock_retrieval_result):
        """Verify that the default output is minimal."""
        mock_get_projects.return_value = ["test_project"]
        mock_retrieval_service = AsyncMock()
        mock_retrieval_service.search.return_value = mock_retrieval_result
        mock_get_strategy.return_value = mock_retrieval_service

        # Call with default minimal_output=True
        result = await multi_modal_search(query="test query", target_projects=["test_project"])

        # Assert that detailed keys are NOT in the response
        assert "performance" not in result
        assert "multi_modal_metadata" not in result
        assert "query_analysis" not in result

        # Assert that the result object itself is minimal
        assert "local_score" not in result["results"][0]
        assert "confidence_level" not in result["results"][0]

        # Assert that essential keys ARE in the response
        assert "query" in result
        assert "results" in result
        assert "total" in result

    @patch("src.tools.indexing.multi_modal_search_tools.get_available_project_names")
    @patch("src.tools.indexing.multi_modal_search_tools.get_multi_modal_retrieval_strategy")
    async def test_minimal_output_false_returns_full_response(self, mock_get_strategy, mock_get_projects, mock_retrieval_result):
        """Verify that minimal_output=False returns the full, detailed response."""
        mock_get_projects.return_value = ["test_project"]
        mock_retrieval_service = AsyncMock()
        mock_retrieval_service.search.return_value = mock_retrieval_result
        mock_get_strategy.return_value = mock_retrieval_service

        # Call with minimal_output=False
        result = await multi_modal_search(
            query="test query",
            target_projects=["test_project"],
            minimal_output=False,
            include_analysis=False,  # To simplify test, we assume it's not added
        )

        # Assert that detailed keys ARE in the response
        assert "performance" in result
        assert "multi_modal_metadata" in result

        # Assert that the result object is the full version
        assert "local_score" in result["results"][0]
        assert "confidence_level" in result["results"][0]
        assert result["performance"]["average_confidence"] == 0.95


if __name__ == "__main__":
    pytest.main([__file__])
