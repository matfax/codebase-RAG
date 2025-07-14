"""Unit tests for search tools implementation.

This module contains comprehensive tests for the search functionality,
including bug fixes for n_results parameter and empty content handling.
"""

from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.tools.core.errors import EmbeddingError, QdrantConnectionError, ValidationError
from src.tools.indexing.search_tools import (
    _expand_search_context,
    _perform_hybrid_search,
    _truncate_content,
    format_search_result_summary,
    get_target_project_collections,
    search_sync,
    validate_search_parameters,
)


class TestPerformHybridSearch:
    """Test suite for _perform_hybrid_search function."""

    def test_n_results_parameter_fix(self):
        """Test that n_results parameter is correctly applied across multiple collections."""
        # Mock Qdrant client
        mock_client = Mock()

        # Create mock search results for multiple collections
        mock_result1 = Mock()
        mock_result1.score = 0.9
        mock_result1.payload = {"content": "def function1(): pass", "file_path": "/test/file1.py", "line_start": 1, "line_end": 2}

        mock_result2 = Mock()
        mock_result2.score = 0.8
        mock_result2.payload = {"content": "def function2(): pass", "file_path": "/test/file2.py", "line_start": 3, "line_end": 4}

        mock_result3 = Mock()
        mock_result3.score = 0.7
        mock_result3.payload = {"content": "def function3(): pass", "file_path": "/test/file3.py", "line_start": 5, "line_end": 6}

        # Mock search method to return different results for each collection
        def mock_search(collection_name, query_vector, query_filter, limit, score_threshold):
            if collection_name == "collection1":
                return [mock_result1, mock_result2]
            elif collection_name == "collection2":
                return [mock_result3]
            else:
                return []

        mock_client.search.side_effect = mock_search

        # Test with n_results=2 and 2 collections
        results = _perform_hybrid_search(
            qdrant_client=mock_client,
            embedding_model=Mock(),
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            search_collections=["collection1", "collection2"],
            n_results=2,
            search_mode="hybrid",
        )

        # Should return exactly 2 results (not 2 * 2 = 4)
        assert len(results) == 2

        # Results should be sorted by score (highest first)
        assert results[0]["score"] == 0.9
        assert results[1]["score"] == 0.8

        # Verify search was called with correct parameters
        assert mock_client.search.call_count == 2

        # Verify per_collection_limit was set correctly (should be max(n_results, 10))
        for call in mock_client.search.call_args_list:
            args, kwargs = call
            assert kwargs["limit"] == 10  # max(2, 10) = 10

    def test_empty_content_filtering(self):
        """Test that results with empty content are filtered out."""
        mock_client = Mock()

        # Create mock results with various content states
        mock_result_valid = Mock()
        mock_result_valid.score = 0.9
        mock_result_valid.payload = {"content": "def valid_function(): pass", "file_path": "/test/file1.py"}

        mock_result_empty = Mock()
        mock_result_empty.score = 0.8
        mock_result_empty.payload = {"content": "", "file_path": "/test/file2.py"}  # Empty content

        mock_result_whitespace = Mock()
        mock_result_whitespace.score = 0.7
        mock_result_whitespace.payload = {"content": "   \n\t  ", "file_path": "/test/file3.py"}  # Whitespace only

        mock_result_missing = Mock()
        mock_result_missing.score = 0.6
        mock_result_missing.payload = {
            "file_path": "/test/file4.py"
            # Missing content field
        }

        mock_client.search.return_value = [mock_result_valid, mock_result_empty, mock_result_whitespace, mock_result_missing]

        results = _perform_hybrid_search(
            qdrant_client=mock_client,
            embedding_model=Mock(),
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            search_collections=["collection1"],
            n_results=10,
            search_mode="hybrid",
        )

        # Should only return the valid result
        assert len(results) == 1
        assert results[0]["content"] == "def valid_function(): pass"
        assert results[0]["score"] == 0.9

    def test_cross_collection_result_aggregation(self):
        """Test correct aggregation and sorting of results across collections."""
        mock_client = Mock()

        # Create results with different scores across collections
        def mock_search(collection_name, query_vector, query_filter, limit, score_threshold):
            if collection_name == "collection1":
                result1 = Mock()
                result1.score = 0.6
                result1.payload = {"content": "content1", "file_path": "/test/file1.py"}

                result2 = Mock()
                result2.score = 0.4
                result2.payload = {"content": "content2", "file_path": "/test/file2.py"}

                return [result1, result2]

            elif collection_name == "collection2":
                result3 = Mock()
                result3.score = 0.8
                result3.payload = {"content": "content3", "file_path": "/test/file3.py"}

                result4 = Mock()
                result4.score = 0.2
                result4.payload = {"content": "content4", "file_path": "/test/file4.py"}

                return [result3, result4]

            return []

        mock_client.search.side_effect = mock_search

        results = _perform_hybrid_search(
            qdrant_client=mock_client,
            embedding_model=Mock(),
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            search_collections=["collection1", "collection2"],
            n_results=3,
            search_mode="hybrid",
        )

        # Should return 3 results sorted by score
        assert len(results) == 3
        assert results[0]["score"] == 0.8  # From collection2
        assert results[1]["score"] == 0.6  # From collection1
        assert results[2]["score"] == 0.4  # From collection1

        # Verify content matches
        assert results[0]["content"] == "content3"
        assert results[1]["content"] == "content1"
        assert results[2]["content"] == "content2"

    def test_search_with_metadata_extractor(self):
        """Test search with custom metadata extractor."""
        mock_client = Mock()

        mock_result = Mock()
        mock_result.score = 0.8
        mock_result.payload = {"content": "test content", "file_path": "/test/file.py", "custom_field": "custom_value"}

        mock_client.search.return_value = [mock_result]

        def custom_metadata_extractor(payload):
            return {"extracted_field": payload.get("custom_field", "")}

        results = _perform_hybrid_search(
            qdrant_client=mock_client,
            embedding_model=Mock(),
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            search_collections=["collection1"],
            n_results=5,
            search_mode="hybrid",
            metadata_extractor=custom_metadata_extractor,
        )

        assert len(results) == 1
        assert results[0]["extracted_field"] == "custom_value"

    def test_error_handling_in_search(self):
        """Test error handling when search fails for some collections."""
        mock_client = Mock()

        # First collection succeeds, second fails
        def mock_search(collection_name, query_vector, query_filter, limit, score_threshold):
            if collection_name == "collection1":
                result = Mock()
                result.score = 0.8
                result.payload = {"content": "valid content", "file_path": "/test/file.py"}
                return [result]
            else:
                raise Exception("Search failed for collection2")

        mock_client.search.side_effect = mock_search

        with patch("tools.indexing.search_tools.logger") as mock_logger:
            results = _perform_hybrid_search(
                qdrant_client=mock_client,
                embedding_model=Mock(),
                query="test query",
                query_embedding=[0.1, 0.2, 0.3],
                search_collections=["collection1", "collection2"],
                n_results=5,
                search_mode="hybrid",
            )

            # Should return results from successful collection
            assert len(results) == 1
            assert results[0]["content"] == "valid content"

            # Should log the error for failed collection
            mock_logger.debug.assert_called_with("Error searching collection collection2: Search failed for collection2")


class TestExpandSearchContext:
    """Test suite for _expand_search_context function."""

    def test_context_expansion_with_adjacent_chunks(self):
        """Test context expansion finds adjacent chunks."""
        mock_client = Mock()

        # Main result
        main_result = {
            "file_path": "/test/file.py",
            "line_start": 10,
            "line_end": 15,
            "content": "def main_function(): pass",
            "collection": "test_collection",
        }

        # Context chunks
        context_before = Mock()
        context_before.payload = {"content": "# Comment before", "line_start": 5, "line_end": 8, "file_path": "/test/file.py"}

        context_after = Mock()
        context_after.payload = {"content": "# Comment after", "line_start": 17, "line_end": 20, "file_path": "/test/file.py"}

        mock_client.search.return_value = [context_before, context_after]

        results = _expand_search_context(
            results=[main_result],
            qdrant_client=mock_client,
            search_collections=["test_collection"],
            context_chunks=1,
            embedding_dimension=384,
        )

        assert len(results) == 1
        assert "expanded_content" in results[0]
        assert "context_info" in results[0]
        assert "# Comment before" in results[0]["expanded_content"]
        assert "# Comment after" in results[0]["expanded_content"]
        assert results[0]["context_info"]["chunks_before"] == 1
        assert results[0]["context_info"]["chunks_after"] == 1

    def test_context_expansion_error_handling(self):
        """Test context expansion handles errors gracefully."""
        mock_client = Mock()
        mock_client.search.side_effect = Exception("Context search failed")

        main_result = {
            "file_path": "/test/file.py",
            "line_start": 10,
            "line_end": 15,
            "content": "def main_function(): pass",
            "collection": "test_collection",
        }

        with patch("tools.indexing.search_tools.logger") as mock_logger:
            results = _expand_search_context(
                results=[main_result],
                qdrant_client=mock_client,
                search_collections=["test_collection"],
                context_chunks=1,
                embedding_dimension=384,
            )

            # Should return original results without expansion
            assert len(results) == 1
            assert "expanded_content" not in results[0]

            # Should log debug message
            mock_logger.debug.assert_called()


class TestSearchValidation:
    """Test suite for search parameter validation."""

    def test_validate_search_parameters_valid_inputs(self):
        """Test validation with valid inputs."""
        errors = validate_search_parameters(query="test query", n_results=5, search_mode="hybrid", context_chunks=1)

        assert errors == []

    def test_validate_search_parameters_invalid_query(self):
        """Test validation with invalid query."""
        errors = validate_search_parameters(query="", n_results=5, search_mode="hybrid", context_chunks=1)

        assert len(errors) == 1
        assert "non-empty string" in errors[0]

    def test_validate_search_parameters_invalid_n_results(self):
        """Test validation with invalid n_results."""
        errors = validate_search_parameters(query="test query", n_results=0, search_mode="hybrid", context_chunks=1)

        assert len(errors) == 1
        assert "between 1 and 100" in errors[0]

        errors = validate_search_parameters(query="test query", n_results=101, search_mode="hybrid", context_chunks=1)

        assert len(errors) == 1
        assert "between 1 and 100" in errors[0]

    def test_validate_search_parameters_invalid_search_mode(self):
        """Test validation with invalid search mode."""
        errors = validate_search_parameters(query="test query", n_results=5, search_mode="invalid_mode", context_chunks=1)

        assert len(errors) == 1
        assert "semantic" in errors[0] and "keyword" in errors[0] and "hybrid" in errors[0]

    def test_validate_search_parameters_invalid_context_chunks(self):
        """Test validation with invalid context_chunks."""
        errors = validate_search_parameters(query="test query", n_results=5, search_mode="hybrid", context_chunks=-1)

        assert len(errors) == 1
        assert "between 0 and 5" in errors[0]


class TestSearchIntegration:
    """Integration tests for search functionality."""

    @patch("tools.indexing.search_tools.get_embeddings_manager_instance")
    @patch("tools.indexing.search_tools.get_qdrant_client")
    @patch("tools.indexing.search_tools.get_current_project")
    def test_search_sync_integration(self, mock_get_project, mock_get_client, mock_get_embeddings):
        """Test complete search workflow."""
        # Mock dependencies
        mock_client = Mock()
        mock_embeddings = Mock()
        mock_project = {"name": "test_project", "collection_prefix": "project_test"}

        mock_get_client.return_value = mock_client
        mock_get_embeddings.return_value = mock_embeddings
        mock_get_project.return_value = mock_project

        # Mock collections
        mock_collection = Mock()
        mock_collection.name = "project_test_code"
        mock_client.get_collections.return_value = Mock(collections=[mock_collection])

        # Mock embedding generation
        mock_embeddings.generate_embeddings.return_value = [0.1, 0.2, 0.3]

        # Mock search results
        mock_result = Mock()
        mock_result.score = 0.8
        mock_result.payload = {
            "content": "def test_function(): pass",
            "file_path": "/test/file.py",
            "line_start": 1,
            "line_end": 2,
            "language": "python",
            "chunk_type": "function",
        }

        mock_client.search.return_value = [mock_result]

        # Execute search
        result = search_sync(
            query="test function", n_results=5, cross_project=False, search_mode="hybrid", include_context=False, context_chunks=0
        )

        # Verify results
        assert result["total"] == 1
        assert len(result["results"]) == 1
        assert result["results"][0]["content"] == "def test_function(): pass"
        assert result["query"] == "test function"
        assert result["search_mode"] == "hybrid"

        # Verify embedding was generated
        mock_embeddings.generate_embeddings.assert_called_once()

        # Verify search was performed
        mock_client.search.assert_called_once()

    @patch("tools.indexing.search_tools.get_embeddings_manager_instance")
    def test_search_sync_embedding_error(self, mock_get_embeddings):
        """Test search handles embedding generation errors."""
        mock_embeddings = Mock()
        mock_embeddings.generate_embeddings.side_effect = Exception("Embedding failed")
        mock_get_embeddings.return_value = mock_embeddings

        result = search_sync(query="test query", n_results=5)

        assert "error" in result
        assert result["error_type"] == "EmbeddingError"
        assert result["total"] == 0

    def test_search_sync_validation_error(self):
        """Test search handles validation errors."""
        result = search_sync(query="", n_results=5)  # Invalid empty query

        assert "error" in result
        assert result["error_type"] == "ValidationError"
        assert result["total"] == 0


class TestUtilityFunctions:
    """Test suite for utility functions."""

    def test_truncate_content_no_truncation(self):
        """Test content truncation when content is short."""
        content = "short content"
        result = _truncate_content(content, max_length=100)
        assert result == "short content"

    def test_truncate_content_with_truncation(self):
        """Test content truncation when content exceeds limit."""
        content = "a" * 2000
        result = _truncate_content(content, max_length=1500)
        assert len(result) == 1500 + len("\n... (truncated)")
        assert result.endswith("\n... (truncated)")

    def test_format_search_result_summary_empty(self):
        """Test summary formatting with empty results."""
        summary = format_search_result_summary([], "test query")
        assert "No results found" in summary
        assert "test query" in summary

    def test_format_search_result_summary_with_results(self):
        """Test summary formatting with results."""
        results = [
            {"type": "code", "file_path": "/test/file1.py", "score": 0.8},
            {"type": "code", "file_path": "/test/file2.py", "score": 0.6},
            {"type": "docs", "file_path": "/test/readme.md", "score": 0.4},
        ]

        summary = format_search_result_summary(results, "test query")
        assert "Found 3 results" in summary
        assert "test query" in summary
        assert "2 code" in summary
        assert "1 docs" in summary
        assert "Across 3 files" in summary
        assert "0.800" in summary  # Top score

    def test_get_target_project_collections_exact_match(self):
        """Test getting collections for target projects with exact match."""
        all_collections = [
            "project_MyProject_code",
            "project_MyProject_docs",
            "project_OtherProject_code",
            "project_MyProject_file_metadata",
        ]

        result = get_target_project_collections(["MyProject"], all_collections)

        assert len(result) == 2
        assert "project_MyProject_code" in result
        assert "project_MyProject_docs" in result
        assert "project_MyProject_file_metadata" not in result

    def test_get_target_project_collections_normalized_match(self):
        """Test getting collections with normalized name matching."""
        all_collections = ["project_my_project_code", "project_my_project_docs"]

        result = get_target_project_collections(["My-Project"], all_collections)

        assert len(result) == 2
        assert "project_my_project_code" in result
        assert "project_my_project_docs" in result

    def test_get_target_project_collections_partial_match(self):
        """Test getting collections with partial matching."""
        all_collections = ["project_my_awesome_project_code", "project_other_project_code"]

        result = get_target_project_collections(["awesome"], all_collections)

        assert len(result) == 1
        assert "project_my_awesome_project_code" in result


class TestSearchBugFixes:
    """Test suite specifically for verifying bug fixes."""

    def test_n_results_bug_fix_verification(self):
        """Verify that the n_results multiplication bug is fixed."""
        mock_client = Mock()

        # Create enough results to test the bug
        mock_results = []
        for i in range(15):  # 15 results across collections
            result = Mock()
            result.score = 0.9 - (i * 0.01)  # Decreasing scores
            result.payload = {"content": f"content_{i}", "file_path": f"/test/file_{i}.py"}
            mock_results.append(result)

        # Mock search to return 5 results per collection
        def mock_search(collection_name, query_vector, query_filter, limit, score_threshold):
            if collection_name == "collection1":
                return mock_results[0:5]
            elif collection_name == "collection2":
                return mock_results[5:10]
            elif collection_name == "collection3":
                return mock_results[10:15]
            return []

        mock_client.search.side_effect = mock_search

        # Test with n_results=7 and 3 collections
        # Before fix: would return 7 * 3 = 21 results
        # After fix: should return exactly 7 results
        results = _perform_hybrid_search(
            qdrant_client=mock_client,
            embedding_model=Mock(),
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            search_collections=["collection1", "collection2", "collection3"],
            n_results=7,
            search_mode="hybrid",
        )

        # Should return exactly 7 results, not 21
        assert len(results) == 7

        # Results should be sorted by score (highest first)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

        # Should include top results from all collections
        assert results[0]["score"] == 0.9  # Best from collection1
        assert results[1]["score"] == 0.89  # Second best from collection1
        # ... and so on

    def test_empty_content_bug_fix_verification(self):
        """Verify that empty content filtering is working correctly."""
        mock_client = Mock()

        # Mix of valid and invalid results
        results_with_empty_content = []

        # Valid result
        valid_result = Mock()
        valid_result.score = 0.9
        valid_result.payload = {"content": "def valid_function(): pass", "file_path": "/test/valid.py"}
        results_with_empty_content.append(valid_result)

        # Empty content result
        empty_result = Mock()
        empty_result.score = 0.8
        empty_result.payload = {"content": "", "file_path": "/test/empty.py"}
        results_with_empty_content.append(empty_result)

        # Whitespace-only result
        whitespace_result = Mock()
        whitespace_result.score = 0.7
        whitespace_result.payload = {"content": "   \n\t   ", "file_path": "/test/whitespace.py"}
        results_with_empty_content.append(whitespace_result)

        # Another valid result
        valid_result2 = Mock()
        valid_result2.score = 0.6
        valid_result2.payload = {"content": "class ValidClass: pass", "file_path": "/test/valid2.py"}
        results_with_empty_content.append(valid_result2)

        mock_client.search.return_value = results_with_empty_content

        results = _perform_hybrid_search(
            qdrant_client=mock_client,
            embedding_model=Mock(),
            query="test query",
            query_embedding=[0.1, 0.2, 0.3],
            search_collections=["collection1"],
            n_results=10,
            search_mode="hybrid",
        )

        # Should only return the 2 valid results
        assert len(results) == 2
        assert results[0]["content"] == "def valid_function(): pass"
        assert results[1]["content"] == "class ValidClass: pass"

        # Should maintain score ordering
        assert results[0]["score"] == 0.9
        assert results[1]["score"] == 0.6
