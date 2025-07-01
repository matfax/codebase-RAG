"""
Unit tests for RAG Search Strategy Service

Tests the multi-query RAG search capabilities for project exploration.
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any

from services.rag_search_strategy import (
    RAGSearchStrategy, SearchQuery, SearchResult, RAGSearchResults,
    SearchQueryType
)


class TestRAGSearchStrategy(unittest.TestCase):
    """Test cases for RAG Search Strategy service."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.search_strategy = RAGSearchStrategy()
        
        # Mock search result for testing
        self.mock_search_result = {
            "score": 0.85,
            "file_path": "src/services/user_service.py",
            "content": "def get_user(user_id: int) -> User:\n    return user_repository.find_by_id(user_id)",
            "chunk_type": "function",
            "name": "get_user",
            "signature": "get_user(user_id: int) -> User",
            "docstring": "Retrieve user by ID from the repository",
            "language": "Python",
            "breadcrumb": "user_service.py > UserService > get_user",
            "line_start": 15,
            "line_end": 17
        }
    
    def test_initialization(self):
        """Test service initialization."""
        self.assertIsNotNone(self.search_strategy.query_templates)
        self.assertIsNotNone(self.search_strategy.scoring_weights)
        
        # Check that all search query types have templates
        for query_type in SearchQueryType:
            self.assertIn(query_type, self.search_strategy.query_templates)
            self.assertGreater(len(self.search_strategy.query_templates[query_type]), 0)
    
    def test_build_search_queries_all_types(self):
        """Test building search queries for all types."""
        queries = self.search_strategy._build_search_queries(None, 5)
        
        self.assertGreater(len(queries), 0)
        
        # Check that high priority queries come first
        for i in range(len(queries) - 1):
            self.assertLessEqual(queries[i].priority, queries[i + 1].priority)
        
        # Check query structure
        for query in queries:
            self.assertIsInstance(query.query_type, SearchQueryType)
            self.assertIsInstance(query.query_text, str)
            self.assertGreater(len(query.query_text), 0)
            self.assertIn(query.priority, [1, 2, 3])
    
    def test_build_search_queries_focused(self):
        """Test building search queries for specific focus areas."""
        focus_areas = [SearchQueryType.ENTRY_POINTS, SearchQueryType.CORE_COMPONENTS]
        queries = self.search_strategy._build_search_queries(focus_areas, 3)
        
        self.assertEqual(len(queries), 2)
        query_types = [q.query_type for q in queries]
        self.assertIn(SearchQueryType.ENTRY_POINTS, query_types)
        self.assertIn(SearchQueryType.CORE_COMPONENTS, query_types)
    
    def test_get_query_priority(self):
        """Test query priority assignment."""
        # High priority
        self.assertEqual(self.search_strategy._get_query_priority(SearchQueryType.ENTRY_POINTS), 1)
        self.assertEqual(self.search_strategy._get_query_priority(SearchQueryType.ARCHITECTURE_PATTERNS), 1)
        
        # Medium priority
        self.assertEqual(self.search_strategy._get_query_priority(SearchQueryType.DATA_FLOW), 2)
        self.assertEqual(self.search_strategy._get_query_priority(SearchQueryType.API_ENDPOINTS), 2)
        
        # Low priority
        self.assertEqual(self.search_strategy._get_query_priority(SearchQueryType.DEPENDENCIES), 3)
        self.assertEqual(self.search_strategy._get_query_priority(SearchQueryType.TESTING_PATTERNS), 3)
    
    def test_enrich_search_result(self):
        """Test search result enrichment."""
        enriched = self.search_strategy._enrich_search_result(
            self.mock_search_result, 
            SearchQueryType.CORE_COMPONENTS
        )
        
        self.assertIsInstance(enriched, SearchResult)
        self.assertEqual(enriched.original_result, self.mock_search_result)
        self.assertEqual(enriched.query_type, SearchQueryType.CORE_COMPONENTS)
        self.assertGreater(enriched.confidence_score, 0.0)
        self.assertLessEqual(enriched.confidence_score, 1.0)
        self.assertGreater(len(enriched.relevance_indicators), 0)
        self.assertGreater(len(enriched.breadcrumb_context), 0)
    
    def test_calculate_confidence_score(self):
        """Test confidence score calculation."""
        score = self.search_strategy._calculate_confidence_score(
            self.mock_search_result,
            SearchQueryType.CORE_COMPONENTS
        )
        
        self.assertGreater(score, 0.0)
        self.assertLessEqual(score, 1.0)
        
        # Test with high semantic score
        high_score_result = self.mock_search_result.copy()
        high_score_result["score"] = 0.95
        high_score = self.search_strategy._calculate_confidence_score(
            high_score_result,
            SearchQueryType.CORE_COMPONENTS
        )
        
        self.assertGreater(high_score, score)
    
    def test_calculate_keyword_bonus(self):
        """Test keyword matching bonus calculation."""
        # Test entry points keywords
        bonus = self.search_strategy._calculate_keyword_bonus(
            "main function startup", 
            "main_handler", 
            "src/main.py",
            SearchQueryType.ENTRY_POINTS
        )
        self.assertGreater(bonus, 0.0)
        
        # Test no matching keywords
        no_bonus = self.search_strategy._calculate_keyword_bonus(
            "random content", 
            "random_function", 
            "src/utils/helper.py",
            SearchQueryType.ENTRY_POINTS
        )
        self.assertEqual(no_bonus, 0.0)
    
    def test_calculate_location_bonus(self):
        """Test file location bonus calculation."""
        # Test main file location
        bonus = self.search_strategy._calculate_location_bonus(
            "src/main.py",
            SearchQueryType.ENTRY_POINTS
        )
        self.assertGreater(bonus, 0.0)
        
        # Test config file location
        config_bonus = self.search_strategy._calculate_location_bonus(
            "config/settings.py",
            SearchQueryType.CONFIGURATION
        )
        self.assertGreater(config_bonus, 0.0)
        
        # Test no matching location
        no_bonus = self.search_strategy._calculate_location_bonus(
            "src/utils/helper.py",
            SearchQueryType.ENTRY_POINTS
        )
        self.assertEqual(no_bonus, 0.0)
    
    def test_extract_relevance_indicators(self):
        """Test relevance indicators extraction."""
        indicators = self.search_strategy._extract_relevance_indicators(
            self.mock_search_result,
            SearchQueryType.CORE_COMPONENTS
        )
        
        self.assertIsInstance(indicators, list)
        self.assertGreater(len(indicators), 0)
        
        # Check for expected indicators
        indicator_text = " ".join(indicators).lower()
        self.assertIn("relevance", indicator_text)
        self.assertIn("function", indicator_text)
        self.assertIn("documented", indicator_text)
    
    def test_build_breadcrumb_context(self):
        """Test breadcrumb context building."""
        # Test with existing breadcrumb
        context = self.search_strategy._build_breadcrumb_context(self.mock_search_result)
        self.assertEqual(context, "user_service.py > UserService > get_user")
        
        # Test without breadcrumb
        result_no_breadcrumb = self.mock_search_result.copy()
        del result_no_breadcrumb["breadcrumb"]
        context = self.search_strategy._build_breadcrumb_context(result_no_breadcrumb)
        self.assertIn("user_service.py", context)
        self.assertIn("get_user", context)
    
    def test_determine_component_type(self):
        """Test component type determination."""
        # Test function in service file
        component_type = self.search_strategy._determine_component_type(
            self.mock_search_result,
            SearchQueryType.CORE_COMPONENTS
        )
        self.assertEqual(component_type, "Service Function")
        
        # Test entry point
        main_result = self.mock_search_result.copy()
        main_result["name"] = "main"
        main_result["file_path"] = "main.py"
        entry_type = self.search_strategy._determine_component_type(
            main_result,
            SearchQueryType.ENTRY_POINTS
        )
        self.assertEqual(entry_type, "Entry Point Function")
    
    def test_organize_search_results(self):
        """Test search results organization."""
        mock_results = {
            SearchQueryType.ENTRY_POINTS: [
                SearchResult(
                    original_result=self.mock_search_result,
                    query_type=SearchQueryType.ENTRY_POINTS,
                    confidence_score=0.9
                )
            ],
            SearchQueryType.CORE_COMPONENTS: [
                SearchResult(
                    original_result=self.mock_search_result,
                    query_type=SearchQueryType.CORE_COMPONENTS,
                    confidence_score=0.8
                )
            ]
        }
        
        organized = self.search_strategy._organize_search_results(mock_results)
        
        self.assertIsInstance(organized, RAGSearchResults)
        self.assertEqual(len(organized.entry_points), 1)
        self.assertEqual(len(organized.core_components), 1)
        self.assertEqual(len(organized.architecture_insights), 0)
    
    def test_count_total_results(self):
        """Test total results counting."""
        results = RAGSearchResults(
            entry_points=[Mock(), Mock()],
            core_components=[Mock()],
            architecture_insights=[Mock(), Mock(), Mock()]
        )
        
        total = self.search_strategy._count_total_results(results)
        self.assertEqual(total, 6)
    
    @patch('services.rag_search_strategy.search_sync')
    def test_execute_focused_search_success(self, mock_search):
        """Test successful focused search execution."""
        # Mock successful search response
        mock_search.return_value = {
            "results": [self.mock_search_result],
            "total": 1,
            "query": "test query"
        }
        
        query = SearchQuery(
            query_type=SearchQueryType.CORE_COMPONENTS,
            query_text="core components",
            expected_results=5
        )
        
        results = self.search_strategy.execute_focused_search(query)
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], SearchResult)
        self.assertEqual(results[0].query_type, SearchQueryType.CORE_COMPONENTS)
        
        # Verify search was called with correct parameters
        mock_search.assert_called_once_with(
            query="core components",
            n_results=5,
            cross_project=False,
            search_mode="hybrid",
            include_context=True,
            context_chunks=1
        )
    
    @patch('services.rag_search_strategy.search_sync')
    def test_execute_focused_search_error(self, mock_search):
        """Test focused search with error response."""
        # Mock error response
        mock_search.return_value = {
            "error": "Search failed",
            "results": []
        }
        
        query = SearchQuery(
            query_type=SearchQueryType.ENTRY_POINTS,
            query_text="main entry",
            expected_results=3
        )
        
        results = self.search_strategy.execute_focused_search(query)
        
        self.assertEqual(len(results), 0)
    
    @patch('services.rag_search_strategy.get_current_project')
    @patch.object(RAGSearchStrategy, '_execute_sequential_searches')
    def test_execute_comprehensive_search(self, mock_sequential, mock_project):
        """Test comprehensive search execution."""
        # Mock project context
        mock_project.return_value = {
            "name": "test_project",
            "root": "/path/to/project"
        }
        
        # Mock search results
        mock_sequential.return_value = {
            SearchQueryType.ENTRY_POINTS: [
                SearchResult(
                    original_result=self.mock_search_result,
                    query_type=SearchQueryType.ENTRY_POINTS,
                    confidence_score=0.9
                )
            ]
        }
        
        results = self.search_strategy.execute_comprehensive_search(
            project_path="/path/to/project",
            enable_parallel_search=False
        )
        
        self.assertIsInstance(results, RAGSearchResults)
        self.assertEqual(results.project_context, "test_project")
        self.assertEqual(results.search_strategy, "comprehensive_multi_query")
        self.assertGreater(results.queries_executed, 0)
        self.assertGreater(results.total_results, 0)
        self.assertGreater(results.total_search_time, 0.0)
    
    def test_get_search_strategy_info(self):
        """Test search strategy information retrieval."""
        info = self.search_strategy.get_search_strategy_info()
        
        self.assertIn("available_query_types", info)
        self.assertIn("search_modes", info)
        self.assertIn("scoring_weights", info)
        self.assertIn("supported_features", info)
        
        # Check query types
        query_types = info["available_query_types"]
        self.assertIn("entry_points", query_types)
        self.assertIn("architecture_patterns", query_types)
        self.assertIn("core_components", query_types)
        
        # Check search modes
        search_modes = info["search_modes"]
        self.assertIn("semantic", search_modes)
        self.assertIn("hybrid", search_modes)
        self.assertIn("keyword", search_modes)
    
    def test_analyze_pattern_evidence_mvc(self):
        """Test pattern evidence analysis for MVC pattern."""
        # Create result that indicates MVC pattern
        mvc_result = self.mock_search_result.copy()
        mvc_result.update({
            "file_path": "src/controllers/user_controller.py",
            "content": "class UserController:\n    def create_user(self):\n        user = User.objects.create()",
            "name": "UserController",
            "docstring": "Controller for handling user operations"
        })
        
        search_result = SearchResult(
            original_result=mvc_result,
            query_type=SearchQueryType.ARCHITECTURE_PATTERNS,
            confidence_score=0.8
        )
        
        score = self.search_strategy._analyze_pattern_evidence(search_result, "mvc")
        
        self.assertGreater(score, 0.5)  # Should detect MVC indicators
    
    def test_analyze_pattern_evidence_microservices(self):
        """Test pattern evidence analysis for microservices pattern."""
        microservice_result = self.mock_search_result.copy()
        microservice_result.update({
            "file_path": "services/payment_service/api.py",
            "content": "class PaymentService:\n    async def process_payment(self):\n        # Microservice logic",
            "name": "PaymentService",
            "docstring": "Independent payment processing service"
        })
        
        search_result = SearchResult(
            original_result=microservice_result,
            query_type=SearchQueryType.ARCHITECTURE_PATTERNS,
            confidence_score=0.9
        )
        
        score = self.search_strategy._analyze_pattern_evidence(search_result, "microservices")
        
        self.assertGreater(score, 0.4)  # Should detect microservices indicators
    
    def test_analyze_pattern_evidence_no_match(self):
        """Test pattern evidence analysis with no pattern match."""
        generic_result = self.mock_search_result.copy()
        
        search_result = SearchResult(
            original_result=generic_result,
            query_type=SearchQueryType.ARCHITECTURE_PATTERNS,
            confidence_score=0.3
        )
        
        score = self.search_strategy._analyze_pattern_evidence(search_result, "mvc")
        
        # Should have low score since generic result doesn't match MVC pattern
        self.assertLess(score, 0.3)
    
    def test_calculate_pattern_confidence(self):
        """Test pattern confidence calculation."""
        evidence_list = [
            {"confidence": 0.8},
            {"confidence": 0.7},
            {"confidence": 0.6}
        ]
        
        confidence = self.search_strategy._calculate_pattern_confidence(evidence_list)
        
        self.assertGreater(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Should be influenced by evidence count bonus
        single_evidence = [{"confidence": 0.8}]
        single_confidence = self.search_strategy._calculate_pattern_confidence(single_evidence)
        
        self.assertGreater(confidence, single_confidence)
    
    def test_calculate_pattern_confidence_empty(self):
        """Test pattern confidence calculation with empty evidence."""
        confidence = self.search_strategy._calculate_pattern_confidence([])
        self.assertEqual(confidence, 0.0)
    
    @patch.object(RAGSearchStrategy, 'execute_focused_search')
    def test_detect_architecture_patterns_success(self, mock_search):
        """Test successful architecture pattern detection."""
        # Mock search results that indicate MVC pattern
        mvc_result = SearchResult(
            original_result={
                "score": 0.9,
                "file_path": "src/models/user.py",
                "content": "class User(Model):\n    name = CharField()",
                "name": "User",
                "docstring": "User model for the application",
                "line_start": 1,
                "line_end": 3
            },
            query_type=SearchQueryType.ARCHITECTURE_PATTERNS,
            confidence_score=0.85
        )
        
        mock_search.return_value = [mvc_result]
        
        result = self.search_strategy.detect_architecture_patterns(max_results=5)
        
        self.assertIn("primary_pattern", result)
        self.assertIn("all_patterns", result)
        self.assertIn("detection_summary", result)
        
        # Check that searches were executed for different patterns
        self.assertGreater(mock_search.call_count, 5)  # Should search multiple patterns
        
        # Check detection summary
        summary = result["detection_summary"]
        self.assertIn("total_patterns_analyzed", summary)
        self.assertIn("analysis_method", summary)
        self.assertEqual(summary["analysis_method"], "RAG semantic search with pattern matching")
    
    @patch.object(RAGSearchStrategy, 'execute_focused_search')
    def test_detect_architecture_patterns_no_patterns(self, mock_search):
        """Test architecture pattern detection with no clear patterns."""
        # Mock empty search results
        mock_search.return_value = []
        
        result = self.search_strategy.detect_architecture_patterns()
        
        self.assertIsNone(result["primary_pattern"])
        self.assertEqual(len(result["all_patterns"]), 0)
        self.assertEqual(result["detection_summary"]["patterns_detected"], 0)
    
    @patch.object(RAGSearchStrategy, 'execute_focused_search')
    def test_detect_architecture_patterns_error_handling(self, mock_search):
        """Test architecture pattern detection error handling."""
        # Mock search that raises an exception
        mock_search.side_effect = Exception("Search failed")
        
        result = self.search_strategy.detect_architecture_patterns()
        
        self.assertIn("error", result)
        self.assertIsNone(result["primary_pattern"])
        self.assertEqual(result["detection_summary"]["analysis_method"], "failed")


class TestSearchQuery(unittest.TestCase):
    """Test cases for SearchQuery dataclass."""
    
    def test_search_query_creation(self):
        """Test SearchQuery creation with defaults."""
        query = SearchQuery(
            query_type=SearchQueryType.ENTRY_POINTS,
            query_text="main function"
        )
        
        self.assertEqual(query.query_type, SearchQueryType.ENTRY_POINTS)
        self.assertEqual(query.query_text, "main function")
        self.assertEqual(query.priority, 1)  # Default
        self.assertEqual(query.expected_results, 5)  # Default
        self.assertEqual(query.search_mode, "hybrid")  # Default
        self.assertEqual(query.context_chunks, 1)  # Default
    
    def test_search_query_custom_values(self):
        """Test SearchQuery with custom values."""
        query = SearchQuery(
            query_type=SearchQueryType.TESTING_PATTERNS,
            query_text="test cases",
            priority=3,
            expected_results=10,
            search_mode="semantic",
            context_chunks=2
        )
        
        self.assertEqual(query.priority, 3)
        self.assertEqual(query.expected_results, 10)
        self.assertEqual(query.search_mode, "semantic")
        self.assertEqual(query.context_chunks, 2)


class TestSearchResult(unittest.TestCase):
    """Test cases for SearchResult dataclass."""
    
    def test_search_result_creation(self):
        """Test SearchResult creation."""
        original_result = {"score": 0.8, "content": "test"}
        
        result = SearchResult(
            original_result=original_result,
            query_type=SearchQueryType.CORE_COMPONENTS,
            confidence_score=0.85
        )
        
        self.assertEqual(result.original_result, original_result)
        self.assertEqual(result.query_type, SearchQueryType.CORE_COMPONENTS)
        self.assertEqual(result.confidence_score, 0.85)
        self.assertEqual(result.relevance_indicators, [])  # Default
        self.assertEqual(result.breadcrumb_context, "")  # Default


class TestRAGSearchResults(unittest.TestCase):
    """Test cases for RAGSearchResults dataclass."""
    
    def test_rag_search_results_creation(self):
        """Test RAGSearchResults creation with defaults."""
        results = RAGSearchResults()
        
        self.assertEqual(results.architecture_insights, [])
        self.assertEqual(results.entry_points, [])
        self.assertEqual(results.total_search_time, 0.0)
        self.assertEqual(results.queries_executed, 0)
        self.assertEqual(results.total_results, 0)
        self.assertEqual(results.project_context, "")
        self.assertEqual(results.search_strategy, "")
    
    def test_rag_search_results_with_data(self):
        """Test RAGSearchResults with actual data."""
        mock_result = SearchResult(
            original_result={"score": 0.8},
            query_type=SearchQueryType.ENTRY_POINTS,
            confidence_score=0.9
        )
        
        results = RAGSearchResults(
            entry_points=[mock_result],
            total_search_time=1.5,
            queries_executed=5,
            total_results=10,
            project_context="test_project",
            search_strategy="comprehensive"
        )
        
        self.assertEqual(len(results.entry_points), 1)
        self.assertEqual(results.total_search_time, 1.5)
        self.assertEqual(results.queries_executed, 5)
        self.assertEqual(results.total_results, 10)
        self.assertEqual(results.project_context, "test_project")
        self.assertEqual(results.search_strategy, "comprehensive")


if __name__ == '__main__':
    unittest.main()