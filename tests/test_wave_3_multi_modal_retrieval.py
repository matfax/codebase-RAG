"""
Test Suite for Wave 3.0 Multi-Modal Retrieval System

This test suite verifies the implementation of the LightRAG-inspired
four retrieval modes: Local, Global, Hybrid, and Mix.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.models.query_features import (
    GlobalModeConfig,
    HybridModeConfig,
    KeywordExtraction,
    LocalModeConfig,
    MixModeConfig,
    QueryComplexity,
    QueryFeatures,
    QueryType,
    RetrievalResult,
)
from src.services.multi_modal_retrieval_strategy import (
    MultiModalRetrievalStrategy,
    MultiModalSearchResult,
)
from src.services.query_analyzer import QueryAnalyzer
from src.services.retrieval_mode_performance_monitor import RetrievalModePerformanceMonitor
from src.tools.indexing.multi_modal_search_tools import (
    analyze_query_features,
    get_retrieval_mode_performance,
    multi_modal_search,
)
from src.utils.keyword_extractor import KeywordExtractor


class TestKeywordExtractor:
    """Test the keyword extraction functionality."""

    def test_keyword_extractor_initialization(self):
        """Test KeywordExtractor initializes correctly."""
        extractor = KeywordExtractor()
        assert extractor is not None
        assert hasattr(extractor, "patterns")
        assert hasattr(extractor, "technical_vocabulary")
        assert hasattr(extractor, "concept_vocabulary")

    def test_entity_focused_query_extraction(self):
        """Test extraction of entity-focused keywords."""
        extractor = KeywordExtractor()
        query = "Show me the getUserData() function implementation"

        extraction = extractor.extract_keywords(query)

        assert len(extraction.entity_names) > 0
        assert "getUserData" in extraction.entity_names or any("getUserData" in name for name in extraction.entity_names)
        assert len(extraction.low_level_keywords) > 0

    def test_relationship_focused_query_extraction(self):
        """Test extraction of relationship-focused keywords."""
        extractor = KeywordExtractor()
        query = "How does UserService connect to DatabaseManager?"

        extraction = extractor.extract_keywords(query)

        assert len(extraction.relationship_indicators) > 0
        assert any(indicator in ["connect", "connects"] for indicator in extraction.relationship_indicators)
        assert len(extraction.high_level_keywords) > 0

    def test_concept_focused_query_extraction(self):
        """Test extraction of concept-focused keywords."""
        extractor = KeywordExtractor()
        query = "Explain the authentication pattern architecture"

        extraction = extractor.extract_keywords(query)

        assert len(extraction.concept_terms) > 0
        assert any(term in ["pattern", "architecture"] for term in extraction.concept_terms)
        assert len(extraction.high_level_keywords) > 0

    def test_mode_recommendation(self):
        """Test retrieval mode recommendation."""
        extractor = KeywordExtractor()

        # Entity-focused query should recommend local mode
        entity_query = "Find the calculateTotal() method"
        mode, confidence = extractor.recommend_retrieval_mode(entity_query)
        assert mode == "local"
        assert confidence > 0.5

        # Relationship-focused query should recommend global mode
        relationship_query = "How are User and Order models connected?"
        mode, confidence = extractor.recommend_retrieval_mode(relationship_query)
        assert mode == "global"
        assert confidence > 0.5


class TestQueryAnalyzer:
    """Test the query analysis functionality."""

    @pytest.mark.asyncio
    async def test_query_analyzer_initialization(self):
        """Test QueryAnalyzer initializes correctly."""
        analyzer = QueryAnalyzer()
        assert analyzer is not None
        assert hasattr(analyzer, "keyword_extractor")
        assert hasattr(analyzer, "mode_configs")

    @pytest.mark.asyncio
    async def test_entity_focused_query_analysis(self):
        """Test analysis of entity-focused queries."""
        analyzer = QueryAnalyzer()
        query = "Show me the UserService.authenticate method"

        features = await analyzer.analyze_query(query)

        assert features.query_type == QueryType.ENTITY_FOCUSED
        assert features.has_specific_entities is True
        assert features.recommended_mode in ["local", "hybrid"]
        assert features.mode_confidence > 0.0
        assert len(features.keywords.entity_names) > 0

    @pytest.mark.asyncio
    async def test_relationship_focused_query_analysis(self):
        """Test analysis of relationship-focused queries."""
        analyzer = QueryAnalyzer()
        query = "How does the API layer connect to the database layer?"

        features = await analyzer.analyze_query(query)

        assert features.query_type == QueryType.RELATIONSHIP_FOCUSED
        assert features.has_relationships is True
        assert features.recommended_mode in ["global", "hybrid"]
        assert len(features.keywords.relationship_indicators) > 0

    @pytest.mark.asyncio
    async def test_conceptual_query_analysis(self):
        """Test analysis of conceptual queries."""
        analyzer = QueryAnalyzer()
        query = "Explain the MVC architectural pattern implementation"

        features = await analyzer.analyze_query(query)

        assert features.query_type == QueryType.CONCEPTUAL
        assert features.has_conceptual_focus is True
        assert features.recommended_mode in ["global", "hybrid"]
        assert len(features.keywords.concept_terms) > 0

    @pytest.mark.asyncio
    async def test_complex_query_analysis(self):
        """Test analysis of complex queries."""
        analyzer = QueryAnalyzer()
        query = "How does the authentication system integrate with the user management service and handle different user roles across multiple database tables?"

        features = await analyzer.analyze_query(query)

        assert features.complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_FACETED]
        assert features.recommended_mode in ["hybrid", "mix"]
        assert features.word_count > 10


class TestMultiModalRetrievalStrategy:
    """Test the multi-modal retrieval strategy."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_qdrant_service = MagicMock()
        self.mock_embedding_service = MagicMock()
        self.strategy = MultiModalRetrievalStrategy(
            qdrant_service=self.mock_qdrant_service,
            embedding_service=self.mock_embedding_service,
        )

    @pytest.mark.asyncio
    async def test_strategy_initialization(self):
        """Test MultiModalRetrievalStrategy initializes correctly."""
        assert self.strategy is not None
        assert hasattr(self.strategy, "mode_configs")
        assert hasattr(self.strategy, "performance_metrics")
        assert "local" in self.strategy.mode_configs
        assert "global" in self.strategy.mode_configs
        assert "hybrid" in self.strategy.mode_configs
        assert "mix" in self.strategy.mode_configs

    @pytest.mark.asyncio
    @patch("src.services.multi_modal_retrieval_strategy.get_hybrid_search_service")
    @patch("src.services.multi_modal_retrieval_strategy.get_query_analyzer")
    async def test_local_mode_search(self, mock_query_analyzer, mock_hybrid_search):
        """Test local mode search functionality."""
        # Mock query analyzer
        mock_analyzer = AsyncMock()
        mock_features = QueryFeatures(
            original_query="Find getUserData function",
            normalized_query="find getuserdata function",
            query_length=20,
            word_count=3,
            query_type=QueryType.ENTITY_FOCUSED,
            complexity=QueryComplexity.SIMPLE,
            confidence_score=0.8,
            keywords=KeywordExtraction(
                entity_names=["getUserData"],
                low_level_keywords=["getUserData", "function"],
                high_level_keywords=[],
            ),
            recommended_mode="local",
            mode_confidence=0.9,
        )
        mock_analyzer.analyze_query.return_value = mock_features
        mock_query_analyzer.return_value = mock_analyzer

        # Mock hybrid search service
        mock_search_service = AsyncMock()
        mock_search_summary = MagicMock()
        mock_search_summary.results = []
        mock_search_service.hybrid_search.return_value = mock_search_summary
        mock_hybrid_search.return_value = mock_search_service

        # Perform search
        result = await self.strategy.search(
            query="Find getUserData function",
            project_names=["test_project"],
            mode="local",
            enable_manual_mode_selection=True,
        )

        assert result.mode_used == "local"
        assert result.query == "Find getUserData function"
        assert result.total_execution_time_ms > 0

    @pytest.mark.asyncio
    @patch("src.services.multi_modal_retrieval_strategy.get_hybrid_search_service")
    @patch("src.services.multi_modal_retrieval_strategy.get_query_analyzer")
    async def test_global_mode_search(self, mock_query_analyzer, mock_hybrid_search):
        """Test global mode search functionality."""
        # Mock query analyzer
        mock_analyzer = AsyncMock()
        mock_features = QueryFeatures(
            original_query="How are users connected to orders?",
            normalized_query="how are users connected to orders",
            query_length=30,
            word_count=6,
            query_type=QueryType.RELATIONSHIP_FOCUSED,
            complexity=QueryComplexity.MODERATE,
            confidence_score=0.7,
            keywords=KeywordExtraction(
                relationship_indicators=["connected"],
                high_level_keywords=["users", "orders", "connected"],
                low_level_keywords=[],
            ),
            recommended_mode="global",
            mode_confidence=0.8,
        )
        mock_analyzer.analyze_query.return_value = mock_features
        mock_query_analyzer.return_value = mock_analyzer

        # Mock hybrid search service
        mock_search_service = AsyncMock()
        mock_search_summary = MagicMock()
        mock_search_summary.results = []
        mock_search_service.hybrid_search.return_value = mock_search_summary
        mock_hybrid_search.return_value = mock_search_service

        # Perform search
        result = await self.strategy.search(
            query="How are users connected to orders?",
            project_names=["test_project"],
            mode="global",
            enable_manual_mode_selection=True,
        )

        assert result.mode_used == "global"
        assert result.query == "How are users connected to orders?"

    @pytest.mark.asyncio
    @patch("src.services.multi_modal_retrieval_strategy.get_hybrid_search_service")
    @patch("src.services.multi_modal_retrieval_strategy.get_query_analyzer")
    async def test_mix_mode_selection(self, mock_query_analyzer, mock_hybrid_search):
        """Test mix mode automatic mode selection."""
        # Mock query analyzer
        mock_analyzer = AsyncMock()
        mock_features = QueryFeatures(
            original_query="Complex multi-faceted query",
            normalized_query="complex multi-faceted query",
            query_length=25,
            word_count=4,
            query_type=QueryType.EXPLORATION,
            complexity=QueryComplexity.MULTI_FACETED,
            confidence_score=0.6,
            keywords=KeywordExtraction(),
            recommended_mode="mix",
            mode_confidence=0.7,
        )
        mock_analyzer.analyze_query.return_value = mock_features
        mock_query_analyzer.return_value = mock_analyzer

        # Mock hybrid search service
        mock_search_service = AsyncMock()
        mock_search_summary = MagicMock()
        mock_search_summary.results = []
        mock_search_service.hybrid_search.return_value = mock_search_summary
        mock_hybrid_search.return_value = mock_search_service

        # Perform search
        result = await self.strategy.search(
            query="Complex multi-faceted query",
            project_names=["test_project"],
            mode="mix",
            enable_manual_mode_selection=True,
        )

        assert result.mode_used.startswith("mix(")  # Should show the sub-mode selected


class TestPerformanceMonitor:
    """Test the performance monitoring functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = RetrievalModePerformanceMonitor(max_history_size=100)

    def test_monitor_initialization(self):
        """Test RetrievalModePerformanceMonitor initializes correctly."""
        assert self.monitor is not None
        assert hasattr(self.monitor, "mode_stats")
        assert "local" in self.monitor.mode_stats
        assert "global" in self.monitor.mode_stats
        assert "hybrid" in self.monitor.mode_stats
        assert "mix" in self.monitor.mode_stats

    def test_record_successful_query(self):
        """Test recording a successful query result."""
        result = RetrievalResult(
            query="test query",
            mode_used="local",
            config_used=LocalModeConfig(),
            results=[],
            total_execution_time_ms=150.0,
            total_results=5,
            average_confidence=0.8,
            result_diversity_score=0.6,
        )

        self.monitor.record_query_result(result)

        stats = self.monitor.mode_stats["local"]
        assert stats.total_queries == 1
        assert stats.successful_queries == 1
        assert stats.failed_queries == 0
        assert stats.success_rate == 1.0
        assert stats.average_execution_time_ms == 150.0

    def test_record_failed_query(self):
        """Test recording a failed query result."""
        result = RetrievalResult(
            query="test query",
            mode_used="global",
            config_used=GlobalModeConfig(),
            results=[],
            total_execution_time_ms=200.0,
            error_message="Test error",
        )

        self.monitor.record_query_result(result)

        stats = self.monitor.mode_stats["global"]
        assert stats.total_queries == 1
        assert stats.successful_queries == 0
        assert stats.failed_queries == 1
        assert stats.success_rate == 0.0

    def test_mode_comparison(self):
        """Test comparison between different modes."""
        # Record some results for different modes
        local_result = RetrievalResult(
            query="local query",
            mode_used="local",
            config_used=LocalModeConfig(),
            results=[],
            total_execution_time_ms=100.0,
            total_results=3,
            average_confidence=0.9,
        )

        global_result = RetrievalResult(
            query="global query",
            mode_used="global",
            config_used=GlobalModeConfig(),
            results=[],
            total_execution_time_ms=200.0,
            total_results=8,
            average_confidence=0.7,
        )

        self.monitor.record_query_result(local_result)
        self.monitor.record_query_result(global_result)

        comparison = self.monitor.compare_modes(["local", "global"])

        assert comparison is not None
        assert len(comparison.modes_compared) == 2
        assert "local" in comparison.modes_compared
        assert "global" in comparison.modes_compared
        assert "average_execution_time_ms" in comparison.metrics_compared
        assert len(comparison.recommendations) > 0


class TestMCPTools:
    """Test the MCP tool integration."""

    @pytest.mark.asyncio
    @patch("src.tools.indexing.multi_modal_search_tools.get_multi_modal_retrieval_strategy")
    @patch("src.tools.indexing.multi_modal_search_tools.get_query_analyzer")
    async def test_multi_modal_search_tool(self, mock_query_analyzer, mock_strategy):
        """Test the multi_modal_search MCP tool."""
        # Mock query analyzer
        mock_analyzer = AsyncMock()
        mock_features = QueryFeatures(
            original_query="test query",
            normalized_query="test query",
            query_length=10,
            word_count=2,
            query_type=QueryType.ENTITY_FOCUSED,
            complexity=QueryComplexity.SIMPLE,
            confidence_score=0.8,
            keywords=KeywordExtraction(),
            recommended_mode="local",
            mode_confidence=0.9,
        )
        mock_analyzer.analyze_query.return_value = mock_features
        mock_query_analyzer.return_value = mock_analyzer

        # Mock retrieval strategy
        mock_retrieval_service = AsyncMock()
        mock_result = RetrievalResult(
            query="test query",
            mode_used="local",
            config_used=LocalModeConfig(),
            results=[],
            total_execution_time_ms=100.0,
            total_results=5,
            average_confidence=0.8,
            result_diversity_score=0.6,
        )
        mock_retrieval_service.search.return_value = mock_result
        mock_strategy.return_value = mock_retrieval_service

        # Mock current project
        with patch("src.tools.indexing.multi_modal_search_tools.get_current_project") as mock_project:
            mock_project.return_value = {"name": "test_project"}

            response = await multi_modal_search(
                query="test query",
                n_results=10,
                mode="local",
                enable_manual_mode_selection=True,
                include_analysis=True,
            )

        assert response is not None
        assert response["query"] == "test query"
        assert response["mode_used"] == "local"
        assert response["total"] == 5
        assert "performance" in response
        assert "query_analysis" in response

    @pytest.mark.asyncio
    async def test_analyze_query_features_tool(self):
        """Test the analyze_query_features MCP tool."""
        with patch("src.tools.indexing.multi_modal_search_tools.get_query_analyzer") as mock_analyzer_getter:
            mock_analyzer = AsyncMock()
            mock_features = QueryFeatures(
                original_query="test query",
                normalized_query="test query",
                query_length=10,
                word_count=2,
                query_type=QueryType.ENTITY_FOCUSED,
                complexity=QueryComplexity.SIMPLE,
                confidence_score=0.8,
                keywords=KeywordExtraction(entity_names=["test"]),
                recommended_mode="local",
                mode_confidence=0.9,
                has_specific_entities=True,
            )
            mock_analyzer.analyze_query.return_value = mock_features
            mock_analyzer_getter.return_value = mock_analyzer

            response = await analyze_query_features("test query")

        assert response is not None
        assert response["query"] == "test query"
        assert response["analysis"]["query_type"] == "entity_focused"
        assert response["recommendation"]["recommended_mode"] == "local"
        assert "keywords" in response

    @pytest.mark.asyncio
    async def test_get_retrieval_mode_performance_tool(self):
        """Test the get_retrieval_mode_performance MCP tool."""
        with patch("src.tools.indexing.multi_modal_search_tools.get_performance_monitor") as mock_monitor_getter:
            mock_monitor = MagicMock()
            mock_monitor.get_mode_statistics.return_value = {
                "local": {
                    "mode_name": "local",
                    "total_queries": 10,
                    "successful_queries": 9,
                    "success_rate": 0.9,
                }
            }
            mock_monitor.compare_modes.return_value = MagicMock(
                timestamp="2024-07-24T12:00:00",
                modes_compared=["local", "global"],
                metrics_compared={},
                best_performing_mode={},
                recommendations=[],
            )
            mock_monitor.get_active_alerts.return_value = []
            mock_monitor_getter.return_value = mock_monitor

            response = await get_retrieval_mode_performance(
                mode="local",
                include_comparison=True,
                include_alerts=True,
            )

        assert response is not None
        assert "metrics" in response
        assert "summary" in response
        assert response["summary"]["monitoring_status"] == "active"


@pytest.mark.asyncio
async def test_end_to_end_workflow():
    """Test the complete end-to-end workflow."""
    # This test would require more complex mocking and setup
    # For now, we'll just verify the components can be instantiated together

    extractor = KeywordExtractor()
    analyzer = QueryAnalyzer()

    # Test basic functionality
    query = "Show me the authentication service implementation"
    extraction = extractor.extract_keywords(query)
    assert extraction is not None

    features = await analyzer.analyze_query(query)
    assert features is not None
    assert features.recommended_mode in ["local", "global", "hybrid", "mix"]


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
