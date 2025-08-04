"""
Test Suite for Enhanced Complexity Detection System

This test suite validates the complexity detection functionality across
multiple dimensions and ensures accurate complexity classification.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest

from ..models.query_features import KeywordExtraction, QueryComplexity
from .complexity_detector import ComplexityDimension, ComplexityProfile, EnhancedComplexityDetector, get_complexity_detector


class TestEnhancedComplexityDetector:
    """Test cases for the EnhancedComplexityDetector class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.detector = EnhancedComplexityDetector()

    @pytest.mark.asyncio
    async def test_simple_query_complexity(self):
        """Test complexity analysis for simple queries."""

        simple_queries = ["find user", "show class", "what is API", "hello world", "get data"]

        for query in simple_queries:
            profile = await self.detector.analyze_complexity(query)

            assert profile.overall_complexity in [QueryComplexity.SIMPLE, QueryComplexity.MODERATE]
            assert 0.0 <= profile.complexity_score <= 0.6
            assert profile.confidence > 0.3
            assert profile.analysis_quality > 0.5

    @pytest.mark.asyncio
    async def test_moderate_query_complexity(self):
        """Test complexity analysis for moderate queries."""

        moderate_queries = [
            "find all users who have admin privileges",
            "show me the connection between classes and interfaces",
            "how does authentication work in this system",
            "what are the best practices for error handling",
            "implement a simple caching mechanism",
        ]

        for query in moderate_queries:
            profile = await self.detector.analyze_complexity(query)

            assert profile.overall_complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
            assert 0.2 <= profile.complexity_score <= 0.8
            assert profile.confidence > 0.4

    @pytest.mark.asyncio
    async def test_complex_query_complexity(self):
        """Test complexity analysis for complex queries."""

        complex_queries = [
            "analyze the architectural patterns used in this microservice infrastructure and compare them with monolithic approaches",
            "implement a distributed caching system that handles concurrent access patterns while maintaining consistency",
            "explain how the authentication middleware integrates with the authorization service and session management",
            "design a scalable data processing pipeline that can handle real-time analytics and batch processing",
        ]

        for query in complex_queries:
            profile = await self.detector.analyze_complexity(query)

            assert profile.overall_complexity in [QueryComplexity.COMPLEX, QueryComplexity.MULTI_FACETED]
            assert profile.complexity_score >= 0.4
            assert profile.confidence > 0.3

    @pytest.mark.asyncio
    async def test_multi_faceted_query_complexity(self):
        """Test complexity analysis for multi-faceted queries."""

        multi_faceted_queries = [
            "provide a comprehensive analysis of the system architecture including performance optimization strategies, security considerations, scalability patterns, and integration approaches while considering the trade-offs between different implementation methodologies",
            "design and implement a full-stack solution that incorporates machine learning algorithms, real-time data processing, user authentication, API management, database optimization, and monitoring systems",
        ]

        for query in multi_faceted_queries:
            profile = await self.detector.analyze_complexity(query)

            assert profile.overall_complexity == QueryComplexity.MULTI_FACETED
            assert profile.complexity_score >= 0.6
            # Multi-faceted queries should have high complexity in multiple dimensions
            high_complexity_dims = sum(
                1
                for dim_score in [
                    profile.lexical_complexity,
                    profile.syntactic_complexity,
                    profile.semantic_complexity,
                    profile.computational_complexity,
                ]
                if dim_score > 0.5
            )
            assert high_complexity_dims >= 2

    @pytest.mark.asyncio
    async def test_lexical_complexity_analysis(self):
        """Test lexical complexity dimension analysis."""

        # High lexical complexity query
        high_lexical_query = "instantiate polymorphic encapsulation methodology using sophisticated algorithmic optimization"
        profile = await self.detector.analyze_complexity(high_lexical_query)
        assert profile.lexical_complexity > 0.6

        # Low lexical complexity query
        low_lexical_query = "get user data from table"
        profile = await self.detector.analyze_complexity(low_lexical_query)
        assert profile.lexical_complexity < 0.5

    @pytest.mark.asyncio
    async def test_syntactic_complexity_analysis(self):
        """Test syntactic complexity dimension analysis."""

        # High syntactic complexity query
        high_syntactic_query = (
            "find all users who (have admin privileges and (created_date > '2023-01-01' or status == 'active')) but not archived"
        )
        profile = await self.detector.analyze_complexity(high_syntactic_query)
        assert profile.syntactic_complexity > 0.4

        # Low syntactic complexity query
        low_syntactic_query = "show users"
        profile = await self.detector.analyze_complexity(low_syntactic_query)
        assert profile.syntactic_complexity < 0.4

    @pytest.mark.asyncio
    async def test_semantic_complexity_analysis(self):
        """Test semantic complexity dimension analysis."""

        # High semantic complexity query
        high_semantic_query = "analyze the conceptual framework underlying the architectural paradigm and its methodological implications"
        profile = await self.detector.analyze_complexity(high_semantic_query)
        assert profile.semantic_complexity > 0.5

        # Low semantic complexity query
        low_semantic_query = "get user by id"
        profile = await self.detector.analyze_complexity(low_semantic_query)
        assert profile.semantic_complexity < 0.5

    @pytest.mark.asyncio
    async def test_computational_complexity_analysis(self):
        """Test computational complexity dimension analysis."""

        # High computational complexity query
        high_computational_query = "optimize the recursive algorithm for processing large datasets with parallel execution and caching"
        profile = await self.detector.analyze_complexity(high_computational_query)
        assert profile.computational_complexity > 0.4

        # Low computational complexity query
        low_computational_query = "display user name"
        profile = await self.detector.analyze_complexity(low_computational_query)
        assert profile.computational_complexity < 0.4

    @pytest.mark.asyncio
    async def test_domain_complexity_analysis(self):
        """Test domain-specific complexity dimension analysis."""

        keywords = KeywordExtraction(
            technical_terms=["API", "microservice", "authentication", "middleware"],
            domain_specific_keywords=["OAuth", "JWT", "REST", "GraphQL"],
        )

        # High domain complexity query
        high_domain_query = "implement OAuth2 authentication with JWT tokens for GraphQL API endpoints"
        profile = await self.detector.analyze_complexity(high_domain_query, keywords=keywords)
        assert profile.domain_complexity > 0.4

        # Low domain complexity query
        low_domain_query = "hello world example"
        profile = await self.detector.analyze_complexity(low_domain_query)
        assert profile.domain_complexity < 0.3

    @pytest.mark.asyncio
    async def test_context_aware_threshold_adjustment(self):
        """Test dynamic threshold adjustment based on context."""

        programming_context = {"detected_domains": {"programming": 0.9}, "technology_stack": ["python", "javascript"]}

        query = "implement function with parameters"

        # Test with programming context
        profile_with_context = await self.detector.analyze_complexity(query, context_hints=programming_context)

        # Test without context
        profile_without_context = await self.detector.analyze_complexity(query)

        # Programming context might affect complexity assessment
        assert isinstance(profile_with_context.overall_complexity, QueryComplexity)
        assert isinstance(profile_without_context.overall_complexity, QueryComplexity)

    @pytest.mark.asyncio
    async def test_advanced_metrics_calculation(self):
        """Test advanced metrics calculation."""

        query = "analyze system architecture patterns and implementation strategies"
        profile = await self.detector.analyze_complexity(query)

        # Check that advanced metrics are calculated
        assert 0.0 <= profile.complexity_variance <= 1.0
        assert 0.0 <= profile.complexity_stability <= 1.0
        assert 0.0 <= profile.processing_difficulty <= 1.0
        assert isinstance(profile.word_complexity_distribution, dict)
        assert isinstance(profile.phrase_complexity_scores, list)
        assert isinstance(profile.structural_patterns, list)

    @pytest.mark.asyncio
    async def test_analysis_quality_assessment(self):
        """Test analysis quality assessment."""

        # Good quality query (clear, well-formed)
        good_query = "find all users with admin privileges created after 2023"
        profile = await self.detector.analyze_complexity(good_query)
        assert profile.analysis_quality > 0.7

        # Poor quality query (very short, ambiguous)
        poor_query = "it"
        profile = await self.detector.analyze_complexity(poor_query)
        assert profile.analysis_quality < 0.7
        assert "very_short_query" in profile.uncertainty_factors

    @pytest.mark.asyncio
    async def test_caching_functionality(self):
        """Test caching of complexity analysis results."""

        query = "test caching functionality"

        # First analysis (cache miss)
        start_time = time.time()
        profile1 = await self.detector.analyze_complexity(query)
        first_time = time.time() - start_time

        # Second analysis (should be cache hit)
        start_time = time.time()
        profile2 = await self.detector.analyze_complexity(query)
        second_time = time.time() - start_time

        # Results should be identical
        assert profile1.complexity_score == profile2.complexity_score
        assert profile1.overall_complexity == profile2.overall_complexity

        # Second analysis should be faster (cached)
        assert second_time < first_time or second_time < 0.001  # Very fast for cache hit

    @pytest.mark.asyncio
    async def test_batch_analysis(self):
        """Test batch complexity analysis."""

        queries = [
            "simple query",
            "moderately complex query with multiple concepts",
            "highly complex architectural analysis with sophisticated methodology",
        ]

        profiles = await self.detector.batch_analyze_complexity(queries)

        assert len(profiles) == len(queries)

        # Should generally show increasing complexity
        assert profiles[0].complexity_score <= profiles[1].complexity_score
        assert profiles[1].complexity_score <= profiles[2].complexity_score

    @pytest.mark.asyncio
    async def test_fallback_profile_creation(self):
        """Test fallback profile creation on analysis failure."""

        # Simulate analysis failure
        with patch.object(self.detector, "_analyze_all_dimensions", side_effect=Exception("Test error")):
            profile = await self.detector.analyze_complexity("test query")

            # Should return a valid fallback profile
            assert isinstance(profile, ComplexityProfile)
            assert isinstance(profile.overall_complexity, QueryComplexity)
            assert 0.0 <= profile.complexity_score <= 1.0
            assert profile.confidence < 0.5  # Low confidence for fallback
            assert "analysis_fallback" in profile.uncertainty_factors

    def test_statistics_tracking(self):
        """Test analysis statistics tracking."""

        initial_stats = self.detector.get_analysis_statistics()
        assert "analyses_performed" in initial_stats
        assert "cache_hit_rate" in initial_stats
        assert "complexity_distribution" in initial_stats

    def test_cache_management(self):
        """Test cache management functionality."""

        initial_cache_size = len(self.detector.analysis_cache)

        # Clear cache
        self.detector.clear_cache()

        # Cache should be empty
        assert len(self.detector.analysis_cache) == 0

    @pytest.mark.asyncio
    async def test_edge_cases(self):
        """Test edge cases and boundary conditions."""

        edge_cases = [
            "",  # Empty string
            " ",  # Whitespace only
            "a",  # Single character
            "a" * 1000,  # Very long string
            "123 456 789",  # Numbers only
            "!@#$%^&*()",  # Special characters only
            "CamelCaseWordsEverywhere",  # All CamelCase
            "ALLUPPERCASE",  # All uppercase
        ]

        for query in edge_cases:
            profile = await self.detector.analyze_complexity(query)

            # Should always return valid profile
            assert isinstance(profile, ComplexityProfile)
            assert isinstance(profile.overall_complexity, QueryComplexity)
            assert 0.0 <= profile.complexity_score <= 1.0
            assert 0.0 <= profile.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_consistency_across_runs(self):
        """Test consistency of complexity analysis across multiple runs."""

        query = "analyze system architecture patterns"

        # Run analysis multiple times
        profiles = []
        for _ in range(5):
            profile = await self.detector.analyze_complexity(query)
            profiles.append(profile)

        # Results should be consistent (identical due to caching)
        first_profile = profiles[0]
        for profile in profiles[1:]:
            assert profile.complexity_score == first_profile.complexity_score
            assert profile.overall_complexity == first_profile.overall_complexity


class TestComplexityDetectorFactory:
    """Test the factory function for ComplexityDetector."""

    def test_singleton_pattern(self):
        """Test that factory returns singleton instance."""

        detector1 = get_complexity_detector()
        detector2 = get_complexity_detector()

        assert detector1 is detector2
        assert isinstance(detector1, EnhancedComplexityDetector)


# Integration tests
class TestComplexityDetectorIntegration:
    """Integration tests for complexity detector with other components."""

    @pytest.mark.asyncio
    async def test_integration_with_keyword_extraction(self):
        """Test integration with keyword extraction."""

        keywords = KeywordExtraction(
            entity_names=["User", "Database"],
            concept_terms=["architecture", "pattern"],
            technical_terms=["API", "REST", "JSON"],
            relationship_indicators=["connects", "implements"],
        )

        query = "how does the User API connect to the Database using REST patterns"

        detector = get_complexity_detector()
        profile = await detector.analyze_complexity(query, keywords=keywords)

        # Should produce reasonable complexity assessment
        assert profile.overall_complexity in [QueryComplexity.MODERATE, QueryComplexity.COMPLEX]
        assert profile.semantic_complexity > 0.3  # Should detect concepts and relationships
        assert profile.domain_complexity > 0.3  # Should detect technical terms

    @pytest.mark.asyncio
    async def test_performance_characteristics(self):
        """Test performance characteristics of complexity analysis."""

        queries = [
            "simple",
            "moderately complex query with several technical concepts",
            "extremely complex architectural analysis involving multiple sophisticated paradigms and methodological frameworks",
        ]

        detector = get_complexity_detector()

        for query in queries:
            start_time = time.time()
            profile = await detector.analyze_complexity(query)
            analysis_time = (time.time() - start_time) * 1000  # Convert to ms

            # Analysis should complete within reasonable time
            assert analysis_time < 100  # Less than 100ms
            assert isinstance(profile, ComplexityProfile)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
