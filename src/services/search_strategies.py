"""
Search strategies for intelligent RAG-based information retrieval.

This module implements the Strategy pattern for different types of search approaches,
providing flexible and extensible search capabilities for various use cases.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from tools.indexing.search_tools import search_sync


class SearchMode(Enum):
    """Enumeration of supported search modes."""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"  
    HYBRID = "hybrid"


@dataclass
class SearchParameters:
    """Parameters for configuring search behavior."""
    n_results: int = 5
    search_mode: SearchMode = SearchMode.HYBRID
    include_context: bool = True
    context_chunks: int = 1
    cross_project: bool = False
    similarity_threshold: float = 0.7
    max_results_per_query: int = 10
    enable_parallel_search: bool = True


@dataclass
class EnrichedSearchResult:
    """Search result enriched with additional metadata and scoring."""
    content: str
    file_path: str
    chunk_id: str
    score: float
    confidence_score: float = 0.0
    keyword_matches: List[str] = None
    metadata: Dict[str, Any] = None
    chunk_type: str = None
    language: str = None
    context_before: str = None
    context_after: str = None
    
    def __post_init__(self):
        if self.keyword_matches is None:
            self.keyword_matches = []
        if self.metadata is None:
            self.metadata = {}


class BaseSearchStrategy(ABC):
    """
    Abstract base class for search strategies.
    
    This class defines the interface that all search strategies must implement,
    providing a consistent way to handle different search approaches while
    allowing for strategy-specific customizations.
    """
    
    def __init__(self, parameters: SearchParameters = None):
        """
        Initialize the search strategy.
        
        Args:
            parameters: Search parameters configuration
        """
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.parameters = parameters or SearchParameters()
        
        # Default scoring weights - can be overridden by subclasses
        self.scoring_weights = {
            "semantic_score": 0.4,
            "keyword_match": 0.3,
            "file_location": 0.2,
            "function_metadata": 0.1
        }
    
    @abstractmethod
    def search(self, query: str, **kwargs) -> List[EnrichedSearchResult]:
        """
        Execute search using this strategy.
        
        Args:
            query: Search query string
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            List of enriched search results
        """
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get the name of this search strategy."""
        pass
    
    def preprocess_query(self, query: str) -> str:
        """
        Preprocess the query before search execution.
        
        Args:
            query: Original query string
            
        Returns:
            Preprocessed query string
        """
        # Default implementation - can be overridden
        return query.strip()
    
    def postprocess_results(self, results: List[Any]) -> List[EnrichedSearchResult]:
        """
        Post-process raw search results into enriched results.
        
        Args:
            results: Raw search results from the search engine
            
        Returns:
            List of enriched search results
        """
        enriched_results = []
        
        for result in results:
            enriched_result = self._enrich_search_result(result)
            if enriched_result:
                enriched_results.append(enriched_result)
        
        return enriched_results
    
    def _enrich_search_result(self, result: Any) -> Optional[EnrichedSearchResult]:
        """
        Enrich a single search result with additional metadata.
        
        Args:
            result: Raw search result
            
        Returns:
            Enriched search result or None if enrichment fails
        """
        try:
            # Extract basic information from result
            content = getattr(result, 'content', '') or str(result)
            file_path = getattr(result, 'file_path', '')
            chunk_id = getattr(result, 'chunk_id', '')
            score = getattr(result, 'score', 0.0)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(result)
            
            # Extract metadata
            metadata = getattr(result, 'metadata', {}) or {}
            chunk_type = metadata.get('chunk_type', 'unknown')
            language = metadata.get('language', 'unknown')
            
            # Extract context if available
            context_before = getattr(result, 'context_before', None)
            context_after = getattr(result, 'context_after', None)
            
            return EnrichedSearchResult(
                content=content,
                file_path=file_path,
                chunk_id=chunk_id,
                score=score,
                confidence_score=confidence_score,
                metadata=metadata,
                chunk_type=chunk_type,
                language=language,
                context_before=context_before,
                context_after=context_after
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to enrich search result: {e}")
            return None
    
    def _calculate_confidence_score(self, result: Any) -> float:
        """
        Calculate confidence score for a search result.
        
        Args:
            result: Search result to score
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        score_components = {}
        
        # Base semantic score
        semantic_score = getattr(result, 'score', 0.0)
        score_components['semantic'] = semantic_score * self.scoring_weights.get('semantic_score', 0.4)
        
        # Keyword matching bonus
        keyword_bonus = self._calculate_keyword_bonus(result)
        score_components['keyword'] = keyword_bonus * self.scoring_weights.get('keyword_match', 0.3)
        
        # File location bonus
        location_bonus = self._calculate_location_bonus(result)
        score_components['location'] = location_bonus * self.scoring_weights.get('file_location', 0.2)
        
        # Metadata bonus
        metadata_bonus = self._calculate_metadata_bonus(result)
        score_components['metadata'] = metadata_bonus * self.scoring_weights.get('function_metadata', 0.1)
        
        # Calculate weighted sum
        total_score = sum(score_components.values())
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, total_score))
    
    def _calculate_keyword_bonus(self, result: Any) -> float:
        """Calculate keyword matching bonus."""
        # Default implementation - can be overridden
        return 0.5
    
    def _calculate_location_bonus(self, result: Any) -> float:
        """Calculate file location relevance bonus."""
        # Default implementation - can be overridden
        file_path = getattr(result, 'file_path', '').lower()
        
        # Boost for important directories
        if any(important in file_path for important in ['src/', 'lib/', 'core/', 'main/']):
            return 0.8
        elif any(test in file_path for test in ['test/', 'tests/', 'spec/']):
            return 0.3
        else:
            return 0.5
    
    def _calculate_metadata_bonus(self, result: Any) -> float:
        """Calculate metadata-based bonus."""
        # Default implementation - can be overridden
        metadata = getattr(result, 'metadata', {}) or {}
        
        # Boost for important chunk types
        chunk_type = metadata.get('chunk_type', '').lower()
        if chunk_type in ['function', 'class', 'method']:
            return 0.8
        elif chunk_type in ['constant', 'variable']:
            return 0.4
        else:
            return 0.5
    
    def validate_parameters(self) -> bool:
        """
        Validate search parameters.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        if self.parameters.n_results <= 0:
            self.logger.error("n_results must be positive")
            return False
        
        if not isinstance(self.parameters.search_mode, SearchMode):
            self.logger.error("search_mode must be a SearchMode enum")
            return False
        
        if self.parameters.similarity_threshold < 0 or self.parameters.similarity_threshold > 1:
            self.logger.error("similarity_threshold must be between 0 and 1")
            return False
        
        return True


class SearchStrategyRegistry:
    """
    Registry for managing and discovering search strategies.
    
    This class provides a centralized way to register and retrieve
    search strategies for different search approaches.
    """
    
    def __init__(self):
        """Initialize the strategy registry."""
        self.logger = logging.getLogger(__name__)
        self._strategies: Dict[str, BaseSearchStrategy] = {}
    
    def register_strategy(self, name: str, strategy: BaseSearchStrategy) -> None:
        """
        Register a search strategy.
        
        Args:
            name: Strategy name/identifier
            strategy: Search strategy instance
        """
        self._strategies[name] = strategy
        self.logger.info(f"Registered search strategy: {name}")
    
    def get_strategy(self, name: str) -> Optional[BaseSearchStrategy]:
        """
        Get a search strategy by name.
        
        Args:
            name: Strategy name/identifier
            
        Returns:
            Search strategy instance or None if not found
        """
        return self._strategies.get(name)
    
    def has_strategy(self, name: str) -> bool:
        """
        Check if a strategy is registered.
        
        Args:
            name: Strategy name/identifier
            
        Returns:
            True if strategy exists, False otherwise
        """
        return name in self._strategies
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return list(self._strategies.keys())
    
    def unregister_strategy(self, name: str) -> bool:
        """
        Unregister a search strategy.
        
        Args:
            name: Strategy name/identifier
            
        Returns:
            True if strategy was removed, False if not found
        """
        if name in self._strategies:
            del self._strategies[name]
            self.logger.info(f"Unregistered search strategy: {name}")
            return True
        return False


# Global registry instance
search_strategy_registry = SearchStrategyRegistry()


def register_search_strategy(name: str):
    """
    Decorator for automatically registering search strategies.
    
    Args:
        name: Strategy name/identifier
        
    Returns:
        Decorator function
    """
    def decorator(strategy_class):
        """Register the strategy class."""
        strategy_instance = strategy_class()
        search_strategy_registry.register_strategy(name, strategy_instance)
        return strategy_class
    
    return decorator


class BaseExecutor:
    """Base class for search execution utilities."""
    
    def __init__(self, parameters: SearchParameters = None):
        """Initialize the executor."""
        self.logger = logging.getLogger(__name__)
        self.parameters = parameters or SearchParameters()
    
    def execute_search(self, query: str, search_mode: SearchMode = None) -> List[Any]:
        """
        Execute a single search query.
        
        Args:
            query: Search query string
            search_mode: Search mode to use (overrides default)
            
        Returns:
            List of raw search results
        """
        mode = search_mode or self.parameters.search_mode
        
        try:
            # Use the existing search_sync function
            results = search_sync(
                query=query,
                n_results=self.parameters.n_results,
                search_mode=mode.value,
                include_context=self.parameters.include_context,
                context_chunks=self.parameters.context_chunks,
                cross_project=self.parameters.cross_project
            )
            
            return results if results else []
            
        except Exception as e:
            self.logger.error(f"Search execution failed for query '{query}': {e}")
            return []
    
    def execute_parallel_searches(self, queries: List[str], search_mode: SearchMode = None) -> Dict[str, List[Any]]:
        """
        Execute multiple searches in parallel.
        
        Args:
            queries: List of search queries
            search_mode: Search mode to use for all queries
            
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        
        if not self.parameters.enable_parallel_search or len(queries) == 1:
            # Fall back to sequential execution
            return self.execute_sequential_searches(queries, search_mode)
        
        try:
            with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
                # Submit all queries
                future_to_query = {
                    executor.submit(self.execute_search, query, search_mode): query
                    for query in queries
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_query):
                    query = future_to_query[future]
                    try:
                        result = future.result()
                        results[query] = result
                    except Exception as e:
                        self.logger.error(f"Parallel search failed for query '{query}': {e}")
                        results[query] = []
        
        except Exception as e:
            self.logger.error(f"Parallel search execution failed: {e}")
            # Fall back to sequential execution
            return self.execute_sequential_searches(queries, search_mode)
        
        return results
    
    def execute_sequential_searches(self, queries: List[str], search_mode: SearchMode = None) -> Dict[str, List[Any]]:
        """
        Execute multiple searches sequentially.
        
        Args:
            queries: List of search queries
            search_mode: Search mode to use for all queries
            
        Returns:
            Dictionary mapping queries to their results
        """
        results = {}
        
        for query in queries:
            results[query] = self.execute_search(query, search_mode)
        
        return results


# =================== Concrete Search Strategy Implementations ===================

@register_search_strategy('semantic')
class SemanticSearchStrategy(BaseSearchStrategy):
    """
    Semantic search strategy using vector similarity.
    
    This strategy focuses on finding semantically similar content using
    embeddings and vector similarity matching.
    """
    
    def __init__(self, parameters: SearchParameters = None):
        """Initialize the semantic search strategy."""
        super().__init__(parameters)
        self.executor = BaseExecutor(self.parameters)
        
        # Adjust scoring weights for semantic search
        self.scoring_weights = {
            "semantic_score": 0.7,  # Higher weight for semantic similarity
            "keyword_match": 0.1,
            "file_location": 0.1,
            "function_metadata": 0.1
        }
    
    def get_strategy_name(self) -> str:
        """Get the name of this search strategy."""
        return "semantic"
    
    def search(self, query: str, **kwargs) -> List[EnrichedSearchResult]:
        """
        Execute semantic search using vector similarity.
        
        Args:
            query: Search query string
            **kwargs: Additional parameters (boost_semantic, semantic_threshold, etc.)
            
        Returns:
            List of enriched search results
        """
        # Preprocess query for better semantic matching
        processed_query = self.preprocess_query(query)
        
        # Add semantic-specific query enhancements
        if kwargs.get('boost_semantic', True):
            processed_query = self._enhance_semantic_query(processed_query)
        
        # Execute search with semantic mode
        raw_results = self.executor.execute_search(processed_query, SearchMode.SEMANTIC)
        
        # Post-process and enrich results
        enriched_results = self.postprocess_results(raw_results)
        
        # Apply semantic-specific filtering
        semantic_threshold = kwargs.get('semantic_threshold', 0.5)
        filtered_results = [
            result for result in enriched_results
            if result.score >= semantic_threshold
        ]
        
        # Sort by semantic score
        filtered_results.sort(key=lambda x: x.score, reverse=True)
        
        return filtered_results[:self.parameters.n_results]
    
    def _enhance_semantic_query(self, query: str) -> str:
        """Enhance query for better semantic matching."""
        # Add conceptual expansions for common programming terms
        semantic_expansions = {
            'function': 'function method procedure routine',
            'class': 'class object type structure',
            'error': 'error exception failure bug issue',
            'config': 'configuration settings options parameters',
            'test': 'test testing spec specification validation',
            'api': 'api interface endpoint service',
            'database': 'database db storage persistence data',
            'auth': 'authentication authorization login security'
        }
        
        enhanced_query = query
        for term, expansion in semantic_expansions.items():
            if term.lower() in query.lower():
                enhanced_query += f" {expansion}"
        
        return enhanced_query
    
    def _calculate_keyword_bonus(self, result: Any) -> float:
        """Calculate keyword matching bonus for semantic search."""
        # Lower weight for keyword matching in semantic search
        return 0.3
    
    def _calculate_metadata_bonus(self, result: Any) -> float:
        """Calculate metadata bonus with semantic considerations."""
        metadata = getattr(result, 'metadata', {}) or {}
        
        # Boost for semantically rich content types
        chunk_type = metadata.get('chunk_type', '').lower()
        if chunk_type in ['function', 'class', 'method']:
            return 0.9
        elif chunk_type in ['docstring', 'comment']:
            return 0.8  # High value for documentation in semantic search
        elif chunk_type in ['interface', 'type_alias']:
            return 0.7
        else:
            return 0.5


@register_search_strategy('keyword')
class KeywordSearchStrategy(BaseSearchStrategy):
    """
    Keyword-based search strategy using exact matching.
    
    This strategy focuses on finding exact keyword matches and
    term frequency-based relevance scoring.
    """
    
    def __init__(self, parameters: SearchParameters = None):
        """Initialize the keyword search strategy."""
        super().__init__(parameters)
        self.executor = BaseExecutor(self.parameters)
        
        # Adjust scoring weights for keyword search
        self.scoring_weights = {
            "semantic_score": 0.2,
            "keyword_match": 0.6,  # Higher weight for keyword matching
            "file_location": 0.1,
            "function_metadata": 0.1
        }
    
    def get_strategy_name(self) -> str:
        """Get the name of this search strategy."""
        return "keyword"
    
    def search(self, query: str, **kwargs) -> List[EnrichedSearchResult]:
        """
        Execute keyword search using exact matching.
        
        Args:
            query: Search query string
            **kwargs: Additional parameters (exact_match, case_sensitive, etc.)
            
        Returns:
            List of enriched search results
        """
        # Preprocess query for keyword matching
        processed_query = self.preprocess_query(query)
        
        # Extract keywords for matching
        keywords = self._extract_keywords(processed_query)
        
        # Execute search with keyword mode
        raw_results = self.executor.execute_search(processed_query, SearchMode.KEYWORD)
        
        # Post-process and enrich results with keyword analysis
        enriched_results = self.postprocess_results(raw_results)
        
        # Apply keyword-specific scoring
        for result in enriched_results:
            result.keyword_matches = self._find_keyword_matches(result.content, keywords)
            # Recalculate confidence with keyword emphasis
            result.confidence_score = self._calculate_keyword_confidence(result, keywords)
        
        # Filter by keyword match quality
        min_keyword_matches = kwargs.get('min_keyword_matches', 1)
        filtered_results = [
            result for result in enriched_results
            if len(result.keyword_matches) >= min_keyword_matches
        ]
        
        # Sort by keyword match quality and confidence
        filtered_results.sort(
            key=lambda x: (len(x.keyword_matches), x.confidence_score), 
            reverse=True
        )
        
        return filtered_results[:self.parameters.n_results]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from the query."""
        # Simple keyword extraction (can be enhanced with NLP)
        import re
        
        # Remove common stop words and extract meaningful terms
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'in', 'with', 'to', 'for', 'of', 'as', 'by'}
        
        # Extract words (including programming identifiers)
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
    def _find_keyword_matches(self, content: str, keywords: List[str]) -> List[str]:
        """Find keyword matches in content."""
        content_lower = content.lower()
        matches = []
        
        for keyword in keywords:
            if keyword in content_lower:
                matches.append(keyword)
        
        return matches
    
    def _calculate_keyword_confidence(self, result: EnrichedSearchResult, keywords: List[str]) -> float:
        """Calculate confidence score with emphasis on keyword matching."""
        base_score = result.score * 0.3  # Lower weight for base semantic score
        
        # Keyword match score
        if keywords:
            match_ratio = len(result.keyword_matches) / len(keywords)
            keyword_score = match_ratio * 0.6
        else:
            keyword_score = 0.0
        
        # Position bonus (keywords at the beginning are more important)
        position_bonus = self._calculate_keyword_position_bonus(result.content, result.keyword_matches) * 0.1
        
        total_score = base_score + keyword_score + position_bonus
        return min(1.0, max(0.0, total_score))
    
    def _calculate_keyword_position_bonus(self, content: str, matches: List[str]) -> float:
        """Calculate bonus based on keyword positions in content."""
        if not matches or not content:
            return 0.0
        
        content_lower = content.lower()
        total_bonus = 0.0
        
        for match in matches:
            position = content_lower.find(match)
            if position != -1:
                # Earlier positions get higher bonus
                position_bonus = max(0, 1.0 - (position / len(content)))
                total_bonus += position_bonus
        
        return total_bonus / len(matches)
    
    def _calculate_keyword_bonus(self, result: Any) -> float:
        """Calculate keyword matching bonus for keyword search."""
        # Higher weight for keyword matching in keyword search
        return 0.8


@register_search_strategy('hybrid')
class HybridSearchStrategy(BaseSearchStrategy):
    """
    Hybrid search strategy combining semantic and keyword approaches.
    
    This strategy balances semantic understanding with exact keyword matching
    to provide comprehensive search results.
    """
    
    def __init__(self, parameters: SearchParameters = None):
        """Initialize the hybrid search strategy."""
        super().__init__(parameters)
        self.executor = BaseExecutor(self.parameters)
        
        # Initialize sub-strategies
        self.semantic_strategy = SemanticSearchStrategy(parameters)
        self.keyword_strategy = KeywordSearchStrategy(parameters)
        
        # Balanced scoring weights
        self.scoring_weights = {
            "semantic_score": 0.4,
            "keyword_match": 0.4,
            "file_location": 0.1,
            "function_metadata": 0.1
        }
    
    def get_strategy_name(self) -> str:
        """Get the name of this search strategy."""
        return "hybrid"
    
    def search(self, query: str, **kwargs) -> List[EnrichedSearchResult]:
        """
        Execute hybrid search combining semantic and keyword approaches.
        
        Args:
            query: Search query string
            **kwargs: Additional parameters (semantic_weight, keyword_weight, etc.)
            
        Returns:
            List of enriched search results
        """
        # Configure strategy weights
        semantic_weight = kwargs.get('semantic_weight', 0.6)
        keyword_weight = kwargs.get('keyword_weight', 0.4)
        
        # Execute base hybrid search
        raw_results = self.executor.execute_search(query, SearchMode.HYBRID)
        enriched_results = self.postprocess_results(raw_results)
        
        # Get additional results from sub-strategies if enabled
        combine_strategies = kwargs.get('combine_strategies', True)
        
        if combine_strategies:
            # Get semantic results
            semantic_results = self.semantic_strategy.search(
                query, 
                semantic_threshold=kwargs.get('semantic_threshold', 0.4)
            )
            
            # Get keyword results  
            keyword_results = self.keyword_strategy.search(
                query,
                min_keyword_matches=kwargs.get('min_keyword_matches', 1)
            )
            
            # Combine and deduplicate results
            all_results = self._combine_results(
                enriched_results, semantic_results, keyword_results,
                semantic_weight, keyword_weight
            )
        else:
            all_results = enriched_results
        
        # Apply hybrid-specific scoring
        for result in all_results:
            result.confidence_score = self._calculate_hybrid_confidence(
                result, semantic_weight, keyword_weight
            )
        
        # Sort by hybrid confidence score
        all_results.sort(key=lambda x: x.confidence_score, reverse=True)
        
        # Remove duplicates while preserving order
        unique_results = self._deduplicate_results(all_results)
        
        return unique_results[:self.parameters.n_results]
    
    def _combine_results(self, base_results: List[EnrichedSearchResult],
                        semantic_results: List[EnrichedSearchResult],
                        keyword_results: List[EnrichedSearchResult],
                        semantic_weight: float, keyword_weight: float) -> List[EnrichedSearchResult]:
        """Combine results from different strategies."""
        # Create a result map to handle duplicates
        result_map = {}
        
        # Add base results
        for result in base_results:
            key = (result.file_path, result.chunk_id)
            result_map[key] = result
        
        # Add semantic results with weight adjustment
        for result in semantic_results:
            key = (result.file_path, result.chunk_id)
            if key in result_map:
                # Boost existing result
                existing = result_map[key]
                existing.confidence_score = max(
                    existing.confidence_score,
                    result.confidence_score * semantic_weight
                )
            else:
                # Add new result with semantic weight
                result.confidence_score *= semantic_weight
                result_map[key] = result
        
        # Add keyword results with weight adjustment
        for result in keyword_results:
            key = (result.file_path, result.chunk_id)
            if key in result_map:
                # Boost existing result
                existing = result_map[key]
                existing.confidence_score = max(
                    existing.confidence_score,
                    result.confidence_score * keyword_weight
                )
            else:
                # Add new result with keyword weight
                result.confidence_score *= keyword_weight
                result_map[key] = result
        
        return list(result_map.values())
    
    def _calculate_hybrid_confidence(self, result: EnrichedSearchResult,
                                   semantic_weight: float, keyword_weight: float) -> float:
        """Calculate hybrid confidence score."""
        # Base semantic component
        semantic_component = result.score * semantic_weight
        
        # Keyword component
        keyword_component = len(result.keyword_matches) * 0.1 * keyword_weight
        
        # Metadata and location components
        metadata_bonus = self._calculate_metadata_bonus(result) * 0.1
        location_bonus = self._calculate_location_bonus(result) * 0.1
        
        total_score = semantic_component + keyword_component + metadata_bonus + location_bonus
        
        return min(1.0, max(0.0, total_score))
    
    def _deduplicate_results(self, results: List[EnrichedSearchResult]) -> List[EnrichedSearchResult]:
        """Remove duplicate results while preserving order."""
        seen = set()
        unique_results = []
        
        for result in results:
            key = (result.file_path, result.chunk_id)
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results


@register_search_strategy('comprehensive')
class ComprehensiveSearchStrategy(BaseSearchStrategy):
    """
    Comprehensive search strategy for deep project exploration.
    
    This strategy executes multiple focused searches across different
    aspects of a project to provide comprehensive analysis.
    """
    
    def __init__(self, parameters: SearchParameters = None):
        """Initialize the comprehensive search strategy."""
        super().__init__(parameters)
        self.executor = BaseExecutor(self.parameters)
        
        # Query templates for different exploration areas
        self.query_templates = {
            'architecture': [
                'main application entry point',
                'service layer architecture',
                'data access layer',
                'configuration management'
            ],
            'patterns': [
                'design patterns implementation',
                'dependency injection',
                'factory pattern',
                'observer pattern'
            ],
            'features': [
                'core business logic',
                'user interface components',
                'API endpoints',
                'data processing'
            ],
            'infrastructure': [
                'error handling',
                'logging configuration',
                'testing framework',
                'deployment scripts'
            ]
        }
    
    def get_strategy_name(self) -> str:
        """Get the name of this search strategy."""
        return "comprehensive"
    
    def search(self, query: str, **kwargs) -> List[EnrichedSearchResult]:
        """
        Execute comprehensive search across multiple dimensions.
        
        Args:
            query: Base search query
            **kwargs: Additional parameters (focus_areas, parallel_execution, etc.)
            
        Returns:
            List of enriched search results from comprehensive analysis
        """
        focus_areas = kwargs.get('focus_areas', ['architecture', 'patterns', 'features'])
        
        # Build comprehensive query set
        queries = self._build_comprehensive_queries(query, focus_areas)
        
        # Execute searches
        if self.parameters.enable_parallel_search:
            search_results = self.executor.execute_parallel_searches(queries, SearchMode.HYBRID)
        else:
            search_results = self.executor.execute_sequential_searches(queries, SearchMode.HYBRID)
        
        # Collect and process all results
        all_results = []
        for query_text, results in search_results.items():
            enriched = self.postprocess_results(results)
            for result in enriched:
                # Add query context to metadata
                result.metadata['source_query'] = query_text
                result.metadata['search_area'] = self._get_query_area(query_text, focus_areas)
            all_results.extend(enriched)
        
        # Calculate comprehensive confidence scores
        for result in all_results:
            result.confidence_score = self._calculate_comprehensive_confidence(result)
        
        # Sort and deduplicate
        all_results.sort(key=lambda x: x.confidence_score, reverse=True)
        unique_results = self._deduplicate_results(all_results)
        
        return unique_results[:self.parameters.n_results]
    
    def _build_comprehensive_queries(self, base_query: str, focus_areas: List[str]) -> List[str]:
        """Build comprehensive query set."""
        queries = [base_query]  # Always include the base query
        
        for area in focus_areas:
            if area in self.query_templates:
                for template in self.query_templates[area]:
                    # Combine base query with template
                    combined_query = f"{base_query} {template}"
                    queries.append(combined_query)
        
        return queries
    
    def _get_query_area(self, query_text: str, focus_areas: List[str]) -> str:
        """Determine which focus area a query belongs to."""
        for area in focus_areas:
            if area in self.query_templates:
                for template in self.query_templates[area]:
                    if template in query_text:
                        return area
        return "general"
    
    def _calculate_comprehensive_confidence(self, result: EnrichedSearchResult) -> float:
        """Calculate confidence for comprehensive search results."""
        base_confidence = self._calculate_confidence_score(result)
        
        # Boost for results from multiple query areas
        search_area = result.metadata.get('search_area', 'general')
        area_bonus = 0.1 if search_area != 'general' else 0.0
        
        return min(1.0, base_confidence + area_bonus)
    
    def _deduplicate_results(self, results: List[EnrichedSearchResult]) -> List[EnrichedSearchResult]:
        """Remove duplicate results while preserving order."""
        seen = set()
        unique_results = []
        
        for result in results:
            key = (result.file_path, result.chunk_id)
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        return unique_results