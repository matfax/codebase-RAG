"""
RAG Search Strategy Service - Refactored Coordinator

This service orchestrates intelligent multi-query RAG search strategies for project exploration
by coordinating specialized services and search strategies.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from services.query_builder_service import (
    FilterOperator,
    QueryBuilderService,
    QueryFilter,
    QueryParameters,
    SearchContext,
)
from services.result_processing_service import (
    ProcessedResult,
    ProcessingOptions,
    ProcessingStats,
    ResultFormat,
    ResultProcessingService,
    SortOrder,
)
from services.search_strategies import (
    BaseSearchStrategy,
    SearchMode,
    SearchParameters,
    search_strategy_registry,
)
from src.tools.indexing.search_tools import (
    get_current_project,
    search_sync,
)

logger = logging.getLogger(__name__)


class SearchQueryType(Enum):
    """Types of search queries for project exploration."""

    ARCHITECTURE_PATTERNS = "architecture_patterns"
    ENTRY_POINTS = "entry_points"
    CORE_COMPONENTS = "core_components"
    DATA_FLOW = "data_flow"
    CONFIGURATION = "configuration"
    DEPENDENCIES = "dependencies"
    TESTING_PATTERNS = "testing_patterns"
    API_ENDPOINTS = "api_endpoints"


@dataclass
class SearchQuery:
    """Represents a structured search query for project exploration."""

    query_type: SearchQueryType
    query_text: str
    priority: int = 1  # 1=high, 2=medium, 3=low
    expected_results: int = 5
    search_mode: str = "hybrid"
    context_chunks: int = 1


@dataclass
class SearchResult:
    """Enriched search result with metadata and confidence scoring."""

    original_result: dict[str, Any]
    query_type: SearchQueryType
    confidence_score: float
    relevance_indicators: list[str] = field(default_factory=list)
    breadcrumb_context: str = ""
    function_signature: str = ""
    component_type: str = ""


@dataclass
class RAGSearchResults:
    """Comprehensive results from multi-query RAG search."""

    architecture_insights: list[SearchResult] = field(default_factory=list)
    entry_points: list[SearchResult] = field(default_factory=list)
    core_components: list[SearchResult] = field(default_factory=list)
    data_flow_patterns: list[SearchResult] = field(default_factory=list)
    configuration_elements: list[SearchResult] = field(default_factory=list)
    dependencies: list[SearchResult] = field(default_factory=list)
    testing_patterns: list[SearchResult] = field(default_factory=list)
    api_endpoints: list[SearchResult] = field(default_factory=list)

    # Metadata
    total_search_time: float = 0.0
    queries_executed: int = 0
    total_results: int = 0
    project_context: str = ""
    search_strategy: str = ""


class RAGSearchStrategyCoordinator:
    """
    Refactored RAG search strategy coordinator.

    This service coordinates specialized search strategies, query builders,
    and result processors to provide comprehensive project exploration capabilities.
    """

    def __init__(self):
        """Initialize the search strategy coordinator."""
        self.logger = logger

        # Initialize specialized services
        self.query_builder = QueryBuilderService()
        self.result_processor = ResultProcessingService()

        # Search query templates for different exploration aspects
        self.query_templates = self._initialize_query_templates()

        # Default search configuration
        self.default_search_params = SearchParameters(
            n_results=5,
            search_mode=SearchMode.HYBRID,
            include_context=True,
            context_chunks=1,
            cross_project=False,
            enable_parallel_search=True,
        )

    def execute_comprehensive_search(
        self,
        project_path: str | None = None,
        focus_areas: list[SearchQueryType] | None = None,
        max_results_per_query: int = 5,
        enable_parallel_search: bool = True,
        search_strategy_name: str = "hybrid",
    ) -> RAGSearchResults:
        """
        Execute comprehensive multi-query search for project exploration.

        Args:
            project_path: Path to project (None for current project)
            focus_areas: Specific areas to focus on (None for all areas)
            max_results_per_query: Maximum results per individual query
            enable_parallel_search: Whether to run searches in parallel
            search_strategy_name: Name of search strategy to use

        Returns:
            RAGSearchResults with comprehensive project insights
        """
        start_time = time.time()

        try:
            # Get project context
            current_project = get_current_project(project_path)
            project_name = current_project.get("name", "unknown") if current_project else "unknown"

            # Determine search queries based on focus areas
            search_queries = self._build_search_queries(focus_areas, max_results_per_query)

            self.logger.info(f"Starting comprehensive RAG search for project: {project_name}")
            self.logger.info(f"Executing {len(search_queries)} targeted queries using strategy: {search_strategy_name}")

            # Get search strategy
            search_strategy = search_strategy_registry.get_strategy(search_strategy_name)
            if not search_strategy:
                self.logger.warning(f"Strategy '{search_strategy_name}' not found, using default hybrid")
                search_strategy = search_strategy_registry.get_strategy("hybrid")

            # Execute searches
            if enable_parallel_search and len(search_queries) > 1:
                search_results = self._execute_parallel_searches(search_queries, search_strategy)
            else:
                search_results = self._execute_sequential_searches(search_queries, search_strategy)

            # Process and organize results
            organized_results = self._organize_search_results(search_results)

            # Calculate metadata
            organized_results.total_search_time = time.time() - start_time
            organized_results.queries_executed = len(search_queries)
            organized_results.total_results = self._count_total_results(organized_results)
            organized_results.project_context = project_name
            organized_results.search_strategy = search_strategy_name

            self.logger.info(f"RAG search completed in {organized_results.total_search_time:.2f}s")
            self.logger.info(f"Found {organized_results.total_results} relevant results across {len(search_queries)} queries")

            return organized_results

        except Exception as e:
            self.logger.error(f"Comprehensive RAG search failed: {e}")
            return RAGSearchResults(
                total_search_time=time.time() - start_time,
                queries_executed=0,
                total_results=0,
                project_context=project_name if "project_name" in locals() else "unknown",
                search_strategy="failed",
            )

    def execute_focused_search(
        self,
        search_query: SearchQuery,
        project_path: str | None = None,
        search_strategy_name: str = "hybrid",
    ) -> list[SearchResult]:
        """
        Execute a single focused search query with enhanced analysis.

        Args:
            search_query: Structured search query to execute
            project_path: Path to project (None for current project)
            search_strategy_name: Name of search strategy to use

        Returns:
            List of enriched search results
        """
        try:
            self.logger.debug(f"Executing focused search: {search_query.query_type.value}")

            # Get search strategy
            search_strategy = search_strategy_registry.get_strategy(search_strategy_name)
            if not search_strategy:
                search_strategy = search_strategy_registry.get_strategy("hybrid")

            # Execute the search using the strategy
            enriched_results = search_strategy.search(
                query=search_query.query_text,
                n_results=search_query.expected_results,
                search_mode=SearchMode(search_query.search_mode),
                context_chunks=search_query.context_chunks,
            )

            # Convert enriched results to SearchResult format
            search_results = []
            for result in enriched_results:
                search_result = self._convert_to_search_result(result, search_query.query_type)
                search_results.append(search_result)

            # Sort by confidence score
            search_results.sort(key=lambda x: x.confidence_score, reverse=True)

            return search_results

        except Exception as e:
            self.logger.error(f"Focused search failed for {search_query.query_type.value}: {e}")
            return []

    def detect_architecture_patterns(
        self,
        project_path: str | None = None,
        max_results: int = 10,
        search_strategy_name: str = "semantic",
    ) -> dict[str, Any]:
        """
        Detect architecture patterns using RAG semantic search.

        Args:
            project_path: Path to project (None for current project)
            max_results: Maximum results per pattern search
            search_strategy_name: Search strategy to use

        Returns:
            Dictionary containing detected patterns with confidence scores
        """
        try:
            self.logger.info("Starting architecture pattern detection using RAG search")

            # Define architecture pattern search queries
            pattern_queries = {
                "mvc": [
                    "model view controller MVC pattern",
                    "models directory view templates controller handlers",
                ],
                "microservices": [
                    "microservices service architecture distributed",
                    "API gateway service discovery load balancer",
                ],
                "layered": [
                    "layered architecture layers presentation business data",
                    "repository pattern data access layer service layer",
                ],
                "component_based": [
                    "component based architecture modular components",
                    "component composition reusable modules",
                ],
                "event_driven": [
                    "event driven architecture events publisher subscriber",
                    "message queue event bus asynchronous processing",
                ],
            }

            # Execute pattern detection searches
            pattern_results = {}

            for pattern_name, queries in pattern_queries.items():
                pattern_evidence = []

                for query_text in queries:
                    # Execute focused search for this pattern
                    search_query = SearchQuery(
                        query_type=SearchQueryType.ARCHITECTURE_PATTERNS,
                        query_text=query_text,
                        expected_results=max_results,
                        search_mode="semantic",
                        context_chunks=0,
                    )

                    search_results = self.execute_focused_search(search_query, project_path, search_strategy_name)

                    # Analyze results for pattern indicators
                    for result in search_results:
                        if result.confidence_score > 0.3:  # Minimum threshold
                            pattern_evidence.append(
                                {
                                    "file_path": result.original_result.get("file_path", ""),
                                    "content_snippet": result.original_result.get("content", "")[:200],
                                    "confidence": result.confidence_score,
                                    "line_range": f"{result.original_result.get('line_start', 0)}-{result.original_result.get('line_end', 0)}",
                                    "component_type": result.component_type,
                                    "search_score": result.original_result.get("score", 0.0),
                                }
                            )

                # Calculate overall pattern confidence
                if pattern_evidence:
                    pattern_evidence.sort(key=lambda x: x["confidence"], reverse=True)
                    pattern_evidence = pattern_evidence[:5]  # Top 5 evidence pieces

                    aggregate_confidence = self._calculate_pattern_confidence(pattern_evidence)

                    pattern_results[pattern_name] = {
                        "pattern_name": pattern_name.replace("_", " ").title(),
                        "confidence": aggregate_confidence,
                        "evidence_count": len(pattern_evidence),
                        "evidence": pattern_evidence,
                        "detected": aggregate_confidence > 0.5,
                    }

            # Identify the most likely architecture pattern
            detected_patterns = {name: info for name, info in pattern_results.items() if info["detected"]}

            if detected_patterns:
                primary_pattern = max(detected_patterns.items(), key=lambda x: x[1]["confidence"])
            else:
                primary_pattern = None

            result = {
                "primary_pattern": primary_pattern[1] if primary_pattern else None,
                "all_patterns": pattern_results,
                "detection_summary": {
                    "total_patterns_analyzed": len(pattern_queries),
                    "patterns_detected": len(detected_patterns),
                    "highest_confidence": max([info["confidence"] for info in pattern_results.values()], default=0.0),
                    "analysis_method": "RAG semantic search with pattern matching",
                },
            }

            self.logger.info("Architecture pattern detection completed")
            self.logger.info(f"Detected {len(detected_patterns)} patterns with confidence > 0.5")

            return result

        except Exception as e:
            self.logger.error(f"Architecture pattern detection failed: {e}")
            return {
                "error": str(e),
                "primary_pattern": None,
                "all_patterns": {},
                "detection_summary": {
                    "total_patterns_analyzed": 0,
                    "patterns_detected": 0,
                    "highest_confidence": 0.0,
                    "analysis_method": "failed",
                },
            }

    def discover_entry_points(
        self,
        project_path: str | None = None,
        max_results: int = 10,
        search_strategy_name: str = "hybrid",
    ) -> dict[str, Any]:
        """
        Discover project entry points using function-level RAG search.

        Args:
            project_path: Path to project (None for current project)
            max_results: Maximum results per entry point search
            search_strategy_name: Search strategy to use

        Returns:
            Dictionary containing discovered entry points with metadata
        """
        try:
            self.logger.info("Starting entry point discovery using function-level RAG search")

            # Define entry point search queries
            entry_point_queries = {
                "main_functions": [
                    "main function entry point startup",
                    "if __name__ == '__main__' main function",
                ],
                "cli_entry_points": [
                    "command line interface CLI argparse click",
                    "parser.add_argument command line arguments",
                ],
                "web_server_startup": [
                    "app.run() flask server startup",
                    "uvicorn.run() fastapi server start",
                ],
                "application_factories": [
                    "create_app() application factory function",
                    "app_factory application builder setup",
                ],
            }

            # Execute entry point discovery searches
            entry_point_results = {}

            for category, queries in entry_point_queries.items():
                category_results = []

                for query_text in queries:
                    search_query = SearchQuery(
                        query_type=SearchQueryType.ENTRY_POINTS,
                        query_text=query_text,
                        expected_results=max_results,
                        search_mode="hybrid",
                        context_chunks=1,
                    )

                    search_results = self.execute_focused_search(search_query, project_path, search_strategy_name)

                    for result in search_results:
                        if result.confidence_score > 0.4:  # Threshold for entry point candidacy
                            category_results.append(
                                {
                                    "file_path": result.original_result.get("file_path", ""),
                                    "function_name": result.original_result.get("name", ""),
                                    "function_signature": result.original_result.get("signature", ""),
                                    "content_snippet": result.original_result.get("content", "")[:300],
                                    "confidence": result.confidence_score,
                                    "line_range": f"{result.original_result.get('line_start', 0)}-{result.original_result.get('line_end', 0)}",
                                    "breadcrumb": result.breadcrumb_context,
                                    "docstring": result.original_result.get("docstring", ""),
                                    "language": result.original_result.get("language", ""),
                                    "search_score": result.original_result.get("score", 0.0),
                                    "entry_type": category.replace("_", " ").title(),
                                }
                            )

                # Sort and deduplicate
                if category_results:
                    category_results = self._deduplicate_entry_points(category_results)
                    category_results.sort(key=lambda x: x["confidence"], reverse=True)
                    category_results = category_results[:5]  # Top 5 per category
                    entry_point_results[category] = category_results

            # Identify primary entry points
            all_entry_points = []
            for category, results in entry_point_results.items():
                all_entry_points.extend(results)

            all_entry_points.sort(key=lambda x: x["confidence"], reverse=True)

            # Get primary entry points (highest confidence, different files)
            primary_entry_points = []
            seen_files = set()

            for entry_point in all_entry_points:
                file_path = entry_point["file_path"]
                if file_path not in seen_files and len(primary_entry_points) < 3:
                    primary_entry_points.append(entry_point)
                    seen_files.add(file_path)

            result = {
                "primary_entry_points": primary_entry_points,
                "entry_points_by_category": entry_point_results,
                "all_entry_points": all_entry_points[:10],  # Top 10 overall
                "discovery_summary": {
                    "total_categories_searched": len(entry_point_queries),
                    "categories_with_results": len(entry_point_results),
                    "total_entry_points_found": len(all_entry_points),
                    "primary_entry_points_count": len(primary_entry_points),
                    "highest_confidence": max([ep["confidence"] for ep in all_entry_points], default=0.0),
                    "analysis_method": "Function-level RAG search with signature analysis",
                },
            }

            self.logger.info("Entry point discovery completed")
            self.logger.info(f"Found {len(all_entry_points)} entry points across {len(entry_point_results)} categories")

            return result

        except Exception as e:
            self.logger.error(f"Entry point discovery failed: {e}")
            return {
                "error": str(e),
                "primary_entry_points": [],
                "entry_points_by_category": {},
                "all_entry_points": [],
                "discovery_summary": {
                    "total_categories_searched": 0,
                    "categories_with_results": 0,
                    "total_entry_points_found": 0,
                    "primary_entry_points_count": 0,
                    "highest_confidence": 0.0,
                    "analysis_method": "failed",
                },
            }

    def get_search_strategy_info(self) -> dict[str, Any]:
        """Get information about available search strategies and configurations."""
        available_strategies = search_strategy_registry.get_available_strategies()

        return {
            "available_query_types": [query_type.value for query_type in SearchQueryType],
            "available_search_strategies": available_strategies,
            "search_modes": ["semantic", "hybrid", "keyword"],
            "priority_levels": {"1": "high", "2": "medium", "3": "low"},
            "query_templates_count": {query_type.value: len(templates) for query_type, templates in self.query_templates.items()},
            "max_parallel_searches": 4,
            "default_results_per_query": 5,
            "supported_features": [
                "Multi-query parallel search",
                "Pluggable search strategies",
                "Advanced query building",
                "Result processing and formatting",
                "Confidence scoring",
                "Metadata enrichment",
                "Breadcrumb context",
                "Component type detection",
                "Relevance indicators",
            ],
        }

    def _initialize_query_templates(self) -> dict[SearchQueryType, list[str]]:
        """Initialize search query templates for different exploration aspects."""
        return {
            SearchQueryType.ARCHITECTURE_PATTERNS: [
                "MVC model view controller pattern",
                "microservices architecture service",
                "layered architecture layers components",
                "design patterns factory singleton strategy",
                "application structure main components",
            ],
            SearchQueryType.ENTRY_POINTS: [
                "main function entry point startup",
                "app.py main.py server.js index.js",
                "application startup initialization bootstrap",
                "command line interface CLI entry",
                "web server setup route handler",
            ],
            SearchQueryType.CORE_COMPONENTS: [
                "core modules main components",
                "business logic service layer",
                "data access repository model",
                "utility helper functions common",
                "shared modules libraries components",
            ],
            SearchQueryType.DATA_FLOW: [
                "data flow processing pipeline",
                "request response handler",
                "database query operations CRUD",
                "API calls HTTP requests",
                "event handling message passing",
            ],
            SearchQueryType.CONFIGURATION: [
                "configuration settings config",
                "environment variables env",
                "database connection setup",
                "logging configuration setup",
                "application settings parameters",
            ],
            SearchQueryType.DEPENDENCIES: [
                "import dependencies modules",
                "external libraries packages",
                "third party integrations",
                "requirements dependencies setup",
                "package imports modules",
            ],
            SearchQueryType.TESTING_PATTERNS: [
                "test cases unit testing",
                "integration tests test suite",
                "mock fixtures test data",
                "testing patterns test framework",
                "test coverage validation",
            ],
            SearchQueryType.API_ENDPOINTS: [
                "API endpoints routes handlers",
                "REST API HTTP methods",
                "URL routing path handlers",
                "web service endpoints",
                "GraphQL resolvers mutations",
            ],
        }

    def _build_search_queries(
        self,
        focus_areas: list[SearchQueryType] | None,
        max_results: int,
    ) -> list[SearchQuery]:
        """Build search queries based on focus areas."""
        if focus_areas is None:
            # Use all search types with priorities
            focus_areas = [
                SearchQueryType.ENTRY_POINTS,  # High priority
                SearchQueryType.ARCHITECTURE_PATTERNS,  # High priority
                SearchQueryType.CORE_COMPONENTS,  # High priority
                SearchQueryType.DATA_FLOW,  # Medium priority
                SearchQueryType.CONFIGURATION,  # Medium priority
                SearchQueryType.DEPENDENCIES,  # Low priority
                SearchQueryType.TESTING_PATTERNS,  # Low priority
                SearchQueryType.API_ENDPOINTS,  # Medium priority
            ]

        queries = []

        for query_type in focus_areas:
            templates = self.query_templates.get(query_type, [])
            if not templates:
                continue

            # Use the first (most general) template for each type
            primary_query = templates[0]

            # Set priority based on query type
            priority = self._get_query_priority(query_type)

            query = SearchQuery(
                query_type=query_type,
                query_text=primary_query,
                priority=priority,
                expected_results=max_results,
                search_mode="hybrid",
                context_chunks=1,
            )
            queries.append(query)

        # Sort by priority (high priority first)
        queries.sort(key=lambda x: x.priority)

        return queries

    def _get_query_priority(self, query_type: SearchQueryType) -> int:
        """Get priority level for query type."""
        high_priority = {
            SearchQueryType.ENTRY_POINTS,
            SearchQueryType.ARCHITECTURE_PATTERNS,
            SearchQueryType.CORE_COMPONENTS,
        }

        medium_priority = {
            SearchQueryType.DATA_FLOW,
            SearchQueryType.API_ENDPOINTS,
            SearchQueryType.CONFIGURATION,
        }

        if query_type in high_priority:
            return 1
        elif query_type in medium_priority:
            return 2
        else:
            return 3

    def _execute_parallel_searches(
        self,
        search_queries: list[SearchQuery],
        search_strategy: BaseSearchStrategy,
    ) -> dict[SearchQueryType, list[SearchResult]]:
        """Execute multiple searches in parallel."""
        results = {}

        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all searches
            future_to_query = {executor.submit(self.execute_focused_search, query): query for query in search_queries}

            # Collect results as they complete
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    query_results = future.result()
                    results[query.query_type] = query_results
                except Exception as e:
                    self.logger.error(f"Parallel search failed for {query.query_type.value}: {e}")
                    results[query.query_type] = []

        return results

    def _execute_sequential_searches(
        self,
        search_queries: list[SearchQuery],
        search_strategy: BaseSearchStrategy,
    ) -> dict[SearchQueryType, list[SearchResult]]:
        """Execute searches sequentially."""
        results = {}

        for query in search_queries:
            query_results = self.execute_focused_search(query)
            results[query.query_type] = query_results

        return results

    def _convert_to_search_result(
        self,
        enriched_result: Any,
        query_type: SearchQueryType,
    ) -> SearchResult:
        """Convert enriched search result to SearchResult format."""
        # Extract original result data
        original_result = {
            "content": enriched_result.content,
            "file_path": enriched_result.file_path,
            "chunk_id": enriched_result.chunk_id,
            "name": getattr(enriched_result, "chunk_name", ""),
            "score": enriched_result.score,
            "line_start": getattr(enriched_result, "line_start", 0),
            "line_end": getattr(enriched_result, "line_end", 0),
            "signature": getattr(enriched_result, "signature", ""),
            "docstring": getattr(enriched_result, "docstring", ""),
            "language": getattr(enriched_result, "language", ""),
            "chunk_type": getattr(enriched_result, "chunk_type", ""),
        }

        return SearchResult(
            original_result=original_result,
            query_type=query_type,
            confidence_score=enriched_result.confidence_score,
            relevance_indicators=getattr(enriched_result, "relevance_indicators", []),
            breadcrumb_context=getattr(enriched_result, "breadcrumb", ""),
            function_signature=original_result["signature"],
            component_type=getattr(enriched_result, "component_type", ""),
        )

    def _organize_search_results(
        self,
        search_results: dict[SearchQueryType, list[SearchResult]],
    ) -> RAGSearchResults:
        """Organize search results into structured format."""
        return RAGSearchResults(
            architecture_insights=search_results.get(SearchQueryType.ARCHITECTURE_PATTERNS, []),
            entry_points=search_results.get(SearchQueryType.ENTRY_POINTS, []),
            core_components=search_results.get(SearchQueryType.CORE_COMPONENTS, []),
            data_flow_patterns=search_results.get(SearchQueryType.DATA_FLOW, []),
            configuration_elements=search_results.get(SearchQueryType.CONFIGURATION, []),
            dependencies=search_results.get(SearchQueryType.DEPENDENCIES, []),
            testing_patterns=search_results.get(SearchQueryType.TESTING_PATTERNS, []),
            api_endpoints=search_results.get(SearchQueryType.API_ENDPOINTS, []),
        )

    def _count_total_results(self, results: RAGSearchResults) -> int:
        """Count total results across all categories."""
        return (
            len(results.architecture_insights)
            + len(results.entry_points)
            + len(results.core_components)
            + len(results.data_flow_patterns)
            + len(results.configuration_elements)
            + len(results.dependencies)
            + len(results.testing_patterns)
            + len(results.api_endpoints)
        )

    def _calculate_pattern_confidence(self, evidence_list: list[dict[str, Any]]) -> float:
        """Calculate aggregate confidence for a pattern based on evidence."""
        if not evidence_list:
            return 0.0

        # Calculate weighted average with diminishing returns
        total_weight = 0.0
        total_score = 0.0

        for i, evidence in enumerate(evidence_list):
            # Diminishing weight for subsequent evidence
            weight = 1.0 / (i + 1)
            confidence = evidence["confidence"]

            total_weight += weight
            total_score += confidence * weight

        base_confidence = total_score / total_weight if total_weight > 0 else 0.0

        # Bonus for multiple evidence pieces
        evidence_bonus = min(0.2, len(evidence_list) * 0.05)

        return min(1.0, base_confidence + evidence_bonus)

    def _deduplicate_entry_points(self, entry_points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Remove duplicate entry points based on file path and function name."""
        seen = {}

        for entry_point in entry_points:
            key = (entry_point["file_path"], entry_point["function_name"])

            if key not in seen or entry_point["confidence"] > seen[key]["confidence"]:
                seen[key] = entry_point

        return list(seen.values())
