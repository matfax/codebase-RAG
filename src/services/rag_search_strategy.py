"""
RAG Search Strategy Service

This service implements intelligent multi-query RAG search strategies for project exploration,
enabling comprehensive project analysis by leveraging the existing vectorized knowledge base.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from tools.indexing.search_tools import get_qdrant_client, get_embeddings_manager_instance, get_current_project
from tools.indexing.search_tools import search_sync


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
    original_result: Dict[str, Any]
    query_type: SearchQueryType
    confidence_score: float
    relevance_indicators: List[str] = field(default_factory=list)
    breadcrumb_context: str = ""
    function_signature: str = ""
    component_type: str = ""
    

@dataclass
class RAGSearchResults:
    """Comprehensive results from multi-query RAG search."""
    architecture_insights: List[SearchResult] = field(default_factory=list)
    entry_points: List[SearchResult] = field(default_factory=list)
    core_components: List[SearchResult] = field(default_factory=list)
    data_flow_patterns: List[SearchResult] = field(default_factory=list)
    configuration_elements: List[SearchResult] = field(default_factory=list)
    dependencies: List[SearchResult] = field(default_factory=list)
    testing_patterns: List[SearchResult] = field(default_factory=list)
    api_endpoints: List[SearchResult] = field(default_factory=list)
    
    # Metadata
    total_search_time: float = 0.0
    queries_executed: int = 0
    total_results: int = 0
    project_context: str = ""
    search_strategy: str = ""


class RAGSearchStrategy:
    """
    Advanced RAG search strategy for comprehensive project exploration.
    
    This service performs multiple targeted searches to gather insights about:
    - Project architecture and patterns
    - Entry points and main components
    - Data flow and relationships
    - Configuration and dependencies
    - Testing patterns and API endpoints
    """
    
    def __init__(self):
        self.logger = logger
        
        # Search query templates for different exploration aspects
        self.query_templates = self._initialize_query_templates()
        
        # Confidence scoring weights
        self.scoring_weights = {
            "semantic_score": 0.4,
            "keyword_match": 0.3,
            "file_location": 0.2,
            "function_metadata": 0.1
        }
    
    def execute_comprehensive_search(
        self,
        project_path: Optional[str] = None,
        focus_areas: Optional[List[SearchQueryType]] = None,
        max_results_per_query: int = 5,
        enable_parallel_search: bool = True
    ) -> RAGSearchResults:
        """
        Execute comprehensive multi-query search for project exploration.
        
        Args:
            project_path: Path to project (None for current project)
            focus_areas: Specific areas to focus on (None for all areas)
            max_results_per_query: Maximum results per individual query
            enable_parallel_search: Whether to run searches in parallel
            
        Returns:
            RAGSearchResults with comprehensive project insights
        """
        start_time = time.time()
        
        try:
            # Determine search queries based on focus areas
            search_queries = self._build_search_queries(focus_areas, max_results_per_query)
            
            # Get project context
            current_project = get_current_project(project_path)
            project_name = current_project.get("name", "unknown") if current_project else "unknown"
            
            self.logger.info(f"Starting comprehensive RAG search for project: {project_name}")
            self.logger.info(f"Executing {len(search_queries)} targeted queries")
            
            # Execute searches
            if enable_parallel_search and len(search_queries) > 1:
                search_results = self._execute_parallel_searches(search_queries)
            else:
                search_results = self._execute_sequential_searches(search_queries)
            
            # Process and organize results
            organized_results = self._organize_search_results(search_results)
            
            # Calculate metadata
            organized_results.total_search_time = time.time() - start_time
            organized_results.queries_executed = len(search_queries)
            organized_results.total_results = self._count_total_results(organized_results)
            organized_results.project_context = project_name
            organized_results.search_strategy = "comprehensive_multi_query"
            
            self.logger.info(f"RAG search completed in {organized_results.total_search_time:.2f}s")
            self.logger.info(f"Found {organized_results.total_results} relevant results across {len(search_queries)} queries")
            
            return organized_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive RAG search failed: {e}")
            # Return empty results with error info
            return RAGSearchResults(
                total_search_time=time.time() - start_time,
                queries_executed=0,
                total_results=0,
                project_context=project_name if 'project_name' in locals() else "unknown",
                search_strategy="failed"
            )
    
    def execute_focused_search(
        self,
        search_query: SearchQuery,
        project_path: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Execute a single focused search query with enhanced analysis.
        
        Args:
            search_query: Structured search query to execute
            project_path: Path to project (None for current project)
            
        Returns:
            List of enriched search results
        """
        try:
            self.logger.debug(f"Executing focused search: {search_query.query_type.value}")
            
            # Execute the search using existing search infrastructure
            raw_results = search_sync(
                query=search_query.query_text,
                n_results=search_query.expected_results,
                cross_project=False,  # Focus on current project for exploration
                search_mode=search_query.search_mode,
                include_context=True,
                context_chunks=search_query.context_chunks
            )
            
            if "error" in raw_results:
                self.logger.warning(f"Search failed for {search_query.query_type.value}: {raw_results['error']}")
                return []
            
            # Enrich results with confidence scoring and metadata
            enriched_results = []
            for result in raw_results.get("results", []):
                enriched_result = self._enrich_search_result(result, search_query.query_type)
                enriched_results.append(enriched_result)
            
            # Sort by confidence score
            enriched_results.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return enriched_results
            
        except Exception as e:
            self.logger.error(f"Focused search failed for {search_query.query_type.value}: {e}")
            return []
    
    def _initialize_query_templates(self) -> Dict[SearchQueryType, List[str]]:
        """Initialize search query templates for different exploration aspects."""
        return {
            SearchQueryType.ARCHITECTURE_PATTERNS: [
                "MVC model view controller pattern",
                "microservices architecture service",
                "layered architecture layers components",
                "design patterns factory singleton strategy",
                "application structure main components"
            ],
            SearchQueryType.ENTRY_POINTS: [
                "main function entry point startup",
                "app.py main.py server.js index.js",
                "application startup initialization bootstrap",
                "command line interface CLI entry",
                "web server setup route handler"
            ],
            SearchQueryType.CORE_COMPONENTS: [
                "core modules main components",
                "business logic service layer",
                "data access repository model",
                "utility helper functions common",
                "shared modules libraries components"
            ],
            SearchQueryType.DATA_FLOW: [
                "data flow processing pipeline",
                "request response handler",
                "database query operations CRUD",
                "API calls HTTP requests",
                "event handling message passing"
            ],
            SearchQueryType.CONFIGURATION: [
                "configuration settings config",
                "environment variables env",
                "database connection setup",
                "logging configuration setup",
                "application settings parameters"
            ],
            SearchQueryType.DEPENDENCIES: [
                "import dependencies modules",
                "external libraries packages",
                "third party integrations",
                "requirements dependencies setup",
                "package imports modules"
            ],
            SearchQueryType.TESTING_PATTERNS: [
                "test cases unit testing",
                "integration tests test suite",
                "mock fixtures test data",
                "testing patterns test framework",
                "test coverage validation"
            ],
            SearchQueryType.API_ENDPOINTS: [
                "API endpoints routes handlers",
                "REST API HTTP methods",
                "URL routing path handlers",
                "web service endpoints",
                "GraphQL resolvers mutations"
            ]
        }
    
    def _build_search_queries(
        self, 
        focus_areas: Optional[List[SearchQueryType]], 
        max_results: int
    ) -> List[SearchQuery]:
        """Build search queries based on focus areas."""
        if focus_areas is None:
            # Use all search types with priorities
            focus_areas = [
                SearchQueryType.ENTRY_POINTS,      # High priority
                SearchQueryType.ARCHITECTURE_PATTERNS,  # High priority
                SearchQueryType.CORE_COMPONENTS,   # High priority
                SearchQueryType.DATA_FLOW,         # Medium priority
                SearchQueryType.CONFIGURATION,     # Medium priority
                SearchQueryType.DEPENDENCIES,      # Low priority
                SearchQueryType.TESTING_PATTERNS,  # Low priority
                SearchQueryType.API_ENDPOINTS,     # Medium priority
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
                context_chunks=1
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
            SearchQueryType.CORE_COMPONENTS
        }
        
        medium_priority = {
            SearchQueryType.DATA_FLOW,
            SearchQueryType.API_ENDPOINTS,
            SearchQueryType.CONFIGURATION
        }
        
        if query_type in high_priority:
            return 1
        elif query_type in medium_priority:
            return 2
        else:
            return 3
    
    def _execute_parallel_searches(self, search_queries: List[SearchQuery]) -> Dict[SearchQueryType, List[SearchResult]]:
        """Execute multiple searches in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit all searches
            future_to_query = {
                executor.submit(self.execute_focused_search, query): query 
                for query in search_queries
            }
            
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
    
    def _execute_sequential_searches(self, search_queries: List[SearchQuery]) -> Dict[SearchQueryType, List[SearchResult]]:
        """Execute searches sequentially."""
        results = {}
        
        for query in search_queries:
            query_results = self.execute_focused_search(query)
            results[query.query_type] = query_results
        
        return results
    
    def _enrich_search_result(self, raw_result: Dict[str, Any], query_type: SearchQueryType) -> SearchResult:
        """Enrich a raw search result with metadata and confidence scoring."""
        # Calculate confidence score
        confidence_score = self._calculate_confidence_score(raw_result, query_type)
        
        # Extract relevance indicators
        relevance_indicators = self._extract_relevance_indicators(raw_result, query_type)
        
        # Build breadcrumb context
        breadcrumb_context = self._build_breadcrumb_context(raw_result)
        
        # Extract function signature
        function_signature = raw_result.get("signature", "")
        
        # Determine component type
        component_type = self._determine_component_type(raw_result, query_type)
        
        return SearchResult(
            original_result=raw_result,
            query_type=query_type,
            confidence_score=confidence_score,
            relevance_indicators=relevance_indicators,
            breadcrumb_context=breadcrumb_context,
            function_signature=function_signature,
            component_type=component_type
        )
    
    def _calculate_confidence_score(self, result: Dict[str, Any], query_type: SearchQueryType) -> float:
        """Calculate confidence score for a search result."""
        score = 0.0
        
        # Base semantic score
        semantic_score = result.get("score", 0.0)
        score += semantic_score * self.scoring_weights["semantic_score"]
        
        # Keyword matching bonus
        content = result.get("content", "").lower()
        chunk_name = result.get("name", "").lower()
        file_path = result.get("file_path", "").lower()
        
        keyword_bonus = self._calculate_keyword_bonus(content, chunk_name, file_path, query_type)
        score += keyword_bonus * self.scoring_weights["keyword_match"]
        
        # File location bonus
        location_bonus = self._calculate_location_bonus(file_path, query_type)
        score += location_bonus * self.scoring_weights["file_location"]
        
        # Function metadata bonus
        metadata_bonus = self._calculate_metadata_bonus(result, query_type)
        score += metadata_bonus * self.scoring_weights["function_metadata"]
        
        return min(1.0, max(0.0, score))  # Clamp to [0, 1]
    
    def _calculate_keyword_bonus(self, content: str, chunk_name: str, file_path: str, query_type: SearchQueryType) -> float:
        """Calculate keyword matching bonus."""
        bonus = 0.0
        
        # Define relevant keywords for each query type
        keyword_sets = {
            SearchQueryType.ENTRY_POINTS: ["main", "app", "server", "index", "startup", "init", "run", "execute"],
            SearchQueryType.ARCHITECTURE_PATTERNS: ["pattern", "architecture", "mvc", "layer", "component", "service", "controller", "model"],
            SearchQueryType.CORE_COMPONENTS: ["core", "main", "primary", "central", "key", "important", "business", "logic"],
            SearchQueryType.DATA_FLOW: ["flow", "process", "pipeline", "stream", "data", "transform", "handle", "process"],
            SearchQueryType.CONFIGURATION: ["config", "setting", "env", "environment", "setup", "parameter", "option"],
            SearchQueryType.DEPENDENCIES: ["import", "require", "dependency", "package", "library", "module", "external"],
            SearchQueryType.TESTING_PATTERNS: ["test", "spec", "mock", "fixture", "suite", "coverage", "assert"],
            SearchQueryType.API_ENDPOINTS: ["api", "endpoint", "route", "handler", "http", "get", "post", "put", "delete"]
        }
        
        relevant_keywords = keyword_sets.get(query_type, [])
        
        # Check content for keywords
        for keyword in relevant_keywords:
            if keyword in content:
                bonus += 0.1
            if keyword in chunk_name:
                bonus += 0.15
            if keyword in file_path:
                bonus += 0.05
        
        return min(1.0, bonus)
    
    def _calculate_location_bonus(self, file_path: str, query_type: SearchQueryType) -> float:
        """Calculate file location relevance bonus."""
        bonus = 0.0
        
        # Define location patterns for different query types
        location_patterns = {
            SearchQueryType.ENTRY_POINTS: ["main", "app", "server", "index", "cli", "run"],
            SearchQueryType.CORE_COMPONENTS: ["core", "main", "src", "lib", "business", "service"],
            SearchQueryType.CONFIGURATION: ["config", "settings", "env", "conf"],
            SearchQueryType.TESTING_PATTERNS: ["test", "tests", "spec", "specs", "__test__"],
            SearchQueryType.API_ENDPOINTS: ["api", "routes", "endpoints", "handlers", "controllers"]
        }
        
        patterns = location_patterns.get(query_type, [])
        
        for pattern in patterns:
            if pattern in file_path.lower():
                bonus += 0.2
        
        return min(1.0, bonus)
    
    def _calculate_metadata_bonus(self, result: Dict[str, Any], query_type: SearchQueryType) -> float:
        """Calculate metadata-based relevance bonus."""
        bonus = 0.0
        
        chunk_type = result.get("chunk_type", "")
        language = result.get("language", "")
        docstring = result.get("docstring", "")
        
        # Bonus for specific chunk types
        if query_type == SearchQueryType.ENTRY_POINTS and chunk_type == "function":
            bonus += 0.3
        elif query_type == SearchQueryType.CORE_COMPONENTS and chunk_type in ["class", "function"]:
            bonus += 0.2
        elif query_type == SearchQueryType.API_ENDPOINTS and chunk_type == "function":
            bonus += 0.3
        
        # Bonus for good documentation
        if docstring and len(docstring) > 20:
            bonus += 0.1
        
        return min(1.0, bonus)
    
    def _extract_relevance_indicators(self, result: Dict[str, Any], query_type: SearchQueryType) -> List[str]:
        """Extract relevance indicators from search result."""
        indicators = []
        
        # Add score-based indicator
        score = result.get("score", 0.0)
        if score > 0.8:
            indicators.append("High semantic relevance")
        elif score > 0.6:
            indicators.append("Good semantic relevance")
        
        # Add chunk type indicator
        chunk_type = result.get("chunk_type", "")
        if chunk_type:
            indicators.append(f"Code {chunk_type}")
        
        # Add language indicator
        language = result.get("language", "")
        if language:
            indicators.append(f"{language} code")
        
        # Add documentation indicator
        docstring = result.get("docstring", "")
        if docstring:
            indicators.append("Well documented")
        
        return indicators
    
    def _build_breadcrumb_context(self, result: Dict[str, Any]) -> str:
        """Build breadcrumb context for the result."""
        breadcrumb = result.get("breadcrumb", "")
        if breadcrumb:
            return breadcrumb
        
        # Fallback: build from available metadata
        file_path = result.get("file_path", "")
        chunk_name = result.get("name", "")
        parent_name = result.get("parent_name", "")
        
        parts = []
        if file_path:
            parts.append(Path(file_path).name)
        if parent_name:
            parts.append(parent_name)
        if chunk_name:
            parts.append(chunk_name)
        
        return " > ".join(parts)
    
    def _determine_component_type(self, result: Dict[str, Any], query_type: SearchQueryType) -> str:
        """Determine the component type for the result."""
        chunk_type = result.get("chunk_type", "")
        chunk_name = result.get("name", "")
        file_path = result.get("file_path", "")
        
        # Use chunk type as base
        if chunk_type:
            base_type = chunk_type.title()
        else:
            base_type = "Code"
        
        # Add context based on query type and location
        if query_type == SearchQueryType.ENTRY_POINTS:
            if "main" in chunk_name.lower() or "main" in file_path.lower():
                return f"Entry Point {base_type}"
        elif query_type == SearchQueryType.CORE_COMPONENTS:
            if "service" in file_path.lower():
                return f"Service {base_type}"
            elif "model" in file_path.lower():
                return f"Model {base_type}"
            elif "controller" in file_path.lower():
                return f"Controller {base_type}"
        elif query_type == SearchQueryType.API_ENDPOINTS:
            return f"API {base_type}"
        elif query_type == SearchQueryType.TESTING_PATTERNS:
            return f"Test {base_type}"
        
        return base_type
    
    def _organize_search_results(self, search_results: Dict[SearchQueryType, List[SearchResult]]) -> RAGSearchResults:
        """Organize search results into structured format."""
        return RAGSearchResults(
            architecture_insights=search_results.get(SearchQueryType.ARCHITECTURE_PATTERNS, []),
            entry_points=search_results.get(SearchQueryType.ENTRY_POINTS, []),
            core_components=search_results.get(SearchQueryType.CORE_COMPONENTS, []),
            data_flow_patterns=search_results.get(SearchQueryType.DATA_FLOW, []),
            configuration_elements=search_results.get(SearchQueryType.CONFIGURATION, []),
            dependencies=search_results.get(SearchQueryType.DEPENDENCIES, []),
            testing_patterns=search_results.get(SearchQueryType.TESTING_PATTERNS, []),
            api_endpoints=search_results.get(SearchQueryType.API_ENDPOINTS, [])
        )
    
    def _count_total_results(self, results: RAGSearchResults) -> int:
        """Count total results across all categories."""
        return (
            len(results.architecture_insights) +
            len(results.entry_points) +
            len(results.core_components) +
            len(results.data_flow_patterns) +
            len(results.configuration_elements) +
            len(results.dependencies) +
            len(results.testing_patterns) +
            len(results.api_endpoints)
        )
    
    def detect_architecture_patterns(
        self,
        project_path: Optional[str] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Detect architecture patterns using RAG semantic search.
        
        This method performs specialized searches to identify common architectural
        patterns in the codebase by analyzing code structure, naming conventions,
        and functional relationships.
        
        Args:
            project_path: Path to project (None for current project)
            max_results: Maximum results per pattern search
            
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
                    "separation concerns model data view presentation controller logic"
                ],
                "microservices": [
                    "microservices service architecture distributed",
                    "API gateway service discovery load balancer",
                    "independent deployable services communication"
                ],
                "layered": [
                    "layered architecture layers presentation business data",
                    "repository pattern data access layer service layer",
                    "three tier architecture N-tier layer separation"
                ],
                "component_based": [
                    "component based architecture modular components",
                    "component composition reusable modules",
                    "plugin architecture component interfaces"
                ],
                "event_driven": [
                    "event driven architecture events publisher subscriber",
                    "message queue event bus asynchronous processing",
                    "event sourcing CQRS pattern"
                ],
                "domain_driven": [
                    "domain driven design DDD bounded context",
                    "aggregate repository domain model entity",
                    "domain services value objects ubiquitous language"
                ],
                "hexagonal": [
                    "hexagonal architecture ports adapters",
                    "dependency inversion clean architecture",
                    "infrastructure domain application layer separation"
                ],
                "pipe_filter": [
                    "pipeline filter processing chain transform",
                    "data pipeline stream processing ETL",
                    "functional pipeline composition"
                ]
            }
            
            # Execute pattern detection searches
            pattern_results = {}
            
            for pattern_name, queries in pattern_queries.items():
                pattern_evidence = []
                
                for query_text in queries:
                    # Execute focused search for this pattern
                    query = SearchQuery(
                        query_type=SearchQueryType.ARCHITECTURE_PATTERNS,
                        query_text=query_text,
                        expected_results=max_results,
                        search_mode="semantic",
                        context_chunks=0  # Don't need context for pattern detection
                    )
                    
                    search_results = self.execute_focused_search(query, project_path)
                    
                    # Analyze results for pattern indicators
                    for result in search_results:
                        pattern_score = self._analyze_pattern_evidence(result, pattern_name)
                        if pattern_score > 0.3:  # Minimum threshold for pattern evidence
                            pattern_evidence.append({
                                "file_path": result.original_result.get("file_path", ""),
                                "content_snippet": result.original_result.get("content", "")[:200],
                                "confidence": pattern_score,
                                "line_range": f"{result.original_result.get('line_start', 0)}-{result.original_result.get('line_end', 0)}",
                                "component_type": result.component_type,
                                "search_score": result.original_result.get("score", 0.0)
                            })
                
                # Calculate overall pattern confidence
                if pattern_evidence:
                    # Sort by confidence and take top results
                    pattern_evidence.sort(key=lambda x: x["confidence"], reverse=True)
                    pattern_evidence = pattern_evidence[:5]  # Top 5 evidence pieces
                    
                    # Calculate aggregate confidence
                    aggregate_confidence = self._calculate_pattern_confidence(pattern_evidence)
                    
                    pattern_results[pattern_name] = {
                        "pattern_name": pattern_name.replace("_", " ").title(),
                        "confidence": aggregate_confidence,
                        "evidence_count": len(pattern_evidence),
                        "evidence": pattern_evidence,
                        "detected": aggregate_confidence > 0.5
                    }
            
            # Identify the most likely architecture pattern
            detected_patterns = {
                name: info for name, info in pattern_results.items() 
                if info["detected"]
            }
            
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
                    "highest_confidence": max(
                        [info["confidence"] for info in pattern_results.values()], 
                        default=0.0
                    ),
                    "analysis_method": "RAG semantic search with pattern matching"
                }
            }
            
            self.logger.info(f"Architecture pattern detection completed")
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
                    "analysis_method": "failed"
                }
            }
    
    def _analyze_pattern_evidence(self, search_result: SearchResult, pattern_name: str) -> float:
        """
        Analyze a search result for evidence of a specific architecture pattern.
        
        Args:
            search_result: Search result to analyze
            pattern_name: Name of the pattern to check for
            
        Returns:
            Confidence score (0.0-1.0) for pattern evidence
        """
        content = search_result.original_result.get("content", "").lower()
        file_path = search_result.original_result.get("file_path", "").lower()
        chunk_name = search_result.original_result.get("name", "").lower()
        docstring = search_result.original_result.get("docstring", "").lower()
        
        # Pattern-specific indicators
        pattern_indicators = {
            "mvc": {
                "keywords": ["model", "view", "controller", "template", "render", "action", "form"],
                "file_patterns": ["model", "view", "controller", "template"],
                "structure_indicators": ["models/", "views/", "controllers/", "templates/"]
            },
            "microservices": {
                "keywords": ["service", "api", "microservice", "gateway", "discovery", "circuit", "async"],
                "file_patterns": ["service", "api", "gateway", "client"],
                "structure_indicators": ["services/", "api/", "gateway/", "client/"]
            },
            "layered": {
                "keywords": ["layer", "tier", "repository", "service", "dao", "business", "presentation"],
                "file_patterns": ["repository", "service", "dao", "business", "data"],
                "structure_indicators": ["data/", "business/", "service/", "repository/"]
            },
            "component_based": {
                "keywords": ["component", "module", "plugin", "extension", "interface"],
                "file_patterns": ["component", "module", "plugin"],
                "structure_indicators": ["components/", "modules/", "plugins/"]
            },
            "event_driven": {
                "keywords": ["event", "listener", "handler", "publisher", "subscriber", "queue", "bus"],
                "file_patterns": ["event", "listener", "handler", "queue"],
                "structure_indicators": ["events/", "handlers/", "listeners/"]
            },
            "domain_driven": {
                "keywords": ["domain", "aggregate", "entity", "value", "repository", "service", "bounded"],
                "file_patterns": ["domain", "aggregate", "entity", "repository"],
                "structure_indicators": ["domain/", "entities/", "aggregates/"]
            },
            "hexagonal": {
                "keywords": ["port", "adapter", "infrastructure", "domain", "application", "dependency"],
                "file_patterns": ["port", "adapter", "infrastructure"],
                "structure_indicators": ["ports/", "adapters/", "infrastructure/"]
            },
            "pipe_filter": {
                "keywords": ["pipeline", "filter", "transform", "process", "chain", "stream"],
                "file_patterns": ["pipeline", "filter", "processor", "transform"],
                "structure_indicators": ["pipelines/", "filters/", "processors/"]
            }
        }
        
        indicators = pattern_indicators.get(pattern_name, {})
        score = 0.0
        
        # Check keywords in content
        keywords = indicators.get("keywords", [])
        for keyword in keywords:
            if keyword in content:
                score += 0.1
            if keyword in chunk_name:
                score += 0.15
            if keyword in docstring:
                score += 0.1
        
        # Check file path patterns
        file_patterns = indicators.get("file_patterns", [])
        for pattern in file_patterns:
            if pattern in file_path:
                score += 0.2
        
        # Check directory structure indicators
        structure_indicators = indicators.get("structure_indicators", [])
        for indicator in structure_indicators:
            if indicator in file_path:
                score += 0.3
        
        # Boost score based on original search relevance
        search_score = search_result.original_result.get("score", 0.0)
        score += search_score * 0.2
        
        return min(1.0, score)  # Cap at 1.0
    
    def _calculate_pattern_confidence(self, evidence_list: List[Dict[str, Any]]) -> float:
        """
        Calculate aggregate confidence for a pattern based on evidence.
        
        Args:
            evidence_list: List of evidence pieces with confidence scores
            
        Returns:
            Aggregate confidence score (0.0-1.0)
        """
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

    def discover_entry_points(
        self,
        project_path: Optional[str] = None,
        max_results: int = 10
    ) -> Dict[str, Any]:
        """
        Discover project entry points using function-level RAG search.
        
        This method identifies main entry points, startup functions, command-line interfaces,
        and web server initialization points by analyzing function signatures, names, and contexts.
        
        Args:
            project_path: Path to project (None for current project)
            max_results: Maximum results per entry point search
            
        Returns:
            Dictionary containing discovered entry points with metadata
        """
        try:
            self.logger.info("Starting entry point discovery using function-level RAG search")
            
            # Define entry point search queries targeting specific types
            entry_point_queries = {
                "main_functions": [
                    "main function entry point startup",
                    "if __name__ == '__main__' main function",
                    "main() function application entry"
                ],
                "cli_entry_points": [
                    "command line interface CLI argparse click",
                    "parser.add_argument command line arguments",
                    "click.command() CLI decorator command"
                ],
                "web_server_startup": [
                    "app.run() flask server startup",
                    "uvicorn.run() fastapi server start",
                    "app.listen() express server startup",
                    "server.start() web server initialization"
                ],
                "application_factories": [
                    "create_app() application factory function",
                    "app_factory application builder setup",
                    "initialize_app() setup configuration"
                ],
                "script_executables": [
                    "#!/usr/bin/env python shebang executable",
                    "sys.argv command line script execution",
                    "argparse.ArgumentParser() script arguments"
                ],
                "service_launchers": [
                    "daemon service launcher background process",
                    "systemd service unit file startup",
                    "background task scheduler worker"
                ]
            }
            
            # Execute entry point discovery searches
            entry_point_results = {}
            
            for category, queries in entry_point_queries.items():
                category_results = []
                
                for query_text in queries:
                    # Execute focused search for entry points
                    query = SearchQuery(
                        query_type=SearchQueryType.ENTRY_POINTS,
                        query_text=query_text,
                        expected_results=max_results,
                        search_mode="hybrid",  # Hybrid for better keyword + semantic matching
                        context_chunks=1  # Include context for better understanding
                    )
                    
                    search_results = self.execute_focused_search(query, project_path)
                    
                    # Analyze results for entry point characteristics
                    for result in search_results:
                        entry_point_score = self._analyze_entry_point_characteristics(result, category)
                        if entry_point_score > 0.4:  # Threshold for entry point candidacy
                            
                            # Extract additional metadata specific to entry points
                            entry_metadata = self._extract_entry_point_metadata(result)
                            
                            category_results.append({
                                "file_path": result.original_result.get("file_path", ""),
                                "function_name": result.original_result.get("name", ""),
                                "function_signature": result.original_result.get("signature", ""),
                                "content_snippet": result.original_result.get("content", "")[:300],
                                "confidence": entry_point_score,
                                "line_range": f"{result.original_result.get('line_start', 0)}-{result.original_result.get('line_end', 0)}",
                                "breadcrumb": result.breadcrumb_context,
                                "docstring": result.original_result.get("docstring", ""),
                                "language": result.original_result.get("language", ""),
                                "search_score": result.original_result.get("score", 0.0),
                                "entry_type": category.replace("_", " ").title(),
                                **entry_metadata
                            })
                
                # Sort by confidence and deduplicate
                if category_results:
                    category_results = self._deduplicate_entry_points(category_results)
                    category_results.sort(key=lambda x: x["confidence"], reverse=True)
                    category_results = category_results[:5]  # Top 5 per category
                    entry_point_results[category] = category_results
            
            # Identify the most likely primary entry points
            all_entry_points = []
            for category, results in entry_point_results.items():
                all_entry_points.extend(results)
            
            # Sort all entry points by confidence
            all_entry_points.sort(key=lambda x: x["confidence"], reverse=True)
            
            # Identify primary entry points (highest confidence, different files)
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
                    "highest_confidence": max(
                        [ep["confidence"] for ep in all_entry_points], 
                        default=0.0
                    ),
                    "analysis_method": "Function-level RAG search with signature analysis"
                }
            }
            
            self.logger.info(f"Entry point discovery completed")
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
                    "analysis_method": "failed"
                }
            }
    
    def _analyze_entry_point_characteristics(self, search_result: SearchResult, category: str) -> float:
        """
        Analyze a search result for entry point characteristics.
        
        Args:
            search_result: Search result to analyze
            category: Entry point category being analyzed
            
        Returns:
            Confidence score (0.0-1.0) for entry point likelihood
        """
        content = search_result.original_result.get("content", "").lower()
        file_path = search_result.original_result.get("file_path", "").lower()
        function_name = search_result.original_result.get("name", "").lower()
        signature = search_result.original_result.get("signature", "").lower()
        chunk_type = search_result.original_result.get("chunk_type", "")
        
        score = 0.0
        
        # Category-specific indicators
        category_indicators = {
            "main_functions": {
                "function_names": ["main", "run", "start", "execute", "app"],
                "content_patterns": ["if __name__", "main()", "app.run", "start()"],
                "file_patterns": ["main.py", "app.py", "run.py", "start.py"]
            },
            "cli_entry_points": {
                "function_names": ["cli", "main", "parse", "command"],
                "content_patterns": ["argparse", "click.command", "parser.add_argument", "sys.argv"],
                "file_patterns": ["cli.py", "main.py", "command.py"]
            },
            "web_server_startup": {
                "function_names": ["run", "start", "serve", "create_app", "app"],
                "content_patterns": ["app.run", "uvicorn.run", "app.listen", "server.start"],
                "file_patterns": ["app.py", "server.py", "main.py", "wsgi.py"]
            },
            "application_factories": {
                "function_names": ["create_app", "make_app", "init_app", "setup_app"],
                "content_patterns": ["create_app", "app_factory", "flask.Flask", "fastapi.FastAPI"],
                "file_patterns": ["app.py", "factory.py", "application.py"]
            },
            "script_executables": {
                "function_names": ["main", "run", "execute", "script"],
                "content_patterns": ["#!/usr/bin/env", "sys.argv", "if __name__"],
                "file_patterns": ["script.py", "run.py", "execute.py"]
            },
            "service_launchers": {
                "function_names": ["start", "run", "daemon", "service"],
                "content_patterns": ["daemon", "service", "background", "worker"],
                "file_patterns": ["daemon.py", "service.py", "worker.py"]
            }
        }
        
        indicators = category_indicators.get(category, {})
        
        # Check function name patterns
        function_names = indicators.get("function_names", [])
        for name_pattern in function_names:
            if name_pattern in function_name:
                score += 0.3
        
        # Check content patterns
        content_patterns = indicators.get("content_patterns", [])
        for pattern in content_patterns:
            if pattern in content:
                score += 0.2
        
        # Check file path patterns
        file_patterns = indicators.get("file_patterns", [])
        for pattern in file_patterns:
            if pattern in file_path:
                score += 0.25
        
        # Boost for function chunk type (entry points are usually functions)
        if chunk_type == "function":
            score += 0.15
        
        # Check for typical entry point signatures
        entry_point_signatures = [
            "main()", "main(args", "main(argv", "run(", "start(", 
            "create_app(", "if __name__"
        ]
        for sig_pattern in entry_point_signatures:
            if sig_pattern in signature or sig_pattern in content:
                score += 0.1
        
        # Boost based on original search relevance
        search_score = search_result.original_result.get("score", 0.0)
        score += search_score * 0.15
        
        return min(1.0, score)  # Cap at 1.0
    
    def _extract_entry_point_metadata(self, search_result: SearchResult) -> Dict[str, Any]:
        """
        Extract additional metadata specific to entry points.
        
        Args:
            search_result: Search result to extract metadata from
            
        Returns:
            Dictionary with entry point specific metadata
        """
        content = search_result.original_result.get("content", "")
        signature = search_result.original_result.get("signature", "")
        file_path = search_result.original_result.get("file_path", "")
        
        metadata = {
            "is_executable": False,
            "has_cli_interface": False,
            "has_web_interface": False,
            "framework_detected": None,
            "execution_context": "unknown"
        }
        
        content_lower = content.lower()
        
        # Check if executable
        if "if __name__" in content_lower or "#!/usr/bin" in content_lower:
            metadata["is_executable"] = True
            metadata["execution_context"] = "script"
        
        # Check for CLI interface
        cli_indicators = ["argparse", "click", "sys.argv", "command", "parser"]
        if any(indicator in content_lower for indicator in cli_indicators):
            metadata["has_cli_interface"] = True
            metadata["execution_context"] = "cli"
        
        # Check for web interface
        web_indicators = ["app.run", "uvicorn", "gunicorn", "flask", "fastapi", "express"]
        if any(indicator in content_lower for indicator in web_indicators):
            metadata["has_web_interface"] = True
            metadata["execution_context"] = "web"
        
        # Detect framework
        framework_patterns = {
            "Flask": ["flask", "app = flask"],
            "FastAPI": ["fastapi", "fastapi.fastapi"],
            "Django": ["django", "manage.py"],
            "Express": ["express", "app.listen"],
            "Click": ["click.command", "@click"],
            "Argparse": ["argparse", "argumentparser"]
        }
        
        for framework, patterns in framework_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                metadata["framework_detected"] = framework
                break
        
        return metadata
    
    def _deduplicate_entry_points(self, entry_points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate entry points based on file path and function name.
        
        Args:
            entry_points: List of entry point results
            
        Returns:
            Deduplicated list with highest confidence entries kept
        """
        seen = {}
        
        for entry_point in entry_points:
            key = (entry_point["file_path"], entry_point["function_name"])
            
            if key not in seen or entry_point["confidence"] > seen[key]["confidence"]:
                seen[key] = entry_point
        
        return list(seen.values())

    def analyze_component_relationships(
        self,
        project_path: Optional[str] = None,
        similarity_threshold: float = 0.7,
        max_components: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze component relationships using vector similarity.
        
        This method discovers relationships between code components by analyzing
        vector similarity between functions, classes, and modules. It identifies
        related components, potential dependencies, and architectural connections.
        
        Args:
            project_path: Path to project (None for current project)
            similarity_threshold: Minimum similarity score for relationships (0.0-1.0)
            max_components: Maximum number of components to analyze
            
        Returns:
            Dictionary containing component relationships and analysis
        """
        try:
            self.logger.info("Starting component relationship analysis using vector similarity")
            
            # Step 1: Discover core components first
            core_components = self._discover_core_components(project_path, max_components)
            
            if not core_components:
                return {
                    "error": "No core components found for relationship analysis",
                    "relationships": [],
                    "component_clusters": [],
                    "analysis_summary": {
                        "components_analyzed": 0,
                        "relationships_found": 0,
                        "clusters_identified": 0,
                        "analysis_method": "failed"
                    }
                }
            
            # Step 2: Analyze pairwise relationships between components
            relationships = self._analyze_pairwise_relationships(
                core_components, 
                similarity_threshold,
                project_path
            )
            
            # Step 3: Identify component clusters (groups of related components)
            clusters = self._identify_component_clusters(core_components, relationships)
            
            # Step 4: Analyze dependency patterns
            dependency_analysis = self._analyze_dependency_patterns(relationships, core_components)
            
            # Step 5: Identify architectural insights
            architectural_insights = self._extract_architectural_insights(clusters, relationships)
            
            result = {
                "core_components": core_components,
                "relationships": relationships,
                "component_clusters": clusters,
                "dependency_analysis": dependency_analysis,
                "architectural_insights": architectural_insights,
                "analysis_summary": {
                    "components_analyzed": len(core_components),
                    "relationships_found": len(relationships),
                    "clusters_identified": len(clusters),
                    "similarity_threshold": similarity_threshold,
                    "analysis_method": "Vector similarity with clustering analysis"
                }
            }
            
            self.logger.info(f"Component relationship analysis completed")
            self.logger.info(f"Analyzed {len(core_components)} components, found {len(relationships)} relationships")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Component relationship analysis failed: {e}")
            return {
                "error": str(e),
                "relationships": [],
                "component_clusters": [],
                "analysis_summary": {
                    "components_analyzed": 0,
                    "relationships_found": 0,
                    "clusters_identified": 0,
                    "analysis_method": "failed"
                }
            }
    
    def _discover_core_components(
        self, 
        project_path: Optional[str], 
        max_components: int
    ) -> List[Dict[str, Any]]:
        """
        Discover core components for relationship analysis.
        
        Args:
            project_path: Path to project
            max_components: Maximum components to discover
            
        Returns:
            List of core component metadata
        """
        # Search for various types of core components
        component_queries = [
            "class definition main components",
            "function definition service module",
            "API endpoint handler function",
            "data model class entity",
            "utility helper function module",
            "configuration setup function",
            "service layer business logic",
            "repository data access pattern"
        ]
        
        all_components = []
        
        for query_text in component_queries:
            query = SearchQuery(
                query_type=SearchQueryType.CORE_COMPONENTS,
                query_text=query_text,
                expected_results=max_components // len(component_queries) + 2,
                search_mode="semantic",
                context_chunks=0
            )
            
            search_results = self.execute_focused_search(query, project_path)
            
            for result in search_results:
                component_score = self._calculate_component_importance(result)
                if component_score > 0.5:  # Threshold for core component
                    component_info = {
                        "file_path": result.original_result.get("file_path", ""),
                        "name": result.original_result.get("name", ""),
                        "signature": result.original_result.get("signature", ""),
                        "chunk_type": result.original_result.get("chunk_type", ""),
                        "language": result.original_result.get("language", ""),
                        "content": result.original_result.get("content", ""),
                        "docstring": result.original_result.get("docstring", ""),
                        "line_range": f"{result.original_result.get('line_start', 0)}-{result.original_result.get('line_end', 0)}",
                        "breadcrumb": result.breadcrumb_context,
                        "importance_score": component_score,
                        "search_score": result.original_result.get("score", 0.0),
                        "vector_id": result.original_result.get("id", "")  # For vector similarity
                    }
                    all_components.append(component_info)
        
        # Deduplicate and sort by importance
        unique_components = self._deduplicate_components(all_components)
        unique_components.sort(key=lambda x: x["importance_score"], reverse=True)
        
        return unique_components[:max_components]
    
    def _calculate_component_importance(self, search_result: SearchResult) -> float:
        """
        Calculate importance score for a component.
        
        Args:
            search_result: Search result to analyze
            
        Returns:
            Importance score (0.0-1.0)
        """
        score = 0.0
        
        # Base on search relevance
        search_score = search_result.original_result.get("score", 0.0)
        score += search_score * 0.3
        
        # Boost for certain chunk types
        chunk_type = search_result.original_result.get("chunk_type", "")
        if chunk_type in ["class", "function"]:
            score += 0.2
        
        # Boost for good documentation
        docstring = search_result.original_result.get("docstring", "")
        if docstring and len(docstring) > 50:
            score += 0.15
        
        # Boost for certain naming patterns (common important components)
        name = search_result.original_result.get("name", "").lower()
        important_patterns = ["service", "manager", "handler", "controller", "model", "repository"]
        for pattern in important_patterns:
            if pattern in name:
                score += 0.1
                break
        
        # Boost for core file locations
        file_path = search_result.original_result.get("file_path", "").lower()
        if any(pattern in file_path for pattern in ["core/", "main/", "src/", "service/"]):
            score += 0.1
        
        # Confidence boost
        confidence = search_result.confidence_score
        score += confidence * 0.15
        
        return min(1.0, score)
    
    def _deduplicate_components(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate components based on file path and name.
        
        Args:
            components: List of component information
            
        Returns:
            Deduplicated list with highest importance components kept
        """
        seen = {}
        
        for component in components:
            key = (component["file_path"], component["name"])
            
            if key not in seen or component["importance_score"] > seen[key]["importance_score"]:
                seen[key] = component
        
        return list(seen.values())
    
    def _analyze_pairwise_relationships(
        self, 
        components: List[Dict[str, Any]], 
        similarity_threshold: float,
        project_path: Optional[str]
    ) -> List[Dict[str, Any]]:
        """
        Analyze pairwise relationships between components using similarity.
        
        Args:
            components: List of components to analyze
            similarity_threshold: Minimum similarity for relationship
            project_path: Project path for context
            
        Returns:
            List of relationship information
        """
        relationships = []
        
        try:
            # Get embeddings manager for similarity calculation
            embeddings_manager = get_embeddings_manager_instance()
            embedding_model = os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")
            
            # For each pair of components, calculate semantic similarity
            for i, comp1 in enumerate(components):
                for j, comp2 in enumerate(components[i+1:], i+1):
                    
                    # Create combined content for similarity analysis
                    content1 = self._create_component_content_for_similarity(comp1)
                    content2 = self._create_component_content_for_similarity(comp2)
                    
                    if not content1 or not content2:
                        continue
                    
                    try:
                        # Generate embeddings for both components
                        embedding1 = embeddings_manager.generate_embeddings(embedding_model, content1)
                        embedding2 = embeddings_manager.generate_embeddings(embedding_model, content2)
                        
                        if embedding1 is None or embedding2 is None:
                            continue
                        
                        # Calculate cosine similarity
                        similarity = self._calculate_cosine_similarity(embedding1, embedding2)
                        
                        if similarity >= similarity_threshold:
                            relationship_type = self._determine_relationship_type(comp1, comp2, similarity)
                            relationship_strength = self._calculate_relationship_strength(comp1, comp2, similarity)
                            
                            relationship = {
                                "component1": {
                                    "name": comp1["name"],
                                    "file_path": comp1["file_path"],
                                    "type": comp1["chunk_type"]
                                },
                                "component2": {
                                    "name": comp2["name"],
                                    "file_path": comp2["file_path"],
                                    "type": comp2["chunk_type"]
                                },
                                "similarity_score": similarity,
                                "relationship_type": relationship_type,
                                "relationship_strength": relationship_strength,
                                "analysis_method": "vector_similarity"
                            }
                            relationships.append(relationship)
                    
                    except Exception as e:
                        self.logger.debug(f"Failed to analyze relationship between {comp1['name']} and {comp2['name']}: {e}")
                        continue
            
        except Exception as e:
            self.logger.warning(f"Pairwise relationship analysis failed: {e}")
        
        # Sort by similarity score
        relationships.sort(key=lambda x: x["similarity_score"], reverse=True)
        
        return relationships
    
    def _create_component_content_for_similarity(self, component: Dict[str, Any]) -> str:
        """
        Create content string for similarity analysis.
        
        Args:
            component: Component information
            
        Returns:
            Combined content string for embedding
        """
        parts = []
        
        # Add name and signature
        if component.get("name"):
            parts.append(component["name"])
        if component.get("signature"):
            parts.append(component["signature"])
        
        # Add docstring (important for semantic similarity)
        if component.get("docstring"):
            parts.append(component["docstring"])
        
        # Add content excerpt (first 200 chars)
        content = component.get("content", "")
        if content:
            parts.append(content[:200])
        
        return " ".join(parts)
    
    def _calculate_cosine_similarity(self, embedding1, embedding2) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (0.0-1.0)
        """
        if not NUMPY_AVAILABLE:
            self.logger.warning("NumPy not available, using fallback similarity calculation")
            return self._fallback_similarity_calculation(embedding1, embedding2)
            
        try:
            # Convert to numpy arrays if needed
            if hasattr(embedding1, 'tolist'):
                vec1 = np.array(embedding1.tolist())
            else:
                vec1 = np.array(embedding1)
                
            if hasattr(embedding2, 'tolist'):
                vec2 = np.array(embedding2.tolist())
            else:
                vec2 = np.array(embedding2)
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm_a = np.linalg.norm(vec1)
            norm_b = np.linalg.norm(vec2)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            
            # Normalize to 0-1 range (cosine similarity can be -1 to 1)
            return (similarity + 1) / 2
            
        except Exception as e:
            self.logger.debug(f"Failed to calculate cosine similarity: {e}")
            return 0.0
    
    def _fallback_similarity_calculation(self, embedding1, embedding2) -> float:
        """
        Fallback similarity calculation without NumPy.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Approximate similarity score (0.0-1.0)
        """
        try:
            # Convert to lists if needed
            if hasattr(embedding1, 'tolist'):
                vec1 = embedding1.tolist()
            else:
                vec1 = list(embedding1)
                
            if hasattr(embedding2, 'tolist'):
                vec2 = embedding2.tolist()
            else:
                vec2 = list(embedding2)
            
            if len(vec1) != len(vec2):
                return 0.0
            
            # Simple dot product and magnitude calculation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            mag1 = sum(a * a for a in vec1) ** 0.5
            mag2 = sum(b * b for b in vec2) ** 0.5
            
            if mag1 == 0 or mag2 == 0:
                return 0.0
            
            similarity = dot_product / (mag1 * mag2)
            
            # Normalize to 0-1 range
            return (similarity + 1) / 2
            
        except Exception as e:
            self.logger.debug(f"Fallback similarity calculation failed: {e}")
            return 0.0
    
    def _determine_relationship_type(
        self, 
        comp1: Dict[str, Any], 
        comp2: Dict[str, Any], 
        similarity: float
    ) -> str:
        """
        Determine the type of relationship between components.
        
        Args:
            comp1: First component
            comp2: Second component
            similarity: Similarity score
            
        Returns:
            Relationship type string
        """
        # Check if same file (likely related/dependent)
        if comp1["file_path"] == comp2["file_path"]:
            return "same_file_related"
        
        # Check naming patterns for specific relationships
        name1 = comp1["name"].lower()
        name2 = comp2["name"].lower()
        
        # Service-Repository pattern
        if "service" in name1 and "repository" in name2:
            return "service_repository"
        if "repository" in name1 and "service" in name2:
            return "repository_service"
        
        # Controller-Service pattern
        if "controller" in name1 and "service" in name2:
            return "controller_service"
        if "service" in name1 and "controller" in name2:
            return "service_controller"
        
        # Model-Service pattern
        if "model" in name1 and "service" in name2:
            return "model_service"
        if "service" in name1 and "model" in name2:
            return "service_model"
        
        # Similar functionality (high similarity)
        if similarity > 0.9:
            return "similar_functionality"
        elif similarity > 0.8:
            return "related_functionality"
        else:
            return "semantic_similarity"
    
    def _calculate_relationship_strength(
        self, 
        comp1: Dict[str, Any], 
        comp2: Dict[str, Any], 
        similarity: float
    ) -> str:
        """
        Calculate relationship strength category.
        
        Args:
            comp1: First component
            comp2: Second component
            similarity: Similarity score
            
        Returns:
            Relationship strength category
        """
        if similarity > 0.9:
            return "very_strong"
        elif similarity > 0.8:
            return "strong"
        elif similarity > 0.7:
            return "moderate"
        else:
            return "weak"
    
    def _identify_component_clusters(
        self, 
        components: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Identify clusters of related components.
        
        Args:
            components: List of components
            relationships: List of relationships
            
        Returns:
            List of component clusters
        """
        # Build adjacency list from relationships
        adjacency = {}
        for comp in components:
            comp_key = (comp["file_path"], comp["name"])
            adjacency[comp_key] = []
        
        for rel in relationships:
            comp1_key = (rel["component1"]["file_path"], rel["component1"]["name"])
            comp2_key = (rel["component2"]["file_path"], rel["component2"]["name"])
            
            if comp1_key in adjacency and comp2_key in adjacency:
                adjacency[comp1_key].append((comp2_key, rel["similarity_score"]))
                adjacency[comp2_key].append((comp1_key, rel["similarity_score"]))
        
        # Find connected components using DFS
        visited = set()
        clusters = []
        
        for comp in components:
            comp_key = (comp["file_path"], comp["name"])
            if comp_key not in visited:
                cluster = self._dfs_cluster(comp_key, adjacency, visited, components)
                if len(cluster) > 1:  # Only include clusters with multiple components
                    clusters.append(cluster)
        
        # Sort clusters by size and average importance
        clusters.sort(key=lambda x: (len(x["components"]), x["average_importance"]), reverse=True)
        
        return clusters
    
    def _dfs_cluster(
        self, 
        start_key: Tuple[str, str], 
        adjacency: Dict, 
        visited: set, 
        components: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Perform DFS to find connected component cluster.
        
        Args:
            start_key: Starting component key
            adjacency: Adjacency list
            visited: Set of visited components
            components: List of all components
            
        Returns:
            Cluster information
        """
        stack = [start_key]
        cluster_components = []
        total_importance = 0.0
        
        # Create component lookup
        comp_lookup = {(c["file_path"], c["name"]): c for c in components}
        
        while stack:
            current_key = stack.pop()
            
            if current_key in visited:
                continue
                
            visited.add(current_key)
            
            if current_key in comp_lookup:
                component = comp_lookup[current_key]
                cluster_components.append({
                    "name": component["name"],
                    "file_path": component["file_path"],
                    "type": component["chunk_type"],
                    "importance_score": component["importance_score"]
                })
                total_importance += component["importance_score"]
                
                # Add neighbors to stack
                for neighbor_key, similarity in adjacency.get(current_key, []):
                    if neighbor_key not in visited:
                        stack.append(neighbor_key)
        
        average_importance = total_importance / len(cluster_components) if cluster_components else 0.0
        
        return {
            "components": cluster_components,
            "cluster_size": len(cluster_components),
            "average_importance": average_importance,
            "total_importance": total_importance
        }
    
    def _analyze_dependency_patterns(
        self, 
        relationships: List[Dict[str, Any]], 
        components: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze dependency patterns from relationships.
        
        Args:
            relationships: List of relationships
            components: List of components
            
        Returns:
            Dependency analysis results
        """
        # Count relationship types
        relationship_type_counts = {}
        for rel in relationships:
            rel_type = rel["relationship_type"]
            relationship_type_counts[rel_type] = relationship_type_counts.get(rel_type, 0) + 1
        
        # Identify hub components (highly connected)
        component_connections = {}
        for rel in relationships:
            comp1_name = rel["component1"]["name"]
            comp2_name = rel["component2"]["name"]
            
            component_connections[comp1_name] = component_connections.get(comp1_name, 0) + 1
            component_connections[comp2_name] = component_connections.get(comp2_name, 0) + 1
        
        # Sort to find most connected components
        hub_components = sorted(
            component_connections.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        return {
            "relationship_type_distribution": relationship_type_counts,
            "hub_components": hub_components,
            "total_connections": len(relationships),
            "average_connections_per_component": len(relationships) * 2 / len(components) if components else 0
        }
    
    def _extract_architectural_insights(
        self, 
        clusters: List[Dict[str, Any]], 
        relationships: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Extract architectural insights from clusters and relationships.
        
        Args:
            clusters: List of component clusters
            relationships: List of relationships
            
        Returns:
            List of architectural insight strings
        """
        insights = []
        
        # Cluster insights
        if clusters:
            largest_cluster = max(clusters, key=lambda x: x["cluster_size"])
            insights.append(
                f"Largest component cluster contains {largest_cluster['cluster_size']} related components"
            )
            
            if len(clusters) > 3:
                insights.append(
                    f"Identified {len(clusters)} distinct component clusters, suggesting modular architecture"
                )
        
        # Relationship pattern insights
        if relationships:
            strong_relationships = [r for r in relationships if r["relationship_strength"] in ["strong", "very_strong"]]
            if strong_relationships:
                insights.append(
                    f"Found {len(strong_relationships)} strong component relationships indicating tight coupling"
                )
            
            # Check for architectural patterns
            service_repo_rels = [r for r in relationships if "service" in r["relationship_type"] and "repository" in r["relationship_type"]]
            if service_repo_rels:
                insights.append("Service-Repository pattern detected in component relationships")
            
            controller_service_rels = [r for r in relationships if "controller" in r["relationship_type"] and "service" in r["relationship_type"]]
            if controller_service_rels:
                insights.append("Controller-Service pattern detected in component relationships")
        
        return insights

    def get_search_strategy_info(self) -> Dict[str, Any]:
        """Get information about available search strategies and configurations."""
        return {
            "available_query_types": [query_type.value for query_type in SearchQueryType],
            "search_modes": ["semantic", "hybrid", "keyword"],
            "priority_levels": {"1": "high", "2": "medium", "3": "low"},
            "scoring_weights": self.scoring_weights,
            "query_templates_count": {
                query_type.value: len(templates) 
                for query_type, templates in self.query_templates.items()
            },
            "max_parallel_searches": 4,
            "default_results_per_query": 5,
            "supported_features": [
                "Multi-query parallel search",
                "Confidence scoring",
                "Metadata enrichment", 
                "Breadcrumb context",
                "Component type detection",
                "Relevance indicators"
            ]
        }