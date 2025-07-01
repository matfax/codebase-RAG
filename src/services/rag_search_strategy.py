"""
RAG Search Strategy Service

This service implements intelligent multi-query RAG search strategies for project exploration,
enabling comprehensive project analysis by leveraging the existing vectorized knowledge base.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

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