"""
Lightweight Graph Service Implementation for Agentic RAG Performance Enhancement.

This service implements memory indexing mechanism to store key node metadata in memory
for fast querying, supporting the full project processing by removing MCP limitations.
"""

import asyncio
import logging
import re
import time
import weakref
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Optional, Union

from ..models.code_chunk import ChunkType, CodeChunk
from .cache_service import BaseCacheService
from .graph_rag_service import GraphRAGService
from .hybrid_search_service import HybridSearchService
from .structure_relationship_builder import GraphEdge, GraphNode, StructureGraph


class GraphExpansionStrategy(Enum):
    """Strategies for expanding partial graphs."""

    BREADTH_FIRST = "breadth_first"
    DEPTH_FIRST = "depth_first"
    IMPORTANCE_BASED = "importance_based"
    RELEVANCE_SCORED = "relevance_scored"
    ADAPTIVE = "adaptive"


class QueryComplexity(Enum):
    """Query complexity levels for adaptive processing."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class GraphBuildOptions:
    """Options for partial graph construction."""

    max_nodes: int = 50
    expansion_strategy: GraphExpansionStrategy = GraphExpansionStrategy.ADAPTIVE
    include_context: bool = True
    context_depth: int = 2
    importance_threshold: float = 0.3
    relevance_threshold: float = 0.5
    max_expansion_rounds: int = 5
    prefer_connected_components: bool = True


@dataclass
class NodeRelevanceScore:
    """Relevance score for a node in context of a query."""

    node_id: str
    semantic_relevance: float = 0.0
    structural_relevance: float = 0.0
    importance_bonus: float = 0.0
    total_score: float = 0.0

    def calculate_total(self) -> float:
        """Calculate total relevance score."""
        self.total_score = self.semantic_relevance * 0.5 + self.structural_relevance * 0.3 + self.importance_bonus * 0.2
        return self.total_score


@dataclass
class NodeMetadata:
    """Lightweight metadata for graph nodes stored in memory index."""

    node_id: str
    name: str
    chunk_type: ChunkType
    file_path: str
    breadcrumb: str | None = None
    parent_name: str | None = None
    signature: str | None = None
    language: str = ""
    line_start: int = 0
    line_end: int = 0
    importance_score: float = 0.0
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    # Fast lookup indices
    children_ids: set[str] = field(default_factory=set)
    parent_ids: set[str] = field(default_factory=set)
    dependency_ids: set[str] = field(default_factory=set)


@dataclass
class QueryCacheEntry:
    """Cache entry for query results with metadata."""

    result: Any
    timestamp: datetime
    access_count: int = 0
    ttl_seconds: int = 1800
    hit_score: float = 0.0  # For cache replacement algorithms

    def is_expired(self) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() - self.timestamp > timedelta(seconds=self.ttl_seconds)

    def update_access(self) -> None:
        """Update access statistics."""
        self.access_count += 1
        self.hit_score = self.access_count / max(1, (datetime.now() - self.timestamp).total_seconds() / 3600)


@dataclass
class QueryPattern:
    """Recognized pattern for common queries."""

    pattern_type: str  # "entry_point", "api_search", "class_lookup", etc.
    regex_pattern: str
    weight: float = 1.0
    cache_ttl: int = 3600  # Patterns can be cached longer


@dataclass
class MemoryIndex:
    """Memory-based index for fast node lookups."""

    # Primary index: node_id -> NodeMetadata
    nodes: dict[str, NodeMetadata] = field(default_factory=dict)

    # Secondary indices for fast lookups
    by_name: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    by_type: dict[ChunkType, set[str]] = field(default_factory=lambda: defaultdict(set))
    by_file: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    by_breadcrumb: dict[str, str] = field(default_factory=dict)  # breadcrumb -> node_id
    by_language: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    # Relationship indices
    children_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    parent_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))
    dependency_index: dict[str, set[str]] = field(default_factory=lambda: defaultdict(set))

    # Performance metrics
    total_nodes: int = 0
    last_updated: datetime = field(default_factory=datetime.now)
    hit_count: int = 0
    miss_count: int = 0


class LightweightGraphService:
    """
    Lightweight Graph Service implementing memory indexing and on-demand graph building.

    This service addresses the performance limitations of the existing GraphRAGService
    by implementing:
    1. Memory indexing mechanism for fast metadata lookups
    2. On-demand partial graph construction
    3. Removal of max_chunks_for_mcp limitation
    4. Multi-layer caching strategy (L1-L3)
    5. Timeout handling with partial results
    6. Progressive result return with confidence scoring
    """

    def __init__(
        self,
        graph_rag_service: GraphRAGService,
        hybrid_search_service: HybridSearchService,
        cache_service: BaseCacheService | None = None,
        enable_timeout: bool = True,
        default_timeout: int = 15,
        enable_progressive_results: bool = True,
        confidence_threshold: float = 0.7,
    ):
        self.logger = logging.getLogger(__name__)
        self.graph_rag_service = graph_rag_service
        self.hybrid_search_service = hybrid_search_service
        self.cache_service = cache_service

        # Configuration
        self.enable_timeout = enable_timeout
        self.default_timeout = default_timeout
        self.enable_progressive_results = enable_progressive_results
        self.confidence_threshold = confidence_threshold

        # Memory index
        self.memory_index = MemoryIndex()

        # Enhanced pre-computed query cache with TTL and invalidation
        self.precomputed_queries = {
            "entry_points": {},  # main, __main__, index, app functions
            "main_functions": {},  # Top functions by importance
            "public_apis": {},  # Exported/public functions
            "common_patterns": {},  # Common code patterns
            "api_endpoints": {},  # Web API endpoints
            "data_models": {},  # Classes representing data structures
            "utility_functions": {},  # Helper/utility functions
            "test_functions": {},  # Test functions
            "configuration_points": {},  # Config-related functions
            "error_handlers": {},  # Error handling functions
        }

        # Query cache with TTL and metadata
        self.query_cache = {}  # cache_key -> {result, timestamp, access_count, ttl}
        self.query_cache_stats = {"hits": 0, "misses": 0, "evictions": 0, "total_queries": 0}

        # Query pattern recognition cache
        self.query_patterns = {}  # pattern -> cached_results

        # Cache configuration
        self.cache_ttl_seconds = 1800  # 30 minutes default TTL
        self.max_cache_size = 1000
        self.cache_warmup_enabled = True

        # Multi-layer cache configuration
        self.l1_cache = {}  # In-memory fast cache
        self.l2_cache = {}  # Path-based cache
        self.l3_cache = {}  # Query result cache

        # Performance tracking
        self.performance_metrics = {"total_queries": 0, "cache_hits": 0, "timeouts": 0, "partial_results": 0, "average_response_time": 0.0}

        self._lock = asyncio.Lock()

    async def initialize_memory_index(self, project_name: str, force_rebuild: bool = False) -> bool:
        """
        Initialize the memory index with key node metadata.

        Args:
            project_name: Name of the project to index
            force_rebuild: Whether to force rebuild the index

        Returns:
            bool: True if successful, False otherwise
        """
        try:
            start_time = time.time()
            self.logger.info(f"Initializing memory index for project: {project_name}")

            # Check if index already exists and is recent
            if not force_rebuild and self._is_index_fresh(project_name):
                self.logger.info("Using existing fresh memory index")
                return True

            # Get all chunks from hybrid search service (NO LIMITATIONS)
            chunks = await self.hybrid_search_service.get_all_chunks(project_name)
            if not chunks:
                self.logger.warning(f"No chunks found for project: {project_name}")
                return False

            self.logger.info(f"Building memory index from {len(chunks)} chunks (NO MCP LIMITATIONS)")

            # Reset memory index
            async with self._lock:
                self.memory_index = MemoryIndex()

                # Process all chunks to build metadata
                for chunk in chunks:
                    await self._add_chunk_to_index(chunk)

                # Build relationship indices
                await self._build_relationship_indices(chunks)

                # Pre-compute common queries
                await self._precompute_common_queries(project_name)

                self.memory_index.total_nodes = len(self.memory_index.nodes)
                self.memory_index.last_updated = datetime.now()

            elapsed_time = time.time() - start_time
            self.logger.info(f"Memory index initialized successfully: {self.memory_index.total_nodes} nodes in {elapsed_time:.2f}s")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize memory index: {e}")
            return False

    async def _add_chunk_to_index(self, chunk: CodeChunk) -> None:
        """Add a code chunk to the memory index."""

        node_metadata = NodeMetadata(
            node_id=chunk.chunk_id,
            name=chunk.name or "unnamed",
            chunk_type=chunk.chunk_type,
            file_path=chunk.file_path,
            breadcrumb=chunk.breadcrumb,
            parent_name=chunk.parent_name,
            signature=chunk.signature,
            language=chunk.language,
            line_start=chunk.start_line,
            line_end=chunk.end_line,
            importance_score=self._calculate_importance_score(chunk),
        )

        # Add to primary index
        self.memory_index.nodes[chunk.chunk_id] = node_metadata

        # Add to secondary indices
        self.memory_index.by_name[chunk.name or "unnamed"].add(chunk.chunk_id)
        self.memory_index.by_type[chunk.chunk_type].add(chunk.chunk_id)
        self.memory_index.by_file[chunk.file_path].add(chunk.chunk_id)
        self.memory_index.by_language[chunk.language].add(chunk.chunk_id)

        if chunk.breadcrumb:
            self.memory_index.by_breadcrumb[chunk.breadcrumb] = chunk.chunk_id

    async def _build_relationship_indices(self, chunks: list[CodeChunk]) -> None:
        """Build relationship indices for fast graph traversal."""

        for chunk in chunks:
            node_id = chunk.chunk_id

            # Build parent-child relationships based on breadcrumb hierarchy
            if chunk.breadcrumb and "." in chunk.breadcrumb:
                parent_breadcrumb = ".".join(chunk.breadcrumb.split(".")[:-1])
                if parent_breadcrumb in self.memory_index.by_breadcrumb:
                    parent_id = self.memory_index.by_breadcrumb[parent_breadcrumb]

                    # Update indices
                    self.memory_index.children_index[parent_id].add(node_id)
                    self.memory_index.parent_index[node_id].add(parent_id)

                    # Update node metadata
                    if node_id in self.memory_index.nodes:
                        self.memory_index.nodes[node_id].parent_ids.add(parent_id)
                    if parent_id in self.memory_index.nodes:
                        self.memory_index.nodes[parent_id].children_ids.add(node_id)

            # Build dependency relationships based on imports
            if hasattr(chunk, "imports_used") and chunk.imports_used:
                for import_name in chunk.imports_used:
                    # Find matching nodes by name
                    matching_nodes = self.memory_index.by_name.get(import_name, set())
                    for dep_id in matching_nodes:
                        self.memory_index.dependency_index[node_id].add(dep_id)
                        if node_id in self.memory_index.nodes:
                            self.memory_index.nodes[node_id].dependency_ids.add(dep_id)

    async def _precompute_common_queries(self, project_name: str) -> None:
        """Enhanced pre-compute common queries for fast retrieval with comprehensive categorization."""

        try:
            start_time = time.time()
            self.logger.info(f"Pre-computing enhanced queries for project: {project_name}")

            # Initialize query pattern recognition
            await self._initialize_query_patterns()

            # Entry points: main functions, __main__, etc.
            entry_points = await self._find_entry_points()
            self.precomputed_queries["entry_points"][project_name] = entry_points

            # Main functions: functions with high importance scores
            main_functions = await self._find_main_functions()
            self.precomputed_queries["main_functions"][project_name] = main_functions

            # Public APIs: exported functions, public methods
            public_apis = await self._find_public_apis()
            self.precomputed_queries["public_apis"][project_name] = public_apis

            # API endpoints: HTTP endpoints, routes
            api_endpoints = await self._find_api_endpoints()
            self.precomputed_queries["api_endpoints"][project_name] = api_endpoints

            # Data models: Classes representing data structures
            data_models = await self._find_data_models()
            self.precomputed_queries["data_models"][project_name] = data_models

            # Utility functions: Helper/utility functions
            utility_functions = await self._find_utility_functions()
            self.precomputed_queries["utility_functions"][project_name] = utility_functions

            # Test functions: Test-related functions
            test_functions = await self._find_test_functions()
            self.precomputed_queries["test_functions"][project_name] = test_functions

            # Configuration points: Config-related functions
            config_points = await self._find_configuration_points()
            self.precomputed_queries["configuration_points"][project_name] = config_points

            # Error handlers: Error handling functions
            error_handlers = await self._find_error_handlers()
            self.precomputed_queries["error_handlers"][project_name] = error_handlers

            # Common patterns: Identify architectural patterns
            common_patterns = await self._find_common_patterns()
            self.precomputed_queries["common_patterns"][project_name] = common_patterns

            # Cache the results with TTL
            await self._cache_precomputed_results(project_name)

            elapsed_time = time.time() - start_time
            self.logger.info(
                f"Enhanced pre-computed queries completed in {elapsed_time:.2f}s: "
                f"{len(entry_points)} entry points, {len(main_functions)} main functions, "
                f"{len(public_apis)} public APIs, {len(api_endpoints)} API endpoints, "
                f"{len(data_models)} data models, {len(utility_functions)} utilities, "
                f"{len(test_functions)} tests, {len(config_points)} config points, "
                f"{len(error_handlers)} error handlers, {len(common_patterns)} patterns"
            )

        except Exception as e:
            self.logger.error(f"Failed to pre-compute enhanced queries: {e}")

    async def _initialize_query_patterns(self) -> None:
        """Initialize query pattern recognition patterns."""

        patterns = [
            QueryPattern("entry_point", r"\b(main|__main__|index|app|start|run|init)\b", 1.0, 3600),
            QueryPattern("api_endpoint", r"\b(route|endpoint|api|handler|view)\b", 0.9, 3600),
            QueryPattern("data_model", r"\b(model|schema|entity|dto|data)\b", 0.8, 3600),
            QueryPattern("utility", r"\b(util|helper|tool|common|shared)\b", 0.7, 3600),
            QueryPattern("test", r"\b(test|spec|check|validate|mock)\b", 0.6, 3600),
            QueryPattern("config", r"\b(config|setting|option|parameter|env)\b", 0.7, 3600),
            QueryPattern("error", r"\b(error|exception|handle|catch|fail)\b", 0.8, 3600),
            QueryPattern("class_lookup", r"^[A-Z][a-zA-Z0-9]*$", 0.9, 3600),  # PascalCase
            QueryPattern("function_lookup", r"^[a-z][a-zA-Z0-9_]*$", 0.8, 3600),  # camelCase/snake_case
        ]

        for pattern in patterns:
            self.query_patterns[pattern.pattern_type] = pattern

    async def _find_entry_points(self) -> list[str]:
        """Find entry point functions."""

        entry_points = []
        entry_names = {"main", "__main__", "index", "app", "start", "run", "init", "begin", "execute"}

        for node_id, metadata in self.memory_index.nodes.items():
            # Direct name matches
            if metadata.name.lower() in entry_names:
                entry_points.append(node_id)
                continue

            # Functions with 'main' in name
            if metadata.chunk_type == ChunkType.FUNCTION and "main" in metadata.name.lower():
                entry_points.append(node_id)
                continue

            # Files named main, index, app
            if metadata.file_path:
                file_name = metadata.file_path.split("/")[-1].split(".")[0].lower()
                if file_name in entry_names and metadata.chunk_type == ChunkType.FUNCTION:
                    entry_points.append(node_id)

        return entry_points

    async def _find_main_functions(self) -> list[str]:
        """Find main functions by importance score."""

        functions = [
            (node_id, metadata)
            for node_id, metadata in self.memory_index.nodes.items()
            if metadata.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.ASYNC_FUNCTION]
        ]

        # Sort by importance score and select top functions
        sorted_functions = sorted(functions, key=lambda x: x[1].importance_score, reverse=True)
        return [node_id for node_id, _ in sorted_functions[:25]]  # Top 25

    async def _find_public_apis(self) -> list[str]:
        """Find public API functions."""

        public_apis = []

        for node_id, metadata in self.memory_index.nodes.items():
            if metadata.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.ASYNC_FUNCTION]:
                # Not private (doesn't start with _)
                if not metadata.name.startswith("_"):
                    # Additional checks for API indicators
                    is_api = (
                        metadata.importance_score > 0.5
                        or any(keyword in metadata.name.lower() for keyword in ["api", "endpoint", "handler", "route", "service"])
                        or (metadata.signature and "public" in metadata.signature.lower())
                    )

                    if is_api:
                        public_apis.append(node_id)

        return public_apis

    async def _find_api_endpoints(self) -> list[str]:
        """Find API endpoint functions."""

        api_endpoints = []
        api_keywords = {"route", "endpoint", "api", "handler", "view", "controller", "resource"}

        for node_id, metadata in self.memory_index.nodes.items():
            if metadata.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.ASYNC_FUNCTION]:
                # Check name for API keywords
                name_lower = metadata.name.lower()
                if any(keyword in name_lower for keyword in api_keywords):
                    api_endpoints.append(node_id)
                    continue

                # Check file path for API-related directories
                if metadata.file_path:
                    path_lower = metadata.file_path.lower()
                    if any(keyword in path_lower for keyword in ["api", "route", "handler", "controller", "endpoint"]):
                        api_endpoints.append(node_id)
                        continue

                # Check breadcrumb for API patterns
                if metadata.breadcrumb:
                    breadcrumb_lower = metadata.breadcrumb.lower()
                    if any(keyword in breadcrumb_lower for keyword in api_keywords):
                        api_endpoints.append(node_id)

        return api_endpoints

    async def _find_data_models(self) -> list[str]:
        """Find data model classes."""

        data_models = []
        model_keywords = {"model", "schema", "entity", "dto", "data", "struct", "record"}

        for node_id, metadata in self.memory_index.nodes.items():
            if metadata.chunk_type in [ChunkType.CLASS, ChunkType.INTERFACE]:
                # Check name for model keywords
                name_lower = metadata.name.lower()
                if any(keyword in name_lower for keyword in model_keywords):
                    data_models.append(node_id)
                    continue

                # Check for PascalCase naming (common for data models)
                if re.match(r"^[A-Z][a-zA-Z0-9]*$", metadata.name):
                    # Additional heuristic: importance score suggests it's significant
                    if metadata.importance_score > 0.6:
                        data_models.append(node_id)

        return data_models

    async def _find_utility_functions(self) -> list[str]:
        """Find utility/helper functions."""

        utilities = []
        util_keywords = {"util", "helper", "tool", "common", "shared", "support"}

        for node_id, metadata in self.memory_index.nodes.items():
            if metadata.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.ASYNC_FUNCTION]:
                # Check name for utility keywords
                name_lower = metadata.name.lower()
                if any(keyword in name_lower for keyword in util_keywords):
                    utilities.append(node_id)
                    continue

                # Check file path for utility directories
                if metadata.file_path:
                    path_lower = metadata.file_path.lower()
                    if any(keyword in path_lower for keyword in ["util", "helper", "tool", "common", "shared", "lib"]):
                        utilities.append(node_id)

        return utilities

    async def _find_test_functions(self) -> list[str]:
        """Find test functions."""

        test_functions = []
        test_keywords = {"test", "spec", "check", "validate", "mock", "assert", "should"}

        for node_id, metadata in self.memory_index.nodes.items():
            if metadata.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.ASYNC_FUNCTION]:
                # Check name for test keywords
                name_lower = metadata.name.lower()
                if any(keyword in name_lower for keyword in test_keywords):
                    test_functions.append(node_id)
                    continue

                # Check file path for test directories
                if metadata.file_path:
                    path_lower = metadata.file_path.lower()
                    if any(keyword in path_lower for keyword in ["test", "spec", "__test__", "tests"]):
                        test_functions.append(node_id)

        return test_functions

    async def _find_configuration_points(self) -> list[str]:
        """Find configuration-related functions."""

        config_points = []
        config_keywords = {"config", "setting", "option", "parameter", "env", "setup", "init"}

        for node_id, metadata in self.memory_index.nodes.items():
            if metadata.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.CLASS]:
                # Check name for config keywords
                name_lower = metadata.name.lower()
                if any(keyword in name_lower for keyword in config_keywords):
                    config_points.append(node_id)
                    continue

                # Check file path for config files
                if metadata.file_path:
                    path_lower = metadata.file_path.lower()
                    if any(keyword in path_lower for keyword in ["config", "setting", "env", "setup"]):
                        config_points.append(node_id)

        return config_points

    async def _find_error_handlers(self) -> list[str]:
        """Find error handling functions."""

        error_handlers = []
        error_keywords = {"error", "exception", "handle", "catch", "fail", "rescue", "recover"}

        for node_id, metadata in self.memory_index.nodes.items():
            if metadata.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD, ChunkType.ASYNC_FUNCTION]:
                # Check name for error keywords
                name_lower = metadata.name.lower()
                if any(keyword in name_lower for keyword in error_keywords):
                    error_handlers.append(node_id)
                    continue

                # Check signature for exception handling
                if metadata.signature:
                    sig_lower = metadata.signature.lower()
                    if any(keyword in sig_lower for keyword in ["exception", "error", "catch", "raise"]):
                        error_handlers.append(node_id)

        return error_handlers

    async def _find_common_patterns(self) -> dict[str, list[str]]:
        """Find common architectural patterns."""

        patterns = {"singleton": [], "factory": [], "builder": [], "observer": [], "decorator": [], "adapter": [], "strategy": []}

        pattern_keywords = {
            "singleton": ["singleton"],
            "factory": ["factory", "create", "make", "build"],
            "builder": ["builder", "construct"],
            "observer": ["observer", "listener", "subscriber", "notify"],
            "decorator": ["decorator", "decorate", "wrap"],
            "adapter": ["adapter", "adapt", "convert"],
            "strategy": ["strategy", "algorithm", "policy"],
        }

        for node_id, metadata in self.memory_index.nodes.items():
            if metadata.chunk_type in [ChunkType.CLASS, ChunkType.FUNCTION, ChunkType.METHOD]:
                name_lower = metadata.name.lower()

                for pattern_type, keywords in pattern_keywords.items():
                    if any(keyword in name_lower for keyword in keywords):
                        patterns[pattern_type].append(node_id)

        return patterns

    async def _cache_precomputed_results(self, project_name: str) -> None:
        """Cache pre-computed results with TTL."""

        timestamp = datetime.now()

        for query_type, results in self.precomputed_queries.items():
            if project_name in results:
                cache_key = f"precomputed_{query_type}_{project_name}"
                cache_entry = QueryCacheEntry(result=results[project_name], timestamp=timestamp, ttl_seconds=self.cache_ttl_seconds)
                self.query_cache[cache_key] = cache_entry

    def _calculate_importance_score(self, chunk: CodeChunk) -> float:
        """Calculate importance score for a code chunk."""

        score = 0.0

        # Base score by chunk type
        type_scores = {
            ChunkType.FUNCTION: 1.0,
            ChunkType.METHOD: 0.9,
            ChunkType.CLASS: 1.2,
            ChunkType.INTERFACE: 1.1,
            ChunkType.CONSTRUCTOR: 0.8,
            ChunkType.ASYNC_FUNCTION: 1.1,
        }
        score += type_scores.get(chunk.chunk_type, 0.5)

        # Name-based scoring
        if chunk.name:
            if chunk.name in ["main", "__main__", "index", "app"]:
                score += 1.0
            elif chunk.name.startswith("_"):
                score -= 0.3  # Private functions less important
            elif chunk.name.isupper():
                score += 0.2  # Constants are moderately important

        # Size-based scoring (larger functions might be more important)
        lines = chunk.end_line - chunk.start_line + 1
        if lines > 50:
            score += 0.3
        elif lines > 20:
            score += 0.1

        # Docstring presence adds importance
        if chunk.docstring:
            score += 0.2

        return min(score, 3.0)  # Cap at 3.0

    def _is_index_fresh(self, project_name: str, max_age_minutes: int = 30) -> bool:
        """Check if the memory index is fresh enough to use."""

        if not self.memory_index.nodes:
            return False

        age = datetime.now() - self.memory_index.last_updated
        return age < timedelta(minutes=max_age_minutes)

    async def build_partial_graph(
        self, project_name: str, query_scope: str, options: GraphBuildOptions | None = None
    ) -> StructureGraph | None:
        """
        Build partial graph for specific query scope with enhanced algorithms (Task 1.2).

        Args:
            project_name: Project to query
            query_scope: Specific scope (e.g., function name, class name, natural language query)
            options: Graph building options

        Returns:
            StructureGraph: Partial graph or None if failed
        """
        if options is None:
            options = GraphBuildOptions()

        try:
            start_time = time.time()
            self.logger.info(f"Building partial graph for scope: {query_scope} with strategy: {options.expansion_strategy.value}")

            # Analyze query complexity for adaptive processing
            query_complexity = await self._analyze_query_complexity(query_scope)

            # Adjust options based on query complexity
            options = await self._adapt_options_to_complexity(options, query_complexity)

            # Find and score target nodes matching the query scope
            scored_targets = await self._find_and_score_target_nodes(query_scope, options)
            if not scored_targets:
                self.logger.warning(f"No nodes found for scope: {query_scope}")
                return None

            # Initialize selected nodes with highest scoring targets
            budget_for_targets = min(options.max_nodes // 2, len(scored_targets))
            selected_nodes = {score.node_id for score in scored_targets[:budget_for_targets]}
            remaining_budget = options.max_nodes - len(selected_nodes)

            # Expand graph using selected strategy
            if remaining_budget > 0 and options.include_context:
                expanded_nodes = await self._expand_graph_with_strategy(selected_nodes, remaining_budget, query_scope, options)
                selected_nodes.update(expanded_nodes)

            # Ensure graph connectivity if preferred
            if options.prefer_connected_components:
                selected_nodes = await self._ensure_connectivity(selected_nodes, options)

            # Build the final graph structure
            graph = await self._build_enhanced_graph_structure(selected_nodes, project_name, query_scope)

            elapsed_time = time.time() - start_time
            self.logger.info(f"Enhanced partial graph built: {len(selected_nodes)} nodes in {elapsed_time:.2f}s")

            return graph

        except Exception as e:
            self.logger.error(f"Failed to build partial graph: {e}")
            return None

    async def _find_target_nodes(self, query_scope: str) -> list[str]:
        """Find nodes matching the query scope."""

        target_nodes = []

        # Search by name
        if query_scope in self.memory_index.by_name:
            target_nodes.extend(self.memory_index.by_name[query_scope])

        # Search by partial name match
        for name, node_ids in self.memory_index.by_name.items():
            if query_scope.lower() in name.lower():
                target_nodes.extend(node_ids)

        # Search by breadcrumb
        for breadcrumb, node_id in self.memory_index.by_breadcrumb.items():
            if query_scope.lower() in breadcrumb.lower():
                target_nodes.append(node_id)

        # Remove duplicates and sort by importance
        unique_nodes = list(set(target_nodes))
        return sorted(
            unique_nodes,
            key=lambda nid: self.memory_index.nodes.get(nid, NodeMetadata("", "", ChunkType.RAW_CODE, "")).importance_score,
            reverse=True,
        )

    async def _get_context_nodes(self, target_nodes: set[str], max_context: int) -> set[str]:
        """Get context nodes related to target nodes."""

        context_nodes = set()

        for node_id in target_nodes:
            if len(context_nodes) >= max_context:
                break

            metadata = self.memory_index.nodes.get(node_id)
            if not metadata:
                continue

            # Add parent nodes
            context_nodes.update(list(metadata.parent_ids)[:2])

            # Add child nodes (limited)
            context_nodes.update(list(metadata.children_ids)[:3])

            # Add dependency nodes (limited)
            context_nodes.update(list(metadata.dependency_ids)[:2])

        return context_nodes - target_nodes  # Exclude target nodes

    async def _build_graph_structure(self, node_ids: set[str], project_name: str) -> StructureGraph:
        """Build graph structure from selected node IDs."""

        nodes = {}
        edges = []

        # Create nodes
        for node_id in node_ids:
            metadata = self.memory_index.nodes.get(node_id)
            if metadata:
                node = GraphNode(
                    chunk_id=node_id,
                    breadcrumb=metadata.breadcrumb or metadata.name,
                    name=metadata.name,
                    chunk_type=metadata.chunk_type,
                    file_path=metadata.file_path,
                    depth=0,
                    semantic_weight=metadata.importance_score,
                )
                nodes[metadata.breadcrumb or node_id] = node

        # Create edges based on relationships
        for node_id in node_ids:
            metadata = self.memory_index.nodes.get(node_id)
            if not metadata:
                continue

            # Parent-child edges
            for child_id in metadata.children_ids:
                if child_id in node_ids:
                    child_metadata = self.memory_index.nodes.get(child_id)
                    if child_metadata:
                        edges.append(
                            GraphEdge(
                                source_breadcrumb=metadata.breadcrumb or metadata.name,
                                target_breadcrumb=child_metadata.breadcrumb or child_metadata.name,
                                relationship_type="parent_child",
                            )
                        )

            # Dependency edges
            for dep_id in metadata.dependency_ids:
                if dep_id in node_ids:
                    dep_metadata = self.memory_index.nodes.get(dep_id)
                    if dep_metadata:
                        edges.append(
                            GraphEdge(
                                source_breadcrumb=metadata.breadcrumb or metadata.name,
                                target_breadcrumb=dep_metadata.breadcrumb or dep_metadata.name,
                                relationship_type="dependency",
                            )
                        )

        return StructureGraph(nodes=nodes, edges=edges, project_name=project_name)

    async def _analyze_query_complexity(self, query_scope: str) -> QueryComplexity:
        """Analyze query complexity for adaptive processing."""

        # Simple heuristics for query complexity
        query_length = len(query_scope.split())
        has_wildcards = "*" in query_scope or "?" in query_scope
        has_logical_ops = any(op in query_scope.lower() for op in ["and", "or", "not", "&", "|"])
        has_path_notation = "." in query_scope or "/" in query_scope

        complexity_score = 0
        complexity_score += min(query_length * 0.1, 1.0)  # Length contribution
        complexity_score += 0.3 if has_wildcards else 0
        complexity_score += 0.4 if has_logical_ops else 0
        complexity_score += 0.2 if has_path_notation else 0

        if complexity_score < 0.3:
            return QueryComplexity.SIMPLE
        elif complexity_score < 0.6:
            return QueryComplexity.MODERATE
        elif complexity_score < 0.9:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.VERY_COMPLEX

    async def _adapt_options_to_complexity(self, options: GraphBuildOptions, complexity: QueryComplexity) -> GraphBuildOptions:
        """Adapt graph building options based on query complexity."""

        adapted = GraphBuildOptions(
            max_nodes=options.max_nodes,
            expansion_strategy=options.expansion_strategy,
            include_context=options.include_context,
            context_depth=options.context_depth,
            importance_threshold=options.importance_threshold,
            relevance_threshold=options.relevance_threshold,
            max_expansion_rounds=options.max_expansion_rounds,
            prefer_connected_components=options.prefer_connected_components,
        )

        # Adjust based on complexity
        complexity_adjustments = {
            QueryComplexity.SIMPLE: {"max_nodes": min(adapted.max_nodes, 30), "context_depth": 1, "max_expansion_rounds": 3},
            QueryComplexity.MODERATE: {
                "max_nodes": adapted.max_nodes,
                "context_depth": adapted.context_depth,
                "max_expansion_rounds": adapted.max_expansion_rounds,
            },
            QueryComplexity.COMPLEX: {
                "max_nodes": min(adapted.max_nodes * 1.5, 100),
                "context_depth": adapted.context_depth + 1,
                "max_expansion_rounds": adapted.max_expansion_rounds + 2,
            },
            QueryComplexity.VERY_COMPLEX: {
                "max_nodes": min(adapted.max_nodes * 2, 150),
                "context_depth": adapted.context_depth + 2,
                "max_expansion_rounds": adapted.max_expansion_rounds + 3,
            },
        }

        adjustments = complexity_adjustments[complexity]
        adapted.max_nodes = int(adjustments["max_nodes"])
        adapted.context_depth = adjustments["context_depth"]
        adapted.max_expansion_rounds = adjustments["max_expansion_rounds"]

        return adapted

    async def _find_and_score_target_nodes(self, query_scope: str, options: GraphBuildOptions) -> list[NodeRelevanceScore]:
        """Find and score target nodes with relevance scoring."""

        target_scores = []

        # Exact name matches (highest priority)
        exact_matches = self.memory_index.by_name.get(query_scope, set())
        for node_id in exact_matches:
            score = NodeRelevanceScore(node_id=node_id)
            score.semantic_relevance = 1.0
            score.structural_relevance = 0.8
            metadata = self.memory_index.nodes.get(node_id)
            if metadata:
                score.importance_bonus = metadata.importance_score / 3.0
            score.calculate_total()
            target_scores.append(score)

        # Breadcrumb exact matches
        if query_scope in self.memory_index.by_breadcrumb:
            node_id = self.memory_index.by_breadcrumb[query_scope]
            if node_id not in [s.node_id for s in target_scores]:
                score = NodeRelevanceScore(node_id=node_id)
                score.semantic_relevance = 1.0
                score.structural_relevance = 1.0
                metadata = self.memory_index.nodes.get(node_id)
                if metadata:
                    score.importance_bonus = metadata.importance_score / 3.0
                score.calculate_total()
                target_scores.append(score)

        # Partial name matches
        query_lower = query_scope.lower()
        for name, node_ids in self.memory_index.by_name.items():
            if query_lower in name.lower() and query_lower != name.lower():
                for node_id in node_ids:
                    if node_id not in [s.node_id for s in target_scores]:
                        score = NodeRelevanceScore(node_id=node_id)
                        # Calculate semantic similarity based on string overlap
                        overlap_ratio = len(query_lower) / len(name.lower())
                        score.semantic_relevance = min(overlap_ratio * 0.8, 0.8)
                        score.structural_relevance = 0.6
                        metadata = self.memory_index.nodes.get(node_id)
                        if metadata:
                            score.importance_bonus = metadata.importance_score / 3.0
                        score.calculate_total()
                        target_scores.append(score)

        # Breadcrumb partial matches
        for breadcrumb, node_id in self.memory_index.by_breadcrumb.items():
            if query_lower in breadcrumb.lower() and node_id not in [s.node_id for s in target_scores]:
                score = NodeRelevanceScore(node_id=node_id)
                overlap_ratio = len(query_lower) / len(breadcrumb.lower())
                score.semantic_relevance = min(overlap_ratio * 0.7, 0.7)
                score.structural_relevance = 0.8
                metadata = self.memory_index.nodes.get(node_id)
                if metadata:
                    score.importance_bonus = metadata.importance_score / 3.0
                score.calculate_total()
                target_scores.append(score)

        # Filter by relevance threshold and sort by total score
        filtered_scores = [s for s in target_scores if s.total_score >= options.relevance_threshold]
        return sorted(filtered_scores, key=lambda x: x.total_score, reverse=True)

    async def _expand_graph_with_strategy(
        self, selected_nodes: set[str], budget: int, query_scope: str, options: GraphBuildOptions
    ) -> set[str]:
        """Expand graph using the specified strategy."""

        if options.expansion_strategy == GraphExpansionStrategy.BREADTH_FIRST:
            return await self._expand_breadth_first(selected_nodes, budget, options)
        elif options.expansion_strategy == GraphExpansionStrategy.DEPTH_FIRST:
            return await self._expand_depth_first(selected_nodes, budget, options)
        elif options.expansion_strategy == GraphExpansionStrategy.IMPORTANCE_BASED:
            return await self._expand_importance_based(selected_nodes, budget, options)
        elif options.expansion_strategy == GraphExpansionStrategy.RELEVANCE_SCORED:
            return await self._expand_relevance_scored(selected_nodes, budget, query_scope, options)
        else:  # ADAPTIVE
            return await self._expand_adaptive(selected_nodes, budget, query_scope, options)

    async def _expand_breadth_first(self, selected_nodes: set[str], budget: int, options: GraphBuildOptions) -> set[str]:
        """Expand graph using breadth-first strategy."""

        expanded = set()
        queue = deque(selected_nodes)
        visited = set(selected_nodes)
        current_depth = 0

        while queue and len(expanded) < budget and current_depth < options.context_depth:
            level_size = len(queue)

            for _ in range(level_size):
                if not queue or len(expanded) >= budget:
                    break

                current_node = queue.popleft()
                neighbors = await self._get_node_neighbors(current_node)

                for neighbor in neighbors:
                    if neighbor not in visited and len(expanded) < budget:
                        metadata = self.memory_index.nodes.get(neighbor)
                        if metadata and metadata.importance_score >= options.importance_threshold:
                            expanded.add(neighbor)
                            visited.add(neighbor)
                            queue.append(neighbor)

            current_depth += 1

        return expanded

    async def _expand_depth_first(self, selected_nodes: set[str], budget: int, options: GraphBuildOptions) -> set[str]:
        """Expand graph using depth-first strategy."""

        expanded = set()

        for start_node in selected_nodes:
            if len(expanded) >= budget:
                break
            await self._dfs_expand(start_node, expanded, {start_node}, budget, 0, options)

        return expanded

    async def _dfs_expand(
        self, node_id: str, expanded: set[str], visited: set[str], budget: int, depth: int, options: GraphBuildOptions
    ) -> None:
        """Recursive DFS expansion helper."""

        if len(expanded) >= budget or depth >= options.context_depth:
            return

        neighbors = await self._get_node_neighbors(node_id)

        for neighbor in neighbors:
            if neighbor not in visited and len(expanded) < budget:
                metadata = self.memory_index.nodes.get(neighbor)
                if metadata and metadata.importance_score >= options.importance_threshold:
                    expanded.add(neighbor)
                    visited.add(neighbor)
                    await self._dfs_expand(neighbor, expanded, visited, budget, depth + 1, options)

    async def _expand_importance_based(self, selected_nodes: set[str], budget: int, options: GraphBuildOptions) -> set[str]:
        """Expand graph based on node importance scores."""

        candidates = set()

        # Collect all neighboring nodes
        for node_id in selected_nodes:
            neighbors = await self._get_node_neighbors(node_id)
            candidates.update(neighbors)

        # Remove already selected nodes
        candidates -= selected_nodes

        # Score candidates by importance and select top ones
        scored_candidates = []
        for candidate in candidates:
            metadata = self.memory_index.nodes.get(candidate)
            if metadata and metadata.importance_score >= options.importance_threshold:
                scored_candidates.append((candidate, metadata.importance_score))

        # Sort by importance and take top budget candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return {candidate for candidate, _ in scored_candidates[:budget]}

    async def _expand_relevance_scored(
        self, selected_nodes: set[str], budget: int, query_scope: str, options: GraphBuildOptions
    ) -> set[str]:
        """Expand graph based on relevance to query scope."""

        candidates = set()

        # Collect neighboring nodes
        for node_id in selected_nodes:
            neighbors = await self._get_node_neighbors(node_id)
            candidates.update(neighbors)

        candidates -= selected_nodes

        # Score candidates by relevance to query
        scored_candidates = []
        query_lower = query_scope.lower()

        for candidate in candidates:
            metadata = self.memory_index.nodes.get(candidate)
            if not metadata or metadata.importance_score < options.importance_threshold:
                continue

            relevance_score = 0.0

            # Name similarity
            if query_lower in metadata.name.lower():
                relevance_score += 0.4

            # Breadcrumb similarity
            if metadata.breadcrumb and query_lower in metadata.breadcrumb.lower():
                relevance_score += 0.3

            # Importance bonus
            relevance_score += metadata.importance_score * 0.3

            if relevance_score >= options.relevance_threshold:
                scored_candidates.append((candidate, relevance_score))

        # Sort by relevance and take top candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return {candidate for candidate, _ in scored_candidates[:budget]}

    async def _expand_adaptive(self, selected_nodes: set[str], budget: int, query_scope: str, options: GraphBuildOptions) -> set[str]:
        """Adaptive expansion combining multiple strategies."""

        # Use 50% budget for importance-based expansion
        importance_budget = budget // 2
        importance_nodes = await self._expand_importance_based(selected_nodes, importance_budget, options)

        # Use remaining budget for relevance-scored expansion
        remaining_budget = budget - len(importance_nodes)
        if remaining_budget > 0:
            relevance_nodes = await self._expand_relevance_scored(selected_nodes | importance_nodes, remaining_budget, query_scope, options)
            return importance_nodes | relevance_nodes

        return importance_nodes

    async def _get_node_neighbors(self, node_id: str) -> set[str]:
        """Get all neighboring nodes for expansion."""

        neighbors = set()
        metadata = self.memory_index.nodes.get(node_id)

        if metadata:
            neighbors.update(metadata.children_ids)
            neighbors.update(metadata.parent_ids)
            neighbors.update(metadata.dependency_ids)

        return neighbors

    async def _ensure_connectivity(self, selected_nodes: set[str], options: GraphBuildOptions) -> set[str]:
        """Ensure graph connectivity by adding bridging nodes if needed."""

        # Find connected components
        components = await self._find_connected_components(selected_nodes)

        if len(components) <= 1:
            return selected_nodes  # Already connected

        # Find minimal bridges to connect components
        bridging_nodes = await self._find_bridging_nodes(components, options)

        return selected_nodes | bridging_nodes

    async def _find_connected_components(self, nodes: set[str]) -> list[set[str]]:
        """Find connected components in the selected nodes."""

        components = []
        unvisited = set(nodes)

        while unvisited:
            # Start new component with arbitrary unvisited node
            start_node = next(iter(unvisited))
            component = set()
            queue = deque([start_node])

            while queue:
                current = queue.popleft()
                if current in unvisited:
                    unvisited.remove(current)
                    component.add(current)

                    # Add connected neighbors
                    neighbors = await self._get_node_neighbors(current)
                    for neighbor in neighbors:
                        if neighbor in nodes and neighbor in unvisited:
                            queue.append(neighbor)

            if component:
                components.append(component)

        return components

    async def _find_bridging_nodes(self, components: list[set[str]], options: GraphBuildOptions) -> set[str]:
        """Find minimal nodes to bridge disconnected components."""

        bridging_nodes = set()

        if len(components) < 2:
            return bridging_nodes

        # Try to connect each component to the largest one
        largest_component = max(components, key=len)
        other_components = [c for c in components if c != largest_component]

        for component in other_components:
            # Find best bridge between this component and largest component
            best_bridge = await self._find_best_bridge(component, largest_component, options)
            if best_bridge:
                bridging_nodes.update(best_bridge)

        return bridging_nodes

    async def _find_best_bridge(self, component1: set[str], component2: set[str], options: GraphBuildOptions) -> set[str]:
        """Find best bridging path between two components."""

        # Find shortest path between any nodes in the two components
        min_path_length = float("inf")
        best_path = []

        for node1 in list(component1)[:5]:  # Limit search for performance
            for node2 in list(component2)[:5]:
                path = await self._bfs_path_search(node1, node2, options.context_depth + 2)
                if path and len(path) < min_path_length:
                    min_path_length = len(path)
                    best_path = path

        # Return intermediate nodes (excluding endpoints which are already in components)
        if len(best_path) > 2:
            return set(best_path[1:-1])  # Exclude first and last

        return set()

    async def _build_enhanced_graph_structure(self, node_ids: set[str], project_name: str, query_scope: str) -> StructureGraph:
        """Build enhanced graph structure with query context."""

        nodes = {}
        edges = []

        # Create nodes with enhanced metadata
        for node_id in node_ids:
            metadata = self.memory_index.nodes.get(node_id)
            if metadata:
                # Calculate query relevance for this node
                query_relevance = self._calculate_query_relevance(metadata, query_scope)

                node = GraphNode(
                    chunk_id=node_id,
                    breadcrumb=metadata.breadcrumb or metadata.name,
                    name=metadata.name,
                    chunk_type=metadata.chunk_type,
                    file_path=metadata.file_path,
                    depth=0,
                    semantic_weight=metadata.importance_score,
                )

                # Add query relevance as additional metadata
                node.query_relevance = query_relevance
                nodes[metadata.breadcrumb or node_id] = node

        # Create edges with enhanced relationship types
        edge_types_added = set()

        for node_id in node_ids:
            metadata = self.memory_index.nodes.get(node_id)
            if not metadata:
                continue

            # Parent-child edges
            for child_id in metadata.children_ids:
                if child_id in node_ids:
                    child_metadata = self.memory_index.nodes.get(child_id)
                    if child_metadata:
                        edge_key = (metadata.breadcrumb or metadata.name, child_metadata.breadcrumb or child_metadata.name, "parent_child")
                        if edge_key not in edge_types_added:
                            edges.append(
                                GraphEdge(
                                    source_breadcrumb=metadata.breadcrumb or metadata.name,
                                    target_breadcrumb=child_metadata.breadcrumb or child_metadata.name,
                                    relationship_type="parent_child",
                                )
                            )
                            edge_types_added.add(edge_key)

            # Dependency edges
            for dep_id in metadata.dependency_ids:
                if dep_id in node_ids:
                    dep_metadata = self.memory_index.nodes.get(dep_id)
                    if dep_metadata:
                        edge_key = (metadata.breadcrumb or metadata.name, dep_metadata.breadcrumb or dep_metadata.name, "dependency")
                        if edge_key not in edge_types_added:
                            edges.append(
                                GraphEdge(
                                    source_breadcrumb=metadata.breadcrumb or metadata.name,
                                    target_breadcrumb=dep_metadata.breadcrumb or dep_metadata.name,
                                    relationship_type="dependency",
                                )
                            )
                            edge_types_added.add(edge_key)

        graph = StructureGraph(nodes=nodes, edges=edges, project_name=project_name)
        graph.query_scope = query_scope  # Add query context to graph
        return graph

    def _calculate_query_relevance(self, metadata: NodeMetadata, query_scope: str) -> float:
        """Calculate how relevant a node is to the query scope."""

        relevance = 0.0
        query_lower = query_scope.lower()

        # Exact name match
        if metadata.name.lower() == query_lower:
            relevance += 1.0
        elif query_lower in metadata.name.lower():
            relevance += 0.6

        # Breadcrumb match
        if metadata.breadcrumb:
            if query_lower in metadata.breadcrumb.lower():
                relevance += 0.4

        # Importance bonus
        relevance += metadata.importance_score * 0.2

        return min(relevance, 1.0)

    async def get_precomputed_query(self, project_name: str, query_type: str) -> list[str]:
        """Enhanced get pre-computed query results with caching and pattern recognition (Task 1.3)."""

        try:
            self.query_cache_stats["total_queries"] += 1

            # Check cache first
            cache_key = f"precomputed_{query_type}_{project_name}"
            if cache_key in self.query_cache:
                cache_entry = self.query_cache[cache_key]
                if not cache_entry.is_expired():
                    cache_entry.update_access()
                    self.query_cache_stats["hits"] += 1
                    return cache_entry.result
                else:
                    # Remove expired entry
                    del self.query_cache[cache_key]
                    self.query_cache_stats["evictions"] += 1

            self.query_cache_stats["misses"] += 1

            # Check if query needs recomputation
            if query_type in self.precomputed_queries and project_name in self.precomputed_queries[query_type]:
                result = self.precomputed_queries[query_type][project_name]

                # Cache the result
                cache_entry = QueryCacheEntry(result=result, timestamp=datetime.now(), ttl_seconds=self.cache_ttl_seconds)
                self.query_cache[cache_key] = cache_entry

                return result

            # Fallback: trigger recomputation if not found
            await self._precompute_common_queries(project_name)
            return self.precomputed_queries.get(query_type, {}).get(project_name, [])

        except Exception as e:
            self.logger.error(f"Failed to get precomputed query {query_type}: {e}")
            return []

    async def query_with_pattern_recognition(self, project_name: str, query: str) -> dict[str, Any]:
        """
        Enhanced query with pattern recognition for common query types.

        Args:
            project_name: Project to query
            query: Natural language query

        Returns:
            Dict containing results and pattern information
        """

        try:
            self.query_cache_stats["total_queries"] += 1

            # Check pattern cache first
            pattern_cache_key = f"pattern_{hash(query)}_{project_name}"
            if pattern_cache_key in self.query_cache:
                cache_entry = self.query_cache[pattern_cache_key]
                if not cache_entry.is_expired():
                    cache_entry.update_access()
                    self.query_cache_stats["hits"] += 1
                    return cache_entry.result

            self.query_cache_stats["misses"] += 1

            # Recognize query patterns
            recognized_patterns = await self._recognize_query_patterns(query)

            results = {"query": query, "patterns": recognized_patterns, "results": {}, "confidence": 0.0}

            # Process each recognized pattern
            for pattern_info in recognized_patterns:
                pattern_type = pattern_info["type"]
                confidence = pattern_info["confidence"]

                if pattern_type in self.precomputed_queries:
                    pattern_results = await self.get_precomputed_query(project_name, pattern_type)
                    results["results"][pattern_type] = {"nodes": pattern_results, "confidence": confidence, "count": len(pattern_results)}

            # Calculate overall confidence
            if recognized_patterns:
                results["confidence"] = sum(p["confidence"] for p in recognized_patterns) / len(recognized_patterns)

            # Cache the pattern recognition result
            cache_entry = QueryCacheEntry(result=results, timestamp=datetime.now(), ttl_seconds=self.cache_ttl_seconds)
            self.query_cache[pattern_cache_key] = cache_entry

            return results

        except Exception as e:
            self.logger.error(f"Pattern recognition query failed: {e}")
            return {"query": query, "patterns": [], "results": {}, "confidence": 0.0}

    async def _recognize_query_patterns(self, query: str) -> list[dict[str, Any]]:
        """Recognize patterns in natural language queries."""

        recognized = []
        query_lower = query.lower()

        for pattern_type, pattern in self.query_patterns.items():
            # Check regex match
            if re.search(pattern.regex_pattern, query_lower, re.IGNORECASE):
                confidence = pattern.weight

                # Boost confidence for exact matches
                if pattern_type == "entry_point" and any(word in query_lower for word in ["main", "entry", "start"]):
                    confidence = min(confidence + 0.2, 1.0)

                recognized.append({"type": pattern_type, "confidence": confidence, "pattern": pattern.regex_pattern})

        # Sort by confidence
        return sorted(recognized, key=lambda x: x["confidence"], reverse=True)

    async def get_query_suggestions(self, project_name: str, partial_query: str) -> list[dict[str, Any]]:
        """Get query suggestions based on pre-computed results."""

        suggestions = []
        partial_lower = partial_query.lower()

        # Search through pre-computed query types
        for query_type, projects in self.precomputed_queries.items():
            if project_name in projects and projects[project_name]:
                # Calculate relevance to partial query
                type_relevance = 0.0

                # Check if query type matches partial query
                if any(word in query_type for word in partial_lower.split()):
                    type_relevance += 0.8

                # Check pattern matches
                if query_type in self.query_patterns:
                    pattern = self.query_patterns[query_type]
                    if re.search(pattern.regex_pattern, partial_lower, re.IGNORECASE):
                        type_relevance += pattern.weight * 0.5

                if type_relevance > 0.3:
                    suggestions.append(
                        {
                            "query_type": query_type,
                            "relevance": type_relevance,
                            "count": len(projects[project_name]),
                            "suggestion": f"Find {query_type.replace('_', ' ')} in {project_name}",
                        }
                    )

        # Sort by relevance
        return sorted(suggestions, key=lambda x: x["relevance"], reverse=True)[:10]

    async def warm_query_cache(self, project_name: str) -> dict[str, Any]:
        """Warm up the query cache with common queries."""

        if not self.cache_warmup_enabled:
            return {"status": "disabled", "warmed": 0}

        try:
            start_time = time.time()
            warmed_count = 0

            # Pre-compute all query types
            await self._precompute_common_queries(project_name)

            # Warm up pattern-based queries
            common_queries = [
                "main function",
                "entry point",
                "api endpoints",
                "data models",
                "utility functions",
                "test functions",
                "error handlers",
            ]

            for query in common_queries:
                await self.query_with_pattern_recognition(project_name, query)
                warmed_count += 1

            elapsed_time = time.time() - start_time

            return {"status": "completed", "warmed": warmed_count, "time_seconds": elapsed_time, "cache_size": len(self.query_cache)}

        except Exception as e:
            self.logger.error(f"Cache warming failed: {e}")
            return {"status": "failed", "error": str(e), "warmed": 0}

    async def get_cache_statistics(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""

        # Calculate cache metrics
        total_entries = len(self.query_cache)
        expired_entries = sum(1 for entry in self.query_cache.values() if entry.is_expired())

        # Hit rate calculation
        total_requests = self.query_cache_stats["hits"] + self.query_cache_stats["misses"]
        hit_rate = self.query_cache_stats["hits"] / max(1, total_requests)

        return {
            "cache_stats": self.query_cache_stats,
            "hit_rate": hit_rate,
            "total_entries": total_entries,
            "expired_entries": expired_entries,
            "cache_utilization": min(total_entries / self.max_cache_size, 1.0),
            "precomputed_types": list(self.precomputed_queries.keys()),
            "pattern_types": list(self.query_patterns.keys()),
            "memory_index_stats": self.get_memory_index_stats(),
        }

    async def clear_expired_cache_entries(self) -> int:
        """Clear expired cache entries and return count cleared."""

        cleared_count = 0
        expired_keys = []

        for key, entry in self.query_cache.items():
            if entry.is_expired():
                expired_keys.append(key)

        for key in expired_keys:
            del self.query_cache[key]
            cleared_count += 1

        self.query_cache_stats["evictions"] += cleared_count
        return cleared_count

    async def invalidate_cache_for_project(self, project_name: str) -> int:
        """Invalidate all cache entries for a specific project."""

        invalidated_count = 0
        keys_to_remove = []

        for key in self.query_cache.keys():
            if project_name in key:
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.query_cache[key]
            invalidated_count += 1

        # Also clear precomputed queries for the project
        for query_type in self.precomputed_queries:
            if project_name in self.precomputed_queries[query_type]:
                del self.precomputed_queries[query_type][project_name]

        return invalidated_count

    async def find_intelligent_path(self, start_node: str, end_node: str, max_depth: int = 10) -> list[str] | None:
        """
        Find intelligent path between nodes using cache and index (Task 1.4).

        Args:
            start_node: Starting node ID or name
            end_node: Target node ID or name
            max_depth: Maximum search depth

        Returns:
            List[str]: Path as list of node IDs, or None if no path found
        """

        try:
            # Check L1 cache first
            cache_key = f"path_{start_node}_{end_node}"
            if cache_key in self.l1_cache:
                self.performance_metrics["cache_hits"] += 1
                return self.l1_cache[cache_key]

            # Resolve node names to IDs if needed
            start_id = await self._resolve_node_id(start_node)
            end_id = await self._resolve_node_id(end_node)

            if not start_id or not end_id:
                return None

            # BFS path finding using memory index
            path = await self._bfs_path_search(start_id, end_id, max_depth)

            # Cache result
            self.l1_cache[cache_key] = path

            return path

        except Exception as e:
            self.logger.error(f"Failed to find intelligent path: {e}")
            return None

    async def _resolve_node_id(self, identifier: str) -> str | None:
        """Resolve node name or breadcrumb to node ID."""

        # Check if it's already a node ID
        if identifier in self.memory_index.nodes:
            return identifier

        # Check breadcrumb index
        if identifier in self.memory_index.by_breadcrumb:
            return self.memory_index.by_breadcrumb[identifier]

        # Check name index (return first match)
        if identifier in self.memory_index.by_name:
            node_ids = self.memory_index.by_name[identifier]
            if node_ids:
                return next(iter(node_ids))

        return None

    async def _bfs_path_search(self, start_id: str, end_id: str, max_depth: int) -> list[str] | None:
        """BFS path search using memory index."""

        if start_id == end_id:
            return [start_id]

        queue = [(start_id, [start_id])]
        visited = {start_id}

        for _ in range(max_depth):
            if not queue:
                break

            current_id, path = queue.pop(0)

            if current_id == end_id:
                return path

            # Get neighbors from memory index
            neighbors = set()
            metadata = self.memory_index.nodes.get(current_id)
            if metadata:
                neighbors.update(metadata.children_ids)
                neighbors.update(metadata.parent_ids)
                neighbors.update(metadata.dependency_ids)

            for neighbor_id in neighbors:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    queue.append((neighbor_id, path + [neighbor_id]))

        return None

    def get_memory_index_stats(self) -> dict[str, Any]:
        """Get memory index statistics."""

        return {
            "total_nodes": self.memory_index.total_nodes,
            "nodes_by_type": {chunk_type.value: len(node_ids) for chunk_type, node_ids in self.memory_index.by_type.items()},
            "nodes_by_language": {lang: len(node_ids) for lang, node_ids in self.memory_index.by_language.items()},
            "hit_rate": (self.memory_index.hit_count / max(1, self.memory_index.hit_count + self.memory_index.miss_count)),
            "last_updated": self.memory_index.last_updated.isoformat(),
            "performance_metrics": self.performance_metrics,
        }
