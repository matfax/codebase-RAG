"""
Hybrid Search Service for Graph RAG enhancement.

This service implements advanced hybrid search algorithms that combine semantic similarity
with structural relationship filtering, providing more accurate and contextually relevant
search results for code understanding and exploration.

Built on top of Wave 2's Graph RAG infrastructure and extends the existing search capabilities.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional, Union

from ..models.code_chunk import ChunkType, CodeChunk
from .cross_project_search_service import CrossProjectMatch, CrossProjectSearchFilter
from .embedding_service import EmbeddingService
from .graph_rag_service import GraphRAGService
from .qdrant_service import QdrantService
from .structure_relationship_builder import GraphNode, StructureGraph


class HybridSearchStrategy(Enum):
    """Different hybrid search strategies."""

    SEMANTIC_FIRST = "semantic_first"  # Semantic similarity as primary, structural as secondary
    STRUCTURAL_FIRST = "structural_first"  # Structural relationships as primary, semantic as secondary
    BALANCED = "balanced"  # Equal weight to semantic and structural factors
    ADAPTIVE = "adaptive"  # Dynamically adjust weights based on query characteristics
    GRAPH_ENHANCED = "graph_enhanced"  # Use graph traversal to expand search scope


@dataclass
class HybridSearchParameters:
    """Parameters for controlling hybrid search behavior."""

    # Search strategy
    strategy: HybridSearchStrategy = HybridSearchStrategy.BALANCED

    # Weight distribution
    semantic_weight: float = 0.5  # Weight for semantic similarity (0.0-1.0)
    structural_weight: float = 0.3  # Weight for structural relationships (0.0-1.0)
    context_weight: float = 0.2  # Weight for contextual factors (0.0-1.0)

    # Similarity thresholds
    min_semantic_similarity: float = 0.1  # Minimum semantic similarity threshold (lowered for testing)
    min_structural_similarity: float = 0.4  # Minimum structural similarity threshold

    # Graph traversal parameters
    max_traversal_depth: int = 3  # Maximum depth for graph traversal
    expand_search_scope: bool = True  # Whether to use graph traversal to expand search

    # Result filtering
    max_results: int = 20  # Maximum number of results to return
    diversity_factor: float = 0.1  # Factor for promoting result diversity

    # Performance tuning
    enable_caching: bool = True  # Whether to use search result caching
    timeout_seconds: float = 30.0  # Search timeout

    def __post_init__(self):
        """Validate and normalize weights."""
        total_weight = self.semantic_weight + self.structural_weight + self.context_weight
        if total_weight > 0:
            # Normalize weights to sum to 1.0
            self.semantic_weight /= total_weight
            self.structural_weight /= total_weight
            self.context_weight /= total_weight


@dataclass
class HybridSearchResult:
    """Result from hybrid search with detailed scoring breakdown."""

    chunk: CodeChunk
    project_name: str

    # Detailed scoring
    semantic_score: float
    structural_score: float
    context_score: float
    final_score: float

    # Supporting information
    related_nodes: list[GraphNode] = None
    structural_path: list[str] = None  # Path through structure graph
    semantic_matches: list[str] = None  # Semantic match explanations
    context_factors: dict[str, float] = None  # Context scoring factors

    # Ranking information
    rank: int = 0
    confidence: float = 0.0  # Confidence in the result (0.0-1.0)

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.related_nodes is None:
            self.related_nodes = []
        if self.structural_path is None:
            self.structural_path = []
        if self.semantic_matches is None:
            self.semantic_matches = []
        if self.context_factors is None:
            self.context_factors = {}


@dataclass
class HybridSearchSummary:
    """Summary of hybrid search execution and results."""

    query: str
    strategy_used: HybridSearchStrategy
    results: list[HybridSearchResult]

    # Execution statistics
    total_candidates_examined: int
    semantic_search_time_ms: float
    structural_analysis_time_ms: float
    total_execution_time_ms: float

    # Search effectiveness metrics
    avg_semantic_score: float = 0.0
    avg_structural_score: float = 0.0
    avg_context_score: float = 0.0
    score_distribution: dict[str, int] = None  # Score range distribution

    # Strategy effectiveness
    weight_adjustments_made: dict[str, float] = None  # If adaptive strategy
    graph_expansion_used: bool = False
    cache_hit_rate: float = 0.0

    def __post_init__(self):
        """Initialize default values and compute statistics."""
        if self.score_distribution is None:
            self.score_distribution = {}
        if self.weight_adjustments_made is None:
            self.weight_adjustments_made = {}

        # Compute statistics
        if self.results:
            self.avg_semantic_score = sum(r.semantic_score for r in self.results) / len(self.results)
            self.avg_structural_score = sum(r.structural_score for r in self.results) / len(self.results)
            self.avg_context_score = sum(r.context_score for r in self.results) / len(self.results)

            # Compute score distribution
            for result in self.results:
                score_range = f"{int(result.final_score * 10) * 10}-{int(result.final_score * 10) * 10 + 9}%"
                self.score_distribution[score_range] = self.score_distribution.get(score_range, 0) + 1


class HybridSearchService:
    """
    Service for performing hybrid searches that combine semantic similarity with
    structural relationship analysis using Graph RAG capabilities.

    This service enhances traditional semantic search with structural intelligence,
    providing more accurate and contextually relevant results for code exploration.
    """

    def __init__(
        self,
        qdrant_service: QdrantService,
        embedding_service: EmbeddingService,
        graph_rag_service: GraphRAGService,
    ):
        """Initialize the hybrid search service.

        Args:
            qdrant_service: Service for vector database operations
            embedding_service: Service for generating embeddings
            graph_rag_service: Service for graph operations and structural analysis
        """
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service
        self.graph_rag_service = graph_rag_service
        self.logger = logging.getLogger(__name__)

        # Cache for search results (in-memory for now)
        self._search_cache = {}
        self._cache_stats = {"hits": 0, "misses": 0}

    async def hybrid_search(
        self,
        query: str,
        project_names: list[str],
        search_params: HybridSearchParameters,
        filters: CrossProjectSearchFilter | None = None,
    ) -> HybridSearchSummary:
        """
        Perform hybrid search across specified projects.

        Args:
            query: Natural language search query
            project_names: List of project names to search in
            search_params: Parameters controlling search behavior
            filters: Optional filters for search scope

        Returns:
            HybridSearchSummary with results and execution statistics
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting hybrid search: '{query}' across {len(project_names)} projects")

            # Step 1: Check cache if enabled
            cache_key = self._generate_cache_key(query, project_names, search_params, filters)
            if search_params.enable_caching and cache_key in self._search_cache:
                self._cache_stats["hits"] += 1
                self.logger.info("Returning cached search results")
                cached_result = self._search_cache[cache_key]
                cached_result.cache_hit_rate = self._calculate_cache_hit_rate()
                return cached_result

            self._cache_stats["misses"] += 1

            # Step 2: Adaptive strategy adjustment if needed
            adjusted_params = await self._adapt_search_strategy(query, search_params)

            # Step 3: Perform semantic search
            semantic_start = time.time()
            semantic_candidates = await self._perform_semantic_search(query, project_names, adjusted_params, filters)
            semantic_time_ms = (time.time() - semantic_start) * 1000

            # Step 4: Enhance with structural analysis
            structural_start = time.time()
            enhanced_candidates = await self._enhance_with_structural_analysis(semantic_candidates, query, adjusted_params, project_names)
            structural_time_ms = (time.time() - structural_start) * 1000

            # Step 5: Calculate final scores and rank results
            final_results = await self._calculate_final_scores_and_rank(enhanced_candidates, adjusted_params)

            # Step 6: Apply diversity filtering if needed
            if adjusted_params.diversity_factor > 0:
                final_results = self._apply_diversity_filtering(final_results, adjusted_params)

            total_time_ms = (time.time() - start_time) * 1000

            # Step 7: Create summary
            summary = HybridSearchSummary(
                query=query,
                strategy_used=adjusted_params.strategy,
                results=final_results[: adjusted_params.max_results],
                total_candidates_examined=len(semantic_candidates),
                semantic_search_time_ms=semantic_time_ms,
                structural_analysis_time_ms=structural_time_ms,
                total_execution_time_ms=total_time_ms,
                graph_expansion_used=adjusted_params.expand_search_scope,
                cache_hit_rate=self._calculate_cache_hit_rate(),
            )

            # Step 8: Cache result if enabled
            if search_params.enable_caching:
                self._search_cache[cache_key] = summary
                # Simple cache size management (keep last 100 searches)
                if len(self._search_cache) > 100:
                    oldest_key = next(iter(self._search_cache))
                    del self._search_cache[oldest_key]

            self.logger.info(
                f"Hybrid search completed in {total_time_ms:.2f}ms. "
                f"Found {len(final_results)} results from {len(semantic_candidates)} candidates."
            )

            return summary

        except Exception as e:
            self.logger.error(f"Error in hybrid search: {e}")
            # Return empty result with error information
            total_time_ms = (time.time() - start_time) * 1000
            return HybridSearchSummary(
                query=query,
                strategy_used=search_params.strategy,
                results=[],
                total_candidates_examined=0,
                semantic_search_time_ms=0.0,
                structural_analysis_time_ms=0.0,
                total_execution_time_ms=total_time_ms,
            )

    async def search_with_graph_expansion(
        self,
        query: str,
        seed_breadcrumbs: list[str],
        project_names: list[str],
        search_params: HybridSearchParameters,
    ) -> HybridSearchSummary:
        """
        Perform search starting from specific breadcrumbs and expanding through graph.

        Args:
            query: Natural language search query
            seed_breadcrumbs: Starting points for graph expansion
            project_names: List of project names to search in
            search_params: Parameters controlling search behavior

        Returns:
            HybridSearchSummary with expanded search results
        """
        self.logger.info(f"Starting graph-expanded search from {len(seed_breadcrumbs)} seed points")

        # Step 1: Expand search scope using graph traversal
        expanded_breadcrumbs = []
        for project_name in project_names:
            for seed_breadcrumb in seed_breadcrumbs:
                try:
                    traversal_result = await self.graph_rag_service.traverse_from_breadcrumb(
                        breadcrumb=seed_breadcrumb,
                        project_name=project_name,
                        max_depth=search_params.max_traversal_depth,
                        strategy_name="semantic",
                    )

                    if traversal_result:
                        expanded_breadcrumbs.extend([node.breadcrumb for node in traversal_result.related_components])

                except Exception as e:
                    self.logger.error(f"Error expanding from breadcrumb {seed_breadcrumb}: {e}")

        # Step 2: Create filters based on expanded breadcrumbs
        expanded_filters = CrossProjectSearchFilter(
            target_projects=project_names,
            # Add any specific filtering based on expanded scope
        )

        # Step 3: Perform hybrid search with expanded scope
        search_params.graph_expansion_used = True
        return await self.hybrid_search(query, project_names, search_params, expanded_filters)

    async def _adapt_search_strategy(
        self,
        query: str,
        params: HybridSearchParameters,
    ) -> HybridSearchParameters:
        """Adapt search strategy based on query characteristics."""
        if params.strategy != HybridSearchStrategy.ADAPTIVE:
            return params

        # Create a copy to avoid modifying the original
        adapted_params = HybridSearchParameters(
            strategy=params.strategy,
            semantic_weight=params.semantic_weight,
            structural_weight=params.structural_weight,
            context_weight=params.context_weight,
            min_semantic_similarity=params.min_semantic_similarity,
            min_structural_similarity=params.min_structural_similarity,
            max_traversal_depth=params.max_traversal_depth,
            expand_search_scope=params.expand_search_scope,
            max_results=params.max_results,
            diversity_factor=params.diversity_factor,
            enable_caching=params.enable_caching,
            timeout_seconds=params.timeout_seconds,
        )

        # Analyze query characteristics
        query_lower = query.lower()

        # If query contains structural keywords, increase structural weight
        structural_keywords = [
            "class",
            "function",
            "method",
            "module",
            "package",
            "namespace",
            "parent",
            "child",
            "hierarchy",
            "structure",
            "architecture",
            "inherit",
            "implement",
            "extend",
            "override",
        ]

        if any(keyword in query_lower for keyword in structural_keywords):
            adapted_params.structural_weight = min(0.6, adapted_params.structural_weight + 0.2)
            adapted_params.semantic_weight = max(0.2, adapted_params.semantic_weight - 0.1)
            self.logger.info("Adapted strategy: Increased structural weight due to structural keywords")

        # If query contains semantic keywords, increase semantic weight
        semantic_keywords = [
            "similar",
            "like",
            "resembles",
            "equivalent",
            "alternative",
            "behavior",
            "functionality",
            "purpose",
            "logic",
            "algorithm",
        ]

        if any(keyword in query_lower for keyword in semantic_keywords):
            adapted_params.semantic_weight = min(0.7, adapted_params.semantic_weight + 0.2)
            adapted_params.structural_weight = max(0.1, adapted_params.structural_weight - 0.1)
            self.logger.info("Adapted strategy: Increased semantic weight due to semantic keywords")

        # If query mentions specific patterns, enable graph expansion
        pattern_keywords = [
            "pattern",
            "design pattern",
            "architecture",
            "framework",
            "template",
            "example",
            "implementation",
            "best practice",
        ]

        if any(keyword in query_lower for keyword in pattern_keywords):
            adapted_params.expand_search_scope = True
            adapted_params.max_traversal_depth = min(4, adapted_params.max_traversal_depth + 1)
            self.logger.info("Adapted strategy: Enabled graph expansion for pattern search")

        # Renormalize weights
        total_weight = adapted_params.semantic_weight + adapted_params.structural_weight + adapted_params.context_weight
        if total_weight > 0:
            adapted_params.semantic_weight /= total_weight
            adapted_params.structural_weight /= total_weight
            adapted_params.context_weight /= total_weight

        return adapted_params

    async def _perform_semantic_search(
        self,
        query: str,
        project_names: list[str],
        params: HybridSearchParameters,
        filters: CrossProjectSearchFilter | None,
    ) -> list[tuple[CodeChunk, str, float]]:  # (chunk, project_name, semantic_score)
        """Perform semantic search across projects."""
        try:
            # Generate query embedding
            query_embedding_tensor = await self.embedding_service.generate_embeddings("nomic-embed-text", query)
            if query_embedding_tensor is None:
                self.logger.error("Failed to generate query embedding")
                return []

            # Convert tensor to list - ensure proper conversion
            if hasattr(query_embedding_tensor, "tolist"):
                query_embedding = query_embedding_tensor.tolist()
            elif hasattr(query_embedding_tensor, "numpy"):
                query_embedding = query_embedding_tensor.numpy().tolist()
            else:
                self.logger.error(f"Unexpected embedding type: {type(query_embedding_tensor)}")
                return []

            # Validate embedding format and dimensions
            if not isinstance(query_embedding, list) or len(query_embedding) == 0:
                self.logger.error(
                    f"Generated query embedding is invalid: type={type(query_embedding)}, len={len(query_embedding) if hasattr(query_embedding, '__len__') else 'N/A'}"
                )
                return []

            if len(query_embedding) != 768:
                self.logger.error(f"Query embedding dimension mismatch: expected 768, got {len(query_embedding)}")
                return []

            self.logger.info(f"Successfully generated query embedding: {len(query_embedding)} dimensions")

            # Search across all target projects
            semantic_candidates = []

            for project_name in project_names:
                collection_name = f"project_{project_name}_code"

                try:
                    # Build filter for this project
                    qdrant_filter = self._build_qdrant_filter(filters) if filters else None

                    # Log search parameters
                    self.logger.info(
                        f"Qdrant search params - collection: {collection_name}, "
                        f"vector_dim: {len(query_embedding)}, "
                        f"limit: {params.max_results * 2}, "
                        f"filter: {qdrant_filter is not None}, "
                        f"threshold: {params.min_semantic_similarity}"
                    )

                    # Perform vector search - only pass supported parameters
                    search_kwargs = {}
                    if qdrant_filter is not None:
                        search_kwargs["query_filter"] = qdrant_filter

                    search_response = await self.qdrant_service.search_vectors(
                        collection_name=collection_name,
                        query_vector=query_embedding,
                        limit=params.max_results * 2,  # Get more candidates for reranking
                        score_threshold=0.0,  # Override threshold for debugging
                        **search_kwargs,
                    )

                    # Log raw response details
                    self.logger.info(f"Raw Qdrant response type: {type(search_response)}, " f"is_dict: {isinstance(search_response, dict)}")
                    if isinstance(search_response, dict):
                        self.logger.info(f"Response keys: {list(search_response.keys())}")

                        # Check for errors first
                        if "error" in search_response:
                            self.logger.error(f"Qdrant search error for {collection_name}: {search_response['error']}")
                            continue

                        results_count = len(search_response.get("results", []))
                        self.logger.info(f"Results in response: {results_count}")
                        if results_count > 0:
                            first_result = search_response["results"][0]
                            self.logger.info(
                                f"First result keys: {list(first_result.keys()) if isinstance(first_result, dict) else 'not dict'}"
                            )
                            if isinstance(first_result, dict) and "score" in first_result:
                                self.logger.info(f"First result score: {first_result['score']}")
                    else:
                        self.logger.warning(f"Unexpected response format: {search_response}")

                    # Extract results from response
                    search_results = search_response.get("results", []) if isinstance(search_response, dict) else []

                    # Debug logging
                    self.logger.info(f"Search response for {collection_name}: {len(search_results)} results")
                    if search_results:
                        scores = [r.get("score", 0) for r in search_results if isinstance(r, dict)]
                        if scores:
                            self.logger.info(
                                f"Score range: {min(scores):.4f} - {max(scores):.4f}, threshold: {params.min_semantic_similarity}"
                            )

                    # Convert to CodeChunk objects
                    for result in search_results:
                        if isinstance(result, dict) and result.get("score", 0) >= params.min_semantic_similarity:
                            chunk = self._create_code_chunk_from_payload(result.get("payload", {}))
                            semantic_candidates.append((chunk, project_name, result.get("score", 0)))

                except Exception as e:
                    self.logger.error(f"Error searching project {project_name}: {e}")
                    continue

            return semantic_candidates

        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            return []

    async def _enhance_with_structural_analysis(
        self,
        semantic_candidates: list[tuple[CodeChunk, str, float]],
        query: str,
        params: HybridSearchParameters,
        project_names: list[str],
    ) -> list[tuple[CodeChunk, str, float, float, dict]]:  # (chunk, project, semantic_score, structural_score, context)
        """Enhance semantic candidates with structural analysis."""
        enhanced_candidates = []

        for chunk, project_name, semantic_score in semantic_candidates:
            try:
                # Calculate structural score
                structural_score = await self._calculate_structural_relevance(chunk, query, project_name, params)

                # Get contextual information
                context = await self._gather_contextual_information(chunk, project_name, params)

                enhanced_candidates.append((chunk, project_name, semantic_score, structural_score, context))

            except Exception as e:
                self.logger.error(f"Error enhancing candidate with structural analysis: {e}")
                # Add with default structural score
                enhanced_candidates.append((chunk, project_name, semantic_score, 0.5, {}))

        return enhanced_candidates

    async def _calculate_structural_relevance(
        self,
        chunk: CodeChunk,
        query: str,
        project_name: str,
        params: HybridSearchParameters,
    ) -> float:
        """Calculate structural relevance score for a chunk."""
        try:
            score_factors = []

            # Factor 1: Breadcrumb quality and depth
            if chunk.breadcrumb:
                depth_score = min(1.0, chunk.breadcrumb_depth / 5.0)  # Normalize depth
                breadcrumb_quality = len(chunk.get_breadcrumb_components()) / max(1, chunk.breadcrumb_depth)
                score_factors.append(depth_score * breadcrumb_quality)
            else:
                score_factors.append(0.3)  # Penalty for missing breadcrumb

            # Factor 2: Hierarchical context
            if chunk.parent_name:
                score_factors.append(0.8)  # Good hierarchical context
            else:
                score_factors.append(0.5)  # Neutral if no parent

            # Factor 3: Related components analysis
            if params.expand_search_scope and chunk.breadcrumb:
                try:
                    traversal_result = await self.graph_rag_service.traverse_from_breadcrumb(
                        breadcrumb=chunk.breadcrumb,
                        project_name=project_name,
                        max_depth=2,  # Limited depth for performance
                        strategy_name="semantic",
                    )

                    if traversal_result and traversal_result.related_components:
                        # Score based on number and quality of related components
                        related_score = min(1.0, len(traversal_result.related_components) / 10.0)
                        score_factors.append(related_score)
                    else:
                        score_factors.append(0.4)  # Some penalty for isolation

                except Exception:
                    score_factors.append(0.5)  # Neutral if traversal fails
            else:
                score_factors.append(0.6)  # Neutral if expansion disabled

            # Factor 4: Chunk type relevance
            # Score based on how likely this chunk type is to be relevant
            chunk_type_scores = {
                ChunkType.CLASS: 0.9,
                ChunkType.FUNCTION: 0.8,
                ChunkType.METHOD: 0.8,
                ChunkType.INTERFACE: 0.9,
                ChunkType.CONSTANT: 0.6,
                ChunkType.VARIABLE: 0.5,
                # Add more as needed
            }
            chunk_type_score = chunk_type_scores.get(chunk.chunk_type, 0.6)
            score_factors.append(chunk_type_score)

            # Calculate weighted average
            return sum(score_factors) / len(score_factors) if score_factors else 0.5

        except Exception as e:
            self.logger.error(f"Error calculating structural relevance: {e}")
            return 0.5

    async def _gather_contextual_information(
        self,
        chunk: CodeChunk,
        project_name: str,
        params: HybridSearchParameters,
    ) -> dict[str, Any]:
        """Gather contextual information for scoring."""
        context = {}

        try:
            # Basic context
            context["project"] = project_name
            context["language"] = chunk.language
            context["chunk_type"] = chunk.chunk_type.value
            context["has_docstring"] = bool(chunk.docstring)
            context["line_count"] = chunk.line_count

            # Structural context
            if chunk.breadcrumb:
                context["breadcrumb_depth"] = chunk.breadcrumb_depth
                context["breadcrumb_components"] = chunk.get_breadcrumb_components()

            # Complexity indicators
            context["complexity_indicators"] = {
                "has_parent": bool(chunk.parent_name),
                "has_dependencies": bool(chunk.dependencies),
                "has_imports": bool(chunk.imports_used),
                "line_count_category": ("small" if chunk.line_count < 10 else "medium" if chunk.line_count < 50 else "large"),
            }

            # File context
            if chunk.file_path:
                file_parts = chunk.file_path.split("/")
                context["file_context"] = {
                    "filename": file_parts[-1] if file_parts else "",
                    "directory_depth": len(file_parts) - 1,
                    "file_extension": chunk.file_path.split(".")[-1] if "." in chunk.file_path else "",
                }

            return context

        except Exception as e:
            self.logger.error(f"Error gathering contextual information: {e}")
            return {"project": project_name}

    async def _calculate_final_scores_and_rank(
        self,
        enhanced_candidates: list[tuple[CodeChunk, str, float, float, dict]],
        params: HybridSearchParameters,
    ) -> list[HybridSearchResult]:
        """Calculate final scores and rank results."""
        results = []

        for chunk, project_name, semantic_score, structural_score, context in enhanced_candidates:
            try:
                # Calculate context score
                context_score = self._calculate_context_score(context, params)

                # Calculate final weighted score
                final_score = (
                    params.semantic_weight * semantic_score
                    + params.structural_weight * structural_score
                    + params.context_weight * context_score
                )

                # Calculate confidence based on score distribution
                confidence = self._calculate_confidence(semantic_score, structural_score, context_score, params)

                # Create result
                result = HybridSearchResult(
                    chunk=chunk,
                    project_name=project_name,
                    semantic_score=semantic_score,
                    structural_score=structural_score,
                    context_score=context_score,
                    final_score=final_score,
                    confidence=confidence,
                    context_factors=context,
                )

                results.append(result)

            except Exception as e:
                self.logger.error(f"Error calculating final score: {e}")
                continue

        # Sort by final score (descending)
        sorted_results = sorted(results, key=lambda r: r.final_score, reverse=True)

        # Assign ranks
        for i, result in enumerate(sorted_results):
            result.rank = i + 1

        return sorted_results

    def _calculate_context_score(self, context: dict[str, Any], params: HybridSearchParameters) -> float:
        """Calculate context score based on contextual factors."""
        try:
            score_factors = []

            # Documentation quality
            if context.get("has_docstring"):
                score_factors.append(0.8)
            else:
                score_factors.append(0.4)

            # Code complexity appropriateness
            complexity = context.get("complexity_indicators", {})
            if complexity.get("has_parent") and complexity.get("has_dependencies"):
                score_factors.append(0.9)  # Well-integrated code
            elif complexity.get("has_parent") or complexity.get("has_dependencies"):
                score_factors.append(0.7)  # Some integration
            else:
                score_factors.append(0.5)  # Standalone code

            # File organization
            file_context = context.get("file_context", {})
            if file_context.get("directory_depth", 0) > 0:
                score_factors.append(0.7)  # Organized in directories
            else:
                score_factors.append(0.5)  # Root level files

            # Language-specific factors
            language = context.get("language", "unknown")
            language_scores = {"python": 0.8, "javascript": 0.8, "typescript": 0.9, "java": 0.8, "cpp": 0.7, "rust": 0.8, "go": 0.8}
            lang_score = language_scores.get(language.lower(), 0.6)
            score_factors.append(lang_score)

            return sum(score_factors) / len(score_factors) if score_factors else 0.5

        except Exception as e:
            self.logger.error(f"Error calculating context score: {e}")
            return 0.5

    def _calculate_confidence(
        self,
        semantic_score: float,
        structural_score: float,
        context_score: float,
        params: HybridSearchParameters,
    ) -> float:
        """Calculate confidence in the result."""
        try:
            # Confidence based on score consistency
            scores = [semantic_score, structural_score, context_score]
            avg_score = sum(scores) / len(scores)
            variance = sum((s - avg_score) ** 2 for s in scores) / len(scores)
            consistency = 1.0 - min(1.0, variance)  # Lower variance = higher confidence

            # Confidence based on absolute scores
            min_score = min(scores)

            # Combined confidence
            confidence = consistency * 0.6 + min_score * 0.4

            return max(0.0, min(1.0, confidence))

        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5

    def _apply_diversity_filtering(
        self,
        results: list[HybridSearchResult],
        params: HybridSearchParameters,
    ) -> list[HybridSearchResult]:
        """Apply diversity filtering to promote result variety."""
        if params.diversity_factor <= 0 or len(results) <= 1:
            return results

        try:
            # Group results by similarity characteristics
            diversity_groups = {}

            for result in results:
                # Create diversity key based on project, language, and chunk type
                diversity_key = (
                    result.project_name,
                    result.chunk.language,
                    result.chunk.chunk_type.value,
                )

                if diversity_key not in diversity_groups:
                    diversity_groups[diversity_key] = []
                diversity_groups[diversity_key].append(result)

            # Select diverse results
            diverse_results = []
            max_per_group = max(1, int(params.max_results / len(diversity_groups)))

            for group_results in diversity_groups.values():
                # Sort group by score and take top results
                group_results.sort(key=lambda r: r.final_score, reverse=True)
                diverse_results.extend(group_results[:max_per_group])

            # Sort final results by score
            diverse_results.sort(key=lambda r: r.final_score, reverse=True)

            return diverse_results

        except Exception as e:
            self.logger.error(f"Error applying diversity filtering: {e}")
            return results

    def _generate_cache_key(
        self,
        query: str,
        project_names: list[str],
        params: HybridSearchParameters,
        filters: CrossProjectSearchFilter | None,
    ) -> str:
        """Generate cache key for search results."""
        import hashlib

        # Create key from search parameters
        key_components = [
            query,
            "|".join(sorted(project_names)),
            params.strategy.value,
            f"{params.semantic_weight:.2f}",
            f"{params.structural_weight:.2f}",
            f"{params.context_weight:.2f}",
            str(params.max_results),
        ]

        if filters:
            key_components.extend(
                [
                    "|".join(sorted(filters.target_projects)) if filters.target_projects else "",
                    "|".join(sorted(filters.languages)) if filters.languages else "",
                    "|".join(ct.value for ct in filters.chunk_types) if filters.chunk_types else "",
                ]
            )

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        return self._cache_stats["hits"] / total_requests if total_requests > 0 else 0.0

    def _build_qdrant_filter(self, filters: CrossProjectSearchFilter) -> Any | None:
        """Build Qdrant filter from search filters."""
        conditions = []

        # Filter by chunk types
        if filters.chunk_types:
            chunk_type_values = [ct.value for ct in filters.chunk_types]
            conditions.append({"key": "chunk_type", "match": {"any": chunk_type_values}})

        # Filter by languages
        if filters.languages:
            conditions.append({"key": "language", "match": {"any": filters.languages}})

        # Exclude languages
        if filters.exclude_languages:
            conditions.append({"key": "language", "match": {"except": filters.exclude_languages}})

        if not conditions:
            return None

        return {"must": conditions} if len(conditions) > 1 else conditions[0]

    async def get_all_chunks(self, project_name: str) -> list[CodeChunk]:
        """
        Get all chunks for a project (needed for LightweightGraphService initialization).

        Args:
            project_name: Name of the project to get chunks for

        Returns:
            List of all CodeChunk objects for the project
        """
        try:
            collection_name = f"project_{project_name}_code"

            # Get all points from the collection (using a large limit)
            search_results = await self.qdrant_service.scroll(
                collection_name=collection_name,
                limit=10000,  # Large limit to get all chunks
                with_payload=True,
                with_vectors=False,  # Don't need vectors for metadata extraction
            )

            chunks = []
            for result in search_results:
                chunk = self._create_code_chunk_from_payload(result.payload)
                chunks.append(chunk)

            self.logger.info(f"Retrieved {len(chunks)} chunks for project {project_name}")
            return chunks

        except Exception as e:
            self.logger.error(f"Error getting all chunks for project {project_name}: {e}")
            return []

    def _create_code_chunk_from_payload(self, payload: dict) -> CodeChunk:
        """Create a CodeChunk object from Qdrant payload."""
        try:
            # Map Qdrant payload to CodeChunk fields
            chunk_type_str = payload.get("chunk_type", "function")
            chunk_type = ChunkType(chunk_type_str) if chunk_type_str in [ct.value for ct in ChunkType] else ChunkType.FUNCTION

            return CodeChunk(
                chunk_id=payload.get("chunk_id", ""),
                file_path=payload.get("file_path", ""),
                content=payload.get("content", ""),
                chunk_type=chunk_type,
                language=payload.get("language", "unknown"),
                start_line=payload.get("start_line", 0),
                end_line=payload.get("end_line", 0),
                start_byte=payload.get("start_byte", 0),
                end_byte=payload.get("end_byte", 0),
                name=payload.get("name"),
                parent_name=payload.get("parent_name"),
                signature=payload.get("signature"),
                docstring=payload.get("docstring"),
                breadcrumb=payload.get("breadcrumb"),
                content_hash=payload.get("content_hash"),
                tags=payload.get("tags", []),
                dependencies=payload.get("dependencies", []),
                access_modifier=payload.get("access_modifier"),
                imports_used=payload.get("imports_used", []),
                has_syntax_errors=payload.get("has_syntax_errors", False),
                error_details=payload.get("error_details"),
            )

        except Exception as e:
            self.logger.error(f"Error creating CodeChunk from payload: {e}")
            # Return a minimal CodeChunk
            return CodeChunk(
                chunk_id="error",
                file_path="",
                content="",
                chunk_type=ChunkType.FUNCTION,
                language="unknown",
                start_line=0,
                end_line=0,
                start_byte=0,
                end_byte=0,
            )


# Factory function for dependency injection
_hybrid_search_service_instance = None


def get_hybrid_search_service(
    qdrant_service: QdrantService = None,
    embedding_service: EmbeddingService = None,
    graph_rag_service: GraphRAGService = None,
) -> HybridSearchService:
    """
    Get or create a HybridSearchService instance.

    Args:
        qdrant_service: Qdrant service instance (optional, will be created if not provided)
        embedding_service: Embedding service instance (optional, will be created if not provided)
        graph_rag_service: Graph RAG service instance (optional, will be created if not provided)

    Returns:
        HybridSearchService instance
    """
    global _hybrid_search_service_instance

    if _hybrid_search_service_instance is None:
        from .embedding_service import EmbeddingService
        from .graph_rag_service import get_graph_rag_service
        from .qdrant_service import QdrantService

        if qdrant_service is None:
            qdrant_service = QdrantService()
        if embedding_service is None:
            embedding_service = EmbeddingService()
        if graph_rag_service is None:
            graph_rag_service = get_graph_rag_service()

        _hybrid_search_service_instance = HybridSearchService(
            qdrant_service=qdrant_service,
            embedding_service=embedding_service,
            graph_rag_service=graph_rag_service,
        )

    return _hybrid_search_service_instance
