"""
Multi-Modal Retrieval Strategy Service

This service implements the four LightRAG-inspired retrieval modes:
- Local: Deep entity-focused retrieval using low-level keywords
- Global: Broad relationship-focused retrieval using high-level keywords
- Hybrid: Combined local+global with balanced context
- Mix: Intelligent automatic mode selection based on query analysis

Built on top of Wave 1.0 & 2.0 foundations with existing Graph RAG capabilities.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from ..models.code_chunk import CodeChunk
from ..models.query_features import (
    GlobalModeConfig,
    HybridModeConfig,
    LocalModeConfig,
    MixModeConfig,
    PerformanceMetrics,
    QueryFeatures,
    RetrievalResult,
)
from .embedding_service import EmbeddingService
from .hybrid_search_service import HybridSearchParameters, HybridSearchStrategy, get_hybrid_search_service
from .qdrant_service import QdrantService
from .query_analyzer import get_query_analyzer

logger = logging.getLogger(__name__)


@dataclass
class MultiModalSearchResult:
    """Extended search result with multi-modal metadata."""

    chunk: CodeChunk
    project_name: str
    mode_used: str

    # Multi-modal scoring
    local_score: float = 0.0
    global_score: float = 0.0
    combined_score: float = 0.0

    # Source information
    retrieval_source: str = "vector"  # vector, graph, hybrid
    confidence_level: str = "medium"  # low, medium, high

    # Context information
    local_context: list[str] = None
    global_context: list[str] = None

    def __post_init__(self):
        if self.local_context is None:
            self.local_context = []
        if self.global_context is None:
            self.global_context = []


class MultiModalRetrievalStrategy:
    """
    Core service implementing the four retrieval modes for the multi-modal system.

    This service orchestrates different retrieval strategies based on query analysis
    and provides a unified interface for all retrieval modes.
    """

    def __init__(
        self,
        qdrant_service: QdrantService | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        """Initialize the multi-modal retrieval strategy service."""
        # Ensure we have proper service instances
        if qdrant_service is None or embedding_service is None:
            raise ValueError("qdrant_service and embedding_service are required for first initialization")

        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service
        self.logger = logging.getLogger(__name__)

        # Initialize dependent services
        self._hybrid_search_service = None
        self._query_analyzer = None

        # Performance tracking
        self.performance_metrics = {
            "local": PerformanceMetrics("local"),
            "global": PerformanceMetrics("global"),
            "hybrid": PerformanceMetrics("hybrid"),
            "mix": PerformanceMetrics("mix"),
        }

        # Configuration
        self.mode_configs = {
            "local": LocalModeConfig(),
            "global": GlobalModeConfig(),
            "hybrid": HybridModeConfig(),
            "mix": MixModeConfig(),
        }

        # Cache for frequent queries
        self._query_cache = {}
        self._cache_stats = {"hits": 0, "misses": 0}

    async def _get_hybrid_search_service(self):
        """Lazy initialization of hybrid search service."""
        if self._hybrid_search_service is None:
            self._hybrid_search_service = get_hybrid_search_service(
                qdrant_service=self.qdrant_service,
                embedding_service=self.embedding_service,
            )
        return self._hybrid_search_service

    async def _get_query_analyzer(self):
        """Lazy initialization of query analyzer."""
        if self._query_analyzer is None:
            self._query_analyzer = await get_query_analyzer()
        return self._query_analyzer

    async def search(
        self,
        query: str,
        project_names: list[str],
        mode: str | None = None,
        n_results: int = 10,
        enable_manual_mode_selection: bool = False,
    ) -> RetrievalResult:
        """
        Perform multi-modal search with automatic or manual mode selection.

        Args:
            query: Natural language search query
            project_names: List of project names to search in
            mode: Optional manual mode selection ('local', 'global', 'hybrid', 'mix')
            n_results: Number of results to return
            enable_manual_mode_selection: Whether to allow manual mode override

        Returns:
            RetrievalResult with search results and performance metrics
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting multi-modal search: '{query[:50]}...' across {len(project_names)} projects")

            # Analyze query to determine optimal retrieval strategy
            query_analyzer = await self._get_query_analyzer()
            analysis_start = time.time()
            query_features = await query_analyzer.analyze_query(query)
            analysis_time = (time.time() - analysis_start) * 1000

            # Determine retrieval mode
            if enable_manual_mode_selection and mode and mode in self.mode_configs:
                selected_mode = mode
                self.logger.info(f"Using manually selected mode: {selected_mode}")
            else:
                selected_mode = query_features.recommended_mode
                self.logger.info(f"Using AI-recommended mode: {selected_mode} (confidence: {query_features.mode_confidence:.2f})")

            # Get mode configuration
            mode_config = self.mode_configs[selected_mode]
            adapted_config = query_analyzer.adapt_mode_config(mode_config, query_features)

            # Perform retrieval based on selected mode
            retrieval_start = time.time()

            if selected_mode == "local":
                results = await self._local_mode_retrieval(query, project_names, adapted_config, query_features)
            elif selected_mode == "global":
                results = await self._global_mode_retrieval(query, project_names, adapted_config, query_features)
            elif selected_mode == "hybrid":
                results = await self._hybrid_mode_retrieval(query, project_names, adapted_config, query_features)
            elif selected_mode == "mix":
                results = await self._mix_mode_retrieval(query, project_names, adapted_config, query_features)
            else:
                # Fallback to hybrid mode
                self.logger.warning(f"Unknown mode '{selected_mode}', falling back to hybrid")
                results = await self._hybrid_mode_retrieval(query, project_names, adapted_config, query_features)
                selected_mode = "hybrid"

            retrieval_time = (time.time() - retrieval_start) * 1000

            # Post-process results
            post_process_start = time.time()
            processed_results = await self._post_process_results(results, query_features, selected_mode)
            post_process_time = (time.time() - post_process_start) * 1000

            # Calculate metrics
            total_time = (time.time() - start_time) * 1000

            # Build result
            retrieval_result = RetrievalResult(
                query=query,
                mode_used=selected_mode,
                config_used=adapted_config,
                results=processed_results,
                total_execution_time_ms=total_time,
                query_analysis_time_ms=analysis_time,
                retrieval_time_ms=retrieval_time,
                post_processing_time_ms=post_process_time,
                total_results=len(processed_results),
                average_confidence=self._calculate_average_confidence(processed_results),
                result_diversity_score=self._calculate_diversity_score(processed_results),
            )

            # Update performance metrics
            self.performance_metrics[selected_mode].update_metrics(retrieval_result)

            self.logger.info(
                f"Multi-modal search completed: {len(processed_results)} results in {total_time:.2f}ms " f"using {selected_mode} mode"
            )

            return retrieval_result

        except Exception as e:
            self.logger.error(f"Error in multi-modal search: {e}")
            total_time = (time.time() - start_time) * 1000

            return RetrievalResult(
                query=query,
                mode_used=mode or "hybrid",
                config_used=self.mode_configs.get(mode, self.mode_configs["hybrid"]),
                results=[],
                total_execution_time_ms=total_time,
                error_message=str(e),
            )

    async def _local_mode_retrieval(
        self,
        query: str,
        project_names: list[str],
        config: LocalModeConfig,
        query_features: QueryFeatures,
    ) -> list[MultiModalSearchResult]:
        """
        Implement Local mode retrieval - focused on specific entities and their direct relationships.

        This mode emphasizes:
        - Entity-specific searches using low-level keywords
        - Limited graph expansion (depth=1)
        - High weight on entity tokens
        - Precise, focused results
        """
        self.logger.debug(f"Executing local mode retrieval for query: '{query[:50]}...'")

        hybrid_service = await self._get_hybrid_search_service()

        # Configure for local mode - entity-focused search
        search_params = HybridSearchParameters(
            strategy=HybridSearchStrategy.SEMANTIC_FIRST,
            semantic_weight=config.weight_local,  # 0.8
            structural_weight=config.weight_global,  # 0.2
            context_weight=0.1,
            max_results=config.max_results,
            diversity_factor=0.1,  # Lower diversity for focused results
        )

        # Use entity names and low-level keywords for focused search
        entity_focused_query = self._build_entity_focused_query(query, query_features)

        # Perform hybrid search with local configuration
        search_summary = await hybrid_service.hybrid_search(
            query=entity_focused_query,
            project_names=project_names,
            search_params=search_params,
        )

        # Convert to multi-modal results
        results = []
        for hybrid_result in search_summary.results:
            multi_modal_result = MultiModalSearchResult(
                chunk=hybrid_result.chunk,
                project_name=hybrid_result.project_name,
                mode_used="local",
                local_score=hybrid_result.semantic_score,
                global_score=hybrid_result.structural_score * 0.3,  # De-emphasize global
                combined_score=hybrid_result.final_score,
                retrieval_source="hybrid",
                confidence_level=self._determine_confidence_level(hybrid_result.confidence),
                local_context=self._extract_local_context(hybrid_result),
            )
            results.append(multi_modal_result)

        self.logger.debug(f"Local mode retrieval completed: {len(results)} results")
        return results

    async def _global_mode_retrieval(
        self,
        query: str,
        project_names: list[str],
        config: GlobalModeConfig,
        query_features: QueryFeatures,
    ) -> list[MultiModalSearchResult]:
        """
        Implement Global mode retrieval - focused on relationships and broad conceptual connections.

        This mode emphasizes:
        - Relationship-focused searches using high-level keywords
        - Deep graph expansion (depth=3)
        - High weight on relationship tokens
        - Broad, interconnected results
        """
        self.logger.debug(f"Executing global mode retrieval for query: '{query[:50]}...'")

        hybrid_service = await self._get_hybrid_search_service()

        # Configure for global mode - relationship-focused search
        search_params = HybridSearchParameters(
            strategy=HybridSearchStrategy.BALANCED,  # Fallback to BALANCED
            semantic_weight=config.weight_local,  # 0.2
            structural_weight=config.weight_global,  # 0.8
            context_weight=0.3,
            max_results=config.max_results,
            diversity_factor=0.3,  # Higher diversity for broad results
        )

        # Use concept terms and high-level keywords for broad search
        concept_focused_query = self._build_concept_focused_query(query, query_features)

        # Perform hybrid search with global configuration
        search_summary = await hybrid_service.hybrid_search(
            query=concept_focused_query,
            project_names=project_names,
            search_params=search_params,
        )

        # Enhance with graph-based relationship discovery
        enhanced_results = []
        for hybrid_result in search_summary.results:
            multi_modal_result = MultiModalSearchResult(
                chunk=hybrid_result.chunk,
                project_name=hybrid_result.project_name,
                mode_used="global",
                local_score=hybrid_result.semantic_score * 0.3,  # De-emphasize local
                global_score=hybrid_result.structural_score,
                combined_score=hybrid_result.final_score,
                retrieval_source="hybrid",  # Fallback to hybrid
                confidence_level=self._determine_confidence_level(hybrid_result.confidence),
                global_context=self._extract_global_context(hybrid_result),
            )
            enhanced_results.append(multi_modal_result)

        self.logger.debug(f"Global mode retrieval completed: {len(enhanced_results)} results")
        return enhanced_results

    async def _hybrid_mode_retrieval(
        self,
        query: str,
        project_names: list[str],
        config: HybridModeConfig,
        query_features: QueryFeatures,
    ) -> list[MultiModalSearchResult]:
        """
        Implement Hybrid mode retrieval - balanced combination of local and global approaches.

        This mode provides:
        - Balanced entity and relationship focus
        - Moderate graph expansion (depth=2)
        - Equal weight distribution
        - Comprehensive results with both depth and breadth
        """
        self.logger.debug(f"Executing hybrid mode retrieval for query: '{query[:50]}...'")

        hybrid_service = await self._get_hybrid_search_service()

        # Configure for hybrid mode - balanced search
        search_params = HybridSearchParameters(
            strategy=HybridSearchStrategy.BALANCED,
            semantic_weight=config.weight_local,  # 0.5
            structural_weight=config.weight_global,  # 0.5
            context_weight=0.2,
            max_results=config.max_results,
            diversity_factor=0.2,  # Moderate diversity
        )

        # Use balanced query construction
        balanced_query = self._build_balanced_query(query, query_features)

        # Perform hybrid search with balanced configuration
        search_summary = await hybrid_service.hybrid_search(
            query=balanced_query,
            project_names=project_names,
            search_params=search_params,
        )

        # Process results with balanced scoring
        results = []
        for hybrid_result in search_summary.results:
            multi_modal_result = MultiModalSearchResult(
                chunk=hybrid_result.chunk,
                project_name=hybrid_result.project_name,
                mode_used="hybrid",
                local_score=hybrid_result.semantic_score,
                global_score=hybrid_result.structural_score,
                combined_score=hybrid_result.final_score,
                retrieval_source="hybrid",
                confidence_level=self._determine_confidence_level(hybrid_result.confidence),
                local_context=self._extract_local_context(hybrid_result),
                global_context=self._extract_global_context(hybrid_result),
            )
            results.append(multi_modal_result)

        self.logger.debug(f"Hybrid mode retrieval completed: {len(results)} results")
        return results

    async def _mix_mode_retrieval(
        self,
        query: str,
        project_names: list[str],
        config: MixModeConfig,
        query_features: QueryFeatures,
    ) -> list[MultiModalSearchResult]:
        """
        Implement Mix mode retrieval - intelligent automatic mode selection with adaptive parameters.

        This mode:
        - Analyzes query characteristics in real-time
        - Dynamically selects the best approach
        - Adapts parameters based on query features
        - Falls back to hybrid if uncertain
        """
        self.logger.debug(f"Executing mix mode retrieval for query: '{query[:50]}...'")

        # Analyze query for dynamic mode selection
        if query_features.mode_confidence >= config.confidence_threshold:
            # High confidence - use recommended mode
            selected_sub_mode = query_features.recommended_mode
            self.logger.debug(f"Mix mode: High confidence, using {selected_sub_mode}")
        else:
            # Low confidence - use fallback
            selected_sub_mode = config.fallback_mode
            self.logger.debug(f"Mix mode: Low confidence, using fallback {selected_sub_mode}")

        # Recursively call the appropriate mode
        if selected_sub_mode == "local":
            results = await self._local_mode_retrieval(query, project_names, LocalModeConfig(), query_features)
        elif selected_sub_mode == "global":
            results = await self._global_mode_retrieval(query, project_names, GlobalModeConfig(), query_features)
        else:  # hybrid
            results = await self._hybrid_mode_retrieval(query, project_names, HybridModeConfig(), query_features)

        # Update mode_used to reflect mix mode
        for result in results:
            result.mode_used = f"mix({selected_sub_mode})"

        self.logger.debug(f"Mix mode retrieval completed: {len(results)} results using {selected_sub_mode}")
        return results

    def _build_entity_focused_query(self, query: str, features: QueryFeatures) -> str:
        """Build a query focused on entities for local mode."""
        # Emphasize entity names and low-level keywords
        entity_terms = features.keywords.entity_names + features.keywords.low_level_keywords
        if entity_terms:
            # Add entity emphasis to the original query
            entity_emphasis = " ".join(entity_terms[:3])  # Top 3 entity terms
            return f"{query} {entity_emphasis}"
        return query

    def _build_concept_focused_query(self, query: str, features: QueryFeatures) -> str:
        """Build a query focused on concepts for global mode."""
        # Emphasize concept terms and high-level keywords
        concept_terms = features.keywords.concept_terms + features.keywords.high_level_keywords
        if concept_terms:
            # Add concept emphasis to the original query
            concept_emphasis = " ".join(concept_terms[:3])  # Top 3 concept terms
            return f"{query} {concept_emphasis}"
        return query

    def _build_balanced_query(self, query: str, features: QueryFeatures) -> str:
        """Build a balanced query for hybrid mode."""
        # Use original query with minimal modification for balanced approach
        return query

    def _extract_local_context(self, hybrid_result) -> list[str]:
        """Extract local context information from hybrid result."""
        context = []

        if hasattr(hybrid_result, "chunk") and hybrid_result.chunk:
            # Add breadcrumb information
            if hybrid_result.chunk.breadcrumb:
                context.append(f"breadcrumb: {hybrid_result.chunk.breadcrumb}")

            # Add parent information
            if hybrid_result.chunk.parent_name:
                context.append(f"parent: {hybrid_result.chunk.parent_name}")

            # Add signature if available
            if hybrid_result.chunk.signature:
                context.append(f"signature: {hybrid_result.chunk.signature}")

        return context

    def _extract_global_context(self, hybrid_result) -> list[str]:
        """Extract global context information from hybrid result."""
        context = []

        if hasattr(hybrid_result, "chunk") and hybrid_result.chunk:
            # Add dependencies
            if hybrid_result.chunk.dependencies:
                context.extend([f"depends_on: {dep}" for dep in hybrid_result.chunk.dependencies[:3]])

            # Add imports
            if hybrid_result.chunk.imports_used:
                context.extend([f"imports: {imp}" for imp in hybrid_result.chunk.imports_used[:3]])

            # Add tags
            if hybrid_result.chunk.tags:
                context.extend([f"tag: {tag}" for tag in hybrid_result.chunk.tags[:3]])

        return context

    def _determine_confidence_level(self, confidence: float) -> str:
        """Determine confidence level from numeric confidence."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.5:
            return "medium"
        else:
            return "low"

    async def _post_process_results(
        self,
        results: list[MultiModalSearchResult],
        query_features: QueryFeatures,
        mode_used: str,
    ) -> list[dict[str, Any]]:
        """Post-process results for final output."""
        processed_results = []

        for i, result in enumerate(results):
            processed_result = {
                # Basic information
                "rank": i + 1,
                "file_path": result.chunk.file_path,
                "content": result.chunk.content,
                "breadcrumb": result.chunk.breadcrumb,
                "chunk_type": result.chunk.chunk_type.value if result.chunk.chunk_type else "unknown",
                "language": result.chunk.language,
                "project": result.project_name,
                # Multi-modal specific
                "retrieval_mode": result.mode_used,
                "local_score": result.local_score,
                "global_score": result.global_score,
                "combined_score": result.combined_score,
                "confidence_level": result.confidence_level,
                "retrieval_source": result.retrieval_source,
                # Context information
                "local_context": result.local_context,
                "global_context": result.global_context,
                # Metadata
                "line_start": result.chunk.start_line,
                "line_end": result.chunk.end_line,
                "name": result.chunk.name,
                "parent_name": result.chunk.parent_name,
                "signature": result.chunk.signature,
                "docstring": result.chunk.docstring,
            }

            processed_results.append(processed_result)

        return processed_results

    def _calculate_average_confidence(self, results: list[dict[str, Any]]) -> float:
        """Calculate average confidence from processed results."""
        if not results:
            return 0.0

        total_confidence = sum(result.get("combined_score", 0.0) for result in results)
        return total_confidence / len(results)

    def _calculate_diversity_score(self, results: list[dict[str, Any]]) -> float:
        """Calculate diversity score based on result variety."""
        if not results:
            return 0.0

        # Calculate diversity based on different dimensions
        unique_projects = len({result.get("project", "") for result in results})
        unique_languages = len({result.get("language", "") for result in results})
        unique_chunk_types = len({result.get("chunk_type", "") for result in results})

        # Normalize by total results
        total_results = len(results)
        diversity_score = (
            (unique_projects / total_results) * 0.4 + (unique_languages / total_results) * 0.3 + (unique_chunk_types / total_results) * 0.3
        )

        return min(diversity_score, 1.0)

    def get_performance_metrics(self) -> dict[str, PerformanceMetrics]:
        """Get performance metrics for all modes."""
        return self.performance_metrics.copy()

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        total_requests = self._cache_stats["hits"] + self._cache_stats["misses"]
        hit_rate = self._cache_stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            "cache_hits": self._cache_stats["hits"],
            "cache_misses": self._cache_stats["misses"],
            "total_requests": total_requests,
            "hit_rate": hit_rate,
        }


# Factory function
_multi_modal_retrieval_strategy_instance = None


def get_multi_modal_retrieval_strategy(
    qdrant_service: QdrantService | None = None,
    embedding_service: EmbeddingService | None = None,
) -> MultiModalRetrievalStrategy:
    """Get or create a MultiModalRetrievalStrategy instance."""
    global _multi_modal_retrieval_strategy_instance

    # Force recreation if services are explicitly provided and instance exists
    if (qdrant_service is not None or embedding_service is not None) and _multi_modal_retrieval_strategy_instance is not None:
        logger.info("Recreating MultiModalRetrievalStrategy with new service dependencies")
        _multi_modal_retrieval_strategy_instance = None

    if _multi_modal_retrieval_strategy_instance is None:
        # Create services if not provided
        if qdrant_service is None:
            logger.debug("Creating new QdrantService instance")
            qdrant_service = QdrantService()

        if embedding_service is None:
            logger.debug("Creating new EmbeddingService instance")
            embedding_service = EmbeddingService()

        _multi_modal_retrieval_strategy_instance = MultiModalRetrievalStrategy(
            qdrant_service=qdrant_service,
            embedding_service=embedding_service,
        )
        logger.info(
            f"MultiModalRetrievalStrategy instance created with services: "
            f"qdrant={qdrant_service is not None}, embedding={embedding_service is not None}"
        )
    else:
        logger.debug("MultiModalRetrievalStrategy instance already exists, using existing instance")

    return _multi_modal_retrieval_strategy_instance
