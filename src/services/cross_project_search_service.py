"""
Cross-Project Search Service for Graph RAG enhancement.

This service provides cross-project search capabilities using structural relationships,
enabling developers to find similar implementations, patterns, and architectural
solutions across multiple indexed projects.

Built on top of Wave 2's Graph RAG infrastructure and relationship building capabilities.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Optional, Union

from src.models.code_chunk import ChunkType, CodeChunk

from .embedding_service import EmbeddingService
from .graph_rag_service import GraphRAGService, GraphTraversalResult
from .qdrant_service import QdrantService
from .structure_relationship_builder import GraphNode, StructureGraph


@dataclass
class CrossProjectSearchFilter:
    """Filters for cross-project search operations."""

    # Project filtering
    target_projects: list[str] = None  # Specific projects to search in
    exclude_projects: list[str] = None  # Projects to exclude

    # Structural filtering
    chunk_types: list[ChunkType] = None  # Specific chunk types to find
    min_depth: int = 0  # Minimum breadcrumb depth
    max_depth: int = 10  # Maximum breadcrumb depth

    # Semantic filtering
    similarity_threshold: float = 0.7  # Minimum semantic similarity
    structural_weight: float = 0.5  # Weight for structural vs semantic similarity

    # Language filtering
    languages: list[str] = None  # Programming languages to include
    exclude_languages: list[str] = None  # Languages to exclude

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.target_projects is None:
            self.target_projects = []
        if self.exclude_projects is None:
            self.exclude_projects = []
        if self.chunk_types is None:
            self.chunk_types = []
        if self.languages is None:
            self.languages = []
        if self.exclude_languages is None:
            self.exclude_languages = []


@dataclass
class CrossProjectMatch:
    """Represents a match found across projects."""

    chunk: CodeChunk
    project_name: str
    similarity_score: float
    structural_score: float
    combined_score: float

    # Context information
    related_components: list[GraphNode] = None
    architectural_context: dict[str, Any] = None
    usage_patterns: list[str] = None

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if self.related_components is None:
            self.related_components = []
        if self.architectural_context is None:
            self.architectural_context = {}
        if self.usage_patterns is None:
            self.usage_patterns = []


@dataclass
class CrossProjectSearchResult:
    """Result of cross-project search operation."""

    query: str
    matches: list[CrossProjectMatch]
    projects_searched: list[str]
    total_chunks_examined: int
    execution_time_ms: float

    # Search strategy information
    search_filters: CrossProjectSearchFilter
    semantic_weight_used: float
    structural_weight_used: float

    # Statistics
    matches_by_project: dict[str, int] = None
    matches_by_language: dict[str, int] = None
    avg_similarity_score: float = 0.0

    def __post_init__(self):
        """Initialize default values and compute statistics."""
        if self.matches_by_project is None:
            self.matches_by_project = {}
        if self.matches_by_language is None:
            self.matches_by_language = {}

        # Compute statistics
        if self.matches:
            self.avg_similarity_score = sum(m.combined_score for m in self.matches) / len(self.matches)

            # Count matches by project
            for match in self.matches:
                project = match.project_name
                self.matches_by_project[project] = self.matches_by_project.get(project, 0) + 1

            # Count matches by language
            for match in self.matches:
                lang = match.chunk.language
                self.matches_by_language[lang] = self.matches_by_language.get(lang, 0) + 1


class CrossProjectSearchService:
    """
    Service for performing cross-project searches with structural relationship filtering.

    This service combines semantic similarity with structural relationships to find
    relevant implementations across multiple projects, enabling learning from
    existing architectural patterns and solutions.
    """

    def __init__(
        self,
        qdrant_service: QdrantService,
        embedding_service: EmbeddingService,
        graph_rag_service: GraphRAGService,
    ):
        """Initialize the cross-project search service.

        Args:
            qdrant_service: Service for vector database operations
            embedding_service: Service for generating embeddings
            graph_rag_service: Service for graph operations and relationship analysis
        """
        self.qdrant_service = qdrant_service
        self.embedding_service = embedding_service
        self.graph_rag_service = graph_rag_service
        self.logger = logging.getLogger(__name__)

    async def search_across_projects(
        self,
        query: str,
        search_filters: CrossProjectSearchFilter,
        max_results: int = 20,
    ) -> CrossProjectSearchResult:
        """
        Search for similar implementations across multiple projects.

        Args:
            query: Natural language query describing what to find
            search_filters: Filters to apply to the search
            max_results: Maximum number of results to return

        Returns:
            CrossProjectSearchResult with matching implementations
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting cross-project search for: {query}")

            # Step 1: Get available projects
            available_projects = await self._get_available_projects()
            target_projects = self._filter_target_projects(available_projects, search_filters)

            if not target_projects:
                self.logger.warning("No target projects found for cross-project search")
                return CrossProjectSearchResult(
                    query=query,
                    matches=[],
                    projects_searched=[],
                    total_chunks_examined=0,
                    execution_time_ms=(time.time() - start_time) * 1000,
                    search_filters=search_filters,
                    semantic_weight_used=1.0 - search_filters.structural_weight,
                    structural_weight_used=search_filters.structural_weight,
                )

            # Step 2: Generate query embedding
            query_embedding = await self._generate_query_embedding(query)

            # Step 3: Search across target projects
            all_matches = []
            total_chunks_examined = 0

            for project in target_projects:
                self.logger.info(f"Searching in project: {project}")

                # Get semantic matches from this project
                semantic_matches = await self._search_project_semantically(project, query_embedding, search_filters, max_results * 2)

                # Enhance matches with structural information
                enhanced_matches = await self._enhance_with_structural_context(semantic_matches, search_filters, project)

                all_matches.extend(enhanced_matches)
                total_chunks_examined += len(semantic_matches)

            # Step 4: Rank and filter final results
            ranked_matches = self._rank_and_filter_matches(all_matches, search_filters, max_results)

            execution_time_ms = (time.time() - start_time) * 1000

            result = CrossProjectSearchResult(
                query=query,
                matches=ranked_matches,
                projects_searched=target_projects,
                total_chunks_examined=total_chunks_examined,
                execution_time_ms=execution_time_ms,
                search_filters=search_filters,
                semantic_weight_used=1.0 - search_filters.structural_weight,
                structural_weight_used=search_filters.structural_weight,
            )

            self.logger.info(
                f"Cross-project search completed in {execution_time_ms:.2f}ms. "
                f"Found {len(ranked_matches)} matches across {len(target_projects)} projects."
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in cross-project search: {e}")
            execution_time_ms = (time.time() - start_time) * 1000
            return CrossProjectSearchResult(
                query=query,
                matches=[],
                projects_searched=[],
                total_chunks_examined=0,
                execution_time_ms=execution_time_ms,
                search_filters=search_filters,
                semantic_weight_used=1.0 - search_filters.structural_weight,
                structural_weight_used=search_filters.structural_weight,
            )

    async def find_similar_implementations(
        self,
        source_chunk: CodeChunk,
        search_filters: CrossProjectSearchFilter,
        max_results: int = 10,
    ) -> CrossProjectSearchResult:
        """
        Find similar implementations to a given code chunk across projects.

        Args:
            source_chunk: The code chunk to find similar implementations for
            search_filters: Filters to apply to the search
            max_results: Maximum number of results to return

        Returns:
            CrossProjectSearchResult with similar implementations
        """
        # Create a query from the source chunk
        query = self._create_query_from_chunk(source_chunk)

        # Add structural constraints based on the source chunk
        enhanced_filters = self._enhance_filters_from_chunk(source_chunk, search_filters)

        return await self.search_across_projects(query, enhanced_filters, max_results)

    async def search_by_architectural_pattern(
        self,
        pattern_description: str,
        pattern_type: str,  # e.g., "service", "factory", "observer", "mvc"
        search_filters: CrossProjectSearchFilter,
        max_results: int = 15,
    ) -> CrossProjectSearchResult:
        """
        Search for implementations of specific architectural patterns across projects.

        Args:
            pattern_description: Description of the pattern to find
            pattern_type: Type of architectural pattern
            search_filters: Filters to apply to the search
            max_results: Maximum number of results to return

        Returns:
            CrossProjectSearchResult with pattern implementations
        """
        # Enhance query with pattern-specific terms
        enhanced_query = f"{pattern_type} pattern: {pattern_description}"

        # Adjust filters for pattern search
        pattern_filters = self._create_pattern_search_filters(pattern_type, search_filters)

        return await self.search_across_projects(enhanced_query, pattern_filters, max_results)

    async def _get_available_projects(self) -> list[str]:
        """Get list of available projects in the vector database."""
        try:
            # Get all collection names from Qdrant
            collections_info = await self.qdrant_service.client.get_collections()

            # Extract project names from collection names
            # Collections follow pattern: project_{name}_code, project_{name}_config, project_{name}_documentation
            project_names = set()
            for collection in collections_info.collections:
                if collection.name.startswith("project_") and collection.name.endswith("_code"):
                    # Extract project name: project_Name_code -> Name
                    project_name = collection.name[8:-5]  # Remove "project_" prefix and "_code" suffix
                    project_names.add(project_name)

            return list(project_names)

        except Exception as e:
            self.logger.error(f"Error getting available projects: {e}")
            return []

    def _filter_target_projects(self, available_projects: list[str], filters: CrossProjectSearchFilter) -> list[str]:
        """Filter available projects based on search filters."""
        if filters.target_projects:
            # Use specific target projects if specified
            target_projects = [p for p in filters.target_projects if p in available_projects]
        else:
            # Use all available projects
            target_projects = available_projects.copy()

        # Exclude projects if specified
        if filters.exclude_projects:
            target_projects = [p for p in target_projects if p not in filters.exclude_projects]

        return target_projects

    async def _generate_query_embedding(self, query: str) -> list[float]:
        """Generate embedding for the search query."""
        try:
            # Use embedding service to generate query embedding
            query_embedding_tensor = await self.embedding_service.generate_embeddings("nomic-embed-text", query)
            if query_embedding_tensor is None:
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
            return query_embedding

        except Exception as e:
            self.logger.error(f"Error generating query embedding: {e}")
            return []

    async def _search_project_semantically(
        self,
        project_name: str,
        query_embedding: list[float],
        filters: CrossProjectSearchFilter,
        max_results: int,
    ) -> list[tuple[CodeChunk, float]]:
        """Search a specific project using semantic similarity."""
        try:
            collection_name = f"project_{project_name}_code"

            # Build Qdrant filter based on search filters
            qdrant_filter = self._build_qdrant_filter(filters)

            # Perform vector search
            search_results = await self.qdrant_service.search_vectors(
                collection_name=collection_name,
                query_vector=query_embedding,
                limit=max_results,
                filter=qdrant_filter,
            )

            # Convert results to CodeChunk objects with similarity scores
            matches = []
            for result in search_results:
                chunk = self._create_code_chunk_from_payload(result.payload, project_name)
                similarity_score = result.score
                matches.append((chunk, similarity_score))

            return matches

        except Exception as e:
            self.logger.error(f"Error searching project {project_name}: {e}")
            return []

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

        # Filter by breadcrumb depth (if breadcrumb field exists)
        if filters.min_depth > 0 or filters.max_depth < 10:
            # This would require a custom field for breadcrumb depth
            # For now, we'll skip this filter in Qdrant and apply it post-search
            pass

        if not conditions:
            return None

        return {"must": conditions} if len(conditions) > 1 else conditions[0]

    def _create_code_chunk_from_payload(self, payload: dict, project_name: str) -> CodeChunk:
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

    async def _enhance_with_structural_context(
        self,
        semantic_matches: list[tuple[CodeChunk, float]],
        filters: CrossProjectSearchFilter,
        project_name: str,
    ) -> list[CrossProjectMatch]:
        """Enhance semantic matches with structural context using Graph RAG."""
        enhanced_matches = []

        for chunk, similarity_score in semantic_matches:
            try:
                # Get structural context using Graph RAG service
                structural_score = await self._calculate_structural_score(chunk, filters, project_name)

                # Calculate combined score
                semantic_weight = 1.0 - filters.structural_weight
                combined_score = semantic_weight * similarity_score + filters.structural_weight * structural_score

                # Get related components for context
                related_components = await self._get_related_components(chunk, project_name)

                # Create enhanced match
                enhanced_match = CrossProjectMatch(
                    chunk=chunk,
                    project_name=project_name,
                    similarity_score=similarity_score,
                    structural_score=structural_score,
                    combined_score=combined_score,
                    related_components=related_components,
                    architectural_context=await self._get_architectural_context(chunk, project_name),
                    usage_patterns=self._extract_usage_patterns(chunk),
                )

                enhanced_matches.append(enhanced_match)

            except Exception as e:
                self.logger.error(f"Error enhancing match with structural context: {e}")
                # Add match with minimal structural info
                enhanced_match = CrossProjectMatch(
                    chunk=chunk,
                    project_name=project_name,
                    similarity_score=similarity_score,
                    structural_score=0.5,  # Default structural score
                    combined_score=similarity_score,
                )
                enhanced_matches.append(enhanced_match)

        return enhanced_matches

    async def _calculate_structural_score(
        self,
        chunk: CodeChunk,
        filters: CrossProjectSearchFilter,
        project_name: str,
    ) -> float:
        """Calculate structural relevance score for a chunk."""
        try:
            score = 0.0
            factors = 0

            # Factor 1: Breadcrumb depth alignment
            if chunk.breadcrumb and filters.min_depth <= chunk.breadcrumb_depth <= filters.max_depth:
                score += 1.0
                factors += 1
            elif chunk.breadcrumb:
                # Penalize if outside desired depth range
                depth_penalty = abs(chunk.breadcrumb_depth - (filters.min_depth + filters.max_depth) / 2) / 10.0
                score += max(0.0, 1.0 - depth_penalty)
                factors += 1

            # Factor 2: Chunk type relevance
            if not filters.chunk_types or chunk.chunk_type in filters.chunk_types:
                score += 1.0
                factors += 1

            # Factor 3: Parent-child relationship quality
            if chunk.parent_name and chunk.breadcrumb:
                score += 0.8  # Has good hierarchical context
                factors += 1
            elif chunk.parent_name or chunk.breadcrumb:
                score += 0.5  # Has some hierarchical context
                factors += 1

            # Factor 4: Language preference
            if not filters.languages or chunk.language in filters.languages:
                score += 1.0
                factors += 1

            return score / max(factors, 1) if factors > 0 else 0.5

        except Exception as e:
            self.logger.error(f"Error calculating structural score: {e}")
            return 0.5

    async def _get_related_components(self, chunk: CodeChunk, project_name: str) -> list[GraphNode]:
        """Get related components for a chunk using Graph RAG traversal."""
        try:
            if not chunk.breadcrumb:
                return []

            # Use Graph RAG service to find related components
            traversal_result = await self.graph_rag_service.traverse_from_breadcrumb(
                breadcrumb=chunk.breadcrumb,
                project_name=project_name,
                max_depth=2,  # Limit depth for performance
                strategy_name="semantic",  # Use semantic traversal
            )

            return traversal_result.related_components if traversal_result else []

        except Exception as e:
            self.logger.error(f"Error getting related components: {e}")
            return []

    async def _get_architectural_context(self, chunk: CodeChunk, project_name: str) -> dict[str, Any]:
        """Get architectural context for a chunk."""
        try:
            context = {
                "project": project_name,
                "file_type": chunk.file_path.split(".")[-1] if "." in chunk.file_path else "unknown",
                "directory_structure": "/".join(chunk.file_path.split("/")[:-1]) if "/" in chunk.file_path else "",
                "complexity_indicators": {
                    "has_docstring": bool(chunk.docstring),
                    "has_parent": bool(chunk.parent_name),
                    "has_breadcrumb": bool(chunk.breadcrumb),
                    "line_count": chunk.line_count,
                    "dependency_count": len(chunk.dependencies) if chunk.dependencies else 0,
                    "import_count": len(chunk.imports_used) if chunk.imports_used else 0,
                },
            }

            # Add hierarchical context if available
            if chunk.breadcrumb:
                context["hierarchical_depth"] = chunk.breadcrumb_depth
                context["breadcrumb_components"] = chunk.get_breadcrumb_components()

            return context

        except Exception as e:
            self.logger.error(f"Error getting architectural context: {e}")
            return {"project": project_name}

    def _extract_usage_patterns(self, chunk: CodeChunk) -> list[str]:
        """Extract usage patterns from a code chunk."""
        patterns = []

        try:
            # Pattern 1: Chunk type patterns
            patterns.append(f"{chunk.chunk_type.value}_implementation")

            # Pattern 2: Access modifier patterns
            if chunk.access_modifier:
                patterns.append(f"{chunk.access_modifier}_access")

            # Pattern 3: Documentation patterns
            if chunk.docstring:
                patterns.append("documented_code")

            # Pattern 4: Hierarchical patterns
            if chunk.parent_name:
                patterns.append("nested_structure")

            # Pattern 5: Complexity patterns
            if chunk.line_count > 50:
                patterns.append("complex_implementation")
            elif chunk.line_count < 10:
                patterns.append("simple_implementation")

            # Pattern 6: Language-specific patterns
            patterns.append(f"{chunk.language}_code")

            return patterns

        except Exception as e:
            self.logger.error(f"Error extracting usage patterns: {e}")
            return ["unknown_pattern"]

    def _rank_and_filter_matches(
        self,
        all_matches: list[CrossProjectMatch],
        filters: CrossProjectSearchFilter,
        max_results: int,
    ) -> list[CrossProjectMatch]:
        """Rank and filter matches based on combined scores and filters."""
        try:
            # Filter by similarity threshold
            filtered_matches = [match for match in all_matches if match.combined_score >= filters.similarity_threshold]

            # Sort by combined score (descending)
            sorted_matches = sorted(filtered_matches, key=lambda m: m.combined_score, reverse=True)

            # Apply result limit
            return sorted_matches[:max_results]

        except Exception as e:
            self.logger.error(f"Error ranking and filtering matches: {e}")
            return all_matches[:max_results]

    def _create_query_from_chunk(self, chunk: CodeChunk) -> str:
        """Create a search query from a code chunk."""
        query_parts = []

        # Add chunk name and type
        if chunk.name:
            query_parts.append(chunk.name)
        query_parts.append(chunk.chunk_type.value)

        # Add signature if available
        if chunk.signature:
            query_parts.append(chunk.signature)

        # Add docstring excerpt if available
        if chunk.docstring:
            # Take first line of docstring
            first_line = chunk.docstring.split("\n")[0].strip()
            if first_line:
                query_parts.append(first_line)

        # Add language
        query_parts.append(f"{chunk.language} code")

        return " ".join(query_parts)

    def _enhance_filters_from_chunk(
        self,
        chunk: CodeChunk,
        filters: CrossProjectSearchFilter,
    ) -> CrossProjectSearchFilter:
        """Enhance search filters based on a source chunk."""
        enhanced_filters = CrossProjectSearchFilter(
            target_projects=filters.target_projects.copy(),
            exclude_projects=filters.exclude_projects.copy(),
            chunk_types=filters.chunk_types.copy() if filters.chunk_types else [chunk.chunk_type],
            min_depth=filters.min_depth,
            max_depth=filters.max_depth,
            similarity_threshold=filters.similarity_threshold,
            structural_weight=filters.structural_weight,
            languages=filters.languages.copy() if filters.languages else [chunk.language],
            exclude_languages=filters.exclude_languages.copy(),
        )

        # Adjust depth filters based on source chunk
        if chunk.breadcrumb and not filters.chunk_types:
            chunk_depth = chunk.breadcrumb_depth
            enhanced_filters.min_depth = max(0, chunk_depth - 2)
            enhanced_filters.max_depth = min(10, chunk_depth + 2)

        return enhanced_filters

    def _create_pattern_search_filters(
        self,
        pattern_type: str,
        base_filters: CrossProjectSearchFilter,
    ) -> CrossProjectSearchFilter:
        """Create specialized filters for architectural pattern search."""
        pattern_filters = CrossProjectSearchFilter(
            target_projects=base_filters.target_projects.copy(),
            exclude_projects=base_filters.exclude_projects.copy(),
            chunk_types=base_filters.chunk_types.copy(),
            min_depth=base_filters.min_depth,
            max_depth=base_filters.max_depth,
            similarity_threshold=base_filters.similarity_threshold * 0.8,  # Lower threshold for patterns
            structural_weight=max(0.6, base_filters.structural_weight),  # Higher structural weight
            languages=base_filters.languages.copy(),
            exclude_languages=base_filters.exclude_languages.copy(),
        )

        # Adjust chunk types based on pattern type
        if pattern_type.lower() in ["service", "factory", "singleton"]:
            pattern_filters.chunk_types = [ChunkType.CLASS, ChunkType.FUNCTION]
        elif pattern_type.lower() in ["observer", "decorator"]:
            pattern_filters.chunk_types = [ChunkType.CLASS, ChunkType.INTERFACE]
        elif pattern_type.lower() in ["mvc", "repository"]:
            pattern_filters.chunk_types = [ChunkType.CLASS]

        return pattern_filters


# Factory function for dependency injection
_cross_project_search_service_instance = None


def get_cross_project_search_service(
    qdrant_service: QdrantService = None,
    embedding_service: EmbeddingService = None,
    graph_rag_service: GraphRAGService = None,
) -> CrossProjectSearchService:
    """
    Get or create a CrossProjectSearchService instance.

    Args:
        qdrant_service: Qdrant service instance (optional, will be created if not provided)
        embedding_service: Embedding service instance (optional, will be created if not provided)
        graph_rag_service: Graph RAG service instance (optional, will be created if not provided)

    Returns:
        CrossProjectSearchService instance
    """
    global _cross_project_search_service_instance

    if _cross_project_search_service_instance is None:
        from .embedding_service import EmbeddingService
        from .graph_rag_service import get_graph_rag_service
        from .qdrant_service import QdrantService

        if qdrant_service is None:
            qdrant_service = QdrantService()
        if embedding_service is None:
            embedding_service = EmbeddingService()
        if graph_rag_service is None:
            graph_rag_service = get_graph_rag_service()

        _cross_project_search_service_instance = CrossProjectSearchService(
            qdrant_service=qdrant_service,
            embedding_service=embedding_service,
            graph_rag_service=graph_rag_service,
        )

    return _cross_project_search_service_instance
