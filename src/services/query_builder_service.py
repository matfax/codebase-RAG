"""
Query Builder Service for constructing Qdrant search queries.

This service provides a centralized way to build and customize Qdrant vector search queries
for different search scenarios and requirements.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from qdrant_client.http import models as qdrant_models

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries for different search scenarios."""

    SIMPLE_VECTOR = "simple_vector"
    FILTERED_VECTOR = "filtered_vector"
    HYBRID_SEARCH = "hybrid_search"
    SIMILARITY_SEARCH = "similarity_search"
    METADATA_FILTER = "metadata_filter"
    RANGE_SEARCH = "range_search"
    COMPOUND_QUERY = "compound_query"


class FilterOperator(Enum):
    """Filter operators for metadata filtering."""

    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    IN = "in"
    NOT_IN = "not_in"
    GREATER_THAN = "gt"
    GREATER_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_EQUAL = "lte"
    RANGE = "range"
    MATCH = "match"
    REGEX = "regex"


@dataclass
class QueryFilter:
    """Represents a filter condition for search queries."""

    field: str
    operator: FilterOperator
    value: Any
    boost: float = 1.0


@dataclass
class SearchContext:
    """Context information for building search queries."""

    collection_name: str
    query_vector: list[float] | None = None
    query_text: str | None = None
    search_mode: str = "hybrid"
    similarity_threshold: float = 0.7
    max_results: int = 10
    include_metadata: bool = True
    include_vectors: bool = False
    offset: int = 0


@dataclass
class QueryParameters:
    """Parameters for customizing query behavior."""

    filters: list[QueryFilter] = field(default_factory=list)
    boost_conditions: dict[str, float] = field(default_factory=dict)
    score_threshold: float | None = None
    must_conditions: list[dict[str, Any]] = field(default_factory=list)
    should_conditions: list[dict[str, Any]] = field(default_factory=list)
    must_not_conditions: list[dict[str, Any]] = field(default_factory=list)
    facet_fields: list[str] = field(default_factory=list)
    grouping_field: str | None = None


class QueryBuilderService:
    """
    Service for building sophisticated Qdrant search queries.

    This service provides methods to construct various types of vector search queries
    with filtering, boosting, and metadata conditions for different search scenarios.
    """

    def __init__(self):
        """Initialize the query builder service."""
        self.logger = logger

        # Default query configurations
        self.default_score_threshold = 0.5
        self.default_limit = 10
        self.default_offset = 0

        # Prebuilt filter conditions for common scenarios
        self.common_filters = {
            "code_only": QueryFilter("chunk_type", FilterOperator.IN, ["function", "class", "method"]),
            "documentation_only": QueryFilter("chunk_type", FilterOperator.IN, ["docstring", "comment", "markdown"]),
            "recent_files": QueryFilter("file_modified", FilterOperator.GREATER_THAN, "2024-01-01"),
            "python_only": QueryFilter("language", FilterOperator.EQUALS, "python"),
            "javascript_only": QueryFilter("language", FilterOperator.EQUALS, "javascript"),
            "typescript_only": QueryFilter("language", FilterOperator.EQUALS, "typescript"),
            "main_files": QueryFilter("file_path", FilterOperator.REGEX, r".*(Union[main, index]|app)\.(Union[py, js]|ts)$"),
            "test_files": QueryFilter("file_path", FilterOperator.REGEX, r".*(Union[test, spec]).*\.(Union[py, js]|ts)$"),
            "high_importance": QueryFilter("importance_score", FilterOperator.GREATER_EQUAL, 0.7),
        }

    def build_simple_vector_query(self, context: SearchContext, parameters: QueryParameters | None = None) -> qdrant_models.SearchRequest:
        """
        Build a simple vector similarity search query.

        Args:
            context: Search context with query vector and parameters
            parameters: Additional query parameters and filters

        Returns:
            Qdrant search request object
        """
        if not context.query_vector:
            raise ValueError("Query vector is required for vector search")

        parameters = parameters or QueryParameters()

        # Build basic query
        search_request = qdrant_models.SearchRequest(
            vector=context.query_vector,
            limit=context.max_results,
            offset=context.offset,
            score_threshold=parameters.score_threshold or context.similarity_threshold,
            with_payload=context.include_metadata,
            with_vector=context.include_vectors,
        )

        # Add filters if specified
        if parameters.filters:
            search_request.filter = self._build_filter_conditions(parameters.filters)

        self.logger.debug(f"Built simple vector query for collection: {context.collection_name}")
        return search_request

    def build_filtered_vector_query(self, context: SearchContext, parameters: QueryParameters) -> qdrant_models.SearchRequest:
        """
        Build a vector search query with advanced filtering.

        Args:
            context: Search context with query vector and parameters
            parameters: Query parameters with filters and conditions

        Returns:
            Qdrant search request with filtering
        """
        if not context.query_vector:
            raise ValueError("Query vector is required for filtered vector search")

        # Start with basic vector query
        search_request = self.build_simple_vector_query(context, parameters)

        # Add advanced filtering
        filter_conditions = []

        # Add must conditions
        if parameters.must_conditions:
            filter_conditions.extend(parameters.must_conditions)

        # Add filter conditions
        if parameters.filters:
            for query_filter in parameters.filters:
                condition = self._create_filter_condition(query_filter)
                if condition:
                    filter_conditions.append(condition)

        # Combine filters
        if filter_conditions:
            if len(filter_conditions) == 1:
                search_request.filter = qdrant_models.Filter(must=[filter_conditions[0]])
            else:
                search_request.filter = qdrant_models.Filter(must=filter_conditions)

        # Add should conditions (optional boost conditions)
        if parameters.should_conditions:
            if search_request.filter:
                search_request.filter.should = parameters.should_conditions
            else:
                search_request.filter = qdrant_models.Filter(should=parameters.should_conditions)

        # Add must_not conditions (exclusions)
        if parameters.must_not_conditions:
            if search_request.filter:
                search_request.filter.must_not = parameters.must_not_conditions
            else:
                search_request.filter = qdrant_models.Filter(must_not=parameters.must_not_conditions)

        self.logger.debug(f"Built filtered vector query with {len(filter_conditions)} filter conditions")
        return search_request

    def build_hybrid_query(
        self, context: SearchContext, parameters: QueryParameters | None = None, text_weight: float = 0.5, vector_weight: float = 0.5
    ) -> qdrant_models.SearchRequest:
        """
        Build a hybrid search query combining vector and text search.

        Args:
            context: Search context with both vector and text query
            parameters: Additional query parameters
            text_weight: Weight for text search component
            vector_weight: Weight for vector search component

        Returns:
            Qdrant search request for hybrid search
        """
        if not context.query_vector:
            raise ValueError("Query vector is required for hybrid search")

        parameters = parameters or QueryParameters()

        # Build base vector query
        search_request = self.build_filtered_vector_query(context, parameters)

        # Add text search component if query text is provided
        if context.query_text:
            # Create text search conditions
            text_conditions = self._build_text_search_conditions(context.query_text, text_weight)

            # Add to should conditions for hybrid scoring
            if search_request.filter and search_request.filter.should:
                search_request.filter.should.extend(text_conditions)
            elif search_request.filter:
                search_request.filter.should = text_conditions
            else:
                search_request.filter = qdrant_models.Filter(should=text_conditions)

        self.logger.debug(f"Built hybrid query with text_weight={text_weight}, vector_weight={vector_weight}")
        return search_request

    def build_similarity_search_query(
        self, context: SearchContext, target_ids: list[str], parameters: QueryParameters | None = None
    ) -> qdrant_models.SearchRequest:
        """
        Build a query to find similar items to specified target IDs.

        Args:
            context: Search context
            target_ids: List of point IDs to find similar items for
            parameters: Additional query parameters

        Returns:
            Qdrant search request for similarity search
        """
        parameters = parameters or QueryParameters()

        # Create similarity conditions
        similarity_conditions = []
        for target_id in target_ids:
            condition = qdrant_models.FieldCondition(key="id", match=qdrant_models.MatchValue(value=target_id))
            similarity_conditions.append(condition)

        # Build base query with similarity targeting
        search_request = qdrant_models.SearchRequest(
            vector=context.query_vector or [0.0] * 768,  # Placeholder if no vector
            limit=context.max_results,
            offset=context.offset,
            score_threshold=parameters.score_threshold or context.similarity_threshold,
            with_payload=context.include_metadata,
            with_vector=context.include_vectors,
        )

        # Add similarity conditions as should conditions
        if similarity_conditions:
            search_request.filter = qdrant_models.Filter(should=similarity_conditions)

        # Add additional filters
        if parameters.filters:
            filter_conditions = [self._create_filter_condition(f) for f in parameters.filters]
            filter_conditions = [c for c in filter_conditions if c]  # Remove None

            if filter_conditions:
                if search_request.filter:
                    search_request.filter.must = filter_conditions
                else:
                    search_request.filter = qdrant_models.Filter(must=filter_conditions)

        self.logger.debug(f"Built similarity search query for {len(target_ids)} target IDs")
        return search_request

    def build_metadata_filter_query(
        self, context: SearchContext, metadata_filters: dict[str, Any], parameters: QueryParameters | None = None
    ) -> qdrant_models.SearchRequest:
        """
        Build a query primarily focused on metadata filtering.

        Args:
            context: Search context
            metadata_filters: Dictionary of metadata field filters
            parameters: Additional query parameters

        Returns:
            Qdrant search request with metadata filtering
        """
        parameters = parameters or QueryParameters()

        # Convert metadata filters to QueryFilter objects
        query_filters = []
        for field_name, value in metadata_filters.items():
            if isinstance(value, list):
                query_filters.append(QueryFilter(field_name, FilterOperator.IN, value))
            elif isinstance(value, dict) and "operator" in value:
                operator = FilterOperator(value["operator"])
                query_filters.append(QueryFilter(field_name, operator, value["value"]))
            else:
                query_filters.append(QueryFilter(field_name, FilterOperator.EQUALS, value))

        # Add to existing filters
        parameters.filters.extend(query_filters)

        # Build filtered query
        if context.query_vector:
            return self.build_filtered_vector_query(context, parameters)
        else:
            # Metadata-only query
            search_request = qdrant_models.SearchRequest(
                vector=[0.0] * 768,  # Placeholder vector
                limit=context.max_results,
                offset=context.offset,
                with_payload=context.include_metadata,
                with_vector=context.include_vectors,
            )

            if parameters.filters:
                search_request.filter = self._build_filter_conditions(parameters.filters)

            return search_request

    def build_range_search_query(
        self, context: SearchContext, range_field: str, min_value: Any, max_value: Any, parameters: QueryParameters | None = None
    ) -> qdrant_models.SearchRequest:
        """
        Build a query for range-based searching.

        Args:
            context: Search context
            range_field: Field to apply range filter on
            min_value: Minimum value for range
            max_value: Maximum value for range
            parameters: Additional query parameters

        Returns:
            Qdrant search request with range filtering
        """
        parameters = parameters or QueryParameters()

        # Add range filter
        range_filter = QueryFilter(range_field, FilterOperator.RANGE, [min_value, max_value])
        parameters.filters.append(range_filter)

        return self.build_filtered_vector_query(context, parameters)

    def build_compound_query(
        self,
        context: SearchContext,
        sub_queries: list[dict[str, Any]],
        combination_mode: str = "should",
        parameters: QueryParameters | None = None,
    ) -> qdrant_models.SearchRequest:
        """
        Build a compound query combining multiple search conditions.

        Args:
            context: Search context
            sub_queries: List of sub-query definitions
            combination_mode: How to combine queries ("must", "should", "must_not")
            parameters: Additional query parameters

        Returns:
            Qdrant search request with compound conditions
        """
        parameters = parameters or QueryParameters()

        # Build individual query conditions
        combined_conditions = []

        for sub_query in sub_queries:
            query_type = sub_query.get("type", "filter")

            if query_type == "filter":
                condition = self._create_filter_condition(
                    QueryFilter(sub_query["field"], FilterOperator(sub_query["operator"]), sub_query["value"])
                )
                if condition:
                    combined_conditions.append(condition)

            elif query_type == "text":
                text_conditions = self._build_text_search_conditions(sub_query["text"], sub_query.get("weight", 1.0))
                combined_conditions.extend(text_conditions)

        # Build base query
        search_request = qdrant_models.SearchRequest(
            vector=context.query_vector or [0.0] * 768,
            limit=context.max_results,
            offset=context.offset,
            score_threshold=parameters.score_threshold or context.similarity_threshold,
            with_payload=context.include_metadata,
            with_vector=context.include_vectors,
        )

        # Apply combination mode
        if combined_conditions:
            if combination_mode == "must":
                search_request.filter = qdrant_models.Filter(must=combined_conditions)
            elif combination_mode == "should":
                search_request.filter = qdrant_models.Filter(should=combined_conditions)
            elif combination_mode == "must_not":
                search_request.filter = qdrant_models.Filter(must_not=combined_conditions)

        self.logger.debug(f"Built compound query with {len(combined_conditions)} conditions in {combination_mode} mode")
        return search_request

    def add_common_filter(self, parameters: QueryParameters, filter_name: str) -> bool:
        """
        Add a predefined common filter to query parameters.

        Args:
            parameters: Query parameters to modify
            filter_name: Name of the common filter to add

        Returns:
            True if filter was added, False if filter name not found
        """
        if filter_name in self.common_filters:
            parameters.filters.append(self.common_filters[filter_name])
            self.logger.debug(f"Added common filter: {filter_name}")
            return True
        else:
            self.logger.warning(f"Common filter not found: {filter_name}")
            return False

    def create_project_scoped_context(self, project_name: str, base_context: SearchContext) -> SearchContext:
        """
        Create a search context scoped to a specific project.

        Args:
            project_name: Name of the project to scope to
            base_context: Base search context to modify

        Returns:
            Modified search context with project scoping
        """
        # Create project-specific collection name
        if not base_context.collection_name.startswith(f"project_{project_name}"):
            scoped_collection = f"project_{project_name}_code"  # Default to code collection
        else:
            scoped_collection = base_context.collection_name

        # Create new context with project scoping
        scoped_context = SearchContext(
            collection_name=scoped_collection,
            query_vector=base_context.query_vector,
            query_text=base_context.query_text,
            search_mode=base_context.search_mode,
            similarity_threshold=base_context.similarity_threshold,
            max_results=base_context.max_results,
            include_metadata=base_context.include_metadata,
            include_vectors=base_context.include_vectors,
            offset=base_context.offset,
        )

        self.logger.debug(f"Created project-scoped context for project: {project_name}")
        return scoped_context

    def optimize_query_for_performance(
        self, search_request: qdrant_models.SearchRequest, performance_mode: str = "balanced"
    ) -> qdrant_models.SearchRequest:
        """
        Optimize query for better performance based on the specified mode.

        Args:
            search_request: Original search request
            performance_mode: Performance optimization mode ("speed", "accuracy", "balanced")

        Returns:
            Optimized search request
        """
        if performance_mode == "speed":
            # Optimize for speed
            search_request.limit = min(search_request.limit, 20)
            search_request.score_threshold = max(search_request.score_threshold or 0.5, 0.7)
            search_request.with_vector = False  # Skip vectors for speed

        elif performance_mode == "accuracy":
            # Optimize for accuracy
            search_request.limit = max(search_request.limit, 50)
            search_request.score_threshold = min(search_request.score_threshold or 0.5, 0.3)
            search_request.with_vector = True  # Include vectors for analysis

        elif performance_mode == "balanced":
            # Balanced optimization
            search_request.limit = min(max(search_request.limit, 10), 30)
            search_request.score_threshold = search_request.score_threshold or 0.5

        self.logger.debug(f"Optimized query for {performance_mode} performance")
        return search_request

    def _build_filter_conditions(self, filters: list[QueryFilter]) -> qdrant_models.Filter:
        """Build Qdrant filter conditions from QueryFilter objects."""
        conditions = []

        for query_filter in filters:
            condition = self._create_filter_condition(query_filter)
            if condition:
                conditions.append(condition)

        return qdrant_models.Filter(must=conditions) if conditions else None

    def _create_filter_condition(self, query_filter: QueryFilter) -> qdrant_models.FieldCondition | None:
        """Create a Qdrant field condition from a QueryFilter."""
        try:
            field_key = query_filter.field
            operator = query_filter.operator
            value = query_filter.value

            if operator == FilterOperator.EQUALS:
                return qdrant_models.FieldCondition(key=field_key, match=qdrant_models.MatchValue(value=value))

            elif operator == FilterOperator.NOT_EQUALS:
                return qdrant_models.FieldCondition(key=field_key, match=qdrant_models.MatchExcept(except_=value))

            elif operator == FilterOperator.IN:
                return qdrant_models.FieldCondition(key=field_key, match=qdrant_models.MatchAny(any=value))

            elif operator == FilterOperator.NOT_IN:
                return qdrant_models.FieldCondition(key=field_key, match=qdrant_models.MatchExcept(except_=value))

            elif operator == FilterOperator.GREATER_THAN:
                return qdrant_models.FieldCondition(key=field_key, range=qdrant_models.Range(gt=value))

            elif operator == FilterOperator.GREATER_EQUAL:
                return qdrant_models.FieldCondition(key=field_key, range=qdrant_models.Range(gte=value))

            elif operator == FilterOperator.LESS_THAN:
                return qdrant_models.FieldCondition(key=field_key, range=qdrant_models.Range(lt=value))

            elif operator == FilterOperator.LESS_EQUAL:
                return qdrant_models.FieldCondition(key=field_key, range=qdrant_models.Range(lte=value))

            elif operator == FilterOperator.RANGE:
                if isinstance(value, Union[list, tuple]) and len(value) == 2:
                    return qdrant_models.FieldCondition(key=field_key, range=qdrant_models.Range(gte=value[0], lte=value[1]))

            elif operator == FilterOperator.MATCH:
                return qdrant_models.FieldCondition(key=field_key, match=qdrant_models.MatchText(text=str(value)))

            elif operator == FilterOperator.REGEX:
                # Note: Qdrant might not support regex directly, using text match as fallback
                return qdrant_models.FieldCondition(key=field_key, match=qdrant_models.MatchText(text=str(value)))

            else:
                self.logger.warning(f"Unsupported filter operator: {operator}")
                return None

        except Exception as e:
            self.logger.error(f"Failed to create filter condition: {e}")
            return None

    def _build_text_search_conditions(self, query_text: str, weight: float = 1.0) -> list[qdrant_models.FieldCondition]:
        """Build text search conditions for hybrid queries."""
        conditions = []

        # Search in content field
        conditions.append(qdrant_models.FieldCondition(key="content", match=qdrant_models.MatchText(text=query_text)))

        # Search in name/title fields if query is short (likely a function/class name)
        if len(query_text.split()) <= 3:
            conditions.append(qdrant_models.FieldCondition(key="name", match=qdrant_models.MatchText(text=query_text)))

        # Search in docstring for documentation queries
        if any(doc_word in query_text.lower() for doc_word in ["what", "how", "why", "explain", "describe"]):
            conditions.append(qdrant_models.FieldCondition(key="docstring", match=qdrant_models.MatchText(text=query_text)))

        return conditions

    def get_query_statistics(self, search_request: qdrant_models.SearchRequest) -> dict[str, Any]:
        """
        Get statistics about a constructed query.

        Args:
            search_request: Qdrant search request to analyze

        Returns:
            Dictionary with query statistics
        """
        stats = {
            "limit": search_request.limit,
            "offset": search_request.offset,
            "score_threshold": search_request.score_threshold,
            "with_payload": search_request.with_payload,
            "with_vector": search_request.with_vector,
            "has_filter": search_request.filter is not None,
            "filter_complexity": 0,
        }

        if search_request.filter:
            filter_obj = search_request.filter
            complexity = 0

            if filter_obj.must:
                complexity += len(filter_obj.must)
            if filter_obj.should:
                complexity += len(filter_obj.should)
            if filter_obj.must_not:
                complexity += len(filter_obj.must_not)

            stats["filter_complexity"] = complexity

        return stats
