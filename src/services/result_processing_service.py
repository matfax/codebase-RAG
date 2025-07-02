"""
Result Processing Service for formatting and enriching search results.

This service provides centralized processing of search results, including
formatting, enrichment, aggregation, and presentation optimization.
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)


class ResultFormat(Enum):
    """Output formats for search results."""

    DETAILED = "detailed"
    SUMMARY = "summary"
    COMPACT = "compact"
    JSON = "json"
    MARKDOWN = "markdown"
    HIERARCHICAL = "hierarchical"


class SortOrder(Enum):
    """Sort orders for result ordering."""

    RELEVANCE = "relevance"
    SCORE = "score"
    ALPHABETICAL = "alphabetical"
    FILE_PATH = "file_path"
    RECENT = "recent"
    IMPORTANCE = "importance"
    CHUNK_TYPE = "chunk_type"


@dataclass
class ProcessingOptions:
    """Options for configuring result processing."""

    format: ResultFormat = ResultFormat.DETAILED
    sort_order: SortOrder = SortOrder.RELEVANCE
    max_results: int = 50
    include_context: bool = True
    include_breadcrumbs: bool = True
    include_metadata: bool = True
    include_code_snippets: bool = True
    snippet_length: int = 200
    context_lines: int = 3
    highlight_query_terms: bool = True
    deduplicate_results: bool = True
    group_by_file: bool = False
    group_by_type: bool = False
    apply_confidence_threshold: float = 0.0
    enhance_with_stats: bool = False


@dataclass
class ProcessedResult:
    """A processed and enriched search result."""

    # Core content
    content: str
    file_path: str
    chunk_id: str
    chunk_name: str = ""
    chunk_type: str = "unknown"

    # Scoring and ranking
    score: float = 0.0
    confidence: float = 0.0
    relevance_rank: int = 0

    # Context and location
    line_start: int = 0
    line_end: int = 0
    breadcrumb: str = ""
    context_before: str = ""
    context_after: str = ""

    # Metadata
    language: str = "unknown"
    file_size: int = 0
    file_modified: datetime | None = None
    signature: str = ""
    docstring: str = ""

    # Processing artifacts
    highlighted_content: str = ""
    content_snippet: str = ""
    keyword_matches: list[str] = field(default_factory=list)
    relevance_indicators: list[str] = field(default_factory=list)

    # Classification
    component_type: str = ""
    importance_level: str = "medium"

    # Raw data
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "content": self.content,
            "file_path": self.file_path,
            "chunk_id": self.chunk_id,
            "chunk_name": self.chunk_name,
            "chunk_type": self.chunk_type,
            "score": self.score,
            "confidence": self.confidence,
            "relevance_rank": self.relevance_rank,
            "line_range": f"{self.line_start}-{self.line_end}",
            "breadcrumb": self.breadcrumb,
            "language": self.language,
            "signature": self.signature,
            "docstring": self.docstring,
            "highlighted_content": self.highlighted_content,
            "content_snippet": self.content_snippet,
            "keyword_matches": self.keyword_matches,
            "relevance_indicators": self.relevance_indicators,
            "component_type": self.component_type,
            "importance_level": self.importance_level,
            "metadata": self.raw_metadata,
        }


@dataclass
class ProcessingStats:
    """Statistics about result processing."""

    total_results: int = 0
    processed_results: int = 0
    filtered_results: int = 0
    duplicates_removed: int = 0
    average_score: float = 0.0
    highest_score: float = 0.0
    languages_found: set[str] = field(default_factory=set)
    chunk_types_found: set[str] = field(default_factory=set)
    files_covered: set[str] = field(default_factory=set)
    processing_time_ms: float = 0.0


class ResultProcessingService:
    """
    Service for processing and formatting search results.

    This service takes raw search results and transforms them into
    well-formatted, enriched results suitable for different use cases.
    """

    def __init__(self):
        """Initialize the result processing service."""
        self.logger = logger

        # Processing configuration
        self.default_snippet_length = 200
        self.default_context_lines = 3
        self.max_breadcrumb_length = 100

        # Patterns for highlighting and extraction
        self.code_patterns = {
            "function_def": r"(def\s+\w+\s*\([^)]*\))",
            "class_def": r"(class\s+\w+\s*\([^)]*\)?)",
            "import_stmt": r"(import\s+\w+|from\s+\w+\s+import\s+\w+)",
            "variable_assignment": r"(\w+\s*=\s*[^=])",
        }

        # Stop words for keyword extraction
        self.stop_words = {
            "the",
            "is",
            "at",
            "which",
            "on",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "with",
            "to",
            "for",
            "of",
            "as",
            "by",
            "from",
            "up",
            "about",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "among",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "who",
            "whom",
            "whose",
            "am",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
        }

    def process_results(
        self, raw_results: list[dict[str, Any]], query: str | None = None, options: ProcessingOptions | None = None
    ) -> tuple[list[ProcessedResult], ProcessingStats]:
        """
        Process raw search results into enriched, formatted results.

        Args:
            raw_results: List of raw search result dictionaries
            query: Original search query for highlighting and analysis
            options: Processing configuration options

        Returns:
            Tuple of (processed results, processing statistics)
        """
        import time

        start_time = time.time()

        options = options or ProcessingOptions()
        stats = ProcessingStats(total_results=len(raw_results))

        self.logger.info(f"Processing {len(raw_results)} search results with format: {options.format.value}")

        processed_results = []
        seen_results = set()

        for idx, raw_result in enumerate(raw_results):
            try:
                # Convert raw result to ProcessedResult
                processed = self._convert_raw_result(raw_result, idx)

                # Apply confidence filtering
                if processed.confidence < options.apply_confidence_threshold:
                    stats.filtered_results += 1
                    continue

                # Deduplicate if requested
                if options.deduplicate_results:
                    result_key = (processed.file_path, processed.chunk_id)
                    if result_key in seen_results:
                        stats.duplicates_removed += 1
                        continue
                    seen_results.add(result_key)

                # Enrich the result
                self._enrich_result(processed, query, options)

                # Update statistics
                self._update_stats(stats, processed)

                processed_results.append(processed)

            except Exception as e:
                self.logger.warning(f"Failed to process result {idx}: {e}")
                continue

        # Apply sorting
        processed_results = self._sort_results(processed_results, options.sort_order)

        # Apply result limit
        if options.max_results > 0:
            processed_results = processed_results[: options.max_results]

        # Group results if requested
        if options.group_by_file or options.group_by_type:
            processed_results = self._group_results(processed_results, options)

        # Finalize statistics
        stats.processed_results = len(processed_results)
        stats.processing_time_ms = (time.time() - start_time) * 1000

        if processed_results:
            scores = [r.score for r in processed_results]
            stats.average_score = sum(scores) / len(scores)
            stats.highest_score = max(scores)

        self.logger.info(f"Processing completed: {stats.processed_results} results in {stats.processing_time_ms:.2f}ms")

        return processed_results, stats

    def format_results(
        self,
        processed_results: list[ProcessedResult],
        format_type: ResultFormat = ResultFormat.DETAILED,
        options: dict[str, Any] | None = None,
    ) -> str | list[dict[str, Any]]:
        """
        Format processed results for output.

        Args:
            processed_results: List of processed results
            format_type: Desired output format
            options: Additional formatting options

        Returns:
            Formatted results as string or structured data
        """
        options = options or {}

        if format_type == ResultFormat.JSON:
            return [result.to_dict() for result in processed_results]

        elif format_type == ResultFormat.MARKDOWN:
            return self._format_as_markdown(processed_results, options)

        elif format_type == ResultFormat.SUMMARY:
            return self._format_as_summary(processed_results, options)

        elif format_type == ResultFormat.COMPACT:
            return self._format_as_compact(processed_results, options)

        elif format_type == ResultFormat.HIERARCHICAL:
            return self._format_as_hierarchical(processed_results, options)

        else:  # DETAILED
            return self._format_as_detailed(processed_results, options)

    def aggregate_results_by_file(self, processed_results: list[ProcessedResult]) -> dict[str, list[ProcessedResult]]:
        """
        Aggregate results grouped by file path.

        Args:
            processed_results: List of processed results

        Returns:
            Dictionary mapping file paths to lists of results
        """
        file_groups = {}

        for result in processed_results:
            file_path = result.file_path
            if file_path not in file_groups:
                file_groups[file_path] = []
            file_groups[file_path].append(result)

        # Sort results within each file by line number
        for file_path in file_groups:
            file_groups[file_path].sort(key=lambda x: x.line_start)

        return file_groups

    def aggregate_results_by_type(self, processed_results: list[ProcessedResult]) -> dict[str, list[ProcessedResult]]:
        """
        Aggregate results grouped by chunk type.

        Args:
            processed_results: List of processed results

        Returns:
            Dictionary mapping chunk types to lists of results
        """
        type_groups = {}

        for result in processed_results:
            chunk_type = result.chunk_type or "unknown"
            if chunk_type not in type_groups:
                type_groups[chunk_type] = []
            type_groups[chunk_type].append(result)

        # Sort results within each type by relevance
        for chunk_type in type_groups:
            type_groups[chunk_type].sort(key=lambda x: x.score, reverse=True)

        return type_groups

    def extract_insights(self, processed_results: list[ProcessedResult], stats: ProcessingStats) -> dict[str, Any]:
        """
        Extract insights and patterns from processed results.

        Args:
            processed_results: List of processed results
            stats: Processing statistics

        Returns:
            Dictionary with extracted insights
        """
        insights = {
            "overview": {
                "total_results": len(processed_results),
                "unique_files": len(stats.files_covered),
                "languages_found": list(stats.languages_found),
                "chunk_types": list(stats.chunk_types_found),
                "average_confidence": sum(r.confidence for r in processed_results) / len(processed_results) if processed_results else 0,
            },
            "top_files": self._get_top_files(processed_results),
            "quality_distribution": self._analyze_quality_distribution(processed_results),
            "content_patterns": self._analyze_content_patterns(processed_results),
            "relevance_indicators": self._aggregate_relevance_indicators(processed_results),
        }

        return insights

    def _convert_raw_result(self, raw_result: dict[str, Any], rank: int) -> ProcessedResult:
        """Convert raw search result to ProcessedResult."""
        # Extract basic fields with fallbacks
        content = raw_result.get("content", "")
        file_path = raw_result.get("file_path", "")
        chunk_id = raw_result.get("id", raw_result.get("chunk_id", ""))

        # Extract metadata
        metadata = raw_result.get("metadata", {})

        processed = ProcessedResult(
            content=content,
            file_path=file_path,
            chunk_id=chunk_id,
            chunk_name=raw_result.get("name", metadata.get("name", "")),
            chunk_type=raw_result.get("chunk_type", metadata.get("chunk_type", "unknown")),
            score=float(raw_result.get("score", 0.0)),
            confidence=float(raw_result.get("confidence", raw_result.get("score", 0.0))),
            relevance_rank=rank + 1,
            line_start=int(raw_result.get("line_start", metadata.get("line_start", 0))),
            line_end=int(raw_result.get("line_end", metadata.get("line_end", 0))),
            breadcrumb=raw_result.get("breadcrumb", ""),
            language=raw_result.get("language", metadata.get("language", "unknown")),
            signature=raw_result.get("signature", metadata.get("signature", "")),
            docstring=raw_result.get("docstring", metadata.get("docstring", "")),
            raw_metadata=metadata,
        )

        return processed

    def _enrich_result(self, result: ProcessedResult, query: str | None, options: ProcessingOptions) -> None:
        """Enrich a processed result with additional information."""

        # Generate content snippet
        if options.include_code_snippets:
            result.content_snippet = self._generate_snippet(result.content, options.snippet_length)

        # Extract and highlight keyword matches
        if query and options.highlight_query_terms:
            result.keyword_matches = self._extract_keyword_matches(result.content, query)
            result.highlighted_content = self._highlight_query_terms(result.content, query, options.snippet_length)

        # Generate breadcrumb if missing
        if not result.breadcrumb and options.include_breadcrumbs:
            result.breadcrumb = self._generate_breadcrumb(result)

        # Classify component type
        result.component_type = self._classify_component_type(result)

        # Determine importance level
        result.importance_level = self._determine_importance_level(result)

        # Extract relevance indicators
        result.relevance_indicators = self._extract_relevance_indicators(result, query)

    def _generate_snippet(self, content: str, max_length: int) -> str:
        """Generate a content snippet of specified length."""
        if len(content) <= max_length:
            return content

        # Try to break at word boundaries
        snippet = content[:max_length]
        last_space = snippet.rfind(" ")

        if last_space > max_length // 2:  # If we can find a reasonable break point
            snippet = snippet[:last_space]

        return snippet + "..." if len(content) > max_length else snippet

    def _extract_keyword_matches(self, content: str, query: str) -> list[str]:
        """Extract keyword matches from content."""
        query_words = [word.lower() for word in re.findall(r"\b\w+\b", query)]
        query_words = [word for word in query_words if word not in self.stop_words and len(word) > 2]

        content_lower = content.lower()
        matches = []

        for word in query_words:
            if word in content_lower:
                matches.append(word)

        return matches

    def _highlight_query_terms(self, content: str, query: str, max_length: int) -> str:
        """Highlight query terms in content."""
        query_words = list(re.findall(r"\b\w+\b", query))
        query_words = [word for word in query_words if word not in self.stop_words and len(word) > 2]

        highlighted = content

        # Simple highlighting - in production, you might want more sophisticated highlighting
        for word in query_words:
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(f"**{word}**", highlighted)

        # Truncate if needed
        if len(highlighted) > max_length:
            highlighted = self._generate_snippet(highlighted, max_length)

        return highlighted

    def _generate_breadcrumb(self, result: ProcessedResult) -> str:
        """Generate breadcrumb navigation for result."""
        parts = []

        # Add file name
        if result.file_path:
            file_name = Path(result.file_path).name
            parts.append(file_name)

        # Add parent context if available
        if result.raw_metadata.get("parent_name"):
            parts.append(result.raw_metadata["parent_name"])

        # Add chunk name
        if result.chunk_name:
            parts.append(result.chunk_name)

        breadcrumb = " > ".join(parts)

        # Truncate if too long
        if len(breadcrumb) > self.max_breadcrumb_length:
            breadcrumb = "..." + breadcrumb[-(self.max_breadcrumb_length - 3) :]

        return breadcrumb

    def _classify_component_type(self, result: ProcessedResult) -> str:
        """Classify the component type of the result."""
        chunk_type = result.chunk_type.lower()
        chunk_name = result.chunk_name.lower()
        file_path = result.file_path.lower()

        # Base classification on chunk type
        if chunk_type == "function":
            if "main" in chunk_name:
                return "Entry Point Function"
            elif "test" in chunk_name or "test" in file_path:
                return "Test Function"
            elif any(prefix in chunk_name for prefix in ["get_", "set_", "is_", "has_"]):
                return "Accessor Function"
            else:
                return "Function"

        elif chunk_type == "class":
            if "service" in chunk_name:
                return "Service Class"
            elif "controller" in chunk_name:
                return "Controller Class"
            elif "model" in chunk_name:
                return "Model Class"
            elif "test" in chunk_name or "test" in file_path:
                return "Test Class"
            else:
                return "Class"

        elif chunk_type == "method":
            return "Method"

        elif chunk_type in ["constant", "variable"]:
            return "Constant/Variable"

        elif chunk_type in ["docstring", "comment"]:
            return "Documentation"

        else:
            return chunk_type.title() if chunk_type else "Code Block"

    def _determine_importance_level(self, result: ProcessedResult) -> str:
        """Determine importance level of the result."""
        score = result.score
        confidence = result.confidence
        chunk_type = result.chunk_type.lower()

        # Calculate importance based on multiple factors
        importance_score = 0

        # Base score
        importance_score += score * 0.4
        importance_score += confidence * 0.3

        # Chunk type bonus
        important_types = ["function", "class", "method"]
        if chunk_type in important_types:
            importance_score += 0.2

        # Name patterns that indicate importance
        chunk_name = result.chunk_name.lower()
        important_patterns = ["main", "core", "primary", "key", "important", "critical"]
        if any(pattern in chunk_name for pattern in important_patterns):
            importance_score += 0.1

        # File location patterns
        file_path = result.file_path.lower()
        if any(pattern in file_path for pattern in ["src/", "core/", "main/"]):
            importance_score += 0.1

        # Determine level
        if importance_score >= 0.8:
            return "high"
        elif importance_score >= 0.5:
            return "medium"
        else:
            return "low"

    def _extract_relevance_indicators(self, result: ProcessedResult, query: str | None) -> list[str]:
        """Extract relevance indicators for the result."""
        indicators = []

        # Score-based indicators
        if result.score > 0.8:
            indicators.append("High semantic relevance")
        elif result.score > 0.6:
            indicators.append("Good semantic relevance")

        # Chunk type indicators
        if result.chunk_type:
            indicators.append(f"Code {result.chunk_type}")

        # Language indicator
        if result.language and result.language != "unknown":
            indicators.append(f"{result.language.title()} code")

        # Documentation indicator
        if result.docstring:
            indicators.append("Well documented")

        # Keyword match indicators
        if query and result.keyword_matches:
            match_count = len(result.keyword_matches)
            if match_count > 2:
                indicators.append(f"Multiple keyword matches ({match_count})")
            elif match_count > 0:
                indicators.append("Keyword match")

        # Importance indicator
        if result.importance_level == "high":
            indicators.append("High importance")

        return indicators

    def _sort_results(self, results: list[ProcessedResult], sort_order: SortOrder) -> list[ProcessedResult]:
        """Sort results according to the specified order."""
        if sort_order == SortOrder.RELEVANCE:
            return sorted(results, key=lambda x: (x.score, x.confidence), reverse=True)

        elif sort_order == SortOrder.SCORE:
            return sorted(results, key=lambda x: x.score, reverse=True)

        elif sort_order == SortOrder.ALPHABETICAL:
            return sorted(results, key=lambda x: (x.file_path, x.chunk_name))

        elif sort_order == SortOrder.FILE_PATH:
            return sorted(results, key=lambda x: x.file_path)

        elif sort_order == SortOrder.IMPORTANCE:
            importance_order = {"high": 3, "medium": 2, "low": 1}
            return sorted(results, key=lambda x: importance_order.get(x.importance_level, 0), reverse=True)

        elif sort_order == SortOrder.CHUNK_TYPE:
            return sorted(results, key=lambda x: x.chunk_type)

        else:  # Default to relevance
            return sorted(results, key=lambda x: x.score, reverse=True)

    def _group_results(self, results: list[ProcessedResult], options: ProcessingOptions) -> list[ProcessedResult]:
        """Group results according to the specified grouping options."""
        # For now, just return the results as-is
        # In a full implementation, you might restructure the results into groups
        return results

    def _update_stats(self, stats: ProcessingStats, result: ProcessedResult) -> None:
        """Update processing statistics with a new result."""
        stats.languages_found.add(result.language)
        stats.chunk_types_found.add(result.chunk_type)
        stats.files_covered.add(result.file_path)

    def _format_as_markdown(self, results: list[ProcessedResult], options: dict[str, Any]) -> str:
        """Format results as Markdown."""
        output = ["# Search Results\n"]

        for i, result in enumerate(results, 1):
            output.append(f"## {i}. {result.chunk_name or 'Unnamed'}")
            output.append(f"**File:** `{result.file_path}` (lines {result.line_start}-{result.line_end})")
            output.append(f"**Type:** {result.component_type}")
            output.append(f"**Score:** {result.score:.3f}")

            if result.signature:
                output.append(f"**Signature:** `{result.signature}`")

            if result.breadcrumb:
                output.append(f"**Location:** {result.breadcrumb}")

            if result.docstring:
                output.append(f"**Description:** {result.docstring[:200]}...")

            if result.highlighted_content:
                output.append("**Content:**")
                output.append(f"```{result.language}")
                output.append(result.highlighted_content)
                output.append("```")

            if result.relevance_indicators:
                output.append(f"**Relevance:** {', '.join(result.relevance_indicators)}")

            output.append("")  # Empty line between results

        return "\n".join(output)

    def _format_as_summary(self, results: list[ProcessedResult], options: dict[str, Any]) -> str:
        """Format results as summary."""
        if not results:
            return "No results found."

        output = [f"Found {len(results)} results:\n"]

        for i, result in enumerate(results, 1):
            file_name = Path(result.file_path).name
            line_info = f":{result.line_start}" if result.line_start else ""

            summary_line = f"{i}. {result.chunk_name or 'Unnamed'} in {file_name}{line_info} (score: {result.score:.3f})"
            output.append(summary_line)

        return "\n".join(output)

    def _format_as_compact(self, results: list[ProcessedResult], options: dict[str, Any]) -> str:
        """Format results in compact format."""
        output = []

        for result in results:
            file_name = Path(result.file_path).name
            compact_line = f"{result.chunk_name or 'Unnamed'} | {file_name}:{result.line_start} | {result.score:.3f}"
            output.append(compact_line)

        return "\n".join(output)

    def _format_as_detailed(self, results: list[ProcessedResult], options: dict[str, Any]) -> str:
        """Format results with full details."""
        output = ["=== DETAILED SEARCH RESULTS ===\n"]

        for i, result in enumerate(results, 1):
            output.append(f"Result #{i}")
            output.append("-" * 40)
            output.append(f"Name: {result.chunk_name or 'Unnamed'}")
            output.append(f"File: {result.file_path}")
            output.append(f"Lines: {result.line_start}-{result.line_end}")
            output.append(f"Type: {result.component_type}")
            output.append(f"Language: {result.language}")
            output.append(f"Score: {result.score:.3f}")
            output.append(f"Confidence: {result.confidence:.3f}")
            output.append(f"Importance: {result.importance_level}")

            if result.signature:
                output.append(f"Signature: {result.signature}")

            if result.breadcrumb:
                output.append(f"Breadcrumb: {result.breadcrumb}")

            if result.keyword_matches:
                output.append(f"Keyword matches: {', '.join(result.keyword_matches)}")

            if result.relevance_indicators:
                output.append(f"Relevance: {', '.join(result.relevance_indicators)}")

            if result.docstring:
                output.append(f"Documentation: {result.docstring[:150]}...")

            if result.content_snippet:
                output.append("Content snippet:")
                output.append(result.content_snippet)

            output.append("")  # Empty line between results

        return "\n".join(output)

    def _format_as_hierarchical(self, results: list[ProcessedResult], options: dict[str, Any]) -> str:
        """Format results in hierarchical structure."""
        # Group by file
        file_groups = self.aggregate_results_by_file(results)

        output = ["=== HIERARCHICAL RESULTS ===\n"]

        for file_path, file_results in file_groups.items():
            output.append(f"📁 {file_path}")

            for result in file_results:
                indent = "  "
                output.append(f"{indent}📄 {result.chunk_name or 'Unnamed'} ({result.chunk_type})")
                output.append(f"{indent}   Score: {result.score:.3f} | Lines: {result.line_start}-{result.line_end}")

                if result.signature:
                    output.append(f"{indent}   Signature: {result.signature}")

            output.append("")  # Empty line between files

        return "\n".join(output)

    def _get_top_files(self, results: list[ProcessedResult]) -> list[dict[str, Any]]:
        """Get top files by result count and average score."""
        file_stats = {}

        for result in results:
            file_path = result.file_path
            if file_path not in file_stats:
                file_stats[file_path] = {"count": 0, "total_score": 0.0, "results": []}

            file_stats[file_path]["count"] += 1
            file_stats[file_path]["total_score"] += result.score
            file_stats[file_path]["results"].append(result)

        # Calculate averages and sort
        top_files = []
        for file_path, stats in file_stats.items():
            avg_score = stats["total_score"] / stats["count"]
            top_files.append(
                {
                    "file_path": file_path,
                    "result_count": stats["count"],
                    "average_score": avg_score,
                    "total_score": stats["total_score"],
                }
            )

        return sorted(top_files, key=lambda x: (x["result_count"], x["average_score"]), reverse=True)[:10]

    def _analyze_quality_distribution(self, results: list[ProcessedResult]) -> dict[str, int]:
        """Analyze quality distribution of results."""
        distribution = {"high": 0, "medium": 0, "low": 0}

        for result in results:
            if result.score > 0.8:
                distribution["high"] += 1
            elif result.score > 0.5:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution

    def _analyze_content_patterns(self, results: list[ProcessedResult]) -> dict[str, int]:
        """Analyze common content patterns in results."""
        patterns = {}

        for result in results:
            content = result.content.lower()

            # Look for common code patterns
            for pattern_name, pattern_regex in self.code_patterns.items():
                if re.search(pattern_regex, content):
                    patterns[pattern_name] = patterns.get(pattern_name, 0) + 1

        return patterns

    def _aggregate_relevance_indicators(self, results: list[ProcessedResult]) -> dict[str, int]:
        """Aggregate relevance indicators across all results."""
        indicator_counts = {}

        for result in results:
            for indicator in result.relevance_indicators:
                indicator_counts[indicator] = indicator_counts.get(indicator, 0) + 1

        return dict(sorted(indicator_counts.items(), key=lambda x: x[1], reverse=True))
