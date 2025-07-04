"""Search diagnostics utilities for debugging and monitoring search operations.

This module provides comprehensive diagnostic tools for identifying and resolving
search-related issues in the codebase RAG system.
"""

import logging
import time
from collections import defaultdict
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, MatchValue

logger = logging.getLogger(__name__)


class SearchDiagnostics:
    """Diagnostic tools for search operations."""

    def __init__(self, qdrant_client: QdrantClient):
        self.qdrant_client = qdrant_client
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def analyze_collection_health(self, collection_name: str) -> dict[str, Any]:
        """Analyze the health and content quality of a collection.

        Args:
            collection_name: Name of the collection to analyze

        Returns:
            Dictionary containing health metrics and issues
        """
        try:
            # Get collection info
            collection_info = self.qdrant_client.get_collection(collection_name)

            # Sample points to analyze content quality
            sample_points = self.qdrant_client.scroll(collection_name=collection_name, limit=100, with_payload=True, with_vectors=False)[0]

            analysis = {
                "collection_name": collection_name,
                "total_points": collection_info.points_count,
                "vector_size": collection_info.config.params.vectors.size,
                "distance_metric": collection_info.config.params.vectors.distance.name,
                "content_analysis": self._analyze_content_quality(sample_points),
                "metadata_analysis": self._analyze_metadata_quality(sample_points),
                "issues": [],
                "recommendations": [],
            }

            # Identify issues
            if analysis["content_analysis"]["empty_content_count"] > 0:
                analysis["issues"].append(
                    {
                        "type": "empty_content",
                        "count": analysis["content_analysis"]["empty_content_count"],
                        "description": "Found chunks with empty or missing content",
                    }
                )
                analysis["recommendations"].append("Run content validation and reindex files with empty content")

            if analysis["metadata_analysis"]["missing_file_paths"] > 0:
                analysis["issues"].append(
                    {
                        "type": "missing_metadata",
                        "count": analysis["metadata_analysis"]["missing_file_paths"],
                        "description": "Found chunks with missing file path metadata",
                    }
                )
                analysis["recommendations"].append("Reindex files to ensure all chunks have proper metadata")

            return analysis

        except Exception as e:
            self.logger.error(f"Failed to analyze collection {collection_name}: {e}")
            return {"collection_name": collection_name, "error": str(e), "issues": [{"type": "analysis_error", "description": str(e)}]}

    def _analyze_content_quality(self, points: list[Any]) -> dict[str, Any]:
        """Analyze the quality of content in sample points."""
        empty_content_count = 0
        total_content_length = 0
        content_lengths = []

        for point in points:
            if not point.payload:
                empty_content_count += 1
                continue

            content = point.payload.get("content", "")
            if not content or not content.strip():
                empty_content_count += 1
            else:
                length = len(content)
                total_content_length += length
                content_lengths.append(length)

        return {
            "total_sampled": len(points),
            "empty_content_count": empty_content_count,
            "average_content_length": total_content_length / max(len(content_lengths), 1),
            "min_content_length": min(content_lengths) if content_lengths else 0,
            "max_content_length": max(content_lengths) if content_lengths else 0,
            "valid_content_rate": (len(points) - empty_content_count) / len(points) if points else 0,
        }

    def _analyze_metadata_quality(self, points: list[Any]) -> dict[str, Any]:
        """Analyze the quality of metadata in sample points."""
        missing_file_paths = 0
        missing_line_info = 0
        missing_language = 0
        file_types = defaultdict(int)
        languages = defaultdict(int)

        for point in points:
            if not point.payload:
                missing_file_paths += 1
                missing_line_info += 1
                missing_language += 1
                continue

            # Check file path
            file_path = point.payload.get("file_path", "")
            if not file_path:
                missing_file_paths += 1
            else:
                # Extract file extension
                ext = file_path.split(".")[-1] if "." in file_path else "no_extension"
                file_types[ext] += 1

            # Check line information
            line_start = point.payload.get("line_start", 0)
            line_end = point.payload.get("line_end", 0)
            if not line_start or not line_end:
                missing_line_info += 1

            # Check language
            language = point.payload.get("language", "")
            if not language:
                missing_language += 1
            else:
                languages[language] += 1

        return {
            "total_sampled": len(points),
            "missing_file_paths": missing_file_paths,
            "missing_line_info": missing_line_info,
            "missing_language": missing_language,
            "file_types": dict(file_types),
            "languages": dict(languages),
        }

    def validate_search_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Validate search results for common issues.

        Args:
            results: List of search results to validate

        Returns:
            Dictionary containing validation results and issues
        """
        validation = {"total_results": len(results), "valid_results": 0, "issues": [], "issue_counts": defaultdict(int)}

        for i, result in enumerate(results):
            result_issues = []

            # Check for empty content
            if not result.get("content", "").strip():
                result_issues.append("empty_content")
                validation["issue_counts"]["empty_content"] += 1

            # Check for missing file path
            if not result.get("file_path", "").strip():
                result_issues.append("missing_file_path")
                validation["issue_counts"]["missing_file_path"] += 1

            # Check for missing score
            if "score" not in result or result["score"] is None:
                result_issues.append("missing_score")
                validation["issue_counts"]["missing_score"] += 1

            # Check for invalid line numbers
            line_start = result.get("line_start", 0)
            line_end = result.get("line_end", 0)
            if line_start <= 0 or line_end <= 0 or line_start > line_end:
                result_issues.append("invalid_line_numbers")
                validation["issue_counts"]["invalid_line_numbers"] += 1

            if result_issues:
                validation["issues"].append(
                    {
                        "result_index": i,
                        "issues": result_issues,
                        "file_path": result.get("file_path", "unknown"),
                        "collection": result.get("collection", "unknown"),
                    }
                )
            else:
                validation["valid_results"] += 1

        validation["validity_rate"] = validation["valid_results"] / validation["total_results"] if validation["total_results"] > 0 else 0

        return validation

    def check_vector_database_consistency(self, collection_names: list[str]) -> dict[str, Any]:
        """Check consistency across vector database collections.

        Args:
            collection_names: List of collection names to check

        Returns:
            Dictionary containing consistency check results
        """
        consistency_report = {
            "collections_checked": collection_names,
            "total_collections": len(collection_names),
            "issues": [],
            "recommendations": [],
        }

        collection_stats = {}

        for collection_name in collection_names:
            try:
                collection_info = self.qdrant_client.get_collection(collection_name)
                collection_stats[collection_name] = {
                    "points_count": collection_info.points_count,
                    "vector_size": collection_info.config.params.vectors.size,
                    "distance_metric": collection_info.config.params.vectors.distance.name,
                }
            except Exception as e:
                consistency_report["issues"].append({"type": "collection_access_error", "collection": collection_name, "error": str(e)})
                continue

        # Check for inconsistent vector sizes
        vector_sizes = {name: stats["vector_size"] for name, stats in collection_stats.items()}
        unique_sizes = set(vector_sizes.values())

        if len(unique_sizes) > 1:
            consistency_report["issues"].append(
                {
                    "type": "inconsistent_vector_sizes",
                    "description": "Collections have different vector dimensions",
                    "details": vector_sizes,
                }
            )
            consistency_report["recommendations"].append("Ensure all collections use the same embedding model and dimension")

        # Check for empty collections
        empty_collections = [name for name, stats in collection_stats.items() if stats["points_count"] == 0]
        if empty_collections:
            consistency_report["issues"].append(
                {"type": "empty_collections", "collections": empty_collections, "description": "Found collections with no indexed content"}
            )
            consistency_report["recommendations"].append("Remove empty collections or reindex missing content")

        consistency_report["collection_stats"] = collection_stats

        return consistency_report

    def measure_search_performance(
        self, query: str, query_embedding: list[float], collection_names: list[str], n_results: int = 10
    ) -> dict[str, Any]:
        """Measure search performance across collections.

        Args:
            query: Search query string
            query_embedding: Query embedding vector
            collection_names: List of collections to search
            n_results: Number of results to retrieve per collection

        Returns:
            Dictionary containing performance metrics
        """
        performance_metrics = {
            "query": query,
            "collections_tested": collection_names,
            "n_results": n_results,
            "collection_performance": {},
            "total_time": 0,
            "total_results": 0,
            "issues": [],
        }

        start_time = time.time()

        for collection_name in collection_names:
            collection_start = time.time()

            try:
                # Perform search
                search_results = self.qdrant_client.search(
                    collection_name=collection_name, query_vector=query_embedding, limit=n_results, score_threshold=0.1
                )

                collection_time = time.time() - collection_start

                # Analyze results
                result_count = len(search_results)
                scores = [result.score for result in search_results]

                performance_metrics["collection_performance"][collection_name] = {
                    "search_time": collection_time,
                    "result_count": result_count,
                    "average_score": sum(scores) / len(scores) if scores else 0,
                    "max_score": max(scores) if scores else 0,
                    "min_score": min(scores) if scores else 0,
                    "empty_content_count": sum(1 for result in search_results if not result.payload.get("content", "").strip()),
                }

                performance_metrics["total_results"] += result_count

            except Exception as e:
                performance_metrics["issues"].append({"collection": collection_name, "error": str(e), "type": "search_error"})

        performance_metrics["total_time"] = time.time() - start_time
        performance_metrics["average_time_per_collection"] = (
            performance_metrics["total_time"] / len(collection_names) if collection_names else 0
        )

        return performance_metrics

    def diagnose_empty_content_issue(self, collection_name: str) -> dict[str, Any]:
        """Diagnose the root cause of empty content in a collection.

        Args:
            collection_name: Name of the collection to diagnose

        Returns:
            Dictionary containing diagnostic information
        """
        try:
            # Get a larger sample to analyze
            points, _ = self.qdrant_client.scroll(collection_name=collection_name, limit=500, with_payload=True, with_vectors=False)

            diagnosis = {
                "collection_name": collection_name,
                "total_sampled": len(points),
                "empty_content_analysis": {"completely_empty": 0, "whitespace_only": 0, "missing_content_field": 0, "null_content": 0},
                "file_pattern_analysis": {},
                "chunk_type_analysis": {},
                "recommendations": [],
            }

            file_empty_counts = defaultdict(int)
            chunk_type_empty_counts = defaultdict(int)

            for point in points:
                if not point.payload:
                    diagnosis["empty_content_analysis"]["missing_content_field"] += 1
                    continue

                content = point.payload.get("content")
                file_path = point.payload.get("file_path", "unknown")
                chunk_type = point.payload.get("chunk_type", "unknown")

                if content is None:
                    diagnosis["empty_content_analysis"]["null_content"] += 1
                    file_empty_counts[file_path] += 1
                    chunk_type_empty_counts[chunk_type] += 1
                elif content == "":
                    diagnosis["empty_content_analysis"]["completely_empty"] += 1
                    file_empty_counts[file_path] += 1
                    chunk_type_empty_counts[chunk_type] += 1
                elif not content.strip():
                    diagnosis["empty_content_analysis"]["whitespace_only"] += 1
                    file_empty_counts[file_path] += 1
                    chunk_type_empty_counts[chunk_type] += 1

            # Analyze patterns
            diagnosis["file_pattern_analysis"] = dict(file_empty_counts)
            diagnosis["chunk_type_analysis"] = dict(chunk_type_empty_counts)

            total_empty = sum(diagnosis["empty_content_analysis"].values())
            diagnosis["empty_content_rate"] = total_empty / len(points) if points else 0

            # Generate recommendations
            if total_empty > 0:
                diagnosis["recommendations"].append(
                    f"Found {total_empty} chunks with empty content ({diagnosis['empty_content_rate']:.1%} of sample)"
                )

                if diagnosis["empty_content_analysis"]["missing_content_field"] > 0:
                    diagnosis["recommendations"].append("Some chunks are missing the content field entirely - check indexing process")

                if file_empty_counts:
                    top_problematic_files = sorted(file_empty_counts.items(), key=lambda x: x[1], reverse=True)[:5]
                    diagnosis["recommendations"].append(
                        f"Files with most empty chunks: {', '.join(f'{file}({count})' for file, count in top_problematic_files)}"
                    )

                if chunk_type_empty_counts:
                    diagnosis["recommendations"].append(f"Chunk types with empty content: {dict(chunk_type_empty_counts)}")

            return diagnosis

        except Exception as e:
            return {
                "collection_name": collection_name,
                "error": str(e),
                "recommendations": ["Unable to diagnose - check collection accessibility"],
            }

    def create_reindexing_recommendations(self, collection_name: str) -> dict[str, Any]:
        """Create recommendations for reindexing problematic files.

        Args:
            collection_name: Name of the collection to analyze

        Returns:
            Dictionary containing reindexing recommendations
        """
        try:
            # Get empty content diagnosis
            empty_diagnosis = self.diagnose_empty_content_issue(collection_name)

            recommendations = {
                "collection_name": collection_name,
                "reindex_needed": False,
                "problematic_files": [],
                "actions": [],
                "priority": "low",
            }

            if empty_diagnosis.get("empty_content_rate", 0) > 0.1:  # More than 10% empty
                recommendations["reindex_needed"] = True
                recommendations["priority"] = "high" if empty_diagnosis["empty_content_rate"] > 0.3 else "medium"

                # Get problematic files from file pattern analysis
                file_issues = empty_diagnosis.get("file_pattern_analysis", {})
                problematic_files = [
                    {"file_path": file_path, "empty_chunks": count} for file_path, count in file_issues.items() if count > 0
                ]

                # Sort by number of empty chunks (worst first)
                problematic_files.sort(key=lambda x: x["empty_chunks"], reverse=True)
                recommendations["problematic_files"] = problematic_files[:20]  # Top 20

                # Create action recommendations
                recommendations["actions"] = [
                    f"Reindex {len(problematic_files)} files with empty content issues",
                    "Verify file encoding and accessibility",
                    "Check indexing process for content extraction failures",
                    "Consider running incremental reindexing for affected files",
                ]

                if empty_diagnosis["empty_content_rate"] > 0.5:
                    recommendations["actions"].insert(0, "URGENT: Over 50% of content is empty - full reindex recommended")

            return recommendations

        except Exception as e:
            return {
                "collection_name": collection_name,
                "error": str(e),
                "reindex_needed": False,
                "actions": ["Unable to analyze - check collection accessibility"],
            }


def create_search_quality_report(
    qdrant_client: QdrantClient, collection_names: list[str], test_queries: list[str] | None = None
) -> dict[str, Any]:
    """Create a comprehensive search quality report.

    Args:
        qdrant_client: Qdrant client instance
        collection_names: List of collection names to analyze
        test_queries: Optional list of test queries for performance testing

    Returns:
        Dictionary containing comprehensive search quality report
    """
    if test_queries is None:
        test_queries = ["function definition", "class implementation", "error handling", "database connection", "API endpoint"]

    diagnostics = SearchDiagnostics(qdrant_client)

    report = {
        "timestamp": time.time(),
        "collections_analyzed": collection_names,
        "collection_health": {},
        "consistency_check": {},
        "performance_metrics": {},
        "overall_recommendations": [],
    }

    # Analyze each collection
    for collection_name in collection_names:
        logger.info(f"Analyzing collection: {collection_name}")
        health_analysis = diagnostics.analyze_collection_health(collection_name)
        report["collection_health"][collection_name] = health_analysis

        # Diagnose empty content if issues found
        if any(issue["type"] == "empty_content" for issue in health_analysis.get("issues", [])):
            logger.info(f"Diagnosing empty content in collection: {collection_name}")
            empty_content_diagnosis = diagnostics.diagnose_empty_content_issue(collection_name)
            report["collection_health"][collection_name]["empty_content_diagnosis"] = empty_content_diagnosis

    # Check consistency across collections
    logger.info("Checking vector database consistency")
    report["consistency_check"] = diagnostics.check_vector_database_consistency(collection_names)

    # Generate overall recommendations
    total_issues = sum(len(health.get("issues", [])) for health in report["collection_health"].values())
    if total_issues > 0:
        report["overall_recommendations"].append(f"Found {total_issues} issues across all collections")

    consistency_issues = len(report["consistency_check"].get("issues", []))
    if consistency_issues > 0:
        report["overall_recommendations"].append(f"Found {consistency_issues} consistency issues")

    return report
