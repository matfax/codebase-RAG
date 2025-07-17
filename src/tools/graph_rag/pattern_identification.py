"""Pattern Identification Graph RAG Tool

This module provides MCP tools for identifying architectural patterns
in codebases using Graph RAG pattern recognition capabilities.
"""

import logging
from typing import Any, Optional

from src.services.embedding_service import EmbeddingService
from src.services.graph_rag_service import GraphRAGService
from src.services.pattern_comparison_service import PatternComparisonService
from src.services.pattern_recognition_service import PatternRecognitionService
from src.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


async def graph_identify_patterns(
    project_name: str,
    pattern_types: list[str] = None,
    scope_breadcrumb: str = None,
    min_confidence: float = 0.6,
    include_comparisons: bool = True,
    include_improvements: bool = False,
    max_patterns: int = 20,
    analysis_depth: str = "comprehensive",
) -> dict[str, Any]:
    """
    Identify architectural patterns in a codebase using Graph RAG capabilities.

    This tool leverages pattern recognition algorithms to detect common
    architectural patterns, design patterns, and code organization structures
    within the analyzed codebase.

    Args:
        project_name: Name of the project to analyze
        pattern_types: List of specific pattern types to look for
                      ("structural", "behavioral", "creational", "naming", "architectural")
        scope_breadcrumb: Optional breadcrumb to limit analysis scope (e.g., "MyClass")
        min_confidence: Minimum confidence threshold for pattern detection (0.0-1.0, default: 0.6)
        include_comparisons: Whether to include pattern comparison analysis
        include_improvements: Whether to suggest pattern improvements
        max_patterns: Maximum number of patterns to return (1-50, default: 20)
        analysis_depth: Depth of analysis ("basic", "comprehensive", "detailed")

    Returns:
        Dictionary containing identified patterns with confidence scores,
        pattern types, architectural context, and optional improvement suggestions
    """
    try:
        logger.info(f"Starting pattern identification for project '{project_name}'")

        # Initialize core services
        qdrant_service = QdrantService()
        embedding_service = EmbeddingService()
        graph_rag_service = GraphRAGService(qdrant_service, embedding_service)
        pattern_service = PatternRecognitionService(graph_rag_service)

        # Validate parameters
        if not project_name or not project_name.strip():
            return {"success": False, "error": "Project name is required and cannot be empty"}

        min_confidence = max(0.0, min(min_confidence, 1.0))  # Clamp between 0 and 1
        max_patterns = max(1, min(max_patterns, 50))  # Clamp between 1 and 50

        # Validate pattern types
        valid_pattern_types = {"structural", "behavioral", "creational", "naming", "architectural"}

        if pattern_types:
            invalid_types = [pt for pt in pattern_types if pt not in valid_pattern_types]
            if invalid_types:
                return {"success": False, "error": f"Invalid pattern types: {invalid_types}. Valid types: {list(valid_pattern_types)}"}

        # Initialize results structure
        results = {
            "success": True,
            "project_name": project_name,
            "scope_breadcrumb": scope_breadcrumb,
            "min_confidence": min_confidence,
            "analysis_depth": analysis_depth,
            "pattern_types_requested": pattern_types,
            "max_patterns": max_patterns,
        }

        # Perform pattern analysis
        pattern_analysis = await pattern_service.analyze_project_patterns(
            project_name=project_name, scope_breadcrumb=scope_breadcrumb, min_confidence=min_confidence
        )

        if not pattern_analysis.patterns_found:
            return {
                "success": True,
                "project_name": project_name,
                "patterns_identified": [],
                "message": f"No patterns found with confidence >= {min_confidence}",
                "analysis_statistics": {
                    "total_components_analyzed": pattern_analysis.total_components_analyzed,
                    "analysis_time_ms": pattern_analysis.analysis_time_ms,
                    "coverage_percentage": pattern_analysis.coverage_percentage,
                },
            }

        # Filter patterns by requested types if specified
        filtered_patterns = pattern_analysis.patterns_found
        if pattern_types:
            filtered_patterns = [pattern for pattern in pattern_analysis.patterns_found if pattern.pattern_type.lower() in pattern_types]

        # Limit to max_patterns
        filtered_patterns = filtered_patterns[:max_patterns]

        # Process patterns into response format
        identified_patterns = []

        for pattern in filtered_patterns:
            pattern_info = {
                "pattern_name": pattern.pattern_name,
                "pattern_type": pattern.pattern_type,
                "confidence_score": pattern.confidence_score,
                "description": pattern.description,
                "evidence": pattern.evidence,
                "components_involved": [
                    {
                        "breadcrumb": comp.breadcrumb,
                        "file_path": comp.file_path,
                        "chunk_type": comp.chunk_type,
                        "role_in_pattern": comp.role_in_pattern if hasattr(comp, "role_in_pattern") else None,
                    }
                    for comp in pattern.components_involved
                ],
                "quality_metrics": pattern.quality_metrics,
            }

            # Add detailed analysis for comprehensive mode
            if analysis_depth in ["comprehensive", "detailed"]:
                pattern_info.update(
                    {
                        "structural_characteristics": getattr(pattern, "structural_characteristics", {}),
                        "usage_frequency": getattr(pattern, "usage_frequency", 0),
                        "complexity_score": getattr(pattern, "complexity_score", 0.0),
                    }
                )

            # Add even more detail for detailed mode
            if analysis_depth == "detailed":
                pattern_info.update(
                    {
                        "related_patterns": getattr(pattern, "related_patterns", []),
                        "anti_patterns_detected": getattr(pattern, "anti_patterns_detected", []),
                        "maintainability_impact": getattr(pattern, "maintainability_impact", {}),
                    }
                )

            identified_patterns.append(pattern_info)

        results["patterns_identified"] = identified_patterns

        # Add pattern comparisons if requested
        if include_comparisons and len(identified_patterns) > 1:
            try:
                comparison_service = PatternComparisonService(qdrant_service, embedding_service)

                # Compare architectural patterns
                architectural_analysis = await comparison_service.analyze_architectural_patterns(
                    patterns=[pattern for pattern in pattern_analysis.patterns_found if pattern.pattern_type.lower() == "architectural"],
                    project_name=project_name,
                )

                results["pattern_comparisons"] = {
                    "architectural_consistency": architectural_analysis.get("consistency_score", 0.0),
                    "dominant_patterns": architectural_analysis.get("dominant_patterns", []),
                    "pattern_conflicts": architectural_analysis.get("conflicts", []),
                    "recommendations": architectural_analysis.get("recommendations", []),
                }

            except Exception as e:
                logger.warning(f"Could not perform pattern comparisons: {e}")
                results["pattern_comparisons"] = {}

        # Add improvement suggestions if requested
        if include_improvements and identified_patterns:
            try:
                improvement_suggestions = await pattern_service.suggest_pattern_improvements(
                    patterns=pattern_analysis.patterns_found[:5],
                    project_name=project_name,  # Top 5 patterns
                )

                results["improvement_suggestions"] = [
                    {
                        "pattern_name": suggestion.pattern_name,
                        "current_quality": suggestion.current_quality_score,
                        "improvement_opportunities": suggestion.improvement_opportunities,
                        "suggested_changes": suggestion.suggested_changes,
                        "expected_benefit": suggestion.expected_benefit,
                        "implementation_effort": suggestion.implementation_effort,
                    }
                    for suggestion in improvement_suggestions.suggestions
                ]

            except Exception as e:
                logger.warning(f"Could not generate improvement suggestions: {e}")
                results["improvement_suggestions"] = []

        # Add analysis statistics
        results["analysis_statistics"] = {
            "total_components_analyzed": pattern_analysis.total_components_analyzed,
            "patterns_found_total": len(pattern_analysis.patterns_found),
            "patterns_returned": len(identified_patterns),
            "analysis_time_ms": pattern_analysis.analysis_time_ms,
            "coverage_percentage": pattern_analysis.coverage_percentage,
            "confidence_distribution": {
                "high_confidence": len([p for p in filtered_patterns if p.confidence_score >= 0.8]),
                "medium_confidence": len([p for p in filtered_patterns if 0.6 <= p.confidence_score < 0.8]),
                "low_confidence": len([p for p in filtered_patterns if p.confidence_score < 0.6]),
            },
            "pattern_type_distribution": {
                pattern_type: len([p for p in filtered_patterns if p.pattern_type.lower() == pattern_type])
                for pattern_type in valid_pattern_types
            },
        }

        # Add quality insights
        if identified_patterns:
            avg_confidence = sum(p["confidence_score"] for p in identified_patterns) / len(identified_patterns)
            highest_confidence = max(p["confidence_score"] for p in identified_patterns)

            results["quality_insights"] = {
                "average_pattern_confidence": avg_confidence,
                "highest_confidence_pattern": highest_confidence,
                "pattern_diversity": len({p["pattern_type"] for p in identified_patterns}),
                "code_organization_quality": "high" if avg_confidence >= 0.8 else "medium" if avg_confidence >= 0.6 else "low",
            }

        logger.info(f"Identified {len(identified_patterns)} patterns for project '{project_name}'")
        return results

    except Exception as e:
        error_msg = f"Error identifying patterns for project '{project_name}': {str(e)}"
        logger.error(error_msg, exc_info=True)
        return {"success": False, "error": error_msg, "project_name": project_name}
