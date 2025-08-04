"""Project Chain Analysis Tool

This module provides MCP tools for analyzing function chains across entire projects,
providing comprehensive insights into project-wide patterns, complexity, and
architecture. It supports project-scope analysis with breadcrumb pattern matching,
complexity calculation, hotspot identification, and refactoring recommendations.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from src.models.code_chunk import ChunkType
from src.services.breadcrumb_resolver_service import BreadcrumbResolver
from src.services.implementation_chain_service import (
    ChainType,
    ImplementationChainService,
    get_implementation_chain_service,
)
from src.utils.complexity_calculator import (
    ComplexityCalculator,
    ComplexityWeights,
    create_complexity_calculator,
    get_default_complexity_weights,
)
from src.utils.output_formatters import (
    MermaidStyle,
    OutputFormat,
    format_comprehensive_output,
)

logger = logging.getLogger(__name__)


class AnalysisScope(Enum):
    """Analysis scope types for project chain analysis."""

    FULL_PROJECT = "full_project"
    SCOPED_BREADCRUMBS = "scoped_breadcrumbs"
    SPECIFIC_MODULES = "specific_modules"
    FUNCTION_PATTERNS = "function_patterns"


class ChainAnalysisType(Enum):
    """Types of chain analysis to perform."""

    COMPLEXITY_ANALYSIS = "complexity_analysis"
    HOTSPOT_IDENTIFICATION = "hotspot_identification"
    COVERAGE_ANALYSIS = "coverage_analysis"
    REFACTORING_RECOMMENDATIONS = "refactoring_recommendations"
    PROJECT_METRICS = "project_metrics"
    COMPREHENSIVE = "comprehensive"


@dataclass
class ProjectChainMetrics:
    """Metrics for project-wide chain analysis."""

    # Basic metrics
    total_functions: int = 0
    total_chains: int = 0
    average_chain_depth: float = 0.0
    total_entry_points: int = 0

    # Complexity metrics
    overall_complexity_score: float = 0.0
    complexity_distribution: dict[str, int] = None

    # Connectivity metrics
    connectivity_score: float = 0.0
    isolated_functions: int = 0
    highly_connected_functions: int = 0

    # Hotspot metrics
    hotspot_functions: list[str] = None
    critical_paths: list[str] = None

    # Coverage metrics
    coverage_percentage: float = 0.0
    uncovered_functions: list[str] = None

    def __post_init__(self):
        """Initialize mutable fields."""
        if self.complexity_distribution is None:
            self.complexity_distribution = {}
        if self.hotspot_functions is None:
            self.hotspot_functions = []
        if self.critical_paths is None:
            self.critical_paths = []
        if self.uncovered_functions is None:
            self.uncovered_functions = []


@dataclass
class FunctionAnalysisResult:
    """Analysis result for a single function."""

    breadcrumb: str
    name: str
    file_path: str

    # Complexity metrics
    complexity_score: float = 0.0
    branching_factor: int = 0
    cyclomatic_complexity: int = 0
    call_depth: int = 0
    function_length: int = 0

    # Connectivity metrics
    incoming_connections: int = 0
    outgoing_connections: int = 0
    connectivity_score: float = 0.0

    # Hotspot metrics
    usage_frequency: int = 0
    criticality_score: float = 0.0
    is_hotspot: bool = False

    # Chain information
    chain_types: list[str] = None
    is_entry_point: bool = False
    is_terminal_point: bool = False

    # Refactoring recommendations
    refactoring_suggestions: list[str] = None

    def __post_init__(self):
        """Initialize mutable fields."""
        if self.chain_types is None:
            self.chain_types = []
        if self.refactoring_suggestions is None:
            self.refactoring_suggestions = []


async def analyze_project_chains(
    project_name: str,
    analysis_types: list[str] = None,
    scope_pattern: str = "*",
    complexity_weights: dict[str, float] = None,
    chain_types: list[str] = None,
    min_complexity_threshold: float = 0.3,
    max_functions_to_analyze: int = 5000,  # Increased for large projects
    include_refactoring_suggestions: bool = True,
    output_format: str = "comprehensive",
    performance_monitoring: bool = True,
    batch_size: int = 50,
) -> dict[str, Any]:
    """
    Analyze function chains across an entire project with comprehensive insights.

    This tool provides project-wide analysis of function chains, including complexity
    calculation, hotspot identification, coverage analysis, and refactoring recommendations.
    It supports flexible scoping and can handle large projects with batch processing.

    Args:
        project_name: Name of the project to analyze
        analysis_types: List of analysis types to perform (default: ["comprehensive"])
        scope_pattern: Breadcrumb pattern to limit analysis scope (default: "*")
        complexity_weights: Custom weights for complexity calculation components
        chain_types: List of chain types to analyze (default: all types)
        min_complexity_threshold: Minimum complexity threshold for reporting
        max_functions_to_analyze: Maximum number of functions to analyze
        include_refactoring_suggestions: Whether to include refactoring recommendations
        output_format: Output format ("comprehensive", "summary", "detailed")
        performance_monitoring: Whether to include performance monitoring
        batch_size: Batch size for processing large projects

    Returns:
        Dictionary containing comprehensive project chain analysis results
    """
    start_time = time.time()

    try:
        logger.info(f"Starting project chain analysis for '{project_name}'")

        # Initialize results structure
        results = {
            "success": True,
            "project_name": project_name,
            "analysis_types": analysis_types or ["comprehensive"],
            "scope_pattern": scope_pattern,
            "timestamp": time.time(),
            "configuration": {
                "complexity_weights": complexity_weights,
                "chain_types": chain_types,
                "min_complexity_threshold": min_complexity_threshold,
                "max_functions_to_analyze": max_functions_to_analyze,
                "include_refactoring_suggestions": include_refactoring_suggestions,
                "batch_size": batch_size,
            },
        }

        # Initialize performance monitoring
        if performance_monitoring:
            results["performance"] = {
                "start_time": start_time,
                "function_discovery_time": 0.0,
                "complexity_analysis_time": 0.0,
                "hotspot_analysis_time": 0.0,
                "coverage_analysis_time": 0.0,
                "report_generation_time": 0.0,
                "total_time": 0.0,
            }

        # Validate input parameters
        validation_result = _validate_analysis_parameters(
            project_name, analysis_types, scope_pattern, complexity_weights, chain_types, min_complexity_threshold, max_functions_to_analyze
        )

        if not validation_result["valid"]:
            results["success"] = False
            results["error"] = validation_result["error"]
            results["suggestions"] = validation_result.get("suggestions", [])
            return results

        # Initialize services
        from src.services.embedding_service import EmbeddingService
        from src.services.graph_rag_service import get_graph_rag_service
        from src.services.hybrid_search_service import get_hybrid_search_service
        from src.services.implementation_chain_service import get_implementation_chain_service
        from src.services.qdrant_service import QdrantService

        breadcrumb_resolver = BreadcrumbResolver()
        qdrant_service = QdrantService()
        embedding_service = EmbeddingService()

        # Initialize Graph RAG and Hybrid Search services with required dependencies
        graph_rag_service = get_graph_rag_service(qdrant_service, embedding_service)
        hybrid_search_service = get_hybrid_search_service()

        implementation_chain_service = get_implementation_chain_service(
            graph_rag_service=graph_rag_service, hybrid_search_service=hybrid_search_service
        )

        # Step 1: Discover functions matching the scope pattern
        discovery_start_time = time.time()

        discovered_functions = await _discover_project_functions(
            project_name, scope_pattern, max_functions_to_analyze, implementation_chain_service, breadcrumb_resolver
        )

        if not discovered_functions:
            results["success"] = False
            results["error"] = "No functions found matching the scope pattern"
            results["suggestions"] = [
                "Check if the project has been indexed",
                "Verify the scope pattern is correct",
                "Try a broader scope pattern like '*'",
                "Use the search tool to explore available functions",
            ]
            return results

        results["discovered_functions_count"] = len(discovered_functions)

        # Add pattern analysis information
        pattern_analysis = _validate_scope_pattern(scope_pattern)
        if pattern_analysis["valid"]:
            results["scope_pattern_analysis"] = pattern_analysis["pattern_analysis"]
            results["pattern_examples"] = pattern_analysis["examples"]

        if performance_monitoring:
            results["performance"]["function_discovery_time"] = (time.time() - discovery_start_time) * 1000

        # Step 2: Initialize complexity calculator with weights
        if complexity_weights:
            weights = ComplexityWeights(
                branching_factor=complexity_weights.get("branching_factor", 0.35),
                cyclomatic_complexity=complexity_weights.get("cyclomatic_complexity", 0.30),
                call_depth=complexity_weights.get("call_depth", 0.25),
                function_length=complexity_weights.get("function_length", 0.10),
            )
            complexity_calculator = ComplexityCalculator(weights)
        else:
            complexity_calculator = create_complexity_calculator()

        weights = complexity_calculator.weights

        # Step 3: Perform requested analysis types
        analysis_results = {}

        # Parse analysis types
        if not analysis_types:
            analysis_types = ["comprehensive"]

        parsed_analysis_types = []
        for analysis_type in analysis_types:
            try:
                parsed_analysis_types.append(ChainAnalysisType(analysis_type.lower()))
            except ValueError:
                logger.warning(f"Unknown analysis type: {analysis_type}")
                continue

        # If comprehensive is requested, include all analysis types
        if ChainAnalysisType.COMPREHENSIVE in parsed_analysis_types:
            parsed_analysis_types = [
                ChainAnalysisType.COMPLEXITY_ANALYSIS,
                ChainAnalysisType.HOTSPOT_IDENTIFICATION,
                ChainAnalysisType.COVERAGE_ANALYSIS,
                ChainAnalysisType.REFACTORING_RECOMMENDATIONS,
                ChainAnalysisType.PROJECT_METRICS,
            ]

        # Perform complexity analysis
        if ChainAnalysisType.COMPLEXITY_ANALYSIS in parsed_analysis_types:
            complexity_start_time = time.time()

            complexity_results = await _perform_complexity_analysis(
                discovered_functions, complexity_calculator, min_complexity_threshold, implementation_chain_service, batch_size
            )

            analysis_results["complexity_analysis"] = complexity_results

            if performance_monitoring:
                results["performance"]["complexity_analysis_time"] = (time.time() - complexity_start_time) * 1000

        # Perform hotspot identification
        if ChainAnalysisType.HOTSPOT_IDENTIFICATION in parsed_analysis_types:
            hotspot_start_time = time.time()

            hotspot_results = await _perform_hotspot_analysis(discovered_functions, implementation_chain_service, batch_size, chain_types)

            analysis_results["hotspot_analysis"] = hotspot_results

            if performance_monitoring:
                results["performance"]["hotspot_analysis_time"] = (time.time() - hotspot_start_time) * 1000

        # Perform coverage analysis
        if ChainAnalysisType.COVERAGE_ANALYSIS in parsed_analysis_types:
            coverage_start_time = time.time()

            coverage_results = await _perform_coverage_analysis(discovered_functions, implementation_chain_service, batch_size, chain_types)

            analysis_results["coverage_analysis"] = coverage_results

            if performance_monitoring:
                results["performance"]["coverage_analysis_time"] = (time.time() - coverage_start_time) * 1000

        # Perform refactoring recommendations
        if ChainAnalysisType.REFACTORING_RECOMMENDATIONS in parsed_analysis_types and include_refactoring_suggestions:
            refactoring_results = await _generate_refactoring_recommendations(
                analysis_results, discovered_functions, min_complexity_threshold
            )

            analysis_results["refactoring_recommendations"] = refactoring_results

        # Calculate project-level metrics
        if ChainAnalysisType.PROJECT_METRICS in parsed_analysis_types:
            project_metrics = await _calculate_project_metrics(
                discovered_functions, analysis_results, implementation_chain_service, chain_types
            )

            analysis_results["project_metrics"] = project_metrics

        # Step 4: Generate comprehensive report
        report_start_time = time.time()

        report = await _generate_analysis_report(analysis_results, discovered_functions, weights, output_format)

        results.update(report)

        # Add chain type filtering information to the results
        if chain_types:
            results["chain_type_filtering"] = {
                "enabled": True,
                "requested_chain_types": chain_types,
                "supported_chain_types": [t.value for t in ChainType],
                "filtering_impact": {
                    "description": f"Analysis filtered to focus on {', '.join(chain_types)} chain types",
                    "coverage_note": "Results represent only the specified chain types",
                },
            }
        else:
            results["chain_type_filtering"] = {"enabled": False, "description": "Analysis includes all available chain types"}

        if performance_monitoring:
            results["performance"]["report_generation_time"] = (time.time() - report_start_time) * 1000
            results["performance"]["total_time"] = (time.time() - start_time) * 1000

        logger.info(f"Project chain analysis completed successfully for '{project_name}' - analyzed {len(discovered_functions)} functions")

        return results

    except Exception as e:
        error_msg = f"Error analyzing project chains for '{project_name}': {str(e)}"
        logger.error(error_msg, exc_info=True)

        results = {
            "success": False,
            "error": error_msg,
            "project_name": project_name,
            "analysis_types": analysis_types or ["comprehensive"],
            "scope_pattern": scope_pattern,
            "suggestions": [
                "Check if the project has been indexed properly",
                "Verify the project name is correct",
                "Try a simpler scope pattern",
                "Reduce the number of functions to analyze",
                "Check system logs for more details",
            ],
            "timestamp": time.time(),
        }

        if performance_monitoring:
            results["performance"] = {
                "total_time": (time.time() - start_time) * 1000,
                "error_occurred": True,
            }

        return results


def _validate_analysis_parameters(
    project_name: str,
    analysis_types: list[str],
    scope_pattern: str,
    complexity_weights: dict[str, float],
    chain_types: list[str],
    min_complexity_threshold: float,
    max_functions_to_analyze: int,
) -> dict[str, Any]:
    """
    Validate input parameters for project chain analysis.

    Args:
        project_name: Name of the project
        analysis_types: List of analysis types
        scope_pattern: Breadcrumb scope pattern
        complexity_weights: Complexity calculation weights
        chain_types: List of chain types
        min_complexity_threshold: Minimum complexity threshold
        max_functions_to_analyze: Maximum number of functions to analyze

    Returns:
        Dictionary with validation results
    """
    if not project_name or not project_name.strip():
        return {"valid": False, "error": "Project name is required and cannot be empty", "suggestions": ["Provide a valid project name"]}

    if analysis_types:
        valid_analysis_types = [t.value for t in ChainAnalysisType]
        for analysis_type in analysis_types:
            if analysis_type.lower() not in valid_analysis_types:
                return {
                    "valid": False,
                    "error": f"Invalid analysis type: {analysis_type}",
                    "suggestions": [f"Valid analysis types: {', '.join(valid_analysis_types)}"],
                }

    if not scope_pattern or not scope_pattern.strip():
        return {
            "valid": False,
            "error": "Scope pattern cannot be empty",
            "suggestions": ["Use '*' for full project analysis or provide a specific pattern"],
        }

    # Validate scope pattern
    pattern_validation = _validate_scope_pattern(scope_pattern)
    if not pattern_validation["valid"]:
        return pattern_validation

    if complexity_weights:
        required_keys = ["branching_factor", "cyclomatic_complexity", "call_depth", "function_length"]
        for key in complexity_weights:
            if key not in required_keys:
                return {
                    "valid": False,
                    "error": f"Invalid complexity weight key: {key}",
                    "suggestions": [f"Valid weight keys: {', '.join(required_keys)}"],
                }

            if not isinstance(complexity_weights[key], int | float) or complexity_weights[key] < 0:
                return {
                    "valid": False,
                    "error": f"Invalid complexity weight value for {key}: {complexity_weights[key]}",
                    "suggestions": ["Complexity weights must be non-negative numbers"],
                }

    if chain_types:
        valid_chain_types = [t.value for t in ChainType]
        for chain_type in chain_types:
            if chain_type.lower() not in valid_chain_types:
                return {
                    "valid": False,
                    "error": f"Invalid chain type: {chain_type}",
                    "suggestions": [f"Valid chain types: {', '.join(valid_chain_types)}"],
                }

    if not isinstance(min_complexity_threshold, int | float) or min_complexity_threshold < 0.0 or min_complexity_threshold > 1.0:
        return {
            "valid": False,
            "error": f"Invalid min_complexity_threshold: {min_complexity_threshold}. Must be between 0.0 and 1.0",
            "suggestions": ["Use a complexity threshold between 0.0 and 1.0"],
        }

    if not isinstance(max_functions_to_analyze, int) or max_functions_to_analyze < 1 or max_functions_to_analyze > 10000:
        return {
            "valid": False,
            "error": f"Invalid max_functions_to_analyze: {max_functions_to_analyze}. Must be between 1 and 10000",
            "suggestions": ["Use a reasonable number of functions between 1 and 10000"],
        }

    return {"valid": True}


async def _discover_project_functions(
    project_name: str,
    scope_pattern: str,
    max_functions: int,
    implementation_chain_service: ImplementationChainService,
    breadcrumb_resolver: BreadcrumbResolver,
) -> list[dict[str, Any]]:
    """
    Discover functions in the project matching the scope pattern.

    Args:
        project_name: Name of the project
        scope_pattern: Breadcrumb pattern to match
        max_functions: Maximum number of functions to return
        implementation_chain_service: Service for chain analysis
        breadcrumb_resolver: Service for breadcrumb resolution

    Returns:
        List of discovered function information
    """
    logger.info(f"Discovering functions in project '{project_name}' with pattern '{scope_pattern}'")

    try:
        # Get the project structure graph
        project_graph = await implementation_chain_service.graph_rag_service.build_structure_graph(project_name, force_rebuild=True)

        if not project_graph:
            logger.warning(f"No structure graph found for project: {project_name}")
            return []

        logger.info(f"Project graph has {len(project_graph.nodes)} total nodes")

        # Extract all functions from the graph
        all_functions = []
        chunk_type_counts = {}
        for node in project_graph.nodes:
            # Safety check for node attributes
            if not hasattr(node, "chunk_type"):
                continue

            chunk_type = getattr(node, "chunk_type", "unknown")
            chunk_type_counts[chunk_type] = chunk_type_counts.get(chunk_type, 0) + 1

            # Convert to string values for comparison
            if chunk_type in [ChunkType.FUNCTION.value, ChunkType.METHOD.value] or chunk_type in ["function", "method"]:
                function_info = {
                    "breadcrumb": getattr(node, "breadcrumb", ""),
                    "name": getattr(node, "name", ""),
                    "file_path": getattr(node, "file_path", ""),
                    "chunk_type": chunk_type,
                    "chunk_id": getattr(node, "chunk_id", ""),
                    "line_start": getattr(node, "line_start", 0),
                    "line_end": getattr(node, "line_end", 0),
                    "language": getattr(node, "language", ""),
                    "content": getattr(node, "content", ""),
                }
                all_functions.append(function_info)

        logger.info(f"Found chunk types: {chunk_type_counts}")
        logger.info(f"Found {len(all_functions)} functions/methods in graph")

        # Filter functions based on scope pattern
        filtered_functions = []
        for func in all_functions:
            if _matches_scope_pattern(func["breadcrumb"], scope_pattern):
                filtered_functions.append(func)

        # Limit the number of functions
        if len(filtered_functions) > max_functions:
            logger.info(f"Limiting analysis to {max_functions} functions (found {len(filtered_functions)})")
            filtered_functions = filtered_functions[:max_functions]

        logger.info(f"Discovered {len(filtered_functions)} functions matching pattern '{scope_pattern}'")
        return filtered_functions

    except Exception as e:
        logger.error(f"Error discovering functions: {e}")
        return []


def _matches_scope_pattern(breadcrumb: str, pattern: str) -> bool:
    """
    Check if a breadcrumb matches the scope pattern.

    Supports advanced pattern matching including:
    - Wildcards (*) for any sequence
    - Prefix matching (api.*)
    - Suffix matching (*.handler)
    - Multiple wildcards (api.*.handler)
    - Class/module specific patterns (MyClass.*)
    - Regex-style patterns

    Args:
        breadcrumb: Function breadcrumb to check
        pattern: Scope pattern to match against

    Returns:
        True if the breadcrumb matches the pattern
    """
    if not breadcrumb or not pattern:
        return False

    # "*" matches everything
    if pattern == "*":
        return True

    # Exact match
    if pattern == breadcrumb:
        return True

    # Convert pattern to regex for advanced matching
    regex_pattern = _convert_pattern_to_regex(pattern)

    try:
        import re

        return bool(re.match(regex_pattern, breadcrumb))
    except Exception:
        # Fallback to simple matching if regex fails
        return _simple_pattern_match(breadcrumb, pattern)


def _convert_pattern_to_regex(pattern: str) -> str:
    """
    Convert a breadcrumb pattern to a regex pattern.

    Args:
        pattern: Breadcrumb pattern with wildcards

    Returns:
        Regex pattern string
    """
    import re

    # Escape special regex characters except * and ?
    escaped = re.escape(pattern)

    # Replace escaped wildcards with regex equivalents
    regex_pattern = escaped.replace(r"\*", ".*")  # * matches any sequence
    regex_pattern = regex_pattern.replace(r"\?", ".")  # ? matches any single character

    # Add anchors to match the full string
    regex_pattern = f"^{regex_pattern}$"

    return regex_pattern


def _simple_pattern_match(breadcrumb: str, pattern: str) -> bool:
    """
    Simple pattern matching fallback implementation.

    Args:
        breadcrumb: Function breadcrumb to check
        pattern: Pattern to match against

    Returns:
        True if the breadcrumb matches the pattern
    """
    # Split pattern by wildcards
    parts = pattern.split("*")

    if len(parts) == 1:
        # No wildcards, exact match
        return breadcrumb == pattern

    # Check if breadcrumb matches all parts in sequence
    current_pos = 0

    for i, part in enumerate(parts):
        if not part:  # Empty part (consecutive wildcards)
            continue

        if i == 0:
            # First part - must match at the beginning
            if not breadcrumb.startswith(part):
                return False
            current_pos = len(part)
        elif i == len(parts) - 1:
            # Last part - must match at the end
            if not breadcrumb.endswith(part):
                return False
            # Check if there's enough space for this part
            if current_pos > len(breadcrumb) - len(part):
                return False
        else:
            # Middle part - must be found after current position
            pos = breadcrumb.find(part, current_pos)
            if pos == -1:
                return False
            current_pos = pos + len(part)

    return True


def _get_advanced_pattern_examples() -> list[dict[str, str]]:
    """
    Get examples of advanced pattern matching.

    Returns:
        List of pattern examples with descriptions
    """
    return [
        {"pattern": "*", "description": "Matches all functions", "example": "matches: api.user.get_user, core.database.connect"},
        {
            "pattern": "api.*",
            "description": "Matches all functions starting with 'api.'",
            "example": "matches: api.user.get_user, api.auth.login",
        },
        {
            "pattern": "*.handler",
            "description": "Matches all functions ending with '.handler'",
            "example": "matches: user.request.handler, auth.login.handler",
        },
        {
            "pattern": "api.*.get_*",
            "description": "Matches API getter functions",
            "example": "matches: api.user.get_user, api.product.get_product",
        },
        {"pattern": "MyClass.*", "description": "Matches all methods in MyClass", "example": "matches: MyClass.method1, MyClass.method2"},
        {"pattern": "*.test_*", "description": "Matches all test functions", "example": "matches: user.test_create, auth.test_login"},
        {
            "pattern": "core.*.database.*",
            "description": "Matches database-related functions in core modules",
            "example": "matches: core.user.database.save, core.auth.database.query",
        },
    ]


def _validate_scope_pattern(pattern: str) -> dict[str, Any]:
    """
    Validate and analyze a scope pattern.

    Args:
        pattern: Scope pattern to validate

    Returns:
        Dictionary with validation results and pattern analysis
    """
    if not pattern:
        return {
            "valid": False,
            "error": "Pattern cannot be empty",
            "suggestions": ["Use '*' for all functions", "Provide a specific pattern"],
        }

    # Check for invalid characters
    invalid_chars = ["|", "&", "(", ")", "[", "]", "{", "}", "^", "$", "+"]
    for char in invalid_chars:
        if char in pattern:
            return {
                "valid": False,
                "error": f"Invalid character '{char}' in pattern",
                "suggestions": ["Use only alphanumeric characters, dots, and wildcards (*)"],
            }

    # Analyze pattern complexity
    wildcard_count = pattern.count("*")
    dot_count = pattern.count(".")

    pattern_analysis = {
        "type": _classify_pattern_type(pattern),
        "complexity": _calculate_pattern_complexity(pattern),
        "wildcard_count": wildcard_count,
        "dot_count": dot_count,
        "estimated_matches": _estimate_pattern_matches(pattern),
    }

    return {"valid": True, "pattern_analysis": pattern_analysis, "examples": _get_pattern_examples(pattern)}


def _classify_pattern_type(pattern: str) -> str:
    """Classify the type of pattern."""
    if pattern == "*":
        return "universal"
    elif pattern.endswith(".*"):
        return "prefix"
    elif pattern.startswith("*."):
        return "suffix"
    elif "*" in pattern:
        return "wildcard"
    else:
        return "exact"


def _calculate_pattern_complexity(pattern: str) -> str:
    """Calculate the complexity of a pattern."""
    wildcard_count = pattern.count("*")

    if wildcard_count == 0:
        return "simple"
    elif wildcard_count == 1:
        return "moderate"
    else:
        return "complex"


def _estimate_pattern_matches(pattern: str) -> str:
    """Estimate how many matches a pattern might return."""
    if pattern == "*":
        return "all"
    elif pattern.endswith(".*") or pattern.startswith("*."):
        return "many"
    elif "*" in pattern:
        return "some"
    else:
        return "one"


def _get_pattern_examples(pattern: str) -> list[str]:
    """Get example matches for a pattern."""
    if pattern == "*":
        return ["api.user.get_user", "core.database.connect", "utils.helper.format"]
    elif pattern.endswith(".*"):
        prefix = pattern[:-2]
        return [f"{prefix}.function1", f"{prefix}.function2", f"{prefix}.method"]
    elif pattern.startswith("*."):
        suffix = pattern[2:]
        return [f"module1.{suffix}", f"module2.{suffix}", f"class.{suffix}"]
    else:
        return [pattern]


async def _perform_complexity_analysis(
    functions: list[dict[str, Any]],
    complexity_calculator: ComplexityCalculator,
    min_threshold: float,
    implementation_chain_service: ImplementationChainService,
    batch_size: int,
) -> dict[str, Any]:
    """
    Perform complexity analysis on discovered functions using the complexity calculator.

    Args:
        functions: List of functions to analyze
        complexity_calculator: Configured complexity calculator
        min_threshold: Minimum complexity threshold
        implementation_chain_service: Service for chain analysis
        batch_size: Batch size for processing

    Returns:
        Dictionary with complexity analysis results
    """
    logger.info(f"Performing complexity analysis on {len(functions)} functions using real complexity calculator")

    complex_functions = []
    all_metrics = []

    # Process functions in batches
    for i in range(0, len(functions), batch_size):
        batch = functions[i : i + batch_size]
        logger.debug(f"Processing complexity batch {i // batch_size + 1}/{(len(functions) + batch_size - 1) // batch_size}")

        # Calculate complexity for each function in the batch
        batch_metrics = complexity_calculator.calculate_batch_complexity(batch)
        all_metrics.extend(batch_metrics)

        # Process results for functions meeting threshold
        for func, metrics in zip(batch, batch_metrics, strict=False):
            if metrics.overall_complexity >= min_threshold:
                complexity_breakdown = complexity_calculator.get_complexity_breakdown(metrics)

                complex_functions.append(
                    {
                        "breadcrumb": func["breadcrumb"],
                        "name": func["name"],
                        "file_path": func["file_path"],
                        "complexity_score": metrics.overall_complexity,
                        "complexity_category": metrics.complexity_category,
                        "complexity_breakdown": complexity_breakdown,
                        "raw_metrics": {
                            "branching_factor": metrics.branching_factor,
                            "cyclomatic_complexity": metrics.cyclomatic_complexity,
                            "call_depth": metrics.call_depth,
                            "function_length": metrics.function_length,
                        },
                        "normalized_scores": {
                            "branching_score": metrics.branching_score,
                            "cyclomatic_score": metrics.cyclomatic_score,
                            "call_depth_score": metrics.call_depth_score,
                            "function_length_score": metrics.function_length_score,
                        },
                    }
                )

    # Get overall statistics
    complexity_stats = complexity_calculator.get_complexity_statistics(all_metrics)

    # Calculate average complexity
    avg_complexity = sum(m.overall_complexity for m in all_metrics) / len(all_metrics) if all_metrics else 0

    # Create complexity distribution from statistics
    complexity_distribution = complexity_stats.get("category_distribution", {"low": 0, "medium": 0, "high": 0})

    # Identify top complex functions
    top_complex_functions = sorted(
        [f for f in complex_functions if f["complexity_score"] >= min_threshold], key=lambda x: x["complexity_score"], reverse=True
    )[:10]  # Top 10 most complex functions

    return {
        "total_functions_analyzed": len(functions),
        "complex_functions_count": len(complex_functions),
        "complex_functions": complex_functions,
        "top_complex_functions": top_complex_functions,
        "complexity_distribution": complexity_distribution,
        "complexity_statistics": complexity_stats,
        "average_complexity": avg_complexity,
        "complexity_weights": complexity_calculator.weights.to_dict(),
        "threshold_used": min_threshold,
        "batch_processing": {
            "batch_size": batch_size,
            "total_batches": (len(functions) + batch_size - 1) // batch_size,
            "functions_per_batch": batch_size,
        },
    }


async def _perform_hotspot_analysis(
    functions: list[dict[str, Any]],
    implementation_chain_service: ImplementationChainService,
    batch_size: int,
    chain_types: list[str] = None,
) -> dict[str, Any]:
    """
    Perform hotspot analysis to identify frequently used and critical functions.

    This function analyzes functions to identify:
    1. Usage frequency - How often functions are called by others
    2. Criticality score - Architectural importance based on connectivity patterns
    3. Critical paths - Important execution chains that impact system performance
    4. Hotspot categories - Performance bottlenecks, architectural hubs, entry points

    Args:
        functions: List of functions to analyze
        implementation_chain_service: Service for chain analysis
        batch_size: Batch size for processing
        chain_types: List of chain types to filter analysis (default: all types)

    Returns:
        Dictionary with comprehensive hotspot analysis results
    """
    logger.info(f"Performing hotspot analysis on {len(functions)} functions")

    hotspot_functions = []
    critical_paths = []
    function_connectivity_map = {}

    # Step 1: Build connectivity analysis for all functions
    logger.debug("Building function connectivity map")
    try:
        # Get project structure graph to analyze connections
        project_name = implementation_chain_service.project_name if hasattr(implementation_chain_service, "project_name") else "unknown"
        project_graph = await implementation_chain_service.graph_rag_service.build_structure_graph(project_name, force_rebuild=True)

        if project_graph:
            # Build connectivity map from graph relationships
            for node in project_graph.nodes:
                if node.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD]:
                    function_connectivity_map[node.breadcrumb] = {
                        "incoming_connections": [],
                        "outgoing_connections": [],
                        "connection_count": 0,
                        "is_entry_point": True,  # Will be updated if connections found
                        "architectural_layer": _determine_architectural_layer(node.breadcrumb),
                    }

            # Analyze relationships to build connection patterns
            for relationship in getattr(project_graph, "relationships", []):
                source_breadcrumb = getattr(relationship, "source_breadcrumb", "")
                target_breadcrumb = getattr(relationship, "target_breadcrumb", "")

                if source_breadcrumb in function_connectivity_map:
                    function_connectivity_map[source_breadcrumb]["outgoing_connections"].append(target_breadcrumb)
                    function_connectivity_map[source_breadcrumb]["is_entry_point"] = False

                if target_breadcrumb in function_connectivity_map:
                    function_connectivity_map[target_breadcrumb]["incoming_connections"].append(source_breadcrumb)

            # Update connection counts
            for breadcrumb, connectivity in function_connectivity_map.items():
                connectivity["connection_count"] = len(connectivity["incoming_connections"]) + len(connectivity["outgoing_connections"])

    except Exception as e:
        logger.warning(f"Could not build full connectivity map: {e}. Using fallback analysis.")

    # Step 2: Process functions in batches for hotspot analysis
    all_hotspot_data = []

    for i in range(0, len(functions), batch_size):
        batch = functions[i : i + batch_size]
        logger.debug(f"Processing hotspot batch {i // batch_size + 1}/{(len(functions) + batch_size - 1) // batch_size}")

        for func in batch:
            breadcrumb = func["breadcrumb"]

            # Calculate usage frequency based on incoming connections
            usage_frequency = _calculate_usage_frequency(func, function_connectivity_map.get(breadcrumb, {}))

            # Calculate criticality score based on architectural importance
            criticality_score = _calculate_criticality_score(func, function_connectivity_map.get(breadcrumb, {}))

            # Determine hotspot category
            hotspot_category = _determine_hotspot_category(
                usage_frequency, criticality_score, function_connectivity_map.get(breadcrumb, {})
            )

            # Calculate performance impact score
            performance_impact = _calculate_performance_impact(func, usage_frequency, criticality_score)

            hotspot_data = {
                "breadcrumb": breadcrumb,
                "name": func["name"],
                "file_path": func["file_path"],
                "usage_frequency": usage_frequency,
                "criticality_score": criticality_score,
                "performance_impact": performance_impact,
                "hotspot_category": hotspot_category,
                "connectivity_metrics": function_connectivity_map.get(breadcrumb, {}),
                "is_hotspot": usage_frequency >= 5 or criticality_score >= 0.7 or performance_impact >= 0.6,
                "hotspot_reasons": _generate_hotspot_reasons(usage_frequency, criticality_score, performance_impact, hotspot_category),
            }

            all_hotspot_data.append(hotspot_data)

            # Add to hotspot functions if it meets criteria
            if hotspot_data["is_hotspot"]:
                hotspot_functions.append(hotspot_data)

    # Step 3: Identify critical paths
    critical_paths = await _identify_critical_paths(functions, function_connectivity_map, implementation_chain_service, chain_types)

    # Step 4: Generate hotspot statistics and insights
    hotspot_statistics = _generate_hotspot_statistics(all_hotspot_data, function_connectivity_map)

    return {
        "total_functions_analyzed": len(functions),
        "hotspot_functions_count": len(hotspot_functions),
        "hotspot_functions": hotspot_functions,
        "critical_paths": critical_paths,
        "hotspot_statistics": hotspot_statistics,
        "hotspot_thresholds": {
            "usage_frequency": 5,
            "criticality_score": 0.7,
            "performance_impact": 0.6,
        },
        "hotspot_categories": {
            "performance_bottleneck": len([h for h in hotspot_functions if h["hotspot_category"] == "performance_bottleneck"]),
            "architectural_hub": len([h for h in hotspot_functions if h["hotspot_category"] == "architectural_hub"]),
            "entry_point": len([h for h in hotspot_functions if h["hotspot_category"] == "entry_point"]),
            "critical_utility": len([h for h in hotspot_functions if h["hotspot_category"] == "critical_utility"]),
        },
        "batch_processing": {
            "batch_size": batch_size,
            "total_batches": (len(functions) + batch_size - 1) // batch_size,
        },
    }


def _calculate_usage_frequency(func: dict[str, Any], connectivity_info: dict[str, Any]) -> int:
    """
    Calculate usage frequency based on actual function connections.

    Args:
        func: Function information
        connectivity_info: Connectivity data from graph analysis

    Returns:
        Usage frequency score (number of incoming connections + weighted factors)
    """
    if not connectivity_info:
        # Fallback to heuristic analysis based on function characteristics
        return _calculate_heuristic_usage_frequency(func)

    # Base frequency from incoming connections
    incoming_count = len(connectivity_info.get("incoming_connections", []))

    # Apply architectural layer multiplier
    layer_multiplier = _get_architectural_layer_multiplier(connectivity_info.get("architectural_layer", "unknown"))

    # Apply function name pattern weights
    name_weight = _calculate_name_pattern_weight(func["name"])

    # Calculate final usage frequency
    usage_frequency = max(1, int(incoming_count * layer_multiplier * name_weight))

    return usage_frequency


def _calculate_criticality_score(func: dict[str, Any], connectivity_info: dict[str, Any]) -> float:
    """
    Calculate criticality score based on architectural importance.

    Args:
        func: Function information
        connectivity_info: Connectivity data from graph analysis

    Returns:
        Criticality score (0.0 to 1.0)
    """
    if not connectivity_info:
        return _calculate_heuristic_criticality(func)

    # Base criticality factors
    factors = {
        "connection_density": 0.0,
        "architectural_position": 0.0,
        "entry_point_factor": 0.0,
        "name_importance": 0.0,
    }

    # Connection density score (0.0 to 0.4)
    total_connections = connectivity_info.get("connection_count", 0)
    if total_connections > 0:
        # Normalize to 0-1 scale, then multiply by weight
        factors["connection_density"] = min(0.4, total_connections * 0.05)

    # Architectural position score (0.0 to 0.3)
    layer = connectivity_info.get("architectural_layer", "unknown")
    layer_scores = {
        "api": 0.3,
        "service": 0.25,
        "core": 0.2,
        "utils": 0.15,
        "unknown": 0.1,
    }
    factors["architectural_position"] = layer_scores.get(layer, 0.1)

    # Entry point factor (0.0 to 0.2)
    if connectivity_info.get("is_entry_point", False):
        factors["entry_point_factor"] = 0.2

    # Name importance score (0.0 to 0.1)
    factors["name_importance"] = _calculate_name_importance_score(func["name"])

    # Calculate total criticality score
    criticality_score = sum(factors.values())

    return min(1.0, criticality_score)


def _determine_architectural_layer(breadcrumb: str) -> str:
    """Determine the architectural layer of a function based on its breadcrumb."""
    breadcrumb_lower = breadcrumb.lower()

    if any(keyword in breadcrumb_lower for keyword in ["api", "endpoint", "handler", "route"]):
        return "api"
    elif any(keyword in breadcrumb_lower for keyword in ["service", "manager", "controller"]):
        return "service"
    elif any(keyword in breadcrumb_lower for keyword in ["core", "engine", "processor"]):
        return "core"
    elif any(keyword in breadcrumb_lower for keyword in ["util", "helper", "tool"]):
        return "utils"
    else:
        return "unknown"


def _get_architectural_layer_multiplier(layer: str) -> float:
    """Get multiplier for architectural layer importance."""
    multipliers = {
        "api": 1.5,
        "service": 1.3,
        "core": 1.2,
        "utils": 1.0,
        "unknown": 1.0,
    }
    return multipliers.get(layer, 1.0)


def _calculate_name_pattern_weight(function_name: str) -> float:
    """Calculate weight based on function name patterns."""
    name_lower = function_name.lower()

    # High importance patterns
    if any(pattern in name_lower for pattern in ["main", "init", "start", "run", "execute", "process"]):
        return 1.5
    # Medium importance patterns
    elif any(pattern in name_lower for pattern in ["get", "set", "create", "update", "delete", "save", "load"]):
        return 1.2
    # Test functions (lower importance)
    elif "test" in name_lower:
        return 0.8
    else:
        return 1.0


def _calculate_name_importance_score(function_name: str) -> float:
    """Calculate importance score based on function name patterns."""
    name_lower = function_name.lower()

    # Critical function patterns
    if any(pattern in name_lower for pattern in ["main", "init", "start", "run"]):
        return 0.1
    # Important function patterns
    elif any(pattern in name_lower for pattern in ["create", "update", "delete", "process", "execute"]):
        return 0.07
    # Standard function patterns
    elif any(pattern in name_lower for pattern in ["get", "set", "save", "load"]):
        return 0.05
    else:
        return 0.02


def _determine_hotspot_category(usage_frequency: int, criticality_score: float, connectivity_info: dict[str, Any]) -> str:
    """Determine the category of hotspot based on metrics."""
    if usage_frequency >= 10 and criticality_score >= 0.8:
        return "performance_bottleneck"
    elif connectivity_info.get("connection_count", 0) >= 8:
        return "architectural_hub"
    elif connectivity_info.get("is_entry_point", False):
        return "entry_point"
    elif criticality_score >= 0.7:
        return "critical_utility"
    else:
        return "standard"


def _calculate_performance_impact(func: dict[str, Any], usage_frequency: int, criticality_score: float) -> float:
    """Calculate performance impact score."""
    # Base impact from usage frequency (0.0 to 0.6)
    frequency_impact = min(0.6, usage_frequency * 0.06)

    # Add criticality impact (0.0 to 0.4)
    criticality_impact = criticality_score * 0.4

    # Function complexity impact (estimated from name and content)
    complexity_impact = _estimate_function_complexity_impact(func)

    return min(1.0, frequency_impact + criticality_impact + complexity_impact)


def _estimate_function_complexity_impact(func: dict[str, Any]) -> float:
    """Estimate function complexity impact for performance scoring."""
    content = func.get("content", "")
    name = func.get("name", "")

    # Base complexity indicators
    complexity_score = 0.0

    # Length-based complexity
    if len(content) > 500:
        complexity_score += 0.1
    elif len(content) > 200:
        complexity_score += 0.05

    # Pattern-based complexity
    if any(pattern in content for pattern in ["for", "while", "if", "loop"]):
        complexity_score += 0.05

    # Name-based complexity indicators
    if any(pattern in name.lower() for pattern in ["process", "calculate", "analyze", "transform"]):
        complexity_score += 0.05

    return min(0.2, complexity_score)


def _generate_hotspot_reasons(usage_frequency: int, criticality_score: float, performance_impact: float, category: str) -> list[str]:
    """Generate reasons why a function is considered a hotspot."""
    reasons = []

    if usage_frequency >= 5:
        reasons.append(f"High usage frequency ({usage_frequency} connections)")

    if criticality_score >= 0.7:
        reasons.append(f"High criticality score ({criticality_score:.2f})")

    if performance_impact >= 0.6:
        reasons.append(f"High performance impact ({performance_impact:.2f})")

    if category == "performance_bottleneck":
        reasons.append("Identified as performance bottleneck")
    elif category == "architectural_hub":
        reasons.append("Functions as architectural hub")
    elif category == "entry_point":
        reasons.append("System entry point")
    elif category == "critical_utility":
        reasons.append("Critical utility function")

    return reasons if reasons else ["Meets hotspot threshold criteria"]


async def _identify_critical_paths(
    functions: list[dict[str, Any]],
    connectivity_map: dict[str, Any],
    implementation_chain_service: ImplementationChainService,
    chain_types: list[str] = None,
) -> list[dict[str, Any]]:
    """
    Identify critical execution paths in the project.

    Args:
        functions: List of functions to analyze
        connectivity_map: Function connectivity mapping
        implementation_chain_service: Service for chain analysis
        chain_types: List of chain types to analyze (default: ["execution_flow"])

    Returns:
        List of critical paths with their metrics
    """
    critical_paths = []

    # Determine which chain types to analyze
    target_chain_types = chain_types if chain_types else ["execution_flow"]

    # Find entry points (functions with no incoming connections)
    entry_points = [func for func in functions if connectivity_map.get(func["breadcrumb"], {}).get("is_entry_point", False)]

    # For each entry point and chain type, trace critical paths
    for entry_point in entry_points[:5]:  # Limit to top 5 entry points
        for chain_type in target_chain_types:
            try:
                # Use implementation chain service to trace from entry point
                chain_result = await implementation_chain_service.trace_function_chain(
                    entry_point["breadcrumb"],
                    max_depth=8,
                    chain_type=chain_type,
                    include_cycles=False,
                )

                if chain_result and "chain_links" in chain_result:
                    # Calculate path criticality
                    path_length = len(chain_result["chain_links"])
                    path_complexity = sum(
                        connectivity_map.get(link.get("target_breadcrumb", ""), {}).get("connection_count", 0)
                        for link in chain_result["chain_links"]
                    )

                    if path_length >= 3 and path_complexity >= 10:  # Significant paths only
                        critical_paths.append(
                            {
                                "entry_point": entry_point["breadcrumb"],
                                "chain_type": chain_type,
                                "path_length": path_length,
                                "path_complexity": path_complexity,
                                "chain_links": chain_result["chain_links"][:10],  # Limit to first 10 links
                                "criticality_score": min(1.0, (path_length * 0.1) + (path_complexity * 0.02)),
                            }
                        )

            except Exception as e:
                logger.debug(f"Could not trace critical path from {entry_point['breadcrumb']} for chain type {chain_type}: {e}")
                continue

    # Sort by criticality score
    critical_paths.sort(key=lambda x: x["criticality_score"], reverse=True)

    return critical_paths[:10]  # Return top 10 critical paths


def _generate_hotspot_statistics(hotspot_data: list[dict[str, Any]], connectivity_map: dict[str, Any]) -> dict[str, Any]:
    """Generate statistics and insights from hotspot analysis."""
    if not hotspot_data:
        return {}

    # Basic statistics
    usage_frequencies = [data["usage_frequency"] for data in hotspot_data]
    criticality_scores = [data["criticality_score"] for data in hotspot_data]
    performance_impacts = [data["performance_impact"] for data in hotspot_data]

    stats = {
        "usage_frequency_stats": {
            "min": min(usage_frequencies),
            "max": max(usage_frequencies),
            "avg": sum(usage_frequencies) / len(usage_frequencies),
            "median": sorted(usage_frequencies)[len(usage_frequencies) // 2],
        },
        "criticality_stats": {
            "min": min(criticality_scores),
            "max": max(criticality_scores),
            "avg": sum(criticality_scores) / len(criticality_scores),
            "median": sorted(criticality_scores)[len(criticality_scores) // 2],
        },
        "performance_impact_stats": {
            "min": min(performance_impacts),
            "max": max(performance_impacts),
            "avg": sum(performance_impacts) / len(performance_impacts),
            "median": sorted(performance_impacts)[len(performance_impacts) // 2],
        },
        "category_distribution": {},
        "architectural_layer_distribution": {},
    }

    # Category distribution
    for data in hotspot_data:
        category = data.get("hotspot_category", "unknown")
        stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1

    # Architectural layer distribution
    for connectivity in connectivity_map.values():
        layer = connectivity.get("architectural_layer", "unknown")
        stats["architectural_layer_distribution"][layer] = stats["architectural_layer_distribution"].get(layer, 0) + 1

    return stats


def _calculate_heuristic_usage_frequency(func: dict[str, Any]) -> int:
    """Fallback heuristic calculation for usage frequency."""
    name = func.get("name", "")
    content = func.get("content", "")

    # Base frequency from function characteristics
    frequency = 1

    # Name pattern indicators
    if any(pattern in name.lower() for pattern in ["get", "set", "create", "update"]):
        frequency += 2

    if any(pattern in name.lower() for pattern in ["main", "init", "process"]):
        frequency += 3

    # Content length indicator (larger functions may be called more)
    if len(content) > 100:
        frequency += 1

    return min(10, frequency)


def _calculate_heuristic_criticality(func: dict[str, Any]) -> float:
    """Fallback heuristic calculation for criticality score."""
    name = func.get("name", "")
    breadcrumb = func.get("breadcrumb", "")

    criticality = 0.0

    # Name-based criticality
    if any(pattern in name.lower() for pattern in ["main", "init", "start"]):
        criticality += 0.4

    if any(pattern in name.lower() for pattern in ["process", "execute", "run"]):
        criticality += 0.3

    # Position-based criticality
    if any(pattern in breadcrumb.lower() for pattern in ["api", "service", "core"]):
        criticality += 0.2

    return min(1.0, criticality)


async def _perform_coverage_analysis(
    functions: list[dict[str, Any]],
    implementation_chain_service: ImplementationChainService,
    batch_size: int,
    chain_types: list[str] = None,
) -> dict[str, Any]:
    """
    Perform comprehensive coverage analysis to identify function connectivity patterns.

    This analysis includes:
    1. Function connectivity mapping - incoming and outgoing connections
    2. Coverage percentage calculation - functions with vs without connections
    3. Connectivity statistics - distribution of connection patterns
    4. Isolation detection - functions with no connections
    5. Hub identification - highly connected functions
    6. Network analysis - connectivity strength and clustering

    Args:
        functions: List of functions to analyze
        implementation_chain_service: Service for chain analysis
        batch_size: Batch size for processing
        chain_types: List of chain types to filter analysis (default: all types)

    Returns:
        Dictionary with comprehensive coverage analysis results
    """
    logger.info(f"Performing coverage analysis on {len(functions)} functions")

    # Step 1: Build comprehensive connectivity map
    connectivity_map = await _build_comprehensive_connectivity_map(functions, implementation_chain_service, chain_types)

    # Step 2: Analyze function coverage in batches
    covered_functions = []
    uncovered_functions = []
    all_connectivity_data = []

    for i in range(0, len(functions), batch_size):
        batch = functions[i : i + batch_size]
        logger.debug(f"Processing coverage batch {i // batch_size + 1}/{(len(functions) + batch_size - 1) // batch_size}")

        for func in batch:
            breadcrumb = func["breadcrumb"]
            connectivity_info = connectivity_map.get(breadcrumb, {})

            # Calculate detailed connectivity metrics
            connectivity_metrics = _calculate_detailed_connectivity_metrics(func, connectivity_info, connectivity_map)

            all_connectivity_data.append(connectivity_metrics)

            # Determine if function is covered (has meaningful connections)
            is_covered = _determine_coverage_status(connectivity_metrics)

            if is_covered:
                covered_functions.append(connectivity_metrics)
            else:
                uncovered_functions.append(
                    {
                        "breadcrumb": breadcrumb,
                        "name": func["name"],
                        "file_path": func["file_path"],
                        "isolation_reasons": _analyze_isolation_reasons(connectivity_metrics),
                        "connectivity_score": connectivity_metrics["connectivity_score"],
                        "suggestion": _generate_coverage_suggestion(connectivity_metrics),
                    }
                )

    # Step 3: Calculate comprehensive connectivity statistics
    connectivity_statistics = _calculate_comprehensive_connectivity_statistics(all_connectivity_data, connectivity_map)

    # Step 4: Identify network patterns and clusters
    network_analysis = _analyze_network_patterns(connectivity_map, all_connectivity_data)

    # Step 5: Calculate coverage metrics
    coverage_percentage = (len(covered_functions) / len(functions)) * 100 if functions else 0

    # Step 6: Generate coverage insights and recommendations
    coverage_insights = _generate_coverage_insights(covered_functions, uncovered_functions, connectivity_statistics)

    return {
        "total_functions_analyzed": len(functions),
        "covered_functions_count": len(covered_functions),
        "uncovered_functions_count": len(uncovered_functions),
        "coverage_percentage": coverage_percentage,
        "covered_functions": covered_functions[:50],  # Limit for performance
        "uncovered_functions": uncovered_functions,
        "connectivity_statistics": connectivity_statistics,
        "network_analysis": network_analysis,
        "coverage_insights": coverage_insights,
        "connectivity_thresholds": {
            "minimum_connections_for_coverage": 1,
            "highly_connected_threshold": 8,
            "isolated_function_threshold": 0,
            "hub_function_threshold": 10,
        },
        "batch_processing": {
            "batch_size": batch_size,
            "total_batches": (len(functions) + batch_size - 1) // batch_size,
        },
    }


async def _build_comprehensive_connectivity_map(
    functions: list[dict[str, Any]],
    implementation_chain_service: ImplementationChainService,
    chain_types: list[str] = None,
) -> dict[str, Any]:
    """
    Build a comprehensive connectivity map for all functions.

    Args:
        functions: List of functions to analyze
        implementation_chain_service: Service for chain analysis
        chain_types: List of chain types to filter relationships (default: all types)

    Returns:
        Dictionary mapping breadcrumbs to connectivity information
    """
    connectivity_map = {}

    # Determine which chain types to analyze
    target_chain_types = chain_types if chain_types else ["execution_flow", "data_flow", "dependency_chain"]

    # Map relationship types to chain types
    relationship_to_chain_map = {
        "call": "execution_flow",
        "import": "dependency_chain",
        "inherit": "execution_flow",
        "implement": "execution_flow",
        "reference": "data_flow",
        "dependency": "dependency_chain",
        "data_transform": "data_flow",
        "invoke": "execution_flow",
        "access": "data_flow",
    }

    try:
        # Get project structure graph for detailed connectivity analysis
        project_name = implementation_chain_service.project_name if hasattr(implementation_chain_service, "project_name") else "unknown"
        project_graph = await implementation_chain_service.graph_rag_service.build_structure_graph(project_name, force_rebuild=True)

        if project_graph:
            # Initialize connectivity map from graph nodes
            for node in project_graph.nodes:
                if node.chunk_type in [ChunkType.FUNCTION, ChunkType.METHOD]:
                    connectivity_map[node.breadcrumb] = {
                        "incoming_connections": [],
                        "outgoing_connections": [],
                        "connection_strength": {},
                        "architectural_layer": _determine_architectural_layer(node.breadcrumb),
                        "function_type": _classify_function_type(node.name),
                        "file_path": node.file_path,
                        "is_entry_point": True,  # Will be updated based on connections
                        "cluster_id": None,  # For network clustering analysis
                    }

            # Build connections from relationships, filtered by chain types
            for relationship in getattr(project_graph, "relationships", []):
                source_breadcrumb = getattr(relationship, "source_breadcrumb", "")
                target_breadcrumb = getattr(relationship, "target_breadcrumb", "")
                relationship_type = getattr(relationship, "relationship_type", "call")

                # Map relationship type to chain type
                mapped_chain_type = relationship_to_chain_map.get(relationship_type, "execution_flow")

                # Only include relationships that match target chain types
                if mapped_chain_type in target_chain_types:
                    if source_breadcrumb in connectivity_map:
                        connectivity_map[source_breadcrumb]["outgoing_connections"].append(target_breadcrumb)
                        connectivity_map[source_breadcrumb]["connection_strength"][target_breadcrumb] = _calculate_connection_strength(
                            relationship_type
                        )
                        connectivity_map[source_breadcrumb]["is_entry_point"] = False

                    if target_breadcrumb in connectivity_map:
                        connectivity_map[target_breadcrumb]["incoming_connections"].append(source_breadcrumb)

    except Exception as e:
        logger.warning(f"Could not build comprehensive connectivity map: {e}. Using fallback analysis.")

    # Fallback analysis for functions not in graph
    for func in functions:
        breadcrumb = func["breadcrumb"]
        if breadcrumb not in connectivity_map:
            connectivity_map[breadcrumb] = {
                "incoming_connections": [],
                "outgoing_connections": [],
                "connection_strength": {},
                "architectural_layer": _determine_architectural_layer(breadcrumb),
                "function_type": _classify_function_type(func["name"]),
                "file_path": func["file_path"],
                "is_entry_point": _is_likely_entry_point(func),
                "cluster_id": None,
            }

    return connectivity_map


def _classify_function_type(function_name: str) -> str:
    """Classify function type based on name patterns."""
    name_lower = function_name.lower()

    if any(pattern in name_lower for pattern in ["test_", "_test", "test"]):
        return "test"
    elif any(pattern in name_lower for pattern in ["main", "init", "start"]):
        return "entry_point"
    elif any(pattern in name_lower for pattern in ["get_", "fetch_", "read_"]):
        return "getter"
    elif any(pattern in name_lower for pattern in ["set_", "write_", "save_", "create_"]):
        return "setter"
    elif any(pattern in name_lower for pattern in ["process_", "calculate_", "analyze_"]):
        return "processor"
    elif any(pattern in name_lower for pattern in ["_helper", "_util", "_format"]):
        return "utility"
    else:
        return "standard"


def _calculate_connection_strength(relationship_type: str) -> float:
    """Calculate connection strength based on relationship type."""
    strength_map = {
        "call": 1.0,
        "import": 0.8,
        "inherit": 0.9,
        "implement": 0.7,
        "reference": 0.5,
        "dependency": 0.6,
    }
    return strength_map.get(relationship_type, 0.5)


def _is_likely_entry_point(func: dict[str, Any]) -> bool:
    """Determine if a function is likely an entry point."""
    name = func.get("name", "").lower()
    breadcrumb = func.get("breadcrumb", "").lower()

    # Check for common entry point patterns
    entry_patterns = ["main", "init", "start", "run", "execute", "handler", "endpoint"]
    return any(pattern in name or pattern in breadcrumb for pattern in entry_patterns)


def _calculate_detailed_connectivity_metrics(
    func: dict[str, Any],
    connectivity_info: dict[str, Any],
    full_connectivity_map: dict[str, Any],
) -> dict[str, Any]:
    """Calculate detailed connectivity metrics for a function."""
    breadcrumb = func["breadcrumb"]

    # Basic connection counts
    incoming_count = len(connectivity_info.get("incoming_connections", []))
    outgoing_count = len(connectivity_info.get("outgoing_connections", []))
    total_connections = incoming_count + outgoing_count

    # Calculate connection strength score
    connection_strengths = list(connectivity_info.get("connection_strength", {}).values())
    avg_connection_strength = sum(connection_strengths) / len(connection_strengths) if connection_strengths else 0.0

    # Calculate connectivity score (0.0 to 1.0)
    connectivity_score = _calculate_connectivity_score(incoming_count, outgoing_count, avg_connection_strength, connectivity_info)

    # Calculate network centrality metrics
    centrality_metrics = _calculate_network_centrality(breadcrumb, full_connectivity_map)

    # Determine connectivity category
    connectivity_category = _determine_connectivity_category(incoming_count, outgoing_count, connectivity_score)

    return {
        "breadcrumb": breadcrumb,
        "name": func["name"],
        "file_path": func["file_path"],
        "incoming_connections": incoming_count,
        "outgoing_connections": outgoing_count,
        "total_connections": total_connections,
        "connectivity_score": connectivity_score,
        "connectivity_category": connectivity_category,
        "connection_strength": avg_connection_strength,
        "centrality_metrics": centrality_metrics,
        "architectural_layer": connectivity_info.get("architectural_layer", "unknown"),
        "function_type": connectivity_info.get("function_type", "standard"),
        "is_entry_point": connectivity_info.get("is_entry_point", False),
        "connected_functions": {
            "incoming": connectivity_info.get("incoming_connections", [])[:5],  # Limit for performance
            "outgoing": connectivity_info.get("outgoing_connections", [])[:5],
        },
    }


def _calculate_connectivity_score(
    incoming_count: int,
    outgoing_count: int,
    avg_strength: float,
    connectivity_info: dict[str, Any],
) -> float:
    """Calculate overall connectivity score."""
    # Base score from connection counts (0.0 to 0.6)
    connection_score = min(0.6, (incoming_count + outgoing_count) * 0.05)

    # Strength bonus (0.0 to 0.2)
    strength_bonus = avg_strength * 0.2

    # Architectural layer bonus (0.0 to 0.1)
    layer = connectivity_info.get("architectural_layer", "unknown")
    layer_bonus = {"api": 0.1, "service": 0.08, "core": 0.06, "utils": 0.04, "unknown": 0.02}.get(layer, 0.02)

    # Entry point bonus (0.0 to 0.1)
    entry_bonus = 0.1 if connectivity_info.get("is_entry_point", False) else 0.0

    total_score = connection_score + strength_bonus + layer_bonus + entry_bonus
    return min(1.0, total_score)


def _calculate_network_centrality(breadcrumb: str, connectivity_map: dict[str, Any]) -> dict[str, float]:
    """Calculate network centrality metrics for a function."""
    # Simplified centrality calculations
    function_info = connectivity_map.get(breadcrumb, {})

    incoming_count = len(function_info.get("incoming_connections", []))
    outgoing_count = len(function_info.get("outgoing_connections", []))
    total_functions = len(connectivity_map)

    # Degree centrality (normalized by total functions)
    degree_centrality = (incoming_count + outgoing_count) / max(1, total_functions - 1)

    # In-degree centrality
    indegree_centrality = incoming_count / max(1, total_functions - 1)

    # Out-degree centrality
    outdegree_centrality = outgoing_count / max(1, total_functions - 1)

    # Betweenness centrality (simplified estimation)
    betweenness_centrality = _estimate_betweenness_centrality(breadcrumb, connectivity_map)

    return {
        "degree_centrality": degree_centrality,
        "indegree_centrality": indegree_centrality,
        "outdegree_centrality": outdegree_centrality,
        "betweenness_centrality": betweenness_centrality,
    }


def _estimate_betweenness_centrality(breadcrumb: str, connectivity_map: dict[str, Any]) -> float:
    """Estimate betweenness centrality (simplified calculation)."""
    function_info = connectivity_map.get(breadcrumb, {})
    incoming = set(function_info.get("incoming_connections", []))
    outgoing = set(function_info.get("outgoing_connections", []))

    # Simple estimation: functions that bridge different clusters
    if incoming and outgoing:
        # Check if this function connects different parts of the graph
        bridge_score = len(incoming.intersection(outgoing)) / max(1, len(incoming.union(outgoing)))
        return min(1.0, bridge_score * 2)  # Scale up bridge functions

    return 0.0


def _determine_connectivity_category(incoming_count: int, outgoing_count: int, connectivity_score: float) -> str:
    """Determine connectivity category for a function."""
    total_connections = incoming_count + outgoing_count

    if total_connections == 0:
        return "isolated"
    elif total_connections >= 10:
        return "hub"
    elif incoming_count >= 5 and outgoing_count <= 2:
        return "sink"
    elif outgoing_count >= 5 and incoming_count <= 2:
        return "source"
    elif total_connections >= 4:
        return "well_connected"
    else:
        return "lightly_connected"


def _determine_coverage_status(connectivity_metrics: dict[str, Any]) -> bool:
    """Determine if a function is considered 'covered' based on connectivity."""
    # A function is considered covered if it has meaningful connections
    total_connections = connectivity_metrics.get("total_connections", 0)
    connectivity_score = connectivity_metrics.get("connectivity_score", 0.0)
    is_entry_point = connectivity_metrics.get("is_entry_point", False)

    # Entry points are always considered covered
    if is_entry_point:
        return True

    # Functions with connections or decent connectivity score are covered
    return total_connections > 0 or connectivity_score > 0.2


def _analyze_isolation_reasons(connectivity_metrics: dict[str, Any]) -> list[str]:
    """Analyze why a function might be isolated."""
    reasons = []

    if connectivity_metrics.get("total_connections", 0) == 0:
        reasons.append("No incoming or outgoing connections found")

    function_type = connectivity_metrics.get("function_type", "standard")
    if function_type == "test":
        reasons.append("Test function - may be intentionally isolated")
    elif function_type == "utility":
        reasons.append("Utility function - may be called indirectly")

    architectural_layer = connectivity_metrics.get("architectural_layer", "unknown")
    if architectural_layer == "unknown":
        reasons.append("Function in unclassified architectural layer")

    connectivity_score = connectivity_metrics.get("connectivity_score", 0.0)
    if connectivity_score < 0.1:
        reasons.append("Very low connectivity score")

    return reasons if reasons else ["Isolated for unknown reasons"]


def _generate_coverage_suggestion(connectivity_metrics: dict[str, Any]) -> str:
    """Generate suggestions for improving function coverage."""
    function_type = connectivity_metrics.get("function_type", "standard")
    architectural_layer = connectivity_metrics.get("architectural_layer", "unknown")
    total_connections = connectivity_metrics.get("total_connections", 0)

    if function_type == "test":
        return "Test function isolation is expected"
    elif total_connections == 0:
        return "Consider if this function is still needed or should be integrated"
    elif architectural_layer == "unknown":
        return "Consider moving to appropriate architectural layer"
    else:
        return "Review function usage and consider refactoring if needed"


def _calculate_comprehensive_connectivity_statistics(
    all_connectivity_data: list[dict[str, Any]],
    connectivity_map: dict[str, Any],
) -> dict[str, Any]:
    """Calculate comprehensive connectivity statistics."""
    if not all_connectivity_data:
        return {}

    # Extract metrics
    incoming_counts = [data["incoming_connections"] for data in all_connectivity_data]
    outgoing_counts = [data["outgoing_connections"] for data in all_connectivity_data]
    total_connections = [data["total_connections"] for data in all_connectivity_data]
    connectivity_scores = [data["connectivity_score"] for data in all_connectivity_data]

    # Calculate basic statistics
    stats = {
        "connection_statistics": {
            "incoming_connections": {
                "min": min(incoming_counts),
                "max": max(incoming_counts),
                "avg": sum(incoming_counts) / len(incoming_counts),
                "median": sorted(incoming_counts)[len(incoming_counts) // 2],
            },
            "outgoing_connections": {
                "min": min(outgoing_counts),
                "max": max(outgoing_counts),
                "avg": sum(outgoing_counts) / len(outgoing_counts),
                "median": sorted(outgoing_counts)[len(outgoing_counts) // 2],
            },
            "total_connections": {
                "min": min(total_connections),
                "max": max(total_connections),
                "avg": sum(total_connections) / len(total_connections),
                "median": sorted(total_connections)[len(total_connections) // 2],
            },
        },
        "connectivity_score_statistics": {
            "min": min(connectivity_scores),
            "max": max(connectivity_scores),
            "avg": sum(connectivity_scores) / len(connectivity_scores),
            "median": sorted(connectivity_scores)[len(connectivity_scores) // 2],
        },
        "category_distribution": {},
        "function_type_distribution": {},
        "architectural_layer_distribution": {},
        "centrality_statistics": {},
    }

    # Category distribution
    for data in all_connectivity_data:
        category = data.get("connectivity_category", "unknown")
        stats["category_distribution"][category] = stats["category_distribution"].get(category, 0) + 1

    # Function type distribution
    for data in all_connectivity_data:
        func_type = data.get("function_type", "standard")
        stats["function_type_distribution"][func_type] = stats["function_type_distribution"].get(func_type, 0) + 1

    # Architectural layer distribution
    for data in all_connectivity_data:
        layer = data.get("architectural_layer", "unknown")
        stats["architectural_layer_distribution"][layer] = stats["architectural_layer_distribution"].get(layer, 0) + 1

    # Centrality statistics
    degree_centralities = [data["centrality_metrics"]["degree_centrality"] for data in all_connectivity_data]
    betweenness_centralities = [data["centrality_metrics"]["betweenness_centrality"] for data in all_connectivity_data]

    stats["centrality_statistics"] = {
        "degree_centrality": {
            "min": min(degree_centralities),
            "max": max(degree_centralities),
            "avg": sum(degree_centralities) / len(degree_centralities),
        },
        "betweenness_centrality": {
            "min": min(betweenness_centralities),
            "max": max(betweenness_centralities),
            "avg": sum(betweenness_centralities) / len(betweenness_centralities),
        },
    }

    # Advanced metrics
    stats["advanced_metrics"] = {
        "highly_connected_functions": len([d for d in all_connectivity_data if d["total_connections"] >= 8]),
        "isolated_functions": len([d for d in all_connectivity_data if d["total_connections"] == 0]),
        "hub_functions": len([d for d in all_connectivity_data if d["connectivity_category"] == "hub"]),
        "entry_point_functions": len([d for d in all_connectivity_data if d["is_entry_point"]]),
        "bridge_functions": len([d for d in all_connectivity_data if d["centrality_metrics"]["betweenness_centrality"] > 0.1]),
    }

    return stats


def _analyze_network_patterns(connectivity_map: dict[str, Any], all_connectivity_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Analyze network patterns and clustering."""
    # Calculate network density
    total_functions = len(connectivity_map)
    total_possible_connections = total_functions * (total_functions - 1)
    actual_connections = sum(len(info.get("outgoing_connections", [])) for info in connectivity_map.values())
    network_density = actual_connections / max(1, total_possible_connections)

    # Identify clusters (simplified approach)
    clusters = _identify_function_clusters(connectivity_map)

    # Calculate modularity (simplified)
    modularity_score = _calculate_network_modularity(connectivity_map, clusters)

    return {
        "network_density": network_density,
        "total_functions": total_functions,
        "total_connections": actual_connections,
        "cluster_analysis": {
            "number_of_clusters": len(clusters),
            "cluster_sizes": [len(cluster) for cluster in clusters.values()],
            "modularity_score": modularity_score,
        },
        "connectivity_patterns": {
            "strongly_connected_components": len([c for c in clusters.values() if len(c) > 1]),
            "isolated_components": len([c for c in clusters.values() if len(c) == 1]),
        },
    }


def _identify_function_clusters(connectivity_map: dict[str, Any]) -> dict[int, list[str]]:
    """Identify function clusters using simplified algorithm."""
    clusters = {}
    visited = set()
    cluster_id = 0

    for breadcrumb in connectivity_map:
        if breadcrumb not in visited:
            cluster = _explore_cluster(breadcrumb, connectivity_map, visited)
            if cluster:
                clusters[cluster_id] = cluster
                cluster_id += 1

    return clusters


def _explore_cluster(start_breadcrumb: str, connectivity_map: dict[str, Any], visited: set) -> list[str]:
    """Explore connected components to form clusters."""
    cluster = []
    stack = [start_breadcrumb]

    while stack:
        current = stack.pop()
        if current in visited:
            continue

        visited.add(current)
        cluster.append(current)

        # Add connected functions
        function_info = connectivity_map.get(current, {})
        connected = function_info.get("incoming_connections", []) + function_info.get("outgoing_connections", [])

        for connected_func in connected:
            if connected_func not in visited and connected_func in connectivity_map:
                stack.append(connected_func)

    return cluster


def _calculate_network_modularity(connectivity_map: dict[str, Any], clusters: dict[int, list[str]]) -> float:
    """Calculate network modularity (simplified)."""
    if not clusters:
        return 0.0

    total_edges = sum(len(info.get("outgoing_connections", [])) for info in connectivity_map.values())
    if total_edges == 0:
        return 0.0

    modularity = 0.0
    for cluster in clusters.values():
        internal_edges = 0
        cluster_degree = 0

        for func in cluster:
            func_info = connectivity_map.get(func, {})
            outgoing = func_info.get("outgoing_connections", [])
            cluster_degree += len(outgoing)

            # Count internal edges
            internal_edges += len([conn for conn in outgoing if conn in cluster])

        if total_edges > 0:
            expected_internal = (cluster_degree**2) / (2 * total_edges)
            modularity += (internal_edges - expected_internal) / total_edges

    return modularity


def _generate_coverage_insights(
    covered_functions: list[dict[str, Any]],
    uncovered_functions: list[dict[str, Any]],
    connectivity_statistics: dict[str, Any],
) -> list[str]:
    """Generate insights about function coverage."""
    insights = []

    total_functions = len(covered_functions) + len(uncovered_functions)
    coverage_percentage = (len(covered_functions) / total_functions) * 100 if total_functions > 0 else 0

    # Coverage insights
    if coverage_percentage > 80:
        insights.append(f"Excellent connectivity: {coverage_percentage:.1f}% of functions are well-connected")
    elif coverage_percentage > 60:
        insights.append(f"Good connectivity: {coverage_percentage:.1f}% of functions have connections")
    else:
        insights.append(f"Poor connectivity: Only {coverage_percentage:.1f}% of functions are connected")

    # Hub function insights
    hub_count = connectivity_statistics.get("advanced_metrics", {}).get("hub_functions", 0)
    if hub_count > 0:
        insights.append(f"Architecture has {hub_count} hub functions that coordinate many connections")

    # Isolation insights
    isolated_count = len(uncovered_functions)
    if isolated_count > total_functions * 0.3:
        insights.append(f"High isolation: {isolated_count} functions may need architectural review")
    elif isolated_count > 0:
        insights.append(f"Some isolation detected: {isolated_count} functions have no connections")

    # Entry point insights
    entry_points = connectivity_statistics.get("advanced_metrics", {}).get("entry_point_functions", 0)
    if entry_points == 0:
        insights.append("No clear entry points identified - consider architectural patterns")
    elif entry_points > 1:
        insights.append(f"Multiple entry points detected: {entry_points} functions serve as system entry points")

    return insights


async def _generate_refactoring_recommendations(
    analysis_results: dict[str, Any],
    functions: list[dict[str, Any]],
    min_complexity_threshold: float,
) -> dict[str, Any]:
    """
    Generate comprehensive refactoring recommendations based on detailed analysis results.

    This function analyzes multiple dimensions of code quality including:
    1. Complexity analysis - functions that exceed complexity thresholds
    2. Hotspot analysis - performance and architectural bottlenecks
    3. Coverage analysis - connectivity and isolation issues
    4. Architectural patterns - layer violations and design issues
    5. Technical debt - maintainability and code quality concerns

    Args:
        analysis_results: Results from various analyses
        functions: List of functions
        min_complexity_threshold: Minimum complexity threshold

    Returns:
        Dictionary with comprehensive refactoring recommendations
    """
    logger.info("Generating comprehensive refactoring recommendations")

    recommendations = []

    # Advanced complexity-based recommendations
    if "complexity_analysis" in analysis_results:
        complexity_recommendations = _generate_complexity_refactoring_recommendations(
            analysis_results["complexity_analysis"], min_complexity_threshold
        )
        recommendations.extend(complexity_recommendations)

    # Hotspot-based performance recommendations
    if "hotspot_analysis" in analysis_results:
        hotspot_recommendations = _generate_hotspot_refactoring_recommendations(analysis_results["hotspot_analysis"])
        recommendations.extend(hotspot_recommendations)

    # Coverage and connectivity recommendations
    if "coverage_analysis" in analysis_results:
        coverage_recommendations = _generate_coverage_refactoring_recommendations(analysis_results["coverage_analysis"])
        recommendations.extend(coverage_recommendations)

    # Cross-analysis pattern detection
    cross_analysis_recommendations = _generate_cross_analysis_recommendations(analysis_results, functions)
    recommendations.extend(cross_analysis_recommendations)

    # Architectural recommendations
    architectural_recommendations = _generate_architectural_recommendations(analysis_results, functions)
    recommendations.extend(architectural_recommendations)

    # Remove duplicates and merge similar recommendations
    recommendations = _deduplicate_and_merge_recommendations(recommendations)

    # Calculate refactoring impact and effort
    recommendations = _calculate_refactoring_impact(recommendations, analysis_results)

    # Group recommendations by priority and type
    recommendations_by_priority = {
        "critical": [r for r in recommendations if r["priority"] == "critical"],
        "high": [r for r in recommendations if r["priority"] == "high"],
        "medium": [r for r in recommendations if r["priority"] == "medium"],
        "low": [r for r in recommendations if r["priority"] == "low"],
    }

    recommendations_by_type = {}
    for rec in recommendations:
        rec_type = rec.get("type", "unknown")
        if rec_type not in recommendations_by_type:
            recommendations_by_type[rec_type] = []
        recommendations_by_type[rec_type].append(rec)

    # Generate refactoring strategy recommendations
    refactoring_strategy = _generate_refactoring_strategy(recommendations, analysis_results)

    # Calculate technical debt metrics
    technical_debt_metrics = _calculate_technical_debt_metrics(recommendations, analysis_results)

    return {
        "total_recommendations": len(recommendations),
        "recommendations_by_priority": recommendations_by_priority,
        "recommendations_by_type": recommendations_by_type,
        "all_recommendations": recommendations,
        "refactoring_strategy": refactoring_strategy,
        "technical_debt_metrics": technical_debt_metrics,
        "summary": {
            "critical_priority_count": len(recommendations_by_priority["critical"]),
            "high_priority_count": len(recommendations_by_priority["high"]),
            "medium_priority_count": len(recommendations_by_priority["medium"]),
            "low_priority_count": len(recommendations_by_priority["low"]),
            "most_common_issues": _get_most_common_issues(recommendations),
            "estimated_refactoring_effort": _estimate_total_refactoring_effort(recommendations),
        },
    }


def _generate_complexity_refactoring_recommendations(complexity_analysis: dict[str, Any], min_threshold: float) -> list[dict[str, Any]]:
    """Generate refactoring recommendations based on complexity analysis."""
    recommendations = []
    complex_functions = complexity_analysis.get("complex_functions", [])

    for func in complex_functions:
        complexity_score = func.get("complexity_score", 0.0)
        complexity_breakdown = func.get("complexity_breakdown", {})
        raw_metrics = func.get("raw_metrics", {})

        # Determine recommendation priority based on complexity level
        if complexity_score > 0.9:
            priority = "critical"
        elif complexity_score > 0.8:
            priority = "high"
        elif complexity_score > 0.6:
            priority = "medium"
        else:
            priority = "low"

        # Generate specific recommendations based on complexity breakdown
        specific_suggestions = _generate_complexity_specific_suggestions(complexity_breakdown, raw_metrics)

        recommendations.append(
            {
                "type": "complexity_reduction",
                "priority": priority,
                "target_function": func["breadcrumb"],
                "issue": f"High complexity score ({complexity_score:.2f})",
                "suggestions": specific_suggestions,
                "complexity_metrics": {
                    "complexity_score": complexity_score,
                    "complexity_category": func.get("complexity_category", "unknown"),
                    "breakdown": complexity_breakdown,
                    "raw_metrics": raw_metrics,
                },
                "estimated_effort": _estimate_complexity_refactoring_effort(complexity_score, raw_metrics),
                "impact": "maintainability",
            }
        )

    return recommendations


def _generate_complexity_specific_suggestions(complexity_breakdown: dict[str, Any], raw_metrics: dict[str, Any]) -> list[str]:
    """Generate specific suggestions based on complexity breakdown."""
    suggestions = []

    # Analyze the highest contributing factors
    branching_contribution = complexity_breakdown.get("branching_factor", 0.0)
    cyclomatic_contribution = complexity_breakdown.get("cyclomatic_complexity", 0.0)
    call_depth_contribution = complexity_breakdown.get("call_depth", 0.0)
    length_contribution = complexity_breakdown.get("function_length", 0.0)

    # Branching complexity suggestions
    if branching_contribution > 0.3:
        branching_count = raw_metrics.get("branching_factor", 0)
        suggestions.append(
            f"Reduce branching complexity ({branching_count} branches) by extracting conditional logic into separate functions"
        )
        if branching_count > 8:
            suggestions.append("Consider using polymorphism or strategy pattern to replace complex conditional logic")

    # Cyclomatic complexity suggestions
    if cyclomatic_contribution > 0.3:
        cyclomatic_score = raw_metrics.get("cyclomatic_complexity", 0)
        suggestions.append(f"Reduce cyclomatic complexity ({cyclomatic_score}) by simplifying control flow")
        if cyclomatic_score > 15:
            suggestions.append("Function has very high cyclomatic complexity - consider splitting into multiple smaller functions")

    # Call depth suggestions
    if call_depth_contribution > 0.3:
        call_depth = raw_metrics.get("call_depth", 0)
        suggestions.append(f"Reduce call depth ({call_depth} levels) by flattening function call hierarchy")
        if call_depth > 6:
            suggestions.append("Consider introducing intermediate functions to reduce deep nesting")

    # Function length suggestions
    if length_contribution > 0.3:
        func_length = raw_metrics.get("function_length", 0)
        suggestions.append(f"Reduce function length ({func_length} lines) by extracting logical blocks into separate functions")
        if func_length > 100:
            suggestions.append("Function is very long - apply Extract Method refactoring pattern")

    # General suggestions if none specific
    if not suggestions:
        suggestions.append("Consider breaking this function into smaller, more focused functions")

    return suggestions


def _generate_hotspot_refactoring_recommendations(hotspot_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate refactoring recommendations based on hotspot analysis."""
    recommendations = []
    hotspot_functions = hotspot_analysis.get("hotspot_functions", [])

    for func in hotspot_functions:
        usage_frequency = func.get("usage_frequency", 0)
        criticality_score = func.get("criticality_score", 0.0)
        performance_impact = func.get("performance_impact", 0.0)
        hotspot_category = func.get("hotspot_category", "standard")

        # Determine priority based on hotspot severity
        if performance_impact > 0.8:
            priority = "critical"
        elif usage_frequency > 15 or criticality_score > 0.8:
            priority = "high"
        elif usage_frequency > 8 or criticality_score > 0.6:
            priority = "medium"
        else:
            priority = "low"

        # Generate category-specific suggestions
        suggestions = _generate_hotspot_specific_suggestions(hotspot_category, usage_frequency, criticality_score, performance_impact)

        recommendations.append(
            {
                "type": "hotspot_optimization",
                "priority": priority,
                "target_function": func["breadcrumb"],
                "issue": f"Performance hotspot ({hotspot_category})",
                "suggestions": suggestions,
                "hotspot_metrics": {
                    "usage_frequency": usage_frequency,
                    "criticality_score": criticality_score,
                    "performance_impact": performance_impact,
                    "category": hotspot_category,
                    "reasons": func.get("hotspot_reasons", []),
                },
                "estimated_effort": _estimate_hotspot_refactoring_effort(hotspot_category, performance_impact),
                "impact": "performance",
            }
        )

    return recommendations


def _generate_hotspot_specific_suggestions(
    category: str, usage_frequency: int, criticality_score: float, performance_impact: float
) -> list[str]:
    """Generate specific suggestions based on hotspot category."""
    suggestions = []

    if category == "performance_bottleneck":
        suggestions.extend(
            [
                "Optimize algorithm complexity and data structures",
                "Consider caching frequently computed results",
                "Profile function execution to identify specific bottlenecks",
                "Consider asynchronous processing for I/O bound operations",
            ]
        )

    elif category == "architectural_hub":
        suggestions.extend(
            [
                "Consider breaking down into smaller, more focused functions",
                "Apply dependency injection to reduce coupling",
                "Implement facade or mediator pattern to manage complexity",
                "Consider event-driven architecture to reduce direct dependencies",
            ]
        )

    elif category == "entry_point":
        suggestions.extend(
            [
                "Minimize logic in entry point functions",
                "Delegate work to specialized service functions",
                "Implement proper error handling and validation",
                "Consider using command pattern for complex entry point logic",
            ]
        )

    elif category == "critical_utility":
        suggestions.extend(
            [
                "Ensure comprehensive unit test coverage",
                "Consider defensive programming practices",
                "Optimize for reliability and error handling",
                "Document usage patterns and constraints clearly",
            ]
        )

    # Add frequency-based suggestions
    if usage_frequency > 20:
        suggestions.append("Extremely high usage - consider performance profiling and optimization")
    elif usage_frequency > 10:
        suggestions.append("High usage frequency - ensure optimal performance")

    return suggestions


def _generate_coverage_refactoring_recommendations(coverage_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate refactoring recommendations based on coverage analysis."""
    recommendations = []
    uncovered_functions = coverage_analysis.get("uncovered_functions", [])

    for func in uncovered_functions:
        isolation_reasons = func.get("isolation_reasons", [])
        connectivity_score = func.get("connectivity_score", 0.0)

        # Determine priority based on isolation severity
        if "No incoming or outgoing connections found" in isolation_reasons:
            priority = "medium"
        elif "Very low connectivity score" in isolation_reasons:
            priority = "low"
        else:
            priority = "low"

        # Generate isolation-specific suggestions
        suggestions = _generate_isolation_specific_suggestions(isolation_reasons, func)

        recommendations.append(
            {
                "type": "connectivity_improvement",
                "priority": priority,
                "target_function": func["breadcrumb"],
                "issue": "Function isolation detected",
                "suggestions": suggestions,
                "isolation_metrics": {
                    "connectivity_score": connectivity_score,
                    "isolation_reasons": isolation_reasons,
                    "suggestion": func.get("suggestion", ""),
                },
                "estimated_effort": "low",
                "impact": "architecture",
            }
        )

    return recommendations


def _generate_isolation_specific_suggestions(isolation_reasons: list[str], func_info: dict[str, Any]) -> list[str]:
    """Generate specific suggestions for isolated functions."""
    suggestions = []

    if "Test function - may be intentionally isolated" in isolation_reasons:
        suggestions.append("Test function isolation is expected and acceptable")
        return suggestions

    if "No incoming or outgoing connections found" in isolation_reasons:
        suggestions.extend(
            [
                "Review if this function is still needed in the codebase",
                "Consider removing if it's dead code",
                "If needed, integrate it into the application flow",
            ]
        )

    if "Utility function - may be called indirectly" in isolation_reasons:
        suggestions.extend(
            [
                "Verify that utility function is actually being used",
                "Consider making it a static method or module-level function",
                "Document its purpose and usage patterns",
            ]
        )

    if "Function in unclassified architectural layer" in isolation_reasons:
        suggestions.extend(
            [
                "Move function to appropriate architectural layer",
                "Follow established project structure patterns",
                "Consider if function belongs in service, utility, or domain layer",
            ]
        )

    return suggestions if suggestions else ["Review function necessity and integration"]


def _generate_cross_analysis_recommendations(analysis_results: dict[str, Any], functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate recommendations based on cross-analysis patterns."""
    recommendations = []

    # Functions that are both complex and hotspots (highest priority)
    complex_hotspots = _identify_complex_hotspots(analysis_results)
    for func_breadcrumb in complex_hotspots:
        recommendations.append(
            {
                "type": "critical_refactoring",
                "priority": "critical",
                "target_function": func_breadcrumb,
                "issue": "Function is both complex and a performance hotspot",
                "suggestions": [
                    "Immediate refactoring required - high complexity + high usage",
                    "Break down into smaller, optimized functions",
                    "Consider performance profiling before refactoring",
                    "Implement comprehensive testing before changes",
                ],
                "estimated_effort": "high",
                "impact": "performance_and_maintainability",
            }
        )

    # Functions with high complexity but low connectivity (dead code candidates)
    complex_isolated = _identify_complex_isolated_functions(analysis_results)
    for func_breadcrumb in complex_isolated:
        recommendations.append(
            {
                "type": "dead_code_analysis",
                "priority": "medium",
                "target_function": func_breadcrumb,
                "issue": "Complex function with low connectivity - potential dead code",
                "suggestions": [
                    "Verify if this function is actually needed",
                    "Consider removing if it's unused complex code",
                    "If needed, simplify before integrating into application flow",
                ],
                "estimated_effort": "medium",
                "impact": "maintainability",
            }
        )

    return recommendations


def _generate_architectural_recommendations(analysis_results: dict[str, Any], functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Generate architectural-level recommendations."""
    recommendations = []

    # Analyze project-level patterns
    coverage_analysis = analysis_results.get("coverage_analysis", {})

    # High isolation rate
    coverage_percentage = coverage_analysis.get("coverage_percentage", 100)
    if coverage_percentage < 50:
        recommendations.append(
            {
                "type": "architectural_improvement",
                "priority": "high",
                "target_function": "project_wide",
                "issue": f"Low connectivity coverage ({coverage_percentage:.1f}%)",
                "suggestions": [
                    "Review overall project architecture",
                    "Identify and connect isolated components",
                    "Consider architectural patterns (DI, Event-driven, etc.)",
                    "Establish clear layer boundaries and dependencies",
                ],
                "estimated_effort": "high",
                "impact": "architecture",
            }
        )

    # Too many entry points
    entry_points = analysis_results.get("hotspot_analysis", {}).get("hotspot_categories", {}).get("entry_point", 0)
    if entry_points > 10:
        recommendations.append(
            {
                "type": "architectural_consolidation",
                "priority": "medium",
                "target_function": "project_wide",
                "issue": f"Too many entry points ({entry_points})",
                "suggestions": [
                    "Consolidate entry points through facade pattern",
                    "Implement unified API gateway or controller",
                    "Reduce system complexity by minimizing entry points",
                ],
                "estimated_effort": "medium",
                "impact": "architecture",
            }
        )

    return recommendations


def _identify_complex_hotspots(analysis_results: dict[str, Any]) -> list[str]:
    """Identify functions that are both complex and hotspots."""
    complex_functions = set()
    hotspot_functions = set()

    # Get complex function breadcrumbs
    if "complexity_analysis" in analysis_results:
        complex_funcs = analysis_results["complexity_analysis"].get("complex_functions", [])
        complex_functions = {func["breadcrumb"] for func in complex_funcs if func.get("complexity_score", 0) > 0.7}

    # Get hotspot function breadcrumbs
    if "hotspot_analysis" in analysis_results:
        hotspot_funcs = analysis_results["hotspot_analysis"].get("hotspot_functions", [])
        hotspot_functions = {func["breadcrumb"] for func in hotspot_funcs}

    return list(complex_functions.intersection(hotspot_functions))


def _identify_complex_isolated_functions(analysis_results: dict[str, Any]) -> list[str]:
    """Identify functions that are complex but isolated."""
    complex_functions = set()
    isolated_functions = set()

    # Get complex function breadcrumbs
    if "complexity_analysis" in analysis_results:
        complex_funcs = analysis_results["complexity_analysis"].get("complex_functions", [])
        complex_functions = {func["breadcrumb"] for func in complex_funcs if func.get("complexity_score", 0) > 0.6}

    # Get isolated function breadcrumbs
    if "coverage_analysis" in analysis_results:
        isolated_funcs = analysis_results["coverage_analysis"].get("uncovered_functions", [])
        isolated_functions = {func["breadcrumb"] for func in isolated_funcs}

    return list(complex_functions.intersection(isolated_functions))


def _deduplicate_and_merge_recommendations(recommendations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate recommendations and merge similar ones."""
    # Group by target function
    by_function = {}
    for rec in recommendations:
        target = rec["target_function"]
        if target not in by_function:
            by_function[target] = []
        by_function[target].append(rec)

    merged_recommendations = []
    for target_function, recs in by_function.items():
        if len(recs) == 1:
            merged_recommendations.extend(recs)
        else:
            # Merge multiple recommendations for the same function
            merged = _merge_function_recommendations(recs)
            merged_recommendations.append(merged)

    return merged_recommendations


def _merge_function_recommendations(recommendations: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge multiple recommendations for the same function."""
    # Use the highest priority
    priorities = ["critical", "high", "medium", "low"]
    highest_priority = "low"
    for rec in recommendations:
        rec_priority = rec.get("priority", "low")
        if priorities.index(rec_priority) < priorities.index(highest_priority):
            highest_priority = rec_priority

    # Combine all issues and suggestions
    all_issues = []
    all_suggestions = []
    all_types = []

    for rec in recommendations:
        all_issues.append(rec.get("issue", ""))
        if isinstance(rec.get("suggestions"), list):
            all_suggestions.extend(rec["suggestions"])
        else:
            all_suggestions.append(rec.get("suggestion", ""))
        all_types.append(rec.get("type", ""))

    return {
        "type": "combined_refactoring",
        "priority": highest_priority,
        "target_function": recommendations[0]["target_function"],
        "issue": "Multiple issues identified: " + "; ".join(filter(None, all_issues)),
        "suggestions": list(set(filter(None, all_suggestions))),  # Remove duplicates
        "combined_types": list(set(all_types)),
        "estimated_effort": "medium",  # Conservative estimate for combined work
        "impact": "comprehensive",
    }


def _calculate_refactoring_impact(recommendations: list[dict[str, Any]], analysis_results: dict[str, Any]) -> list[dict[str, Any]]:
    """Calculate the potential impact of each refactoring recommendation."""
    for rec in recommendations:
        rec["impact_analysis"] = {
            "maintainability_impact": _calculate_maintainability_impact(rec),
            "performance_impact": _calculate_performance_impact_potential(rec),
            "risk_level": _calculate_refactoring_risk(rec),
            "dependencies_affected": _estimate_dependencies_affected(rec, analysis_results),
        }

    return recommendations


def _calculate_maintainability_impact(recommendation: dict[str, Any]) -> str:
    """Calculate maintainability impact of a recommendation."""
    rec_type = recommendation.get("type", "")
    priority = recommendation.get("priority", "low")

    if rec_type in ["complexity_reduction", "critical_refactoring"]:
        return "high" if priority in ["critical", "high"] else "medium"
    elif rec_type in ["hotspot_optimization", "architectural_improvement"]:
        return "medium"
    else:
        return "low"


def _calculate_performance_impact_potential(recommendation: dict[str, Any]) -> str:
    """Calculate potential performance impact of a recommendation."""
    rec_type = recommendation.get("type", "")

    if rec_type in ["hotspot_optimization", "critical_refactoring"]:
        return "high"
    elif rec_type in ["complexity_reduction", "architectural_improvement"]:
        return "medium"
    else:
        return "low"


def _calculate_refactoring_risk(recommendation: dict[str, Any]) -> str:
    """Calculate the risk level of performing the refactoring."""
    rec_type = recommendation.get("type", "")
    priority = recommendation.get("priority", "low")

    if rec_type in ["critical_refactoring", "architectural_improvement"] and priority == "critical":
        return "high"
    elif rec_type in ["hotspot_optimization", "complexity_reduction"] and priority in ["critical", "high"]:
        return "medium"
    else:
        return "low"


def _estimate_dependencies_affected(recommendation: dict[str, Any], analysis_results: dict[str, Any]) -> int:
    """Estimate number of dependencies that might be affected by the refactoring."""
    target_function = recommendation.get("target_function", "")

    # Look up function in hotspot analysis for connectivity info
    if "hotspot_analysis" in analysis_results:
        hotspot_functions = analysis_results["hotspot_analysis"].get("hotspot_functions", [])
        for func in hotspot_functions:
            if func.get("breadcrumb") == target_function:
                connectivity = func.get("connectivity_metrics", {})
                return connectivity.get("connection_count", 0)

    # Look up in coverage analysis
    if "coverage_analysis" in analysis_results:
        covered_functions = analysis_results["coverage_analysis"].get("covered_functions", [])
        for func in covered_functions:
            if func.get("breadcrumb") == target_function:
                return func.get("total_connections", 0)

    return 0  # Default if not found


def _generate_refactoring_strategy(recommendations: list[dict[str, Any]], analysis_results: dict[str, Any]) -> dict[str, Any]:
    """Generate an overall refactoring strategy."""
    # Group by priority
    critical_count = len([r for r in recommendations if r.get("priority") == "critical"])
    high_count = len([r for r in recommendations if r.get("priority") == "high"])

    # Determine strategy approach
    if critical_count > 0:
        approach = "immediate_action_required"
        strategy = "Address critical issues immediately, then proceed with high-priority items"
    elif high_count > 5:
        approach = "systematic_refactoring"
        strategy = "Plan systematic refactoring campaign focusing on high-impact areas"
    else:
        approach = "incremental_improvement"
        strategy = "Implement gradual improvements during regular development cycles"

    # Calculate estimated timeline
    total_effort = sum({"low": 1, "medium": 3, "high": 8}.get(r.get("estimated_effort", "low"), 1) for r in recommendations)

    timeline = "1-2 weeks" if total_effort < 10 else "2-4 weeks" if total_effort < 30 else "1-2 months"

    return {
        "approach": approach,
        "strategy": strategy,
        "estimated_timeline": timeline,
        "priority_distribution": {
            "critical": critical_count,
            "high": high_count,
            "medium": len([r for r in recommendations if r.get("priority") == "medium"]),
            "low": len([r for r in recommendations if r.get("priority") == "low"]),
        },
        "recommended_order": _generate_recommended_refactoring_order(recommendations),
    }


def _generate_recommended_refactoring_order(recommendations: list[dict[str, Any]]) -> list[str]:
    """Generate recommended order for implementing refactoring recommendations."""
    # Sort by priority first, then by impact
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}

    sorted_recs = sorted(
        recommendations,
        key=lambda r: (
            priority_order.get(r.get("priority", "low"), 3),
            -_get_impact_score(r),  # Higher impact first
        ),
    )

    return [rec.get("target_function", "unknown") for rec in sorted_recs[:10]]  # Top 10


def _get_impact_score(recommendation: dict[str, Any]) -> int:
    """Get numerical impact score for sorting."""
    impact_analysis = recommendation.get("impact_analysis", {})
    score = 0

    # Add points for different impact types
    if impact_analysis.get("maintainability_impact") == "high":
        score += 3
    elif impact_analysis.get("maintainability_impact") == "medium":
        score += 2

    if impact_analysis.get("performance_impact") == "high":
        score += 3
    elif impact_analysis.get("performance_impact") == "medium":
        score += 2

    score += impact_analysis.get("dependencies_affected", 0) // 5  # More dependencies = higher impact

    return score


def _calculate_technical_debt_metrics(recommendations: list[dict[str, Any]], analysis_results: dict[str, Any]) -> dict[str, Any]:
    """Calculate technical debt metrics based on recommendations."""
    total_debt_score = 0
    debt_categories = {"complexity": 0, "performance": 0, "architecture": 0, "connectivity": 0}

    for rec in recommendations:
        priority_weight = {"critical": 10, "high": 7, "medium": 4, "low": 2}.get(rec.get("priority", "low"), 2)
        total_debt_score += priority_weight

        # Categorize debt
        rec_type = rec.get("type", "")
        if "complexity" in rec_type:
            debt_categories["complexity"] += priority_weight
        elif "hotspot" in rec_type or "performance" in rec_type:
            debt_categories["performance"] += priority_weight
        elif "architectural" in rec_type:
            debt_categories["architecture"] += priority_weight
        elif "connectivity" in rec_type:
            debt_categories["connectivity"] += priority_weight

    # Calculate debt level
    if total_debt_score > 50:
        debt_level = "high"
    elif total_debt_score > 20:
        debt_level = "medium"
    else:
        debt_level = "low"

    return {
        "total_debt_score": total_debt_score,
        "debt_level": debt_level,
        "debt_categories": debt_categories,
        "critical_issues": len([r for r in recommendations if r.get("priority") == "critical"]),
        "refactoring_urgency": "immediate" if debt_level == "high" else "planned" if debt_level == "medium" else "optional",
    }


def _get_most_common_issues(recommendations: list[dict[str, Any]]) -> list[dict[str, int]]:
    """Get the most common issues from recommendations."""
    issue_counts = {}
    for rec in recommendations:
        rec_type = rec.get("type", "unknown")
        issue_counts[rec_type] = issue_counts.get(rec_type, 0) + 1

    # Sort by count
    sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
    return [{"issue": issue, "count": count} for issue, count in sorted_issues[:5]]


def _estimate_total_refactoring_effort(recommendations: list[dict[str, Any]]) -> str:
    """Estimate total effort required for all refactoring recommendations."""
    total_effort = sum({"low": 1, "medium": 3, "high": 8}.get(r.get("estimated_effort", "low"), 1) for r in recommendations)

    if total_effort < 10:
        return "low (1-2 weeks)"
    elif total_effort < 30:
        return "medium (2-4 weeks)"
    else:
        return "high (1-2 months)"


def _estimate_complexity_refactoring_effort(complexity_score: float, raw_metrics: dict[str, Any]) -> str:
    """Estimate effort for complexity refactoring."""
    func_length = raw_metrics.get("function_length", 0)
    cyclomatic_complexity = raw_metrics.get("cyclomatic_complexity", 0)

    if complexity_score > 0.8 or func_length > 100 or cyclomatic_complexity > 15:
        return "high"
    elif complexity_score > 0.6 or func_length > 50 or cyclomatic_complexity > 10:
        return "medium"
    else:
        return "low"


def _estimate_hotspot_refactoring_effort(category: str, performance_impact: float) -> str:
    """Estimate effort for hotspot refactoring."""
    if category == "performance_bottleneck" and performance_impact > 0.8:
        return "high"
    elif category in ["architectural_hub", "critical_utility"] or performance_impact > 0.6:
        return "medium"
    else:
        return "low"


async def _calculate_project_metrics(
    functions: list[dict[str, Any]],
    analysis_results: dict[str, Any],
    implementation_chain_service: ImplementationChainService,
    chain_types: list[str] = None,
) -> dict[str, Any]:
    """
    Calculate comprehensive project-level metrics including average chain depth,
    total entry points, connectivity scores, and architectural quality indicators.

    This function provides project-wide insights by aggregating individual function
    analysis results into meaningful project-level statistics and quality indicators.

    Args:
        functions: List of functions to analyze
        analysis_results: Results from various analyses (complexity, hotspot, coverage)
        implementation_chain_service: Service for chain analysis

    Returns:
        Dictionary with comprehensive project metrics and quality indicators
    """
    logger.info("Calculating comprehensive project-level metrics")

    # Step 1: Calculate basic structural metrics
    basic_metrics = await _calculate_basic_project_metrics(functions, analysis_results, implementation_chain_service)

    # Step 2: Calculate average chain depth using real chain analysis
    chain_depth_metrics = await _calculate_real_average_chain_depth(functions, implementation_chain_service, chain_types)

    # Step 3: Calculate connectivity metrics and scores
    connectivity_metrics = _calculate_comprehensive_connectivity_metrics_for_project(analysis_results)

    # Step 4: Calculate complexity distribution and quality metrics
    complexity_metrics = _calculate_project_complexity_metrics(analysis_results)

    # Step 5: Calculate architectural quality indicators
    architectural_metrics = _calculate_architectural_quality_metrics(analysis_results, functions)

    # Step 6: Calculate performance and hotspot metrics
    performance_metrics = _calculate_project_performance_metrics(analysis_results)

    # Step 7: Calculate technical debt and maintainability metrics
    maintainability_metrics = _calculate_project_maintainability_metrics(analysis_results)

    # Step 8: Calculate project health score
    health_score = _calculate_project_health_score(basic_metrics, connectivity_metrics, complexity_metrics, architectural_metrics)

    return {
        "basic_metrics": basic_metrics,
        "chain_depth_metrics": chain_depth_metrics,
        "connectivity_metrics": connectivity_metrics,
        "complexity_metrics": complexity_metrics,
        "architectural_metrics": architectural_metrics,
        "performance_metrics": performance_metrics,
        "maintainability_metrics": maintainability_metrics,
        "project_health": health_score,
        "quality_indicators": _generate_quality_indicators(basic_metrics, connectivity_metrics, complexity_metrics, architectural_metrics),
        "recommendations": _generate_project_level_recommendations(
            basic_metrics, connectivity_metrics, complexity_metrics, architectural_metrics
        ),
    }


async def _calculate_basic_project_metrics(
    functions: list[dict[str, Any]],
    analysis_results: dict[str, Any],
    implementation_chain_service: ImplementationChainService,
) -> dict[str, Any]:
    """Calculate basic project structural metrics."""
    total_functions = len(functions)

    # Count entry points from coverage analysis
    total_entry_points = 0
    if "coverage_analysis" in analysis_results:
        connectivity_stats = analysis_results["coverage_analysis"].get("connectivity_statistics", {})
        total_entry_points = connectivity_stats.get("advanced_metrics", {}).get("entry_point_functions", 0)

    # If no coverage analysis, use heuristic detection
    if total_entry_points == 0:
        for func in functions:
            if _is_likely_entry_point_heuristic(func):
                total_entry_points += 1

    # Calculate total chain connections
    total_chain_connections = 0
    if "coverage_analysis" in analysis_results:
        network_analysis = analysis_results["coverage_analysis"].get("network_analysis", {})
        total_chain_connections = network_analysis.get("total_connections", 0)

    # Calculate function distribution by type
    function_distribution = _calculate_function_type_distribution(functions, analysis_results)

    # Calculate file distribution metrics
    file_distribution = _calculate_file_distribution_metrics(functions)

    return {
        "total_functions": total_functions,
        "total_entry_points": total_entry_points,
        "total_chain_connections": total_chain_connections,
        "function_distribution": function_distribution,
        "file_distribution": file_distribution,
        "files_analyzed": len({func.get("file_path", "") for func in functions}),
        "average_functions_per_file": total_functions / max(1, len({func.get("file_path", "") for func in functions})),
    }


async def _calculate_real_average_chain_depth(
    functions: list[dict[str, Any]],
    implementation_chain_service: ImplementationChainService,
    chain_types: list[str] = None,
) -> dict[str, Any]:
    """
    Calculate real average chain depth by analyzing actual function chains.

    Args:
        functions: List of functions to analyze
        implementation_chain_service: Service for chain analysis
        chain_types: List of chain types to analyze (default: ["execution_flow"])

    Returns:
        Dictionary with chain depth metrics and analysis coverage
    """
    chain_depths = []
    total_chains_analyzed = 0
    successful_traces = 0
    chain_depths_by_type = {}

    # Determine which chain types to analyze
    target_chain_types = chain_types if chain_types else ["execution_flow"]

    # Initialize chain depths tracking by type
    for chain_type in target_chain_types:
        chain_depths_by_type[chain_type] = []

    # Sample entry points for chain analysis (limit for performance)
    entry_point_functions = [func for func in functions if _is_likely_entry_point_heuristic(func)]
    sample_entry_points = entry_point_functions[: min(10, len(entry_point_functions))]

    logger.debug(f"Analyzing chain depth for {len(sample_entry_points)} entry points across {len(target_chain_types)} chain types")

    for entry_point in sample_entry_points:
        for chain_type in target_chain_types:
            try:
                # Trace chain from entry point for specific chain type
                chain_result = await implementation_chain_service.trace_function_chain(
                    entry_point["breadcrumb"], max_depth=15, chain_type=chain_type, include_cycles=False
                )

                if chain_result and "chain_links" in chain_result:
                    chain_depth = len(chain_result["chain_links"])
                    chain_depths.append(chain_depth)
                    chain_depths_by_type[chain_type].append(chain_depth)
                    successful_traces += 1

                total_chains_analyzed += 1

            except Exception as e:
                logger.debug(f"Could not trace {chain_type} chain for {entry_point['breadcrumb']}: {e}")
                continue

    # Calculate statistics
    if chain_depths:
        average_chain_depth = sum(chain_depths) / len(chain_depths)
        max_chain_depth = max(chain_depths)
        min_chain_depth = min(chain_depths)
        median_chain_depth = sorted(chain_depths)[len(chain_depths) // 2] if chain_depths else 0
    else:
        # Fallback to heuristic calculation
        average_chain_depth = _estimate_average_chain_depth_heuristic(functions)
        max_chain_depth = int(average_chain_depth * 2)
        min_chain_depth = max(1, int(average_chain_depth * 0.5))
        median_chain_depth = average_chain_depth

    # Calculate chain type specific metrics
    chain_type_metrics = {}
    for chain_type, depths in chain_depths_by_type.items():
        if depths:
            chain_type_metrics[chain_type] = {
                "average_depth": round(sum(depths) / len(depths), 2),
                "max_depth": max(depths),
                "min_depth": min(depths),
                "count": len(depths),
                "distribution": _calculate_chain_depth_distribution(depths),
            }
        else:
            chain_type_metrics[chain_type] = {
                "average_depth": 0.0,
                "max_depth": 0,
                "min_depth": 0,
                "count": 0,
                "distribution": {"shallow": 0, "medium": 0, "deep": 0, "very_deep": 0},
            }

    return {
        "average_chain_depth": round(average_chain_depth, 2),
        "max_chain_depth": max_chain_depth,
        "min_chain_depth": min_chain_depth,
        "median_chain_depth": median_chain_depth,
        "total_chains_analyzed": total_chains_analyzed,
        "successful_traces": successful_traces,
        "chain_depth_distribution": _calculate_chain_depth_distribution(chain_depths),
        "analysis_coverage": (successful_traces / max(1, total_chains_analyzed)) * 100,
        "chain_types_analyzed": target_chain_types,
        "chain_type_metrics": chain_type_metrics,
    }


def _calculate_comprehensive_connectivity_metrics_for_project(analysis_results: dict[str, Any]) -> dict[str, Any]:
    """Calculate comprehensive connectivity metrics for the entire project."""
    if "coverage_analysis" not in analysis_results:
        return _default_connectivity_metrics()

    coverage_data = analysis_results["coverage_analysis"]
    connectivity_stats = coverage_data.get("connectivity_statistics", {})
    network_analysis = coverage_data.get("network_analysis", {})

    # Basic connectivity metrics
    coverage_percentage = coverage_data.get("coverage_percentage", 0.0)
    connectivity_score = coverage_percentage / 100.0

    # Advanced connectivity metrics
    connection_stats = connectivity_stats.get("connection_statistics", {})
    total_connections = connection_stats.get("total_connections", {})

    avg_incoming = total_connections.get("avg", 0.0)
    avg_outgoing = total_connections.get("avg", 0.0)

    # Network quality metrics
    network_density = network_analysis.get("network_density", 0.0)
    cluster_analysis = network_analysis.get("cluster_analysis", {})
    modularity_score = cluster_analysis.get("modularity_score", 0.0)

    # Hub and isolation metrics
    advanced_metrics = connectivity_stats.get("advanced_metrics", {})
    highly_connected_functions = advanced_metrics.get("highly_connected_functions", 0)
    isolated_functions = advanced_metrics.get("isolated_functions", 0)
    hub_functions = advanced_metrics.get("hub_functions", 0)
    bridge_functions = advanced_metrics.get("bridge_functions", 0)

    return {
        "connectivity_score": connectivity_score,
        "coverage_percentage": coverage_percentage,
        "network_density": network_density,
        "modularity_score": modularity_score,
        "average_incoming_connections": avg_incoming,
        "average_outgoing_connections": avg_outgoing,
        "highly_connected_functions": highly_connected_functions,
        "isolated_functions": isolated_functions,
        "hub_functions": hub_functions,
        "bridge_functions": bridge_functions,
        "connectivity_quality": _calculate_connectivity_quality_score(
            connectivity_score, network_density, modularity_score, isolated_functions
        ),
    }


def _calculate_project_complexity_metrics(analysis_results: dict[str, Any]) -> dict[str, Any]:
    """Calculate project-wide complexity metrics and distribution."""
    if "complexity_analysis" not in analysis_results:
        return _default_complexity_metrics()

    complexity_data = analysis_results["complexity_analysis"]

    # Basic complexity metrics
    average_complexity = complexity_data.get("average_complexity", 0.0)
    complexity_distribution = complexity_data.get("complexity_distribution", {})
    total_complex_functions = complexity_data.get("complex_functions_count", 0)
    total_functions = complexity_data.get("total_functions_analyzed", 1)

    # Calculate complexity quality indicators
    complexity_ratio = total_complex_functions / max(1, total_functions)
    complexity_quality = _calculate_complexity_quality_score(average_complexity, complexity_ratio, complexity_distribution)

    # Get complexity statistics
    complexity_stats = complexity_data.get("complexity_statistics", {})

    return {
        "overall_complexity_score": average_complexity,
        "complexity_distribution": complexity_distribution,
        "complex_functions_count": total_complex_functions,
        "complexity_ratio": complexity_ratio,
        "complexity_quality": complexity_quality,
        "complexity_statistics": complexity_stats,
        "complexity_trends": _analyze_complexity_trends(complexity_data),
    }


def _calculate_architectural_quality_metrics(analysis_results: dict[str, Any], functions: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate architectural quality metrics."""
    # Analyze layer distribution
    layer_distribution = _analyze_architectural_layers(functions)

    # Analyze hotspot patterns
    hotspot_metrics = {}
    if "hotspot_analysis" in analysis_results:
        hotspot_data = analysis_results["hotspot_analysis"]
        hotspot_categories = hotspot_data.get("hotspot_categories", {})
        hotspot_metrics = {
            "performance_bottlenecks": hotspot_categories.get("performance_bottleneck", 0),
            "architectural_hubs": hotspot_categories.get("architectural_hub", 0),
            "entry_points": hotspot_categories.get("entry_point", 0),
            "critical_utilities": hotspot_categories.get("critical_utility", 0),
        }

    # Calculate architectural health score
    architectural_health = _calculate_architectural_health_score(layer_distribution, hotspot_metrics, analysis_results)

    return {
        "layer_distribution": layer_distribution,
        "hotspot_metrics": hotspot_metrics,
        "architectural_health": architectural_health,
        "design_pattern_indicators": _identify_design_pattern_indicators(analysis_results, functions),
        "coupling_indicators": _calculate_coupling_indicators(analysis_results),
        "cohesion_indicators": _calculate_cohesion_indicators(analysis_results),
    }


def _calculate_project_performance_metrics(analysis_results: dict[str, Any]) -> dict[str, Any]:
    """Calculate project performance and hotspot metrics."""
    if "hotspot_analysis" not in analysis_results:
        return _default_performance_metrics()

    hotspot_data = analysis_results["hotspot_analysis"]

    # Performance hotspot metrics
    hotspot_functions_count = hotspot_data.get("hotspot_functions_count", 0)
    critical_paths_count = len(hotspot_data.get("critical_paths", []))

    # Get hotspot statistics
    hotspot_stats = hotspot_data.get("hotspot_statistics", {})
    performance_impact_stats = hotspot_stats.get("performance_impact_stats", {})

    # Calculate performance risk score
    performance_risk = _calculate_performance_risk_score(hotspot_functions_count, critical_paths_count, performance_impact_stats)

    return {
        "hotspot_functions_count": hotspot_functions_count,
        "critical_paths_count": critical_paths_count,
        "performance_impact_statistics": performance_impact_stats,
        "performance_risk_score": performance_risk,
        "hotspot_distribution": hotspot_data.get("hotspot_categories", {}),
        "performance_recommendations": _generate_performance_recommendations(hotspot_data),
    }


def _calculate_project_maintainability_metrics(analysis_results: dict[str, Any]) -> dict[str, Any]:
    """Calculate project maintainability and technical debt metrics."""
    maintainability_score = 0.0
    technical_debt_indicators = {}

    # Factor in complexity metrics
    if "complexity_analysis" in analysis_results:
        complexity_data = analysis_results["complexity_analysis"]
        avg_complexity = complexity_data.get("average_complexity", 0.0)
        maintainability_score += (1.0 - min(1.0, avg_complexity)) * 0.4  # 40% weight

    # Factor in connectivity metrics
    if "coverage_analysis" in analysis_results:
        coverage_data = analysis_results["coverage_analysis"]
        coverage_percentage = coverage_data.get("coverage_percentage", 0.0)
        maintainability_score += (coverage_percentage / 100.0) * 0.3  # 30% weight

    # Factor in hotspot metrics
    if "hotspot_analysis" in analysis_results:
        hotspot_data = analysis_results["hotspot_analysis"]
        hotspot_count = hotspot_data.get("hotspot_functions_count", 0)
        total_functions = analysis_results.get("complexity_analysis", {}).get("total_functions_analyzed", 1)
        hotspot_ratio = 1.0 - min(1.0, hotspot_count / max(1, total_functions))
        maintainability_score += hotspot_ratio * 0.2  # 20% weight

    # Factor in refactoring recommendations
    if "refactoring_recommendations" in analysis_results:
        refactoring_data = analysis_results["refactoring_recommendations"]
        technical_debt_metrics = refactoring_data.get("technical_debt_metrics", {})
        debt_level = technical_debt_metrics.get("debt_level", "low")
        debt_impact = {"low": 0.1, "medium": 0.05, "high": 0.0}.get(debt_level, 0.1)
        maintainability_score += debt_impact * 0.1  # 10% weight

    return {
        "maintainability_score": round(maintainability_score, 3),
        "technical_debt_indicators": technical_debt_indicators,
        "refactoring_urgency": _determine_refactoring_urgency(analysis_results),
        "code_quality_trends": _analyze_code_quality_trends(analysis_results),
    }


def _calculate_project_health_score(
    basic_metrics: dict[str, Any],
    connectivity_metrics: dict[str, Any],
    complexity_metrics: dict[str, Any],
    architectural_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Calculate overall project health score."""
    # Calculate component scores (0.0 to 1.0)
    connectivity_score = connectivity_metrics.get("connectivity_quality", 0.0)
    complexity_score = complexity_metrics.get("complexity_quality", 0.0)
    architectural_score = architectural_metrics.get("architectural_health", 0.0)

    # Calculate weighted overall health score
    health_weights = {"connectivity": 0.35, "complexity": 0.35, "architecture": 0.30}

    overall_health = (
        connectivity_score * health_weights["connectivity"]
        + complexity_score * health_weights["complexity"]
        + architectural_score * health_weights["architecture"]
    )

    # Determine health category
    if overall_health >= 0.8:
        health_category = "excellent"
    elif overall_health >= 0.6:
        health_category = "good"
    elif overall_health >= 0.4:
        health_category = "fair"
    else:
        health_category = "poor"

    return {
        "overall_health_score": round(overall_health, 3),
        "health_category": health_category,
        "component_scores": {
            "connectivity": round(connectivity_score, 3),
            "complexity": round(complexity_score, 3),
            "architecture": round(architectural_score, 3),
        },
        "health_trends": _analyze_health_trends(overall_health, connectivity_score, complexity_score, architectural_score),
    }


# Helper functions for project metrics calculations


def _is_likely_entry_point_heuristic(func: dict[str, Any]) -> bool:
    """Heuristic to determine if a function is likely an entry point."""
    name = func.get("name", "").lower()
    breadcrumb = func.get("breadcrumb", "").lower()

    # Check for common entry point patterns
    entry_patterns = ["main", "init", "start", "run", "execute", "handler", "endpoint", "app", "server"]
    return any(pattern in name or pattern in breadcrumb for pattern in entry_patterns)


def _calculate_function_type_distribution(functions: list[dict[str, Any]], analysis_results: dict[str, Any]) -> dict[str, int]:
    """Calculate distribution of function types."""
    distribution = {"entry_point": 0, "getter": 0, "setter": 0, "processor": 0, "utility": 0, "test": 0, "standard": 0}

    # Use coverage analysis function types if available
    if "coverage_analysis" in analysis_results:
        coverage_stats = analysis_results["coverage_analysis"].get("connectivity_statistics", {})
        function_type_dist = coverage_stats.get("function_type_distribution", {})
        distribution.update(function_type_dist)
    else:
        # Fallback to heuristic classification
        for func in functions:
            func_type = _classify_function_type(func["name"])
            distribution[func_type] = distribution.get(func_type, 0) + 1

    return distribution


def _calculate_file_distribution_metrics(functions: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate file distribution metrics."""
    file_function_counts = {}
    for func in functions:
        file_path = func.get("file_path", "unknown")
        file_function_counts[file_path] = file_function_counts.get(file_path, 0) + 1

    counts = list(file_function_counts.values())
    if counts:
        return {
            "total_files": len(file_function_counts),
            "avg_functions_per_file": sum(counts) / len(counts),
            "max_functions_per_file": max(counts),
            "min_functions_per_file": min(counts),
            "files_with_many_functions": len([c for c in counts if c > 10]),  # Potential refactoring candidates
        }
    else:
        return {"total_files": 0, "avg_functions_per_file": 0, "max_functions_per_file": 0, "min_functions_per_file": 0}


def _estimate_average_chain_depth_heuristic(functions: list[dict[str, Any]]) -> float:
    """Estimate average chain depth using heuristics."""
    # Simple heuristic based on function content and naming patterns
    total_estimated_depth = 0.0
    processed_functions = 0

    for func in functions:
        content = func.get("content", "")
        name = func.get("name", "")

        # Estimate depth based on function characteristics
        estimated_depth = 1.0  # Base depth

        # Functions with many lines tend to have deeper call chains
        if len(content) > 100:
            estimated_depth += 1.0
        if len(content) > 500:
            estimated_depth += 1.0

        # Processor functions tend to have deeper chains
        if any(pattern in name.lower() for pattern in ["process", "calculate", "analyze", "transform"]):
            estimated_depth += 1.5

        # Entry points tend to have deeper chains
        if _is_likely_entry_point_heuristic(func):
            estimated_depth += 2.0

        total_estimated_depth += estimated_depth
        processed_functions += 1

    return total_estimated_depth / max(1, processed_functions)


def _calculate_chain_depth_distribution(chain_depths: list[int]) -> dict[str, int]:
    """Calculate distribution of chain depths."""
    if not chain_depths:
        return {"shallow": 0, "medium": 0, "deep": 0, "very_deep": 0}

    distribution = {"shallow": 0, "medium": 0, "deep": 0, "very_deep": 0}

    for depth in chain_depths:
        if depth <= 3:
            distribution["shallow"] += 1
        elif depth <= 6:
            distribution["medium"] += 1
        elif depth <= 10:
            distribution["deep"] += 1
        else:
            distribution["very_deep"] += 1

    return distribution


def _calculate_connectivity_quality_score(
    connectivity_score: float, network_density: float, modularity_score: float, isolated_functions: int
) -> float:
    """Calculate overall connectivity quality score."""
    # Base score from connectivity percentage (0.0 to 0.5)
    base_score = min(0.5, connectivity_score * 0.5)

    # Network density bonus (0.0 to 0.2)
    density_bonus = min(0.2, network_density * 0.2)

    # Modularity bonus (0.0 to 0.2)
    modularity_bonus = min(0.2, max(0, modularity_score) * 0.2)

    # Isolation penalty (0.0 to -0.1)
    isolation_penalty = min(0.1, isolated_functions * 0.01)

    total_score = base_score + density_bonus + modularity_bonus - isolation_penalty
    return max(0.0, min(1.0, total_score))


def _calculate_complexity_quality_score(average_complexity: float, complexity_ratio: float, distribution: dict) -> float:
    """Calculate complexity quality score."""
    # Lower complexity is better
    complexity_score = max(0.0, 1.0 - average_complexity)

    # Lower ratio of complex functions is better
    ratio_score = max(0.0, 1.0 - complexity_ratio)

    # Distribution balance (prefer more low/medium complexity functions)
    low_count = distribution.get("low", 0)
    medium_count = distribution.get("medium", 0)
    high_count = distribution.get("high", 0)
    total = max(1, low_count + medium_count + high_count)

    distribution_score = (low_count * 1.0 + medium_count * 0.5 + high_count * 0.1) / total

    # Weighted combination
    return (complexity_score * 0.4) + (ratio_score * 0.4) + (distribution_score * 0.2)


def _analyze_architectural_layers(functions: list[dict[str, Any]]) -> dict[str, int]:
    """Analyze distribution across architectural layers."""
    layer_counts = {"api": 0, "service": 0, "core": 0, "utils": 0, "unknown": 0}

    for func in functions:
        breadcrumb = func.get("breadcrumb", "")
        layer = _determine_architectural_layer(breadcrumb)
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    return layer_counts


def _calculate_architectural_health_score(
    layer_distribution: dict[str, int], hotspot_metrics: dict[str, int], analysis_results: dict[str, Any]
) -> float:
    """Calculate architectural health score."""
    total_functions = sum(layer_distribution.values())
    if total_functions == 0:
        return 0.0

    # Layer balance score (better when functions are distributed across layers)
    unknown_ratio = layer_distribution.get("unknown", 0) / total_functions
    layer_balance = 1.0 - unknown_ratio

    # Hotspot distribution score (better when hotspots are manageable)
    total_hotspots = sum(hotspot_metrics.values())
    hotspot_ratio = total_hotspots / max(1, total_functions)
    hotspot_score = max(0.0, 1.0 - hotspot_ratio * 2)  # Penalty for high hotspot ratio

    # Entry point management (better when entry points are reasonable)
    entry_points = hotspot_metrics.get("entry_points", 0)
    entry_point_score = 1.0 if entry_points <= 5 else max(0.0, 1.0 - (entry_points - 5) * 0.1)

    # Weighted combination
    return (layer_balance * 0.4) + (hotspot_score * 0.4) + (entry_point_score * 0.2)


# Default metrics for when analysis data is not available


def _default_connectivity_metrics() -> dict[str, Any]:
    """Return default connectivity metrics when analysis is not available."""
    return {
        "connectivity_score": 0.0,
        "coverage_percentage": 0.0,
        "network_density": 0.0,
        "modularity_score": 0.0,
        "average_incoming_connections": 0.0,
        "average_outgoing_connections": 0.0,
        "highly_connected_functions": 0,
        "isolated_functions": 0,
        "hub_functions": 0,
        "bridge_functions": 0,
        "connectivity_quality": 0.0,
    }


def _default_complexity_metrics() -> dict[str, Any]:
    """Return default complexity metrics when analysis is not available."""
    return {
        "overall_complexity_score": 0.0,
        "complexity_distribution": {"low": 0, "medium": 0, "high": 0},
        "complex_functions_count": 0,
        "complexity_ratio": 0.0,
        "complexity_quality": 0.0,
        "complexity_statistics": {},
        "complexity_trends": {"status": "unknown"},
    }


def _default_performance_metrics() -> dict[str, Any]:
    """Return default performance metrics when analysis is not available."""
    return {
        "hotspot_functions_count": 0,
        "critical_paths_count": 0,
        "performance_impact_statistics": {},
        "performance_risk_score": 0.0,
        "hotspot_distribution": {},
        "performance_recommendations": [],
    }


# Additional analysis functions


def _analyze_complexity_trends(complexity_data: dict[str, Any]) -> dict[str, str]:
    """Analyze complexity trends from complexity data."""
    avg_complexity = complexity_data.get("average_complexity", 0.0)

    if avg_complexity < 0.3:
        return {"status": "excellent", "description": "Low complexity across the project"}
    elif avg_complexity < 0.5:
        return {"status": "good", "description": "Moderate complexity levels"}
    elif avg_complexity < 0.7:
        return {"status": "concerning", "description": "High complexity requiring attention"}
    else:
        return {"status": "critical", "description": "Very high complexity requiring immediate action"}


def _identify_design_pattern_indicators(analysis_results: dict[str, Any], functions: list[dict[str, Any]]) -> dict[str, int]:
    """Identify potential design pattern usage indicators."""
    patterns = {"factory": 0, "singleton": 0, "observer": 0, "strategy": 0, "adapter": 0, "facade": 0}

    for func in functions:
        name = func.get("name", "").lower()

        # Simple pattern detection based on naming and content
        if "factory" in name or "create" in name:
            patterns["factory"] += 1
        if "singleton" in name or "instance" in name:
            patterns["singleton"] += 1
        if "observer" in name or "notify" in name or "update" in name:
            patterns["observer"] += 1
        if "strategy" in name or "algorithm" in name:
            patterns["strategy"] += 1
        if "adapter" in name or "wrapper" in name:
            patterns["adapter"] += 1
        if "facade" in name or "interface" in name:
            patterns["facade"] += 1

    return patterns


def _calculate_coupling_indicators(analysis_results: dict[str, Any]) -> dict[str, float]:
    """Calculate coupling indicators from analysis results."""
    if "coverage_analysis" not in analysis_results:
        return {"afferent_coupling": 0.0, "efferent_coupling": 0.0, "instability": 0.0}

    connectivity_stats = analysis_results["coverage_analysis"].get("connectivity_statistics", {})
    connection_stats = connectivity_stats.get("connection_statistics", {})

    avg_incoming = connection_stats.get("incoming_connections", {}).get("avg", 0.0)
    avg_outgoing = connection_stats.get("outgoing_connections", {}).get("avg", 0.0)

    # Calculate instability (I = Ce / (Ca + Ce))
    instability = avg_outgoing / max(1, avg_incoming + avg_outgoing)

    return {"afferent_coupling": avg_incoming, "efferent_coupling": avg_outgoing, "instability": instability}


def _calculate_cohesion_indicators(analysis_results: dict[str, Any]) -> dict[str, float]:
    """Calculate cohesion indicators from analysis results."""
    # Simplified cohesion calculation based on function clustering
    if "coverage_analysis" not in analysis_results:
        return {"module_cohesion": 0.0, "functional_cohesion": 0.0}

    network_analysis = analysis_results["coverage_analysis"].get("network_analysis", {})
    cluster_analysis = network_analysis.get("cluster_analysis", {})
    modularity_score = cluster_analysis.get("modularity_score", 0.0)

    # Use modularity as a proxy for cohesion
    module_cohesion = max(0.0, modularity_score)
    functional_cohesion = module_cohesion * 0.8  # Conservative estimate

    return {"module_cohesion": module_cohesion, "functional_cohesion": functional_cohesion}


def _calculate_performance_risk_score(hotspot_count: int, critical_paths: int, performance_stats: dict[str, Any]) -> float:
    """Calculate performance risk score."""
    if not performance_stats:
        return min(1.0, (hotspot_count + critical_paths) * 0.1)

    avg_impact = performance_stats.get("avg", 0.0)
    max_impact = performance_stats.get("max", 0.0)

    # Risk based on hotspot count, impact, and critical paths
    hotspot_risk = min(0.4, hotspot_count * 0.05)
    impact_risk = min(0.4, avg_impact * 0.4)
    max_impact_risk = min(0.2, max_impact * 0.2)

    return hotspot_risk + impact_risk + max_impact_risk


def _generate_performance_recommendations(hotspot_data: dict[str, Any]) -> list[str]:
    """Generate performance recommendations based on hotspot data."""
    recommendations = []

    hotspot_count = hotspot_data.get("hotspot_functions_count", 0)
    critical_paths = len(hotspot_data.get("critical_paths", []))

    if hotspot_count > 10:
        recommendations.append("High number of hotspot functions detected - prioritize performance optimization")
    if critical_paths > 5:
        recommendations.append("Multiple critical paths identified - consider architectural review")

    hotspot_categories = hotspot_data.get("hotspot_categories", {})
    if hotspot_categories.get("performance_bottleneck", 0) > 0:
        recommendations.append("Performance bottlenecks identified - profile and optimize critical functions")
    if hotspot_categories.get("architectural_hub", 0) > 3:
        recommendations.append("Many architectural hubs - consider breaking down complex components")

    return recommendations if recommendations else ["Performance appears well-distributed"]


def _determine_refactoring_urgency(analysis_results: dict[str, Any]) -> str:
    """Determine refactoring urgency based on analysis results."""
    if "refactoring_recommendations" in analysis_results:
        refactoring_data = analysis_results["refactoring_recommendations"]
        debt_metrics = refactoring_data.get("technical_debt_metrics", {})
        debt_level = debt_metrics.get("debt_level", "low")

        urgency_map = {"low": "optional", "medium": "planned", "high": "immediate"}
        return urgency_map.get(debt_level, "optional")

    return "unknown"


def _analyze_code_quality_trends(analysis_results: dict[str, Any]) -> dict[str, str]:
    """Analyze overall code quality trends."""
    # This is a simplified implementation - in reality would compare historical data
    quality_indicators = []

    if "complexity_analysis" in analysis_results:
        avg_complexity = analysis_results["complexity_analysis"].get("average_complexity", 0.0)
        quality_indicators.append(("complexity", avg_complexity))

    if "coverage_analysis" in analysis_results:
        coverage_percentage = analysis_results["coverage_analysis"].get("coverage_percentage", 0.0)
        quality_indicators.append(("connectivity", coverage_percentage / 100.0))

    if quality_indicators:
        avg_quality = sum(score for _, score in quality_indicators) / len(quality_indicators)
        if avg_quality > 0.8:
            return {"trend": "improving", "description": "High code quality indicators"}
        elif avg_quality > 0.6:
            return {"trend": "stable", "description": "Moderate code quality"}
        else:
            return {"trend": "declining", "description": "Code quality needs attention"}

    return {"trend": "unknown", "description": "Insufficient data for trend analysis"}


def _analyze_health_trends(overall: float, connectivity: float, complexity: float, architecture: float) -> dict[str, str]:
    """Analyze health trends across different metrics."""
    scores = [overall, connectivity, complexity, architecture]
    avg_score = sum(scores) / len(scores)

    if avg_score > 0.8:
        return {"trend": "excellent", "description": "Project health is excellent across all metrics"}
    elif avg_score > 0.6:
        return {"trend": "good", "description": "Project health is good with some areas for improvement"}
    elif avg_score > 0.4:
        return {"trend": "concerning", "description": "Project health shows concerning patterns"}
    else:
        return {"trend": "critical", "description": "Project health requires immediate attention"}


def _generate_quality_indicators(
    basic_metrics: dict[str, Any],
    connectivity_metrics: dict[str, Any],
    complexity_metrics: dict[str, Any],
    architectural_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    """Generate quality indicators for the project."""
    indicators = []

    # Connectivity quality indicator
    connectivity_score = connectivity_metrics.get("connectivity_quality", 0.0)
    indicators.append(
        {
            "metric": "connectivity_quality",
            "score": connectivity_score,
            "status": "excellent" if connectivity_score > 0.8 else "good" if connectivity_score > 0.6 else "needs_improvement",
            "description": f"Function connectivity quality: {connectivity_score:.2f}",
        }
    )

    # Complexity quality indicator
    complexity_score = complexity_metrics.get("complexity_quality", 0.0)
    indicators.append(
        {
            "metric": "complexity_quality",
            "score": complexity_score,
            "status": "excellent" if complexity_score > 0.8 else "good" if complexity_score > 0.6 else "needs_improvement",
            "description": f"Code complexity quality: {complexity_score:.2f}",
        }
    )

    # Architectural quality indicator
    architectural_score = architectural_metrics.get("architectural_health", 0.0)
    indicators.append(
        {
            "metric": "architectural_quality",
            "score": architectural_score,
            "status": "excellent" if architectural_score > 0.8 else "good" if architectural_score > 0.6 else "needs_improvement",
            "description": f"Architectural quality: {architectural_score:.2f}",
        }
    )

    return indicators


def _generate_project_level_recommendations(
    basic_metrics: dict[str, Any],
    connectivity_metrics: dict[str, Any],
    complexity_metrics: dict[str, Any],
    architectural_metrics: dict[str, Any],
) -> list[str]:
    """Generate project-level recommendations based on metrics."""
    recommendations = []

    # Connectivity recommendations
    connectivity_score = connectivity_metrics.get("connectivity_quality", 0.0)
    if connectivity_score < 0.5:
        recommendations.append("Improve function connectivity and reduce isolated functions")

    # Complexity recommendations
    complexity_score = complexity_metrics.get("complexity_quality", 0.0)
    if complexity_score < 0.5:
        recommendations.append("Focus on reducing code complexity through refactoring")

    # Architectural recommendations
    architectural_score = architectural_metrics.get("architectural_health", 0.0)
    if architectural_score < 0.5:
        recommendations.append("Review and improve overall project architecture")

    # Function distribution recommendations
    function_dist = basic_metrics.get("function_distribution", {})
    if function_dist.get("unknown", 0) > function_dist.get("standard", 0):
        recommendations.append("Classify and organize functions into appropriate architectural layers")

    return recommendations if recommendations else ["Project shows good overall quality metrics"]


async def _generate_analysis_report(
    analysis_results: dict[str, Any],
    functions: list[dict[str, Any]],
    weights: ComplexityWeights,
    output_format: str,
) -> dict[str, Any]:
    """
    Generate comprehensive analysis report.

    Args:
        analysis_results: Results from various analyses
        functions: List of functions
        weights: Complexity calculation weights
        output_format: Output format preference

    Returns:
        Dictionary with formatted analysis report
    """
    logger.info("Generating analysis report")

    report = {
        "analysis_summary": {
            "total_functions_analyzed": len(functions),
            "analysis_types_performed": list(analysis_results.keys()),
            "complexity_weights_used": {
                "branching_factor": weights.branching_factor,
                "cyclomatic_complexity": weights.cyclomatic_complexity,
                "call_depth": weights.call_depth,
                "function_length": weights.function_length,
            },
        },
        "detailed_results": analysis_results,
        "recommendations_summary": {
            "total_recommendations": 0,
            "high_priority_recommendations": 0,
            "medium_priority_recommendations": 0,
            "low_priority_recommendations": 0,
        },
    }

    # Add recommendation summary if available
    if "refactoring_recommendations" in analysis_results:
        refactoring_data = analysis_results["refactoring_recommendations"]
        report["recommendations_summary"] = {
            "total_recommendations": refactoring_data.get("total_recommendations", 0),
            "high_priority_recommendations": refactoring_data.get("summary", {}).get("high_priority_count", 0),
            "medium_priority_recommendations": refactoring_data.get("summary", {}).get("medium_priority_count", 0),
            "low_priority_recommendations": refactoring_data.get("summary", {}).get("low_priority_count", 0),
        }

    # Add project metrics summary if available
    if "project_metrics" in analysis_results:
        report["project_overview"] = analysis_results["project_metrics"]

    # Generate insights and suggestions
    insights = _generate_analysis_insights(analysis_results, functions)
    report["insights"] = insights

    # Generate final recommendations
    final_recommendations = _generate_final_recommendations(analysis_results, functions)
    report["final_recommendations"] = final_recommendations

    return report


def _generate_analysis_insights(analysis_results: dict[str, Any], functions: list[dict[str, Any]]) -> list[str]:
    """Generate insights from analysis results."""
    insights = []

    # Insights from complexity analysis
    if "complexity_analysis" in analysis_results:
        complexity_data = analysis_results["complexity_analysis"]
        complex_count = complexity_data.get("complex_functions_count", 0)
        total_count = complexity_data.get("total_functions_analyzed", 0)

        if total_count > 0:
            complexity_ratio = complex_count / total_count
            if complexity_ratio > 0.3:
                insights.append(
                    f"High complexity detected: {complex_count} out of {total_count} functions ({complexity_ratio:.1%}) are complex"
                )
            elif complexity_ratio > 0.1:
                insights.append(f"Moderate complexity: {complex_count} out of {total_count} functions ({complexity_ratio:.1%}) are complex")
            else:
                insights.append(f"Low complexity: Only {complex_count} out of {total_count} functions ({complexity_ratio:.1%}) are complex")

    # Insights from hotspot analysis
    if "hotspot_analysis" in analysis_results:
        hotspot_data = analysis_results["hotspot_analysis"]
        hotspot_count = hotspot_data.get("hotspot_functions_count", 0)

        if hotspot_count > 0:
            insights.append(f"Performance attention needed: {hotspot_count} hotspot functions identified")
        else:
            insights.append("Good performance distribution: No significant hotspots detected")

    # Insights from coverage analysis
    if "coverage_analysis" in analysis_results:
        coverage_data = analysis_results["coverage_analysis"]
        coverage_percentage = coverage_data.get("coverage_percentage", 0.0)

        if coverage_percentage > 80:
            insights.append(f"Excellent connectivity: {coverage_percentage:.1f}% of functions are well-connected")
        elif coverage_percentage > 60:
            insights.append(f"Good connectivity: {coverage_percentage:.1f}% of functions are connected")
        else:
            insights.append(f"Poor connectivity: Only {coverage_percentage:.1f}% of functions are connected")

    return insights


def _generate_final_recommendations(analysis_results: dict[str, Any], functions: list[dict[str, Any]]) -> list[str]:
    """Generate final recommendations based on all analysis results."""
    recommendations = []

    # Recommendations based on complexity
    if "complexity_analysis" in analysis_results:
        complex_functions = analysis_results["complexity_analysis"].get("complex_functions", [])
        if len(complex_functions) > 5:
            recommendations.append("Prioritize refactoring complex functions to improve maintainability")

    # Recommendations based on hotspots
    if "hotspot_analysis" in analysis_results:
        hotspot_functions = analysis_results["hotspot_analysis"].get("hotspot_functions", [])
        if len(hotspot_functions) > 3:
            recommendations.append("Consider optimizing hotspot functions for better performance")

    # Recommendations based on coverage
    if "coverage_analysis" in analysis_results:
        coverage_percentage = analysis_results["coverage_analysis"].get("coverage_percentage", 0.0)
        if coverage_percentage < 60:
            recommendations.append("Improve function connectivity to enhance code integration")

    # Recommendations based on refactoring analysis
    if "refactoring_recommendations" in analysis_results:
        high_priority_count = analysis_results["refactoring_recommendations"].get("summary", {}).get("high_priority_count", 0)
        if high_priority_count > 0:
            recommendations.append(f"Address {high_priority_count} high-priority refactoring recommendations")

    # General recommendations
    if not recommendations:
        recommendations.append("Project shows good overall structure and maintainability")

    return recommendations
