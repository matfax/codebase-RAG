"""
MCP tools for cascade invalidation management.

This module provides tools to manage cascade invalidation rules, monitor
dependency relationships, and analyze cascade invalidation performance.
"""

import logging
from typing import Any

from ...services.cache_invalidation_service import get_cache_invalidation_service
from ...services.cascade_invalidation_service import CascadeStrategy, DependencyType

logger = logging.getLogger(__name__)


async def add_cascade_dependency_rule_tool(
    source_pattern: str,
    target_pattern: str,
    dependency_type: str = "file_content",
    cascade_strategy: str = "immediate",
    condition: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Add a cascade invalidation dependency rule.

    Args:
        source_pattern: Pattern matching source cache keys (supports wildcards)
        target_pattern: Pattern matching dependent cache keys (supports wildcards)
        dependency_type: Type of dependency (file_content, file_metadata, project_structure, etc.)
        cascade_strategy: Strategy for cascade invalidation (immediate, parallel, delayed, lazy, selective)
        condition: Optional condition for dependency
        metadata: Optional metadata for the rule

    Returns:
        Dictionary with rule creation results
    """
    try:
        # Validate dependency type
        try:
            dep_type = DependencyType(dependency_type.lower())
        except ValueError:
            valid_types = [t.value for t in DependencyType]
            return {
                "success": False,
                "error": f"Invalid dependency type: {dependency_type}",
                "valid_types": valid_types,
            }

        # Validate cascade strategy
        try:
            strategy = CascadeStrategy(cascade_strategy.lower())
        except ValueError:
            valid_strategies = [s.value for s in CascadeStrategy]
            return {
                "success": False,
                "error": f"Invalid cascade strategy: {cascade_strategy}",
                "valid_strategies": valid_strategies,
            }

        # Get invalidation service and add rule
        invalidation_service = await get_cache_invalidation_service()
        invalidation_service.add_cascade_dependency_rule(
            source_pattern=source_pattern,
            target_pattern=target_pattern,
            dependency_type=dep_type,
            cascade_strategy=strategy,
            condition=condition,
            metadata=metadata or {},
        )

        return {
            "success": True,
            "source_pattern": source_pattern,
            "target_pattern": target_pattern,
            "dependency_type": dependency_type,
            "cascade_strategy": cascade_strategy,
            "message": f"Added cascade dependency rule: {source_pattern} -> {target_pattern}",
        }

    except Exception as e:
        logger.error(f"Failed to add cascade dependency rule: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to add cascade dependency rule",
        }


async def add_explicit_cascade_dependency_tool(source_key: str, dependent_keys: list[str]) -> dict[str, Any]:
    """
    Add explicit cascade dependency relationship.

    Args:
        source_key: Source cache key
        dependent_keys: List of cache keys that depend on the source

    Returns:
        Dictionary with dependency creation results
    """
    try:
        invalidation_service = await get_cache_invalidation_service()
        invalidation_service.add_explicit_cascade_dependency(source_key, dependent_keys)

        return {
            "success": True,
            "source_key": source_key,
            "dependent_keys": dependent_keys,
            "dependency_count": len(dependent_keys),
            "message": f"Added explicit dependency: {source_key} -> {len(dependent_keys)} keys",
        }

    except Exception as e:
        logger.error(f"Failed to add explicit cascade dependency: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to add explicit cascade dependency",
        }


async def get_cascade_stats_tool() -> dict[str, Any]:
    """
    Get cascade invalidation statistics and performance metrics.

    Returns:
        Dictionary with cascade statistics and metrics
    """
    try:
        invalidation_service = await get_cache_invalidation_service()
        cascade_stats = invalidation_service.get_cascade_stats()

        return {
            "success": True,
            "cascade_statistics": cascade_stats,
            "message": "Retrieved cascade invalidation statistics",
        }

    except Exception as e:
        logger.error(f"Failed to get cascade stats: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve cascade statistics",
        }


async def get_dependency_graph_tool() -> dict[str, Any]:
    """
    Get the cache dependency graph showing all relationships.

    Returns:
        Dictionary with dependency graph information
    """
    try:
        invalidation_service = await get_cache_invalidation_service()
        dependency_graph = invalidation_service.get_dependency_graph()

        return {
            "success": True,
            "dependency_graph": dependency_graph,
            "message": "Retrieved cache dependency graph",
        }

    except Exception as e:
        logger.error(f"Failed to get dependency graph: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to retrieve dependency graph",
        }


async def detect_circular_dependencies_tool() -> dict[str, Any]:
    """
    Detect circular dependencies in the cache dependency graph.

    Returns:
        Dictionary with circular dependency detection results
    """
    try:
        invalidation_service = await get_cache_invalidation_service()
        circular_deps = invalidation_service.detect_circular_dependencies()

        return {
            "success": True,
            "circular_dependencies": circular_deps,
            "circular_count": len(circular_deps),
            "has_circular_dependencies": len(circular_deps) > 0,
            "message": f"Found {len(circular_deps)} circular dependencies" if circular_deps else "No circular dependencies detected",
        }

    except Exception as e:
        logger.error(f"Failed to detect circular dependencies: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to detect circular dependencies",
        }


async def test_cascade_invalidation_tool(test_keys: list[str], max_depth: int = 3) -> dict[str, Any]:
    """
    Test cascade invalidation for specific cache keys.

    Args:
        test_keys: List of cache keys to test cascade invalidation for
        max_depth: Maximum cascade depth to test

    Returns:
        Dictionary with cascade test results
    """
    try:
        invalidation_service = await get_cache_invalidation_service()

        # This is a simulation of cascade invalidation
        # In practice, you'd want to test without actually invalidating
        test_results = {
            "test_keys": test_keys,
            "max_depth": max_depth,
            "simulated_cascades": [],
            "total_affected_keys": 0,
        }

        # Get dependency graph to simulate cascade
        dependency_graph = invalidation_service.get_dependency_graph()

        # Simulate cascade for each test key
        for test_key in test_keys:
            cascade_simulation = await _simulate_cascade(test_key, dependency_graph, max_depth)
            test_results["simulated_cascades"].append(cascade_simulation)
            test_results["total_affected_keys"] += len(cascade_simulation.get("affected_keys", []))

        return {
            "success": True,
            "test_results": test_results,
            "message": f"Cascade test completed for {len(test_keys)} keys",
        }

    except Exception as e:
        logger.error(f"Failed to test cascade invalidation: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to test cascade invalidation",
        }


async def _simulate_cascade(start_key: str, dependency_graph: dict[str, Any], max_depth: int) -> dict[str, Any]:
    """
    Simulate cascade invalidation for a specific key.

    Args:
        start_key: Starting cache key
        dependency_graph: Dependency graph information
        max_depth: Maximum cascade depth

    Returns:
        Simulation results
    """
    affected_keys = [start_key]
    cascade_levels = {0: [start_key]}

    current_level = 0
    current_keys = [start_key]

    while current_level < max_depth and current_keys:
        next_level_keys = []

        # Find dependencies for current level keys
        explicit_deps = dependency_graph.get("explicit_dependencies", {})
        for key in current_keys:
            if key in explicit_deps:
                deps = explicit_deps[key]
                next_level_keys.extend(deps)
                affected_keys.extend(deps)

        current_level += 1
        if next_level_keys:
            cascade_levels[current_level] = list(set(next_level_keys))
            current_keys = next_level_keys
        else:
            break

    return {
        "start_key": start_key,
        "affected_keys": list(set(affected_keys)),
        "cascade_levels": cascade_levels,
        "max_depth_reached": current_level,
        "total_affected": len(set(affected_keys)),
    }


async def configure_cascade_settings_tool(
    max_cascade_depth: int | None = None,
    enable_circular_detection: bool | None = None,
    enable_lazy_invalidation: bool | None = None,
    batch_delay_seconds: float | None = None,
) -> dict[str, Any]:
    """
    Configure cascade invalidation settings.

    Args:
        max_cascade_depth: Maximum depth for cascade invalidation
        enable_circular_detection: Whether to detect circular dependencies
        enable_lazy_invalidation: Whether to enable lazy invalidation
        batch_delay_seconds: Delay for batch invalidation in seconds

    Returns:
        Dictionary with configuration results
    """
    try:
        invalidation_service = await get_cache_invalidation_service()

        # Configure cascade service if available
        if hasattr(invalidation_service, "_cascade_service") and invalidation_service._cascade_service:
            invalidation_service._cascade_service.configure(
                max_cascade_depth=max_cascade_depth,
                enable_circular_detection=enable_circular_detection,
                enable_lazy_invalidation=enable_lazy_invalidation,
                batch_delay_seconds=batch_delay_seconds,
            )

            configuration = {
                "max_cascade_depth": invalidation_service._cascade_service.max_cascade_depth,
                "enable_circular_detection": invalidation_service._cascade_service._enable_circular_detection,
                "enable_lazy_invalidation": invalidation_service._cascade_service._enable_lazy_invalidation,
                "batch_delay_seconds": invalidation_service._cascade_service._batch_delay_seconds,
            }

            return {
                "success": True,
                "configuration": configuration,
                "message": "Updated cascade invalidation configuration",
            }
        else:
            return {
                "success": False,
                "error": "Cascade service not available",
                "message": "Cannot configure cascade settings - service not initialized",
            }

    except Exception as e:
        logger.error(f"Failed to configure cascade settings: {e}")
        return {
            "success": False,
            "error": str(e),
            "message": "Failed to configure cascade settings",
        }


def register_cascade_invalidation_tools(server) -> None:
    """
    Register cascade invalidation tools with the MCP server.

    Args:
        server: The MCP server instance
    """

    @server.tool()
    async def add_cascade_dependency_rule(
        source_pattern: str,
        target_pattern: str,
        dependency_type: str = "file_content",
        cascade_strategy: str = "immediate",
        condition: str = None,
        metadata: dict[str, Any] = None,
    ) -> dict[str, Any]:
        """Add a cascade invalidation dependency rule.

        Args:
            source_pattern: Pattern matching source cache keys (supports wildcards)
            target_pattern: Pattern matching dependent cache keys (supports wildcards)
            dependency_type: Type of dependency (file_content, file_metadata, project_structure,
                cross_project, computed_result, aggregated_data, derived_cache)
            cascade_strategy: Strategy for cascade invalidation (immediate, parallel, delayed, lazy, selective)
            condition: Optional condition for dependency
            metadata: Optional metadata for the rule
        """
        return await add_cascade_dependency_rule_tool(
            source_pattern, target_pattern, dependency_type, cascade_strategy, condition, metadata
        )

    @server.tool()
    async def add_explicit_cascade_dependency(source_key: str, dependent_keys: list[str]) -> dict[str, Any]:
        """Add explicit cascade dependency relationship.

        Args:
            source_key: Source cache key
            dependent_keys: List of cache keys that depend on the source
        """
        return await add_explicit_cascade_dependency_tool(source_key, dependent_keys)

    @server.tool()
    async def get_cascade_stats() -> dict[str, Any]:
        """Get cascade invalidation statistics and performance metrics."""
        return await get_cascade_stats_tool()

    @server.tool()
    async def get_dependency_graph() -> dict[str, Any]:
        """Get the cache dependency graph showing all relationships."""
        return await get_dependency_graph_tool()

    @server.tool()
    async def detect_circular_dependencies() -> dict[str, Any]:
        """Detect circular dependencies in the cache dependency graph."""
        return await detect_circular_dependencies_tool()

    @server.tool()
    async def test_cascade_invalidation(test_keys: list[str], max_depth: int = 3) -> dict[str, Any]:
        """Test cascade invalidation for specific cache keys.

        Args:
            test_keys: List of cache keys to test cascade invalidation for
            max_depth: Maximum cascade depth to test
        """
        return await test_cascade_invalidation_tool(test_keys, max_depth)

    @server.tool()
    async def configure_cascade_settings(
        max_cascade_depth: int = None,
        enable_circular_detection: bool = None,
        enable_lazy_invalidation: bool = None,
        batch_delay_seconds: float = None,
    ) -> dict[str, Any]:
        """Configure cascade invalidation settings.

        Args:
            max_cascade_depth: Maximum depth for cascade invalidation
            enable_circular_detection: Whether to detect circular dependencies
            enable_lazy_invalidation: Whether to enable lazy invalidation
            batch_delay_seconds: Delay for batch invalidation in seconds
        """
        return await configure_cascade_settings_tool(
            max_cascade_depth, enable_circular_detection, enable_lazy_invalidation, batch_delay_seconds
        )

    logger.info("Registered cascade invalidation MCP tools")
