"""
Cache warmup utilities for memory-aware cache preloading.

This module provides utility functions for cache warmup operations,
including memory budget calculations, warmup item creation, and
integration with the existing cache services.
"""

import asyncio
import logging
import time
from typing import Any, Optional

from ..services.cache_warmup_service import WarmupItem, WarmupStrategy
from .memory_utils import (
    MemoryPressureLevel,
    get_cache_memory_stats,
    get_system_memory_pressure,
    get_total_cache_memory_usage,
)

logger = logging.getLogger(__name__)


def calculate_memory_budget(available_memory_mb: float, strategy: WarmupStrategy, pressure_level: MemoryPressureLevel) -> float:
    """Calculate memory budget for cache warmup based on strategy and pressure."""
    base_budgets = {
        WarmupStrategy.AGGRESSIVE: 0.8,  # 80% of available memory
        WarmupStrategy.BALANCED: 0.6,  # 60% of available memory
        WarmupStrategy.CONSERVATIVE: 0.2,  # 20% of available memory
        WarmupStrategy.ADAPTIVE: 0.5,  # 50% base, adjusted below
        WarmupStrategy.PRIORITY_BASED: 0.4,  # 40% of available memory
    }

    base_factor = base_budgets.get(strategy, 0.5)

    # Adjust based on memory pressure
    pressure_adjustments = {
        MemoryPressureLevel.CRITICAL: 0.0,  # No warmup
        MemoryPressureLevel.HIGH: 0.2,  # 20% of base
        MemoryPressureLevel.MODERATE: 0.6,  # 60% of base
        MemoryPressureLevel.LOW: 1.0,  # 100% of base
    }

    pressure_factor = pressure_adjustments.get(pressure_level, 1.0)

    # Calculate final budget
    budget = available_memory_mb * base_factor * pressure_factor

    # Ensure minimum budget if any warmup is allowed
    if pressure_level != MemoryPressureLevel.CRITICAL:
        budget = max(budget, 10.0)  # Minimum 10MB

    # Cap at reasonable maximum
    budget = min(budget, available_memory_mb * 0.9)  # Max 90% of available

    return budget


def estimate_item_memory_usage(cache_name: str, item_key: str, item_data: Any, cache_service: Any) -> float:
    """Estimate memory usage for a cache item."""
    try:
        # Try to get estimate from cache service
        if hasattr(cache_service, "estimate_item_size"):
            return cache_service.estimate_item_size(item_key, item_data)

        # Fallback to basic estimation
        if isinstance(item_data, str):
            # String data - estimate UTF-8 encoding
            return len(item_data.encode("utf-8")) / (1024 * 1024)  # Convert to MB
        elif isinstance(item_data, list | tuple):
            # List/tuple - estimate based on length
            return len(item_data) * 0.001  # Rough estimate: 1KB per item
        elif isinstance(item_data, dict):
            # Dictionary - estimate based on key/value pairs
            return len(str(item_data).encode("utf-8")) / (1024 * 1024)
        elif hasattr(item_data, "__sizeof__"):
            # Object with size method
            return item_data.__sizeof__() / (1024 * 1024)
        else:
            # Default estimation
            return 0.01  # 10KB default

    except Exception as e:
        logger.warning(f"Failed to estimate size for {cache_name}:{item_key}: {e}")
        return 0.01  # Default 10KB


def create_warmup_item(
    cache_name: str,
    item_key: str,
    item_data: Any,
    cache_service: Any,
    priority: int = 5,
    usage_frequency: float = 0.0,
    dependencies: set[str] | None = None,
) -> WarmupItem:
    """Create a warmup item with proper metadata."""
    estimated_size = estimate_item_memory_usage(cache_name, item_key, item_data, cache_service)

    # Estimate warmup cost based on data type and size
    warmup_cost = 0.1  # Base cost
    if isinstance(item_data, str) and len(item_data) > 10000:
        warmup_cost += 0.5  # Large string
    elif isinstance(item_data, list | tuple) and len(item_data) > 1000:
        warmup_cost += 0.3  # Large collection
    elif isinstance(item_data, dict) and len(item_data) > 100:
        warmup_cost += 0.2  # Large dictionary

    return WarmupItem(
        cache_name=cache_name,
        item_key=item_key,
        item_data=item_data,
        priority=priority,
        estimated_size_mb=estimated_size,
        usage_frequency=usage_frequency,
        dependencies=dependencies or set(),
        warmup_cost=warmup_cost,
    )


async def get_embedding_cache_warmup_candidates(embedding_cache_service: Any, historical_data: dict[str, Any]) -> list[WarmupItem]:
    """Get warmup candidates for embedding cache."""
    candidates = []

    try:
        # Get frequently used queries from historical data
        frequent_queries = historical_data.get("frequent_queries", [])

        for query_data in frequent_queries:
            query_text = query_data.get("query", "")
            frequency = query_data.get("frequency", 0.0)
            last_used = query_data.get("last_used", time.time())

            if query_text and frequency > 0.1:  # Only items used more than 10% of the time
                priority = min(10, int(frequency * 10))  # Convert frequency to priority

                item = create_warmup_item(
                    cache_name="embedding_cache",
                    item_key=query_text,
                    item_data=query_text,
                    cache_service=embedding_cache_service,
                    priority=priority,
                    usage_frequency=frequency,
                )

                item.last_accessed = last_used
                candidates.append(item)

        # Get model-specific embeddings
        cached_models = historical_data.get("cached_models", [])
        for model_info in cached_models:
            model_name = model_info.get("model", "")
            usage_count = model_info.get("usage_count", 0)

            if model_name and usage_count > 5:  # Only frequently used models
                priority = min(10, usage_count // 10)  # Convert usage to priority

                item = create_warmup_item(
                    cache_name="embedding_cache",
                    item_key=f"model:{model_name}",
                    item_data=model_info,
                    cache_service=embedding_cache_service,
                    priority=priority,
                    usage_frequency=min(1.0, usage_count / 100.0),
                )

                candidates.append(item)

    except Exception as e:
        logger.error(f"Failed to get embedding cache warmup candidates: {e}")

    return candidates


async def get_search_cache_warmup_candidates(search_cache_service: Any, historical_data: dict[str, Any]) -> list[WarmupItem]:
    """Get warmup candidates for search cache."""
    candidates = []

    try:
        # Get frequently used search queries
        frequent_searches = historical_data.get("frequent_searches", [])

        for search_data in frequent_searches:
            query = search_data.get("query", "")
            parameters = search_data.get("parameters", {})
            frequency = search_data.get("frequency", 0.0)
            results = search_data.get("cached_results", [])

            if query and frequency > 0.05:  # Only items used more than 5% of the time
                priority = min(10, int(frequency * 20))  # Convert frequency to priority

                # Create composite key for search query + parameters
                param_key = "_".join(f"{k}:{v}" for k, v in sorted(parameters.items()))
                cache_key = f"{query}_{param_key}"

                item = create_warmup_item(
                    cache_name="search_cache",
                    item_key=cache_key,
                    item_data={"query": query, "parameters": parameters, "results": results},
                    cache_service=search_cache_service,
                    priority=priority,
                    usage_frequency=frequency,
                )

                candidates.append(item)

        # Get frequently accessed projects
        project_searches = historical_data.get("project_searches", [])
        for project_data in project_searches:
            project_name = project_data.get("project", "")
            search_count = project_data.get("search_count", 0)

            if project_name and search_count > 10:  # Only frequently searched projects
                priority = min(10, search_count // 20)

                item = create_warmup_item(
                    cache_name="search_cache",
                    item_key=f"project:{project_name}",
                    item_data=project_data,
                    cache_service=search_cache_service,
                    priority=priority,
                    usage_frequency=min(1.0, search_count / 200.0),
                )

                candidates.append(item)

    except Exception as e:
        logger.error(f"Failed to get search cache warmup candidates: {e}")

    return candidates


async def get_file_cache_warmup_candidates(file_cache_service: Any, historical_data: dict[str, Any]) -> list[WarmupItem]:
    """Get warmup candidates for file cache."""
    candidates = []

    try:
        # Get frequently parsed files
        frequent_files = historical_data.get("frequent_files", [])

        for file_data in frequent_files:
            file_path = file_data.get("path", "")
            parse_count = file_data.get("parse_count", 0)
            file_size = file_data.get("size", 0)
            language = file_data.get("language", "")

            if file_path and parse_count > 3:  # Only files parsed more than 3 times
                priority = min(10, parse_count // 5)  # Convert parse count to priority

                # Adjust priority based on file size and language
                if language in ["python", "javascript", "typescript"]:
                    priority += 1  # Common languages get higher priority

                if file_size > 100000:  # Large files (>100KB)
                    priority += 2  # Large files benefit more from caching

                priority = min(10, priority)  # Cap at 10

                item = create_warmup_item(
                    cache_name="file_cache",
                    item_key=file_path,
                    item_data={"path": file_path, "size": file_size, "language": language, "parse_count": parse_count},
                    cache_service=file_cache_service,
                    priority=priority,
                    usage_frequency=min(1.0, parse_count / 50.0),
                )

                candidates.append(item)

        # Get frequently used language parsers
        language_usage = historical_data.get("language_usage", {})
        for language, usage_data in language_usage.items():
            usage_count = usage_data.get("count", 0)

            if usage_count > 20:  # Only frequently used languages
                priority = min(10, usage_count // 30)

                item = create_warmup_item(
                    cache_name="file_cache",
                    item_key=f"parser:{language}",
                    item_data={"language": language, "usage_count": usage_count},
                    cache_service=file_cache_service,
                    priority=priority,
                    usage_frequency=min(1.0, usage_count / 200.0),
                )

                candidates.append(item)

    except Exception as e:
        logger.error(f"Failed to get file cache warmup candidates: {e}")

    return candidates


async def get_project_cache_warmup_candidates(project_cache_service: Any, historical_data: dict[str, Any]) -> list[WarmupItem]:
    """Get warmup candidates for project cache."""
    candidates = []

    try:
        # Get frequently accessed projects
        frequent_projects = historical_data.get("frequent_projects", [])

        for project_data in frequent_projects:
            project_name = project_data.get("name", "")
            access_count = project_data.get("access_count", 0)
            project_info = project_data.get("info", {})

            if project_name and access_count > 5:  # Only frequently accessed projects
                priority = min(10, access_count // 10)

                item = create_warmup_item(
                    cache_name="project_cache",
                    item_key=project_name,
                    item_data=project_info,
                    cache_service=project_cache_service,
                    priority=priority,
                    usage_frequency=min(1.0, access_count / 100.0),
                )

                candidates.append(item)

        # Get frequently used project statistics
        project_stats = historical_data.get("project_stats", [])
        for stats_data in project_stats:
            project_name = stats_data.get("project", "")
            stats_requests = stats_data.get("requests", 0)

            if project_name and stats_requests > 10:
                priority = min(10, stats_requests // 15)

                item = create_warmup_item(
                    cache_name="project_cache",
                    item_key=f"stats:{project_name}",
                    item_data=stats_data,
                    cache_service=project_cache_service,
                    priority=priority,
                    usage_frequency=min(1.0, stats_requests / 150.0),
                )

                candidates.append(item)

    except Exception as e:
        logger.error(f"Failed to get project cache warmup candidates: {e}")

    return candidates


def prioritize_warmup_items(items: list[WarmupItem], memory_budget: float, strategy: WarmupStrategy) -> list[WarmupItem]:
    """Prioritize warmup items based on strategy and memory budget."""
    if not items:
        return []

    # Sort by warmup score (higher is better)
    items.sort(key=lambda x: x.warmup_score, reverse=True)

    if strategy == WarmupStrategy.PRIORITY_BASED:
        # Focus on highest priority items only
        items = [item for item in items if item.priority >= 7]
    elif strategy == WarmupStrategy.CONSERVATIVE:
        # Focus on high priority and frequent items
        items = [item for item in items if item.priority >= 6 and item.usage_frequency >= 0.3]
    elif strategy == WarmupStrategy.BALANCED:
        # Include medium priority items with decent frequency
        items = [item for item in items if item.priority >= 4 and item.usage_frequency >= 0.1]
    # AGGRESSIVE strategy uses all items

    # Filter by memory budget
    selected_items = []
    total_memory = 0.0

    for item in items:
        if total_memory + item.estimated_size_mb <= memory_budget:
            selected_items.append(item)
            total_memory += item.estimated_size_mb
        else:
            break

    return selected_items


def group_items_by_dependencies(items: list[WarmupItem]) -> list[list[WarmupItem]]:
    """Group warmup items by their dependencies to ensure proper loading order."""
    if not items:
        return []

    # Create dependency graph
    dependency_graph = {}
    items_by_key = {}

    for item in items:
        key = f"{item.cache_name}:{item.item_key}"
        items_by_key[key] = item
        dependency_graph[key] = item.dependencies

    # Topological sort to resolve dependencies
    visited = set()
    temp_visited = set()
    ordered_items = []

    def visit(key: str):
        if key in temp_visited:
            # Circular dependency - skip
            return
        if key in visited:
            return

        temp_visited.add(key)

        # Visit dependencies first
        for dep in dependency_graph.get(key, set()):
            if dep in items_by_key:
                visit(dep)

        temp_visited.remove(key)
        visited.add(key)

        if key in items_by_key:
            ordered_items.append(items_by_key[key])

    # Visit all items
    for key in items_by_key:
        visit(key)

    # Group into phases based on dependencies
    phases = []
    current_phase = []

    for item in ordered_items:
        # Check if all dependencies are satisfied by previous phases
        dependencies_satisfied = True

        for dep in item.dependencies:
            if dep not in [f"{i.cache_name}:{i.item_key}" for phase in phases for i in phase]:
                dependencies_satisfied = False
                break

        if dependencies_satisfied:
            current_phase.append(item)
        else:
            # Start new phase
            if current_phase:
                phases.append(current_phase)
                current_phase = [item]
            else:
                current_phase.append(item)

    # Add final phase
    if current_phase:
        phases.append(current_phase)

    return phases


async def validate_warmup_environment() -> dict[str, Any]:
    """Validate that the environment is suitable for cache warmup."""
    system_pressure = get_system_memory_pressure()
    cache_memory_usage = get_total_cache_memory_usage()

    validation_result = {
        "suitable_for_warmup": True,
        "issues": [],
        "recommendations": [],
        "system_info": {
            "memory_pressure": system_pressure.level.value,
            "available_memory_mb": system_pressure.available_mb,
            "cache_memory_usage_mb": cache_memory_usage,
            "total_memory_mb": system_pressure.total_mb,
        },
    }

    # Check memory pressure
    if system_pressure.level == MemoryPressureLevel.CRITICAL:
        validation_result["suitable_for_warmup"] = False
        validation_result["issues"].append("Critical memory pressure detected")
        validation_result["recommendations"].append("Wait for memory pressure to decrease before warmup")

    elif system_pressure.level == MemoryPressureLevel.HIGH:
        validation_result["issues"].append("High memory pressure detected")
        validation_result["recommendations"].append("Use conservative warmup strategy")

    # Check available memory
    if system_pressure.available_mb < 100:
        validation_result["suitable_for_warmup"] = False
        validation_result["issues"].append("Insufficient available memory (<100MB)")
        validation_result["recommendations"].append("Free up memory before attempting warmup")

    # Check cache memory usage
    if cache_memory_usage > system_pressure.total_mb * 0.5:
        validation_result["issues"].append("Cache memory usage is already high (>50% of total)")
        validation_result["recommendations"].append("Consider clearing some caches before warmup")

    # Check individual cache stats
    cache_stats = get_cache_memory_stats()
    if isinstance(cache_stats, dict):
        for cache_name, stats in cache_stats.items():
            if stats.pressure_events > 5:
                validation_result["issues"].append(f"Cache '{cache_name}' has frequent pressure events")
                validation_result["recommendations"].append(f"Optimize '{cache_name}' cache before warmup")

    return validation_result


def calculate_warmup_time_estimate(items: list[WarmupItem]) -> float:
    """Estimate time required for warmup based on items."""
    if not items:
        return 0.0

    # Base time estimates (in seconds)
    base_time_per_item = 0.1  # 100ms per item
    size_factor = 0.01  # 10ms per MB

    total_time = 0.0
    for item in items:
        item_time = base_time_per_item + (item.estimated_size_mb * size_factor) + item.warmup_cost
        total_time += item_time

    # Add overhead for batching and coordination
    overhead_factor = 1.2  # 20% overhead
    total_time *= overhead_factor

    return total_time


def get_warmup_memory_recommendations(available_memory_mb: float, current_cache_usage_mb: float) -> dict[str, Any]:
    """Get memory recommendations for cache warmup."""
    recommendations = {"memory_budget_options": {}, "recommended_strategy": None, "warnings": [], "optimizations": []}

    # Calculate budget options
    strategies = [(WarmupStrategy.CONSERVATIVE, 0.2), (WarmupStrategy.BALANCED, 0.5), (WarmupStrategy.AGGRESSIVE, 0.8)]

    for strategy, factor in strategies:
        budget = available_memory_mb * factor
        recommendations["memory_budget_options"][strategy.value] = {
            "budget_mb": budget,
            "percentage_of_available": factor * 100,
            "safe_for_system": budget < available_memory_mb * 0.9,
        }

    # Recommend strategy based on current usage
    usage_ratio = current_cache_usage_mb / available_memory_mb

    if usage_ratio > 0.7:
        recommendations["recommended_strategy"] = WarmupStrategy.CONSERVATIVE.value
        recommendations["warnings"].append("High current cache usage - use conservative warmup")
    elif usage_ratio > 0.4:
        recommendations["recommended_strategy"] = WarmupStrategy.BALANCED.value
    else:
        recommendations["recommended_strategy"] = WarmupStrategy.BALANCED.value

    # Add optimization suggestions
    if current_cache_usage_mb > 500:  # 500MB
        recommendations["optimizations"].append("Consider clearing unused caches before warmup")

    if available_memory_mb < 1000:  # 1GB
        recommendations["optimizations"].append("Limited available memory - consider increasing system memory")

    return recommendations
