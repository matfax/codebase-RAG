"""
Cascade invalidation service for managing complex cache dependencies.

This service provides sophisticated cascade invalidation capabilities for handling
complex cache dependencies, cross-project invalidation, and intelligent dependency
tracking to ensure cache consistency across the entire system.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional

from ..models.file_metadata import FileMetadata
from ..services.cache_invalidation_service import InvalidationEvent, InvalidationReason


class DependencyType(Enum):
    """Types of cache dependencies."""

    FILE_CONTENT = "file_content"  # Cache depends on file content
    FILE_METADATA = "file_metadata"  # Cache depends on file metadata
    PROJECT_STRUCTURE = "project_structure"  # Cache depends on project structure
    CROSS_PROJECT = "cross_project"  # Cache depends on other projects
    COMPUTED_RESULT = "computed_result"  # Cache depends on computed results
    AGGREGATED_DATA = "aggregated_data"  # Cache depends on aggregated data
    DERIVED_CACHE = "derived_cache"  # Cache depends on other caches


class CascadeStrategy(Enum):
    """Strategies for cascade invalidation."""

    IMMEDIATE = "immediate"  # Invalidate immediately in sequence
    PARALLEL = "parallel"  # Invalidate dependents in parallel
    DELAYED = "delayed"  # Delay invalidation with batching
    LAZY = "lazy"  # Mark for lazy invalidation
    SELECTIVE = "selective"  # Only invalidate if content actually changed


@dataclass
class DependencyRule:
    """Rule defining a cache dependency relationship."""

    source_pattern: str  # Pattern matching source cache keys
    target_pattern: str  # Pattern matching dependent cache keys
    dependency_type: DependencyType
    cascade_strategy: CascadeStrategy = CascadeStrategy.IMMEDIATE
    condition: str | None = None  # Optional condition for dependency
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches_source(self, cache_key: str) -> bool:
        """Check if cache key matches source pattern."""
        import fnmatch

        return fnmatch.fnmatch(cache_key, self.source_pattern)

    def get_dependent_keys(self, source_key: str, all_keys: list[str]) -> list[str]:
        """Get dependent cache keys based on target pattern."""
        import fnmatch

        # If target pattern contains variables from source, substitute them
        target_pattern = self._substitute_pattern_variables(source_key, self.target_pattern)

        return [key for key in all_keys if fnmatch.fnmatch(key, target_pattern)]

    def _substitute_pattern_variables(self, source_key: str, target_pattern: str) -> str:
        """Substitute pattern variables from source key into target pattern."""
        # Simple substitution for common patterns
        # This could be enhanced with more sophisticated pattern matching
        if "{project}" in target_pattern and ":" in source_key:
            parts = source_key.split(":")
            if len(parts) >= 2:
                project_part = parts[1] if parts[0] in ["file", "embedding", "search"] else parts[0]
                target_pattern = target_pattern.replace("{project}", project_part)

        return target_pattern


@dataclass
class CascadeEvent:
    """Represents a cascade invalidation event."""

    cascade_id: str
    root_event: InvalidationEvent
    cascade_level: int
    dependency_type: DependencyType
    cascade_strategy: CascadeStrategy
    source_keys: list[str]
    target_keys: list[str]
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            "cascade_id": self.cascade_id,
            "root_event_id": self.root_event.event_id,
            "cascade_level": self.cascade_level,
            "dependency_type": self.dependency_type.value,
            "cascade_strategy": self.cascade_strategy.value,
            "source_keys": self.source_keys,
            "target_keys": self.target_keys,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class CascadeStats:
    """Statistics for cascade invalidation operations."""

    total_cascades: int = 0
    cascade_levels_reached: dict[int, int] = field(default_factory=dict)
    strategies_used: dict[CascadeStrategy, int] = field(default_factory=dict)
    dependency_types: dict[DependencyType, int] = field(default_factory=dict)
    keys_invalidated: int = 0
    avg_cascade_time: float = 0.0
    max_cascade_depth: int = 0
    circular_dependencies_detected: int = 0
    last_cascade: datetime | None = None

    def update(self, cascade_event: CascadeEvent, duration: float, keys_invalidated: int, max_depth: int) -> None:
        """Update statistics with cascade event data."""
        self.total_cascades += 1
        self.keys_invalidated += keys_invalidated
        self.last_cascade = datetime.now()

        # Update cascade level tracking
        level = cascade_event.cascade_level
        if level not in self.cascade_levels_reached:
            self.cascade_levels_reached[level] = 0
        self.cascade_levels_reached[level] += 1

        # Update strategy usage
        strategy = cascade_event.cascade_strategy
        if strategy not in self.strategies_used:
            self.strategies_used[strategy] = 0
        self.strategies_used[strategy] += 1

        # Update dependency type usage
        dep_type = cascade_event.dependency_type
        if dep_type not in self.dependency_types:
            self.dependency_types[dep_type] = 0
        self.dependency_types[dep_type] += 1

        # Update averages
        if self.total_cascades > 0:
            self.avg_cascade_time = (self.avg_cascade_time * (self.total_cascades - 1) + duration) / self.total_cascades

        # Update max depth
        self.max_cascade_depth = max(self.max_cascade_depth, max_depth)


class CascadeInvalidationService:
    """
    Service for managing cascade invalidation of dependent caches.

    This service provides sophisticated dependency tracking and cascade invalidation
    to ensure cache consistency across complex dependency relationships.
    """

    def __init__(self, max_cascade_depth: int = 5):
        """
        Initialize the cascade invalidation service.

        Args:
            max_cascade_depth: Maximum depth for cascade invalidation
        """
        self.logger = logging.getLogger(__name__)
        self.max_cascade_depth = max_cascade_depth

        # Dependency management
        self._dependency_rules: list[DependencyRule] = []
        self._explicit_dependencies: dict[str, set[str]] = {}  # key -> dependent_keys
        self._reverse_dependencies: dict[str, set[str]] = {}  # key -> source_keys

        # Cascade tracking
        self._active_cascades: dict[str, CascadeEvent] = {}
        self._cascade_history: list[CascadeEvent] = []
        self._circular_dependency_cache: dict[str, set[str]] = {}

        # Statistics
        self._stats = CascadeStats()

        # Configuration
        self._enable_circular_detection = True
        self._enable_lazy_invalidation = True
        self._batch_delay_seconds = 1.0
        self._max_history_size = 1000

        # Default dependency rules
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Set up default dependency rules for common cache patterns."""
        # File content changes invalidate related caches
        self.add_dependency_rule(
            DependencyRule(
                source_pattern="file:*",
                target_pattern="embedding:*:{project}:*",
                dependency_type=DependencyType.FILE_CONTENT,
                cascade_strategy=CascadeStrategy.IMMEDIATE,
            )
        )

        # File changes invalidate search results
        self.add_dependency_rule(
            DependencyRule(
                source_pattern="file:*",
                target_pattern="search:*",
                dependency_type=DependencyType.FILE_CONTENT,
                cascade_strategy=CascadeStrategy.PARALLEL,
            )
        )

        # Embedding changes invalidate derived search results
        self.add_dependency_rule(
            DependencyRule(
                source_pattern="embedding:*",
                target_pattern="search:*:{project}:*",
                dependency_type=DependencyType.DERIVED_CACHE,
                cascade_strategy=CascadeStrategy.DELAYED,
            )
        )

        # Project changes invalidate aggregated data
        self.add_dependency_rule(
            DependencyRule(
                source_pattern="project:*",
                target_pattern="aggregated:*:{project}:*",
                dependency_type=DependencyType.PROJECT_STRUCTURE,
                cascade_strategy=CascadeStrategy.SELECTIVE,
            )
        )

        # File metadata changes have limited impact
        self.add_dependency_rule(
            DependencyRule(
                source_pattern="file_metadata:*",
                target_pattern="project_stats:*",
                dependency_type=DependencyType.FILE_METADATA,
                cascade_strategy=CascadeStrategy.LAZY,
            )
        )

    def add_dependency_rule(self, rule: DependencyRule) -> None:
        """
        Add a dependency rule for cascade invalidation.

        Args:
            rule: Dependency rule to add
        """
        self._dependency_rules.append(rule)
        self.logger.debug(f"Added dependency rule: {rule.source_pattern} -> {rule.target_pattern}")

    def remove_dependency_rule(self, source_pattern: str, target_pattern: str) -> bool:
        """
        Remove a dependency rule.

        Args:
            source_pattern: Source pattern of the rule to remove
            target_pattern: Target pattern of the rule to remove

        Returns:
            True if rule was found and removed
        """
        for i, rule in enumerate(self._dependency_rules):
            if rule.source_pattern == source_pattern and rule.target_pattern == target_pattern:
                del self._dependency_rules[i]
                self.logger.debug(f"Removed dependency rule: {source_pattern} -> {target_pattern}")
                return True
        return False

    def add_explicit_dependency(self, source_key: str, dependent_keys: list[str]) -> None:
        """
        Add explicit dependency relationships.

        Args:
            source_key: Source cache key
            dependent_keys: List of keys that depend on the source
        """
        if source_key not in self._explicit_dependencies:
            self._explicit_dependencies[source_key] = set()

        self._explicit_dependencies[source_key].update(dependent_keys)

        # Update reverse dependencies
        for dep_key in dependent_keys:
            if dep_key not in self._reverse_dependencies:
                self._reverse_dependencies[dep_key] = set()
            self._reverse_dependencies[dep_key].add(source_key)

        self.logger.debug(f"Added explicit dependencies for {source_key}: {dependent_keys}")

    def remove_explicit_dependency(self, source_key: str, dependent_key: str) -> None:
        """
        Remove an explicit dependency relationship.

        Args:
            source_key: Source cache key
            dependent_key: Dependent cache key to remove
        """
        if source_key in self._explicit_dependencies:
            self._explicit_dependencies[source_key].discard(dependent_key)
            if not self._explicit_dependencies[source_key]:
                del self._explicit_dependencies[source_key]

        if dependent_key in self._reverse_dependencies:
            self._reverse_dependencies[dependent_key].discard(source_key)
            if not self._reverse_dependencies[dependent_key]:
                del self._reverse_dependencies[dependent_key]

    async def cascade_invalidate(
        self, root_event: InvalidationEvent, available_keys: list[str], invalidation_callback: Any
    ) -> list[CascadeEvent]:
        """
        Perform cascade invalidation based on dependency rules.

        Args:
            root_event: The root invalidation event that triggered cascade
            available_keys: List of all available cache keys
            invalidation_callback: Callback function to perform actual invalidation

        Returns:
            List of cascade events that were processed
        """
        start_time = time.time()
        cascade_events = []

        try:
            # Generate cascade ID
            cascade_id = f"cascade_{int(time.time() * 1000)}"

            # Track processed keys to prevent circular dependencies
            processed_keys = set(root_event.affected_keys)
            cascade_queue = [(root_event.affected_keys, 0)]  # (keys, level)

            while cascade_queue and len(cascade_events) < 100:  # Safety limit
                current_keys, level = cascade_queue.pop(0)

                if level >= self.max_cascade_depth:
                    self.logger.warning(f"Maximum cascade depth ({self.max_cascade_depth}) reached")
                    break

                # Find dependent keys for current level
                dependent_keys = await self._find_dependent_keys(current_keys, available_keys)

                # Remove already processed keys to prevent cycles
                new_dependent_keys = [key for key in dependent_keys if key not in processed_keys]

                if not new_dependent_keys:
                    continue

                # Group dependent keys by strategy
                strategy_groups = self._group_keys_by_strategy(current_keys, new_dependent_keys, available_keys)

                # Process each strategy group
                for strategy, (dep_type, keys_to_invalidate) in strategy_groups.items():
                    if not keys_to_invalidate:
                        continue

                    # Create cascade event
                    cascade_event = CascadeEvent(
                        cascade_id=cascade_id,
                        root_event=root_event,
                        cascade_level=level + 1,
                        dependency_type=dep_type,
                        cascade_strategy=strategy,
                        source_keys=current_keys,
                        target_keys=keys_to_invalidate,
                        metadata={"level": level + 1},
                    )

                    # Execute invalidation based on strategy
                    invalidated_keys = await self._execute_cascade_strategy(cascade_event, keys_to_invalidate, invalidation_callback)

                    if invalidated_keys:
                        cascade_events.append(cascade_event)
                        processed_keys.update(invalidated_keys)

                        # Add to queue for next level if not at max depth
                        if level + 1 < self.max_cascade_depth:
                            cascade_queue.append((invalidated_keys, level + 1))

            # Update statistics
            duration = time.time() - start_time
            total_keys = sum(len(event.target_keys) for event in cascade_events)
            max_depth = max((event.cascade_level for event in cascade_events), default=0)

            for event in cascade_events:
                self._stats.update(event, duration / len(cascade_events) if cascade_events else 0, len(event.target_keys), max_depth)

            # Store cascade history
            self._cascade_history.extend(cascade_events)
            if len(self._cascade_history) > self._max_history_size:
                self._cascade_history = self._cascade_history[-self._max_history_size :]

            self.logger.info(
                f"Cascade invalidation completed: {len(cascade_events)} events, " f"{total_keys} keys invalidated, max depth: {max_depth}"
            )

            return cascade_events

        except Exception as e:
            self.logger.error(f"Error during cascade invalidation: {e}")
            return cascade_events

    async def _find_dependent_keys(self, source_keys: list[str], available_keys: list[str]) -> list[str]:
        """
        Find all keys that depend on the given source keys.

        Args:
            source_keys: Source cache keys
            available_keys: All available cache keys

        Returns:
            List of dependent cache keys
        """
        dependent_keys = set()

        for source_key in source_keys:
            # Check explicit dependencies
            if source_key in self._explicit_dependencies:
                dependent_keys.update(self._explicit_dependencies[source_key])

            # Check rule-based dependencies
            for rule in self._dependency_rules:
                if rule.matches_source(source_key):
                    rule_dependents = rule.get_dependent_keys(source_key, available_keys)
                    dependent_keys.update(rule_dependents)

        return list(dependent_keys)

    def _group_keys_by_strategy(
        self, source_keys: list[str], dependent_keys: list[str], available_keys: list[str]
    ) -> dict[CascadeStrategy, tuple[DependencyType, list[str]]]:
        """
        Group dependent keys by their cascade strategy.

        Args:
            source_keys: Source cache keys
            dependent_keys: Dependent cache keys
            available_keys: All available cache keys

        Returns:
            Dictionary mapping strategies to (dependency_type, keys) tuples
        """
        strategy_groups: dict[CascadeStrategy, tuple[DependencyType, list[str]]] = {}

        for dep_key in dependent_keys:
            # Find the rule that created this dependency
            rule = self._find_rule_for_dependency(source_keys, dep_key, available_keys)

            if rule:
                strategy = rule.cascade_strategy
                dep_type = rule.dependency_type

                if strategy not in strategy_groups:
                    strategy_groups[strategy] = (dep_type, [])

                strategy_groups[strategy][1].append(dep_key)

        return strategy_groups

    def _find_rule_for_dependency(self, source_keys: list[str], dependent_key: str, available_keys: list[str]) -> DependencyRule | None:
        """
        Find the rule that created a specific dependency.

        Args:
            source_keys: Source cache keys
            dependent_key: Dependent cache key
            available_keys: All available cache keys

        Returns:
            The dependency rule that matches, or None
        """
        for source_key in source_keys:
            for rule in self._dependency_rules:
                if rule.matches_source(source_key):
                    rule_dependents = rule.get_dependent_keys(source_key, available_keys)
                    if dependent_key in rule_dependents:
                        return rule
        return None

    async def _execute_cascade_strategy(
        self, cascade_event: CascadeEvent, keys_to_invalidate: list[str], invalidation_callback: Any
    ) -> list[str]:
        """
        Execute invalidation based on cascade strategy.

        Args:
            cascade_event: Cascade event information
            keys_to_invalidate: Keys to invalidate
            invalidation_callback: Callback function for invalidation

        Returns:
            List of keys that were actually invalidated
        """
        strategy = cascade_event.cascade_strategy
        invalidated_keys = []

        try:
            if strategy == CascadeStrategy.IMMEDIATE:
                # Invalidate immediately in sequence
                for key in keys_to_invalidate:
                    await invalidation_callback([key])
                    invalidated_keys.append(key)

            elif strategy == CascadeStrategy.PARALLEL:
                # Invalidate all keys in parallel
                tasks = [invalidation_callback([key]) for key in keys_to_invalidate]
                await asyncio.gather(*tasks, return_exceptions=True)
                invalidated_keys = keys_to_invalidate

            elif strategy == CascadeStrategy.DELAYED:
                # Batch invalidation with delay
                await asyncio.sleep(self._batch_delay_seconds)
                await invalidation_callback(keys_to_invalidate)
                invalidated_keys = keys_to_invalidate

            elif strategy == CascadeStrategy.LAZY:
                # Mark for lazy invalidation (implementation specific)
                if self._enable_lazy_invalidation:
                    # For now, we'll just log them
                    self.logger.debug(f"Marked {len(keys_to_invalidate)} keys for lazy invalidation")
                    # In a real implementation, these would be stored for later processing
                    invalidated_keys = keys_to_invalidate

            elif strategy == CascadeStrategy.SELECTIVE:
                # Only invalidate if content actually changed
                # This would require content comparison logic
                # For now, we'll invalidate all keys
                await invalidation_callback(keys_to_invalidate)
                invalidated_keys = keys_to_invalidate

        except Exception as e:
            self.logger.error(f"Error executing cascade strategy {strategy.value}: {e}")

        return invalidated_keys

    def detect_circular_dependencies(self, max_depth: int = 10) -> dict[str, list[str]]:
        """
        Detect circular dependencies in the cache dependency graph.

        Args:
            max_depth: Maximum depth to search for cycles

        Returns:
            Dictionary mapping cache keys to their circular dependency paths
        """
        circular_deps = {}

        # Check explicit dependencies for cycles
        for start_key in self._explicit_dependencies:
            cycle = self._find_circular_path(start_key, max_depth, set(), [])
            if cycle:
                circular_deps[start_key] = cycle

        return circular_deps

    def _find_circular_path(self, current_key: str, max_depth: int, visited: set[str], path: list[str]) -> list[str] | None:
        """
        Find circular dependency path starting from a key.

        Args:
            current_key: Current key in the path
            max_depth: Maximum depth to search
            visited: Set of already visited keys
            path: Current path being explored

        Returns:
            Circular path if found, None otherwise
        """
        if len(path) >= max_depth:
            return None

        if current_key in visited:
            # Found a cycle
            cycle_start = path.index(current_key) if current_key in path else 0
            return path[cycle_start:] + [current_key]

        visited.add(current_key)
        path.append(current_key)

        # Check explicit dependencies
        if current_key in self._explicit_dependencies:
            for dep_key in self._explicit_dependencies[current_key]:
                cycle = self._find_circular_path(dep_key, max_depth, visited.copy(), path.copy())
                if cycle:
                    return cycle

        return None

    def get_dependency_graph(self) -> dict[str, Any]:
        """
        Get a representation of the dependency graph.

        Returns:
            Dictionary representing the dependency graph
        """
        graph = {
            "explicit_dependencies": {key: list(deps) for key, deps in self._explicit_dependencies.items()},
            "dependency_rules": [
                {
                    "source_pattern": rule.source_pattern,
                    "target_pattern": rule.target_pattern,
                    "dependency_type": rule.dependency_type.value,
                    "cascade_strategy": rule.cascade_strategy.value,
                }
                for rule in self._dependency_rules
            ],
            "statistics": {
                "total_explicit_dependencies": len(self._explicit_dependencies),
                "total_dependency_rules": len(self._dependency_rules),
                "total_cascades": self._stats.total_cascades,
                "max_cascade_depth": self._stats.max_cascade_depth,
            },
        }
        return graph

    def get_cascade_stats(self) -> CascadeStats:
        """Get cascade invalidation statistics."""
        return self._stats

    def get_recent_cascades(self, count: int = 10) -> list[CascadeEvent]:
        """
        Get recent cascade events.

        Args:
            count: Number of recent events to return

        Returns:
            List of recent cascade events
        """
        return self._cascade_history[-count:] if count > 0 else []

    def clear_dependency_cache(self) -> None:
        """Clear dependency caches and circular dependency cache."""
        self._circular_dependency_cache.clear()
        self.logger.debug("Cleared dependency caches")

    def configure(
        self,
        max_cascade_depth: int | None = None,
        enable_circular_detection: bool | None = None,
        enable_lazy_invalidation: bool | None = None,
        batch_delay_seconds: float | None = None,
    ) -> None:
        """
        Configure cascade invalidation settings.

        Args:
            max_cascade_depth: Maximum cascade depth
            enable_circular_detection: Whether to detect circular dependencies
            enable_lazy_invalidation: Whether to enable lazy invalidation
            batch_delay_seconds: Delay for batch invalidation
        """
        if max_cascade_depth is not None:
            self.max_cascade_depth = max_cascade_depth
        if enable_circular_detection is not None:
            self._enable_circular_detection = enable_circular_detection
        if enable_lazy_invalidation is not None:
            self._enable_lazy_invalidation = enable_lazy_invalidation
        if batch_delay_seconds is not None:
            self._batch_delay_seconds = batch_delay_seconds

        self.logger.info(
            f"Updated cascade configuration: depth={self.max_cascade_depth}, "
            f"circular_detection={self._enable_circular_detection}, "
            f"lazy={self._enable_lazy_invalidation}, "
            f"batch_delay={self._batch_delay_seconds}"
        )


# Global cascade invalidation service instance
_cascade_invalidation_service: CascadeInvalidationService | None = None


def get_cascade_invalidation_service() -> CascadeInvalidationService:
    """
    Get the global cascade invalidation service instance.

    Returns:
        CascadeInvalidationService: The global cascade service instance
    """
    global _cascade_invalidation_service
    if _cascade_invalidation_service is None:
        _cascade_invalidation_service = CascadeInvalidationService()
    return _cascade_invalidation_service
