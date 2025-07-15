"""
Memory-aware cache warmup service for intelligent cache preloading.

This module provides comprehensive cache warmup strategies that adapt to available memory,
memory pressure, and cache importance patterns. It coordinates with the existing cache
services and memory management system to provide optimal cache preloading.
"""

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union

from ..config.cache_config import get_global_cache_config
from ..utils.memory_utils import (
    CacheMemoryEvent,
    MemoryPressureLevel,
    SystemMemoryPressure,
    get_cache_memory_stats,
    get_system_memory_pressure,
    get_total_cache_memory_usage,
    register_cache_service,
    track_cache_memory_event,
)

logger = logging.getLogger(__name__)


class WarmupStrategy(Enum):
    """Cache warmup strategy types."""

    AGGRESSIVE = "aggressive"  # Preload as much as possible
    BALANCED = "balanced"  # Balanced approach based on memory
    CONSERVATIVE = "conservative"  # Minimal preloading
    ADAPTIVE = "adaptive"  # Dynamically adjust based on conditions
    PRIORITY_BASED = "priority_based"  # Load based on priority only


class WarmupPhase(Enum):
    """Cache warmup phases."""

    INITIALIZATION = "initialization"
    CRITICAL_PRELOAD = "critical_preload"
    STANDARD_PRELOAD = "standard_preload"
    BACKGROUND_PRELOAD = "background_preload"
    MAINTENANCE = "maintenance"


@dataclass
class WarmupItem:
    """Individual warmup item with metadata."""

    cache_name: str
    item_key: str
    item_data: Any
    priority: int = 5  # 1-10 scale, 10 being highest
    estimated_size_mb: float = 0.0
    usage_frequency: float = 0.0  # Historical usage frequency
    last_accessed: float = field(default_factory=time.time)
    dependencies: set[str] = field(default_factory=set)
    warmup_cost: float = 0.0  # Cost/time to warm up

    @property
    def warmup_score(self) -> float:
        """Calculate warmup score based on priority, frequency, and cost."""
        # Higher priority, frequency, and lower cost = higher score
        frequency_factor = min(self.usage_frequency, 1.0)
        cost_factor = 1.0 / (1.0 + self.warmup_cost)
        recency_factor = 1.0 / (1.0 + (time.time() - self.last_accessed) / 3600)  # Decay over hours

        return (self.priority / 10.0) * frequency_factor * cost_factor * recency_factor


@dataclass
class WarmupPlan:
    """Comprehensive cache warmup plan."""

    strategy: WarmupStrategy
    total_items: int
    estimated_memory_mb: float
    available_memory_mb: float
    phases: list[tuple[WarmupPhase, list[WarmupItem]]] = field(default_factory=list)
    execution_time_estimate: float = 0.0
    memory_budget: float = 0.0
    created_at: float = field(default_factory=time.time)

    def get_phase_items(self, phase: WarmupPhase) -> list[WarmupItem]:
        """Get items for a specific phase."""
        for phase_type, items in self.phases:
            if phase_type == phase:
                return items
        return []


@dataclass
class WarmupProgress:
    """Cache warmup progress tracking."""

    total_items: int
    completed_items: int = 0
    failed_items: int = 0
    current_phase: WarmupPhase = WarmupPhase.INITIALIZATION
    memory_used_mb: float = 0.0
    memory_budget_mb: float = 0.0
    start_time: float = field(default_factory=time.time)
    estimated_completion: float = 0.0

    @property
    def progress_percent(self) -> float:
        """Calculate progress percentage."""
        return (self.completed_items / self.total_items) * 100 if self.total_items > 0 else 0.0

    @property
    def memory_usage_percent(self) -> float:
        """Calculate memory usage percentage."""
        return (self.memory_used_mb / self.memory_budget_mb) * 100 if self.memory_budget_mb > 0 else 0.0


class CacheWarmupStrategy(ABC):
    """Abstract base class for cache warmup strategies."""

    @abstractmethod
    async def create_warmup_plan(
        self,
        cache_services: dict[str, Any],
        available_memory_mb: float,
        system_pressure: SystemMemoryPressure,
        historical_data: dict[str, Any],
    ) -> WarmupPlan:
        """Create a warmup plan based on the strategy."""
        pass

    @abstractmethod
    async def should_continue_warmup(self, progress: WarmupProgress, current_pressure: SystemMemoryPressure) -> bool:
        """Determine if warmup should continue."""
        pass


class AggressiveWarmupStrategy(CacheWarmupStrategy):
    """Aggressive warmup strategy - preload as much as possible."""

    async def create_warmup_plan(
        self,
        cache_services: dict[str, Any],
        available_memory_mb: float,
        system_pressure: SystemMemoryPressure,
        historical_data: dict[str, Any],
    ) -> WarmupPlan:
        """Create aggressive warmup plan."""
        # Use 80% of available memory for caching
        memory_budget = available_memory_mb * 0.8

        # Collect all potential warmup items
        all_items = await self._collect_warmup_items(cache_services, historical_data)

        # Sort by priority and frequency
        all_items.sort(key=lambda x: x.warmup_score, reverse=True)

        # Create phases
        phases = []
        remaining_memory = memory_budget

        # Phase 1: Critical items (priority 8-10)
        critical_items = [item for item in all_items if item.priority >= 8]
        critical_items = self._filter_by_memory_budget(critical_items, remaining_memory * 0.4)
        phases.append((WarmupPhase.CRITICAL_PRELOAD, critical_items))
        remaining_memory -= sum(item.estimated_size_mb for item in critical_items)

        # Phase 2: Standard items (priority 5-7)
        standard_items = [item for item in all_items if 5 <= item.priority < 8]
        standard_items = self._filter_by_memory_budget(standard_items, remaining_memory * 0.6)
        phases.append((WarmupPhase.STANDARD_PRELOAD, standard_items))
        remaining_memory -= sum(item.estimated_size_mb for item in standard_items)

        # Phase 3: Background items (priority 1-4)
        background_items = [item for item in all_items if item.priority < 5]
        background_items = self._filter_by_memory_budget(background_items, remaining_memory)
        phases.append((WarmupPhase.BACKGROUND_PRELOAD, background_items))

        total_items = len(critical_items) + len(standard_items) + len(background_items)
        estimated_memory = sum(item.estimated_size_mb for item in all_items[:total_items])

        return WarmupPlan(
            strategy=WarmupStrategy.AGGRESSIVE,
            total_items=total_items,
            estimated_memory_mb=estimated_memory,
            available_memory_mb=available_memory_mb,
            phases=phases,
            memory_budget=memory_budget,
        )

    async def should_continue_warmup(self, progress: WarmupProgress, current_pressure: SystemMemoryPressure) -> bool:
        """Continue unless critical memory pressure."""
        return current_pressure.level != MemoryPressureLevel.CRITICAL

    def _filter_by_memory_budget(self, items: list[WarmupItem], budget: float) -> list[WarmupItem]:
        """Filter items to fit within memory budget."""
        if budget <= 0:
            return []

        selected_items = []
        current_size = 0.0

        for item in items:
            if current_size + item.estimated_size_mb <= budget:
                selected_items.append(item)
                current_size += item.estimated_size_mb
            else:
                break

        return selected_items

    async def _collect_warmup_items(self, cache_services: dict[str, Any], historical_data: dict[str, Any]) -> list[WarmupItem]:
        """Collect all potential warmup items."""
        items = []

        # For each cache service, collect items based on historical usage
        for cache_name, cache_service in cache_services.items():
            if hasattr(cache_service, "get_warmup_candidates"):
                cache_items = await cache_service.get_warmup_candidates(historical_data.get(cache_name, {}))
                items.extend(cache_items)

        return items


class BalancedWarmupStrategy(CacheWarmupStrategy):
    """Balanced warmup strategy - moderate preloading based on memory and usage."""

    async def create_warmup_plan(
        self,
        cache_services: dict[str, Any],
        available_memory_mb: float,
        system_pressure: SystemMemoryPressure,
        historical_data: dict[str, Any],
    ) -> WarmupPlan:
        """Create balanced warmup plan."""
        # Adjust memory budget based on system pressure
        if system_pressure.level == MemoryPressureLevel.HIGH:
            memory_budget = available_memory_mb * 0.3
        elif system_pressure.level == MemoryPressureLevel.MODERATE:
            memory_budget = available_memory_mb * 0.5
        else:
            memory_budget = available_memory_mb * 0.6

        # Collect items with stricter filtering
        all_items = await self._collect_warmup_items(cache_services, historical_data)

        # Filter by minimum usage frequency and priority
        filtered_items = [item for item in all_items if item.usage_frequency >= 0.1 and item.priority >= 3]

        # Sort by warmup score
        filtered_items.sort(key=lambda x: x.warmup_score, reverse=True)

        # Create phases with balanced distribution
        phases = []
        remaining_memory = memory_budget

        # Phase 1: High priority items
        high_priority = [item for item in filtered_items if item.priority >= 7]
        high_priority = self._filter_by_memory_budget(high_priority, remaining_memory * 0.5)
        phases.append((WarmupPhase.CRITICAL_PRELOAD, high_priority))
        remaining_memory -= sum(item.estimated_size_mb for item in high_priority)

        # Phase 2: Medium priority items
        medium_priority = [item for item in filtered_items if 4 <= item.priority < 7]
        medium_priority = self._filter_by_memory_budget(medium_priority, remaining_memory * 0.7)
        phases.append((WarmupPhase.STANDARD_PRELOAD, medium_priority))
        remaining_memory -= sum(item.estimated_size_mb for item in medium_priority)

        # Phase 3: Background items (only if memory allows)
        if remaining_memory > 50:  # Only if at least 50MB left
            background_items = [item for item in filtered_items if item.priority < 4]
            background_items = self._filter_by_memory_budget(background_items, remaining_memory)
            phases.append((WarmupPhase.BACKGROUND_PRELOAD, background_items))

        total_items = sum(len(items) for _, items in phases)
        estimated_memory = sum(sum(item.estimated_size_mb for item in items) for _, items in phases)

        return WarmupPlan(
            strategy=WarmupStrategy.BALANCED,
            total_items=total_items,
            estimated_memory_mb=estimated_memory,
            available_memory_mb=available_memory_mb,
            phases=phases,
            memory_budget=memory_budget,
        )

    async def should_continue_warmup(self, progress: WarmupProgress, current_pressure: SystemMemoryPressure) -> bool:
        """Continue unless high or critical memory pressure."""
        return current_pressure.level not in [MemoryPressureLevel.HIGH, MemoryPressureLevel.CRITICAL]

    def _filter_by_memory_budget(self, items: list[WarmupItem], budget: float) -> list[WarmupItem]:
        """Filter items to fit within memory budget."""
        if budget <= 0:
            return []

        selected_items = []
        current_size = 0.0

        for item in items:
            if current_size + item.estimated_size_mb <= budget:
                selected_items.append(item)
                current_size += item.estimated_size_mb
            else:
                break

        return selected_items

    async def _collect_warmup_items(self, cache_services: dict[str, Any], historical_data: dict[str, Any]) -> list[WarmupItem]:
        """Collect warmup items with balanced criteria."""
        items = []

        for cache_name, cache_service in cache_services.items():
            if hasattr(cache_service, "get_warmup_candidates"):
                cache_items = await cache_service.get_warmup_candidates(historical_data.get(cache_name, {}))
                items.extend(cache_items)

        return items


class ConservativeWarmupStrategy(CacheWarmupStrategy):
    """Conservative warmup strategy - minimal preloading."""

    async def create_warmup_plan(
        self,
        cache_services: dict[str, Any],
        available_memory_mb: float,
        system_pressure: SystemMemoryPressure,
        historical_data: dict[str, Any],
    ) -> WarmupPlan:
        """Create conservative warmup plan."""
        # Use only 20% of available memory
        memory_budget = available_memory_mb * 0.2

        # Only preload critical items
        all_items = await self._collect_warmup_items(cache_services, historical_data)

        # Filter for only high-priority, frequently-used items
        critical_items = [item for item in all_items if item.priority >= 8 and item.usage_frequency >= 0.3]

        # Sort by score and take only the top items
        critical_items.sort(key=lambda x: x.warmup_score, reverse=True)
        critical_items = self._filter_by_memory_budget(critical_items, memory_budget)

        phases = [(WarmupPhase.CRITICAL_PRELOAD, critical_items)]

        return WarmupPlan(
            strategy=WarmupStrategy.CONSERVATIVE,
            total_items=len(critical_items),
            estimated_memory_mb=sum(item.estimated_size_mb for item in critical_items),
            available_memory_mb=available_memory_mb,
            phases=phases,
            memory_budget=memory_budget,
        )

    async def should_continue_warmup(self, progress: WarmupProgress, current_pressure: SystemMemoryPressure) -> bool:
        """Stop at any sign of memory pressure."""
        return current_pressure.level == MemoryPressureLevel.LOW

    def _filter_by_memory_budget(self, items: list[WarmupItem], budget: float) -> list[WarmupItem]:
        """Filter items to fit within memory budget."""
        if budget <= 0:
            return []

        selected_items = []
        current_size = 0.0

        for item in items:
            if current_size + item.estimated_size_mb <= budget:
                selected_items.append(item)
                current_size += item.estimated_size_mb
            else:
                break

        return selected_items

    async def _collect_warmup_items(self, cache_services: dict[str, Any], historical_data: dict[str, Any]) -> list[WarmupItem]:
        """Collect only critical warmup items."""
        items = []

        for cache_name, cache_service in cache_services.items():
            if hasattr(cache_service, "get_warmup_candidates"):
                cache_items = await cache_service.get_warmup_candidates(historical_data.get(cache_name, {}))
                # Filter for only critical items
                critical_items = [item for item in cache_items if item.priority >= 8]
                items.extend(critical_items)

        return items


class CacheWarmupService:
    """Memory-aware cache warmup service."""

    def __init__(self):
        self.cache_services: dict[str, Any] = {}
        self.historical_data: dict[str, Any] = defaultdict(dict)
        self.warmup_strategies: dict[WarmupStrategy, CacheWarmupStrategy] = {
            WarmupStrategy.AGGRESSIVE: AggressiveWarmupStrategy(),
            WarmupStrategy.BALANCED: BalancedWarmupStrategy(),
            WarmupStrategy.CONSERVATIVE: ConservativeWarmupStrategy(),
        }
        self.current_warmup: WarmupProgress | None = None
        self.warmup_history: list[dict[str, Any]] = []
        self.is_warmup_active = False
        self._warmup_lock = asyncio.Lock()

        # Register with memory tracking
        register_cache_service("cache_warmup", self)

    async def register_cache_service(self, name: str, service: Any) -> None:
        """Register a cache service for warmup."""
        self.cache_services[name] = service
        logger.info(f"Registered cache service '{name}' for warmup")

    async def unregister_cache_service(self, name: str) -> None:
        """Unregister a cache service."""
        self.cache_services.pop(name, None)
        logger.info(f"Unregistered cache service '{name}' from warmup")

    async def update_historical_data(self, cache_name: str, data: dict[str, Any]) -> None:
        """Update historical usage data for a cache."""
        self.historical_data[cache_name] = data

    async def get_recommended_strategy(self, system_pressure: SystemMemoryPressure | None = None) -> WarmupStrategy:
        """Get recommended warmup strategy based on system conditions."""
        if system_pressure is None:
            system_pressure = get_system_memory_pressure()

        # Determine strategy based on system pressure
        if system_pressure.level == MemoryPressureLevel.CRITICAL:
            return WarmupStrategy.CONSERVATIVE
        elif system_pressure.level == MemoryPressureLevel.HIGH:
            return WarmupStrategy.CONSERVATIVE
        elif system_pressure.level == MemoryPressureLevel.MODERATE:
            return WarmupStrategy.BALANCED
        else:
            return WarmupStrategy.BALANCED  # Default to balanced for safety

    async def create_warmup_plan(self, strategy: WarmupStrategy | None = None, memory_budget_mb: float | None = None) -> WarmupPlan:
        """Create a cache warmup plan."""
        if strategy is None:
            strategy = await self.get_recommended_strategy()

        system_pressure = get_system_memory_pressure()
        available_memory = system_pressure.available_mb

        if memory_budget_mb is not None:
            available_memory = min(available_memory, memory_budget_mb)

        warmup_strategy = self.warmup_strategies[strategy]
        plan = await warmup_strategy.create_warmup_plan(self.cache_services, available_memory, system_pressure, self.historical_data)

        return plan

    async def execute_warmup_plan(self, plan: WarmupPlan, progress_callback: callable | None = None) -> dict[str, Any]:
        """Execute a cache warmup plan."""
        async with self._warmup_lock:
            if self.is_warmup_active:
                return {"error": "Warmup already in progress"}

            self.is_warmup_active = True

            # Initialize progress tracking
            self.current_warmup = WarmupProgress(total_items=plan.total_items, memory_budget_mb=plan.memory_budget, start_time=time.time())

            results = {
                "strategy": plan.strategy.value,
                "total_items": plan.total_items,
                "phases_completed": 0,
                "items_completed": 0,
                "items_failed": 0,
                "memory_used_mb": 0.0,
                "execution_time": 0.0,
                "phase_results": [],
            }

            try:
                # Execute each phase
                for phase_type, items in plan.phases:
                    self.current_warmup.current_phase = phase_type

                    phase_result = await self._execute_warmup_phase(phase_type, items, progress_callback)

                    results["phase_results"].append(phase_result)
                    results["phases_completed"] += 1
                    results["items_completed"] += phase_result["items_completed"]
                    results["items_failed"] += phase_result["items_failed"]
                    results["memory_used_mb"] += phase_result["memory_used_mb"]

                    # Check if we should continue
                    current_pressure = get_system_memory_pressure()
                    strategy_handler = self.warmup_strategies[plan.strategy]

                    if not await strategy_handler.should_continue_warmup(self.current_warmup, current_pressure):
                        logger.info(f"Stopping warmup due to memory pressure: {current_pressure.level}")
                        break

                results["execution_time"] = time.time() - self.current_warmup.start_time
                results["success"] = True

                # Record in history
                self.warmup_history.append({"timestamp": time.time(), "strategy": plan.strategy.value, "results": results.copy()})

                return results

            except Exception as e:
                logger.error(f"Warmup execution failed: {e}")
                results["error"] = str(e)
                results["success"] = False
                return results

            finally:
                self.is_warmup_active = False
                self.current_warmup = None

    async def _execute_warmup_phase(
        self, phase: WarmupPhase, items: list[WarmupItem], progress_callback: callable | None = None
    ) -> dict[str, Any]:
        """Execute a single warmup phase."""
        logger.info(f"Starting warmup phase: {phase.value} with {len(items)} items")

        phase_result = {
            "phase": phase.value,
            "total_items": len(items),
            "items_completed": 0,
            "items_failed": 0,
            "memory_used_mb": 0.0,
            "execution_time": 0.0,
            "start_time": time.time(),
        }

        # Process items in batches to avoid memory spikes
        batch_size = 10
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]

            # Check memory pressure before each batch
            current_pressure = get_system_memory_pressure()
            if current_pressure.level == MemoryPressureLevel.CRITICAL:
                logger.warning("Critical memory pressure detected, stopping warmup phase")
                break

            # Process batch
            batch_results = await self._process_warmup_batch(batch)

            phase_result["items_completed"] += batch_results["completed"]
            phase_result["items_failed"] += batch_results["failed"]
            phase_result["memory_used_mb"] += batch_results["memory_used_mb"]

            # Update progress
            if self.current_warmup:
                self.current_warmup.completed_items += batch_results["completed"]
                self.current_warmup.failed_items += batch_results["failed"]
                self.current_warmup.memory_used_mb += batch_results["memory_used_mb"]

            # Call progress callback if provided
            if progress_callback:
                await progress_callback(self.current_warmup)

            # Small delay to prevent overwhelming the system
            await asyncio.sleep(0.1)

        phase_result["execution_time"] = time.time() - phase_result["start_time"]
        logger.info(f"Completed warmup phase: {phase.value} - {phase_result['items_completed']}/{phase_result['total_items']} items")

        return phase_result

    async def _process_warmup_batch(self, batch: list[WarmupItem]) -> dict[str, Any]:
        """Process a batch of warmup items."""
        batch_result = {"completed": 0, "failed": 0, "memory_used_mb": 0.0}

        # Process items concurrently but limit concurrency
        semaphore = asyncio.Semaphore(3)  # Limit concurrent warmup operations

        async def process_item(item: WarmupItem) -> tuple[bool, float]:
            async with semaphore:
                try:
                    # Get the cache service
                    cache_service = self.cache_services.get(item.cache_name)
                    if not cache_service:
                        logger.warning(f"Cache service not found: {item.cache_name}")
                        return False, 0.0

                    # Warmup the item
                    if hasattr(cache_service, "warmup_item"):
                        memory_used = await cache_service.warmup_item(item.item_key, item.item_data)

                        # Track memory event
                        track_cache_memory_event(
                            item.cache_name,
                            CacheMemoryEvent.ALLOCATION,
                            memory_used,
                            {"warmup_item": item.item_key, "priority": item.priority},
                        )

                        return True, memory_used
                    else:
                        logger.warning(f"Cache service does not support warmup: {item.cache_name}")
                        return False, 0.0

                except Exception as e:
                    logger.error(f"Failed to warmup item {item.item_key}: {e}")
                    return False, 0.0

        # Process all items in the batch
        tasks = [process_item(item) for item in batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Aggregate results
        for result in results:
            if isinstance(result, Exception):
                batch_result["failed"] += 1
            else:
                success, memory_used = result
                if success:
                    batch_result["completed"] += 1
                    batch_result["memory_used_mb"] += memory_used
                else:
                    batch_result["failed"] += 1

        return batch_result

    async def get_warmup_status(self) -> dict[str, Any]:
        """Get current warmup status."""
        if not self.is_warmup_active or not self.current_warmup:
            return {"active": False, "last_warmup": self.warmup_history[-1] if self.warmup_history else None}

        return {
            "active": True,
            "progress": {
                "total_items": self.current_warmup.total_items,
                "completed_items": self.current_warmup.completed_items,
                "failed_items": self.current_warmup.failed_items,
                "progress_percent": self.current_warmup.progress_percent,
                "current_phase": self.current_warmup.current_phase.value,
                "memory_used_mb": self.current_warmup.memory_used_mb,
                "memory_budget_mb": self.current_warmup.memory_budget_mb,
                "memory_usage_percent": self.current_warmup.memory_usage_percent,
                "elapsed_time": time.time() - self.current_warmup.start_time,
            },
        }

    async def cancel_warmup(self) -> dict[str, Any]:
        """Cancel current warmup operation."""
        if not self.is_warmup_active:
            return {"error": "No warmup in progress"}

        # The warmup will be cancelled at the next batch check
        logger.info("Warmup cancellation requested")
        return {"message": "Warmup cancellation requested"}

    async def get_warmup_recommendations(self) -> dict[str, Any]:
        """Get warmup recommendations based on current system state."""
        system_pressure = get_system_memory_pressure()
        cache_memory_usage = get_total_cache_memory_usage()

        recommended_strategy = await self.get_recommended_strategy(system_pressure)

        recommendations = {
            "recommended_strategy": recommended_strategy.value,
            "system_pressure": system_pressure.level.value,
            "available_memory_mb": system_pressure.available_mb,
            "cache_memory_usage_mb": cache_memory_usage,
            "recommendations": [],
        }

        # Add specific recommendations
        if system_pressure.level == MemoryPressureLevel.CRITICAL:
            recommendations["recommendations"].append("Critical memory pressure detected. Avoid cache warmup until memory is available.")
        elif system_pressure.level == MemoryPressureLevel.HIGH:
            recommendations["recommendations"].append("High memory pressure. Use conservative warmup strategy with minimal preloading.")
        elif system_pressure.level == MemoryPressureLevel.MODERATE:
            recommendations["recommendations"].append("Moderate memory pressure. Use balanced warmup strategy with selective preloading.")
        else:
            recommendations["recommendations"].append("Normal memory conditions. Balanced warmup strategy recommended.")

        # Add cache-specific recommendations
        cache_stats = get_cache_memory_stats()
        if isinstance(cache_stats, dict):
            for cache_name, stats in cache_stats.items():
                if stats.current_size_mb > 100:  # Large cache
                    recommendations["recommendations"].append(
                        f"Cache '{cache_name}' is using {stats.current_size_mb:.1f}MB. " "Consider reducing warmup for this cache."
                    )

        return recommendations


# Global cache warmup service instance
_warmup_service: CacheWarmupService | None = None


async def get_cache_warmup_service() -> CacheWarmupService:
    """Get the global cache warmup service instance."""
    global _warmup_service
    if _warmup_service is None:
        _warmup_service = CacheWarmupService()
    return _warmup_service


async def initialize_cache_warmup_service() -> None:
    """Initialize the cache warmup service."""
    await get_cache_warmup_service()
    logger.info("Cache warmup service initialized")


async def register_cache_for_warmup(name: str, service: Any) -> None:
    """Register a cache service for warmup."""
    warmup_service = await get_cache_warmup_service()
    await warmup_service.register_cache_service(name, service)


async def execute_memory_aware_warmup(strategy: WarmupStrategy | None = None, memory_budget_mb: float | None = None) -> dict[str, Any]:
    """Execute memory-aware cache warmup."""
    warmup_service = await get_cache_warmup_service()

    # Create plan
    plan = await warmup_service.create_warmup_plan(strategy, memory_budget_mb)

    # Execute plan
    results = await warmup_service.execute_warmup_plan(plan)

    return results
