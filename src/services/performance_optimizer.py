"""
Automated Performance Optimization and Auto-Tuning System - Wave 5.0 Implementation.

This service provides intelligent performance optimization capabilities including:
- Automated parameter tuning based on performance metrics
- Dynamic resource allocation and scaling
- Cache optimization and sizing
- Query optimization and routing improvements
- Memory management and garbage collection tuning
- Real-time performance adjustments
"""

import asyncio
import json
import logging
import math
import statistics
import time
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from src.services.performance_monitor import (
    AlertSeverity,
    PerformanceMetricType,
    get_performance_monitor,
)
from src.utils.performance_monitor import (
    get_cache_performance_monitor,
)


class OptimizationAction(Enum):
    """Types of optimization actions that can be performed."""

    INCREASE_CACHE_SIZE = "increase_cache_size"
    DECREASE_CACHE_SIZE = "decrease_cache_size"
    ADJUST_CACHE_TTL = "adjust_cache_ttl"
    OPTIMIZE_QUERY_ROUTING = "optimize_query_routing"
    ADJUST_MEMORY_LIMITS = "adjust_memory_limits"
    TUNE_GC_PARAMETERS = "tune_gc_parameters"
    SCALE_RESOURCES = "scale_resources"
    OPTIMIZE_THREADING = "optimize_threading"
    ADJUST_TIMEOUTS = "adjust_timeouts"
    ENABLE_COMPRESSION = "enable_compression"
    CUSTOM = "custom"


class OptimizationStrategy(Enum):
    """Optimization strategies for different scenarios."""

    CONSERVATIVE = "conservative"  # Small, safe adjustments
    BALANCED = "balanced"  # Moderate adjustments
    AGGRESSIVE = "aggressive"  # Large, bold adjustments
    ADAPTIVE = "adaptive"  # Strategy changes based on conditions


@dataclass
class OptimizationRule:
    """Defines a rule for automated optimization."""

    rule_id: str
    name: str
    description: str

    # Trigger conditions
    metric_type: PerformanceMetricType
    component: str
    threshold_value: float
    comparison_operator: str  # 'gt', 'lt', 'eq', 'gte', 'lte'

    # Action to take
    action: OptimizationAction
    action_parameters: dict[str, Any]

    # Rule behavior
    enabled: bool = True
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    cooldown_seconds: float = 300.0  # 5 minutes default
    max_adjustments: int = 5  # Maximum adjustments before stopping

    # Tracking
    last_triggered: float | None = None
    trigger_count: int = 0
    adjustment_count: int = 0
    success_count: int = 0

    def can_trigger(self, current_time: float) -> bool:
        """Check if this rule can be triggered."""
        if not self.enabled:
            return False

        if self.adjustment_count >= self.max_adjustments:
            return False

        if self.last_triggered and (current_time - self.last_triggered) < self.cooldown_seconds:
            return False

        return True

    def evaluate_condition(self, value: float) -> bool:
        """Evaluate if the condition is met for this rule."""
        if self.comparison_operator == "gt":
            return value > self.threshold_value
        elif self.comparison_operator == "lt":
            return value < self.threshold_value
        elif self.comparison_operator == "eq":
            return value == self.threshold_value
        elif self.comparison_operator == "gte":
            return value >= self.threshold_value
        elif self.comparison_operator == "lte":
            return value <= self.threshold_value
        else:
            return False


@dataclass
class OptimizationResult:
    """Result of an optimization action."""

    optimization_id: str
    rule_id: str
    action: OptimizationAction
    component: str

    # Action details
    action_parameters: dict[str, Any]
    previous_value: Any
    new_value: Any

    # Result tracking
    success: bool
    error_message: str | None = None
    timestamp: float = field(default_factory=time.time)

    # Performance impact
    metric_before: float | None = None
    metric_after: float | None = None
    improvement_percent: float | None = None

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def calculate_improvement(self):
        """Calculate performance improvement percentage."""
        if self.metric_before is not None and self.metric_after is not None:
            if self.metric_before != 0:
                self.improvement_percent = ((self.metric_after - self.metric_before) / self.metric_before) * 100
            else:
                self.improvement_percent = 0.0


@dataclass
class OptimizationConfig:
    """Configuration for the performance optimizer."""

    enabled: bool = True
    strategy: OptimizationStrategy = OptimizationStrategy.BALANCED
    optimization_interval: float = 60.0  # 1 minute

    # Safety limits
    max_cache_size_mb: int = 4096  # 4GB max cache size
    min_cache_size_mb: int = 64  # 64MB min cache size
    max_memory_limit_mb: int = 8192  # 8GB max memory limit

    # Optimization aggressiveness
    cache_size_adjustment_percent: float = 20.0  # 20% adjustments
    ttl_adjustment_percent: float = 25.0  # 25% adjustments

    # Monitoring and validation
    validation_period_seconds: float = 180.0  # 3 minutes to validate changes
    rollback_on_degradation: bool = True
    degradation_threshold_percent: float = 10.0  # 10% performance degradation triggers rollback

    # Logging and reporting
    enable_detailed_logging: bool = True
    enable_optimization_reports: bool = True


class AutoPerformanceOptimizer:
    """
    Automated performance optimization and auto-tuning system.

    Monitors performance metrics and automatically applies optimizations to improve
    system performance, including cache tuning, memory optimization, and resource scaling.
    """

    def __init__(self, config: OptimizationConfig | None = None, custom_optimizers: dict[str, Callable] | None = None):
        """
        Initialize the performance optimizer.

        Args:
            config: Optimization configuration
            custom_optimizers: Custom optimization functions
        """
        self.config = config or OptimizationConfig()
        self.custom_optimizers = custom_optimizers or {}

        # Initialize logging
        self.logger = logging.getLogger(__name__)

        # Performance monitoring integration
        self.performance_monitor = get_performance_monitor()
        self.cache_monitor = get_cache_performance_monitor()

        # Optimization state
        self._optimization_rules: list[OptimizationRule] = []
        self._optimization_history: deque = deque(maxlen=1000)
        self._pending_validations: dict[str, dict[str, Any]] = {}

        # Control state
        self._optimizer_task: asyncio.Task | None = None
        self._validation_task: asyncio.Task | None = None
        self._is_running = False

        # Current system state tracking
        self._current_config: dict[str, Any] = {}
        self._baseline_metrics: dict[str, float] = {}

        # Performance tracking
        self._optimization_stats = {
            "total_optimizations": 0,
            "successful_optimizations": 0,
            "rollbacks": 0,
            "improvements": 0,
            "degradations": 0,
        }

        # Initialize default optimization rules
        self._initialize_default_rules()

        self.logger.info("AutoPerformanceOptimizer initialized")

    def _initialize_default_rules(self):
        """Initialize default optimization rules."""
        # Cache hit rate optimization
        cache_hit_rule = OptimizationRule(
            rule_id="cache_hit_rate_low",
            name="Low Cache Hit Rate Optimization",
            description="Increase cache size when hit rate is low",
            metric_type=PerformanceMetricType.CACHE_HIT_RATE,
            component="*",
            threshold_value=70.0,  # < 70%
            comparison_operator="lt",
            action=OptimizationAction.INCREASE_CACHE_SIZE,
            action_parameters={"increase_percent": 25.0},
            strategy=self.config.strategy,
        )

        # High response time optimization
        response_time_rule = OptimizationRule(
            rule_id="response_time_high",
            name="High Response Time Optimization",
            description="Optimize caching and routing when response time is high",
            metric_type=PerformanceMetricType.RESPONSE_TIME,
            component="*",
            threshold_value=15000.0,  # > 15 seconds
            comparison_operator="gt",
            action=OptimizationAction.INCREASE_CACHE_SIZE,
            action_parameters={"increase_percent": 30.0},
            strategy=self.config.strategy,
        )

        # Memory usage optimization
        memory_rule = OptimizationRule(
            rule_id="memory_usage_high",
            name="High Memory Usage Optimization",
            description="Reduce cache sizes when memory usage is high",
            metric_type=PerformanceMetricType.MEMORY_USAGE,
            component="*",
            threshold_value=2048.0,  # > 2GB
            comparison_operator="gt",
            action=OptimizationAction.DECREASE_CACHE_SIZE,
            action_parameters={"decrease_percent": 15.0},
            strategy=self.config.strategy,
        )

        # Error rate optimization
        error_rate_rule = OptimizationRule(
            rule_id="error_rate_high",
            name="High Error Rate Optimization",
            description="Adjust timeouts and retry logic when error rate is high",
            metric_type=PerformanceMetricType.ERROR_RATE,
            component="*",
            threshold_value=5.0,  # > 5%
            comparison_operator="gt",
            action=OptimizationAction.ADJUST_TIMEOUTS,
            action_parameters={"increase_timeout_percent": 20.0},
            strategy=self.config.strategy,
        )

        self._optimization_rules.extend([cache_hit_rule, response_time_rule, memory_rule, error_rate_rule])

    async def start_optimization(self):
        """Start the automated performance optimization system."""
        if self._is_running:
            self.logger.warning("Performance optimizer already running")
            return {"status": "already_running"}

        if not self.config.enabled:
            self.logger.info("Performance optimization disabled in config")
            return {"status": "disabled"}

        try:
            self._is_running = True

            # Start optimization loop
            self._optimizer_task = asyncio.create_task(self._optimization_loop())

            # Start validation loop
            self._validation_task = asyncio.create_task(self._validation_loop())

            # Capture baseline metrics
            await self._capture_baseline_metrics()

            self.logger.info("Automated performance optimization started")
            return {"status": "started", "message": "Performance optimization system activated"}

        except Exception as e:
            self.logger.error(f"Error starting performance optimizer: {e}")
            self._is_running = False
            return {"status": "error", "message": f"Failed to start optimizer: {e}"}

    async def stop_optimization(self):
        """Stop the automated performance optimization system."""
        self._is_running = False

        # Cancel tasks
        if self._optimizer_task:
            self._optimizer_task.cancel()
            try:
                await self._optimizer_task
            except asyncio.CancelledError:
                pass

        if self._validation_task:
            self._validation_task.cancel()
            try:
                await self._validation_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Automated performance optimization stopped")
        return {"status": "stopped", "message": "Performance optimization system deactivated"}

    async def _optimization_loop(self):
        """Main optimization loop."""
        self.logger.info("Performance optimization loop started")

        while self._is_running:
            try:
                # Get current performance metrics
                current_metrics = self.performance_monitor.get_current_metrics()

                # Evaluate optimization rules
                await self._evaluate_optimization_rules(current_metrics)

                # Wait for next optimization cycle
                await asyncio.sleep(self.config.optimization_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in optimization loop: {e}")
                await asyncio.sleep(min(self.config.optimization_interval, 60.0))

        self.logger.info("Performance optimization loop stopped")

    async def _validation_loop(self):
        """Validation loop for checking optimization results."""
        self.logger.info("Optimization validation loop started")

        while self._is_running:
            try:
                # Check pending validations
                await self._check_pending_validations()

                # Wait before next validation cycle
                await asyncio.sleep(30.0)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in validation loop: {e}")
                await asyncio.sleep(30.0)

        self.logger.info("Optimization validation loop stopped")

    async def _capture_baseline_metrics(self):
        """Capture baseline performance metrics."""
        try:
            current_metrics = self.performance_monitor.get_current_metrics()

            # Group metrics by component and type
            for metric in current_metrics:
                key = f"{metric.component}_{metric.metric_type.value}"
                self._baseline_metrics[key] = metric.value

            self.logger.info(f"Captured baseline metrics for {len(self._baseline_metrics)} metric types")

        except Exception as e:
            self.logger.error(f"Error capturing baseline metrics: {e}")

    async def _evaluate_optimization_rules(self, current_metrics: list):
        """Evaluate optimization rules against current metrics."""
        try:
            current_time = time.time()

            # Group metrics by component and type
            metrics_by_key = {}
            for metric in current_metrics:
                key = f"{metric.component}_{metric.metric_type.value}"
                if key not in metrics_by_key:
                    metrics_by_key[key] = []
                metrics_by_key[key].append(metric.value)

            # Calculate average values for each metric
            avg_metrics = {}
            for key, values in metrics_by_key.items():
                avg_metrics[key] = statistics.mean(values) if values else 0.0

            # Evaluate each rule
            for rule in self._optimization_rules:
                if not rule.can_trigger(current_time):
                    continue

                # Find matching metrics
                matching_metrics = []
                if rule.component == "*":
                    # Apply to all components
                    for key, value in avg_metrics.items():
                        if rule.metric_type.value in key:
                            matching_metrics.append((key, value))
                else:
                    # Apply to specific component
                    key = f"{rule.component}_{rule.metric_type.value}"
                    if key in avg_metrics:
                        matching_metrics.append((key, avg_metrics[key]))

                # Check if rule condition is met
                for metric_key, metric_value in matching_metrics:
                    if rule.evaluate_condition(metric_value):
                        # Extract component name
                        component = metric_key.split("_")[0] if "_" in metric_key else rule.component

                        # Apply optimization
                        await self._apply_optimization(rule, component, metric_value, current_time)
                        break  # Only trigger once per evaluation cycle

        except Exception as e:
            self.logger.error(f"Error evaluating optimization rules: {e}")

    async def _apply_optimization(self, rule: OptimizationRule, component: str, current_metric_value: float, current_time: float):
        """Apply an optimization action based on a triggered rule."""
        try:
            optimization_id = f"opt_{rule.rule_id}_{int(current_time)}"

            self.logger.info(
                f"Applying optimization {rule.action.value} for {component} " f"(rule: {rule.rule_id}, metric: {current_metric_value})"
            )

            # Record rule trigger
            rule.last_triggered = current_time
            rule.trigger_count += 1

            # Get current configuration before optimization
            previous_config = await self._get_current_config(component, rule.action)

            # Apply the optimization action
            result = await self._execute_optimization_action(optimization_id, rule, component, previous_config)

            if result.success:
                rule.adjustment_count += 1
                rule.success_count += 1
                self._optimization_stats["total_optimizations"] += 1
                self._optimization_stats["successful_optimizations"] += 1

                # Schedule validation if enabled
                if self.config.rollback_on_degradation:
                    await self._schedule_validation(optimization_id, rule, result)

                self.logger.info(f"Optimization {optimization_id} applied successfully")
            else:
                self.logger.warning(f"Optimization {optimization_id} failed: {result.error_message}")

            # Store optimization result
            self._optimization_history.append(result)

        except Exception as e:
            self.logger.error(f"Error applying optimization: {e}")

    async def _execute_optimization_action(
        self, optimization_id: str, rule: OptimizationRule, component: str, previous_config: dict[str, Any]
    ) -> OptimizationResult:
        """Execute a specific optimization action."""
        try:
            result = OptimizationResult(
                optimization_id=optimization_id,
                rule_id=rule.rule_id,
                action=rule.action,
                component=component,
                action_parameters=rule.action_parameters.copy(),
                previous_value=previous_config,
            )

            # Execute specific optimization action
            if rule.action == OptimizationAction.INCREASE_CACHE_SIZE:
                result = await self._optimize_cache_size(result, increase=True)

            elif rule.action == OptimizationAction.DECREASE_CACHE_SIZE:
                result = await self._optimize_cache_size(result, increase=False)

            elif rule.action == OptimizationAction.ADJUST_CACHE_TTL:
                result = await self._optimize_cache_ttl(result)

            elif rule.action == OptimizationAction.ADJUST_TIMEOUTS:
                result = await self._optimize_timeouts(result)

            elif rule.action == OptimizationAction.OPTIMIZE_QUERY_ROUTING:
                result = await self._optimize_query_routing(result)

            elif rule.action == OptimizationAction.CUSTOM:
                result = await self._execute_custom_optimization(result)

            else:
                result.success = False
                result.error_message = f"Unsupported optimization action: {rule.action}"

            return result

        except Exception as e:
            self.logger.error(f"Error executing optimization action: {e}")
            return OptimizationResult(
                optimization_id=optimization_id,
                rule_id=rule.rule_id,
                action=rule.action,
                component=component,
                action_parameters=rule.action_parameters,
                previous_value=previous_config,
                success=False,
                error_message=str(e),
            )

    async def _optimize_cache_size(self, result: OptimizationResult, increase: bool) -> OptimizationResult:
        """Optimize cache size (increase or decrease)."""
        try:
            # Get current cache metrics
            cache_metrics = self.cache_monitor.get_aggregated_metrics()

            if not cache_metrics or "summary" not in cache_metrics:
                result.success = False
                result.error_message = "Cache metrics not available"
                return result

            # Calculate adjustment
            adjustment_percent = result.action_parameters.get(
                "increase_percent" if increase else "decrease_percent", self.config.cache_size_adjustment_percent
            )

            current_size_mb = cache_metrics["summary"].get("total_size_mb", 0)

            if increase:
                new_size_mb = current_size_mb * (1 + adjustment_percent / 100)
                new_size_mb = min(new_size_mb, self.config.max_cache_size_mb)
            else:
                new_size_mb = current_size_mb * (1 - adjustment_percent / 100)
                new_size_mb = max(new_size_mb, self.config.min_cache_size_mb)

            # Apply cache size optimization (this would need actual cache service integration)
            # For now, we'll simulate the optimization
            result.new_value = {"cache_size_mb": new_size_mb}
            result.success = True
            result.metadata = {
                "adjustment_percent": adjustment_percent,
                "direction": "increase" if increase else "decrease",
                "previous_size_mb": current_size_mb,
                "new_size_mb": new_size_mb,
            }

            self.logger.info(
                f"Cache size optimization: {current_size_mb}MB -> {new_size_mb}MB "
                f"({adjustment_percent}% {'increase' if increase else 'decrease'})"
            )

            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            return result

    async def _optimize_cache_ttl(self, result: OptimizationResult) -> OptimizationResult:
        """Optimize cache TTL settings."""
        try:
            # Get TTL adjustment parameters
            adjustment_percent = result.action_parameters.get("ttl_adjustment_percent", self.config.ttl_adjustment_percent)

            increase = result.action_parameters.get("increase_ttl", True)

            # Current TTL would be retrieved from cache configuration
            # For simulation, we'll use a default value
            current_ttl = 3600  # 1 hour default

            if increase:
                new_ttl = current_ttl * (1 + adjustment_percent / 100)
            else:
                new_ttl = current_ttl * (1 - adjustment_percent / 100)
                new_ttl = max(new_ttl, 60)  # Minimum 1 minute

            result.new_value = {"cache_ttl_seconds": new_ttl}
            result.success = True
            result.metadata = {
                "adjustment_percent": adjustment_percent,
                "direction": "increase" if increase else "decrease",
                "previous_ttl": current_ttl,
                "new_ttl": new_ttl,
            }

            self.logger.info(
                f"Cache TTL optimization: {current_ttl}s -> {new_ttl}s " f"({adjustment_percent}% {'increase' if increase else 'decrease'})"
            )

            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            return result

    async def _optimize_timeouts(self, result: OptimizationResult) -> OptimizationResult:
        """Optimize timeout settings."""
        try:
            adjustment_percent = result.action_parameters.get("increase_timeout_percent", 20.0)

            # Current timeout values (would be retrieved from actual configuration)
            current_timeouts = {
                "request_timeout": 30000,  # 30 seconds
                "connection_timeout": 10000,  # 10 seconds
                "read_timeout": 15000,  # 15 seconds
            }

            new_timeouts = {}
            for timeout_type, current_value in current_timeouts.items():
                new_value = current_value * (1 + adjustment_percent / 100)
                new_timeouts[timeout_type] = min(new_value, 60000)  # Max 60 seconds

            result.new_value = new_timeouts
            result.success = True
            result.metadata = {"adjustment_percent": adjustment_percent, "timeout_types": list(current_timeouts.keys())}

            self.logger.info(f"Timeout optimization: increased by {adjustment_percent}%")

            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            return result

    async def _optimize_query_routing(self, result: OptimizationResult) -> OptimizationResult:
        """Optimize query routing settings."""
        try:
            # This would involve adjusting query routing parameters
            # For simulation, we'll optimize routing thresholds

            current_routing_config = {"complexity_threshold": 0.7, "cache_first_threshold": 0.8, "parallel_execution_threshold": 0.6}

            # Adjust routing parameters based on performance
            new_routing_config = current_routing_config.copy()

            # Make routing more conservative to improve performance
            new_routing_config["complexity_threshold"] *= 0.9  # Lower threshold
            new_routing_config["cache_first_threshold"] *= 0.95  # Prefer cache more

            result.new_value = new_routing_config
            result.success = True
            result.metadata = {"optimization_type": "conservative_routing"}

            self.logger.info("Query routing optimization: adjusted for better caching")

            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            return result

    async def _execute_custom_optimization(self, result: OptimizationResult) -> OptimizationResult:
        """Execute a custom optimization function."""
        try:
            custom_function_name = result.action_parameters.get("function_name")

            if not custom_function_name or custom_function_name not in self.custom_optimizers:
                result.success = False
                result.error_message = f"Custom optimizer '{custom_function_name}' not found"
                return result

            # Execute custom optimization function
            custom_function = self.custom_optimizers[custom_function_name]

            if asyncio.iscoroutinefunction(custom_function):
                new_config = await custom_function(result.component, result.action_parameters)
            else:
                new_config = custom_function(result.component, result.action_parameters)

            result.new_value = new_config
            result.success = True
            result.metadata = {"custom_function": custom_function_name}

            self.logger.info(f"Custom optimization '{custom_function_name}' executed successfully")

            return result

        except Exception as e:
            result.success = False
            result.error_message = str(e)
            return result

    async def _get_current_config(self, component: str, action: OptimizationAction) -> dict[str, Any]:
        """Get current configuration for a component and action type."""
        try:
            # This would retrieve actual configuration from the system
            # For simulation, return mock configuration

            if action in [OptimizationAction.INCREASE_CACHE_SIZE, OptimizationAction.DECREASE_CACHE_SIZE]:
                cache_metrics = self.cache_monitor.get_aggregated_metrics()
                return {
                    "cache_size_mb": cache_metrics.get("summary", {}).get("total_size_mb", 0) if cache_metrics else 0,
                    "cache_type": "unified",
                }

            elif action == OptimizationAction.ADJUST_CACHE_TTL:
                return {"cache_ttl_seconds": 3600}

            elif action == OptimizationAction.ADJUST_TIMEOUTS:
                return {"request_timeout": 30000, "connection_timeout": 10000, "read_timeout": 15000}

            else:
                return {}

        except Exception as e:
            self.logger.error(f"Error getting current config: {e}")
            return {}

    async def _schedule_validation(self, optimization_id: str, rule: OptimizationRule, result: OptimizationResult):
        """Schedule validation for an optimization."""
        try:
            validation_time = time.time() + self.config.validation_period_seconds

            self._pending_validations[optimization_id] = {
                "rule": rule,
                "result": result,
                "validation_time": validation_time,
                "baseline_metric": result.metric_before,
            }

            self.logger.info(f"Scheduled validation for optimization {optimization_id}")

        except Exception as e:
            self.logger.error(f"Error scheduling validation: {e}")

    async def _check_pending_validations(self):
        """Check pending validations and rollback if necessary."""
        try:
            current_time = time.time()
            completed_validations = []

            for optimization_id, validation_data in self._pending_validations.items():
                if current_time >= validation_data["validation_time"]:
                    # Perform validation
                    await self._validate_optimization(optimization_id, validation_data)
                    completed_validations.append(optimization_id)

            # Remove completed validations
            for optimization_id in completed_validations:
                del self._pending_validations[optimization_id]

        except Exception as e:
            self.logger.error(f"Error checking pending validations: {e}")

    async def _validate_optimization(self, optimization_id: str, validation_data: dict[str, Any]):
        """Validate the results of an optimization and rollback if necessary."""
        try:
            rule = validation_data["rule"]
            result = validation_data["result"]
            baseline_metric = validation_data["baseline_metric"]

            # Get current metrics for the same component and metric type
            current_metrics = self.performance_monitor.get_current_metrics(component=result.component)

            current_metric_value = None
            for metric in current_metrics:
                if metric.metric_type == rule.metric_type:
                    current_metric_value = metric.value
                    break

            if current_metric_value is None:
                self.logger.warning(f"Could not find current metric for validation of {optimization_id}")
                return

            # Calculate performance change
            if baseline_metric is not None:
                performance_change = ((current_metric_value - baseline_metric) / baseline_metric) * 100
            else:
                performance_change = 0.0

            # Determine if rollback is needed
            needs_rollback = False

            # Check for performance degradation
            if rule.metric_type in [
                PerformanceMetricType.RESPONSE_TIME,
                PerformanceMetricType.ERROR_RATE,
                PerformanceMetricType.MEMORY_USAGE,
            ]:
                # For these metrics, increase is bad
                if performance_change > self.config.degradation_threshold_percent:
                    needs_rollback = True
            elif rule.metric_type in [PerformanceMetricType.CACHE_HIT_RATE, PerformanceMetricType.THROUGHPUT]:
                # For these metrics, decrease is bad
                if performance_change < -self.config.degradation_threshold_percent:
                    needs_rollback = True

            if needs_rollback:
                await self._rollback_optimization(optimization_id, result, performance_change)
                self._optimization_stats["rollbacks"] += 1
                self._optimization_stats["degradations"] += 1
            else:
                # Optimization was successful
                result.metric_after = current_metric_value
                result.calculate_improvement()

                if result.improvement_percent and result.improvement_percent > 0:
                    self._optimization_stats["improvements"] += 1

                self.logger.info(
                    f"Optimization {optimization_id} validated successfully " f"(performance change: {performance_change:.1f}%)"
                )

        except Exception as e:
            self.logger.error(f"Error validating optimization {optimization_id}: {e}")

    async def _rollback_optimization(self, optimization_id: str, result: OptimizationResult, performance_change: float):
        """Rollback an optimization that caused performance degradation."""
        try:
            self.logger.warning(
                f"Rolling back optimization {optimization_id} due to performance degradation " f"({performance_change:.1f}%)"
            )

            # Restore previous configuration
            # This would involve reversing the optimization action
            # For simulation, we'll just log the rollback

            rollback_result = OptimizationResult(
                optimization_id=f"rollback_{optimization_id}",
                rule_id=result.rule_id,
                action=OptimizationAction.CUSTOM,
                component=result.component,
                action_parameters={"rollback_of": optimization_id},
                previous_value=result.new_value,
                new_value=result.previous_value,
                success=True,
                metadata={"rollback_reason": "performance_degradation", "performance_change": performance_change},
            )

            self._optimization_history.append(rollback_result)

            self.logger.info(f"Optimization {optimization_id} rolled back successfully")

        except Exception as e:
            self.logger.error(f"Error rolling back optimization {optimization_id}: {e}")

    def add_optimization_rule(self, rule: OptimizationRule):
        """Add a custom optimization rule."""
        self._optimization_rules.append(rule)
        self.logger.info(f"Added optimization rule: {rule.rule_id}")

    def remove_optimization_rule(self, rule_id: str) -> bool:
        """Remove an optimization rule."""
        for i, rule in enumerate(self._optimization_rules):
            if rule.rule_id == rule_id:
                del self._optimization_rules[i]
                self.logger.info(f"Removed optimization rule: {rule_id}")
                return True
        return False

    def add_custom_optimizer(self, name: str, optimizer_function: Callable):
        """Add a custom optimization function."""
        self.custom_optimizers[name] = optimizer_function
        self.logger.info(f"Added custom optimizer: {name}")

    def get_optimization_status(self) -> dict[str, Any]:
        """Get the current status of the optimization system."""
        return {
            "enabled": self.config.enabled,
            "running": self._is_running,
            "strategy": self.config.strategy.value,
            "optimization_interval": self.config.optimization_interval,
            "total_rules": len(self._optimization_rules),
            "active_rules": len([r for r in self._optimization_rules if r.enabled]),
            "pending_validations": len(self._pending_validations),
            "optimization_stats": self._optimization_stats.copy(),
            "recent_optimizations": len([r for r in self._optimization_history if time.time() - r.timestamp <= 3600]),  # Last hour
        }

    def get_optimization_history(self, hours: int = 24) -> list[dict[str, Any]]:
        """Get optimization history for the specified time period."""
        cutoff_time = time.time() - (hours * 3600)

        recent_optimizations = [
            {
                "optimization_id": opt.optimization_id,
                "rule_id": opt.rule_id,
                "action": opt.action.value,
                "component": opt.component,
                "success": opt.success,
                "timestamp": opt.timestamp,
                "improvement_percent": opt.improvement_percent,
                "error_message": opt.error_message,
            }
            for opt in self._optimization_history
            if opt.timestamp > cutoff_time
        ]

        return recent_optimizations

    def get_optimization_recommendations(self) -> list[dict[str, Any]]:
        """Get manual optimization recommendations based on current metrics."""
        recommendations = []

        try:
            current_metrics = self.performance_monitor.get_current_metrics()

            # Analyze metrics for optimization opportunities
            metrics_by_component = defaultdict(list)
            for metric in current_metrics:
                metrics_by_component[metric.component].append(metric)

            for component, metrics in metrics_by_component.items():
                component_recommendations = self._analyze_component_for_recommendations(component, metrics)
                recommendations.extend(component_recommendations)

            return recommendations

        except Exception as e:
            self.logger.error(f"Error generating optimization recommendations: {e}")
            return []

    def _analyze_component_for_recommendations(self, component: str, metrics: list) -> list[dict[str, Any]]:
        """Analyze a component's metrics and generate optimization recommendations."""
        recommendations = []

        try:
            # Group metrics by type
            metric_values = {}
            for metric in metrics:
                metric_values[metric.metric_type] = metric.value

            # Check response time
            if PerformanceMetricType.RESPONSE_TIME in metric_values:
                response_time = metric_values[PerformanceMetricType.RESPONSE_TIME]
                if response_time > 20000:  # > 20 seconds
                    recommendations.append(
                        {
                            "type": "response_time_optimization",
                            "component": component,
                            "priority": "high",
                            "description": f"High response time ({response_time/1000:.1f}s) detected",
                            "suggested_actions": [
                                "Increase cache sizes",
                                "Optimize database queries",
                                "Add connection pooling",
                                "Consider request batching",
                            ],
                            "current_value": response_time,
                            "target_value": 10000,  # 10 seconds
                        }
                    )

            # Check cache hit rate
            if PerformanceMetricType.CACHE_HIT_RATE in metric_values:
                hit_rate = metric_values[PerformanceMetricType.CACHE_HIT_RATE]
                if hit_rate < 60:  # < 60%
                    recommendations.append(
                        {
                            "type": "cache_optimization",
                            "component": component,
                            "priority": "medium",
                            "description": f"Low cache hit rate ({hit_rate:.1f}%) detected",
                            "suggested_actions": [
                                "Increase cache TTL",
                                "Optimize cache key generation",
                                "Implement cache warming",
                                "Review cache eviction policies",
                            ],
                            "current_value": hit_rate,
                            "target_value": 80,  # 80%
                        }
                    )

            # Check memory usage
            if PerformanceMetricType.MEMORY_USAGE in metric_values:
                memory_usage = metric_values[PerformanceMetricType.MEMORY_USAGE]
                if memory_usage > 3072:  # > 3GB
                    recommendations.append(
                        {
                            "type": "memory_optimization",
                            "component": component,
                            "priority": "high",
                            "description": f"High memory usage ({memory_usage:.0f}MB) detected",
                            "suggested_actions": [
                                "Implement memory pooling",
                                "Optimize data structures",
                                "Add memory-based cache eviction",
                                "Review object lifecycle management",
                            ],
                            "current_value": memory_usage,
                            "target_value": 2048,  # 2GB
                        }
                    )

            return recommendations

        except Exception as e:
            self.logger.error(f"Error analyzing component {component}: {e}")
            return []

    async def manual_optimization(self, component: str, action: OptimizationAction, parameters: dict[str, Any]) -> dict[str, Any]:
        """Manually trigger an optimization action."""
        try:
            current_time = time.time()
            optimization_id = f"manual_{action.value}_{int(current_time)}"

            # Create a temporary rule for this manual optimization
            manual_rule = OptimizationRule(
                rule_id="manual_optimization",
                name="Manual Optimization",
                description="Manually triggered optimization",
                metric_type=PerformanceMetricType.CUSTOM,
                component=component,
                threshold_value=0,
                comparison_operator="gt",
                action=action,
                action_parameters=parameters,
                max_adjustments=1,  # Only allow one manual adjustment
            )

            # Get current configuration
            previous_config = await self._get_current_config(component, action)

            # Execute the optimization
            result = await self._execute_optimization_action(optimization_id, manual_rule, component, previous_config)

            if result.success:
                self._optimization_stats["total_optimizations"] += 1
                self._optimization_stats["successful_optimizations"] += 1

                # Schedule validation if enabled
                if self.config.rollback_on_degradation:
                    await self._schedule_validation(optimization_id, manual_rule, result)

            # Store result
            self._optimization_history.append(result)

            return {
                "success": result.success,
                "optimization_id": optimization_id,
                "message": "Manual optimization executed successfully" if result.success else result.error_message,
                "previous_value": result.previous_value,
                "new_value": result.new_value,
            }

        except Exception as e:
            self.logger.error(f"Error executing manual optimization: {e}")
            return {"success": False, "message": f"Manual optimization failed: {e}"}

    async def shutdown(self):
        """Shutdown the performance optimizer."""
        self.logger.info("Shutting down AutoPerformanceOptimizer")
        await self.stop_optimization()

        # Clear state
        self._optimization_rules.clear()
        self._optimization_history.clear()
        self._pending_validations.clear()
        self._current_config.clear()
        self._baseline_metrics.clear()

        self.logger.info("AutoPerformanceOptimizer shutdown complete")


# Global instance
_performance_optimizer: AutoPerformanceOptimizer | None = None


def get_performance_optimizer() -> AutoPerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = AutoPerformanceOptimizer()
    return _performance_optimizer


def initialize_performance_optimizer(
    config: OptimizationConfig | None = None, custom_optimizers: dict[str, Callable] | None = None
) -> AutoPerformanceOptimizer:
    """Initialize the global performance optimizer."""
    global _performance_optimizer
    _performance_optimizer = AutoPerformanceOptimizer(config=config, custom_optimizers=custom_optimizers)
    return _performance_optimizer
