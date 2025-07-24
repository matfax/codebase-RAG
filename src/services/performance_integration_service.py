"""
Performance Integration Service for Enhanced Function Call Detection Pipeline.

This service integrates all performance monitoring components and provides
a unified interface for performance tracking across the entire pipeline.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from src.services.breadcrumb_cache_service import BreadcrumbCacheService
from src.services.concurrent_call_extractor_service import ConcurrentCallExtractor
from src.services.file_change_watcher_service import FileChangeWatcherService
from src.services.incremental_call_detection_service import IncrementalCallDetectionService
from src.services.performance_dashboard_service import DashboardConfig, PerformanceDashboardService
from src.services.performance_monitoring_service import PerformanceAlert, PerformanceMonitoringConfig, PerformanceMonitoringService
from src.utils.enhanced_tree_sitter_manager import EnhancedTreeSitterManager


@dataclass
class PerformanceIntegrationConfig:
    """Configuration for performance integration."""

    enable_monitoring: bool = True
    enable_dashboard: bool = True
    enable_alerts: bool = True
    enable_auto_optimization: bool = True
    performance_reporting_interval_minutes: int = 30
    auto_optimization_interval_minutes: int = 60
    performance_history_retention_hours: int = 168  # 1 week

    # Alert configurations
    critical_error_rate_threshold: float = 10.0  # 10% error rate
    critical_memory_threshold_mb: float = 4096.0  # 4GB
    critical_efficiency_threshold: float = 40.0  # 40% efficiency

    @classmethod
    def from_env(cls) -> "PerformanceIntegrationConfig":
        """Create configuration from environment variables."""
        import os

        return cls(
            enable_monitoring=os.getenv("PERF_INTEGRATION_MONITORING", "true").lower() == "true",
            enable_dashboard=os.getenv("PERF_INTEGRATION_DASHBOARD", "true").lower() == "true",
            enable_alerts=os.getenv("PERF_INTEGRATION_ALERTS", "true").lower() == "true",
            enable_auto_optimization=os.getenv("PERF_INTEGRATION_AUTO_OPT", "true").lower() == "true",
            performance_reporting_interval_minutes=int(os.getenv("PERF_INTEGRATION_REPORT_INTERVAL", "30")),
            auto_optimization_interval_minutes=int(os.getenv("PERF_INTEGRATION_OPT_INTERVAL", "60")),
            performance_history_retention_hours=int(os.getenv("PERF_INTEGRATION_HISTORY_HOURS", "168")),
            critical_error_rate_threshold=float(os.getenv("PERF_INTEGRATION_CRITICAL_ERROR_RATE", "10.0")),
            critical_memory_threshold_mb=float(os.getenv("PERF_INTEGRATION_CRITICAL_MEMORY_MB", "4096.0")),
            critical_efficiency_threshold=float(os.getenv("PERF_INTEGRATION_CRITICAL_EFFICIENCY", "40.0")),
        )


@dataclass
class PipelineHealthStatus:
    """Overall health status of the call detection pipeline."""

    overall_health: str  # 'excellent', 'good', 'warning', 'critical'
    health_score: float  # 0-100
    active_components: int
    total_components: int
    active_alerts: int
    critical_alerts: int
    performance_issues: list[str]
    recommendations: list[str]
    uptime_hours: float
    last_health_check: float = field(default_factory=time.time)


class PerformanceIntegrationService:
    """
    Performance integration service that coordinates all performance monitoring
    components for the enhanced function call detection pipeline.

    This service provides:
    - Unified performance monitoring across all pipeline components
    - Centralized alert management and notification
    - Automatic performance optimization and tuning
    - Health status monitoring and reporting
    - Integration with all caching, concurrent, and incremental services
    """

    def __init__(
        self,
        config: PerformanceIntegrationConfig | None = None,
        # Core services
        breadcrumb_cache_service: BreadcrumbCacheService | None = None,
        concurrent_extractor: ConcurrentCallExtractor | None = None,
        incremental_detection_service: IncrementalCallDetectionService | None = None,
        file_watcher_service: FileChangeWatcherService | None = None,
        tree_sitter_manager: EnhancedTreeSitterManager | None = None,
        # Alert callback
        alert_callback: callable | None = None,
    ):
        """
        Initialize the performance integration service.

        Args:
            config: Integration configuration
            breadcrumb_cache_service: Breadcrumb cache service
            concurrent_extractor: Concurrent call extractor
            incremental_detection_service: Incremental detection service
            file_watcher_service: File change watcher
            tree_sitter_manager: Enhanced Tree-sitter manager
            alert_callback: Optional callback for critical alerts
        """
        self.config = config or PerformanceIntegrationConfig.from_env()
        self.alert_callback = alert_callback
        self.logger = logging.getLogger(__name__)

        # Core services
        self.breadcrumb_cache_service = breadcrumb_cache_service
        self.concurrent_extractor = concurrent_extractor
        self.incremental_detection_service = incremental_detection_service
        self.file_watcher_service = file_watcher_service
        self.tree_sitter_manager = tree_sitter_manager

        # Performance monitoring components
        self.monitoring_service: PerformanceMonitoringService | None = None
        self.dashboard_service: PerformanceDashboardService | None = None

        # Service state
        self._is_running = False
        self._start_time = time.time()
        self._health_check_task: asyncio.Task | None = None
        self._optimization_task: asyncio.Task | None = None
        self._reporting_task: asyncio.Task | None = None

        # Component registry
        self._registered_components: dict[str, Any] = {}
        self._component_health: dict[str, bool] = {}

        # Health tracking
        self._last_health_status: PipelineHealthStatus | None = None
        self._performance_history: list[dict[str, Any]] = []

        self.logger.info(f"PerformanceIntegrationService initialized with config: {self.config}")

    async def initialize(self):
        """Initialize all performance monitoring components."""
        try:
            # Initialize monitoring service
            if self.config.enable_monitoring:
                monitoring_config = PerformanceMonitoringConfig.from_env()
                self.monitoring_service = PerformanceMonitoringService(
                    config=monitoring_config, alert_callback=self._handle_performance_alert
                )

            # Initialize dashboard service
            if self.config.enable_dashboard and self.monitoring_service:
                dashboard_config = DashboardConfig.from_env()
                self.dashboard_service = PerformanceDashboardService(monitoring_service=self.monitoring_service, config=dashboard_config)

            # Register all components
            await self._register_components()

            self.logger.info("Performance integration service initialized successfully")

        except Exception as e:
            self.logger.error(f"Error initializing performance integration service: {e}")
            raise

    async def start(self):
        """Start all performance monitoring services."""
        if self._is_running:
            self.logger.warning("Performance integration service already running")
            return

        try:
            self._is_running = True
            self._start_time = time.time()

            # Start monitoring service
            if self.monitoring_service:
                await self.monitoring_service.start_monitoring()

            # Start dashboard service
            if self.dashboard_service:
                await self.dashboard_service.start_dashboard()

            # Start health monitoring tasks
            self._health_check_task = asyncio.create_task(self._health_check_loop())

            if self.config.enable_auto_optimization:
                self._optimization_task = asyncio.create_task(self._auto_optimization_loop())

            self._reporting_task = asyncio.create_task(self._performance_reporting_loop())

            self.logger.info("Performance integration service started")

        except Exception as e:
            self.logger.error(f"Error starting performance integration service: {e}")
            self._is_running = False
            raise

    async def stop(self):
        """Stop all performance monitoring services."""
        self._is_running = False

        # Stop tasks
        for task in [self._health_check_task, self._optimization_task, self._reporting_task]:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop services
        if self.dashboard_service:
            await self.dashboard_service.stop_dashboard()

        if self.monitoring_service:
            await self.monitoring_service.stop_monitoring()

        self.logger.info("Performance integration service stopped")

    async def _register_components(self):
        """Register all pipeline components for monitoring."""
        components = {
            "breadcrumb_cache": self.breadcrumb_cache_service,
            "concurrent_extractor": self.concurrent_extractor,
            "incremental_detection": self.incremental_detection_service,
            "file_watcher": self.file_watcher_service,
            "tree_sitter_manager": self.tree_sitter_manager,
        }

        for name, service in components.items():
            if service:
                self._registered_components[name] = service
                self._component_health[name] = True
                self.logger.info(f"Registered component: {name}")

    def track_operation(
        self,
        component: str,
        operation_name: str,
        duration_ms: float,
        success: bool = True,
        items_processed: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0,
        metadata: dict[str, Any] | None = None,
    ):
        """
        Track a performance operation across the pipeline.

        Args:
            component: Component name
            operation_name: Operation identifier
            duration_ms: Operation duration in milliseconds
            success: Whether operation was successful
            items_processed: Number of items processed
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            metadata: Optional metadata
        """
        if not self.monitoring_service:
            return

        # Generate unique operation ID
        operation_id = f"{operation_name}_{time.time()}"

        # Start and immediately complete operation
        self.monitoring_service.start_operation(component, operation_id, metadata)
        self.monitoring_service.complete_operation(
            component=component,
            operation_id=operation_id,
            success=success,
            items_processed=items_processed,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            metadata=metadata,
        )

    def record_function_calls_detected(self, component: str, calls_count: int):
        """Record function calls detected by a component."""
        if self.monitoring_service:
            self.monitoring_service.record_calls_detected(component, calls_count)

    def record_cache_operation(self, component: str, operation_type: str, hit: bool, duration_ms: float = 0):
        """Record a cache operation."""
        if self.monitoring_service:
            self.monitoring_service.record_cache_operation(component, operation_type, hit, duration_ms)

    async def _health_check_loop(self):
        """Main health check loop."""
        self.logger.info("Health check loop started")

        while self._is_running:
            try:
                # Perform health check
                health_status = await self._perform_health_check()
                self._last_health_status = health_status

                # Handle critical issues
                if health_status.overall_health == "critical":
                    await self._handle_critical_health_issues(health_status)

                # Store health history
                self._performance_history.append(
                    {
                        "timestamp": time.time(),
                        "health_score": health_status.health_score,
                        "overall_health": health_status.overall_health,
                        "active_alerts": health_status.active_alerts,
                        "performance_issues": health_status.performance_issues,
                    }
                )

                # Limit history size
                if len(self._performance_history) > 1000:
                    self._performance_history = self._performance_history[-1000:]

                # Wait for next check
                await asyncio.sleep(60)  # Check every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)

        self.logger.info("Health check loop stopped")

    async def _perform_health_check(self) -> PipelineHealthStatus:
        """Perform comprehensive health check of the pipeline."""
        try:
            health_scores = []
            performance_issues = []
            recommendations = []
            active_alerts = 0
            critical_alerts = 0

            # Check monitoring service health
            if self.monitoring_service:
                monitoring_report = self.monitoring_service.get_performance_report()
                current_snapshot = monitoring_report["current_snapshot"]

                # Efficiency score
                efficiency = current_snapshot.get("efficiency_score", 0)
                health_scores.append(efficiency)

                if efficiency < self.config.critical_efficiency_threshold:
                    performance_issues.append(f"Low pipeline efficiency: {efficiency:.1f}%")
                    recommendations.append("Consider optimizing slow components or increasing cache sizes")

                # Alert counts
                active_alerts = current_snapshot.get("active_alerts", 0)

                # Component-specific checks
                for component, stats in monitoring_report["component_performance"].items():
                    component_score = 100

                    # Error rate check
                    error_rate = stats.get("error_rate_percent", 0)
                    if error_rate > self.config.critical_error_rate_threshold:
                        performance_issues.append(f"High error rate in {component}: {error_rate:.1f}%")
                        critical_alerts += 1
                        component_score -= 30

                    # Cache hit rate check
                    cache_hit_rate = stats.get("cache_hit_rate_percent", 100)
                    if cache_hit_rate < 50:
                        performance_issues.append(f"Low cache hit rate in {component}: {cache_hit_rate:.1f}%")
                        component_score -= 20

                    health_scores.append(component_score)
                    self._component_health[component] = component_score > 70

                # System resource checks
                system_metrics = monitoring_report.get("system_metrics", {})
                memory_mb = system_metrics.get("memory_usage_mb", 0)

                if memory_mb > self.config.critical_memory_threshold_mb:
                    performance_issues.append(f"High memory usage: {memory_mb:.1f} MB")
                    critical_alerts += 1
                    recommendations.append("Consider reducing cache sizes or optimizing memory usage")

            # Check individual component health
            active_components = 0
            total_components = len(self._registered_components)

            for name, component in self._registered_components.items():
                try:
                    # Basic health check - component exists and is responsive
                    if hasattr(component, "get_statistics"):
                        stats = component.get_statistics()
                        if stats:
                            active_components += 1
                        else:
                            performance_issues.append(f"Component {name} not responding")
                    else:
                        active_components += 1  # Assume healthy if no stats method
                except Exception as e:
                    performance_issues.append(f"Component {name} error: {str(e)}")
                    self._component_health[name] = False

            # Calculate overall health score
            overall_score = sum(health_scores) / len(health_scores) if health_scores else 50

            # Determine overall health status
            if critical_alerts > 0 or overall_score < 40:
                overall_health = "critical"
            elif active_alerts > 5 or overall_score < 60:
                overall_health = "warning"
            elif overall_score < 80:
                overall_health = "good"
            else:
                overall_health = "excellent"

            return PipelineHealthStatus(
                overall_health=overall_health,
                health_score=overall_score,
                active_components=active_components,
                total_components=total_components,
                active_alerts=active_alerts,
                critical_alerts=critical_alerts,
                performance_issues=performance_issues,
                recommendations=recommendations,
                uptime_hours=(time.time() - self._start_time) / 3600,
            )

        except Exception as e:
            self.logger.error(f"Error performing health check: {e}")
            return PipelineHealthStatus(
                overall_health="critical",
                health_score=0,
                active_components=0,
                total_components=len(self._registered_components),
                active_alerts=1,
                critical_alerts=1,
                performance_issues=[f"Health check failed: {str(e)}"],
                recommendations=["Investigate health check failure"],
                uptime_hours=(time.time() - self._start_time) / 3600,
            )

    async def _handle_critical_health_issues(self, health_status: PipelineHealthStatus):
        """Handle critical health issues."""
        self.logger.error(f"Critical health issues detected: {health_status.performance_issues}")

        # Call alert callback if provided
        if self.alert_callback:
            try:
                critical_alert = PerformanceAlert(
                    alert_id=f"critical_health_{time.time()}",
                    alert_type="critical",
                    message=f"Critical pipeline health issues: {', '.join(health_status.performance_issues[:3])}",
                    metric_name="health_score",
                    current_value=health_status.health_score,
                    threshold_value=self.config.critical_efficiency_threshold,
                    timestamp=time.time(),
                    component="pipeline",
                )
                self.alert_callback(critical_alert)
            except Exception as e:
                self.logger.error(f"Error calling alert callback: {e}")

        # Attempt automatic recovery actions
        if self.config.enable_auto_optimization:
            await self._attempt_auto_recovery(health_status)

    async def _attempt_auto_recovery(self, health_status: PipelineHealthStatus):
        """Attempt automatic recovery actions for critical issues."""
        self.logger.info("Attempting automatic recovery actions")

        try:
            # Clear caches if memory usage is high
            if any("memory" in issue.lower() for issue in health_status.performance_issues):
                if self.breadcrumb_cache_service:
                    self.breadcrumb_cache_service.clear_expired_entries()
                    self.logger.info("Cleared expired cache entries")

            # Restart components with errors
            for component_name, is_healthy in self._component_health.items():
                if not is_healthy:
                    component = self._registered_components.get(component_name)
                    if component and hasattr(component, "restart"):
                        try:
                            await component.restart()
                            self.logger.info(f"Restarted component: {component_name}")
                        except Exception as e:
                            self.logger.error(f"Failed to restart component {component_name}: {e}")

        except Exception as e:
            self.logger.error(f"Error during automatic recovery: {e}")

    async def _auto_optimization_loop(self):
        """Automatic optimization loop."""
        self.logger.info("Auto-optimization loop started")

        while self._is_running:
            try:
                await asyncio.sleep(self.config.auto_optimization_interval_minutes * 60)

                if self.monitoring_service:
                    # Get performance report
                    report = self.monitoring_service.get_performance_report()

                    # Apply optimizations based on trends
                    await self._apply_auto_optimizations(report)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in auto-optimization loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

        self.logger.info("Auto-optimization loop stopped")

    async def _apply_auto_optimizations(self, performance_report: dict[str, Any]):
        """Apply automatic optimizations based on performance data."""
        try:
            recommendations = performance_report.get("optimization_recommendations", [])

            for recommendation in recommendations:
                rec_type = recommendation.get("type")
                component = recommendation.get("component")
                priority = recommendation.get("priority", "low")

                if priority == "high":
                    # Apply high-priority optimizations
                    if rec_type == "low_cache_hit_rate" and component == "breadcrumb_cache":
                        if self.breadcrumb_cache_service:
                            # Increase cache size
                            current_config = self.breadcrumb_cache_service.config
                            new_max_size = min(current_config.max_cache_size * 1.5, 10000)
                            current_config.max_cache_size = int(new_max_size)
                            self.logger.info(f"Increased breadcrumb cache size to {new_max_size}")

                    elif rec_type == "performance_degradation":
                        # General performance optimizations
                        if self.tree_sitter_manager and hasattr(self.tree_sitter_manager, "optimize_for_performance"):
                            await self.tree_sitter_manager.optimize_for_performance()
                            self.logger.info("Applied Tree-sitter performance optimizations")

        except Exception as e:
            self.logger.error(f"Error applying auto-optimizations: {e}")

    async def _performance_reporting_loop(self):
        """Performance reporting loop."""
        self.logger.info("Performance reporting loop started")

        while self._is_running:
            try:
                await asyncio.sleep(self.config.performance_reporting_interval_minutes * 60)

                # Generate and log performance summary
                summary = self.get_performance_summary()
                self.logger.info(f"Performance Summary: {summary}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in performance reporting loop: {e}")
                await asyncio.sleep(300)

        self.logger.info("Performance reporting loop stopped")

    def _handle_performance_alert(self, alert: PerformanceAlert):
        """Handle performance alerts from monitoring service."""
        if alert.alert_type == "critical" and self.alert_callback:
            self.alert_callback(alert)

    def get_performance_summary(self) -> dict[str, Any]:
        """Get a summary of current performance status."""
        summary = {
            "service_status": "running" if self._is_running else "stopped",
            "uptime_hours": (time.time() - self._start_time) / 3600,
            "registered_components": len(self._registered_components),
            "healthy_components": sum(1 for is_healthy in self._component_health.values() if is_healthy),
        }

        if self._last_health_status:
            summary.update(
                {
                    "overall_health": self._last_health_status.overall_health,
                    "health_score": self._last_health_status.health_score,
                    "active_alerts": self._last_health_status.active_alerts,
                    "critical_alerts": self._last_health_status.critical_alerts,
                    "performance_issues_count": len(self._last_health_status.performance_issues),
                }
            )

        if self.monitoring_service:
            monitoring_report = self.monitoring_service.get_performance_report()
            current_snapshot = monitoring_report["current_snapshot"]
            summary.update(
                {
                    "efficiency_score": current_snapshot.get("efficiency_score", 0),
                    "total_files_processed": current_snapshot.get("total_files_processed", 0),
                    "total_calls_detected": current_snapshot.get("total_calls_detected", 0),
                }
            )

        return summary

    def get_health_status(self) -> PipelineHealthStatus | None:
        """Get current health status."""
        return self._last_health_status

    def get_component_health(self) -> dict[str, bool]:
        """Get health status of all registered components."""
        return self._component_health.copy()

    def get_dashboard_data(self) -> dict[str, Any] | None:
        """Get dashboard data if dashboard service is available."""
        if self.dashboard_service:
            return self.dashboard_service.get_dashboard_data()
        return None

    async def shutdown(self):
        """Shutdown the performance integration service."""
        self.logger.info("Shutting down PerformanceIntegrationService")
        await self.stop()

        # Shutdown individual services
        if self.dashboard_service:
            await self.dashboard_service.shutdown()

        if self.monitoring_service:
            await self.monitoring_service.shutdown()

        self.logger.info("PerformanceIntegrationService shutdown complete")
