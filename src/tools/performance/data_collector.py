"""
Performance Data Collection Service - Wave 5.0 Implementation.

Provides comprehensive performance data collection capabilities including:
- Multi-source performance data aggregation
- Historical performance data storage and retrieval
- Performance trend analysis and pattern detection
- Data export and reporting capabilities
- Real-time and batch data collection modes
- Performance data visualization support
"""

import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import psutil

from src.services.performance_monitor import get_performance_monitor
from src.utils.performance_monitor import get_cache_performance_monitor


class DataCollectionMode(Enum):
    """Data collection modes."""

    REAL_TIME = "real_time"
    BATCH = "batch"
    ON_DEMAND = "on_demand"
    SCHEDULED = "scheduled"


class DataFormat(Enum):
    """Data export formats."""

    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    METRICS = "metrics"


class AggregationFunction(Enum):
    """Data aggregation functions."""

    MEAN = "mean"
    MEDIAN = "median"
    MIN = "min"
    MAX = "max"
    SUM = "sum"
    COUNT = "count"
    PERCENTILE_95 = "p95"
    PERCENTILE_99 = "p99"
    STDDEV = "stddev"


@dataclass
class DataPoint:
    """Individual performance data point."""

    timestamp: float
    component: str
    metric_name: str
    value: float
    unit: str
    tags: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp,
            "component": self.component,
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "tags": self.tags,
            "metadata": self.metadata,
        }


@dataclass
class AggregatedMetric:
    """Aggregated performance metric over a time period."""

    start_time: float
    end_time: float
    component: str
    metric_name: str
    aggregation_function: AggregationFunction
    value: float
    sample_count: int
    unit: str
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def duration_seconds(self) -> float:
        """Get the duration of the aggregation period."""
        return self.end_time - self.start_time

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "component": self.component,
            "metric_name": self.metric_name,
            "aggregation_function": self.aggregation_function.value,
            "value": self.value,
            "sample_count": self.sample_count,
            "unit": self.unit,
            "tags": self.tags,
        }


@dataclass
class CollectionConfig:
    """Configuration for performance data collection."""

    # Collection settings
    collection_interval_seconds: float = 10.0
    batch_size: int = 1000
    max_memory_usage_mb: float = 500.0

    # Storage settings
    enable_persistent_storage: bool = True
    storage_directory: str = "performance_data"
    data_retention_days: int = 30

    # Aggregation settings
    enable_aggregation: bool = True
    aggregation_intervals: list[int] = field(default_factory=lambda: [60, 300, 3600])  # 1min, 5min, 1hour
    aggregation_functions: list[AggregationFunction] = field(
        default_factory=lambda: [
            AggregationFunction.MEAN,
            AggregationFunction.MIN,
            AggregationFunction.MAX,
            AggregationFunction.PERCENTILE_95,
        ]
    )

    # Collection filters
    component_whitelist: list[str] | None = None
    component_blacklist: list[str] | None = None
    metric_whitelist: list[str] | None = None
    metric_blacklist: list[str] | None = None

    # Export settings
    enable_auto_export: bool = False
    export_interval_hours: int = 24
    export_formats: list[DataFormat] = field(default_factory=lambda: [DataFormat.JSON])


class PerformanceDataCollector:
    """
    Comprehensive performance data collection and management system.

    Provides real-time and batch collection of performance metrics with
    storage, aggregation, and export capabilities.
    """

    def __init__(self, config: CollectionConfig | None = None):
        """
        Initialize the performance data collector.

        Args:
            config: Collection configuration settings
        """
        self.config = config or CollectionConfig()
        self.logger = logging.getLogger(__name__)

        # Performance monitoring integration
        self.performance_monitor = get_performance_monitor()
        self.cache_monitor = get_cache_performance_monitor()

        # Data storage
        self._raw_data: deque = deque(maxlen=100000)  # Raw data points
        self._aggregated_data: dict[str, dict[int, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=10000)))
        self._collection_stats = {
            "total_points_collected": 0,
            "collection_start_time": None,
            "last_collection_time": None,
            "collections_per_second": 0.0,
            "memory_usage_mb": 0.0,
        }

        # Collection control
        self._collection_task: asyncio.Task | None = None
        self._aggregation_task: asyncio.Task | None = None
        self._export_task: asyncio.Task | None = None
        self._is_collecting = False

        # Data processing queues
        self._processing_queue: asyncio.Queue = asyncio.Queue()
        self._export_queue: asyncio.Queue = asyncio.Queue()

        # Storage management
        self._storage_path = Path(self.config.storage_directory)
        self._storage_path.mkdir(parents=True, exist_ok=True)

        self.logger.info("PerformanceDataCollector initialized")

    async def start_collection(self, mode: DataCollectionMode = DataCollectionMode.REAL_TIME):
        """
        Start performance data collection.

        Args:
            mode: Collection mode (real-time, batch, etc.)
        """
        if self._is_collecting:
            return {"status": "already_running", "message": "Data collection already running"}

        try:
            self._is_collecting = True
            self._collection_stats["collection_start_time"] = time.time()

            # Start collection task based on mode
            if mode == DataCollectionMode.REAL_TIME:
                self._collection_task = asyncio.create_task(self._real_time_collection_loop())
            elif mode == DataCollectionMode.BATCH:
                self._collection_task = asyncio.create_task(self._batch_collection_loop())
            elif mode == DataCollectionMode.SCHEDULED:
                self._collection_task = asyncio.create_task(self._scheduled_collection_loop())

            # Start aggregation task
            if self.config.enable_aggregation:
                self._aggregation_task = asyncio.create_task(self._aggregation_loop())

            # Start export task
            if self.config.enable_auto_export:
                self._export_task = asyncio.create_task(self._export_loop())

            self.logger.info(f"Performance data collection started in {mode.value} mode")
            return {"status": "started", "mode": mode.value, "message": "Data collection started successfully"}

        except Exception as e:
            self.logger.error(f"Error starting data collection: {e}")
            self._is_collecting = False
            return {"status": "error", "message": f"Failed to start collection: {e}"}

    async def stop_collection(self):
        """Stop performance data collection."""
        self._is_collecting = False

        # Cancel tasks
        tasks = [self._collection_task, self._aggregation_task, self._export_task]
        for task in tasks:
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Process remaining data
        await self._process_remaining_data()

        self.logger.info("Performance data collection stopped")
        return {"status": "stopped", "message": "Data collection stopped successfully"}

    async def _real_time_collection_loop(self):
        """Real-time data collection loop."""
        self.logger.info("Real-time data collection loop started")

        while self._is_collecting:
            try:
                # Collect current metrics
                await self._collect_current_metrics()

                # Update collection stats
                self._update_collection_stats()

                # Wait for next collection interval
                await asyncio.sleep(self.config.collection_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in real-time collection loop: {e}")
                await asyncio.sleep(self.config.collection_interval_seconds)

        self.logger.info("Real-time data collection loop stopped")

    async def _batch_collection_loop(self):
        """Batch data collection loop."""
        self.logger.info("Batch data collection loop started")

        while self._is_collecting:
            try:
                # Collect batch of metrics
                batch_start = time.time()
                collected_points = []

                for _ in range(self.config.batch_size):
                    if not self._is_collecting:
                        break

                    data_points = await self._collect_current_metrics()
                    collected_points.extend(data_points)

                    await asyncio.sleep(0.1)  # Brief pause between collections

                # Process batch
                if collected_points:
                    await self._process_data_batch(collected_points)

                batch_duration = time.time() - batch_start
                self.logger.info(f"Processed batch of {len(collected_points)} data points in {batch_duration:.2f}s")

                # Wait before next batch
                await asyncio.sleep(60.0)  # 1 minute between batches

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in batch collection loop: {e}")
                await asyncio.sleep(60.0)

        self.logger.info("Batch data collection loop stopped")

    async def _scheduled_collection_loop(self):
        """Scheduled data collection loop."""
        self.logger.info("Scheduled data collection loop started")

        while self._is_collecting:
            try:
                # Collect at regular intervals (every 5 minutes)
                await self._collect_current_metrics()
                self._update_collection_stats()

                # Wait for next scheduled collection
                await asyncio.sleep(300.0)  # 5 minutes

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in scheduled collection loop: {e}")
                await asyncio.sleep(300.0)

        self.logger.info("Scheduled data collection loop stopped")

    async def _collect_current_metrics(self) -> list[DataPoint]:
        """Collect current performance metrics from all sources."""
        try:
            current_time = time.time()
            data_points = []

            # Collect from performance monitor
            performance_metrics = self.performance_monitor.get_current_metrics()
            for metric in performance_metrics:
                if self._should_collect_metric(metric.component, metric.metric_type.value):
                    data_point = DataPoint(
                        timestamp=current_time,
                        component=metric.component,
                        metric_name=metric.metric_type.value,
                        value=metric.value,
                        unit=metric.unit,
                        tags={"source": "performance_monitor"},
                        metadata=metric.metadata,
                    )
                    data_points.append(data_point)

            # Collect from cache monitor
            cache_metrics = self.cache_monitor.get_aggregated_metrics()
            if cache_metrics and "summary" in cache_metrics:
                summary = cache_metrics["summary"]
                cache_data_points = [
                    DataPoint(
                        timestamp=current_time,
                        component="cache_system",
                        metric_name="hit_rate",
                        value=summary.get("overall_hit_rate", 0.0) * 100,
                        unit="%",
                        tags={"source": "cache_monitor"},
                    ),
                    DataPoint(
                        timestamp=current_time,
                        component="cache_system",
                        metric_name="operations_total",
                        value=summary.get("total_operations", 0),
                        unit="count",
                        tags={"source": "cache_monitor"},
                    ),
                    DataPoint(
                        timestamp=current_time,
                        component="cache_system",
                        metric_name="size",
                        value=summary.get("total_size_mb", 0.0),
                        unit="MB",
                        tags={"source": "cache_monitor"},
                    ),
                    DataPoint(
                        timestamp=current_time,
                        component="cache_system",
                        metric_name="error_rate",
                        value=summary.get("overall_error_rate", 0.0) * 100,
                        unit="%",
                        tags={"source": "cache_monitor"},
                    ),
                ]

                for dp in cache_data_points:
                    if self._should_collect_metric(dp.component, dp.metric_name):
                        data_points.append(dp)

            # Collect system metrics
            system_data_points = await self._collect_system_metrics(current_time)
            data_points.extend(system_data_points)

            # Store raw data points
            for data_point in data_points:
                self._raw_data.append(data_point)
                self._collection_stats["total_points_collected"] += 1

            self._collection_stats["last_collection_time"] = current_time

            return data_points

        except Exception as e:
            self.logger.error(f"Error collecting current metrics: {e}")
            return []

    async def _collect_system_metrics(self, timestamp: float) -> list[DataPoint]:
        """Collect system-level performance metrics."""
        try:
            data_points = []

            # Memory metrics
            memory = psutil.virtual_memory()
            data_points.extend(
                [
                    DataPoint(
                        timestamp=timestamp,
                        component="system",
                        metric_name="memory_total",
                        value=memory.total / (1024**3),
                        unit="GB",
                        tags={"source": "system", "type": "memory"},
                    ),
                    DataPoint(
                        timestamp=timestamp,
                        component="system",
                        metric_name="memory_available",
                        value=memory.available / (1024**3),
                        unit="GB",
                        tags={"source": "system", "type": "memory"},
                    ),
                    DataPoint(
                        timestamp=timestamp,
                        component="system",
                        metric_name="memory_percent",
                        value=memory.percent,
                        unit="%",
                        tags={"source": "system", "type": "memory"},
                    ),
                ]
            )

            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            data_points.extend(
                [
                    DataPoint(
                        timestamp=timestamp,
                        component="system",
                        metric_name="cpu_percent",
                        value=cpu_percent,
                        unit="%",
                        tags={"source": "system", "type": "cpu"},
                    ),
                    DataPoint(
                        timestamp=timestamp,
                        component="system",
                        metric_name="cpu_count",
                        value=cpu_count,
                        unit="count",
                        tags={"source": "system", "type": "cpu"},
                    ),
                ]
            )

            # Disk metrics
            disk = psutil.disk_usage("/")
            data_points.extend(
                [
                    DataPoint(
                        timestamp=timestamp,
                        component="system",
                        metric_name="disk_total",
                        value=disk.total / (1024**3),
                        unit="GB",
                        tags={"source": "system", "type": "disk"},
                    ),
                    DataPoint(
                        timestamp=timestamp,
                        component="system",
                        metric_name="disk_used",
                        value=disk.used / (1024**3),
                        unit="GB",
                        tags={"source": "system", "type": "disk"},
                    ),
                    DataPoint(
                        timestamp=timestamp,
                        component="system",
                        metric_name="disk_percent",
                        value=(disk.used / disk.total) * 100,
                        unit="%",
                        tags={"source": "system", "type": "disk"},
                    ),
                ]
            )

            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info()
            data_points.extend(
                [
                    DataPoint(
                        timestamp=timestamp,
                        component="process",
                        metric_name="memory_rss",
                        value=process_memory.rss / (1024**2),
                        unit="MB",
                        tags={"source": "system", "type": "process"},
                    ),
                    DataPoint(
                        timestamp=timestamp,
                        component="process",
                        metric_name="memory_percent",
                        value=process.memory_percent(),
                        unit="%",
                        tags={"source": "system", "type": "process"},
                    ),
                    DataPoint(
                        timestamp=timestamp,
                        component="process",
                        metric_name="threads",
                        value=process.num_threads(),
                        unit="count",
                        tags={"source": "system", "type": "process"},
                    ),
                ]
            )

            return data_points

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return []

    def _should_collect_metric(self, component: str, metric_name: str) -> bool:
        """Check if a metric should be collected based on filters."""
        # Component whitelist/blacklist
        if self.config.component_whitelist:
            if component not in self.config.component_whitelist:
                return False

        if self.config.component_blacklist:
            if component in self.config.component_blacklist:
                return False

        # Metric whitelist/blacklist
        if self.config.metric_whitelist:
            if metric_name not in self.config.metric_whitelist:
                return False

        if self.config.metric_blacklist:
            if metric_name in self.config.metric_blacklist:
                return False

        return True

    def _update_collection_stats(self):
        """Update collection statistics."""
        try:
            current_time = time.time()

            if self._collection_stats["collection_start_time"]:
                duration = current_time - self._collection_stats["collection_start_time"]
                if duration > 0:
                    self._collection_stats["collections_per_second"] = self._collection_stats["total_points_collected"] / duration

            # Calculate memory usage
            self._collection_stats["memory_usage_mb"] = self._estimate_memory_usage()

        except Exception as e:
            self.logger.error(f"Error updating collection stats: {e}")

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage of collected data."""
        try:
            # Rough estimation based on data point count and size
            raw_data_size = len(self._raw_data) * 0.5  # Approximate 0.5KB per data point

            aggregated_data_size = 0
            for component_data in self._aggregated_data.values():
                for interval_data in component_data.values():
                    aggregated_data_size += len(interval_data) * 0.3  # Approximate 0.3KB per aggregated point

            return (raw_data_size + aggregated_data_size) / 1024  # Convert to MB

        except Exception:
            return 0.0

    async def _process_data_batch(self, data_points: list[DataPoint]):
        """Process a batch of data points."""
        try:
            # Store in memory
            for data_point in data_points:
                self._raw_data.append(data_point)

            # Persist to storage if enabled
            if self.config.enable_persistent_storage:
                await self._persist_data_batch(data_points)

            self.logger.debug(f"Processed batch of {len(data_points)} data points")

        except Exception as e:
            self.logger.error(f"Error processing data batch: {e}")

    async def _aggregation_loop(self):
        """Data aggregation loop."""
        self.logger.info("Data aggregation loop started")

        while self._is_collecting:
            try:
                # Perform aggregation for each interval
                for interval_seconds in self.config.aggregation_intervals:
                    await self._aggregate_data(interval_seconds)

                # Wait before next aggregation
                await asyncio.sleep(60.0)  # Aggregate every minute

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in aggregation loop: {e}")
                await asyncio.sleep(60.0)

        self.logger.info("Data aggregation loop stopped")

    async def _aggregate_data(self, interval_seconds: int):
        """Aggregate data for a specific time interval."""
        try:
            current_time = time.time()
            interval_start = current_time - interval_seconds

            # Group data points by component and metric
            grouped_data = defaultdict(lambda: defaultdict(list))

            for data_point in self._raw_data:
                if data_point.timestamp >= interval_start:
                    key = f"{data_point.component}_{data_point.metric_name}"
                    grouped_data[data_point.component][data_point.metric_name].append(data_point.value)

            # Generate aggregated metrics
            for component, metrics in grouped_data.items():
                for metric_name, values in metrics.items():
                    if values:
                        for agg_func in self.config.aggregation_functions:
                            aggregated_value = self._apply_aggregation_function(values, agg_func)

                            aggregated_metric = AggregatedMetric(
                                start_time=interval_start,
                                end_time=current_time,
                                component=component,
                                metric_name=metric_name,
                                aggregation_function=agg_func,
                                value=aggregated_value,
                                sample_count=len(values),
                                unit=self._get_metric_unit(component, metric_name),
                                tags={"aggregation_interval": str(interval_seconds)},
                            )

                            # Store aggregated metric
                            agg_key = f"{component}_{metric_name}_{agg_func.value}"
                            self._aggregated_data[agg_key][interval_seconds].append(aggregated_metric)

        except Exception as e:
            self.logger.error(f"Error aggregating data for {interval_seconds}s interval: {e}")

    def _apply_aggregation_function(self, values: list[float], func: AggregationFunction) -> float:
        """Apply aggregation function to a list of values."""
        try:
            if not values:
                return 0.0

            if func == AggregationFunction.MEAN:
                return statistics.mean(values)
            elif func == AggregationFunction.MEDIAN:
                return statistics.median(values)
            elif func == AggregationFunction.MIN:
                return min(values)
            elif func == AggregationFunction.MAX:
                return max(values)
            elif func == AggregationFunction.SUM:
                return sum(values)
            elif func == AggregationFunction.COUNT:
                return len(values)
            elif func == AggregationFunction.PERCENTILE_95:
                return self._calculate_percentile(values, 95)
            elif func == AggregationFunction.PERCENTILE_99:
                return self._calculate_percentile(values, 99)
            elif func == AggregationFunction.STDDEV:
                return statistics.stdev(values) if len(values) > 1 else 0.0
            else:
                return statistics.mean(values)  # Default to mean

        except Exception:
            return 0.0

    def _calculate_percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0

        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)

        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower = int(index)
            upper = lower + 1
            weight = index - lower

            if upper >= len(sorted_values):
                return sorted_values[lower]

            return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

    def _get_metric_unit(self, component: str, metric_name: str) -> str:
        """Get the unit for a metric (fallback method)."""
        # Default units based on metric name patterns
        if "time" in metric_name.lower():
            return "ms"
        elif "rate" in metric_name.lower() or "percent" in metric_name.lower():
            return "%"
        elif "memory" in metric_name.lower() or "size" in metric_name.lower():
            return "MB"
        elif "count" in metric_name.lower() or "operations" in metric_name.lower():
            return "count"
        else:
            return "value"

    async def _export_loop(self):
        """Automatic data export loop."""
        self.logger.info("Data export loop started")

        while self._is_collecting:
            try:
                # Export data
                await self._export_data_automatically()

                # Wait for next export
                await asyncio.sleep(self.config.export_interval_hours * 3600)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in export loop: {e}")
                await asyncio.sleep(3600)  # Wait 1 hour on error

        self.logger.info("Data export loop stopped")

    async def _export_data_automatically(self):
        """Automatically export data in configured formats."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            for format_type in self.config.export_formats:
                filename = f"performance_data_{timestamp}.{format_type.value}"
                file_path = self._storage_path / filename

                await self.export_data(file_path=str(file_path), format_type=format_type, hours=24)  # Export last 24 hours

        except Exception as e:
            self.logger.error(f"Error in automatic data export: {e}")

    async def _persist_data_batch(self, data_points: list[DataPoint]):
        """Persist data batch to storage."""
        try:
            if not self.config.enable_persistent_storage:
                return

            # Create daily file
            today = datetime.now().strftime("%Y%m%d")
            filename = f"raw_data_{today}.jsonl"
            file_path = self._storage_path / filename

            # Append data points as JSON lines
            with open(file_path, "a") as f:
                for data_point in data_points:
                    json.dump(data_point.to_dict(), f)
                    f.write("\n")

        except Exception as e:
            self.logger.error(f"Error persisting data batch: {e}")

    async def _process_remaining_data(self):
        """Process any remaining data before shutdown."""
        try:
            # Final aggregation
            if self.config.enable_aggregation:
                for interval_seconds in self.config.aggregation_intervals:
                    await self._aggregate_data(interval_seconds)

            # Final export
            if self.config.enable_persistent_storage:
                remaining_points = list(self._raw_data)
                if remaining_points:
                    await self._persist_data_batch(remaining_points)

        except Exception as e:
            self.logger.error(f"Error processing remaining data: {e}")

    async def collect_on_demand(self, duration_seconds: int = 60, collection_interval: float = 1.0) -> dict[str, Any]:
        """
        Collect performance data on-demand for a specific duration.

        Args:
            duration_seconds: How long to collect data
            collection_interval: Interval between collections

        Returns:
            Collection results dictionary
        """
        try:
            start_time = time.time()
            end_time = start_time + duration_seconds
            collected_points = []

            self.logger.info(f"Starting on-demand collection for {duration_seconds} seconds")

            while time.time() < end_time:
                data_points = await self._collect_current_metrics()
                collected_points.extend(data_points)
                await asyncio.sleep(collection_interval)

            actual_duration = time.time() - start_time

            # Generate summary
            summary = self._generate_collection_summary(collected_points, actual_duration)

            return {
                "success": True,
                "collection_duration": actual_duration,
                "points_collected": len(collected_points),
                "collection_rate": len(collected_points) / actual_duration,
                "summary": summary,
                "data_points": [dp.to_dict() for dp in collected_points],
            }

        except Exception as e:
            self.logger.error(f"Error in on-demand collection: {e}")
            return {"success": False, "error": str(e)}

    def _generate_collection_summary(self, data_points: list[DataPoint], duration: float) -> dict[str, Any]:
        """Generate a summary of collected data points."""
        try:
            if not data_points:
                return {"message": "No data points collected"}

            # Group by component and metric
            by_component = defaultdict(lambda: defaultdict(list))
            for dp in data_points:
                by_component[dp.component][dp.metric_name].append(dp.value)

            summary = {
                "total_points": len(data_points),
                "collection_duration": duration,
                "unique_components": len(by_component),
                "components": {},
            }

            for component, metrics in by_component.items():
                component_summary = {
                    "metric_count": len(metrics),
                    "total_points": sum(len(values) for values in metrics.values()),
                    "metrics": {},
                }

                for metric_name, values in metrics.items():
                    if values:
                        component_summary["metrics"][metric_name] = {
                            "count": len(values),
                            "mean": statistics.mean(values),
                            "min": min(values),
                            "max": max(values),
                        }

                summary["components"][component] = component_summary

            return summary

        except Exception as e:
            self.logger.error(f"Error generating collection summary: {e}")
            return {"error": str(e)}

    def get_raw_data(self, hours: int = 1, component: str | None = None, metric_name: str | None = None) -> list[dict[str, Any]]:
        """
        Get raw performance data points.

        Args:
            hours: Number of hours of data to retrieve
            component: Filter by component name
            metric_name: Filter by metric name

        Returns:
            List of data point dictionaries
        """
        try:
            cutoff_time = time.time() - (hours * 3600)

            filtered_data = [dp for dp in self._raw_data if dp.timestamp > cutoff_time]

            if component:
                filtered_data = [dp for dp in filtered_data if dp.component == component]

            if metric_name:
                filtered_data = [dp for dp in filtered_data if dp.metric_name == metric_name]

            return [dp.to_dict() for dp in filtered_data]

        except Exception as e:
            self.logger.error(f"Error getting raw data: {e}")
            return []

    def get_aggregated_data(
        self,
        hours: int = 24,
        interval_seconds: int = 3600,
        aggregation_function: AggregationFunction = AggregationFunction.MEAN,
        component: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get aggregated performance data.

        Args:
            hours: Number of hours of data to retrieve
            interval_seconds: Aggregation interval
            aggregation_function: Aggregation function to use
            component: Filter by component name

        Returns:
            List of aggregated metric dictionaries
        """
        try:
            cutoff_time = time.time() - (hours * 3600)
            aggregated_results = []

            for agg_key, interval_data in self._aggregated_data.items():
                if interval_seconds in interval_data:
                    for agg_metric in interval_data[interval_seconds]:
                        if agg_metric.end_time > cutoff_time and agg_metric.aggregation_function == aggregation_function:

                            if component is None or agg_metric.component == component:
                                aggregated_results.append(agg_metric.to_dict())

            # Sort by timestamp
            aggregated_results.sort(key=lambda x: x["end_time"])

            return aggregated_results

        except Exception as e:
            self.logger.error(f"Error getting aggregated data: {e}")
            return []

    async def export_data(
        self, file_path: str, format_type: DataFormat = DataFormat.JSON, hours: int = 24, include_aggregated: bool = True
    ) -> dict[str, Any]:
        """
        Export performance data to file.

        Args:
            file_path: Output file path
            format_type: Export format
            hours: Hours of data to export
            include_aggregated: Whether to include aggregated data

        Returns:
            Export result dictionary
        """
        try:
            # Get data to export
            raw_data = self.get_raw_data(hours=hours)
            aggregated_data = []

            if include_aggregated:
                for interval in self.config.aggregation_intervals:
                    for func in self.config.aggregation_functions:
                        agg_data = self.get_aggregated_data(hours=hours, interval_seconds=interval, aggregation_function=func)
                        aggregated_data.extend(agg_data)

            export_data = {
                "export_timestamp": time.time(),
                "export_format": format_type.value,
                "data_hours": hours,
                "raw_data_points": len(raw_data),
                "aggregated_data_points": len(aggregated_data),
                "collection_stats": self._collection_stats.copy(),
                "raw_data": raw_data,
                "aggregated_data": aggregated_data,
            }

            # Export in specified format
            if format_type == DataFormat.JSON:
                await self._export_json(export_data, file_path)
            elif format_type == DataFormat.CSV:
                await self._export_csv(export_data, file_path)
            else:
                raise ValueError(f"Unsupported export format: {format_type}")

            file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0

            return {
                "success": True,
                "file_path": file_path,
                "format": format_type.value,
                "raw_points": len(raw_data),
                "aggregated_points": len(aggregated_data),
                "file_size_bytes": file_size,
                "export_duration": time.time() - export_data["export_timestamp"],
            }

        except Exception as e:
            self.logger.error(f"Error exporting data: {e}")
            return {"success": False, "error": str(e)}

    async def _export_json(self, data: dict[str, Any], file_path: str):
        """Export data as JSON."""
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

    async def _export_csv(self, data: dict[str, Any], file_path: str):
        """Export data as CSV."""
        import csv

        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write raw data
            writer.writerow(["Type", "Timestamp", "Component", "Metric", "Value", "Unit"])

            for dp in data["raw_data"]:
                writer.writerow(["raw", dp["timestamp"], dp["component"], dp["metric_name"], dp["value"], dp["unit"]])

            # Write aggregated data
            for agg in data["aggregated_data"]:
                writer.writerow(
                    [
                        f"aggregated_{agg['aggregation_function']}",
                        agg["end_time"],
                        agg["component"],
                        agg["metric_name"],
                        agg["value"],
                        agg["unit"],
                    ]
                )

    def get_collection_status(self) -> dict[str, Any]:
        """Get current collection status."""
        return {
            "collecting": self._is_collecting,
            "collection_mode": "real_time" if self._collection_task else "stopped",
            "aggregation_enabled": self.config.enable_aggregation,
            "auto_export_enabled": self.config.enable_auto_export,
            "storage_enabled": self.config.enable_persistent_storage,
            "collection_stats": self._collection_stats.copy(),
            "raw_data_points": len(self._raw_data),
            "aggregated_data_keys": len(self._aggregated_data),
            "memory_usage_mb": self._collection_stats["memory_usage_mb"],
            "storage_path": str(self._storage_path),
        }

    async def cleanup_old_data(self):
        """Clean up old data based on retention policy."""
        try:
            cutoff_time = time.time() - (self.config.data_retention_days * 24 * 3600)

            # Clean raw data
            original_count = len(self._raw_data)
            self._raw_data = deque((dp for dp in self._raw_data if dp.timestamp > cutoff_time), maxlen=self._raw_data.maxlen)
            cleaned_raw = original_count - len(self._raw_data)

            # Clean aggregated data
            cleaned_aggregated = 0
            for agg_key in list(self._aggregated_data.keys()):
                for interval in list(self._aggregated_data[agg_key].keys()):
                    original_agg_count = len(self._aggregated_data[agg_key][interval])
                    self._aggregated_data[agg_key][interval] = deque(
                        (agg for agg in self._aggregated_data[agg_key][interval] if agg.end_time > cutoff_time),
                        maxlen=self._aggregated_data[agg_key][interval].maxlen,
                    )
                    cleaned_aggregated += original_agg_count - len(self._aggregated_data[agg_key][interval])

            # Clean storage files
            cleaned_files = 0
            if self.config.enable_persistent_storage:
                for file_path in self._storage_path.glob("*.jsonl"):
                    if file_path.stat().st_mtime < cutoff_time:
                        file_path.unlink()
                        cleaned_files += 1

            self.logger.info(
                f"Cleanup completed: {cleaned_raw} raw data points, "
                f"{cleaned_aggregated} aggregated points, {cleaned_files} files removed"
            )

            return {
                "success": True,
                "cleaned_raw_points": cleaned_raw,
                "cleaned_aggregated_points": cleaned_aggregated,
                "cleaned_files": cleaned_files,
            }

        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            return {"success": False, "error": str(e)}

    async def shutdown(self):
        """Shutdown the data collector."""
        self.logger.info("Shutting down PerformanceDataCollector")
        await self.stop_collection()

        # Final cleanup
        await self.cleanup_old_data()

        # Clear memory
        self._raw_data.clear()
        self._aggregated_data.clear()

        self.logger.info("PerformanceDataCollector shutdown complete")


# Global data collector instance
_data_collector: PerformanceDataCollector | None = None


def get_data_collector() -> PerformanceDataCollector:
    """Get the global data collector instance."""
    global _data_collector
    if _data_collector is None:
        _data_collector = PerformanceDataCollector()
    return _data_collector
