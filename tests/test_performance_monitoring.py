"""
Integration tests for performance monitoring - Wave 15.2.5
Tests performance monitoring, metrics collection, and optimization workflows.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.services.cache_service import CacheService
from src.services.cache_performance_service import CachePerformanceService
from src.services.cache_memory_pressure_service import CacheMemoryPressureService
from src.services.cache_memory_profiler import CacheMemoryProfiler
from src.services.cache_memory_reporter import CacheMemoryReporter
from src.utils.performance_monitor import PerformanceMonitor
from src.utils.metrics_collector import MetricsCollector
from src.utils.alerting_system import AlertingSystem


class TestPerformanceMonitoringBase:
    """Base class for performance monitoring tests."""

    @pytest.fixture
    async def cache_service(self):
        """Create cache service for testing."""
        service = CacheService()
        service._redis_client = AsyncMock()
        
        # Mock Redis operations with realistic timing
        async def mock_get(key):
            await asyncio.sleep(0.001)  # 1ms latency
            return json.dumps({"test": "data"}) if "exists" in key else None
        
        async def mock_set(key, value, **kwargs):
            await asyncio.sleep(0.002)  # 2ms latency
            return True
        
        async def mock_delete(*keys):
            await asyncio.sleep(0.001)  # 1ms latency
            return len(keys)
        
        service._redis_client.get.side_effect = mock_get
        service._redis_client.set.side_effect = mock_set
        service._redis_client.delete.side_effect = mock_delete
        service._redis_client.info.return_value = {
            "used_memory": 1024 * 1024 * 50,  # 50MB
            "maxmemory": 1024 * 1024 * 100,   # 100MB
            "connected_clients": 10,
            "total_commands_processed": 1000
        }
        
        yield service
        await service.close()

    @pytest.fixture
    async def performance_service(self):
        """Create performance monitoring service."""
        service = CachePerformanceService()
        yield service
        await service.close()

    @pytest.fixture
    async def memory_service(self):
        """Create memory monitoring service."""
        service = CacheMemoryPressureService()
        yield service
        await service.close()

    @pytest.fixture
    async def metrics_collector(self):
        """Create metrics collector."""
        collector = MetricsCollector()
        yield collector
        await collector.close()


class TestCachePerformanceMonitoring(TestPerformanceMonitoringBase):
    """Test cache performance monitoring functionality."""

    @pytest.mark.asyncio
    async def test_operation_timing_monitoring(
        self, cache_service, performance_service
    ):
        """Test monitoring of cache operation timing."""
        # Connect performance monitoring
        cache_service.enable_performance_monitoring(performance_service)
        
        # Perform various cache operations
        operations = [
            ("get", "exists:key1"),
            ("get", "missing:key1"),
            ("set", "new:key1", {"data": "test1"}),
            ("set", "new:key2", {"data": "test2"}),
            ("delete", "new:key1"),
            ("get", "exists:key2"),
        ]
        
        for op_type, key, *args in operations:
            if op_type == "get":
                await cache_service.get(key)
            elif op_type == "set":
                await cache_service.set(key, args[0])
            elif op_type == "delete":
                await cache_service.delete(key)
        
        # Get performance metrics
        metrics = await performance_service.get_metrics()
        
        # Verify timing metrics
        assert metrics["total_operations"] == 6
        assert metrics["operation_breakdown"]["get"] == 3
        assert metrics["operation_breakdown"]["set"] == 2
        assert metrics["operation_breakdown"]["delete"] == 1
        
        # Verify latency tracking
        assert metrics["average_latency"] > 0
        assert metrics["p95_latency"] > 0
        assert metrics["p99_latency"] > 0

    @pytest.mark.asyncio
    async def test_throughput_monitoring(
        self, cache_service, performance_service
    ):
        """Test cache throughput monitoring."""
        cache_service.enable_performance_monitoring(performance_service)
        
        # Simulate high-throughput operations
        start_time = time.time()
        
        # Perform 100 operations rapidly
        tasks = []
        for i in range(100):
            if i % 3 == 0:
                task = cache_service.get(f"key:{i}")
            elif i % 3 == 1:
                task = cache_service.set(f"key:{i}", {"index": i})
            else:
                task = cache_service.delete(f"key:{i}")
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        # Get throughput metrics
        metrics = await performance_service.get_metrics()
        
        # Verify throughput calculation
        ops_per_second = metrics["operations_per_second"]
        assert ops_per_second > 0
        assert ops_per_second == pytest.approx(100 / elapsed, rel=0.1)
        
        # Verify concurrent operation handling
        assert metrics["peak_concurrent_operations"] > 1

    @pytest.mark.asyncio
    async def test_hit_rate_monitoring(
        self, cache_service, performance_service
    ):
        """Test cache hit rate monitoring."""
        cache_service.enable_performance_monitoring(performance_service)
        
        # Set up some cache data
        for i in range(10):
            await cache_service.set(f"exists:key:{i}", {"index": i})
        
        # Perform mixed hit/miss operations
        hit_operations = [cache_service.get(f"exists:key:{i}") for i in range(10)]
        miss_operations = [cache_service.get(f"missing:key:{i}") for i in range(5)]
        
        await asyncio.gather(*(hit_operations + miss_operations))
        
        # Get hit rate metrics
        metrics = await performance_service.get_metrics()
        
        # Verify hit rate calculation
        expected_hit_rate = 10 / 15  # 10 hits out of 15 gets
        assert metrics["hit_rate"] == pytest.approx(expected_hit_rate, rel=0.01)
        assert metrics["cache_hits"] == 10
        assert metrics["cache_misses"] == 5

    @pytest.mark.asyncio
    async def test_error_rate_monitoring(
        self, cache_service, performance_service
    ):
        """Test cache error rate monitoring."""
        cache_service.enable_performance_monitoring(performance_service)
        
        # Mock some operations to fail
        original_get = cache_service._redis_client.get
        
        async def failing_get(key):
            if "error" in key:
                raise Exception("Simulated Redis error")
            return await original_get(key)
        
        cache_service._redis_client.get.side_effect = failing_get
        
        # Perform operations with some failures
        operations = [
            cache_service.get("normal:key1"),
            cache_service.get("error:key1"),
            cache_service.get("normal:key2"),
            cache_service.get("error:key2"),
            cache_service.get("normal:key3"),
        ]
        
        # Execute with error handling
        for op in operations:
            try:
                await op
            except:
                pass  # Expected for error keys
        
        # Get error metrics
        metrics = await performance_service.get_metrics()
        
        # Verify error tracking
        assert metrics["total_errors"] == 2
        assert metrics["error_rate"] == pytest.approx(2/5, rel=0.01)
        assert "error_breakdown" in metrics


class TestMemoryPressureMonitoring(TestPerformanceMonitoringBase):
    """Test memory pressure monitoring and management."""

    @pytest.mark.asyncio
    async def test_memory_usage_tracking(
        self, cache_service, memory_service
    ):
        """Test memory usage tracking and alerts."""
        # Connect memory monitoring
        cache_service.set_memory_pressure_service(memory_service)
        memory_service.set_cache_service(cache_service)
        
        # Configure memory thresholds
        await memory_service.set_thresholds({
            "warning": 0.7,    # 70%
            "critical": 0.9,   # 90%
            "emergency": 0.95  # 95%
        })
        
        # Mock high memory usage
        cache_service._redis_client.info.return_value = {
            "used_memory": 1024 * 1024 * 85,  # 85MB
            "maxmemory": 1024 * 1024 * 100,   # 100MB (85% usage)
            "evicted_keys": 0
        }
        
        # Check memory status
        status = await memory_service.check_memory_pressure()
        
        # Should trigger warning threshold
        assert status["pressure_level"] == "warning"
        assert status["memory_usage_percent"] == 85.0
        assert status["needs_action"] is True

    @pytest.mark.asyncio
    async def test_automatic_memory_management(
        self, cache_service, memory_service
    ):
        """Test automatic memory management under pressure."""
        cache_service.set_memory_pressure_service(memory_service)
        memory_service.set_cache_service(cache_service)
        
        # Fill cache with test data
        for i in range(100):
            await cache_service.set(
                f"memory:test:{i}",
                {"data": "x" * 1000, "priority": i % 3, "created": time.time()}
            )
        
        # Mock critical memory usage
        cache_service._redis_client.info.return_value = {
            "used_memory": 1024 * 1024 * 95,  # 95MB
            "maxmemory": 1024 * 1024 * 100,   # 100MB (95% usage)
            "evicted_keys": 0
        }
        
        # Trigger memory management
        result = await memory_service.handle_memory_pressure()
        
        # Should have attempted eviction
        assert result["action_taken"] is True
        assert result["memory_freed"] > 0 or result["keys_evicted"] > 0

    @pytest.mark.asyncio
    async def test_memory_leak_detection(
        self, cache_service, memory_service
    ):
        """Test memory leak detection."""
        profiler = CacheMemoryProfiler()
        cache_service.set_memory_profiler(profiler)
        
        # Simulate memory growth pattern
        memory_samples = [
            (time.time() - 300, 50 * 1024 * 1024),  # 50MB 5 min ago
            (time.time() - 240, 60 * 1024 * 1024),  # 60MB 4 min ago
            (time.time() - 180, 70 * 1024 * 1024),  # 70MB 3 min ago
            (time.time() - 120, 80 * 1024 * 1024),  # 80MB 2 min ago
            (time.time() - 60,  90 * 1024 * 1024),  # 90MB 1 min ago
            (time.time(),       100 * 1024 * 1024), # 100MB now
        ]
        
        for timestamp, memory_usage in memory_samples:
            await profiler.record_memory_sample(timestamp, memory_usage)
        
        # Analyze for memory leaks
        analysis = await profiler.analyze_memory_trends()
        
        # Should detect upward trend
        assert analysis["trend"] == "increasing"
        assert analysis["growth_rate"] > 0
        assert analysis["leak_probability"] > 0.7  # High probability


class TestMetricsCollectionAndReporting(TestPerformanceMonitoringBase):
    """Test comprehensive metrics collection and reporting."""

    @pytest.mark.asyncio
    async def test_comprehensive_metrics_collection(
        self, cache_service, performance_service, memory_service, metrics_collector
    ):
        """Test collection of comprehensive performance metrics."""
        # Connect all monitoring services
        cache_service.enable_performance_monitoring(performance_service)
        cache_service.set_memory_pressure_service(memory_service)
        
        # Register metrics collector
        await metrics_collector.register_source("cache", cache_service)
        await metrics_collector.register_source("performance", performance_service)
        await metrics_collector.register_source("memory", memory_service)
        
        # Perform various operations to generate metrics
        operations = [
            lambda: cache_service.set("metrics:test:1", {"data": "test1"}),
            lambda: cache_service.get("metrics:test:1"),
            lambda: cache_service.get("metrics:missing:1"),
            lambda: cache_service.delete("metrics:test:1"),
        ]
        
        for operation in operations:
            await operation()
        
        # Collect comprehensive metrics
        all_metrics = await metrics_collector.collect_all_metrics()
        
        # Verify comprehensive coverage
        assert "cache" in all_metrics
        assert "performance" in all_metrics
        assert "memory" in all_metrics
        
        # Verify metric completeness
        perf_metrics = all_metrics["performance"]
        assert "latency" in perf_metrics
        assert "throughput" in perf_metrics
        assert "hit_rate" in perf_metrics
        assert "error_rate" in perf_metrics

    @pytest.mark.asyncio
    async def test_historical_metrics_tracking(
        self, cache_service, performance_service, metrics_collector
    ):
        """Test historical metrics tracking and analysis."""
        cache_service.enable_performance_monitoring(performance_service)
        await metrics_collector.register_source("performance", performance_service)
        
        # Configure historical tracking
        await metrics_collector.enable_historical_tracking(
            interval_seconds=0.1,  # Fast for testing
            retention_hours=1
        )
        
        # Generate operations over time
        for round_num in range(5):
            # Perform operations
            for i in range(10):
                await cache_service.set(f"historical:{round_num}:{i}", {"round": round_num})
                await cache_service.get(f"historical:{round_num}:{i}")
            
            # Wait between rounds
            await asyncio.sleep(0.1)
        
        # Get historical analysis
        historical_data = await metrics_collector.get_historical_metrics(
            metric_name="operations_per_second",
            time_range_minutes=1
        )
        
        # Verify historical tracking
        assert len(historical_data) >= 3  # Should have multiple data points
        assert all("timestamp" in point for point in historical_data)
        assert all("value" in point for point in historical_data)

    @pytest.mark.asyncio
    async def test_performance_alerting(
        self, cache_service, performance_service
    ):
        """Test performance-based alerting system."""
        alerting_system = AlertingSystem()
        
        # Configure performance alerts
        await alerting_system.add_alert_rule({
            "name": "high_latency",
            "metric": "average_latency",
            "threshold": 0.01,  # 10ms
            "operator": ">",
            "severity": "warning",
            "enabled": True
        })
        
        await alerting_system.add_alert_rule({
            "name": "low_hit_rate",
            "metric": "hit_rate",
            "threshold": 0.5,  # 50%
            "operator": "<",
            "severity": "critical",
            "enabled": True
        })
        
        # Connect to performance service
        performance_service.set_alerting_system(alerting_system)
        cache_service.enable_performance_monitoring(performance_service)
        
        # Mock slow operations to trigger latency alert
        async def slow_operation():
            await asyncio.sleep(0.02)  # 20ms - above threshold
            return "slow_result"
        
        cache_service._redis_client.get.side_effect = lambda key: slow_operation()
        
        # Perform operations that should trigger alerts
        for i in range(5):
            await cache_service.get(f"slow:key:{i}")
        
        # Check for triggered alerts
        active_alerts = await alerting_system.get_active_alerts()
        
        # Should have latency alert
        latency_alerts = [a for a in active_alerts if a["rule_name"] == "high_latency"]
        assert len(latency_alerts) > 0

    @pytest.mark.asyncio
    async def test_performance_optimization_recommendations(
        self, cache_service, performance_service
    ):
        """Test automatic performance optimization recommendations."""
        cache_service.enable_performance_monitoring(performance_service)
        
        # Simulate various performance patterns
        
        # Pattern 1: High miss rate
        for i in range(50):
            await cache_service.get(f"missing:key:{i}")  # All misses
        
        # Pattern 2: Memory pressure
        cache_service._redis_client.info.return_value["used_memory"] = 95 * 1024 * 1024
        
        # Pattern 3: Slow operations
        cache_service._redis_client.get.side_effect = lambda key: asyncio.sleep(0.05)
        for i in range(10):
            await cache_service.get(f"slow:key:{i}")
        
        # Get optimization recommendations
        recommendations = await performance_service.get_optimization_recommendations()
        
        # Should provide actionable recommendations
        assert len(recommendations) > 0
        
        # Check for expected recommendation types
        rec_types = [rec["type"] for rec in recommendations]
        assert any("cache_warming" in rec_type for rec_type in rec_types)
        assert any("memory" in rec_type for rec_type in rec_types)


class TestPerformanceBenchmarking(TestPerformanceMonitoringBase):
    """Test performance benchmarking and baseline establishment."""

    @pytest.mark.asyncio
    async def test_performance_baseline_establishment(
        self, cache_service, performance_service
    ):
        """Test establishment of performance baselines."""
        cache_service.enable_performance_monitoring(performance_service)
        
        # Run baseline benchmark
        baseline_config = {
            "operations": ["get", "set", "delete"],
            "key_patterns": ["baseline:test:{i}"],
            "iterations": 100,
            "concurrency": 10
        }
        
        baseline_results = await performance_service.run_baseline_benchmark(baseline_config)
        
        # Verify baseline metrics
        assert baseline_results["total_operations"] == 300  # 100 × 3 operations
        assert baseline_results["baseline_latency"] > 0
        assert baseline_results["baseline_throughput"] > 0
        
        # Store baseline for comparison
        await performance_service.store_baseline(baseline_results)
        
        # Verify baseline storage
        stored_baseline = await performance_service.get_current_baseline()
        assert stored_baseline["baseline_latency"] == baseline_results["baseline_latency"]

    @pytest.mark.asyncio
    async def test_performance_regression_detection(
        self, cache_service, performance_service
    ):
        """Test detection of performance regressions."""
        cache_service.enable_performance_monitoring(performance_service)
        
        # Establish baseline
        await performance_service.store_baseline({
            "baseline_latency": 0.001,  # 1ms baseline
            "baseline_throughput": 1000,  # 1000 ops/sec baseline
            "baseline_hit_rate": 0.9  # 90% hit rate baseline
        })
        
        # Simulate degraded performance
        async def slow_operation():
            await asyncio.sleep(0.01)  # 10ms - 10x slower than baseline
            return "result"
        
        cache_service._redis_client.get.side_effect = lambda key: slow_operation()
        
        # Perform operations with degraded performance
        for i in range(20):
            await cache_service.get(f"regression:test:{i}")
        
        # Check for regression detection
        regression_analysis = await performance_service.detect_regressions()
        
        # Should detect latency regression
        assert regression_analysis["regressions_detected"] is True
        latency_regression = next(
            (r for r in regression_analysis["regressions"] if r["metric"] == "latency"),
            None
        )
        assert latency_regression is not None
        assert latency_regression["regression_factor"] > 5  # 5x worse than baseline

    @pytest.mark.asyncio
    async def test_load_testing_integration(
        self, cache_service, performance_service
    ):
        """Test integration with load testing scenarios."""
        cache_service.enable_performance_monitoring(performance_service)
        
        # Define load test scenarios
        load_scenarios = [
            {
                "name": "steady_load",
                "operations_per_second": 100,
                "duration_seconds": 1,
                "operation_mix": {"get": 0.7, "set": 0.2, "delete": 0.1}
            },
            {
                "name": "burst_load",
                "operations_per_second": 500,
                "duration_seconds": 0.5,
                "operation_mix": {"get": 0.8, "set": 0.2}
            }
        ]
        
        # Execute load scenarios
        load_results = {}
        
        for scenario in load_scenarios:
            result = await performance_service.execute_load_scenario(scenario)
            load_results[scenario["name"]] = result
        
        # Verify load test results
        assert "steady_load" in load_results
        assert "burst_load" in load_results
        
        # Steady load should have consistent performance
        steady_result = load_results["steady_load"]
        assert steady_result["completed_operations"] >= 90  # Allow some variance
        
        # Burst load should handle higher throughput
        burst_result = load_results["burst_load"]
        assert burst_result["peak_throughput"] > steady_result["average_throughput"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])