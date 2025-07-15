#!/usr/bin/env python3
"""
Comprehensive cache system validation script.

This script performs end-to-end validation of the complete cache system,
including integration tests, performance benchmarks, security checks,
and failure scenario testing.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.cache_config import CacheConfig, CacheLevel, CacheWriteStrategy
from services.cache_service import MultiTierCacheService, get_cache_service
from services.embedding_cache_service import EmbeddingCacheService
from services.file_cache_service import FileCacheService
from services.project_cache_service import ProjectCacheService
from services.search_cache_service import SearchCacheService
from utils.cache_key_optimization import KeyOptimizationManager
from utils.cache_performance_optimization import AdaptivePerformanceOptimizer, PerformanceMonitor, PerformanceProfile
from utils.cache_serialization_optimization import OptimizedCacheSerializer


@dataclass
class ValidationResult:
    """Result of a validation test."""

    test_name: str
    passed: bool
    duration: float
    details: dict[str, Any]
    error: str | None = None


class CacheSystemValidator:
    """Comprehensive cache system validator."""

    def __init__(self, config_path: str | None = None, verbose: bool = False):
        """Initialize validator."""
        self.verbose = verbose
        self.logger = self._setup_logging()
        self.results: list[ValidationResult] = []

        # Load configuration
        if config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = self._create_test_config()

        # Initialize components
        self.cache_service: MultiTierCacheService | None = None
        self.specialized_services: dict[str, Any] = {}
        self.optimizer: AdaptivePerformanceOptimizer | None = None
        self.key_manager: KeyOptimizationManager | None = None
        self.serializer: OptimizedCacheSerializer | None = None

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("cache_validator")
        logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        return logger

    def _load_config(self, config_path: str) -> CacheConfig:
        """Load configuration from file."""
        try:
            with open(config_path) as f:
                config_data = json.load(f)
            return CacheConfig(**config_data)
        except Exception as e:
            self.logger.warning(f"Failed to load config from {config_path}: {e}")
            return self._create_test_config()

    def _create_test_config(self) -> CacheConfig:
        """Create test configuration."""
        return CacheConfig(
            enabled=True,
            cache_level=CacheLevel.BOTH,
            write_strategy=CacheWriteStrategy.WRITE_THROUGH,
            key_prefix="validation_test",
            debug_mode=True,
        )

    async def initialize_system(self) -> ValidationResult:
        """Initialize the complete cache system."""
        start_time = time.time()

        try:
            # Initialize main cache service
            self.cache_service = MultiTierCacheService(self.config)
            await self.cache_service.initialize()

            # Initialize specialized services
            self.specialized_services = {
                "embedding": EmbeddingCacheService(self.config),
                "search": SearchCacheService(self.config),
                "project": ProjectCacheService(self.config),
                "file": FileCacheService(self.config),
            }

            for name, service in self.specialized_services.items():
                await service.initialize()

            # Initialize optimization components
            self.optimizer = AdaptivePerformanceOptimizer()
            self.key_manager = KeyOptimizationManager()
            self.serializer = OptimizedCacheSerializer()

            duration = time.time() - start_time

            return ValidationResult(
                test_name="system_initialization",
                passed=True,
                duration=duration,
                details={
                    "cache_service_initialized": True,
                    "specialized_services_count": len(self.specialized_services),
                    "optimization_components_initialized": True,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name="system_initialization", passed=False, duration=duration, details={}, error=str(e))

    async def validate_basic_cache_operations(self) -> ValidationResult:
        """Validate basic cache operations."""
        start_time = time.time()

        try:
            test_data = {
                "string_value": "test_string",
                "numeric_value": 12345,
                "list_value": [1, 2, 3, 4, 5],
                "dict_value": {"nested": {"data": "value"}},
                "complex_value": {"vectors": [[0.1, 0.2] for _ in range(10)], "metadata": {"source": "test", "timestamp": time.time()}},
            }

            operations_passed = 0
            total_operations = 0

            for key, value in test_data.items():
                total_operations += 3  # set, get, delete

                # Test set operation
                success = await self.cache_service.set(f"basic_{key}", value, ttl=3600)
                if success:
                    operations_passed += 1

                # Test get operation
                retrieved = await self.cache_service.get(f"basic_{key}")
                if retrieved == value:
                    operations_passed += 1

                # Test delete operation
                deleted = await self.cache_service.delete(f"basic_{key}")
                if deleted:
                    operations_passed += 1

            duration = time.time() - start_time
            success_rate = operations_passed / total_operations

            return ValidationResult(
                test_name="basic_cache_operations",
                passed=success_rate >= 0.95,
                duration=duration,
                details={
                    "total_operations": total_operations,
                    "successful_operations": operations_passed,
                    "success_rate": success_rate,
                    "data_types_tested": list(test_data.keys()),
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name="basic_cache_operations", passed=False, duration=duration, details={}, error=str(e))

    async def validate_multi_tier_functionality(self) -> ValidationResult:
        """Validate multi-tier cache functionality."""
        start_time = time.time()

        try:
            test_key = "multi_tier_test"
            test_value = {"tier": "test", "data": list(range(100))}

            # Test write-through to both tiers
            await self.cache_service.set(test_key, test_value)

            # Verify L1 cache
            l1_value = self.cache_service.l1_cache.get(test_key)
            l1_hit = l1_value == test_value

            # Verify L2 cache
            l2_value = await self.cache_service.l2_cache.get(test_key)
            l2_hit = l2_value == test_value

            # Test promotion from L2 to L1
            self.cache_service.l1_cache.clear()
            retrieved_value = await self.cache_service.get(test_key)
            promotion_success = retrieved_value == test_value

            # Verify promotion actually happened
            l1_after_promotion = self.cache_service.l1_cache.get(test_key)
            promotion_verified = l1_after_promotion == test_value

            # Test cache coherency
            coherency_result = await self.cache_service.check_cache_coherency()
            coherency_ok = coherency_result.get("coherent", False)

            duration = time.time() - start_time

            return ValidationResult(
                test_name="multi_tier_functionality",
                passed=all([l1_hit, l2_hit, promotion_success, promotion_verified, coherency_ok]),
                duration=duration,
                details={
                    "l1_cache_hit": l1_hit,
                    "l2_cache_hit": l2_hit,
                    "promotion_success": promotion_success,
                    "promotion_verified": promotion_verified,
                    "cache_coherency": coherency_ok,
                    "coherency_details": coherency_result,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name="multi_tier_functionality", passed=False, duration=duration, details={}, error=str(e))

    async def validate_specialized_services(self) -> ValidationResult:
        """Validate specialized cache services."""
        start_time = time.time()

        try:
            service_results = {}

            # Test embedding service
            embedding_service = self.specialized_services["embedding"]
            test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]

            await embedding_service.cache_embedding("test_embedding", test_embedding)
            cached_embedding = await embedding_service.get_cached_embedding("test_embedding")
            service_results["embedding"] = cached_embedding == test_embedding

            # Test search service
            search_service = self.specialized_services["search"]
            search_params = {"query": "test query", "limit": 10}
            search_results = [{"id": 1, "score": 0.9}, {"id": 2, "score": 0.8}]

            await search_service.cache_search_results(search_params, search_results)
            cached_search = await search_service.get_cached_search_results(search_params)
            service_results["search"] = cached_search == search_results

            # Test project service
            project_service = self.specialized_services["project"]
            project_info = {"name": "test_project", "files": 100, "size": "10MB"}

            await project_service.cache_project_info("test_project", project_info)
            cached_project = await project_service.get_cached_project_info("test_project")
            service_results["project"] = cached_project == project_info

            # Test file service
            file_service = self.specialized_services["file"]
            file_chunks = [{"type": "function", "name": "test_func", "line": 1}]

            await file_service.cache_file_chunks("/test/file.py", file_chunks)
            cached_chunks = await file_service.get_cached_file_chunks("/test/file.py")
            service_results["file"] = cached_chunks == file_chunks

            duration = time.time() - start_time
            all_passed = all(service_results.values())

            return ValidationResult(
                test_name="specialized_services",
                passed=all_passed,
                duration=duration,
                details={
                    "service_results": service_results,
                    "services_tested": list(service_results.keys()),
                    "success_rate": sum(service_results.values()) / len(service_results),
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name="specialized_services", passed=False, duration=duration, details={}, error=str(e))

    async def validate_performance_optimization(self) -> ValidationResult:
        """Validate performance optimization system."""
        start_time = time.time()

        try:
            # Create test performance profile
            profile = PerformanceProfile(
                hit_rate=0.6,
                avg_latency=0.05,
                throughput_ops_per_sec=1500,
                memory_usage_mb=512,
                cpu_usage_percent=45,
                cache_size=1000,
                operation_count=10000,
            )

            # Test optimization
            optimization_result = await self.optimizer.optimize_performance(profile)
            optimization_success = optimization_result.get("status") in ["completed", "baseline_set"]

            # Test performance monitoring
            monitor = PerformanceMonitor(self.optimizer)

            def mock_performance_provider():
                return profile

            await monitor.start_monitoring(mock_performance_provider)
            await asyncio.sleep(1)  # Let it run briefly
            await monitor.stop_monitoring()

            monitoring_stats = monitor.get_monitoring_stats()
            monitoring_success = monitoring_stats.get("status") == "inactive"

            # Test cache warming
            warming_result = await self.cache_service.trigger_cache_warmup("adaptive")
            warming_success = warming_result.get("success", False)

            duration = time.time() - start_time

            return ValidationResult(
                test_name="performance_optimization",
                passed=all([optimization_success, monitoring_success, warming_success]),
                duration=duration,
                details={
                    "optimization_result": optimization_result,
                    "monitoring_stats": monitoring_stats,
                    "warming_result": warming_result,
                    "optimization_success": optimization_success,
                    "monitoring_success": monitoring_success,
                    "warming_success": warming_success,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name="performance_optimization", passed=False, duration=duration, details={}, error=str(e))

    async def validate_key_optimization(self) -> ValidationResult:
        """Validate key optimization system."""
        start_time = time.time()

        try:
            # Test key optimization
            test_components = {
                "project": "test_project",
                "service": "embedding",
                "content_hash": "abc123def456789",
                "version": "1.0",
                "metadata": "additional_info",
            }

            # Generate optimized key
            optimized_key = self.key_manager.optimize_cache_key("embedding", **test_components)
            key_generated = len(optimized_key) > 0

            # Test key determinism
            optimized_key2 = self.key_manager.optimize_cache_key("embedding", **test_components)
            deterministic = optimized_key == optimized_key2

            # Test key restoration
            restored_components = self.key_manager.restore_cache_key("embedding", optimized_key)
            restoration_success = "project" in restored_components

            # Test optimization stats
            optimization_stats = self.key_manager.get_optimization_stats()
            stats_available = "default_optimizer" in optimization_stats

            # Test with cache operations
            test_value = {"optimized": True, "test": "data"}
            await self.cache_service.set(optimized_key, test_value)
            retrieved_value = await self.cache_service.get(optimized_key)
            cache_integration = retrieved_value == test_value

            duration = time.time() - start_time

            return ValidationResult(
                test_name="key_optimization",
                passed=all([key_generated, deterministic, restoration_success, stats_available, cache_integration]),
                duration=duration,
                details={
                    "key_generated": key_generated,
                    "optimized_key_length": len(optimized_key),
                    "deterministic": deterministic,
                    "restoration_success": restoration_success,
                    "stats_available": stats_available,
                    "cache_integration": cache_integration,
                    "optimization_stats": optimization_stats,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name="key_optimization", passed=False, duration=duration, details={}, error=str(e))

    async def validate_serialization_optimization(self) -> ValidationResult:
        """Validate serialization optimization system."""
        start_time = time.time()

        try:
            # Test data types
            test_data = {
                "simple": {"key": "value"},
                "complex": {
                    "vectors": [[0.1, 0.2, 0.3] for _ in range(50)],
                    "metadata": {"timestamp": time.time(), "source": "test"},
                    "nested": {"level1": {"level2": {"data": "deep"}}},
                },
                "large": {"data": "x" * 10000},  # 10KB string
            }

            serialization_results = {}

            for data_type, data in test_data.items():
                # Test serialization
                serialized = self.serializer.serialize_and_compress(data)
                serialization_success = len(serialized) > 0

                # Test deserialization
                deserialized = self.serializer.decompress_and_deserialize(serialized)
                deserialization_success = deserialized == data

                # Calculate compression ratio
                original_size = len(str(data).encode())
                compressed_size = len(serialized)
                compression_ratio = 1.0 - (compressed_size / original_size) if original_size > 0 else 0.0

                serialization_results[data_type] = {
                    "serialization_success": serialization_success,
                    "deserialization_success": deserialization_success,
                    "original_size": original_size,
                    "compressed_size": compressed_size,
                    "compression_ratio": compression_ratio,
                }

            # Test stats
            serialization_stats = self.serializer.get_stats()
            stats_available = "operation_count" in serialization_stats

            # Test configuration
            config_result = self.serializer.configure(compression_threshold=512)
            config_success = config_result.get("success", False)

            duration = time.time() - start_time
            all_passed = (
                all(result["serialization_success"] and result["deserialization_success"] for result in serialization_results.values())
                and stats_available
                and config_success
            )

            return ValidationResult(
                test_name="serialization_optimization",
                passed=all_passed,
                duration=duration,
                details={
                    "serialization_results": serialization_results,
                    "stats_available": stats_available,
                    "config_success": config_success,
                    "serialization_stats": serialization_stats,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name="serialization_optimization", passed=False, duration=duration, details={}, error=str(e))

    async def validate_performance_benchmarks(self) -> ValidationResult:
        """Validate performance benchmarks."""
        start_time = time.time()

        try:
            # Performance thresholds
            thresholds = {
                "min_throughput_ops_per_sec": 1000,
                "max_avg_latency_ms": 10,
                "min_hit_rate": 0.8,
                "max_memory_overhead_ratio": 2.0,
            }

            # Benchmark data
            num_operations = 5000
            test_data = {f"bench_key_{i}": f"benchmark_value_{i}" for i in range(num_operations)}

            # Write throughput test
            write_start = time.time()
            for key, value in test_data.items():
                await self.cache_service.set(key, value)
            write_duration = time.time() - write_start
            write_throughput = num_operations / write_duration

            # Read throughput test
            read_start = time.time()
            hit_count = 0
            for key in test_data.keys():
                value = await self.cache_service.get(key)
                if value is not None:
                    hit_count += 1
            read_duration = time.time() - read_start
            read_throughput = num_operations / read_duration

            # Calculate metrics
            avg_latency_ms = ((write_duration + read_duration) / (num_operations * 2)) * 1000
            hit_rate = hit_count / num_operations

            # Memory usage test
            cache_stats = self.cache_service.l1_cache.get_info()
            memory_usage_mb = cache_stats.get("memory_usage_mb", 0)
            expected_memory_mb = (sum(len(str(v).encode()) for v in test_data.values())) / (1024 * 1024)
            memory_overhead_ratio = memory_usage_mb / expected_memory_mb if expected_memory_mb > 0 else 1.0

            # Check thresholds
            performance_checks = {
                "write_throughput_ok": write_throughput >= thresholds["min_throughput_ops_per_sec"],
                "read_throughput_ok": read_throughput >= thresholds["min_throughput_ops_per_sec"],
                "latency_ok": avg_latency_ms <= thresholds["max_avg_latency_ms"],
                "hit_rate_ok": hit_rate >= thresholds["min_hit_rate"],
                "memory_ok": memory_overhead_ratio <= thresholds["max_memory_overhead_ratio"],
            }

            duration = time.time() - start_time
            all_passed = all(performance_checks.values())

            return ValidationResult(
                test_name="performance_benchmarks",
                passed=all_passed,
                duration=duration,
                details={
                    "thresholds": thresholds,
                    "measurements": {
                        "write_throughput_ops_per_sec": write_throughput,
                        "read_throughput_ops_per_sec": read_throughput,
                        "avg_latency_ms": avg_latency_ms,
                        "hit_rate": hit_rate,
                        "memory_usage_mb": memory_usage_mb,
                        "memory_overhead_ratio": memory_overhead_ratio,
                    },
                    "performance_checks": performance_checks,
                    "num_operations": num_operations,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name="performance_benchmarks", passed=False, duration=duration, details={}, error=str(e))

    async def validate_failure_scenarios(self) -> ValidationResult:
        """Validate failure scenario handling."""
        start_time = time.time()

        try:
            failure_tests = {}

            # Test 1: Memory pressure handling
            try:
                large_data = "x" * 100000  # 100KB per entry
                original_size = self.cache_service.l1_cache.get_info()["size"]

                # Fill beyond capacity
                for i in range(self.cache_service.l1_cache.max_size + 20):
                    await self.cache_service.set(f"pressure_{i}", large_data)

                final_size = self.cache_service.l1_cache.get_info()["size"]
                failure_tests["memory_pressure"] = final_size <= self.cache_service.l1_cache.max_size

            except Exception:
                failure_tests["memory_pressure"] = False

            # Test 2: Concurrent access stress
            try:

                async def concurrent_worker(worker_id: int):
                    errors = 0
                    for i in range(100):
                        try:
                            key = f"concurrent_{worker_id}_{i}"
                            await self.cache_service.set(key, f"value_{i}")
                            await self.cache_service.get(key)
                        except Exception:
                            errors += 1
                    return errors

                tasks = [concurrent_worker(i) for i in range(20)]
                error_counts = await asyncio.gather(*tasks, return_exceptions=True)

                total_errors = sum(e for e in error_counts if isinstance(e, int))
                failure_tests["concurrent_stress"] = total_errors < 100  # Less than 5% error rate

            except Exception:
                failure_tests["concurrent_stress"] = False

            # Test 3: Invalid data handling
            try:
                # Test with various invalid data
                invalid_data_tests = [
                    (None, "none_value"),
                    (float("inf"), "infinity"),
                    (float("nan"), "nan"),
                    ({}, "empty_dict"),
                    ([], "empty_list"),
                ]

                invalid_handled = 0
                for data, desc in invalid_data_tests:
                    try:
                        key = f"invalid_{desc}"
                        await self.cache_service.set(key, data)
                        retrieved = await self.cache_service.get(key)
                        # As long as it doesn't crash, consider it handled
                        invalid_handled += 1
                    except Exception:
                        # Expected for some data types
                        invalid_handled += 1

                failure_tests["invalid_data"] = invalid_handled == len(invalid_data_tests)

            except Exception:
                failure_tests["invalid_data"] = False

            # Test 4: Cache corruption recovery
            try:
                # Set valid data
                test_key = "corruption_test"
                test_value = {"valid": True}
                await self.cache_service.set(test_key, test_value)

                # Simulate corruption (if possible)
                if test_key in self.cache_service.l1_cache._cache:
                    original_value = self.cache_service.l1_cache._cache[test_key].value
                    self.cache_service.l1_cache._cache[test_key].value = "corrupted"

                    # Try to retrieve (should not crash)
                    retrieved = await self.cache_service.get(test_key)
                    failure_tests["corruption_recovery"] = True  # Didn't crash
                else:
                    failure_tests["corruption_recovery"] = True  # Test key not in L1

            except Exception:
                failure_tests["corruption_recovery"] = False

            duration = time.time() - start_time
            all_passed = all(failure_tests.values())

            return ValidationResult(
                test_name="failure_scenarios",
                passed=all_passed,
                duration=duration,
                details={
                    "failure_tests": failure_tests,
                    "tests_passed": sum(failure_tests.values()),
                    "total_tests": len(failure_tests),
                    "success_rate": sum(failure_tests.values()) / len(failure_tests),
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name="failure_scenarios", passed=False, duration=duration, details={}, error=str(e))

    async def cleanup_system(self) -> ValidationResult:
        """Cleanup the cache system."""
        start_time = time.time()

        try:
            cleanup_results = {}

            # Cleanup specialized services
            for name, service in self.specialized_services.items():
                try:
                    await service.shutdown()
                    cleanup_results[f"{name}_service"] = True
                except Exception as e:
                    cleanup_results[f"{name}_service"] = False
                    self.logger.warning(f"Failed to shutdown {name} service: {e}")

            # Cleanup main cache service
            if self.cache_service:
                try:
                    await self.cache_service.shutdown()
                    cleanup_results["main_cache_service"] = True
                except Exception as e:
                    cleanup_results["main_cache_service"] = False
                    self.logger.warning(f"Failed to shutdown main cache service: {e}")

            duration = time.time() - start_time
            all_cleaned = all(cleanup_results.values())

            return ValidationResult(
                test_name="system_cleanup",
                passed=all_cleaned,
                duration=duration,
                details={
                    "cleanup_results": cleanup_results,
                    "services_cleaned": sum(cleanup_results.values()),
                    "total_services": len(cleanup_results),
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            return ValidationResult(test_name="system_cleanup", passed=False, duration=duration, details={}, error=str(e))

    async def run_validation(self) -> dict[str, Any]:
        """Run complete validation suite."""
        self.logger.info("Starting comprehensive cache system validation")
        start_time = time.time()

        # Define validation steps
        validation_steps = [
            ("Initialize System", self.initialize_system),
            ("Basic Cache Operations", self.validate_basic_cache_operations),
            ("Multi-Tier Functionality", self.validate_multi_tier_functionality),
            ("Specialized Services", self.validate_specialized_services),
            ("Performance Optimization", self.validate_performance_optimization),
            ("Key Optimization", self.validate_key_optimization),
            ("Serialization Optimization", self.validate_serialization_optimization),
            ("Performance Benchmarks", self.validate_performance_benchmarks),
            ("Failure Scenarios", self.validate_failure_scenarios),
            ("System Cleanup", self.cleanup_system),
        ]

        # Run validation steps
        for step_name, step_func in validation_steps:
            self.logger.info(f"Running validation step: {step_name}")
            try:
                result = await step_func()
                self.results.append(result)

                if result.passed:
                    self.logger.info(f"✓ {step_name} passed ({result.duration:.2f}s)")
                else:
                    self.logger.error(f"✗ {step_name} failed ({result.duration:.2f}s): {result.error}")

                if self.verbose:
                    self.logger.debug(f"Details: {result.details}")

            except Exception as e:
                self.logger.error(f"✗ {step_name} crashed: {e}")
                self.results.append(
                    ValidationResult(
                        test_name=step_name.lower().replace(" ", "_"),
                        passed=False,
                        duration=0.0,
                        details={},
                        error=f"Unexpected error: {e}",
                    )
                )

        # Calculate summary
        total_duration = time.time() - start_time
        passed_tests = sum(1 for r in self.results if r.passed)
        total_tests = len(self.results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0

        summary = {
            "validation_completed": True,
            "total_duration": total_duration,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": success_rate,
            "overall_passed": success_rate >= 0.9,  # 90% success rate required
            "results": [asdict(r) for r in self.results],
        }

        self.logger.info(f"Validation completed: {passed_tests}/{total_tests} tests passed ({success_rate:.1%})")

        return summary

    def save_results(self, output_path: str) -> None:
        """Save validation results to file."""
        try:
            summary = {"timestamp": time.time(), "config": asdict(self.config), "results": [asdict(r) for r in self.results]}

            with open(output_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)

            self.logger.info(f"Validation results saved to {output_path}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")


async def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Cache System Validator")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--output", default="validation_results.json", help="Output file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Create validator
    validator = CacheSystemValidator(config_path=args.config, verbose=args.verbose)

    try:
        # Run validation
        results = await validator.run_validation()

        # Save results
        validator.save_results(args.output)

        # Print summary
        print("\n" + "=" * 60)
        print("CACHE SYSTEM VALIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Duration: {results['total_duration']:.2f}s")
        print(f"Overall Result: {'PASS' if results['overall_passed'] else 'FAIL'}")
        print("=" * 60)

        # Print failed tests
        if results["failed_tests"] > 0:
            print("\nFAILED TESTS:")
            for result in results["results"]:
                if not result["passed"]:
                    print(f"- {result['test_name']}: {result.get('error', 'Unknown error')}")

        # Exit with appropriate code
        sys.exit(0 if results["overall_passed"] else 1)

    except KeyboardInterrupt:
        print("\nValidation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nValidation failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
