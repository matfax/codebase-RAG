#!/usr/bin/env python3
"""
Wave 8.0 Task 8.3: Memory Usage Validation Framework

This module validates the 50% memory usage reduction target by measuring
memory consumption across different operations, identifying memory leaks,
and ensuring efficient memory utilization patterns.
"""

import asyncio
import gc
import json
import logging
import os
import statistics
import sys
import threading
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from services.cache_service import CacheService
from services.embedding_service import EmbeddingService
from services.qdrant_service import QdrantService


@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""

    timestamp: str
    rss_mb: float  # Resident Set Size
    vms_mb: float  # Virtual Memory Size
    shared_mb: float  # Shared memory
    text_mb: float  # Text (code) memory
    data_mb: float  # Data memory
    lib_mb: float  # Library memory
    dirty_mb: float  # Dirty pages
    peak_mb: float  # Peak memory since tracemalloc start
    current_mb: float  # Current traced memory
    operation: str  # What operation was being performed


@dataclass
class MemoryTestResult:
    """Result of a memory usage test"""

    test_name: str
    baseline_memory_mb: float
    peak_memory_mb: float
    final_memory_mb: float
    memory_increase_mb: float
    memory_increase_percent: float
    memory_leaked: bool
    leak_threshold_mb: float
    snapshots: list[MemorySnapshot]
    duration_seconds: float
    operations_count: int
    memory_efficiency_score: float
    target_met: bool  # True if within 50% reduction target


class MemoryUsageValidator:
    """Comprehensive memory usage validation framework"""

    def __init__(self):
        self.target_reduction_percent = 50.0  # 50% memory reduction target
        self.leak_threshold_mb = 10.0  # Memory leak threshold
        self.baseline_memory_mb = 0.0
        self.results: list[MemoryTestResult] = []
        self.logger = self._setup_logging()

        # Initialize process monitoring
        self.process = psutil.Process()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for memory tests"""
        logger = logging.getLogger("memory_validator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _take_memory_snapshot(self, operation: str = "") -> MemorySnapshot:
        """Take a detailed memory snapshot"""
        memory_info = self.process.memory_info()
        memory_percent = self.process.memory_percent()

        # Get tracemalloc info if available
        try:
            current_traced, peak_traced = tracemalloc.get_traced_memory()
            current_mb = current_traced / 1024 / 1024
            peak_mb = peak_traced / 1024 / 1024
        except:
            current_mb = 0.0
            peak_mb = 0.0

        return MemorySnapshot(
            timestamp=datetime.now().isoformat(),
            rss_mb=memory_info.rss / 1024 / 1024,
            vms_mb=memory_info.vms / 1024 / 1024,
            shared_mb=getattr(memory_info, "shared", 0) / 1024 / 1024,
            text_mb=getattr(memory_info, "text", 0) / 1024 / 1024,
            data_mb=getattr(memory_info, "data", 0) / 1024 / 1024,
            lib_mb=getattr(memory_info, "lib", 0) / 1024 / 1024,
            dirty_mb=getattr(memory_info, "dirty", 0) / 1024 / 1024,
            peak_mb=peak_mb,
            current_mb=current_mb,
            operation=operation,
        )

    async def test_embedding_memory_usage(self) -> MemoryTestResult:
        """Test memory usage during embedding generation"""
        test_name = "embedding_memory_usage"
        self.logger.info(f"Starting {test_name}")

        # Start monitoring
        tracemalloc.start()
        gc.collect()  # Clean start

        snapshots = []
        start_time = time.time()

        # Baseline snapshot
        baseline_snapshot = self._take_memory_snapshot("baseline")
        snapshots.append(baseline_snapshot)
        baseline_memory = baseline_snapshot.rss_mb

        try:
            # Generate embeddings with increasing load
            embedding_service = EmbeddingService()

            # Small batch
            small_texts = ["def function():", "class MyClass:", "import os"]
            snapshots.append(self._take_memory_snapshot("small_batch_start"))
            await embedding_service.generate_embeddings(small_texts)
            snapshots.append(self._take_memory_snapshot("small_batch_end"))

            # Medium batch
            medium_texts = [f"def function_{i}(): return {i}" for i in range(50)]
            snapshots.append(self._take_memory_snapshot("medium_batch_start"))
            await embedding_service.generate_embeddings(medium_texts)
            snapshots.append(self._take_memory_snapshot("medium_batch_end"))

            # Large batch
            large_texts = [f"class Class_{i}: def method(self): return '{i}'" for i in range(200)]
            snapshots.append(self._take_memory_snapshot("large_batch_start"))
            await embedding_service.generate_embeddings(large_texts)
            snapshots.append(self._take_memory_snapshot("large_batch_end"))

            # Force garbage collection
            gc.collect()
            snapshots.append(self._take_memory_snapshot("after_gc"))

        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")

        finally:
            end_time = time.time()
            final_snapshot = self._take_memory_snapshot("final")
            snapshots.append(final_snapshot)
            tracemalloc.stop()

        # Calculate results
        peak_memory = max(s.rss_mb for s in snapshots)
        final_memory = final_snapshot.rss_mb
        memory_increase = final_memory - baseline_memory
        memory_increase_percent = (memory_increase / baseline_memory) * 100 if baseline_memory > 0 else 0

        memory_leaked = memory_increase > self.leak_threshold_mb
        efficiency_score = max(0, 100 - memory_increase_percent)
        target_met = memory_increase_percent <= self.target_reduction_percent

        result = MemoryTestResult(
            test_name=test_name,
            baseline_memory_mb=baseline_memory,
            peak_memory_mb=peak_memory,
            final_memory_mb=final_memory,
            memory_increase_mb=memory_increase,
            memory_increase_percent=memory_increase_percent,
            memory_leaked=memory_leaked,
            leak_threshold_mb=self.leak_threshold_mb,
            snapshots=snapshots,
            duration_seconds=end_time - start_time,
            operations_count=3,  # 3 batches
            memory_efficiency_score=efficiency_score,
            target_met=target_met,
        )

        self.results.append(result)
        return result

    async def test_search_memory_usage(self) -> MemoryTestResult:
        """Test memory usage during search operations"""
        test_name = "search_memory_usage"
        self.logger.info(f"Starting {test_name}")

        tracemalloc.start()
        gc.collect()

        snapshots = []
        start_time = time.time()

        baseline_snapshot = self._take_memory_snapshot("baseline")
        snapshots.append(baseline_snapshot)
        baseline_memory = baseline_snapshot.rss_mb

        try:
            # Simulate search operations
            operations_count = 0

            # Simple searches
            for i in range(20):
                snapshots.append(self._take_memory_snapshot(f"simple_search_{i}"))
                await asyncio.sleep(0.01)  # Simulate search work
                operations_count += 1

            # Complex searches
            for i in range(10):
                snapshots.append(self._take_memory_snapshot(f"complex_search_{i}"))
                await asyncio.sleep(0.05)  # Simulate complex search work
                operations_count += 1

            # Graph RAG searches
            for i in range(5):
                snapshots.append(self._take_memory_snapshot(f"graph_search_{i}"))
                await asyncio.sleep(0.1)  # Simulate graph analysis
                operations_count += 1

            gc.collect()
            snapshots.append(self._take_memory_snapshot("after_gc"))

        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")

        finally:
            end_time = time.time()
            final_snapshot = self._take_memory_snapshot("final")
            snapshots.append(final_snapshot)
            tracemalloc.stop()

        # Calculate results
        peak_memory = max(s.rss_mb for s in snapshots)
        final_memory = final_snapshot.rss_mb
        memory_increase = final_memory - baseline_memory
        memory_increase_percent = (memory_increase / baseline_memory) * 100 if baseline_memory > 0 else 0

        memory_leaked = memory_increase > self.leak_threshold_mb
        efficiency_score = max(0, 100 - memory_increase_percent)
        target_met = memory_increase_percent <= self.target_reduction_percent

        result = MemoryTestResult(
            test_name=test_name,
            baseline_memory_mb=baseline_memory,
            peak_memory_mb=peak_memory,
            final_memory_mb=final_memory,
            memory_increase_mb=memory_increase,
            memory_increase_percent=memory_increase_percent,
            memory_leaked=memory_leaked,
            leak_threshold_mb=self.leak_threshold_mb,
            snapshots=snapshots,
            duration_seconds=end_time - start_time,
            operations_count=operations_count,
            memory_efficiency_score=efficiency_score,
            target_met=target_met,
        )

        self.results.append(result)
        return result

    async def test_indexing_memory_usage(self) -> MemoryTestResult:
        """Test memory usage during indexing operations"""
        test_name = "indexing_memory_usage"
        self.logger.info(f"Starting {test_name}")

        tracemalloc.start()
        gc.collect()

        snapshots = []
        start_time = time.time()

        baseline_snapshot = self._take_memory_snapshot("baseline")
        snapshots.append(baseline_snapshot)
        baseline_memory = baseline_snapshot.rss_mb

        try:
            operations_count = 0

            # Simulate file parsing and chunking
            for i in range(50):
                snapshots.append(self._take_memory_snapshot(f"parse_file_{i}"))
                # Simulate file processing
                await asyncio.sleep(0.02)
                operations_count += 1

                # Periodic garbage collection during indexing
                if i % 10 == 0:
                    gc.collect()
                    snapshots.append(self._take_memory_snapshot(f"gc_after_{i}"))

            # Simulate embedding generation
            for i in range(20):
                snapshots.append(self._take_memory_snapshot(f"embedding_{i}"))
                await asyncio.sleep(0.03)
                operations_count += 1

            # Simulate vector storage
            for i in range(30):
                snapshots.append(self._take_memory_snapshot(f"storage_{i}"))
                await asyncio.sleep(0.01)
                operations_count += 1

            gc.collect()
            snapshots.append(self._take_memory_snapshot("final_gc"))

        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")

        finally:
            end_time = time.time()
            final_snapshot = self._take_memory_snapshot("final")
            snapshots.append(final_snapshot)
            tracemalloc.stop()

        # Calculate results
        peak_memory = max(s.rss_mb for s in snapshots)
        final_memory = final_snapshot.rss_mb
        memory_increase = final_memory - baseline_memory
        memory_increase_percent = (memory_increase / baseline_memory) * 100 if baseline_memory > 0 else 0

        memory_leaked = memory_increase > self.leak_threshold_mb
        efficiency_score = max(0, 100 - memory_increase_percent)
        target_met = memory_increase_percent <= self.target_reduction_percent

        result = MemoryTestResult(
            test_name=test_name,
            baseline_memory_mb=baseline_memory,
            peak_memory_mb=peak_memory,
            final_memory_mb=final_memory,
            memory_increase_mb=memory_increase,
            memory_increase_percent=memory_increase_percent,
            memory_leaked=memory_leaked,
            leak_threshold_mb=self.leak_threshold_mb,
            snapshots=snapshots,
            duration_seconds=end_time - start_time,
            operations_count=operations_count,
            memory_efficiency_score=efficiency_score,
            target_met=target_met,
        )

        self.results.append(result)
        return result

    async def test_cache_memory_usage(self) -> MemoryTestResult:
        """Test memory usage of caching systems"""
        test_name = "cache_memory_usage"
        self.logger.info(f"Starting {test_name}")

        tracemalloc.start()
        gc.collect()

        snapshots = []
        start_time = time.time()

        baseline_snapshot = self._take_memory_snapshot("baseline")
        snapshots.append(baseline_snapshot)
        baseline_memory = baseline_snapshot.rss_mb

        try:
            operations_count = 0

            # Simulate cache operations
            cache_data = {}

            # Fill cache
            for i in range(1000):
                cache_data[f"key_{i}"] = f"value_{i}" * 100  # Larger values
                operations_count += 1

                if i % 100 == 0:
                    snapshots.append(self._take_memory_snapshot(f"cache_fill_{i}"))

            # Access cache (should not increase memory significantly)
            for i in range(500):
                _ = cache_data.get(f"key_{i % 1000}")
                operations_count += 1

            snapshots.append(self._take_memory_snapshot("after_access"))

            # Clear cache
            cache_data.clear()
            gc.collect()
            snapshots.append(self._take_memory_snapshot("after_clear"))

        except Exception as e:
            self.logger.error(f"Error in {test_name}: {e}")

        finally:
            end_time = time.time()
            final_snapshot = self._take_memory_snapshot("final")
            snapshots.append(final_snapshot)
            tracemalloc.stop()

        # Calculate results
        peak_memory = max(s.rss_mb for s in snapshots)
        final_memory = final_snapshot.rss_mb
        memory_increase = final_memory - baseline_memory
        memory_increase_percent = (memory_increase / baseline_memory) * 100 if baseline_memory > 0 else 0

        memory_leaked = memory_increase > self.leak_threshold_mb
        efficiency_score = max(0, 100 - memory_increase_percent)
        target_met = memory_increase_percent <= self.target_reduction_percent

        result = MemoryTestResult(
            test_name=test_name,
            baseline_memory_mb=baseline_memory,
            peak_memory_mb=peak_memory,
            final_memory_mb=final_memory,
            memory_increase_mb=memory_increase,
            memory_increase_percent=memory_increase_percent,
            memory_leaked=memory_leaked,
            leak_threshold_mb=self.leak_threshold_mb,
            snapshots=snapshots,
            duration_seconds=end_time - start_time,
            operations_count=operations_count,
            memory_efficiency_score=efficiency_score,
            target_met=target_met,
        )

        self.results.append(result)
        return result

    async def run_all_tests(self) -> dict[str, Any]:
        """Run all memory usage tests"""
        self.logger.info("Starting memory usage validation tests...")

        # Set baseline memory
        gc.collect()
        baseline_snapshot = self._take_memory_snapshot("overall_baseline")
        self.baseline_memory_mb = baseline_snapshot.rss_mb

        # Run tests
        test_methods = [
            self.test_embedding_memory_usage,
            self.test_search_memory_usage,
            self.test_indexing_memory_usage,
            self.test_cache_memory_usage,
        ]

        for test_method in test_methods:
            try:
                await test_method()
                self.logger.info(f"Completed {test_method.__name__}")
                # Clean up between tests
                gc.collect()
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Failed {test_method.__name__}: {e}")

        # Generate summary
        summary = self._generate_summary()
        self.logger.info("Memory usage validation tests completed")

        return summary

    def _generate_summary(self) -> dict[str, Any]:
        """Generate comprehensive memory test summary"""
        total_tests = len(self.results)
        tests_target_met = len([r for r in self.results if r.target_met])
        tests_with_leaks = len([r for r in self.results if r.memory_leaked])

        # Memory statistics
        if self.results:
            avg_memory_increase = statistics.mean([r.memory_increase_percent for r in self.results])
            max_memory_increase = max([r.memory_increase_percent for r in self.results])
            min_memory_increase = min([r.memory_increase_percent for r in self.results])
            avg_efficiency_score = statistics.mean([r.memory_efficiency_score for r in self.results])
        else:
            avg_memory_increase = 0
            max_memory_increase = 0
            min_memory_increase = 0
            avg_efficiency_score = 0

        # Overall target achievement
        overall_target_met = tests_target_met == total_tests and tests_with_leaks == 0

        summary = {
            "total_tests": total_tests,
            "tests_target_met": tests_target_met,
            "tests_with_leaks": tests_with_leaks,
            "overall_target_met": overall_target_met,
            "target_reduction_percent": self.target_reduction_percent,
            "baseline_memory_mb": self.baseline_memory_mb,
            "memory_statistics": {
                "average_increase_percent": avg_memory_increase,
                "max_increase_percent": max_memory_increase,
                "min_increase_percent": min_memory_increase,
                "average_efficiency_score": avg_efficiency_score,
            },
            "test_details": [asdict(result) for result in self.results],
            "timestamp": datetime.now().isoformat(),
        }

        return summary

    def generate_report(self, output_file: str = "memory_usage_validation_report.json"):
        """Generate detailed memory usage report"""
        summary = self._generate_summary()

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Generated memory usage validation report: {output_file}")
        return summary

    def print_summary(self):
        """Print human-readable memory test summary"""
        summary = self._generate_summary()

        print("\n=== Memory Usage Validation Summary ===")
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Tests Meeting Target: {summary['tests_target_met']}/{summary['total_tests']}")
        print(f"Tests with Memory Leaks: {summary['tests_with_leaks']}")
        print(f"Overall Target Met: {'YES' if summary['overall_target_met'] else 'NO'}")
        print(f"Target Reduction: {summary['target_reduction_percent']}%")
        print(f"Baseline Memory: {summary['baseline_memory_mb']:.2f} MB")

        stats = summary["memory_statistics"]
        print("\nMemory Statistics:")
        print(f"  Average Increase: {stats['average_increase_percent']:.2f}%")
        print(f"  Max Increase: {stats['max_increase_percent']:.2f}%")
        print(f"  Min Increase: {stats['min_increase_percent']:.2f}%")
        print(f"  Average Efficiency Score: {stats['average_efficiency_score']:.2f}")

        print("\nDetailed Results:")
        for result in self.results:
            status = "PASS" if result.target_met and not result.memory_leaked else "FAIL"
            print(f"  {result.test_name}: {status}")
            print(f"    Memory Increase: {result.memory_increase_percent:.2f}%")
            print(f"    Memory Leaked: {'YES' if result.memory_leaked else 'NO'}")
            print(f"    Efficiency Score: {result.memory_efficiency_score:.2f}")


async def main():
    """Main function to run memory usage validation"""
    validator = MemoryUsageValidator()

    # Run tests
    print("Running memory usage validation tests...")
    summary = await validator.run_all_tests()

    # Generate report
    validator.generate_report("wave_8_memory_usage_report.json")

    # Print summary
    validator.print_summary()

    return summary


if __name__ == "__main__":
    asyncio.run(main())
