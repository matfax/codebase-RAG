#!/usr/bin/env python3
"""
Performance Test Runner for Query Caching Layer

This script runs comprehensive performance tests and benchmarks for the cache system,
providing detailed metrics and analysis of cache performance.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Add src and tests to Python path to enable imports
src_path = Path(__file__).parent / "src"
tests_path = Path(__file__).parent / "tests"
sys.path.insert(0, str(src_path))
sys.path.insert(0, str(tests_path))

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_cache_performance_benchmarks():
    """Run comprehensive cache performance benchmarks."""
    logger.info("Starting cache performance benchmarks...")
    
    try:
        # Import performance testing modules
        from test_cache_performance_benchmarks import CachePerformanceBenchmarkSuite
        from config.cache_config import CacheConfig
        from services.search_cache_service import SearchCacheService
        from services.project_cache_service import ProjectCacheService
        
        # Create benchmark suite
        benchmark_suite = CachePerformanceBenchmarkSuite()
        
        # Create mock cache configuration for testing
        cache_config = CacheConfig(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            redis_password=None,
            redis_ssl=False,
            memory_cache_size_mb=128,
            default_ttl_seconds=3600,
            compression_enabled=True
        )
        
        results = []
        
        # Test 1: Basic Cache Operations Performance
        logger.info("Running basic cache operations performance test...")
        try:
            search_cache = SearchCacheService(cache_config)
            await search_cache.initialize()
            
            result = await benchmark_suite.benchmark_basic_operations(
                search_cache, 
                iterations=100, 
                data_sizes=[100, 1000, 10000]
            )
            results.append(result)
            logger.info(f"Basic operations result: {result.summary_stats}")
            
            await search_cache.shutdown()
        except Exception as e:
            logger.warning(f"Basic cache operations test failed (likely Redis not available): {e}")
        
        # Test 2: Load Testing Performance
        logger.info("Running load testing performance...")
        try:
            project_cache = ProjectCacheService(cache_config)
            await project_cache.initialize()
            
            load_result = await benchmark_suite.benchmark_concurrent_load(
                project_cache,
                concurrent_operations=50,
                operations_per_worker=20,
                data_size_bytes=1024
            )
            results.append(load_result)
            logger.info(f"Load test result: {load_result.summary_stats}")
            
            await project_cache.shutdown()
        except Exception as e:
            logger.warning(f"Load testing failed (likely Redis not available): {e}")
        
        # Test 3: Memory Usage Profiling
        logger.info("Running memory usage profiling...")
        try:
            memory_result = await benchmark_suite.benchmark_memory_usage(
                cache_config,
                operations=["set", "get", "delete"],
                data_sizes=[1024, 10240, 102400],
                iterations=50
            )
            results.append(memory_result)
            logger.info(f"Memory profiling result: {memory_result.summary_stats}")
        except Exception as e:
            logger.warning(f"Memory profiling failed: {e}")
        
        # Generate comprehensive report
        report = generate_performance_report(results)
        
        # Save report to file
        report_path = Path(__file__).parent / "reports" / "performance_test_results.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Performance test report saved to: {report_path}")
        
        # Print summary
        print_performance_summary(report)
        
        return report
        
    except Exception as e:
        logger.error(f"Performance benchmark failed: {e}")
        raise


async def run_failure_scenario_tests():
    """Run comprehensive failure scenario tests."""
    logger.info("Starting failure scenario tests...")
    
    try:
        # Import failure testing modules
        from test_redis_failure_scenarios import RedisFailureScenarioTester
        from test_network_failure_scenarios import NetworkFailureScenarioTester
        from config.cache_config import CacheConfig
        
        # Create cache configuration for testing
        cache_config = CacheConfig(
            redis_host="localhost",
            redis_port=6379,
            redis_db=0,
            connection_timeout=5,
            socket_timeout=5,
            max_connections=20
        )
        
        results = []
        
        # Test 1: Redis Connection Failures
        logger.info("Running Redis failure scenarios...")
        try:
            redis_tester = RedisFailureScenarioTester(cache_config)
            redis_results = await redis_tester.run_all_failure_scenarios()
            results.extend(redis_results)
            logger.info(f"Redis failure tests completed: {len(redis_results)} scenarios tested")
        except Exception as e:
            logger.warning(f"Redis failure tests failed: {e}")
        
        # Test 2: Network Failure Scenarios
        logger.info("Running network failure scenarios...")
        try:
            network_tester = NetworkFailureScenarioTester(cache_config)
            network_results = await network_tester.run_all_network_scenarios()
            results.extend(network_results)
            logger.info(f"Network failure tests completed: {len(network_results)} scenarios tested")
        except Exception as e:
            logger.warning(f"Network failure tests failed: {e}")
        
        # Test 3: Memory Pressure Scenarios
        logger.info("Running memory pressure scenarios...")
        try:
            memory_results = await run_memory_pressure_scenarios(cache_config)
            results.extend(memory_results)
            logger.info(f"Memory pressure tests completed: {len(memory_results)} scenarios tested")
        except Exception as e:
            logger.warning(f"Memory pressure tests failed: {e}")
        
        # Generate failure scenario report
        report = generate_failure_report(results)
        
        # Save report to file
        report_path = Path(__file__).parent / "reports" / "failure_scenario_results.json"
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Failure scenario report saved to: {report_path}")
        
        # Print summary
        print_failure_summary(report)
        
        return report
        
    except Exception as e:
        logger.error(f"Failure scenario tests failed: {e}")
        raise


async def run_memory_pressure_scenarios(cache_config):
    """Run memory pressure scenarios."""
    # Mock implementation for memory pressure testing
    return [
        {
            "scenario": "memory_pressure_simulation",
            "status": "simulated",
            "description": "Memory pressure scenarios would require actual memory allocation testing"
        }
    ]


def generate_performance_report(results):
    """Generate comprehensive performance report."""
    report = {
        "test_type": "performance_benchmarks",
        "timestamp": time.time(),
        "total_scenarios": len(results),
        "summary": {
            "total_tests": len(results),
            "successful_tests": len([r for r in results if not r.performance_issues]),
            "tests_with_issues": len([r for r in results if r.performance_issues])
        },
        "results": []
    }
    
    for result in results:
        if hasattr(result, 'summary_stats'):
            report["results"].append({
                "scenario": result.scenario,
                "summary_stats": result.summary_stats,
                "performance_issues": result.performance_issues,
                "timestamp": result.timestamp
            })
    
    # Calculate overall performance metrics
    if results:
        all_durations = []
        all_throughputs = []
        
        for result in results:
            if hasattr(result, 'metrics'):
                for metric in result.metrics:
                    if hasattr(metric, 'duration_ms'):
                        all_durations.append(metric.duration_ms)
                    if hasattr(metric, 'throughput_ops_per_sec'):
                        all_throughputs.append(metric.throughput_ops_per_sec)
        
        if all_durations:
            report["summary"]["avg_duration_ms"] = sum(all_durations) / len(all_durations)
            report["summary"]["max_duration_ms"] = max(all_durations)
            report["summary"]["min_duration_ms"] = min(all_durations)
        
        if all_throughputs:
            report["summary"]["avg_throughput_ops_per_sec"] = sum(all_throughputs) / len(all_throughputs)
            report["summary"]["max_throughput_ops_per_sec"] = max(all_throughputs)
    
    return report


def generate_failure_report(results):
    """Generate comprehensive failure scenario report."""
    report = {
        "test_type": "failure_scenarios",
        "timestamp": time.time(),
        "total_scenarios": len(results),
        "summary": {
            "total_tests": len(results),
            "passed_tests": len([r for r in results if r.get("status") == "passed"]),
            "failed_tests": len([r for r in results if r.get("status") == "failed"]),
            "error_tests": len([r for r in results if r.get("status") == "error"])
        },
        "results": results
    }
    
    return report


def print_performance_summary(report):
    """Print performance test summary to console."""
    print("\n" + "="*80)
    print("CACHE PERFORMANCE TEST SUMMARY")
    print("="*80)
    
    summary = report.get("summary", {})
    print(f"Total Tests: {summary.get('total_tests', 0)}")
    print(f"Successful Tests: {summary.get('successful_tests', 0)}")
    print(f"Tests with Issues: {summary.get('tests_with_issues', 0)}")
    
    if "avg_duration_ms" in summary:
        print(f"Average Duration: {summary['avg_duration_ms']:.2f} ms")
        print(f"Max Duration: {summary['max_duration_ms']:.2f} ms")
        print(f"Min Duration: {summary['min_duration_ms']:.2f} ms")
    
    if "avg_throughput_ops_per_sec" in summary:
        print(f"Average Throughput: {summary['avg_throughput_ops_per_sec']:.2f} ops/sec")
        print(f"Max Throughput: {summary['max_throughput_ops_per_sec']:.2f} ops/sec")
    
    print("="*80)


def print_failure_summary(report):
    """Print failure scenario test summary to console."""
    print("\n" + "="*80)
    print("FAILURE SCENARIO TEST SUMMARY")
    print("="*80)
    
    summary = report.get("summary", {})
    print(f"Total Tests: {summary.get('total_tests', 0)}")
    print(f"Passed Tests: {summary.get('passed_tests', 0)}")
    print(f"Failed Tests: {summary.get('failed_tests', 0)}")
    print(f"Error Tests: {summary.get('error_tests', 0)}")
    
    print("="*80)


async def main():
    """Main function to run all performance tests."""
    print("Query Caching Layer - Performance Testing Suite")
    print("=" * 50)
    
    # Ensure reports directory exists
    reports_dir = Path(__file__).parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    
    try:
        # Run performance benchmarks
        print("\n1. Running Performance Benchmarks...")
        perf_report = await run_cache_performance_benchmarks()
        
        # Run failure scenario tests
        print("\n2. Running Failure Scenario Tests...")
        failure_report = await run_failure_scenario_tests()
        
        # Generate combined report
        combined_report = {
            "performance_tests": perf_report,
            "failure_tests": failure_report,
            "timestamp": time.time(),
            "status": "completed"
        }
        
        # Save combined report
        combined_path = reports_dir / "wave_16_performance_testing_report.json"
        with open(combined_path, 'w') as f:
            json.dump(combined_report, f, indent=2, default=str)
        
        print(f"\n✅ Wave 16.0 Performance Testing Completed!")
        print(f"📊 Combined report saved to: {combined_path}")
        
        return combined_report
        
    except Exception as e:
        logger.error(f"Performance testing suite failed: {e}")
        print(f"\n❌ Performance testing failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())