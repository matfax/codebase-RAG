# Cache Development and Extension Guide

## Overview

This comprehensive guide is designed for developers who want to extend, customize, or contribute to the Query Caching Layer system. It covers development workflows, architecture patterns, API design, testing strategies, and best practices for building robust cache extensions.

## Development Environment Setup

### Prerequisites

```bash
# Core development tools
Python >= 3.9
uv (Python package manager)
Docker >= 20.10
Docker Compose >= 2.0
Git >= 2.30
Redis >= 7.0

# Development dependencies
pytest >= 7.0
black >= 22.0
ruff >= 0.1.0
mypy >= 1.0
pre-commit >= 2.20
```

### Development Environment Configuration

```bash
# 1. Clone and setup repository
git clone <repository-url>
cd query-caching-layer
git checkout -b feature/your-feature-name

# 2. Install development dependencies
uv sync --dev

# 3. Setup pre-commit hooks
pre-commit install

# 4. Configure development environment
cp .env.example .env.development
# Edit .env.development with your settings

# 5. Start development services
docker-compose -f docker-compose.dev.yml up -d

# 6. Run tests to verify setup
uv run pytest tests/ -v
```

### IDE Configuration

#### VS Code Setup

```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": false,
    "python.linting.mypyEnabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.provider": "isort",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".mypy_cache": true,
        ".coverage": true
    }
}
```

```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Debug Cache Service",
            "type": "python",
            "request": "launch",
            "program": "src/main.py",
            "env": {
                "ENVIRONMENT": "development",
                "CACHE_DEBUG_MODE": "true"
            },
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        }
    ]
}
```

## Architecture Patterns for Extensions

### Cache Service Extension Pattern

```python
# src/services/extensions/custom_cache_service.py
"""Example of extending the base cache service."""

from typing import Any, Optional
from src.services.cache_service import BaseCacheService, CacheOperationError

class CustomCacheService(BaseCacheService):
    """Custom cache service with additional functionality."""

    def __init__(self, base_service: BaseCacheService, custom_config: dict):
        super().__init__(base_service.config)
        self.base_service = base_service
        self.custom_config = custom_config
        self.custom_features = {}

    async def initialize(self) -> None:
        """Initialize the custom cache service."""
        await self.base_service.initialize()
        await self._initialize_custom_features()

    async def _initialize_custom_features(self):
        """Initialize custom features."""
        # Example: Initialize custom analytics
        if self.custom_config.get("analytics_enabled", False):
            self.custom_features["analytics"] = AnalyticsCollector()
            await self.custom_features["analytics"].initialize()

    async def get(self, key: str) -> Any | None:
        """Enhanced get operation with custom logic."""
        # Pre-processing hook
        await self._before_get(key)

        try:
            # Delegate to base service
            result = await self.base_service.get(key)

            # Post-processing hook
            await self._after_get(key, result)

            return result

        except Exception as e:
            await self._on_error("get", key, e)
            raise

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Enhanced set operation with custom logic."""
        # Validation hook
        await self._validate_set_operation(key, value, ttl)

        # Pre-processing
        processed_value = await self._preprocess_value(value)

        try:
            result = await self.base_service.set(key, processed_value, ttl)

            # Custom post-processing
            if result:
                await self._on_successful_set(key, value, ttl)

            return result

        except Exception as e:
            await self._on_error("set", key, e)
            raise

    async def _before_get(self, key: str):
        """Hook called before get operation."""
        if "analytics" in self.custom_features:
            await self.custom_features["analytics"].record_access(key)

    async def _after_get(self, key: str, result: Any):
        """Hook called after get operation."""
        if "analytics" in self.custom_features:
            await self.custom_features["analytics"].record_result(key, result is not None)

    async def _validate_set_operation(self, key: str, value: Any, ttl: Optional[int]):
        """Validate set operation parameters."""
        # Example: Size validation
        if isinstance(value, str) and len(value) > self.custom_config.get("max_value_size", 1024*1024):
            raise CacheOperationError(f"Value size exceeds maximum allowed size")

        # Example: Key pattern validation
        if not self._is_valid_key_pattern(key):
            raise CacheOperationError(f"Invalid key pattern: {key}")

    async def _preprocess_value(self, value: Any) -> Any:
        """Preprocess value before caching."""
        # Example: Automatic compression
        if self.custom_config.get("auto_compress", False):
            return await self._compress_value(value)
        return value

    def _is_valid_key_pattern(self, key: str) -> bool:
        """Validate key pattern according to custom rules."""
        allowed_patterns = self.custom_config.get("allowed_key_patterns", [".*"])

        import re
        for pattern in allowed_patterns:
            if re.match(pattern, key):
                return True
        return False
```

### Cache Strategy Extension Pattern

```python
# src/strategies/custom_cache_strategy.py
"""Custom cache strategy implementation."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class CacheStrategy(ABC):
    """Abstract base class for cache strategies."""

    @abstractmethod
    async def should_cache(self, key: str, value: Any) -> bool:
        """Determine if a value should be cached."""
        pass

    @abstractmethod
    async def calculate_ttl(self, key: str, value: Any) -> Optional[int]:
        """Calculate TTL for a cache entry."""
        pass

    @abstractmethod
    async def should_invalidate(self, key: str, context: Dict) -> bool:
        """Determine if a cache entry should be invalidated."""
        pass

class SmartCacheStrategy(CacheStrategy):
    """Smart cache strategy based on access patterns and value characteristics."""

    def __init__(self, config: Dict):
        self.config = config
        self.access_patterns = {}
        self.value_characteristics = {}

    async def should_cache(self, key: str, value: Any) -> bool:
        """Intelligent caching decision based on multiple factors."""

        # Factor 1: Value size
        value_size = self._estimate_size(value)
        if value_size > self.config.get("max_cacheable_size", 1024*1024):
            return False

        # Factor 2: Historical access patterns
        access_pattern = self.access_patterns.get(key, {})
        access_frequency = access_pattern.get("frequency", 0)

        if access_frequency < self.config.get("min_access_frequency", 2):
            return False

        # Factor 3: Value volatility
        volatility = self._calculate_volatility(key, value)
        if volatility > self.config.get("max_volatility", 0.8):
            return False

        return True

    async def calculate_ttl(self, key: str, value: Any) -> Optional[int]:
        """Calculate optimal TTL based on access patterns and value type."""

        base_ttl = self.config.get("base_ttl", 3600)

        # Adjust based on access frequency
        access_pattern = self.access_patterns.get(key, {})
        access_frequency = access_pattern.get("frequency", 1)

        # Higher frequency = longer TTL
        frequency_multiplier = min(access_frequency / 10.0, 3.0)

        # Adjust based on value volatility
        volatility = self._calculate_volatility(key, value)
        volatility_multiplier = max(1.0 - volatility, 0.1)

        # Adjust based on value type
        type_multiplier = self._get_type_multiplier(value)

        optimal_ttl = int(base_ttl * frequency_multiplier * volatility_multiplier * type_multiplier)

        # Apply bounds
        min_ttl = self.config.get("min_ttl", 300)
        max_ttl = self.config.get("max_ttl", 86400)

        return max(min_ttl, min(optimal_ttl, max_ttl))

    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of a value."""
        import sys
        return sys.getsizeof(value)

    def _calculate_volatility(self, key: str, value: Any) -> float:
        """Calculate value volatility (how often it changes)."""
        characteristics = self.value_characteristics.get(key, {})
        change_frequency = characteristics.get("change_frequency", 0.5)
        return change_frequency

    def _get_type_multiplier(self, value: Any) -> float:
        """Get TTL multiplier based on value type."""
        if isinstance(value, dict) and "embedding" in str(value):
            return 2.0  # Embeddings are stable
        elif isinstance(value, list):
            return 0.8  # Lists might change more often
        else:
            return 1.0  # Default
```

### Cache Decorator Pattern

```python
# src/decorators/cache_decorators.py
"""Cache decorators for easy function caching."""

import asyncio
import functools
import hashlib
import json
from typing import Any, Callable, Optional

from src.services.cache_service import get_cache_service

def cache_result(
    ttl: Optional[int] = None,
    key_prefix: str = "",
    cache_name: str = "default",
    serialize_args: bool = True,
    ignore_args: Optional[list] = None
):
    """Decorator to cache function results."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = _generate_cache_key(
                func, args, kwargs, key_prefix, serialize_args, ignore_args
            )

            # Get cache service
            cache_service = await get_cache_service()

            # Try to get cached result
            cached_result = await cache_service.get(cache_key)
            if cached_result is not None:
                return _deserialize_result(cached_result)

            # Execute function
            result = await func(*args, **kwargs)

            # Cache the result
            serialized_result = _serialize_result(result)
            await cache_service.set(cache_key, serialized_result, ttl)

            return result

        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For synchronous functions
            return asyncio.run(async_wrapper(*args, **kwargs))

        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper

    return decorator

def invalidate_cache(
    pattern: str = "",
    cache_name: str = "default"
):
    """Decorator to invalidate cache entries after function execution."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Execute function first
            result = await func(*args, **kwargs)

            # Invalidate cache
            cache_service = await get_cache_service()

            if pattern:
                # Pattern-based invalidation
                await _invalidate_pattern(cache_service, pattern, args, kwargs)
            else:
                # Invalidate based on function name
                function_pattern = f"{func.__module__}.{func.__name__}:*"
                await _invalidate_pattern(cache_service, function_pattern, args, kwargs)

            return result

        return wrapper

    return decorator

def _generate_cache_key(
    func: Callable,
    args: tuple,
    kwargs: dict,
    key_prefix: str,
    serialize_args: bool,
    ignore_args: Optional[list]
) -> str:
    """Generate cache key for function call."""

    # Base key from function
    base_key = f"{func.__module__}.{func.__name__}"

    if key_prefix:
        base_key = f"{key_prefix}:{base_key}"

    if not serialize_args:
        return base_key

    # Filter arguments
    filtered_args = args
    filtered_kwargs = kwargs

    if ignore_args:
        # Remove ignored arguments
        filtered_kwargs = {k: v for k, v in kwargs.items() if k not in ignore_args}

    # Serialize arguments
    args_str = json.dumps([filtered_args, filtered_kwargs], sort_keys=True, default=str)
    args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]

    return f"{base_key}:{args_hash}"

# Usage examples
@cache_result(ttl=3600, key_prefix="embeddings")
async def generate_embedding(text: str, model: str = "default") -> list[float]:
    """Generate embedding with caching."""
    # Expensive embedding generation
    return [0.1, 0.2, 0.3]  # Placeholder

@cache_result(ttl=1800, ignore_args=["user_id"])
async def search_documents(query: str, filters: dict, user_id: str) -> list[dict]:
    """Search documents with caching (ignoring user_id)."""
    # Expensive search operation
    return [{"doc": "result"}]  # Placeholder

@invalidate_cache(pattern="embeddings:*")
async def update_embedding_model(model_path: str):
    """Update embedding model and invalidate related caches."""
    # Model update logic
    pass
```

## Creating Custom Cache Tools

### MCP Tool Development Pattern

```python
# src/tools/custom/advanced_cache_tools.py
"""Custom MCP tools for advanced cache operations."""

import asyncio
from typing import Any, Dict, List, Optional

from mcp.server import Server
from mcp.types import Resource, Tool

from src.services.cache_service import get_cache_service

class AdvancedCacheTools:
    """Advanced cache management tools."""

    def __init__(self, server: Server):
        self.server = server
        self._register_tools()

    def _register_tools(self):
        """Register custom cache tools."""

        @self.server.tool()
        async def cache_bulk_operations(
            operations: List[Dict[str, Any]]
        ) -> Dict[str, Any]:
            """
            Execute bulk cache operations efficiently.

            Args:
                operations: List of operations [{"op": "get|set|delete", "key": "...", "value": "...", "ttl": 3600}]

            Returns:
                Dict with operation results and performance metrics
            """
            cache_service = await get_cache_service()

            start_time = time.perf_counter()
            results = []

            # Group operations by type for optimization
            get_keys = []
            set_operations = {}
            delete_keys = []

            for op in operations:
                if op["op"] == "get":
                    get_keys.append(op["key"])
                elif op["op"] == "set":
                    set_operations[op["key"]] = {
                        "value": op["value"],
                        "ttl": op.get("ttl")
                    }
                elif op["op"] == "delete":
                    delete_keys.append(op["key"])

            # Execute bulk operations
            if get_keys:
                get_results = await cache_service.get_batch(get_keys)
                for key in get_keys:
                    results.append({
                        "operation": "get",
                        "key": key,
                        "result": get_results.get(key),
                        "success": True
                    })

            if set_operations:
                set_items = {k: v["value"] for k, v in set_operations.items()}
                set_results = await cache_service.set_batch(set_items)
                for key, success in set_results.items():
                    results.append({
                        "operation": "set",
                        "key": key,
                        "success": success
                    })

            if delete_keys:
                delete_results = await cache_service.delete_batch(delete_keys)
                for key, success in delete_results.items():
                    results.append({
                        "operation": "delete",
                        "key": key,
                        "success": success
                    })

            execution_time = time.perf_counter() - start_time

            return {
                "results": results,
                "performance": {
                    "total_operations": len(operations),
                    "execution_time_ms": execution_time * 1000,
                    "operations_per_second": len(operations) / execution_time
                }
            }

        @self.server.tool()
        async def cache_pattern_analysis(
            key_pattern: str = "*",
            max_keys: int = 1000
        ) -> Dict[str, Any]:
            """
            Analyze cache key patterns and usage statistics.

            Args:
                key_pattern: Pattern to match keys (Redis glob pattern)
                max_keys: Maximum number of keys to analyze

            Returns:
                Dict with pattern analysis and recommendations
            """
            cache_service = await get_cache_service()

            # Get keys matching pattern
            keys = []
            if hasattr(cache_service, 'get_redis_client'):
                async with cache_service.get_redis_client() as redis:
                    count = 0
                    async for key in redis.scan_iter(match=key_pattern):
                        if count >= max_keys:
                            break
                        keys.append(key.decode())
                        count += 1

            # Analyze patterns
            analysis = {
                "total_keys": len(keys),
                "pattern_breakdown": {},
                "size_distribution": {},
                "ttl_distribution": {},
                "recommendations": []
            }

            # Pattern breakdown
            for key in keys:
                parts = key.split(":")
                if len(parts) >= 2:
                    pattern = ":".join(parts[:2])
                    analysis["pattern_breakdown"][pattern] = analysis["pattern_breakdown"].get(pattern, 0) + 1

            # Generate recommendations
            if len(keys) > max_keys * 0.8:
                analysis["recommendations"].append({
                    "type": "performance",
                    "message": f"Large number of keys ({len(keys)}) - consider key cleanup"
                })

            return analysis

        @self.server.tool()
        async def cache_performance_tuning(
            test_duration_seconds: int = 60
        ) -> Dict[str, Any]:
            """
            Run performance tests and provide tuning recommendations.

            Args:
                test_duration_seconds: Duration of performance test

            Returns:
                Dict with performance metrics and tuning recommendations
            """
            cache_service = await get_cache_service()

            # Run performance tests
            results = await self._run_performance_tests(cache_service, test_duration_seconds)

            # Generate recommendations
            recommendations = self._generate_tuning_recommendations(results)

            return {
                "performance_metrics": results,
                "recommendations": recommendations,
                "test_duration": test_duration_seconds
            }

    async def _run_performance_tests(self, cache_service, duration: int) -> Dict[str, Any]:
        """Run comprehensive performance tests."""

        import time
        import statistics

        start_time = time.time()

        # Test metrics
        get_latencies = []
        set_latencies = []
        operation_count = 0

        while time.time() - start_time < duration:
            # Test GET operation
            start_op = time.perf_counter()
            await cache_service.get(f"perf_test_{operation_count}")
            get_latency = (time.perf_counter() - start_op) * 1000
            get_latencies.append(get_latency)

            # Test SET operation
            start_op = time.perf_counter()
            await cache_service.set(f"perf_test_{operation_count}", f"test_value_{operation_count}")
            set_latency = (time.perf_counter() - start_op) * 1000
            set_latencies.append(set_latency)

            operation_count += 1

            # Small delay to prevent overwhelming
            await asyncio.sleep(0.01)

        actual_duration = time.time() - start_time

        return {
            "operations_per_second": operation_count / actual_duration,
            "get_latency": {
                "avg": statistics.mean(get_latencies),
                "p50": statistics.median(get_latencies),
                "p95": sorted(get_latencies)[int(len(get_latencies) * 0.95)],
                "p99": sorted(get_latencies)[int(len(get_latencies) * 0.99)]
            },
            "set_latency": {
                "avg": statistics.mean(set_latencies),
                "p50": statistics.median(set_latencies),
                "p95": sorted(set_latencies)[int(len(set_latencies) * 0.95)],
                "p99": sorted(set_latencies)[int(len(set_latencies) * 0.99)]
            },
            "total_operations": operation_count,
            "test_duration": actual_duration
        }

    def _generate_tuning_recommendations(self, performance_results: Dict) -> List[Dict]:
        """Generate performance tuning recommendations."""

        recommendations = []

        # Check operation rate
        ops_per_second = performance_results["operations_per_second"]
        if ops_per_second < 1000:
            recommendations.append({
                "category": "throughput",
                "priority": "medium",
                "issue": "Low operation throughput",
                "recommendation": "Consider increasing connection pool size or optimizing Redis configuration",
                "current_value": ops_per_second,
                "target_value": 1000
            })

        # Check latency
        avg_get_latency = performance_results["get_latency"]["avg"]
        if avg_get_latency > 10:  # 10ms
            recommendations.append({
                "category": "latency",
                "priority": "high",
                "issue": "High GET operation latency",
                "recommendation": "Check network latency to Redis or consider local caching",
                "current_value": avg_get_latency,
                "target_value": 5
            })

        return recommendations
```

### Custom Cache Backend

```python
# src/backends/custom_backend.py
"""Custom cache backend implementation."""

from typing import Any, Dict, Optional
from abc import ABC, abstractmethod

class CacheBackend(ABC):
    """Abstract cache backend interface."""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """Clear all cache entries."""
        pass

class MemcachedBackend(CacheBackend):
    """Memcached cache backend implementation."""

    def __init__(self, servers: List[str], config: Dict):
        self.servers = servers
        self.config = config
        self.client = None

    async def initialize(self):
        """Initialize Memcached client."""
        try:
            import aiomemcache
            self.client = aiomemcache.Client(self.servers)
            await self.client.version()  # Test connection
        except ImportError:
            raise ImportError("aiomemcache package required for Memcached backend")

    async def get(self, key: str) -> Optional[Any]:
        """Get value from Memcached."""
        if not self.client:
            await self.initialize()

        try:
            value = await self.client.get(key.encode())
            if value:
                return self._deserialize(value)
            return None
        except Exception as e:
            logger.error(f"Memcached get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in Memcached."""
        if not self.client:
            await self.initialize()

        try:
            serialized_value = self._serialize(value)
            expiry = ttl or self.config.get("default_ttl", 3600)
            return await self.client.set(key.encode(), serialized_value, expiry)
        except Exception as e:
            logger.error(f"Memcached set error: {e}")
            return False

    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        import pickle
        return pickle.dumps(value)

    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        import pickle
        return pickle.loads(data)

class HybridBackend(CacheBackend):
    """Hybrid backend that combines multiple cache backends."""

    def __init__(self, primary: CacheBackend, secondary: CacheBackend):
        self.primary = primary
        self.secondary = secondary

    async def get(self, key: str) -> Optional[Any]:
        """Get from primary, fallback to secondary."""
        # Try primary first
        try:
            value = await self.primary.get(key)
            if value is not None:
                return value
        except Exception:
            pass

        # Fallback to secondary
        try:
            value = await self.secondary.get(key)
            if value is not None:
                # Restore to primary cache
                asyncio.create_task(self.primary.set(key, value))
                return value
        except Exception:
            pass

        return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set in both backends."""
        primary_success = False
        secondary_success = False

        try:
            primary_success = await self.primary.set(key, value, ttl)
        except Exception:
            pass

        try:
            secondary_success = await self.secondary.set(key, value, ttl)
        except Exception:
            pass

        return primary_success or secondary_success
```

## Testing Strategies

### Unit Testing Patterns

```python
# tests/test_custom_cache_service.py
"""Unit tests for custom cache service."""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock

from src.services.extensions.custom_cache_service import CustomCacheService

class TestCustomCacheService:
    """Test cases for CustomCacheService."""

    @pytest.fixture
    async def mock_base_service(self):
        """Create mock base cache service."""
        mock_service = AsyncMock()
        mock_service.config = MagicMock()
        return mock_service

    @pytest.fixture
    def custom_config(self):
        """Custom configuration for testing."""
        return {
            "analytics_enabled": True,
            "max_value_size": 1024,
            "auto_compress": True,
            "allowed_key_patterns": [r"test:.*", r"cache:.*"]
        }

    @pytest.fixture
    async def custom_service(self, mock_base_service, custom_config):
        """Create custom cache service instance."""
        service = CustomCacheService(mock_base_service, custom_config)
        await service.initialize()
        return service

    @pytest.mark.asyncio
    async def test_get_operation_with_analytics(self, custom_service, mock_base_service):
        """Test get operation records analytics."""
        # Setup
        test_key = "test:key1"
        test_value = "test_value"
        mock_base_service.get.return_value = test_value

        # Execute
        result = await custom_service.get(test_key)

        # Verify
        assert result == test_value
        mock_base_service.get.assert_called_once_with(test_key)

        # Verify analytics was recorded
        analytics = custom_service.custom_features.get("analytics")
        if analytics:
            analytics.record_access.assert_called_once_with(test_key)
            analytics.record_result.assert_called_once_with(test_key, True)

    @pytest.mark.asyncio
    async def test_set_validation_value_size(self, custom_service):
        """Test set operation validates value size."""
        # Setup large value
        large_value = "x" * 2048  # Exceeds max_value_size of 1024

        # Execute and verify exception
        with pytest.raises(CacheOperationError, match="Value size exceeds maximum"):
            await custom_service.set("test:key", large_value)

    @pytest.mark.asyncio
    async def test_set_validation_key_pattern(self, custom_service):
        """Test set operation validates key patterns."""
        # Invalid key pattern
        invalid_key = "invalid:pattern:key"

        # Execute and verify exception
        with pytest.raises(CacheOperationError, match="Invalid key pattern"):
            await custom_service.set(invalid_key, "value")

    @pytest.mark.asyncio
    async def test_value_preprocessing(self, custom_service, mock_base_service):
        """Test value preprocessing with compression."""
        # Setup
        test_key = "test:key1"
        test_value = "test_value"
        mock_base_service.set.return_value = True

        # Execute
        result = await custom_service.set(test_key, test_value)

        # Verify
        assert result is True

        # Verify that set was called (value might be compressed)
        mock_base_service.set.assert_called_once()
        call_args = mock_base_service.set.call_args
        assert call_args[0][0] == test_key  # Key should be unchanged

@pytest.mark.integration
class TestCacheServiceIntegration:
    """Integration tests for cache service."""

    @pytest.fixture
    async def redis_service(self):
        """Create real Redis cache service for integration testing."""
        from src.services.cache_service import RedisCacheService
        from src.config.cache_config import CacheConfig

        config = CacheConfig()
        config.redis.host = "localhost"
        config.redis.port = 6379
        config.redis.password = None

        service = RedisCacheService(config)
        await service.initialize()

        yield service

        # Cleanup
        await service.clear()
        await service.shutdown()

    @pytest.mark.asyncio
    async def test_end_to_end_cache_operations(self, redis_service):
        """Test end-to-end cache operations."""
        # Test SET
        success = await redis_service.set("integration:test", "test_value", ttl=60)
        assert success is True

        # Test GET
        value = await redis_service.get("integration:test")
        assert value == "test_value"

        # Test DELETE
        deleted = await redis_service.delete("integration:test")
        assert deleted is True

        # Verify deletion
        value = await redis_service.get("integration:test")
        assert value is None
```

### Performance Testing

```python
# tests/test_performance.py
"""Performance tests for cache system."""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor

class TestCachePerformance:
    """Performance test cases."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, cache_service):
        """Test cache performance under concurrent load."""

        async def worker(worker_id: int, operations: int):
            """Worker function for concurrent testing."""
            latencies = []

            for i in range(operations):
                key = f"perf:worker_{worker_id}:key_{i}"
                value = f"value_{i}"

                # Measure SET latency
                start_time = time.perf_counter()
                await cache_service.set(key, value)
                set_latency = (time.perf_counter() - start_time) * 1000

                # Measure GET latency
                start_time = time.perf_counter()
                result = await cache_service.get(key)
                get_latency = (time.perf_counter() - start_time) * 1000

                assert result == value

                latencies.append({
                    "set_latency": set_latency,
                    "get_latency": get_latency
                })

            return latencies

        # Run concurrent workers
        num_workers = 10
        operations_per_worker = 100

        start_time = time.time()

        tasks = [
            worker(i, operations_per_worker)
            for i in range(num_workers)
        ]

        results = await asyncio.gather(*tasks)

        total_time = time.time() - start_time

        # Analyze results
        all_set_latencies = []
        all_get_latencies = []

        for worker_results in results:
            for op_result in worker_results:
                all_set_latencies.append(op_result["set_latency"])
                all_get_latencies.append(op_result["get_latency"])

        total_operations = num_workers * operations_per_worker

        performance_report = {
            "total_operations": total_operations,
            "total_time": total_time,
            "operations_per_second": total_operations / total_time,
            "set_latency": {
                "avg": statistics.mean(all_set_latencies),
                "p95": sorted(all_set_latencies)[int(len(all_set_latencies) * 0.95)],
                "p99": sorted(all_set_latencies)[int(len(all_set_latencies) * 0.99)]
            },
            "get_latency": {
                "avg": statistics.mean(all_get_latencies),
                "p95": sorted(all_get_latencies)[int(len(all_get_latencies) * 0.95)],
                "p99": sorted(all_get_latencies)[int(len(all_get_latencies) * 0.99)]
            }
        }

        print(f"Performance Report: {performance_report}")

        # Performance assertions
        assert performance_report["operations_per_second"] > 500
        assert performance_report["set_latency"]["avg"] < 50  # 50ms
        assert performance_report["get_latency"]["avg"] < 20  # 20ms

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_scaling(self, cache_service):
        """Test memory usage scaling with data volume."""
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss

        # Store increasing amounts of data
        data_sizes = [1024, 10240, 102400]  # 1KB, 10KB, 100KB

        for size in data_sizes:
            # Store 100 entries of this size
            test_data = "x" * size

            for i in range(100):
                await cache_service.set(f"memory_test_{size}_{i}", test_data)

            current_memory = process.memory_info().rss
            memory_increase = current_memory - initial_memory

            print(f"Memory usage after {size}B x 100 entries: {memory_increase / 1024 / 1024:.1f}MB")

            # Cleanup
            for i in range(100):
                await cache_service.delete(f"memory_test_{size}_{i}")
```

### Load Testing

```python
# tests/load_test.py
"""Load testing for cache system."""

import asyncio
import time
import random
import statistics
from dataclasses import dataclass
from typing import List

@dataclass
class LoadTestResult:
    """Load test result data."""
    total_requests: int
    successful_requests: int
    failed_requests: int
    duration: float
    requests_per_second: float
    avg_latency: float
    p95_latency: float
    p99_latency: float
    error_rate: float

class CacheLoadTester:
    """Load tester for cache system."""

    def __init__(self, cache_service):
        self.cache_service = cache_service

    async def run_load_test(
        self,
        duration_seconds: int = 60,
        concurrent_users: int = 50,
        operations_per_user: int = 100
    ) -> LoadTestResult:
        """Run comprehensive load test."""

        async def user_simulation(user_id: int) -> List[float]:
            """Simulate user operations."""
            latencies = []

            for op_num in range(operations_per_user):
                # Random operation mix
                operation = random.choices(
                    ["get", "set", "delete"],
                    weights=[70, 20, 10]  # 70% read, 20% write, 10% delete
                )[0]

                key = f"load_test:user_{user_id}:op_{op_num}"

                start_time = time.perf_counter()

                try:
                    if operation == "get":
                        await self.cache_service.get(key)
                    elif operation == "set":
                        value = f"test_value_{user_id}_{op_num}"
                        await self.cache_service.set(key, value, ttl=3600)
                    elif operation == "delete":
                        await self.cache_service.delete(key)

                    latency = (time.perf_counter() - start_time) * 1000
                    latencies.append(latency)

                except Exception as e:
                    # Record error but continue
                    latencies.append(-1)  # Error marker

                # Small random delay between operations
                await asyncio.sleep(random.uniform(0.001, 0.01))

            return latencies

        # Start load test
        start_time = time.time()

        # Create user simulation tasks
        tasks = [
            user_simulation(user_id)
            for user_id in range(concurrent_users)
        ]

        # Run all user simulations concurrently
        all_results = await asyncio.gather(*tasks)

        test_duration = time.time() - start_time

        # Aggregate results
        all_latencies = []
        total_requests = 0
        failed_requests = 0

        for user_results in all_results:
            for latency in user_results:
                total_requests += 1
                if latency == -1:
                    failed_requests += 1
                else:
                    all_latencies.append(latency)

        successful_requests = total_requests - failed_requests

        # Calculate statistics
        if all_latencies:
            avg_latency = statistics.mean(all_latencies)
            sorted_latencies = sorted(all_latencies)
            p95_latency = sorted_latencies[int(len(sorted_latencies) * 0.95)]
            p99_latency = sorted_latencies[int(len(sorted_latencies) * 0.99)]
        else:
            avg_latency = p95_latency = p99_latency = 0

        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            duration=test_duration,
            requests_per_second=total_requests / test_duration,
            avg_latency=avg_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            error_rate=failed_requests / total_requests if total_requests > 0 else 0
        )

# Usage
async def main():
    from src.services.cache_service import get_cache_service

    cache_service = await get_cache_service()
    load_tester = CacheLoadTester(cache_service)

    # Run load test
    result = await load_tester.run_load_test(
        duration_seconds=120,
        concurrent_users=100,
        operations_per_user=200
    )

    print(f"Load Test Results:")
    print(f"  Total Requests: {result.total_requests}")
    print(f"  Successful: {result.successful_requests}")
    print(f"  Failed: {result.failed_requests}")
    print(f"  Error Rate: {result.error_rate:.2%}")
    print(f"  Requests/sec: {result.requests_per_second:.1f}")
    print(f"  Avg Latency: {result.avg_latency:.1f}ms")
    print(f"  P95 Latency: {result.p95_latency:.1f}ms")
    print(f"  P99 Latency: {result.p99_latency:.1f}ms")

if __name__ == "__main__":
    asyncio.run(main())
```

## Contribution Guidelines

### Code Style and Standards

```python
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-merge-conflict
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
        language_version: python3.9

  - repo: https://github.com/charliermarsh/ruff-pre-commit
    rev: v0.0.254
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict, --ignore-missing-imports]
```

### Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/cache-enhancement

# 2. Make changes and run tests
uv run pytest tests/ -v

# 3. Run code quality checks
uv run black src/ tests/
uv run ruff check src/ tests/ --fix
uv run mypy src/

# 4. Commit changes
git add .
git commit -m "feat: add cache enhancement feature"

# 5. Push and create PR
git push origin feature/cache-enhancement
```

### Pull Request Guidelines

#### PR Template

```markdown
## Description
Brief description of the changes made.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] This change requires a documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Performance tests added/updated
- [ ] Manual testing performed

## Performance Impact
- [ ] No performance impact
- [ ] Performance improvement
- [ ] Performance regression (explain below)

## Documentation
- [ ] Code is self-documenting
- [ ] Docstrings added/updated
- [ ] README updated
- [ ] Architecture docs updated

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Tests pass locally
- [ ] No breaking changes to existing API
- [ ] Backward compatibility maintained
```

This comprehensive development guide provides everything needed to extend and contribute to the cache system effectively, following best practices and maintaining high code quality.
