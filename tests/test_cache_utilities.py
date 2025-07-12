"""
Unit tests for cache utilities and models - Wave 15.1.2
Tests cache utility functions, data models, and helper classes.
"""

import json
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from src.models.cache_models import (
    CacheEntry,
    CacheMetrics,
    CachePolicy,
    CacheStatus,
    EvictionPolicy,
    InvalidationRule,
    SecurityContext
)
from src.utils.cache_key_utils import (
    generate_cache_key,
    parse_cache_key,
    get_key_prefix,
    sanitize_key_component,
    create_namespace_key
)
from src.utils.cache_warmup_utils import (
    CacheWarmupStrategy,
    PredictiveWarmup,
    ScheduledWarmup,
    warmup_cache_entries,
    analyze_access_patterns
)
from src.utils.secure_cache_utils import (
    SecureCacheWrapper,
    encrypt_cache_value,
    decrypt_cache_value,
    generate_encryption_key,
    validate_security_context
)


class TestCacheModels:
    """Test cache data models."""

    def test_cache_entry_creation(self):
        """Test CacheEntry model creation and validation."""
        entry = CacheEntry(
            key="test:key:123",
            value={"data": "test"},
            ttl=3600,
            created_at=datetime.now(),
            access_count=0,
            last_accessed=None,
            metadata={"source": "api"}
        )
        
        assert entry.key == "test:key:123"
        assert entry.value == {"data": "test"}
        assert entry.ttl == 3600
        assert entry.access_count == 0
        assert entry.is_expired() is False

    def test_cache_entry_expiration(self):
        """Test CacheEntry expiration logic."""
        past_time = datetime.now() - timedelta(hours=2)
        entry = CacheEntry(
            key="expired:key",
            value="data",
            ttl=3600,  # 1 hour TTL
            created_at=past_time
        )
        
        assert entry.is_expired() is True
        assert entry.remaining_ttl() < 0

    def test_cache_metrics_tracking(self):
        """Test CacheMetrics tracking and calculation."""
        metrics = CacheMetrics()
        
        # Record some operations
        metrics.record_hit("key1")
        metrics.record_hit("key2")
        metrics.record_miss("key3")
        metrics.record_miss("key4")
        metrics.record_miss("key5")
        
        assert metrics.total_hits == 2
        assert metrics.total_misses == 3
        assert metrics.hit_rate == 0.4  # 2/5
        
        # Test eviction tracking
        metrics.record_eviction("key1", reason="ttl_expired")
        metrics.record_eviction("key2", reason="memory_pressure")
        
        assert metrics.total_evictions == 2
        assert metrics.eviction_reasons["ttl_expired"] == 1
        assert metrics.eviction_reasons["memory_pressure"] == 1

    def test_cache_policy_configuration(self):
        """Test CachePolicy configuration and validation."""
        policy = CachePolicy(
            eviction_policy=EvictionPolicy.LRU,
            max_size_mb=100,
            default_ttl=3600,
            enable_compression=True,
            compression_threshold=1024,
            enable_encryption=True,
            namespace_isolation=True
        )
        
        assert policy.eviction_policy == EvictionPolicy.LRU
        assert policy.max_size_mb == 100
        assert policy.should_compress(2048) is True
        assert policy.should_compress(512) is False

    def test_invalidation_rule_matching(self):
        """Test InvalidationRule pattern matching."""
        rule = InvalidationRule(
            pattern="cache:user:*",
            dependencies=["cache:profile:*", "cache:preferences:*"],
            cascade=True,
            priority=10
        )
        
        assert rule.matches("cache:user:123") is True
        assert rule.matches("cache:user:456:profile") is True
        assert rule.matches("cache:product:123") is False
        assert rule.should_cascade() is True

    def test_security_context_validation(self):
        """Test SecurityContext creation and validation."""
        context = SecurityContext(
            user_id="user123",
            tenant_id="tenant456",
            permissions=["read", "write"],
            encryption_required=True,
            audit_enabled=True
        )
        
        assert context.has_permission("read") is True
        assert context.has_permission("delete") is False
        assert context.requires_encryption() is True
        assert context.should_audit() is True


class TestCacheKeyUtils:
    """Test cache key utility functions."""

    def test_generate_cache_key_simple(self):
        """Test simple cache key generation."""
        key = generate_cache_key("users", "123")
        assert key == "users:123"

    def test_generate_cache_key_with_namespace(self):
        """Test cache key generation with namespace."""
        key = generate_cache_key("users", "123", namespace="tenant_456")
        assert key == "tenant_456:users:123"

    def test_generate_cache_key_with_params(self):
        """Test cache key generation with parameters."""
        params = {"sort": "name", "filter": "active", "page": 1}
        key = generate_cache_key("users", "list", params=params)
        
        # Key should be deterministic
        key2 = generate_cache_key("users", "list", params=params)
        assert key == key2
        
        # Different param order should produce same key
        params_reordered = {"page": 1, "filter": "active", "sort": "name"}
        key3 = generate_cache_key("users", "list", params=params_reordered)
        assert key == key3

    def test_parse_cache_key(self):
        """Test cache key parsing."""
        key = "tenant_123:users:profile:456"
        components = parse_cache_key(key)
        
        assert components["namespace"] == "tenant_123"
        assert components["prefix"] == "users"
        assert components["identifier"] == "profile:456"

    def test_sanitize_key_component(self):
        """Test key component sanitization."""
        # Test removal of invalid characters
        assert sanitize_key_component("user@email.com") == "user_email_com"
        assert sanitize_key_component("path/to/file") == "path_to_file"
        assert sanitize_key_component("key with spaces") == "key_with_spaces"
        
        # Test length limitation
        long_component = "a" * 200
        sanitized = sanitize_key_component(long_component, max_length=50)
        assert len(sanitized) <= 50

    def test_create_namespace_key(self):
        """Test namespace key creation."""
        key = create_namespace_key("tenant_123", "users", "456")
        assert key == "tenant_123:users:456"
        
        # Test with multiple components
        key = create_namespace_key("global", "cache", "search", "results", "query_789")
        assert key == "global:cache:search:results:query_789"


class TestCacheWarmupUtils:
    """Test cache warmup utilities."""

    @pytest.mark.asyncio
    async def test_predictive_warmup(self):
        """Test predictive cache warmup based on access patterns."""
        warmup = PredictiveWarmup()
        
        # Simulate access pattern
        access_log = [
            {"key": "user:123", "timestamp": time.time() - 3600, "frequency": 10},
            {"key": "user:456", "timestamp": time.time() - 7200, "frequency": 5},
            {"key": "product:789", "timestamp": time.time() - 1800, "frequency": 15}
        ]
        
        predictions = await warmup.predict_next_access(access_log)
        
        # Most frequently accessed should be prioritized
        assert predictions[0]["key"] == "product:789"
        assert predictions[0]["priority"] > predictions[1]["priority"]

    @pytest.mark.asyncio
    async def test_scheduled_warmup(self):
        """Test scheduled cache warmup."""
        warmup = ScheduledWarmup()
        
        # Define warmup schedule
        schedule = {
            "daily_reports": {
                "pattern": "report:daily:*",
                "schedule": "0 6 * * *",  # 6 AM daily
                "priority": "high"
            },
            "user_profiles": {
                "pattern": "user:profile:*",
                "schedule": "*/30 * * * *",  # Every 30 minutes
                "priority": "medium"
            }
        }
        
        warmup.set_schedule(schedule)
        next_runs = warmup.get_next_warmup_times()
        
        assert "daily_reports" in next_runs
        assert "user_profiles" in next_runs

    @pytest.mark.asyncio
    async def test_warmup_cache_entries(self):
        """Test batch cache warmup functionality."""
        mock_cache = MagicMock()
        mock_data_loader = MagicMock()
        
        # Define entries to warmup
        entries = [
            {"key": "user:123", "loader": "load_user", "params": {"id": 123}},
            {"key": "product:456", "loader": "load_product", "params": {"id": 456}}
        ]
        
        mock_data_loader.load_user.return_value = {"id": 123, "name": "Test User"}
        mock_data_loader.load_product.return_value = {"id": 456, "name": "Test Product"}
        
        results = await warmup_cache_entries(entries, mock_cache, mock_data_loader)
        
        assert results["success"] == 2
        assert results["failed"] == 0
        assert mock_cache.set.call_count == 2

    def test_analyze_access_patterns(self):
        """Test access pattern analysis for warmup optimization."""
        access_log = [
            {"key": "user:123", "timestamp": time.time() - 3600},
            {"key": "user:123", "timestamp": time.time() - 7200},
            {"key": "user:123", "timestamp": time.time() - 10800},
            {"key": "product:456", "timestamp": time.time() - 1800},
            {"key": "product:456", "timestamp": time.time() - 5400}
        ]
        
        patterns = analyze_access_patterns(access_log)
        
        assert patterns["user:123"]["frequency"] == 3
        assert patterns["product:456"]["frequency"] == 2
        assert patterns["user:123"]["avg_interval"] > 0


class TestSecureCacheUtils:
    """Test secure cache utilities."""

    def test_secure_cache_wrapper(self):
        """Test SecureCacheWrapper functionality."""
        base_cache = MagicMock()
        security_context = SecurityContext(
            user_id="user123",
            encryption_required=True
        )
        
        secure_cache = SecureCacheWrapper(base_cache, security_context)
        
        # Test secure set
        test_data = {"sensitive": "data"}
        secure_cache.set("key123", test_data)
        
        # Verify encryption was applied
        base_cache.set.assert_called_once()
        stored_value = base_cache.set.call_args[0][1]
        assert stored_value != test_data  # Should be encrypted

    def test_encrypt_decrypt_cache_value(self):
        """Test cache value encryption and decryption."""
        original_value = {"user": "test", "password": "secret123"}
        encryption_key = generate_encryption_key()
        
        # Encrypt
        encrypted = encrypt_cache_value(original_value, encryption_key)
        assert encrypted != original_value
        assert isinstance(encrypted, str)
        
        # Decrypt
        decrypted = decrypt_cache_value(encrypted, encryption_key)
        assert decrypted == original_value

    def test_validate_security_context(self):
        """Test security context validation."""
        # Valid context
        valid_context = SecurityContext(
            user_id="user123",
            tenant_id="tenant456",
            permissions=["read", "write"]
        )
        assert validate_security_context(valid_context, required_permissions=["read"]) is True
        
        # Invalid context - missing permission
        assert validate_security_context(valid_context, required_permissions=["delete"]) is False
        
        # Invalid context - missing user
        invalid_context = SecurityContext(
            user_id=None,
            permissions=["read"]
        )
        assert validate_security_context(invalid_context, require_user=True) is False


class TestCacheUtilityIntegration:
    """Test integration between different cache utilities."""

    @pytest.mark.asyncio
    async def test_secure_cache_with_warmup(self):
        """Test integration of secure cache with warmup utilities."""
        base_cache = MagicMock()
        security_context = SecurityContext(
            user_id="user123",
            encryption_required=True
        )
        
        secure_cache = SecureCacheWrapper(base_cache, security_context)
        
        # Warmup with secure cache
        entries = [
            {"key": "secure:user:123", "data": {"name": "Test", "ssn": "123-45-6789"}}
        ]
        
        for entry in entries:
            await secure_cache.set(entry["key"], entry["data"])
        
        # Verify encryption was applied during warmup
        assert base_cache.set.call_count == 1
        stored_value = base_cache.set.call_args[0][1]
        assert "ssn" not in str(stored_value)  # Should be encrypted

    def test_cache_key_with_security_namespace(self):
        """Test cache key generation with security namespacing."""
        security_context = SecurityContext(
            user_id="user123",
            tenant_id="tenant456"
        )
        
        # Generate tenant-isolated key
        key = generate_cache_key(
            "users",
            "profile",
            namespace=f"tenant:{security_context.tenant_id}"
        )
        
        assert key.startswith("tenant:tenant456:")
        assert "users:profile" in key

    def test_invalidation_with_secure_keys(self):
        """Test cache invalidation with secure key patterns."""
        rule = InvalidationRule(
            pattern="tenant:*:users:*",
            cascade=True,
            security_context=SecurityContext(tenant_id="tenant456")
        )
        
        # Should match tenant-specific keys
        assert rule.matches("tenant:tenant456:users:123") is True
        assert rule.matches("tenant:tenant789:users:123") is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])