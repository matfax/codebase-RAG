#!/usr/bin/env python3
"""
Cache Configuration Validation Script

This script validates the cache configuration for the Codebase RAG MCP Server.
It checks environment variables, Redis connectivity, and configuration consistency.

Usage:
    python validate_cache_config.py [--check-redis] [--verbose]

Options:
    --check-redis    Test Redis connectivity (requires Redis to be running)
    --verbose        Enable verbose output
"""

import argparse
import os
import sys
from pathlib import Path

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from src.config.cache_config import CacheConfig, CacheConfigError, validate_cache_config
except ImportError as e:
    print(f"❌ Failed to import cache configuration: {e}")
    print("Make sure you're running this script from the project root directory.")
    sys.exit(1)


def check_redis_connectivity(config: CacheConfig) -> bool:
    """
    Check Redis connectivity.

    Args:
        config: Cache configuration

    Returns:
        bool: True if Redis is accessible
    """
    try:
        import redis
    except ImportError:
        print("❌ Redis library not installed. Run: pip install redis")
        return False

    try:
        # Create Redis client
        client = redis.Redis(
            host=config.redis.host,
            port=config.redis.port,
            password=config.redis.password,
            db=config.redis.db,
            socket_timeout=config.redis.socket_timeout,
            socket_connect_timeout=config.redis.connection_timeout,
            retry_on_timeout=config.redis.retry_on_timeout,
            ssl=config.redis.ssl_enabled,
            ssl_cert_reqs=None if not config.redis.ssl_enabled else "required",
            ssl_certfile=config.redis.ssl_cert_path,
            ssl_keyfile=config.redis.ssl_key_path,
            ssl_ca_certs=config.redis.ssl_ca_cert_path,
        )

        # Test connection
        response = client.ping()
        if response:
            print(f"✅ Redis connectivity: Connected to {config.redis.host}:{config.redis.port}")

            # Test basic operations
            test_key = f"{config.key_prefix}:test:validation"
            client.set(test_key, "test_value", ex=10)
            value = client.get(test_key)
            client.delete(test_key)

            if value == b"test_value":
                print("✅ Redis operations: Basic set/get/delete operations working")
            else:
                print("⚠️  Redis operations: Basic operations may have issues")
                return False

            return True
        else:
            print(f"❌ Redis connectivity: Failed to ping {config.redis.host}:{config.redis.port}")
            return False

    except redis.ConnectionError as e:
        print(f"❌ Redis connectivity: Connection failed - {e}")
        return False
    except redis.AuthenticationError as e:
        print(f"❌ Redis connectivity: Authentication failed - {e}")
        return False
    except Exception as e:
        print(f"❌ Redis connectivity: Unexpected error - {e}")
        return False


def check_cryptography_availability() -> bool:
    """
    Check if cryptography library is available.

    Returns:
        bool: True if cryptography is available
    """
    try:
        import cryptography

        print(f"✅ Cryptography library: Available (version {cryptography.__version__})")
        return True
    except ImportError:
        print("❌ Cryptography library: Not installed. Run: pip install cryptography")
        return False


def validate_environment() -> bool:
    """
    Validate environment setup.

    Returns:
        bool: True if environment is valid
    """
    success = True

    # Check if .env file exists
    env_file = Path(".env")
    env_example_file = Path(".env.example")

    if env_file.exists():
        print(f"✅ Environment file: {env_file} exists")
    elif env_example_file.exists():
        print(f"⚠️  Environment file: {env_file} not found, but {env_example_file} exists")
        print(f"   Consider copying {env_example_file} to {env_file} and customizing")
    else:
        print(f"❌ Environment file: Neither {env_file} nor {env_example_file} found")
        success = False

    return success


def print_config_summary(config: CacheConfig, verbose: bool = False) -> None:
    """
    Print configuration summary.

    Args:
        config: Cache configuration
        verbose: Enable verbose output
    """
    print("\n📋 Cache Configuration Summary:")
    print(f"   Cache Enabled: {config.enabled}")
    print(f"   Cache Level: {config.cache_level.value}")
    print(f"   Write Strategy: {config.write_strategy.value}")
    print(f"   Default TTL: {config.default_ttl}s")

    print("\n🔧 Redis Configuration:")
    print(f"   Host: {config.redis.host}")
    print(f"   Port: {config.redis.port}")
    print(f"   Database: {config.redis.db}")
    print(f"   Max Connections: {config.redis.max_connections}")
    print(f"   SSL Enabled: {config.redis.ssl_enabled}")

    print("\n💾 Memory Cache Configuration:")
    print(f"   Max Size: {config.memory.max_size}")
    print(f"   TTL: {config.memory.ttl_seconds}s")
    print(f"   Eviction Policy: {config.memory.eviction_policy.value}")
    print(f"   Max Memory: {config.memory.max_memory_mb}MB")

    print("\n📊 Cache Type TTLs:")
    print(f"   Embedding Cache: {config.embedding_cache.ttl_seconds}s")
    print(f"   Search Cache: {config.search_cache.ttl_seconds}s")
    print(f"   Project Cache: {config.project_cache.ttl_seconds}s")
    print(f"   File Cache: {config.file_cache.ttl_seconds}s")

    if verbose:
        print("\n🔐 Security Configuration:")
        print(f"   Encryption Enabled: {config.encryption_enabled}")
        print(f"   Encryption Key Set: {'Yes' if config.encryption_key else 'No'}")

        print("\n⚡ Performance Configuration:")
        print(f"   Batch Size: {config.batch_size}")
        print(f"   Parallel Operations: {config.parallel_operations}")
        print(f"   Connection Pool Size: {config.connection_pool_size}")

        print("\n📈 Monitoring Configuration:")
        print(f"   Metrics Enabled: {config.metrics_enabled}")
        print(f"   Health Check Interval: {config.health_check_interval}s")
        print(f"   Stats Collection Interval: {config.stats_collection_interval}s")


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate cache configuration for Codebase RAG MCP Server")
    parser.add_argument("--check-redis", action="store_true", help="Test Redis connectivity (requires Redis to be running)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    print("🔍 Validating Cache Configuration...")
    print("=" * 50)

    success = True

    # Validate environment setup
    if not validate_environment():
        success = False

    # Load and validate configuration
    try:
        config = CacheConfig.from_env()
        validate_cache_config(config)
        print("✅ Configuration validation: All settings are valid")
    except CacheConfigError as e:
        print(f"❌ Configuration validation: {e}")
        success = False
        return
    except Exception as e:
        print(f"❌ Configuration loading: Unexpected error - {e}")
        success = False
        return

    # Check dependencies
    if not check_cryptography_availability():
        success = False

    # Check Redis connectivity if requested
    if args.check_redis:
        if not check_redis_connectivity(config):
            success = False
    else:
        print("⏭️  Redis connectivity: Skipped (use --check-redis to test)")

    # Print configuration summary
    print_config_summary(config, args.verbose)

    # Final result
    print("\n" + "=" * 50)
    if success:
        print("🎉 Cache configuration validation completed successfully!")
        print("\n💡 Next steps:")
        print("   1. Start Redis: docker-compose -f docker-compose.cache.yml up -d")
        print("   2. Test connectivity: python validate_cache_config.py --check-redis")
        print("   3. Start the MCP server with cache enabled")
    else:
        print("❌ Cache configuration validation failed!")
        print("\n🔧 Please fix the issues above and run validation again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
