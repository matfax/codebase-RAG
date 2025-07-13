#!/usr/bin/env python3
"""
Cache Configuration Validation Script

Validates cache configuration before deployment, checking for:
- Required environment variables
- Valid configuration values
- Security settings
- Performance parameters
"""

import json
import os
import re
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.cache_config import CacheConfig, CacheLevel, WriteStrategy


class ValidationLevel(Enum):
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass
class ValidationResult:
    level: ValidationLevel
    category: str
    message: str
    suggestion: str | None = None


class CacheConfigValidator:
    """Comprehensive cache configuration validator."""

    def __init__(self, env_file: str | None = None):
        self.env_file = env_file or os.path.join(project_root, ".env")
        self.results: list[ValidationResult] = []
        self.config: CacheConfig | None = None

    def validate(self) -> tuple[bool, list[ValidationResult]]:
        """Run all validation checks."""
        self.results = []

        # Load configuration
        if not self._load_configuration():
            return False, self.results

        # Run validation checks
        self._validate_core_settings()
        self._validate_redis_configuration()
        self._validate_memory_configuration()
        self._validate_security_settings()
        self._validate_performance_settings()
        self._validate_cache_type_settings()
        self._validate_monitoring_settings()

        # Check for errors
        has_errors = any(r.level == ValidationLevel.ERROR for r in self.results)
        return not has_errors, self.results

    def _load_configuration(self) -> bool:
        """Load configuration from environment."""
        try:
            # Load environment file if exists
            if Path(self.env_file).exists():
                self._load_env_file(self.env_file)

            # Create configuration
            self.config = CacheConfig.from_env()
            return True

        except Exception as e:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.ERROR,
                    category="Configuration Loading",
                    message=f"Failed to load configuration: {e}",
                    suggestion="Check .env file format and required variables",
                )
            )
            return False

    def _load_env_file(self, env_file: str):
        """Load environment variables from file."""
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, _, value = line.partition("=")
                    if key and value:
                        os.environ[key.strip()] = value.strip().strip("\"'")

    def _validate_core_settings(self):
        """Validate core cache settings."""
        # Check if cache is enabled
        if not self.config.enabled:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="Core Settings",
                    message="Cache is disabled",
                    suggestion="Set CACHE_ENABLED=true to enable caching",
                )
            )

        # Check cache level
        if self.config.cache_level == CacheLevel.L1_MEMORY:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.INFO,
                    category="Core Settings",
                    message="Using memory-only cache (L1)",
                    suggestion="Consider using BOTH for better persistence",
                )
            )

        # Check write strategy
        if self.config.write_strategy == WriteStrategy.WRITE_BACK:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="Core Settings",
                    message="Using write-back strategy - potential data loss on crash",
                    suggestion="Consider WRITE_THROUGH for better consistency",
                )
            )

    def _validate_redis_configuration(self):
        """Validate Redis configuration."""
        if self.config.cache_level in [CacheLevel.L2_REDIS, CacheLevel.BOTH]:
            # Check Redis host
            if self.config.redis.host in ["localhost", "127.0.0.1"]:
                self.results.append(
                    ValidationResult(
                        level=ValidationLevel.INFO,
                        category="Redis Configuration",
                        message="Using localhost for Redis",
                        suggestion="Ensure Redis is running locally or update REDIS_HOST",
                    )
                )

            # Check Redis password
            if not self.config.redis.password:
                self.results.append(
                    ValidationResult(
                        level=ValidationLevel.ERROR,
                        category="Redis Security",
                        message="Redis password not set",
                        suggestion="Set REDIS_PASSWORD for security",
                    )
                )
            elif len(self.config.redis.password) < 8:
                self.results.append(
                    ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="Redis Security",
                        message="Redis password is too short",
                        suggestion="Use a password with at least 8 characters",
                    )
                )

            # Check connection pool size
            if self.config.redis.max_connections < 5:
                self.results.append(
                    ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="Redis Performance",
                        message="Redis connection pool size is very small",
                        suggestion="Increase REDIS_MAX_CONNECTIONS for better performance",
                    )
                )

            # Check SSL configuration
            if self.config.redis.ssl_enabled:
                if not self.config.redis.ssl_cert_path:
                    self.results.append(
                        ValidationResult(
                            level=ValidationLevel.ERROR,
                            category="Redis Security",
                            message="SSL enabled but no certificate path provided",
                            suggestion="Set REDIS_SSL_CERT_PATH or disable SSL",
                        )
                    )
                elif not Path(self.config.redis.ssl_cert_path).exists():
                    self.results.append(
                        ValidationResult(
                            level=ValidationLevel.ERROR,
                            category="Redis Security",
                            message=f"SSL certificate not found: {self.config.redis.ssl_cert_path}",
                            suggestion="Check certificate path or generate new certificates",
                        )
                    )

    def _validate_memory_configuration(self):
        """Validate memory cache configuration."""
        # Check memory limits
        if self.config.memory.max_memory_mb < 64:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="Memory Configuration",
                    message="Memory cache size is very small (<64MB)",
                    suggestion="Increase MEMORY_CACHE_MAX_MEMORY_MB for better performance",
                )
            )
        elif self.config.memory.max_memory_mb > 8192:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="Memory Configuration",
                    message="Memory cache size is very large (>8GB)",
                    suggestion="Consider system memory limits",
                )
            )

        # Check cache size limits
        if self.config.memory.max_size < 100:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="Memory Configuration",
                    message="Maximum cache entries is very low",
                    suggestion="Increase MEMORY_CACHE_MAX_SIZE for better hit rates",
                )
            )

        # Check cleanup interval
        if self.config.memory.cleanup_interval < 60:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="Memory Configuration",
                    message="Cleanup interval is very frequent",
                    suggestion="Increase MEMORY_CACHE_CLEANUP_INTERVAL to reduce overhead",
                )
            )

    def _validate_security_settings(self):
        """Validate security configuration."""
        environment = os.getenv("ENVIRONMENT", "development")

        # Check encryption settings
        if self.config.encryption_enabled:
            if not self.config.encryption_key:
                self.results.append(
                    ValidationResult(
                        level=ValidationLevel.ERROR,
                        category="Security",
                        message="Encryption enabled but no key provided",
                        suggestion="Set CACHE_ENCRYPTION_KEY or disable encryption",
                    )
                )
            elif len(self.config.encryption_key) < 32:
                self.results.append(
                    ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="Security",
                        message="Encryption key appears to be too short",
                        suggestion="Use a key with at least 32 characters",
                    )
                )

        # Production security checks
        if environment == "production":
            if not self.config.encryption_enabled:
                self.results.append(
                    ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="Security",
                        message="Encryption disabled in production",
                        suggestion="Enable CACHE_ENCRYPTION_ENABLED for production",
                    )
                )

            if not self.config.project_isolation:
                self.results.append(
                    ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="Security",
                        message="Project isolation disabled in production",
                        suggestion="Enable CACHE_PROJECT_ISOLATION for security",
                    )
                )

            if not self.config.access_logging:
                self.results.append(
                    ValidationResult(
                        level=ValidationLevel.WARNING,
                        category="Security",
                        message="Access logging disabled in production",
                        suggestion="Enable CACHE_ACCESS_LOGGING for audit trail",
                    )
                )

    def _validate_performance_settings(self):
        """Validate performance configuration."""
        # Check TTL settings
        if self.config.default_ttl < 60:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="Performance",
                    message="Default TTL is very short (<1 minute)",
                    suggestion="Increase CACHE_DEFAULT_TTL to reduce cache misses",
                )
            )

        # Check batch settings
        if self.config.batch_size > 1000:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="Performance",
                    message="Batch size is very large",
                    suggestion="Reduce CACHE_BATCH_SIZE to avoid memory spikes",
                )
            )

        # Check parallel operations
        if self.config.parallel_operations > 32:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="Performance",
                    message="High parallel operations may overwhelm system",
                    suggestion="Reduce CACHE_PARALLEL_OPERATIONS",
                )
            )

    def _validate_cache_type_settings(self):
        """Validate cache type specific settings."""
        # Embedding cache validation
        embedding_ttl = int(os.getenv("EMBEDDING_CACHE_TTL", "7200"))
        if embedding_ttl < 3600:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.INFO,
                    category="Cache Types",
                    message="Embedding cache TTL is short (<1 hour)",
                    suggestion="Embeddings are stable, consider longer TTL",
                )
            )

        # Search cache validation
        search_ttl = int(os.getenv("SEARCH_CACHE_TTL", "1800"))
        if search_ttl > 3600:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.INFO,
                    category="Cache Types",
                    message="Search cache TTL is long (>1 hour)",
                    suggestion="Search results may become stale",
                )
            )

    def _validate_monitoring_settings(self):
        """Validate monitoring configuration."""
        if not self.config.metrics_enabled:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.INFO,
                    category="Monitoring",
                    message="Metrics collection disabled",
                    suggestion="Enable CACHE_METRICS_ENABLED for monitoring",
                )
            )

        # Check health check interval
        if self.config.health_check_interval < 30:
            self.results.append(
                ValidationResult(
                    level=ValidationLevel.WARNING,
                    category="Monitoring",
                    message="Health check interval is very frequent",
                    suggestion="Increase CACHE_HEALTH_CHECK_INTERVAL to reduce overhead",
                )
            )

    def print_results(self, results: list[ValidationResult]):
        """Print validation results in a formatted way."""
        # Group by level
        errors = [r for r in results if r.level == ValidationLevel.ERROR]
        warnings = [r for r in results if r.level == ValidationLevel.WARNING]
        infos = [r for r in results if r.level == ValidationLevel.INFO]

        # Print summary
        print("\n" + "=" * 60)
        print("Cache Configuration Validation Report")
        print("=" * 60)
        print(f"Errors:   {len(errors)}")
        print(f"Warnings: {len(warnings)}")
        print(f"Info:     {len(infos)}")
        print("=" * 60 + "\n")

        # Print errors
        if errors:
            print("ERRORS (must be fixed):")
            print("-" * 60)
            for r in errors:
                print(f"[{r.category}] {r.message}")
                if r.suggestion:
                    print(f"  → {r.suggestion}")
                print()

        # Print warnings
        if warnings:
            print("WARNINGS (should be addressed):")
            print("-" * 60)
            for r in warnings:
                print(f"[{r.category}] {r.message}")
                if r.suggestion:
                    print(f"  → {r.suggestion}")
                print()

        # Print info
        if infos:
            print("INFO (recommendations):")
            print("-" * 60)
            for r in infos:
                print(f"[{r.category}] {r.message}")
                if r.suggestion:
                    print(f"  → {r.suggestion}")
                print()

    def export_results(self, results: list[ValidationResult], output_file: str):
        """Export results to JSON file."""
        data = {
            "timestamp": datetime.utcnow().isoformat(),
            "environment": os.getenv("ENVIRONMENT", "development"),
            "summary": {
                "errors": len([r for r in results if r.level == ValidationLevel.ERROR]),
                "warnings": len([r for r in results if r.level == ValidationLevel.WARNING]),
                "info": len([r for r in results if r.level == ValidationLevel.INFO]),
            },
            "results": [
                {"level": r.level.value, "category": r.category, "message": r.message, "suggestion": r.suggestion} for r in results
            ],
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nResults exported to: {output_file}")


def main():
    """Main validation function."""
    import argparse
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Validate cache configuration")
    parser.add_argument("--env-file", help="Path to environment file")
    parser.add_argument("--export", help="Export results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Only show errors")
    args = parser.parse_args()

    # Create validator
    validator = CacheConfigValidator(env_file=args.env_file)

    # Run validation
    print("Validating cache configuration...")
    is_valid, results = validator.validate()

    # Filter results if quiet mode
    if args.quiet:
        results = [r for r in results if r.level == ValidationLevel.ERROR]

    # Print results
    validator.print_results(results)

    # Export if requested
    if args.export:
        validator.export_results(results, args.export)

    # Exit with appropriate code
    if is_valid:
        print("\n✅ Configuration is valid")
        sys.exit(0)
    else:
        print("\n❌ Configuration has errors")
        sys.exit(1)


if __name__ == "__main__":
    main()
