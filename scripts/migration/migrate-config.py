#!/usr/bin/env python3
"""
Configuration Migration Tool

Handles migration of configuration files between different versions:
- Environment variable migrations
- Docker Compose migrations
- Configuration file format migrations
- Settings validation and updates
"""

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class ConfigMigrationRule:
    """Configuration migration rule."""

    old_key: str
    new_key: str | None = None
    default_value: str | None = None
    transformation: str | None = None  # Function name for value transformation
    required: bool = False
    deprecated: bool = False


@dataclass
class MigrationResult:
    """Result of configuration migration."""

    file_path: str
    success: bool
    changes_made: list[str]
    warnings: list[str]
    errors: list[str]


class ConfigurationMigrator:
    """Handles configuration file migrations."""

    def __init__(self):
        self.project_root = project_root
        self.backup_dir = self.project_root / "config_backups"
        self.migration_rules = self._load_migration_rules()

    def _load_migration_rules(self) -> list[ConfigMigrationRule]:
        """Load configuration migration rules."""
        return [
            # Cache system settings
            ConfigMigrationRule(old_key="CACHE_ENABLE", new_key="CACHE_ENABLED", default_value="true"),
            ConfigMigrationRule(old_key="CACHE_TYPE", new_key="CACHE_LEVEL", transformation="transform_cache_type"),
            ConfigMigrationRule(old_key="REDIS_URL", new_key=None, transformation="split_redis_url"),
            # Memory cache settings
            ConfigMigrationRule(old_key="MEMORY_CACHE_SIZE", new_key="MEMORY_CACHE_MAX_SIZE", default_value="1000"),
            ConfigMigrationRule(old_key="MEMORY_CACHE_MEMORY", new_key="MEMORY_CACHE_MAX_MEMORY_MB", default_value="256"),
            # Redis settings
            ConfigMigrationRule(old_key="REDIS_SERVER", new_key="REDIS_HOST", default_value="localhost"),
            ConfigMigrationRule(old_key="REDIS_PASSWORD", new_key="REDIS_PASSWORD", required=True),
            # New required settings
            ConfigMigrationRule(old_key="CACHE_WRITE_STRATEGY", new_key="CACHE_WRITE_STRATEGY", default_value="WRITE_THROUGH"),
            ConfigMigrationRule(old_key="CACHE_METRICS_ENABLED", new_key="CACHE_METRICS_ENABLED", default_value="true"),
            ConfigMigrationRule(old_key="CACHE_HEALTH_CHECK_INTERVAL", new_key="CACHE_HEALTH_CHECK_INTERVAL", default_value="60"),
            # Cache type specific settings
            ConfigMigrationRule(old_key="EMBEDDING_CACHE_TTL", new_key="EMBEDDING_CACHE_TTL", default_value="7200"),
            ConfigMigrationRule(old_key="SEARCH_CACHE_TTL", new_key="SEARCH_CACHE_TTL", default_value="1800"),
            ConfigMigrationRule(old_key="PROJECT_CACHE_TTL", new_key="PROJECT_CACHE_TTL", default_value="3600"),
            ConfigMigrationRule(old_key="FILE_CACHE_TTL", new_key="FILE_CACHE_TTL", default_value="1800"),
            # Security settings
            ConfigMigrationRule(old_key="CACHE_ENCRYPTION", new_key="CACHE_ENCRYPTION_ENABLED", transformation="transform_boolean"),
            ConfigMigrationRule(old_key="CACHE_PROJECT_ISOLATION", new_key="CACHE_PROJECT_ISOLATION", default_value="true"),
            # Deprecated settings
            ConfigMigrationRule(
                old_key="CACHE_DEBUG", new_key="CACHE_LOG_LEVEL", transformation="transform_debug_to_log_level", deprecated=True
            ),
            ConfigMigrationRule(old_key="REDIS_TIMEOUT", new_key="REDIS_CONNECTION_TIMEOUT", deprecated=True),
        ]

    def create_backup(self, file_path: Path) -> Path:
        """Create backup of configuration file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"{file_path.name}.{timestamp}.backup"

        self.backup_dir.mkdir(exist_ok=True)
        backup_path = self.backup_dir / backup_name

        shutil.copy2(file_path, backup_path)
        return backup_path

    def migrate_env_file(self, env_path: Path | None = None) -> MigrationResult:
        """Migrate .env file configuration."""
        if env_path is None:
            env_path = self.project_root / ".env"

        result = MigrationResult(file_path=str(env_path), success=True, changes_made=[], warnings=[], errors=[])

        if not env_path.exists():
            # Create new .env file from template
            return self._create_new_env_file(env_path)

        # Backup existing file
        backup_path = self.create_backup(env_path)
        result.changes_made.append(f"Created backup: {backup_path}")

        # Read existing configuration
        existing_config = self._read_env_file(env_path)
        new_config = {}

        # Apply migration rules
        for rule in self.migration_rules:
            try:
                if rule.old_key in existing_config:
                    old_value = existing_config[rule.old_key]

                    if rule.deprecated:
                        result.warnings.append(f"Deprecated setting: {rule.old_key}")

                    if rule.transformation:
                        # Apply transformation
                        new_values = self._apply_transformation(rule.transformation, rule.old_key, old_value)
                        for key, value in new_values.items():
                            new_config[key] = value
                            result.changes_made.append(f"Transformed {rule.old_key} -> {key}: {value}")
                    elif rule.new_key:
                        # Direct mapping
                        new_config[rule.new_key] = old_value
                        if rule.old_key != rule.new_key:
                            result.changes_made.append(f"Renamed {rule.old_key} -> {rule.new_key}")
                elif rule.new_key and rule.default_value:
                    # Add missing setting with default value
                    new_config[rule.new_key] = rule.default_value
                    result.changes_made.append(f"Added {rule.new_key}: {rule.default_value}")
                elif rule.required and rule.new_key:
                    # Missing required setting
                    result.errors.append(f"Missing required setting: {rule.new_key}")
                    result.success = False

            except Exception as e:
                result.errors.append(f"Error processing {rule.old_key}: {e}")
                result.success = False

        # Preserve existing settings not covered by migration rules
        for key, value in existing_config.items():
            if key not in [rule.old_key for rule in self.migration_rules]:
                if key not in new_config:
                    new_config[key] = value

        # Write updated configuration
        if result.success:
            self._write_env_file(env_path, new_config)
            result.changes_made.append(f"Updated {env_path}")

        return result

    def _create_new_env_file(self, env_path: Path) -> MigrationResult:
        """Create new .env file with default configuration."""
        result = MigrationResult(file_path=str(env_path), success=True, changes_made=[], warnings=[], errors=[])

        # Default configuration
        default_config = {
            "CACHE_ENABLED": "true",
            "CACHE_LEVEL": "BOTH",
            "CACHE_WRITE_STRATEGY": "WRITE_THROUGH",
            "CACHE_DEFAULT_TTL": "3600",
            # Redis configuration
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379",
            "REDIS_PASSWORD": "changeme",
            "REDIS_MAX_CONNECTIONS": "10",
            "REDIS_CONNECTION_TIMEOUT": "5.0",
            # Memory cache configuration
            "MEMORY_CACHE_MAX_SIZE": "1000",
            "MEMORY_CACHE_MAX_MEMORY_MB": "256",
            "MEMORY_CACHE_TTL": "3600",
            # Cache type specific settings
            "EMBEDDING_CACHE_TTL": "7200",
            "SEARCH_CACHE_TTL": "1800",
            "PROJECT_CACHE_TTL": "3600",
            "FILE_CACHE_TTL": "1800",
            # Security settings
            "CACHE_ENCRYPTION_ENABLED": "false",
            "CACHE_PROJECT_ISOLATION": "true",
            "CACHE_ACCESS_LOGGING": "false",
            # Monitoring settings
            "CACHE_METRICS_ENABLED": "true",
            "CACHE_HEALTH_CHECK_INTERVAL": "60",
            "CACHE_LOG_LEVEL": "INFO",
        }

        self._write_env_file(env_path, default_config)
        result.changes_made.append(f"Created new .env file: {env_path}")

        return result

    def migrate_docker_compose(self, compose_path: Path | None = None) -> MigrationResult:
        """Migrate Docker Compose configuration."""
        if compose_path is None:
            compose_path = self.project_root / "docker-compose.cache.yml"

        result = MigrationResult(file_path=str(compose_path), success=True, changes_made=[], warnings=[], errors=[])

        if not compose_path.exists():
            return self._create_new_docker_compose(compose_path)

        # Backup existing file
        backup_path = self.create_backup(compose_path)
        result.changes_made.append(f"Created backup: {backup_path}")

        try:
            # Read existing compose file
            with open(compose_path) as f:
                compose_data = yaml.safe_load(f)

            # Migrate compose configuration
            updated = False

            # Update Redis service configuration
            if "services" in compose_data and "redis" in compose_data["services"]:
                redis_service = compose_data["services"]["redis"]

                # Update image version
                if redis_service.get("image", "").startswith("redis:6"):
                    redis_service["image"] = "redis:7-alpine"
                    result.changes_made.append("Updated Redis image to 7-alpine")
                    updated = True

                # Add health check if missing
                if "healthcheck" not in redis_service:
                    redis_service["healthcheck"] = {
                        "test": ["CMD", "redis-cli", "--no-auth-warning", "-a", "$REDIS_PASSWORD", "ping"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 3,
                        "start_period": "30s",
                    }
                    result.changes_made.append("Added Redis health check")
                    updated = True

                # Add resource limits if missing
                if "mem_limit" not in redis_service:
                    redis_service["mem_limit"] = "512m"
                    redis_service["mem_reservation"] = "256m"
                    redis_service["cpus"] = 0.5
                    result.changes_made.append("Added Redis resource limits")
                    updated = True

                # Ensure networks configuration
                if "networks" not in redis_service:
                    redis_service["networks"] = ["codebase_rag_network"]
                    result.changes_made.append("Added Redis network configuration")
                    updated = True

            # Ensure networks section exists
            if "networks" not in compose_data:
                compose_data["networks"] = {"codebase_rag_network": {"driver": "bridge"}}
                result.changes_made.append("Added networks configuration")
                updated = True

            # Ensure volumes section exists
            if "volumes" not in compose_data:
                compose_data["volumes"] = {"redis_data": {"driver": "local"}}
                result.changes_made.append("Added volumes configuration")
                updated = True

            # Write updated compose file
            if updated:
                with open(compose_path, "w") as f:
                    yaml.dump(compose_data, f, default_flow_style=False, indent=2)
                result.changes_made.append(f"Updated {compose_path}")

        except Exception as e:
            result.errors.append(f"Error migrating Docker Compose: {e}")
            result.success = False

        return result

    def _create_new_docker_compose(self, compose_path: Path) -> MigrationResult:
        """Create new Docker Compose file."""
        result = MigrationResult(file_path=str(compose_path), success=True, changes_made=[], warnings=[], errors=[])

        compose_data = {
            "version": "3.8",
            "services": {
                "redis": {
                    "image": "redis:7-alpine",
                    "container_name": "codebase_rag_redis",
                    "restart": "unless-stopped",
                    "ports": ["6379:6379"],
                    "volumes": ["redis_data:/data", "./redis.conf:/usr/local/etc/redis/redis.conf"],
                    "command": "redis-server /usr/local/etc/redis/redis.conf",
                    "environment": ["REDIS_PASSWORD=${REDIS_PASSWORD:-changeme}"],
                    "healthcheck": {
                        "test": ["CMD", "redis-cli", "--no-auth-warning", "-a", "$REDIS_PASSWORD", "ping"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 3,
                        "start_period": "30s",
                    },
                    "networks": ["codebase_rag_network"],
                    "mem_limit": "512m",
                    "mem_reservation": "256m",
                    "cpus": 0.5,
                },
                "redis-commander": {
                    "image": "rediscommander/redis-commander:latest",
                    "container_name": "codebase_rag_redis_commander",
                    "restart": "unless-stopped",
                    "depends_on": {"redis": {"condition": "service_healthy"}},
                    "ports": ["8081:8081"],
                    "environment": ["REDIS_HOSTS=local:redis:6379", "REDIS_PASSWORD=${REDIS_PASSWORD:-changeme}"],
                    "networks": ["codebase_rag_network"],
                    "profiles": ["debug"],
                },
            },
            "volumes": {"redis_data": {"driver": "local"}},
            "networks": {"codebase_rag_network": {"driver": "bridge"}},
        }

        with open(compose_path, "w") as f:
            yaml.dump(compose_data, f, default_flow_style=False, indent=2)

        result.changes_made.append(f"Created new Docker Compose file: {compose_path}")
        return result

    def _read_env_file(self, env_path: Path) -> dict[str, str]:
        """Read environment file into dictionary."""
        config = {}

        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip().strip("\"'")

        return config

    def _write_env_file(self, env_path: Path, config: dict[str, str]):
        """Write configuration dictionary to environment file."""
        with open(env_path, "w") as f:
            f.write("# Cache System Configuration\n")
            f.write(f"# Generated on {datetime.now().isoformat()}\n\n")

            # Group settings by category
            categories = {
                "Core Cache Settings": ["CACHE_ENABLED", "CACHE_LEVEL", "CACHE_WRITE_STRATEGY", "CACHE_DEFAULT_TTL"],
                "Redis Configuration": [
                    "REDIS_HOST",
                    "REDIS_PORT",
                    "REDIS_PASSWORD",
                    "REDIS_MAX_CONNECTIONS",
                    "REDIS_CONNECTION_TIMEOUT",
                    "REDIS_SSL_ENABLED",
                ],
                "Memory Cache Settings": ["MEMORY_CACHE_MAX_SIZE", "MEMORY_CACHE_MAX_MEMORY_MB", "MEMORY_CACHE_TTL"],
                "Cache Type Settings": ["EMBEDDING_CACHE_TTL", "SEARCH_CACHE_TTL", "PROJECT_CACHE_TTL", "FILE_CACHE_TTL"],
                "Security Settings": ["CACHE_ENCRYPTION_ENABLED", "CACHE_PROJECT_ISOLATION", "CACHE_ACCESS_LOGGING"],
                "Monitoring Settings": ["CACHE_METRICS_ENABLED", "CACHE_HEALTH_CHECK_INTERVAL", "CACHE_LOG_LEVEL"],
            }

            written_keys = set()

            for category, keys in categories.items():
                category_written = False
                for key in keys:
                    if key in config:
                        if not category_written:
                            f.write(f"# {category}\n")
                            category_written = True
                        f.write(f"{key}={config[key]}\n")
                        written_keys.add(key)
                if category_written:
                    f.write("\n")

            # Write remaining settings
            remaining = {k: v for k, v in config.items() if k not in written_keys}
            if remaining:
                f.write("# Additional Settings\n")
                for key, value in remaining.items():
                    f.write(f"{key}={value}\n")

    def _apply_transformation(self, transformation: str, key: str, value: str) -> dict[str, str]:
        """Apply value transformation."""
        if transformation == "transform_cache_type":
            # Convert old cache type values to new cache level
            type_mapping = {"memory": "L1_MEMORY", "redis": "L2_REDIS", "both": "BOTH", "all": "BOTH"}
            new_value = type_mapping.get(value.lower(), "BOTH")
            return {"CACHE_LEVEL": new_value}

        elif transformation == "split_redis_url":
            # Split Redis URL into host, port, password
            result = {}
            if value.startswith("redis://"):
                # Parse redis://[:password@]host[:port][/db]
                url_part = value[8:]  # Remove redis://

                if "@" in url_part:
                    auth_part, host_part = url_part.split("@", 1)
                    if ":" in auth_part:
                        user, password = auth_part.split(":", 1)
                        result["REDIS_PASSWORD"] = password
                    else:
                        result["REDIS_PASSWORD"] = auth_part
                else:
                    host_part = url_part

                if ":" in host_part:
                    host, port_db = host_part.split(":", 1)
                    if "/" in port_db:
                        port, db = port_db.split("/", 1)
                        result["REDIS_DB"] = db
                    else:
                        port = port_db
                    result["REDIS_PORT"] = port
                else:
                    host = host_part

                result["REDIS_HOST"] = host
            else:
                # Assume it's just a host
                result["REDIS_HOST"] = value

            return result

        elif transformation == "transform_boolean":
            # Convert various boolean representations
            bool_value = value.lower() in ["true", "1", "yes", "on", "enabled"]
            return {key.replace("CACHE_ENCRYPTION", "CACHE_ENCRYPTION_ENABLED"): str(bool_value).lower()}

        elif transformation == "transform_debug_to_log_level":
            # Convert debug flag to log level
            if value.lower() in ["true", "1", "yes", "on"]:
                return {"CACHE_LOG_LEVEL": "DEBUG"}
            else:
                return {"CACHE_LOG_LEVEL": "INFO"}

        else:
            # Unknown transformation
            return {key: value}

    def validate_configuration(self, env_path: Path | None = None) -> list[str]:
        """Validate migrated configuration."""
        if env_path is None:
            env_path = self.project_root / ".env"

        issues = []

        if not env_path.exists():
            issues.append("Environment file does not exist")
            return issues

        config = self._read_env_file(env_path)

        # Check required settings
        required_settings = ["CACHE_ENABLED", "REDIS_HOST", "REDIS_PORT", "REDIS_PASSWORD"]

        for setting in required_settings:
            if setting not in config:
                issues.append(f"Missing required setting: {setting}")
            elif not config[setting]:
                issues.append(f"Empty required setting: {setting}")

        # Validate specific values
        if "CACHE_LEVEL" in config:
            valid_levels = ["L1_MEMORY", "L2_REDIS", "BOTH"]
            if config["CACHE_LEVEL"] not in valid_levels:
                issues.append(f"Invalid CACHE_LEVEL: {config['CACHE_LEVEL']}")

        if "CACHE_WRITE_STRATEGY" in config:
            valid_strategies = ["WRITE_THROUGH", "WRITE_BACK", "WRITE_AROUND"]
            if config["CACHE_WRITE_STRATEGY"] not in valid_strategies:
                issues.append(f"Invalid CACHE_WRITE_STRATEGY: {config['CACHE_WRITE_STRATEGY']}")

        # Check numeric values
        numeric_settings = [
            "REDIS_PORT",
            "REDIS_MAX_CONNECTIONS",
            "MEMORY_CACHE_MAX_SIZE",
            "MEMORY_CACHE_MAX_MEMORY_MB",
            "CACHE_DEFAULT_TTL",
        ]

        for setting in numeric_settings:
            if setting in config:
                try:
                    int(config[setting])
                except ValueError:
                    issues.append(f"Invalid numeric value for {setting}: {config[setting]}")

        return issues


def main():
    """Main configuration migration function."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate cache configuration")
    parser.add_argument("action", choices=["migrate", "validate", "backup"], help="Migration action")
    parser.add_argument("--env-file", help="Path to environment file")
    parser.add_argument("--compose-file", help="Path to Docker Compose file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed")
    args = parser.parse_args()

    migrator = ConfigurationMigrator()

    if args.action == "migrate":
        print("🚀 Starting configuration migration...")

        # Migrate environment file
        env_path = Path(args.env_file) if args.env_file else None
        env_result = migrator.migrate_env_file(env_path)

        print("\n📄 Environment File Migration:")
        print(f"   File: {env_result.file_path}")
        print(f"   Success: {'✅' if env_result.success else '❌'}")

        if env_result.changes_made:
            print("   Changes:")
            for change in env_result.changes_made:
                print(f"     - {change}")

        if env_result.warnings:
            print("   Warnings:")
            for warning in env_result.warnings:
                print(f"     ⚠️ {warning}")

        if env_result.errors:
            print("   Errors:")
            for error in env_result.errors:
                print(f"     ❌ {error}")

        # Migrate Docker Compose file
        compose_path = Path(args.compose_file) if args.compose_file else None
        compose_result = migrator.migrate_docker_compose(compose_path)

        print("\n🐳 Docker Compose Migration:")
        print(f"   File: {compose_result.file_path}")
        print(f"   Success: {'✅' if compose_result.success else '❌'}")

        if compose_result.changes_made:
            print("   Changes:")
            for change in compose_result.changes_made:
                print(f"     - {change}")

        if compose_result.warnings:
            print("   Warnings:")
            for warning in compose_result.warnings:
                print(f"     ⚠️ {warning}")

        if compose_result.errors:
            print("   Errors:")
            for error in compose_result.errors:
                print(f"     ❌ {error}")

        # Overall result
        overall_success = env_result.success and compose_result.success
        print(f"\n🎯 Overall Migration: {'✅ Success' if overall_success else '❌ Failed'}")

        if not overall_success:
            sys.exit(1)

    elif args.action == "validate":
        print("🔍 Validating configuration...")

        env_path = Path(args.env_file) if args.env_file else None
        issues = migrator.validate_configuration(env_path)

        if issues:
            print("❌ Configuration issues found:")
            for issue in issues:
                print(f"   - {issue}")
            sys.exit(1)
        else:
            print("✅ Configuration is valid")

    elif args.action == "backup":
        print("💾 Creating configuration backup...")

        backup_files = []

        # Backup .env file
        env_path = Path(args.env_file) if args.env_file else migrator.project_root / ".env"
        if env_path.exists():
            backup_path = migrator.create_backup(env_path)
            backup_files.append(backup_path)
            print(f"   Created: {backup_path}")

        # Backup Docker Compose file
        compose_path = Path(args.compose_file) if args.compose_file else migrator.project_root / "docker-compose.cache.yml"
        if compose_path.exists():
            backup_path = migrator.create_backup(compose_path)
            backup_files.append(backup_path)
            print(f"   Created: {backup_path}")

        if backup_files:
            print(f"✅ Created {len(backup_files)} backup files")
        else:
            print("⚠️ No configuration files found to backup")


if __name__ == "__main__":
    main()
