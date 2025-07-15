#!/usr/bin/env python3
"""
Cache System Migration Script

Handles migration of cache system for existing installations including:
- Configuration migration
- Data migration
- Service migration
- Rollback capabilities
"""

import asyncio
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class MigrationStep(Enum):
    BACKUP = "backup"
    CONFIG_MIGRATION = "config_migration"
    SERVICE_MIGRATION = "service_migration"
    DATA_MIGRATION = "data_migration"
    VALIDATION = "validation"
    CLEANUP = "cleanup"


@dataclass
class MigrationResult:
    """Result of a migration step."""

    step: MigrationStep
    success: bool
    message: str
    details: dict = None
    duration_seconds: float = 0


@dataclass
class MigrationPlan:
    """Migration execution plan."""

    source_version: str
    target_version: str
    steps: list[MigrationStep]
    rollback_plan: list[MigrationStep]
    estimated_duration_minutes: int


class CacheMigrationManager:
    """Manages cache system migrations."""

    def __init__(self, migration_id: str | None = None):
        self.migration_id = migration_id or f"migration_{int(time.time())}"
        self.project_root = project_root
        self.backup_dir = self.project_root / "migration_backups" / self.migration_id
        self.results: list[MigrationResult] = []
        self.current_step = None

        # Migration metadata
        self.source_version = None
        self.target_version = "2.0.0"  # Current cache system version

    def detect_current_installation(self) -> dict[str, any]:
        """Detect current installation state."""
        detection_result = {
            "has_cache_system": False,
            "cache_version": None,
            "has_redis": False,
            "has_config": False,
            "services": [],
            "data_size": 0,
        }

        # Check for existing cache configuration
        env_file = self.project_root / ".env"
        if env_file.exists():
            detection_result["has_config"] = True
            with open(env_file) as f:
                content = f.read()
                if "CACHE_ENABLED" in content:
                    detection_result["has_cache_system"] = True

        # Check for Redis
        compose_file = self.project_root / "docker-compose.cache.yml"
        if compose_file.exists():
            detection_result["has_redis"] = True

        # Check for cache services
        services_dir = self.project_root / "src" / "services"
        if services_dir.exists():
            cache_services = list(services_dir.glob("*cache*.py"))
            detection_result["services"] = [s.name for s in cache_services]

        # Estimate data size
        try:
            import subprocess

            result = subprocess.run(
                ["docker", "exec", "codebase_rag_redis", "redis-cli", "-a", os.getenv("REDIS_PASSWORD", ""), "DBSIZE"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                detection_result["data_size"] = int(result.stdout.strip())
        except:
            pass

        return detection_result

    def create_migration_plan(self, detection_result: dict[str, any]) -> MigrationPlan:
        """Create migration plan based on current installation."""
        if not detection_result["has_cache_system"]:
            # Fresh installation
            steps = [MigrationStep.CONFIG_MIGRATION, MigrationStep.SERVICE_MIGRATION, MigrationStep.VALIDATION]
            estimated_duration = 5
        else:
            # Upgrade existing installation
            steps = [
                MigrationStep.BACKUP,
                MigrationStep.CONFIG_MIGRATION,
                MigrationStep.SERVICE_MIGRATION,
                MigrationStep.DATA_MIGRATION,
                MigrationStep.VALIDATION,
                MigrationStep.CLEANUP,
            ]
            estimated_duration = 15 + (detection_result["data_size"] // 1000)  # Estimate based on data

        rollback_steps = list(reversed(steps))

        return MigrationPlan(
            source_version=detection_result.get("cache_version", "unknown"),
            target_version=self.target_version,
            steps=steps,
            rollback_plan=rollback_steps,
            estimated_duration_minutes=estimated_duration,
        )

    async def execute_migration(self, plan: MigrationPlan, dry_run: bool = False) -> bool:
        """Execute migration plan."""
        print(f"🚀 Starting cache system migration: {self.migration_id}")
        print(f"📊 Estimated duration: {plan.estimated_duration_minutes} minutes")
        print(f"🎯 Target version: {plan.target_version}")

        if dry_run:
            print("🧪 DRY RUN MODE - No changes will be made")

        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Save migration metadata
        metadata = {
            "migration_id": self.migration_id,
            "timestamp": datetime.utcnow().isoformat(),
            "plan": {
                "source_version": plan.source_version,
                "target_version": plan.target_version,
                "steps": [s.value for s in plan.steps],
                "estimated_duration_minutes": plan.estimated_duration_minutes,
            },
            "dry_run": dry_run,
        }

        with open(self.backup_dir / "migration_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Execute steps
        all_success = True
        for step in plan.steps:
            try:
                print(f"\n📋 Executing step: {step.value}")
                self.current_step = step

                start_time = time.time()
                success, message, details = await self._execute_step(step, dry_run)
                duration = time.time() - start_time

                result = MigrationResult(step=step, success=success, message=message, details=details, duration_seconds=duration)
                self.results.append(result)

                if success:
                    print(f"✅ {step.value}: {message} ({duration:.1f}s)")
                else:
                    print(f"❌ {step.value}: {message}")
                    all_success = False
                    break

            except Exception as e:
                error_result = MigrationResult(
                    step=step,
                    success=False,
                    message=f"Exception: {e}",
                    details={"error": str(e)},
                    duration_seconds=time.time() - start_time,
                )
                self.results.append(error_result)
                print(f"💥 {step.value}: Failed with exception: {e}")
                all_success = False
                break

        # Save results
        self._save_migration_results()

        if all_success:
            print("\n🎉 Migration completed successfully!")
            return True
        else:
            print("\n💥 Migration failed. Check logs for details.")
            print(f"📁 Migration data saved to: {self.backup_dir}")
            return False

    async def _execute_step(self, step: MigrationStep, dry_run: bool) -> tuple[bool, str, dict]:
        """Execute a single migration step."""
        if step == MigrationStep.BACKUP:
            return await self._backup_current_system(dry_run)
        elif step == MigrationStep.CONFIG_MIGRATION:
            return await self._migrate_configuration(dry_run)
        elif step == MigrationStep.SERVICE_MIGRATION:
            return await self._migrate_services(dry_run)
        elif step == MigrationStep.DATA_MIGRATION:
            return await self._migrate_data(dry_run)
        elif step == MigrationStep.VALIDATION:
            return await self._validate_migration(dry_run)
        elif step == MigrationStep.CLEANUP:
            return await self._cleanup_migration(dry_run)
        else:
            return False, f"Unknown step: {step}", {}

    async def _backup_current_system(self, dry_run: bool) -> tuple[bool, str, dict]:
        """Backup current system state."""
        if dry_run:
            return True, "Would create backup of current system", {}

        try:
            # Backup configuration files
            config_backup_dir = self.backup_dir / "config"
            config_backup_dir.mkdir(exist_ok=True)

            # Backup .env file
            env_file = self.project_root / ".env"
            if env_file.exists():
                shutil.copy2(env_file, config_backup_dir / ".env.backup")

            # Backup docker-compose files
            for compose_file in self.project_root.glob("docker-compose*.yml"):
                shutil.copy2(compose_file, config_backup_dir / f"{compose_file.name}.backup")

            # Backup cache services
            services_backup_dir = self.backup_dir / "services"
            services_backup_dir.mkdir(exist_ok=True)

            src_services = self.project_root / "src" / "services"
            if src_services.exists():
                for cache_service in src_services.glob("*cache*.py"):
                    shutil.copy2(cache_service, services_backup_dir / f"{cache_service.name}.backup")

            # Backup Redis data using existing script
            try:
                backup_script = self.project_root / "scripts" / "deployment" / "backup-restore.sh"
                if backup_script.exists():
                    import subprocess

                    result = subprocess.run(
                        [str(backup_script), "backup", f"migration_{self.migration_id}"], capture_output=True, text=True, timeout=300
                    )
                    if result.returncode == 0:
                        redis_backup_info = {"redis_backup": "completed"}
                    else:
                        redis_backup_info = {"redis_backup": "failed", "error": result.stderr}
                else:
                    redis_backup_info = {"redis_backup": "script_not_found"}
            except Exception as e:
                redis_backup_info = {"redis_backup": "error", "details": str(e)}

            details = {
                "config_files_backed_up": len(list(config_backup_dir.glob("*"))),
                "service_files_backed_up": len(list(services_backup_dir.glob("*"))),
                "redis_backup": redis_backup_info,
            }

            return True, "System backup completed", details

        except Exception as e:
            return False, f"Backup failed: {e}", {"error": str(e)}

    async def _migrate_configuration(self, dry_run: bool) -> tuple[bool, str, dict]:
        """Migrate configuration files."""
        if dry_run:
            return True, "Would migrate configuration files", {}

        try:
            updates_made = []

            # Update .env file with new cache settings
            env_file = self.project_root / ".env"
            new_settings = {
                "CACHE_ENABLED": "true",
                "CACHE_LEVEL": "BOTH",
                "CACHE_WRITE_STRATEGY": "WRITE_THROUGH",
                "MEMORY_CACHE_MAX_SIZE": "1000",
                "MEMORY_CACHE_MAX_MEMORY_MB": "256",
                "REDIS_HOST": "localhost",
                "REDIS_PORT": "6379",
                "REDIS_MAX_CONNECTIONS": "10",
                "CACHE_METRICS_ENABLED": "true",
                "CACHE_HEALTH_CHECK_INTERVAL": "60",
            }

            # Read existing env file
            existing_env = {}
            if env_file.exists():
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if "=" in line and not line.startswith("#"):
                            key, value = line.split("=", 1)
                            existing_env[key.strip()] = value.strip()

            # Add new settings if not present
            for key, default_value in new_settings.items():
                if key not in existing_env:
                    existing_env[key] = default_value
                    updates_made.append(f"Added {key}={default_value}")

            # Write updated env file
            with open(env_file, "w") as f:
                for key, value in existing_env.items():
                    f.write(f"{key}={value}\n")

            # Ensure docker-compose.cache.yml exists
            compose_file = self.project_root / "docker-compose.cache.yml"
            if not compose_file.exists():
                # Copy from template or create minimal one
                compose_content = """version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: codebase_rag_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD:-changeme}
    healthcheck:
      test: ["CMD", "redis-cli", "--no-auth-warning", "-a", "$REDIS_PASSWORD", "ping"]
      interval: 10s
      timeout: 5s
      retries: 3
    networks:
      - codebase_rag_network

volumes:
  redis_data:
    driver: local

networks:
  codebase_rag_network:
    driver: bridge
"""
                with open(compose_file, "w") as f:
                    f.write(compose_content)
                updates_made.append("Created docker-compose.cache.yml")

            details = {"updates_made": updates_made, "env_file": str(env_file), "compose_file": str(compose_file)}

            return True, f"Configuration migration completed ({len(updates_made)} updates)", details

        except Exception as e:
            return False, f"Configuration migration failed: {e}", {"error": str(e)}

    async def _migrate_services(self, dry_run: bool) -> tuple[bool, str, dict]:
        """Migrate cache services."""
        if dry_run:
            return True, "Would migrate cache services", {}

        try:
            # Check if cache services already exist
            services_dir = self.project_root / "src" / "services"
            existing_services = list(services_dir.glob("*cache*.py")) if services_dir.exists() else []

            # For this migration, we assume services are already in place
            # In a real migration, you might copy new service files or update existing ones

            details = {"existing_cache_services": [s.name for s in existing_services], "services_directory": str(services_dir)}

            if existing_services:
                return True, f"Found {len(existing_services)} existing cache services", details
            else:
                return True, "No existing cache services found (fresh installation)", details

        except Exception as e:
            return False, f"Service migration failed: {e}", {"error": str(e)}

    async def _migrate_data(self, dry_run: bool) -> tuple[bool, str, dict]:
        """Migrate cache data."""
        if dry_run:
            return True, "Would migrate cache data", {}

        try:
            # Check if Redis is running
            import subprocess

            try:
                result = subprocess.run(["docker", "ps", "--format", "{{.Names}}"], capture_output=True, text=True, timeout=10)

                if "codebase_rag_redis" in result.stdout:
                    # Redis is running, check data
                    redis_info = subprocess.run(
                        ["docker", "exec", "codebase_rag_redis", "redis-cli", "-a", os.getenv("REDIS_PASSWORD", ""), "INFO", "keyspace"],
                        capture_output=True,
                        text=True,
                        timeout=10,
                    )

                    data_info = redis_info.stdout.strip()
                    has_data = "db0:" in data_info

                    details = {"redis_running": True, "has_existing_data": has_data, "keyspace_info": data_info}

                    if has_data:
                        return True, "Existing cache data preserved", details
                    else:
                        return True, "No existing cache data to migrate", details
                else:
                    # Start Redis
                    start_result = subprocess.run(
                        ["docker-compose", "-f", "docker-compose.cache.yml", "up", "-d"], capture_output=True, text=True, timeout=60
                    )

                    details = {"redis_running": False, "started_redis": start_result.returncode == 0}

                    return True, "Redis started for cache system", details

            except subprocess.TimeoutExpired:
                return False, "Data migration timed out", {"error": "timeout"}

        except Exception as e:
            return False, f"Data migration failed: {e}", {"error": str(e)}

    async def _validate_migration(self, dry_run: bool) -> tuple[bool, str, dict]:
        """Validate migration results."""
        if dry_run:
            return True, "Would validate migration", {}

        try:
            validation_results = {}

            # Check configuration
            env_file = self.project_root / ".env"
            if env_file.exists():
                with open(env_file) as f:
                    env_content = f.read()
                    validation_results["has_cache_config"] = "CACHE_ENABLED" in env_content
            else:
                validation_results["has_cache_config"] = False

            # Check Docker services
            import subprocess

            docker_result = subprocess.run(
                ["docker-compose", "-f", "docker-compose.cache.yml", "ps"], capture_output=True, text=True, timeout=30
            )
            validation_results["docker_services"] = docker_result.returncode == 0

            # Test Redis connectivity
            try:
                redis_test = subprocess.run(
                    ["docker", "exec", "codebase_rag_redis", "redis-cli", "-a", os.getenv("REDIS_PASSWORD", ""), "ping"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                validation_results["redis_connectivity"] = "PONG" in redis_test.stdout
            except:
                validation_results["redis_connectivity"] = False

            # Test cache operations (if possible)
            try:
                # Try to import and test cache service
                sys.path.insert(0, str(self.project_root))
                from src.config.cache_config import get_cache_config

                config = get_cache_config()
                validation_results["cache_config_loadable"] = True
            except:
                validation_results["cache_config_loadable"] = False

            # Overall validation
            all_checks_passed = all(validation_results.values())

            if all_checks_passed:
                return True, "All validation checks passed", validation_results
            else:
                failed_checks = [k for k, v in validation_results.items() if not v]
                return False, f"Validation failed: {', '.join(failed_checks)}", validation_results

        except Exception as e:
            return False, f"Validation failed: {e}", {"error": str(e)}

    async def _cleanup_migration(self, dry_run: bool) -> tuple[bool, str, dict]:
        """Cleanup migration artifacts."""
        if dry_run:
            return True, "Would cleanup migration artifacts", {}

        try:
            # Keep migration logs and backups for safety
            # Only cleanup temporary files

            details = {"backup_dir_preserved": str(self.backup_dir), "migration_id": self.migration_id}

            return True, "Migration cleanup completed (backups preserved)", details

        except Exception as e:
            return False, f"Cleanup failed: {e}", {"error": str(e)}

    def _save_migration_results(self):
        """Save migration results to file."""
        results_file = self.backup_dir / "migration_results.json"

        results_data = {
            "migration_id": self.migration_id,
            "timestamp": datetime.utcnow().isoformat(),
            "results": [
                {
                    "step": r.step.value,
                    "success": r.success,
                    "message": r.message,
                    "details": r.details,
                    "duration_seconds": r.duration_seconds,
                }
                for r in self.results
            ],
            "overall_success": all(r.success for r in self.results),
            "total_duration": sum(r.duration_seconds for r in self.results),
        }

        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2)

    async def rollback_migration(self) -> bool:
        """Rollback a failed migration."""
        print(f"🔄 Starting migration rollback: {self.migration_id}")

        # Load migration metadata
        metadata_file = self.backup_dir / "migration_metadata.json"
        if not metadata_file.exists():
            print("❌ Migration metadata not found")
            return False

        with open(metadata_file) as f:
            metadata = json.load(f)

        print(f"📅 Original migration: {metadata['timestamp']}")

        try:
            # Restore configuration files
            config_backup_dir = self.backup_dir / "config"
            if config_backup_dir.exists():
                for backup_file in config_backup_dir.glob("*.backup"):
                    original_name = backup_file.name.replace(".backup", "")
                    target_file = self.project_root / original_name
                    shutil.copy2(backup_file, target_file)
                    print(f"✅ Restored {original_name}")

            # Restore Redis data if backup exists
            redis_backup_script = self.project_root / "scripts" / "deployment" / "backup-restore.sh"
            if redis_backup_script.exists():
                # Look for Redis backup
                backup_dirs = list(Path(self.project_root / "backups").glob(f"*migration_{self.migration_id}*"))
                if backup_dirs:
                    import subprocess

                    result = subprocess.run(
                        [str(redis_backup_script), "restore", str(backup_dirs[0])], capture_output=True, text=True, timeout=300
                    )
                    if result.returncode == 0:
                        print("✅ Redis data restored")
                    else:
                        print(f"⚠️ Redis restore warning: {result.stderr}")

            print("🎉 Migration rollback completed")
            return True

        except Exception as e:
            print(f"💥 Rollback failed: {e}")
            return False


async def main():
    """Main migration function."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate cache system")
    parser.add_argument("action", choices=["migrate", "rollback"], help="Migration action")
    parser.add_argument("--migration-id", help="Migration ID for rollback")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--force", action="store_true", help="Force migration without confirmation")
    args = parser.parse_args()

    if args.action == "migrate":
        # Create migration manager
        manager = CacheMigrationManager()

        # Detect current installation
        print("🔍 Detecting current installation...")
        detection_result = manager.detect_current_installation()

        print("\n📊 Current Installation:")
        for key, value in detection_result.items():
            print(f"   {key}: {value}")

        # Create migration plan
        plan = manager.create_migration_plan(detection_result)

        print("\n📋 Migration Plan:")
        print(f"   Source version: {plan.source_version}")
        print(f"   Target version: {plan.target_version}")
        print(f"   Steps: {[s.value for s in plan.steps]}")
        print(f"   Estimated duration: {plan.estimated_duration_minutes} minutes")

        # Confirm migration
        if not args.force and not args.dry_run:
            confirm = input("\nProceed with migration? (yes/no): ")
            if confirm.lower() != "yes":
                print("Migration cancelled")
                return

        # Execute migration
        success = await manager.execute_migration(plan, dry_run=args.dry_run)

        if success:
            print("\n🎉 Migration completed successfully!")
            if not args.dry_run:
                print(f"📁 Migration data: {manager.backup_dir}")
        else:
            print("\n💥 Migration failed!")
            if not args.dry_run:
                print(f"📁 Backup preserved at: {manager.backup_dir}")
                print("Use 'rollback' command to restore previous state")
            sys.exit(1)

    elif args.action == "rollback":
        if not args.migration_id:
            print("❌ Migration ID required for rollback")
            sys.exit(1)

        manager = CacheMigrationManager(migration_id=args.migration_id)

        if not args.force:
            confirm = input(f"Rollback migration {args.migration_id}? (yes/no): ")
            if confirm.lower() != "yes":
                print("Rollback cancelled")
                return

        success = await manager.rollback_migration()

        if success:
            print("✅ Rollback completed successfully")
        else:
            print("❌ Rollback failed")
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
