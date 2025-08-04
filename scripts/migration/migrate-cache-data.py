#!/usr/bin/env python3
"""
Cache Data Migration Tool

Handles data migration between different cache configurations and versions:
- Redis version migrations
- Cache structure migrations
- Data format migrations
- Selective data migration
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class DataMigrationTask:
    """Individual data migration task."""

    name: str
    description: str
    source_pattern: str
    target_pattern: str
    transformation_function: str | None = None
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class MigrationStats:
    """Migration statistics."""

    total_keys: int = 0
    migrated_keys: int = 0
    skipped_keys: int = 0
    failed_keys: int = 0
    duration_seconds: float = 0
    data_size_mb: float = 0


class CacheDataMigrator:
    """Handles cache data migrations."""

    def __init__(self):
        self.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.redis_container = "codebase_rag_redis"
        self.temp_dir = Path(tempfile.mkdtemp(prefix="cache_migration_"))
        self.stats = MigrationStats()

    def __del__(self):
        """Cleanup temporary directory."""
        import shutil

        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _run_redis_command(self, command: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
        """Run Redis command via Docker."""
        full_command = ["docker", "exec", self.redis_container, "redis-cli"]

        if self.redis_password:
            full_command.extend(["-a", self.redis_password])

        full_command.extend(command)

        return subprocess.run(full_command, capture_output=True, text=True, timeout=timeout)

    def check_redis_connectivity(self) -> bool:
        """Check if Redis is accessible."""
        try:
            result = self._run_redis_command(["ping"])
            return "PONG" in result.stdout
        except:
            return False

    def get_cache_info(self) -> dict[str, Any]:
        """Get current cache information."""
        if not self.check_redis_connectivity():
            return {"error": "Redis not accessible"}

        try:
            # Get basic info
            info_result = self._run_redis_command(["INFO", "keyspace"])
            dbsize_result = self._run_redis_command(["DBSIZE"])
            memory_result = self._run_redis_command(["INFO", "memory"])

            info = {
                "total_keys": int(dbsize_result.stdout.strip()) if dbsize_result.returncode == 0 else 0,
                "keyspace_info": info_result.stdout.strip() if info_result.returncode == 0 else "",
                "memory_info": memory_result.stdout.strip() if memory_result.returncode == 0 else "",
            }

            # Analyze key patterns
            scan_result = self._run_redis_command(["--scan", "--pattern", "codebase_rag:*"], timeout=60)
            if scan_result.returncode == 0:
                keys = scan_result.stdout.strip().split("\n")
                keys = [k for k in keys if k]  # Remove empty strings

                # Categorize keys
                patterns = {
                    "embedding": len([k for k in keys if ":embedding:" in k]),
                    "search": len([k for k in keys if ":search:" in k]),
                    "project": len([k for k in keys if ":project:" in k]),
                    "file": len([k for k in keys if ":file:" in k]),
                    "other": len([k for k in keys if not any(p in k for p in [":embedding:", ":search:", ":project:", ":file:"])]),
                }

                info["key_patterns"] = patterns
                info["cache_keys"] = len(keys)
            else:
                info["key_patterns"] = {}
                info["cache_keys"] = 0

            return info

        except Exception as e:
            return {"error": str(e)}

    def detect_migration_needs(self) -> list[DataMigrationTask]:
        """Detect what data migrations are needed."""
        tasks = []
        cache_info = self.get_cache_info()

        if "error" in cache_info:
            return tasks

        # Check for old key formats that need migration
        if cache_info.get("cache_keys", 0) > 0:
            # Scan for keys that might need format updates
            try:
                # Look for keys without proper versioning
                scan_result = self._run_redis_command(["--scan", "--pattern", "*"], timeout=60)
                if scan_result.returncode == 0:
                    all_keys = scan_result.stdout.strip().split("\n")

                    # Check for keys without codebase_rag prefix
                    old_format_keys = [k for k in all_keys if k and not k.startswith("codebase_rag:")]
                    if old_format_keys:
                        tasks.append(
                            DataMigrationTask(
                                name="key_prefix_migration",
                                description=f"Add codebase_rag prefix to {len(old_format_keys)} keys",
                                source_pattern="*",
                                target_pattern="codebase_rag:*",
                                priority=1,
                            )
                        )

                    # Check for keys with old structure
                    embedding_keys = [k for k in all_keys if "embedding" in k and ":embedding:" not in k]
                    if embedding_keys:
                        tasks.append(
                            DataMigrationTask(
                                name="embedding_structure_migration",
                                description=f"Update structure for {len(embedding_keys)} embedding keys",
                                source_pattern="*embedding*",
                                target_pattern="codebase_rag:embedding:*",
                                priority=1,
                            )
                        )

                    # Check for search cache format
                    search_keys = [k for k in all_keys if "search" in k and ":search:" not in k]
                    if search_keys:
                        tasks.append(
                            DataMigrationTask(
                                name="search_structure_migration",
                                description=f"Update structure for {len(search_keys)} search keys",
                                source_pattern="*search*",
                                target_pattern="codebase_rag:search:*",
                                priority=2,
                            )
                        )

            except Exception:
                pass

        # Add general optimization tasks
        if cache_info.get("total_keys", 0) > 1000:
            tasks.append(
                DataMigrationTask(
                    name="key_optimization",
                    description="Optimize key structure for large dataset",
                    source_pattern="codebase_rag:*",
                    target_pattern="codebase_rag:*",
                    transformation_function="optimize_keys",
                    priority=3,
                )
            )

        return tasks

    async def migrate_data(self, tasks: list[DataMigrationTask], dry_run: bool = False) -> MigrationStats:
        """Execute data migration tasks."""
        if dry_run:
            pass

        start_time = time.time()
        total_stats = MigrationStats()

        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda t: t.priority)

        for i, task in enumerate(sorted_tasks, 1):
            if dry_run:
                continue

            try:
                task_stats = await self._execute_migration_task(task)

                # Aggregate stats
                total_stats.total_keys += task_stats.total_keys
                total_stats.migrated_keys += task_stats.migrated_keys
                total_stats.skipped_keys += task_stats.skipped_keys
                total_stats.failed_keys += task_stats.failed_keys
                total_stats.data_size_mb += task_stats.data_size_mb

                if task_stats.failed_keys > 0:
                    pass

            except Exception:
                total_stats.failed_keys += 1

        total_stats.duration_seconds = time.time() - start_time

        return total_stats

    async def _execute_migration_task(self, task: DataMigrationTask) -> MigrationStats:
        """Execute a single migration task."""
        stats = MigrationStats()

        # Get keys matching the source pattern
        scan_result = self._run_redis_command(["--scan", "--pattern", task.source_pattern], timeout=120)
        if scan_result.returncode != 0:
            raise Exception(f"Failed to scan keys: {scan_result.stderr}")

        keys = scan_result.stdout.strip().split("\n")
        keys = [k for k in keys if k and self._should_migrate_key(k, task)]
        stats.total_keys = len(keys)

        if stats.total_keys == 0:
            return stats

        # Process keys in batches
        batch_size = 100
        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]
            batch_stats = await self._migrate_key_batch(batch, task)

            stats.migrated_keys += batch_stats.migrated_keys
            stats.skipped_keys += batch_stats.skipped_keys
            stats.failed_keys += batch_stats.failed_keys
            stats.data_size_mb += batch_stats.data_size_mb

        return stats

    def _should_migrate_key(self, key: str, task: DataMigrationTask) -> bool:
        """Check if a key should be migrated based on task criteria."""
        if task.name == "key_prefix_migration":
            return not key.startswith("codebase_rag:")
        elif task.name == "embedding_structure_migration":
            return "embedding" in key and ":embedding:" not in key
        elif task.name == "search_structure_migration":
            return "search" in key and ":search:" not in key
        elif task.name == "key_optimization":
            # Only migrate if key is not already optimized
            return True  # Implement specific optimization logic
        else:
            return True

    async def _migrate_key_batch(self, keys: list[str], task: DataMigrationTask) -> MigrationStats:
        """Migrate a batch of keys."""
        stats = MigrationStats()

        for key in keys:
            try:
                # Get current value
                get_result = self._run_redis_command(["GET", key])
                if get_result.returncode != 0:
                    stats.failed_keys += 1
                    continue

                value = get_result.stdout.strip()
                if not value:
                    stats.skipped_keys += 1
                    continue

                # Get TTL
                ttl_result = self._run_redis_command(["TTL", key])
                ttl = int(ttl_result.stdout.strip()) if ttl_result.returncode == 0 else -1

                # Transform key and value
                new_key, new_value = self._transform_key_value(key, value, task)

                if new_key == key and new_value == value:
                    stats.skipped_keys += 1
                    continue

                # Set new key
                set_command = ["SET", new_key, new_value]
                if ttl > 0:
                    set_command.extend(["EX", str(ttl)])

                set_result = self._run_redis_command(set_command)
                if set_result.returncode != 0:
                    stats.failed_keys += 1
                    continue

                # Delete old key if different
                if new_key != key:
                    delete_result = self._run_redis_command(["DEL", key])
                    if delete_result.returncode != 0:
                        pass

                stats.migrated_keys += 1
                stats.data_size_mb += len(value) / (1024 * 1024)

            except Exception:
                stats.failed_keys += 1

        return stats

    def _transform_key_value(self, key: str, value: str, task: DataMigrationTask) -> tuple[str, str]:
        """Transform key and value according to migration task."""
        new_key = key
        new_value = value

        if task.name == "key_prefix_migration":
            if not key.startswith("codebase_rag:"):
                new_key = f"codebase_rag:{key}"

        elif task.name == "embedding_structure_migration":
            if "embedding" in key and ":embedding:" not in key:
                # Transform key to new structure
                parts = key.split(":")
                if len(parts) >= 2:
                    new_key = f"codebase_rag:embedding:{':'.join(parts[1:])}"
                else:
                    new_key = f"codebase_rag:embedding:{key}"

        elif task.name == "search_structure_migration":
            if "search" in key and ":search:" not in key:
                parts = key.split(":")
                if len(parts) >= 2:
                    new_key = f"codebase_rag:search:{':'.join(parts[1:])}"
                else:
                    new_key = f"codebase_rag:search:{key}"

        elif task.name == "key_optimization":
            # Implement key optimization logic
            new_key, new_value = self._optimize_key_value(key, value)

        return new_key, new_value

    def _optimize_key_value(self, key: str, value: str) -> tuple[str, str]:
        """Optimize key and value structure."""
        new_key = key
        new_value = value

        try:
            # Try to parse value as JSON and optimize structure
            if value.startswith("{") or value.startswith("["):
                data = json.loads(value)

                # Add version information if missing
                if isinstance(data, dict) and "version" not in data:
                    data["version"] = "2.0.0"
                    new_value = json.dumps(data, separators=(",", ":"))

                # Optimize timestamps
                if isinstance(data, dict) and "timestamp" in data:
                    # Ensure timestamp is in ISO format
                    timestamp = data["timestamp"]
                    if isinstance(timestamp, int | float):
                        data["timestamp"] = datetime.fromtimestamp(timestamp).isoformat()
                        new_value = json.dumps(data, separators=(",", ":"))

        except (json.JSONDecodeError, ValueError):
            # Not JSON data, leave as is
            pass

        return new_key, new_value

    def export_keys(self, pattern: str, output_file: str) -> int:
        """Export keys matching pattern to file."""

        if not self.check_redis_connectivity():
            raise Exception("Redis not accessible")

        # Scan for keys
        scan_result = self._run_redis_command(["--scan", "--pattern", pattern], timeout=300)
        if scan_result.returncode != 0:
            raise Exception(f"Failed to scan keys: {scan_result.stderr}")

        keys = scan_result.stdout.strip().split("\n")
        keys = [k for k in keys if k]

        # Export data
        export_data = {"export_timestamp": datetime.utcnow().isoformat(), "pattern": pattern, "key_count": len(keys), "data": {}}

        batch_size = 100
        for i in range(0, len(keys), batch_size):
            batch = keys[i : i + batch_size]

            for key in batch:
                try:
                    # Get value
                    get_result = self._run_redis_command(["GET", key])
                    if get_result.returncode == 0:
                        value = get_result.stdout.strip()

                        # Get TTL
                        ttl_result = self._run_redis_command(["TTL", key])
                        ttl = int(ttl_result.stdout.strip()) if ttl_result.returncode == 0 else -1

                        export_data["data"][key] = {"value": value, "ttl": ttl}

                except Exception:
                    pass

        # Save to file
        with open(output_file, "w") as f:
            json.dump(export_data, f, indent=2)

        return len(export_data["data"])

    def import_keys(self, input_file: str, overwrite: bool = False) -> int:
        """Import keys from file."""

        if not self.check_redis_connectivity():
            raise Exception("Redis not accessible")

        with open(input_file) as f:
            import_data = json.load(f)

        keys_data = import_data.get("data", {})

        imported_count = 0
        skipped_count = 0

        for key, data in keys_data.items():
            try:
                # Check if key exists
                if not overwrite:
                    exists_result = self._run_redis_command(["EXISTS", key])
                    if exists_result.returncode == 0 and exists_result.stdout.strip() == "1":
                        skipped_count += 1
                        continue

                value = data["value"]
                ttl = data.get("ttl", -1)

                # Set key
                set_command = ["SET", key, value]
                if ttl > 0:
                    set_command.extend(["EX", str(ttl)])

                set_result = self._run_redis_command(set_command)
                if set_result.returncode == 0:
                    imported_count += 1

            except Exception:
                pass

        if skipped_count > 0:
            pass

        return imported_count


async def main():
    """Main migration function."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate cache data")
    parser.add_argument("action", choices=["analyze", "migrate", "export", "import"], help="Migration action")
    parser.add_argument("--pattern", default="codebase_rag:*", help="Key pattern for operations")
    parser.add_argument("--file", help="File for export/import operations")
    parser.add_argument("--dry-run", action="store_true", help="Dry run mode")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing keys on import")
    args = parser.parse_args()

    migrator = CacheDataMigrator()

    if args.action == "analyze":
        # Check connectivity
        if not migrator.check_redis_connectivity():
            sys.exit(1)

        # Get cache info
        info = migrator.get_cache_info()
        if "error" in info:
            sys.exit(1)

        if info.get("key_patterns"):
            for pattern, count in info["key_patterns"].items():
                pass

        # Detect migration needs
        tasks = migrator.detect_migration_needs()
        if tasks:
            for task in tasks:
                pass
        else:
            pass

    elif args.action == "migrate":
        # Check connectivity
        if not migrator.check_redis_connectivity():
            sys.exit(1)

        # Detect migration tasks
        tasks = migrator.detect_migration_needs()

        if not tasks:
            return

        for task in tasks:
            pass

        if not args.dry_run:
            confirm = input("\nProceed with migration? (yes/no): ")
            if confirm.lower() != "yes":
                return

        # Execute migration
        stats = await migrator.migrate_data(tasks, dry_run=args.dry_run)

        if stats.failed_keys == 0:
            pass
        else:
            pass

    elif args.action == "export":
        if not args.file:
            sys.exit(1)

        try:
            migrator.export_keys(args.pattern, args.file)
        except Exception:
            sys.exit(1)

    elif args.action == "import":
        if not args.file:
            sys.exit(1)

        try:
            migrator.import_keys(args.file, overwrite=args.overwrite)
        except Exception:
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
