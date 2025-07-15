"""
Advanced cache key optimization and compression strategies.

This module provides sophisticated key generation, compression, and optimization
techniques to minimize memory usage and improve cache performance.
"""

import base64
import hashlib
import json
import logging
import pickle
import time
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class KeyCompressionStrategy(Enum):
    """Cache key compression strategies."""

    NONE = "none"
    HASH_MD5 = "hash_md5"
    HASH_SHA256 = "hash_sha256"
    ZLIB = "zlib"
    BASE64 = "base64"
    CUSTOM = "custom"


class SerializationFormat(Enum):
    """Serialization formats for complex keys."""

    JSON = "json"
    PICKLE = "pickle"
    STRING = "string"
    BINARY = "binary"


@dataclass
class KeyOptimizationStats:
    """Statistics for key optimization performance."""

    original_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    compression_time: float = 0.0
    hash_collisions: int = 0
    total_keys: int = 0

    def update(self, original: int, compressed: int, time_taken: float):
        """Update optimization statistics."""
        self.original_size += original
        self.compressed_size += compressed
        self.total_keys += 1
        self.compression_time += time_taken

        if self.original_size > 0:
            self.compression_ratio = 1.0 - (self.compressed_size / self.original_size)


class BaseKeyOptimizer(ABC):
    """Base class for cache key optimizers."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize key optimizer."""
        self.logger = logger or logging.getLogger(__name__)
        self.stats = KeyOptimizationStats()
        self._key_map: dict[str, str] = {}  # Compressed -> Original mapping
        self._reverse_map: dict[str, str] = {}  # Original -> Compressed mapping

    @abstractmethod
    def optimize_key(self, key: str) -> str:
        """Optimize a cache key."""
        pass

    @abstractmethod
    def restore_key(self, optimized_key: str) -> str:
        """Restore original key from optimized version."""
        pass

    def get_optimization_ratio(self) -> float:
        """Get overall compression ratio."""
        return self.stats.compression_ratio

    def get_stats(self) -> dict[str, Any]:
        """Get optimization statistics."""
        return {
            "strategy": self.__class__.__name__,
            "total_keys": self.stats.total_keys,
            "original_size_bytes": self.stats.original_size,
            "compressed_size_bytes": self.stats.compressed_size,
            "compression_ratio": self.stats.compression_ratio,
            "average_compression_time": self.stats.compression_time / max(1, self.stats.total_keys),
            "hash_collisions": self.stats.hash_collisions,
        }


class HashKeyOptimizer(BaseKeyOptimizer):
    """Hash-based key optimizer using cryptographic hashing."""

    def __init__(self, algorithm: str = "sha256", truncate_length: int | None = None, logger: logging.Logger | None = None):
        """Initialize hash optimizer."""
        super().__init__(logger)
        self.algorithm = algorithm.lower()
        self.truncate_length = truncate_length

        # Validate algorithm
        if self.algorithm not in ["md5", "sha1", "sha256", "sha512"]:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    def optimize_key(self, key: str) -> str:
        """Generate hash-based optimized key."""
        start_time = time.time()

        try:
            # Convert to bytes
            key_bytes = key.encode("utf-8")
            original_size = len(key_bytes)

            # Generate hash
            if self.algorithm == "md5":
                hash_obj = hashlib.md5(key_bytes)
            elif self.algorithm == "sha1":
                hash_obj = hashlib.sha1(key_bytes)
            elif self.algorithm == "sha256":
                hash_obj = hashlib.sha256(key_bytes)
            elif self.algorithm == "sha512":
                hash_obj = hashlib.sha512(key_bytes)

            hash_hex = hash_obj.hexdigest()

            # Truncate if specified
            if self.truncate_length:
                hash_hex = hash_hex[: self.truncate_length]

            compressed_size = len(hash_hex.encode("utf-8"))
            time_taken = time.time() - start_time

            # Check for collisions
            if hash_hex in self._key_map and self._key_map[hash_hex] != key:
                self.stats.hash_collisions += 1
                self.logger.warning(f"Hash collision detected for key: {key}")

            # Store mapping
            self._key_map[hash_hex] = key
            self._reverse_map[key] = hash_hex

            # Update stats
            self.stats.update(original_size, compressed_size, time_taken)

            return hash_hex

        except Exception as e:
            self.logger.error(f"Error optimizing key with hash: {e}")
            return key

    def restore_key(self, optimized_key: str) -> str:
        """Restore original key from hash (if stored)."""
        return self._key_map.get(optimized_key, optimized_key)


class CompressionKeyOptimizer(BaseKeyOptimizer):
    """Compression-based key optimizer using zlib."""

    def __init__(self, compression_level: int = 6, min_key_length: int = 50, logger: logging.Logger | None = None):
        """Initialize compression optimizer."""
        super().__init__(logger)
        self.compression_level = max(1, min(9, compression_level))
        self.min_key_length = min_key_length

    def optimize_key(self, key: str) -> str:
        """Compress key using zlib."""
        start_time = time.time()

        try:
            key_bytes = key.encode("utf-8")
            original_size = len(key_bytes)

            # Only compress if key is long enough
            if original_size < self.min_key_length:
                return key

            # Compress
            compressed_bytes = zlib.compress(key_bytes, self.compression_level)

            # Encode to string (base64)
            compressed_key = base64.b64encode(compressed_bytes).decode("ascii")

            compressed_size = len(compressed_key.encode("utf-8"))
            time_taken = time.time() - start_time

            # Only use compressed version if it's actually smaller
            if compressed_size < original_size:
                # Store mapping
                self._key_map[compressed_key] = key
                self._reverse_map[key] = compressed_key

                # Update stats
                self.stats.update(original_size, compressed_size, time_taken)
                return compressed_key
            else:
                return key

        except Exception as e:
            self.logger.error(f"Error compressing key: {e}")
            return key

    def restore_key(self, optimized_key: str) -> str:
        """Restore original key from compressed version."""
        if optimized_key in self._key_map:
            return self._key_map[optimized_key]

        # Try to decompress
        try:
            compressed_bytes = base64.b64decode(optimized_key.encode("ascii"))
            original_bytes = zlib.decompress(compressed_bytes)
            return original_bytes.decode("utf-8")
        except Exception:
            return optimized_key


class HierarchicalKeyOptimizer(BaseKeyOptimizer):
    """Hierarchical key optimizer for structured keys."""

    def __init__(
        self,
        separator: str = ":",
        max_levels: int = 5,
        level_optimizers: dict[int, BaseKeyOptimizer] | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize hierarchical optimizer."""
        super().__init__(logger)
        self.separator = separator
        self.max_levels = max_levels
        self.level_optimizers = level_optimizers or {}

        # Default optimizers for different levels
        if not self.level_optimizers:
            self.level_optimizers = {
                0: HashKeyOptimizer("md5", 8),  # Project level - short hash
                1: HashKeyOptimizer("sha256", 16),  # Service level - medium hash
                2: CompressionKeyOptimizer(9, 20),  # Content level - compression
            }

    def optimize_key(self, key: str) -> str:
        """Optimize hierarchical key by levels."""
        start_time = time.time()

        try:
            # Split key into levels
            levels = key.split(self.separator)
            original_size = len(key.encode("utf-8"))

            # Optimize each level
            optimized_levels = []
            for i, level in enumerate(levels[: self.max_levels]):
                if i in self.level_optimizers:
                    optimized_level = self.level_optimizers[i].optimize_key(level)
                else:
                    optimized_level = level
                optimized_levels.append(optimized_level)

            # Add remaining levels unchanged
            if len(levels) > self.max_levels:
                optimized_levels.extend(levels[self.max_levels :])

            optimized_key = self.separator.join(optimized_levels)
            compressed_size = len(optimized_key.encode("utf-8"))
            time_taken = time.time() - start_time

            # Store mapping
            self._key_map[optimized_key] = key
            self._reverse_map[key] = optimized_key

            # Update stats
            self.stats.update(original_size, compressed_size, time_taken)

            return optimized_key

        except Exception as e:
            self.logger.error(f"Error optimizing hierarchical key: {e}")
            return key

    def restore_key(self, optimized_key: str) -> str:
        """Restore original hierarchical key."""
        if optimized_key in self._key_map:
            return self._key_map[optimized_key]

        try:
            # Split and restore each level
            levels = optimized_key.split(self.separator)
            restored_levels = []

            for i, level in enumerate(levels[: self.max_levels]):
                if i in self.level_optimizers:
                    restored_level = self.level_optimizers[i].restore_key(level)
                else:
                    restored_level = level
                restored_levels.append(restored_level)

            # Add remaining levels unchanged
            if len(levels) > self.max_levels:
                restored_levels.extend(levels[self.max_levels :])

            return self.separator.join(restored_levels)

        except Exception as e:
            self.logger.error(f"Error restoring hierarchical key: {e}")
            return optimized_key


class AdaptiveKeyOptimizer(BaseKeyOptimizer):
    """Adaptive key optimizer that selects best strategy per key."""

    def __init__(
        self, strategies: list[BaseKeyOptimizer] | None = None, evaluation_threshold: int = 100, logger: logging.Logger | None = None
    ):
        """Initialize adaptive optimizer."""
        super().__init__(logger)

        # Default strategies
        if strategies is None:
            self.strategies = [HashKeyOptimizer("sha256", 16), CompressionKeyOptimizer(6, 30), HierarchicalKeyOptimizer()]
        else:
            self.strategies = strategies

        self.evaluation_threshold = evaluation_threshold
        self.strategy_performance: dict[str, list[float]] = {}
        self.key_count = 0

        # Initialize performance tracking
        for strategy in self.strategies:
            self.strategy_performance[strategy.__class__.__name__] = []

    def optimize_key(self, key: str) -> str:
        """Optimize key using best available strategy."""
        start_time = time.time()

        try:
            original_size = len(key.encode("utf-8"))
            best_key = key
            best_ratio = 0.0
            best_strategy = None

            # Try all strategies
            for strategy in self.strategies:
                try:
                    optimized = strategy.optimize_key(key)
                    optimized_size = len(optimized.encode("utf-8"))

                    # Calculate compression ratio
                    ratio = 1.0 - (optimized_size / original_size) if original_size > 0 else 0.0

                    if ratio > best_ratio:
                        best_key = optimized
                        best_ratio = ratio
                        best_strategy = strategy

                except Exception as e:
                    self.logger.debug(f"Strategy {strategy.__class__.__name__} failed for key: {e}")
                    continue

            # Record performance
            if best_strategy:
                strategy_name = best_strategy.__class__.__name__
                self.strategy_performance[strategy_name].append(best_ratio)

                # Keep only recent performance data
                self.strategy_performance[strategy_name] = self.strategy_performance[strategy_name][-50:]

            # Store mapping
            if best_key != key:
                self._key_map[best_key] = key
                self._reverse_map[key] = best_key

            compressed_size = len(best_key.encode("utf-8"))
            time_taken = time.time() - start_time

            # Update stats
            self.stats.update(original_size, compressed_size, time_taken)

            self.key_count += 1

            # Periodic strategy evaluation
            if self.key_count % self.evaluation_threshold == 0:
                self._evaluate_strategies()

            return best_key

        except Exception as e:
            self.logger.error(f"Error in adaptive key optimization: {e}")
            return key

    def restore_key(self, optimized_key: str) -> str:
        """Restore original key using stored mapping."""
        if optimized_key in self._key_map:
            return self._key_map[optimized_key]

        # Try to restore using each strategy
        for strategy in self.strategies:
            try:
                restored = strategy.restore_key(optimized_key)
                if restored != optimized_key:
                    return restored
            except Exception:
                continue

        return optimized_key

    def _evaluate_strategies(self) -> None:
        """Evaluate and rank strategy performance."""
        try:
            strategy_scores = {}

            for strategy_name, performance_data in self.strategy_performance.items():
                if performance_data:
                    avg_performance = sum(performance_data) / len(performance_data)
                    strategy_scores[strategy_name] = avg_performance

            # Sort strategies by performance
            sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)

            self.logger.info(f"Strategy performance ranking: {sorted_strategies}")

            # Reorder strategies based on performance
            strategy_map = {s.__class__.__name__: s for s in self.strategies}
            self.strategies = [strategy_map[name] for name, _ in sorted_strategies if name in strategy_map]

        except Exception as e:
            self.logger.error(f"Error evaluating strategies: {e}")


class ComplexKeySerializer:
    """Serializer for complex key structures."""

    def __init__(self, format_type: SerializationFormat = SerializationFormat.JSON, logger: logging.Logger | None = None):
        """Initialize serializer."""
        self.format_type = format_type
        self.logger = logger or logging.getLogger(__name__)

    def serialize_key_components(self, **components) -> str:
        """Serialize key components into optimized string."""
        try:
            if self.format_type == SerializationFormat.JSON:
                return json.dumps(components, sort_keys=True, separators=(",", ":"))
            elif self.format_type == SerializationFormat.PICKLE:
                return base64.b64encode(pickle.dumps(components)).decode("ascii")
            elif self.format_type == SerializationFormat.STRING:
                # Simple string concatenation with delimiter
                return "|".join(f"{k}={v}" for k, v in sorted(components.items()))
            else:
                return str(components)

        except Exception as e:
            self.logger.error(f"Error serializing key components: {e}")
            return str(components)

    def deserialize_key_components(self, serialized: str) -> dict[str, Any]:
        """Deserialize key components from string."""
        try:
            if self.format_type == SerializationFormat.JSON:
                return json.loads(serialized)
            elif self.format_type == SerializationFormat.PICKLE:
                return pickle.loads(base64.b64decode(serialized.encode("ascii")))
            elif self.format_type == SerializationFormat.STRING:
                components = {}
                for pair in serialized.split("|"):
                    if "=" in pair:
                        k, v = pair.split("=", 1)
                        components[k] = v
                return components
            else:
                return {"key": serialized}

        except Exception as e:
            self.logger.error(f"Error deserializing key components: {e}")
            return {"key": serialized}


class KeyOptimizerFactory:
    """Factory for creating key optimizer instances."""

    @staticmethod
    def create_optimizer(
        strategy: KeyCompressionStrategy, config: dict[str, Any] | None = None, logger: logging.Logger | None = None
    ) -> BaseKeyOptimizer:
        """Create key optimizer instance."""
        config = config or {}

        if strategy == KeyCompressionStrategy.HASH_MD5:
            return HashKeyOptimizer("md5", config.get("truncate_length"), logger)
        elif strategy == KeyCompressionStrategy.HASH_SHA256:
            return HashKeyOptimizer("sha256", config.get("truncate_length"), logger)
        elif strategy == KeyCompressionStrategy.ZLIB:
            return CompressionKeyOptimizer(config.get("compression_level", 6), config.get("min_key_length", 50), logger)
        elif strategy == KeyCompressionStrategy.CUSTOM:
            # For hierarchical keys
            return HierarchicalKeyOptimizer(
                config.get("separator", ":"), config.get("max_levels", 5), config.get("level_optimizers"), logger
            )
        else:
            # Default to adaptive
            return AdaptiveKeyOptimizer(logger=logger)

    @staticmethod
    def get_available_strategies() -> list[str]:
        """Get list of available optimization strategies."""
        return [strategy.value for strategy in KeyCompressionStrategy]


class KeyOptimizationManager:
    """Manager for cache key optimization across the system."""

    def __init__(
        self,
        default_strategy: KeyCompressionStrategy = KeyCompressionStrategy.CUSTOM,
        config: dict[str, Any] | None = None,
        logger: logging.Logger | None = None,
    ):
        """Initialize optimization manager."""
        self.logger = logger or logging.getLogger(__name__)
        self.default_optimizer = KeyOptimizerFactory.create_optimizer(default_strategy, config, self.logger)
        self.serializer = ComplexKeySerializer(SerializationFormat.JSON, self.logger)

        # Specialized optimizers for different key types
        self.specialized_optimizers: dict[str, BaseKeyOptimizer] = {
            "embedding": HashKeyOptimizer("sha256", 16, self.logger),
            "search": AdaptiveKeyOptimizer(logger=self.logger),
            "project": HashKeyOptimizer("md5", 8, self.logger),
            "file": CompressionKeyOptimizer(6, 30, self.logger),
        }

    def optimize_cache_key(self, key_type: str, **key_components) -> str:
        """Optimize cache key based on type and components."""
        try:
            # Serialize components
            serialized_key = self.serializer.serialize_key_components(**key_components)

            # Select optimizer
            optimizer = self.specialized_optimizers.get(key_type, self.default_optimizer)

            # Optimize
            return optimizer.optimize_key(serialized_key)

        except Exception as e:
            self.logger.error(f"Error optimizing cache key: {e}")
            return str(key_components)

    def restore_cache_key(self, key_type: str, optimized_key: str) -> dict[str, Any]:
        """Restore original key components."""
        try:
            # Select optimizer
            optimizer = self.specialized_optimizers.get(key_type, self.default_optimizer)

            # Restore
            serialized_key = optimizer.restore_key(optimized_key)

            # Deserialize
            return self.serializer.deserialize_key_components(serialized_key)

        except Exception as e:
            self.logger.error(f"Error restoring cache key: {e}")
            return {"key": optimized_key}

    def get_optimization_stats(self) -> dict[str, Any]:
        """Get comprehensive optimization statistics."""
        stats = {"default_optimizer": self.default_optimizer.get_stats(), "specialized_optimizers": {}}

        for key_type, optimizer in self.specialized_optimizers.items():
            stats["specialized_optimizers"][key_type] = optimizer.get_stats()

        return stats

    def configure_optimizer(self, key_type: str, strategy: KeyCompressionStrategy, config: dict[str, Any] | None = None) -> bool:
        """Configure optimizer for specific key type."""
        try:
            new_optimizer = KeyOptimizerFactory.create_optimizer(strategy, config, self.logger)

            if key_type == "default":
                self.default_optimizer = new_optimizer
            else:
                self.specialized_optimizers[key_type] = new_optimizer

            self.logger.info(f"Configured {strategy.value} optimizer for {key_type} keys")
            return True

        except Exception as e:
            self.logger.error(f"Error configuring optimizer: {e}")
            return False
