"""
Advanced cache serialization and compression optimization.

This module provides sophisticated serialization strategies, compression algorithms,
and data optimization techniques for maximum cache performance and storage efficiency.
"""

import json
import logging
import pickle
import sys
import time
import zlib
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Union

import brotli
import lz4.frame
import msgpack
import orjson


class CompressionAlgorithm(Enum):
    """Available compression algorithms."""

    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"
    BROTLI = "brotli"
    GZIP = "gzip"


class SerializationFormat(Enum):
    """Available serialization formats."""

    PICKLE = "pickle"
    JSON = "json"
    ORJSON = "orjson"
    MSGPACK = "msgpack"
    BINARY = "binary"


@dataclass
class SerializationStats:
    """Statistics for serialization performance."""

    serialization_time: float = 0.0
    deserialization_time: float = 0.0
    compression_time: float = 0.0
    decompression_time: float = 0.0
    original_size: int = 0
    serialized_size: int = 0
    compressed_size: int = 0
    compression_ratio: float = 0.0
    operations_count: int = 0

    def update_serialization(self, time_taken: float, original_size: int, serialized_size: int):
        """Update serialization statistics."""
        self.serialization_time += time_taken
        self.original_size += original_size
        self.serialized_size += serialized_size
        self.operations_count += 1

    def update_compression(self, time_taken: float, compressed_size: int):
        """Update compression statistics."""
        self.compression_time += time_taken
        self.compressed_size += compressed_size
        if self.serialized_size > 0:
            self.compression_ratio = 1.0 - (self.compressed_size / self.serialized_size)

    def get_average_times(self) -> dict[str, float]:
        """Get average operation times."""
        count = max(1, self.operations_count)
        return {
            "avg_serialization_time": self.serialization_time / count,
            "avg_deserialization_time": self.deserialization_time / count,
            "avg_compression_time": self.compression_time / count,
            "avg_decompression_time": self.decompression_time / count,
        }


class BaseSerializer(ABC):
    """Base class for cache serializers."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize serializer."""
        self.logger = logger or logging.getLogger(__name__)
        self.stats = SerializationStats()

    @abstractmethod
    def serialize(self, data: Any) -> bytes:
        """Serialize data to bytes."""
        pass

    @abstractmethod
    def deserialize(self, data: bytes) -> Any:
        """Deserialize bytes to data."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get serialization statistics."""
        stats_dict = asdict(self.stats)
        stats_dict.update(self.stats.get_average_times())
        stats_dict["format"] = self.__class__.__name__
        return stats_dict


class PickleSerializer(BaseSerializer):
    """Pickle-based serializer with protocol optimization."""

    def __init__(self, protocol: int = pickle.HIGHEST_PROTOCOL, logger: logging.Logger | None = None):
        """Initialize pickle serializer."""
        super().__init__(logger)
        self.protocol = protocol

    def serialize(self, data: Any) -> bytes:
        """Serialize using pickle."""
        start_time = time.time()
        try:
            original_size = sys.getsizeof(data)
            serialized = pickle.dumps(data, protocol=self.protocol)
            serialized_size = len(serialized)

            time_taken = time.time() - start_time
            self.stats.update_serialization(time_taken, original_size, serialized_size)

            return serialized
        except Exception as e:
            self.logger.error(f"Pickle serialization error: {e}")
            raise

    def deserialize(self, data: bytes) -> Any:
        """Deserialize using pickle."""
        start_time = time.time()
        try:
            result = pickle.loads(data)
            time_taken = time.time() - start_time
            self.stats.deserialization_time += time_taken
            return result
        except Exception as e:
            self.logger.error(f"Pickle deserialization error: {e}")
            raise


class JSONSerializer(BaseSerializer):
    """JSON-based serializer with optimization."""

    def __init__(self, ensure_ascii: bool = False, separators: tuple = (",", ":"), logger: logging.Logger | None = None):
        """Initialize JSON serializer."""
        super().__init__(logger)
        self.ensure_ascii = ensure_ascii
        self.separators = separators

    def serialize(self, data: Any) -> bytes:
        """Serialize using JSON."""
        start_time = time.time()
        try:
            original_size = sys.getsizeof(data)
            json_str = json.dumps(data, ensure_ascii=self.ensure_ascii, separators=self.separators)
            serialized = json_str.encode("utf-8")
            serialized_size = len(serialized)

            time_taken = time.time() - start_time
            self.stats.update_serialization(time_taken, original_size, serialized_size)

            return serialized
        except Exception as e:
            self.logger.error(f"JSON serialization error: {e}")
            raise

    def deserialize(self, data: bytes) -> Any:
        """Deserialize using JSON."""
        start_time = time.time()
        try:
            json_str = data.decode("utf-8")
            result = json.loads(json_str)
            time_taken = time.time() - start_time
            self.stats.deserialization_time += time_taken
            return result
        except Exception as e:
            self.logger.error(f"JSON deserialization error: {e}")
            raise


class OrjsonSerializer(BaseSerializer):
    """Orjson-based high-performance serializer."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize orjson serializer."""
        super().__init__(logger)

    def serialize(self, data: Any) -> bytes:
        """Serialize using orjson."""
        start_time = time.time()
        try:
            original_size = sys.getsizeof(data)
            serialized = orjson.dumps(data)
            serialized_size = len(serialized)

            time_taken = time.time() - start_time
            self.stats.update_serialization(time_taken, original_size, serialized_size)

            return serialized
        except Exception as e:
            self.logger.error(f"Orjson serialization error: {e}")
            raise

    def deserialize(self, data: bytes) -> Any:
        """Deserialize using orjson."""
        start_time = time.time()
        try:
            result = orjson.loads(data)
            time_taken = time.time() - start_time
            self.stats.deserialization_time += time_taken
            return result
        except Exception as e:
            self.logger.error(f"Orjson deserialization error: {e}")
            raise


class MsgpackSerializer(BaseSerializer):
    """MessagePack-based efficient serializer."""

    def __init__(self, use_bin_type: bool = True, raw: bool = False, logger: logging.Logger | None = None):
        """Initialize msgpack serializer."""
        super().__init__(logger)
        self.use_bin_type = use_bin_type
        self.raw = raw

    def serialize(self, data: Any) -> bytes:
        """Serialize using msgpack."""
        start_time = time.time()
        try:
            original_size = sys.getsizeof(data)
            serialized = msgpack.packb(data, use_bin_type=self.use_bin_type)
            serialized_size = len(serialized)

            time_taken = time.time() - start_time
            self.stats.update_serialization(time_taken, original_size, serialized_size)

            return serialized
        except Exception as e:
            self.logger.error(f"Msgpack serialization error: {e}")
            raise

    def deserialize(self, data: bytes) -> Any:
        """Deserialize using msgpack."""
        start_time = time.time()
        try:
            result = msgpack.unpackb(data, raw=self.raw)
            time_taken = time.time() - start_time
            self.stats.deserialization_time += time_taken
            return result
        except Exception as e:
            self.logger.error(f"Msgpack deserialization error: {e}")
            raise


class BaseCompressor(ABC):
    """Base class for compression algorithms."""

    def __init__(self, logger: logging.Logger | None = None):
        """Initialize compressor."""
        self.logger = logger or logging.getLogger(__name__)
        self.compression_time = 0.0
        self.decompression_time = 0.0
        self.operations_count = 0

    @abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress data."""
        pass

    @abstractmethod
    def decompress(self, data: bytes) -> bytes:
        """Decompress data."""
        pass

    def get_stats(self) -> dict[str, Any]:
        """Get compression statistics."""
        count = max(1, self.operations_count)
        return {
            "algorithm": self.__class__.__name__,
            "total_compression_time": self.compression_time,
            "total_decompression_time": self.decompression_time,
            "operations_count": self.operations_count,
            "avg_compression_time": self.compression_time / count,
            "avg_decompression_time": self.decompression_time / count,
        }


class ZlibCompressor(BaseCompressor):
    """Zlib compression algorithm."""

    def __init__(self, level: int = 6, logger: logging.Logger | None = None):
        """Initialize zlib compressor."""
        super().__init__(logger)
        self.level = max(1, min(9, level))

    def compress(self, data: bytes) -> bytes:
        """Compress using zlib."""
        start_time = time.time()
        try:
            compressed = zlib.compress(data, self.level)
            self.compression_time += time.time() - start_time
            self.operations_count += 1
            return compressed
        except Exception as e:
            self.logger.error(f"Zlib compression error: {e}")
            raise

    def decompress(self, data: bytes) -> bytes:
        """Decompress using zlib."""
        start_time = time.time()
        try:
            decompressed = zlib.decompress(data)
            self.decompression_time += time.time() - start_time
            return decompressed
        except Exception as e:
            self.logger.error(f"Zlib decompression error: {e}")
            raise


class LZ4Compressor(BaseCompressor):
    """LZ4 high-speed compression algorithm."""

    def __init__(self, compression_level: int = 0, logger: logging.Logger | None = None):
        """Initialize LZ4 compressor."""
        super().__init__(logger)
        self.compression_level = compression_level

    def compress(self, data: bytes) -> bytes:
        """Compress using LZ4."""
        start_time = time.time()
        try:
            compressed = lz4.frame.compress(data, compression_level=self.compression_level)
            self.compression_time += time.time() - start_time
            self.operations_count += 1
            return compressed
        except Exception as e:
            self.logger.error(f"LZ4 compression error: {e}")
            raise

    def decompress(self, data: bytes) -> bytes:
        """Decompress using LZ4."""
        start_time = time.time()
        try:
            decompressed = lz4.frame.decompress(data)
            self.decompression_time += time.time() - start_time
            return decompressed
        except Exception as e:
            self.logger.error(f"LZ4 decompression error: {e}")
            raise


class BrotliCompressor(BaseCompressor):
    """Brotli compression algorithm with high compression ratio."""

    def __init__(self, quality: int = 6, logger: logging.Logger | None = None):
        """Initialize Brotli compressor."""
        super().__init__(logger)
        self.quality = max(0, min(11, quality))

    def compress(self, data: bytes) -> bytes:
        """Compress using Brotli."""
        start_time = time.time()
        try:
            compressed = brotli.compress(data, quality=self.quality)
            self.compression_time += time.time() - start_time
            self.operations_count += 1
            return compressed
        except Exception as e:
            self.logger.error(f"Brotli compression error: {e}")
            raise

    def decompress(self, data: bytes) -> bytes:
        """Decompress using Brotli."""
        start_time = time.time()
        try:
            decompressed = brotli.decompress(data)
            self.decompression_time += time.time() - start_time
            return decompressed
        except Exception as e:
            self.logger.error(f"Brotli decompression error: {e}")
            raise


class OptimizedCacheSerializer:
    """Optimized cache serializer with adaptive algorithm selection."""

    def __init__(
        self,
        serialization_format: SerializationFormat = SerializationFormat.ORJSON,
        compression_algorithm: CompressionAlgorithm = CompressionAlgorithm.LZ4,
        compression_threshold: int = 1024,
        auto_optimize: bool = True,
        logger: logging.Logger | None = None,
    ):
        """Initialize optimized serializer."""
        self.logger = logger or logging.getLogger(__name__)
        self.compression_threshold = compression_threshold
        self.auto_optimize = auto_optimize

        # Initialize serializers
        self.serializers = {
            SerializationFormat.PICKLE: PickleSerializer(logger=self.logger),
            SerializationFormat.JSON: JSONSerializer(logger=self.logger),
            SerializationFormat.ORJSON: OrjsonSerializer(logger=self.logger),
            SerializationFormat.MSGPACK: MsgpackSerializer(logger=self.logger),
        }

        # Initialize compressors
        self.compressors = {
            CompressionAlgorithm.ZLIB: ZlibCompressor(6, self.logger),
            CompressionAlgorithm.LZ4: LZ4Compressor(0, self.logger),
            CompressionAlgorithm.BROTLI: BrotliCompressor(6, self.logger),
        }

        self.current_serializer = self.serializers[serialization_format]
        self.current_compressor = self.compressors.get(compression_algorithm)

        # Performance tracking for optimization
        self.performance_history: list[dict[str, Any]] = []
        self.optimization_interval = 1000
        self.operation_count = 0

    def serialize_and_compress(self, data: Any) -> bytes:
        """Serialize and optionally compress data."""
        self.operation_count += 1

        try:
            # Serialize data
            serialized = self.current_serializer.serialize(data)

            # Compress if over threshold and compressor available
            if len(serialized) >= self.compression_threshold and self.current_compressor is not None:
                start_time = time.time()
                compressed = self.current_compressor.compress(serialized)
                compression_time = time.time() - start_time

                # Only use compressed version if actually smaller
                if len(compressed) < len(serialized):
                    # Add compression marker
                    result = b"COMPRESSED:" + compressed

                    # Track performance
                    if self.auto_optimize:
                        self._track_performance(
                            {
                                "operation": "serialize_compress",
                                "original_size": len(serialized),
                                "final_size": len(result),
                                "compression_time": compression_time,
                                "compression_ratio": 1.0 - (len(compressed) / len(serialized)),
                            }
                        )

                    return result

            # Return uncompressed
            if self.auto_optimize:
                self._track_performance(
                    {
                        "operation": "serialize_only",
                        "original_size": sys.getsizeof(data),
                        "final_size": len(serialized),
                        "compression_time": 0.0,
                        "compression_ratio": 0.0,
                    }
                )

            return serialized

        except Exception as e:
            self.logger.error(f"Serialization/compression error: {e}")
            raise

    def decompress_and_deserialize(self, data: bytes) -> Any:
        """Decompress and deserialize data."""
        try:
            # Check for compression marker
            if data.startswith(b"COMPRESSED:"):
                # Remove marker and decompress
                compressed_data = data[11:]  # Remove 'COMPRESSED:' prefix

                if self.current_compressor is None:
                    raise ValueError("No compressor available for decompression")

                start_time = time.time()
                decompressed = self.current_compressor.decompress(compressed_data)
                decompression_time = time.time() - start_time

                # Deserialize
                result = self.current_serializer.deserialize(decompressed)

                if self.auto_optimize:
                    self._track_performance(
                        {
                            "operation": "decompress_deserialize",
                            "compressed_size": len(compressed_data),
                            "decompressed_size": len(decompressed),
                            "decompression_time": decompression_time,
                        }
                    )

                return result
            else:
                # Direct deserialization
                result = self.current_serializer.deserialize(data)

                if self.auto_optimize:
                    self._track_performance({"operation": "deserialize_only", "data_size": len(data), "decompression_time": 0.0})

                return result

        except Exception as e:
            self.logger.error(f"Decompression/deserialization error: {e}")
            raise

    def _track_performance(self, metrics: dict[str, Any]) -> None:
        """Track performance metrics for optimization."""
        metrics["timestamp"] = time.time()
        self.performance_history.append(metrics)

        # Keep only recent history
        if len(self.performance_history) > self.optimization_interval:
            self.performance_history = self.performance_history[-self.optimization_interval :]

        # Periodic optimization
        if self.operation_count % self.optimization_interval == 0:
            self._optimize_algorithms()

    def _optimize_algorithms(self) -> None:
        """Optimize serialization and compression algorithms based on performance."""
        try:
            if len(self.performance_history) < 100:
                return

            # Analyze performance trends
            recent_metrics = self.performance_history[-100:]

            # Calculate average performance metrics
            avg_compression_ratio = sum(m.get("compression_ratio", 0) for m in recent_metrics) / len(recent_metrics)
            avg_compression_time = sum(m.get("compression_time", 0) for m in recent_metrics) / len(recent_metrics)

            # Optimization decisions
            if avg_compression_ratio < 0.1 and avg_compression_time > 0.01:
                # Low compression benefit, high time cost - consider disabling compression
                self.logger.info("Low compression efficiency detected, consider disabling compression")

            elif avg_compression_ratio > 0.3 and avg_compression_time < 0.005:
                # Good compression benefit, low time cost - consider more aggressive compression
                self.logger.info("High compression efficiency detected, consider upgrading compression")

            # Log current performance
            self.logger.info(
                f"Serialization performance - Compression ratio: {avg_compression_ratio:.3f}, "
                f"Compression time: {avg_compression_time:.6f}s"
            )

        except Exception as e:
            self.logger.error(f"Error optimizing algorithms: {e}")

    def benchmark_algorithms(self, test_data: list[Any]) -> dict[str, Any]:
        """Benchmark different serialization and compression algorithms."""
        results = {"serializers": {}, "compressors": {}, "recommendations": []}

        try:
            # Benchmark serializers
            for format_type, serializer in self.serializers.items():
                serializer_results = []

                for data in test_data[:10]:  # Limit test data
                    try:
                        start_time = time.time()
                        serialized = serializer.serialize(data)
                        serialize_time = time.time() - start_time

                        start_time = time.time()
                        deserialized = serializer.deserialize(serialized)
                        deserialize_time = time.time() - start_time

                        serializer_results.append(
                            {
                                "serialize_time": serialize_time,
                                "deserialize_time": deserialize_time,
                                "serialized_size": len(serialized),
                                "original_size": sys.getsizeof(data),
                            }
                        )

                    except Exception as e:
                        self.logger.warning(f"Serializer {format_type.value} failed: {e}")
                        continue

                if serializer_results:
                    avg_serialize_time = sum(r["serialize_time"] for r in serializer_results) / len(serializer_results)
                    avg_deserialize_time = sum(r["deserialize_time"] for r in serializer_results) / len(serializer_results)
                    avg_size = sum(r["serialized_size"] for r in serializer_results) / len(serializer_results)

                    results["serializers"][format_type.value] = {
                        "avg_serialize_time": avg_serialize_time,
                        "avg_deserialize_time": avg_deserialize_time,
                        "avg_serialized_size": avg_size,
                        "total_time": avg_serialize_time + avg_deserialize_time,
                    }

            # Benchmark compressors
            test_data_serialized = [self.current_serializer.serialize(data) for data in test_data[:5]]

            for algorithm, compressor in self.compressors.items():
                compressor_results = []

                for serialized_data in test_data_serialized:
                    try:
                        start_time = time.time()
                        compressed = compressor.compress(serialized_data)
                        compress_time = time.time() - start_time

                        start_time = time.time()
                        decompressed = compressor.decompress(compressed)
                        decompress_time = time.time() - start_time

                        compressor_results.append(
                            {
                                "compress_time": compress_time,
                                "decompress_time": decompress_time,
                                "compressed_size": len(compressed),
                                "original_size": len(serialized_data),
                                "compression_ratio": 1.0 - (len(compressed) / len(serialized_data)),
                            }
                        )

                    except Exception as e:
                        self.logger.warning(f"Compressor {algorithm.value} failed: {e}")
                        continue

                if compressor_results:
                    avg_compress_time = sum(r["compress_time"] for r in compressor_results) / len(compressor_results)
                    avg_decompress_time = sum(r["decompress_time"] for r in compressor_results) / len(compressor_results)
                    avg_compression_ratio = sum(r["compression_ratio"] for r in compressor_results) / len(compressor_results)

                    results["compressors"][algorithm.value] = {
                        "avg_compress_time": avg_compress_time,
                        "avg_decompress_time": avg_decompress_time,
                        "avg_compression_ratio": avg_compression_ratio,
                        "total_time": avg_compress_time + avg_decompress_time,
                    }

            # Generate recommendations
            if results["serializers"]:
                fastest_serializer = min(results["serializers"].items(), key=lambda x: x[1]["total_time"])
                results["recommendations"].append(f"Fastest serializer: {fastest_serializer[0]}")

                smallest_serializer = min(results["serializers"].items(), key=lambda x: x[1]["avg_serialized_size"])
                results["recommendations"].append(f"Most compact serializer: {smallest_serializer[0]}")

            if results["compressors"]:
                best_compressor = max(results["compressors"].items(), key=lambda x: x[1]["avg_compression_ratio"])
                results["recommendations"].append(f"Best compression ratio: {best_compressor[0]}")

                fastest_compressor = min(results["compressors"].items(), key=lambda x: x[1]["total_time"])
                results["recommendations"].append(f"Fastest compressor: {fastest_compressor[0]}")

            return results

        except Exception as e:
            self.logger.error(f"Error benchmarking algorithms: {e}")
            return results

    def get_stats(self) -> dict[str, Any]:
        """Get comprehensive serialization statistics."""
        stats = {
            "operation_count": self.operation_count,
            "compression_threshold": self.compression_threshold,
            "auto_optimize": self.auto_optimize,
            "current_serializer": self.current_serializer.__class__.__name__,
            "current_compressor": self.current_compressor.__class__.__name__ if self.current_compressor else None,
            "serializer_stats": {},
            "compressor_stats": {},
        }

        # Get serializer stats
        for format_type, serializer in self.serializers.items():
            stats["serializer_stats"][format_type.value] = serializer.get_stats()

        # Get compressor stats
        for algorithm, compressor in self.compressors.items():
            stats["compressor_stats"][algorithm.value] = compressor.get_stats()

        # Performance history summary
        if self.performance_history:
            recent_metrics = self.performance_history[-100:]
            stats["recent_performance"] = {
                "avg_compression_ratio": sum(m.get("compression_ratio", 0) for m in recent_metrics) / len(recent_metrics),
                "avg_compression_time": sum(m.get("compression_time", 0) for m in recent_metrics) / len(recent_metrics),
                "operations_with_compression": sum(1 for m in recent_metrics if m.get("compression_ratio", 0) > 0),
                "total_operations": len(recent_metrics),
            }

        return stats

    def configure(
        self,
        serialization_format: SerializationFormat | None = None,
        compression_algorithm: CompressionAlgorithm | None = None,
        compression_threshold: int | None = None,
        auto_optimize: bool | None = None,
    ) -> dict[str, Any]:
        """Configure serialization parameters."""
        config_result = {"success": True, "changes": []}

        try:
            if serialization_format and serialization_format in self.serializers:
                self.current_serializer = self.serializers[serialization_format]
                config_result["changes"].append(f"Changed serializer to {serialization_format.value}")

            if compression_algorithm:
                if compression_algorithm == CompressionAlgorithm.NONE:
                    self.current_compressor = None
                    config_result["changes"].append("Disabled compression")
                elif compression_algorithm in self.compressors:
                    self.current_compressor = self.compressors[compression_algorithm]
                    config_result["changes"].append(f"Changed compressor to {compression_algorithm.value}")

            if compression_threshold is not None:
                self.compression_threshold = max(0, compression_threshold)
                config_result["changes"].append(f"Changed compression threshold to {self.compression_threshold}")

            if auto_optimize is not None:
                self.auto_optimize = auto_optimize
                config_result["changes"].append(f"{'Enabled' if auto_optimize else 'Disabled'} auto-optimization")

            return config_result

        except Exception as e:
            self.logger.error(f"Error configuring serialization: {e}")
            return {"success": False, "error": str(e)}
