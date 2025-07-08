"""
Cache utility functions for the Codebase RAG MCP Server.

This module provides comprehensive cache utilities including serialization,
compression, size estimation, and debugging tools.
"""

import gzip
import json
import logging
import pickle
import sys
import time
import zlib
from dataclasses import asdict, is_dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import lz4.frame

    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False


class SerializationFormat(Enum):
    """Supported serialization formats."""

    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"


class CompressionFormat(Enum):
    """Supported compression formats."""

    NONE = "none"
    GZIP = "gzip"
    ZLIB = "zlib"
    LZ4 = "lz4"


class CacheUtilsError(Exception):
    """Base exception for cache utils errors."""

    pass


class SerializationError(CacheUtilsError):
    """Exception raised during serialization/deserialization."""

    pass


class CompressionError(CacheUtilsError):
    """Exception raised during compression/decompression."""

    pass


def serialize_data(data: Any, format: SerializationFormat = SerializationFormat.PICKLE, ensure_ascii: bool = False) -> bytes:
    """
    Serialize data to bytes using the specified format.

    Args:
        data: Data to serialize
        format: Serialization format to use
        ensure_ascii: For JSON, ensure ASCII output

    Returns:
        bytes: Serialized data

    Raises:
        SerializationError: If serialization fails
    """
    try:
        if format == SerializationFormat.JSON:
            # Convert dataclasses and enums for JSON compatibility
            json_data = _make_json_serializable(data)
            json_str = json.dumps(json_data, ensure_ascii=ensure_ascii, separators=(",", ":"))
            return json_str.encode("utf-8")

        elif format == SerializationFormat.PICKLE:
            return pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)

        elif format == SerializationFormat.MSGPACK:
            try:
                import msgpack

                return msgpack.packb(data, use_bin_type=True)
            except ImportError:
                raise SerializationError("msgpack library not installed")

        else:
            raise SerializationError(f"Unsupported serialization format: {format}")

    except Exception as e:
        raise SerializationError(f"Failed to serialize data: {e}")


def deserialize_data(data: bytes, format: SerializationFormat = SerializationFormat.PICKLE) -> Any:
    """
    Deserialize bytes to data using the specified format.

    Args:
        data: Serialized data bytes
        format: Serialization format used

    Returns:
        Any: Deserialized data

    Raises:
        SerializationError: If deserialization fails
    """
    try:
        if format == SerializationFormat.JSON:
            json_str = data.decode("utf-8")
            return json.loads(json_str)

        elif format == SerializationFormat.PICKLE:
            return pickle.loads(data)

        elif format == SerializationFormat.MSGPACK:
            try:
                import msgpack

                return msgpack.unpackb(data, raw=False)
            except ImportError:
                raise SerializationError("msgpack library not installed")

        else:
            raise SerializationError(f"Unsupported serialization format: {format}")

    except Exception as e:
        raise SerializationError(f"Failed to deserialize data: {e}")


def compress_data(data: bytes, format: CompressionFormat = CompressionFormat.GZIP, level: int = 6) -> bytes:
    """
    Compress data using the specified format.

    Args:
        data: Data to compress
        format: Compression format to use
        level: Compression level (1-9 for gzip/zlib, 0-16 for lz4)

    Returns:
        bytes: Compressed data

    Raises:
        CompressionError: If compression fails
    """
    try:
        if format == CompressionFormat.NONE:
            return data

        elif format == CompressionFormat.GZIP:
            return gzip.compress(data, compresslevel=level)

        elif format == CompressionFormat.ZLIB:
            return zlib.compress(data, level=level)

        elif format == CompressionFormat.LZ4:
            if not HAS_LZ4:
                raise CompressionError("lz4 library not installed")
            return lz4.frame.compress(data, compression_level=level)

        else:
            raise CompressionError(f"Unsupported compression format: {format}")

    except Exception as e:
        raise CompressionError(f"Failed to compress data: {e}")


def decompress_data(data: bytes, format: CompressionFormat = CompressionFormat.GZIP) -> bytes:
    """
    Decompress data using the specified format.

    Args:
        data: Compressed data
        format: Compression format used

    Returns:
        bytes: Decompressed data

    Raises:
        CompressionError: If decompression fails
    """
    try:
        if format == CompressionFormat.NONE:
            return data

        elif format == CompressionFormat.GZIP:
            return gzip.decompress(data)

        elif format == CompressionFormat.ZLIB:
            return zlib.decompress(data)

        elif format == CompressionFormat.LZ4:
            if not HAS_LZ4:
                raise CompressionError("lz4 library not installed")
            return lz4.frame.decompress(data)

        else:
            raise CompressionError(f"Unsupported compression format: {format}")

    except Exception as e:
        raise CompressionError(f"Failed to decompress data: {e}")


def serialize_and_compress(
    data: Any,
    serialization_format: SerializationFormat = SerializationFormat.PICKLE,
    compression_format: CompressionFormat = CompressionFormat.GZIP,
    compression_level: int = 6,
) -> tuple[bytes, dict[str, Any]]:
    """
    Serialize and compress data with metadata.

    Args:
        data: Data to process
        serialization_format: Serialization format
        compression_format: Compression format
        compression_level: Compression level

    Returns:
        Tuple[bytes, Dict[str, Any]]: (processed_data, metadata)

    Raises:
        CacheUtilsError: If processing fails
    """
    try:
        # Start timing
        start_time = time.time()

        # Serialize data
        serialized_data = serialize_data(data, serialization_format)
        serialized_size = len(serialized_data)

        # Compress data
        compressed_data = compress_data(serialized_data, compression_format, compression_level)
        compressed_size = len(compressed_data)

        # Calculate processing time
        processing_time = time.time() - start_time

        # Build metadata
        metadata = {
            "serialization_format": serialization_format.value,
            "compression_format": compression_format.value,
            "compression_level": compression_level,
            "original_size": estimate_size(data),
            "serialized_size": serialized_size,
            "compressed_size": compressed_size,
            "compression_ratio": serialized_size / compressed_size if compressed_size > 0 else 1.0,
            "processing_time": processing_time,
            "timestamp": time.time(),
        }

        return compressed_data, metadata

    except Exception as e:
        raise CacheUtilsError(f"Failed to serialize and compress data: {e}")


def decompress_and_deserialize(data: bytes, metadata: dict[str, Any]) -> Any:
    """
    Decompress and deserialize data using metadata.

    Args:
        data: Compressed and serialized data
        metadata: Processing metadata

    Returns:
        Any: Original data

    Raises:
        CacheUtilsError: If processing fails
    """
    try:
        # Extract formats from metadata
        serialization_format = SerializationFormat(metadata["serialization_format"])
        compression_format = CompressionFormat(metadata["compression_format"])

        # Decompress data
        decompressed_data = decompress_data(data, compression_format)

        # Deserialize data
        original_data = deserialize_data(decompressed_data, serialization_format)

        return original_data

    except Exception as e:
        raise CacheUtilsError(f"Failed to decompress and deserialize data: {e}")


def estimate_size(obj: Any) -> int:
    """
    Estimate the memory size of an object in bytes.

    Args:
        obj: Object to estimate size for

    Returns:
        int: Estimated size in bytes
    """
    try:
        # Use sys.getsizeof for basic types
        base_size = sys.getsizeof(obj)

        # For collections, estimate recursively
        if isinstance(obj, dict):
            # Add size of keys and values
            items_size = sum(estimate_size(k) + estimate_size(v) for k, v in obj.items())
            return base_size + items_size

        elif isinstance(obj, (list, tuple, set)):
            # Add size of elements
            items_size = sum(estimate_size(item) for item in obj)
            return base_size + items_size

        elif isinstance(obj, str):
            # String size is usually accurate with getsizeof
            return base_size

        elif hasattr(obj, "__dict__"):
            # For objects with attributes
            attrs_size = estimate_size(obj.__dict__)
            return base_size + attrs_size

        else:
            # For other types, use base size
            return base_size

    except Exception:
        # Fallback for objects that can't be sized
        return 1024  # Default estimate


def analyze_compression_efficiency(data: Any, formats: list[CompressionFormat] | None = None) -> dict[str, dict[str, Any]]:
    """
    Analyze compression efficiency for different formats.

    Args:
        data: Data to analyze
        formats: List of compression formats to test

    Returns:
        Dict[str, Dict[str, Any]]: Analysis results per format
    """
    if formats is None:
        formats = [CompressionFormat.NONE, CompressionFormat.GZIP, CompressionFormat.ZLIB]
        if HAS_LZ4:
            formats.append(CompressionFormat.LZ4)

    # Serialize data once
    serialized_data = serialize_data(data, SerializationFormat.PICKLE)
    original_size = len(serialized_data)

    results = {}

    for compression_format in formats:
        try:
            # Test compression
            start_time = time.time()
            compressed_data = compress_data(serialized_data, compression_format)
            compression_time = time.time() - start_time

            # Test decompression
            start_time = time.time()
            decompressed_data = decompress_data(compressed_data, compression_format)
            decompression_time = time.time() - start_time

            # Verify integrity
            integrity_ok = decompressed_data == serialized_data

            compressed_size = len(compressed_data)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0

            results[compression_format.value] = {
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "space_savings": 1.0 - (compressed_size / original_size),
                "compression_time": compression_time,
                "decompression_time": decompression_time,
                "total_time": compression_time + decompression_time,
                "integrity_ok": integrity_ok,
            }

        except Exception as e:
            results[compression_format.value] = {
                "error": str(e),
                "original_size": original_size,
                "compressed_size": None,
                "compression_ratio": None,
                "space_savings": None,
                "compression_time": None,
                "decompression_time": None,
                "total_time": None,
                "integrity_ok": False,
            }

    return results


def generate_cache_key(*args, **kwargs) -> str:
    """
    Generate a deterministic cache key from arguments.

    Args:
        *args: Positional arguments
        **kwargs: Keyword arguments

    Returns:
        str: Generated cache key
    """
    import hashlib

    # Create a hashable representation
    key_data = {
        "args": args,
        "kwargs": sorted(kwargs.items()),
    }

    # Serialize and hash
    serialized = serialize_data(key_data, SerializationFormat.JSON, ensure_ascii=True)
    hash_obj = hashlib.sha256(serialized)
    return hash_obj.hexdigest()


def create_cache_key_with_prefix(prefix: str, *args, **kwargs) -> str:
    """
    Create a cache key with a specific prefix.

    Args:
        prefix: Key prefix
        *args: Arguments for key generation
        **kwargs: Keyword arguments for key generation

    Returns:
        str: Prefixed cache key
    """
    base_key = generate_cache_key(*args, **kwargs)
    return f"{prefix}:{base_key}"


def debug_cache_entry(key: str, value: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Create debug information for a cache entry.

    Args:
        key: Cache key
        value: Cache value
        metadata: Optional metadata

    Returns:
        Dict[str, Any]: Debug information
    """
    debug_info = {
        "key": key,
        "key_length": len(key),
        "value_type": type(value).__name__,
        "value_size": estimate_size(value),
        "timestamp": time.time(),
    }

    # Add value summary based on type
    if isinstance(value, str):
        debug_info["value_preview"] = value[:100] + "..." if len(value) > 100 else value
    elif isinstance(value, (list, tuple)):
        debug_info["value_length"] = len(value)
        debug_info["value_preview"] = f"[{type(value).__name__} with {len(value)} items]"
    elif isinstance(value, dict):
        debug_info["value_keys"] = len(value)
        debug_info["value_preview"] = f"{{dict with {len(value)} keys}}"
    else:
        debug_info["value_preview"] = str(value)[:100]

    # Add metadata if provided
    if metadata:
        debug_info["metadata"] = metadata

    return debug_info


def validate_cache_key(key: str, max_length: int = 250) -> bool:
    """
    Validate a cache key for compliance with Redis and other cache systems.

    Args:
        key: Cache key to validate
        max_length: Maximum key length

    Returns:
        bool: True if key is valid
    """
    if not isinstance(key, str):
        return False

    if not key:
        return False

    if len(key) > max_length:
        return False

    # Check for problematic characters
    problematic_chars = [" ", "\n", "\r", "\t", "\0"]
    for char in problematic_chars:
        if char in key:
            return False

    return True


def get_optimal_compression_format(data: Any, priority: str = "ratio") -> CompressionFormat:  # "ratio", "speed", "balanced"
    """
    Get the optimal compression format for given data and priority.

    Args:
        data: Data to analyze
        priority: Optimization priority

    Returns:
        CompressionFormat: Recommended compression format
    """
    # Quick analysis for small data
    if estimate_size(data) < 1024:  # Less than 1KB
        return CompressionFormat.NONE

    # Analyze compression efficiency
    analysis = analyze_compression_efficiency(data)

    if priority == "speed":
        # Prioritize speed
        best_format = CompressionFormat.NONE
        best_time = float("inf")

        for format_name, result in analysis.items():
            if result.get("total_time") is not None:
                if result["total_time"] < best_time:
                    best_time = result["total_time"]
                    best_format = CompressionFormat(format_name)

    elif priority == "ratio":
        # Prioritize compression ratio
        best_format = CompressionFormat.NONE
        best_ratio = 1.0

        for format_name, result in analysis.items():
            if result.get("compression_ratio") is not None:
                if result["compression_ratio"] > best_ratio:
                    best_ratio = result["compression_ratio"]
                    best_format = CompressionFormat(format_name)

    else:  # balanced
        # Balance ratio and speed
        best_format = CompressionFormat.GZIP
        best_score = 0.0

        for format_name, result in analysis.items():
            if result.get("compression_ratio") is not None and result.get("total_time") is not None:
                # Score based on ratio/time balance
                ratio_score = result["compression_ratio"] / 10.0  # Normalize
                time_score = 1.0 / (1.0 + result["total_time"])  # Invert time
                score = (ratio_score + time_score) / 2.0

                if score > best_score:
                    best_score = score
                    best_format = CompressionFormat(format_name)

    return best_format


def _make_json_serializable(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable format.

    Args:
        obj: Object to convert

    Returns:
        Any: JSON-serializable object
    """
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, Enum):
        return obj.value
    elif is_dataclass(obj):
        return _make_json_serializable(asdict(obj))
    elif hasattr(obj, "isoformat"):  # datetime objects
        return obj.isoformat()
    else:
        # Fallback to string representation
        return str(obj)


# Convenience functions for common cache operations
def cache_serialize(data: Any, compress: bool = True) -> tuple[bytes, dict[str, Any]]:
    """
    Convenience function to serialize data for caching.

    Args:
        data: Data to serialize
        compress: Whether to compress the data

    Returns:
        Tuple[bytes, Dict[str, Any]]: (serialized_data, metadata)
    """
    compression_format = CompressionFormat.GZIP if compress else CompressionFormat.NONE
    return serialize_and_compress(data, compression_format=compression_format)


def cache_deserialize(data: bytes, metadata: dict[str, Any]) -> Any:
    """
    Convenience function to deserialize cached data.

    Args:
        data: Serialized data
        metadata: Serialization metadata

    Returns:
        Any: Deserialized data
    """
    return decompress_and_deserialize(data, metadata)


# Export commonly used functions
__all__ = [
    "SerializationFormat",
    "CompressionFormat",
    "CacheUtilsError",
    "SerializationError",
    "CompressionError",
    "serialize_data",
    "deserialize_data",
    "compress_data",
    "decompress_data",
    "serialize_and_compress",
    "decompress_and_deserialize",
    "estimate_size",
    "analyze_compression_efficiency",
    "generate_cache_key",
    "create_cache_key_with_prefix",
    "debug_cache_entry",
    "validate_cache_key",
    "get_optimal_compression_format",
    "cache_serialize",
    "cache_deserialize",
]
