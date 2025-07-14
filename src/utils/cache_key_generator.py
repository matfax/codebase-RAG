"""
Cache key generation and management system for the Codebase RAG MCP Server.

This module provides comprehensive cache key generation with hierarchical structure,
content-based hashing, versioning, and collision detection/resolution.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union
from uuid import uuid4

from config.cache_config import CacheConfig, get_global_cache_config


class KeyType(Enum):
    """Cache key types for different data categories."""

    EMBEDDING = "embedding"
    SEARCH = "search"
    PROJECT = "project"
    FILE = "file"
    METADATA = "metadata"
    ANALYTICS = "analytics"
    HEALTH = "health"


class HashAlgorithm(Enum):
    """Supported hash algorithms for key generation."""

    SHA256 = "sha256"
    SHA1 = "sha1"
    MD5 = "md5"
    BLAKE2B = "blake2b"


@dataclass
class CacheKeyComponents:
    """Components that make up a cache key."""

    key_type: KeyType
    namespace: str
    project_id: str
    content_hash: str
    version: str = "v1"
    timestamp: float | None = None
    additional_params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Set timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class CacheKeyMetadata:
    """Metadata associated with a cache key."""

    key: str
    components: CacheKeyComponents
    created_at: float
    last_accessed: float
    access_count: int = 0
    collision_count: int = 0
    is_collision_resolved: bool = False

    def __post_init__(self):
        """Initialize timestamps if not provided."""
        if self.created_at == 0:
            self.created_at = time.time()
        if self.last_accessed == 0:
            self.last_accessed = time.time()


class CacheKeyGeneratorError(Exception):
    """Base exception for cache key generator errors."""

    pass


class KeyCollisionError(CacheKeyGeneratorError):
    """Exception raised when key collision cannot be resolved."""

    pass


class InvalidKeyError(CacheKeyGeneratorError):
    """Exception raised when key validation fails."""

    pass


class CacheKeyGenerator:
    """
    Comprehensive cache key generation system with hierarchical structure,
    content-based hashing, versioning, and collision detection.
    """

    def __init__(self, config: CacheConfig | None = None):
        """
        Initialize the cache key generator.

        Args:
            config: Cache configuration instance
        """
        self.config = config or get_global_cache_config()
        self.logger = logging.getLogger(__name__)

        # Key collision tracking
        self._key_registry: dict[str, CacheKeyMetadata] = {}
        self._collision_counter: dict[str, int] = {}

        # Hash algorithm configuration
        self.hash_algorithm = HashAlgorithm.SHA256

        # Key versioning
        self.current_version = "v1"
        self.version_history: list[str] = ["v1"]

        # Maximum key length (Redis limit is 512MB, but we keep it reasonable)
        self.max_key_length = self.config.max_key_length or 250

        # Namespace separator
        self.namespace_separator = ":"

        # Timestamp configuration
        self.include_timestamps = getattr(self.config, "include_timestamps", False)

        # Initialize hash functions
        self._hash_functions = {
            HashAlgorithm.SHA256: hashlib.sha256,
            HashAlgorithm.SHA1: hashlib.sha1,
            HashAlgorithm.MD5: hashlib.md5,
            HashAlgorithm.BLAKE2B: hashlib.blake2b,
        }

    def generate_key(
        self,
        key_type: KeyType,
        namespace: str,
        project_id: str,
        content: str | bytes | dict[str, Any],
        version: str | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a cache key with hierarchical structure and content-based hashing.

        Args:
            key_type: Type of cache key
            namespace: Namespace for the key
            project_id: Project identifier
            content: Content to hash for the key
            version: Key version (defaults to current version)
            additional_params: Additional parameters for key generation

        Returns:
            str: Generated cache key

        Raises:
            CacheKeyGeneratorError: If key generation fails
        """
        try:
            # Validate inputs
            self._validate_inputs(key_type, namespace, project_id, content)

            # Generate content hash
            content_hash = self._generate_content_hash(content)

            # Create key components
            components = CacheKeyComponents(
                key_type=key_type,
                namespace=namespace,
                project_id=project_id,
                content_hash=content_hash,
                version=version or self.current_version,
                additional_params=additional_params or {},
            )

            # Build hierarchical key
            key = self._build_hierarchical_key(components)

            # Check for collisions and resolve if necessary
            key = self._handle_key_collision(key, components)

            # Register the key
            self._register_key(key, components)

            return key

        except Exception as e:
            self.logger.error(f"Failed to generate cache key: {e}")
            raise CacheKeyGeneratorError(f"Key generation failed: {e}")

    def generate_hierarchical_key(
        self,
        key_type: KeyType,
        namespace: str,
        project_id: str,
        content_hash: str,
        version: str | None = None,
        additional_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a hierarchical cache key with pre-computed content hash.

        This method is similar to generate_key but uses a pre-computed content hash
        instead of computing it from content, useful for embedding and other cases
        where the hash is already known.

        Args:
            key_type: Type of cache key
            namespace: Namespace for the key
            project_id: Project identifier
            content_hash: Pre-computed content hash
            version: Key version (defaults to current version)
            additional_params: Additional parameters for key generation

        Returns:
            str: Generated cache key
        """
        try:
            version = version or self.current_version
            additional_params = additional_params or {}

            # Create key components
            components = CacheKeyComponents(
                key_type=key_type,
                namespace=namespace,
                project_id=project_id,
                content_hash=content_hash,
                version=version,
                timestamp=time.time() if self.include_timestamps else None,
                additional_params=additional_params,
            )

            # Build hierarchical key
            key = self._build_hierarchical_key(components)

            # Check for collisions and resolve if necessary
            key = self._handle_key_collision(key, components)

            # Register the key
            self._register_key(key, components)

            return key

        except Exception as e:
            self.logger.error(f"Failed to generate hierarchical cache key: {e}")
            raise CacheKeyGeneratorError(f"Hierarchical key generation failed: {e}")

    def generate_embedding_key(
        self,
        query: str,
        model_name: str,
        project_id: str,
        additional_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a cache key for embedding data.

        Args:
            query: Query text for embedding
            model_name: Name of the embedding model
            project_id: Project identifier
            additional_params: Additional parameters

        Returns:
            str: Generated embedding cache key
        """
        content = {
            "query": query,
            "model": model_name,
            "params": additional_params or {},
        }

        return self.generate_key(
            key_type=KeyType.EMBEDDING,
            namespace="embeddings",
            project_id=project_id,
            content=content,
            additional_params=additional_params,
        )

    def generate_search_key(
        self,
        search_query: str,
        search_params: dict[str, Any],
        project_id: str,
        additional_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a cache key for search results.

        Args:
            search_query: Search query text
            search_params: Search parameters
            project_id: Project identifier
            additional_params: Additional parameters

        Returns:
            str: Generated search cache key
        """
        content = {
            "query": search_query,
            "params": search_params,
            "additional": additional_params or {},
        }

        return self.generate_key(
            key_type=KeyType.SEARCH,
            namespace="search",
            project_id=project_id,
            content=content,
            additional_params=additional_params,
        )

    def generate_project_key(
        self,
        project_path: str,
        project_metadata: dict[str, Any],
        additional_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a cache key for project data.

        Args:
            project_path: Path to the project
            project_metadata: Project metadata
            additional_params: Additional parameters

        Returns:
            str: Generated project cache key
        """
        # Extract project ID from path
        project_id = self._extract_project_id(project_path)

        content = {
            "path": project_path,
            "metadata": project_metadata,
            "additional": additional_params or {},
        }

        return self.generate_key(
            key_type=KeyType.PROJECT,
            namespace="projects",
            project_id=project_id,
            content=content,
            additional_params=additional_params,
        )

    def generate_file_key(
        self,
        file_path: str,
        file_metadata: dict[str, Any],
        project_id: str,
        additional_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a cache key for file processing data.

        Args:
            file_path: Path to the file
            file_metadata: File metadata
            project_id: Project identifier
            additional_params: Additional parameters

        Returns:
            str: Generated file cache key
        """
        content = {
            "path": file_path,
            "metadata": file_metadata,
            "additional": additional_params or {},
        }

        return self.generate_key(
            key_type=KeyType.FILE,
            namespace="files",
            project_id=project_id,
            content=content,
            additional_params=additional_params,
        )

    def generate_file_parsing_key(
        self,
        file_path: str,
        content_hash: str,
        language: str,
        parser_version: str = "1.0.0",
        additional_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a cache key for file parsing results.

        Args:
            file_path: Path to the file
            content_hash: SHA256 hash of file content
            language: Detected programming language
            parser_version: Version of the parser used
            additional_params: Additional parameters

        Returns:
            str: Generated file parsing cache key
        """
        # Extract project ID from file path
        import os

        project_id = self._extract_project_id(os.path.dirname(file_path))

        content = {
            "file_path": file_path,
            "content_hash": content_hash,
            "language": language,
            "parser_version": parser_version,
            "additional": additional_params or {},
        }

        return self.generate_key(
            key_type=KeyType.FILE,
            namespace="file_parsing",
            project_id=project_id,
            content=content,
            additional_params=additional_params,
        )

    def generate_chunking_key(
        self,
        file_path: str,
        content_hash: str,
        language: str,
        chunking_strategy: str = "default",
        additional_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a cache key for chunking results.

        Args:
            file_path: Path to the file
            content_hash: SHA256 hash of file content
            language: Detected programming language
            chunking_strategy: Strategy used for chunking
            additional_params: Additional parameters

        Returns:
            str: Generated chunking cache key
        """
        # Extract project ID from file path
        import os

        project_id = self._extract_project_id(os.path.dirname(file_path))

        content = {
            "file_path": file_path,
            "content_hash": content_hash,
            "language": language,
            "chunking_strategy": chunking_strategy,
            "additional": additional_params or {},
        }

        return self.generate_key(
            key_type=KeyType.FILE,
            namespace="chunking",
            project_id=project_id,
            content=content,
            additional_params=additional_params,
        )

    def generate_ast_parsing_key(
        self,
        file_path: str,
        content_hash: str,
        language: str,
        tree_sitter_version: str = "0.20.0",
        additional_params: dict[str, Any] | None = None,
    ) -> str:
        """
        Generate a cache key for AST parsing results.

        Args:
            file_path: Path to the file
            content_hash: SHA256 hash of file content
            language: Detected programming language
            tree_sitter_version: Version of Tree-sitter used
            additional_params: Additional parameters

        Returns:
            str: Generated AST parsing cache key
        """
        # Extract project ID from file path
        import os

        project_id = self._extract_project_id(os.path.dirname(file_path))

        content = {
            "file_path": file_path,
            "content_hash": content_hash,
            "language": language,
            "tree_sitter_version": tree_sitter_version,
            "additional": additional_params or {},
        }

        return self.generate_key(
            key_type=KeyType.FILE,
            namespace="ast_parsing",
            project_id=project_id,
            content=content,
            additional_params=additional_params,
        )

    def validate_key(self, key: str) -> bool:
        """
        Validate a cache key format and structure.

        Args:
            key: Cache key to validate

        Returns:
            bool: True if key is valid

        Raises:
            InvalidKeyError: If key is invalid
        """
        try:
            # Check key length
            if len(key) > self.max_key_length:
                raise InvalidKeyError(f"Key length exceeds maximum: {len(key)} > {self.max_key_length}")

            # Check key format
            if not key.startswith(self.config.key_prefix):
                raise InvalidKeyError(f"Key doesn't start with prefix: {self.config.key_prefix}")

            # Check namespace structure
            parts = key.split(self.namespace_separator)
            if len(parts) < 6:  # prefix:type:namespace:project:version:hash
                raise InvalidKeyError(f"Key doesn't have minimum required parts: {len(parts)}")

            # Validate key components
            key_type_str = parts[1]
            try:
                KeyType(key_type_str)
            except ValueError:
                raise InvalidKeyError(f"Invalid key type: {key_type_str}")

            # Check for empty components
            if any(not part.strip() for part in parts):
                raise InvalidKeyError("Key contains empty components")

            return True

        except Exception as e:
            self.logger.error(f"Key validation failed: {e}")
            raise InvalidKeyError(f"Key validation failed: {e}")

    def extract_key_components(self, key: str) -> CacheKeyComponents:
        """
        Extract components from a cache key.

        Args:
            key: Cache key to parse

        Returns:
            CacheKeyComponents: Parsed key components

        Raises:
            InvalidKeyError: If key cannot be parsed
        """
        try:
            # Validate key first
            self.validate_key(key)

            # Split key into components
            parts = key.split(self.namespace_separator)

            # Parse components
            # Format: prefix:type:namespace:project:version:hash:timestamp?:additional?
            key_type = KeyType(parts[1])
            namespace = parts[2]
            project_id = parts[3]
            version = parts[4]
            content_hash = parts[5]

            # Optional timestamp
            timestamp = None
            if len(parts) > 6 and parts[6]:
                try:
                    timestamp = float(parts[6])
                except ValueError:
                    pass

            # Optional additional parameters (encoded as key=value pairs)
            additional_params = {}
            if len(parts) > 7:
                for param_str in parts[7:]:
                    if "=" in param_str:
                        key_param, value = param_str.split("=", 1)
                        additional_params[key_param] = value

            return CacheKeyComponents(
                key_type=key_type,
                namespace=namespace,
                project_id=project_id,
                content_hash=content_hash,
                version=version,
                timestamp=timestamp,
                additional_params=additional_params,
            )

        except Exception as e:
            self.logger.error(f"Failed to extract key components: {e}")
            raise InvalidKeyError(f"Key parsing failed: {e}")

    def invalidate_keys_by_pattern(self, pattern: str) -> list[str]:
        """
        Get keys that match a pattern for invalidation.

        Args:
            pattern: Pattern to match keys against

        Returns:
            List[str]: List of keys matching the pattern
        """
        matching_keys = []

        for key in self._key_registry.keys():
            if self._matches_pattern(key, pattern):
                matching_keys.append(key)

        return matching_keys

    def invalidate_keys_by_project(self, project_id: str) -> list[str]:
        """
        Get keys for a specific project for invalidation.

        Args:
            project_id: Project identifier

        Returns:
            List[str]: List of keys for the project
        """
        matching_keys = []

        for key, metadata in self._key_registry.items():
            if metadata.components.project_id == project_id:
                matching_keys.append(key)

        return matching_keys

    def invalidate_keys_by_version(self, version: str) -> list[str]:
        """
        Get keys with a specific version for invalidation.

        Args:
            version: Version to match

        Returns:
            List[str]: List of keys with the specified version
        """
        matching_keys = []

        for key, metadata in self._key_registry.items():
            if metadata.components.version == version:
                matching_keys.append(key)

        return matching_keys

    def update_key_version(self, new_version: str) -> None:
        """
        Update the current key version.

        Args:
            new_version: New version string
        """
        if new_version not in self.version_history:
            self.version_history.append(new_version)

        self.current_version = new_version
        self.logger.info(f"Updated key version to: {new_version}")

    def get_key_statistics(self) -> dict[str, Any]:
        """
        Get statistics about generated keys.

        Returns:
            Dict[str, Any]: Key statistics
        """
        total_keys = len(self._key_registry)
        collision_count = sum(1 for metadata in self._key_registry.values() if metadata.collision_count > 0)

        # Type distribution
        type_distribution = {}
        for metadata in self._key_registry.values():
            key_type = metadata.components.key_type.value
            type_distribution[key_type] = type_distribution.get(key_type, 0) + 1

        # Version distribution
        version_distribution = {}
        for metadata in self._key_registry.values():
            version = metadata.components.version
            version_distribution[version] = version_distribution.get(version, 0) + 1

        return {
            "total_keys": total_keys,
            "collision_count": collision_count,
            "collision_rate": collision_count / total_keys if total_keys > 0 else 0,
            "type_distribution": type_distribution,
            "version_distribution": version_distribution,
            "current_version": self.current_version,
            "version_history": self.version_history,
        }

    def _validate_inputs(
        self,
        key_type: KeyType,
        namespace: str,
        project_id: str,
        content: str | bytes | dict[str, Any],
    ) -> None:
        """Validate inputs for key generation."""
        if not isinstance(key_type, KeyType):
            raise CacheKeyGeneratorError(f"Invalid key type: {key_type}")

        if not namespace or not namespace.strip():
            raise CacheKeyGeneratorError("Namespace cannot be empty")

        if not project_id or not project_id.strip():
            raise CacheKeyGeneratorError("Project ID cannot be empty")

        if content is None:
            raise CacheKeyGeneratorError("Content cannot be None")

        # Check for invalid characters in namespace and project_id
        invalid_chars = set(self.namespace_separator + " \t\n\r")
        if any(char in namespace for char in invalid_chars):
            raise CacheKeyGeneratorError(f"Namespace contains invalid characters: {invalid_chars}")

        if any(char in project_id for char in invalid_chars):
            raise CacheKeyGeneratorError(f"Project ID contains invalid characters: {invalid_chars}")

    def _generate_content_hash(self, content: str | bytes | dict[str, Any]) -> str:
        """Generate content-based hash using SHA-256."""
        hash_func = self._hash_functions[self.hash_algorithm]

        if isinstance(content, str):
            content_bytes = content.encode("utf-8")
        elif isinstance(content, bytes):
            content_bytes = content
        elif isinstance(content, dict):
            # Convert dict to sorted JSON string for consistent hashing
            import json

            content_str = json.dumps(content, sort_keys=True, separators=(",", ":"))
            content_bytes = content_str.encode("utf-8")
        else:
            # Convert to string and then to bytes
            content_bytes = str(content).encode("utf-8")

        return hash_func(content_bytes).hexdigest()

    def _build_hierarchical_key(self, components: CacheKeyComponents) -> str:
        """Build hierarchical key from components."""
        # Format: prefix:type:namespace:project:version:hash:timestamp:additional_params
        key_parts = [
            self.config.key_prefix,
            components.key_type.value,
            components.namespace,
            components.project_id,
            components.version,
            components.content_hash,
        ]

        # Add timestamp if present
        if components.timestamp:
            key_parts.append(str(int(components.timestamp)))
        else:
            key_parts.append("")

        # Add additional parameters
        if components.additional_params:
            for key, value in sorted(components.additional_params.items()):
                key_parts.append(f"{key}={value}")

        return self.namespace_separator.join(key_parts)

    def _handle_key_collision(self, key: str, components: CacheKeyComponents) -> str:
        """Handle key collision detection and resolution."""
        original_key = key
        collision_count = 0

        while key in self._key_registry:
            collision_count += 1

            # Check if existing key has same content hash
            existing_metadata = self._key_registry[key]
            if existing_metadata.components.content_hash == components.content_hash:
                # Same content, return existing key
                existing_metadata.access_count += 1
                existing_metadata.last_accessed = time.time()
                return key

            # Generate new key with collision suffix
            collision_suffix = f"_c{collision_count}"
            key = f"{original_key}{collision_suffix}"

            # Prevent infinite loops
            if collision_count > 1000:
                raise KeyCollisionError(f"Could not resolve collision for key: {original_key}")

        # Update collision statistics
        if collision_count > 0:
            self._collision_counter[original_key] = collision_count
            self.logger.warning(f"Key collision resolved after {collision_count} attempts: {original_key}")

        return key

    def _register_key(self, key: str, components: CacheKeyComponents) -> None:
        """Register a generated key in the registry."""
        collision_count = self._collision_counter.get(key.split("_c")[0], 0)

        metadata = CacheKeyMetadata(
            key=key,
            components=components,
            created_at=time.time(),
            last_accessed=time.time(),
            access_count=1,
            collision_count=collision_count,
            is_collision_resolved=collision_count > 0,
        )

        self._key_registry[key] = metadata

    def _extract_project_id(self, project_path: str) -> str:
        """Extract project ID from project path."""
        # Use the last directory name as project ID
        import os

        project_id = os.path.basename(project_path.rstrip(os.sep))

        # Clean up the project ID
        project_id = project_id.replace(" ", "_").replace("-", "_")

        # If empty, generate a hash-based ID
        if not project_id:
            project_id = self._generate_content_hash(project_path)[:8]

        return project_id

    def _matches_pattern(self, key: str, pattern: str) -> bool:
        """Check if key matches a pattern (simple wildcard matching)."""
        import fnmatch

        return fnmatch.fnmatch(key, pattern)


# Global cache key generator instance
_cache_key_generator: CacheKeyGenerator | None = None


def get_cache_key_generator() -> CacheKeyGenerator:
    """
    Get the global cache key generator instance.

    Returns:
        CacheKeyGenerator: The global cache key generator instance
    """
    global _cache_key_generator
    if _cache_key_generator is None:
        _cache_key_generator = CacheKeyGenerator()
    return _cache_key_generator


def reset_cache_key_generator() -> None:
    """Reset the global cache key generator (mainly for testing)."""
    global _cache_key_generator
    _cache_key_generator = None
