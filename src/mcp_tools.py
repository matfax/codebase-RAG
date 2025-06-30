import os
import sys
import hashlib
import time
import fnmatch
import gc
import logging
import traceback
from typing import Dict, List, Optional, Any, Set, Tuple, Callable, Generator
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import threading
from queue import Queue, Empty
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logging.warning("psutil not available - memory monitoring disabled")

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from mcp.server.fastmcp import FastMCP
from utils.performance_monitor import MemoryMonitor, ProgressTracker

# Load environment variables from the MCP server directory
mcp_server_dir = Path(__file__).parent.parent
env_path = mcp_server_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Configure basic console logging for startup messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
console_logger = logging.getLogger(__name__)

# Custom error types for better error handling
class QdrantConnectionError(Exception):
    """Raised when connection to Qdrant fails."""
    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message)
        self.details = details

class IndexingError(Exception):
    """Raised when file indexing fails."""
    def __init__(self, message: str, file_path: str, details: Optional[str] = None):
        super().__init__(message)
        self.file_path = file_path
        self.details = details

class SearchError(Exception):
    """Raised when search operation fails."""
    def __init__(self, message: str, query: str, details: Optional[str] = None):
        super().__init__(message)
        self.query = query
        self.details = details

# Global variables for lazy initialization
_qdrant_client = None
_embeddings_manager = None

_current_project = None

# Configuration (simplified for now)
PROJECT_MARKERS = ['.git', 'pyproject.toml'] # Simplified

# Memory management configuration
MEMORY_WARNING_THRESHOLD_MB = int(os.getenv("MEMORY_WARNING_THRESHOLD_MB", "1000"))
MEMORY_CLEANUP_INTERVAL = int(os.getenv("MEMORY_CLEANUP_INTERVAL", "5"))  # Cleanup every N batches
FORCE_CLEANUP_THRESHOLD_MB = int(os.getenv("FORCE_CLEANUP_THRESHOLD_MB", "1500"))  # Force cleanup at this threshold

# Helper functions (simplified from reference)
def get_logger():
    return console_logger

def get_memory_usage_mb() -> float:
    """Get current memory usage in MB."""
    if not PSUTIL_AVAILABLE:
        return 0.0
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        return memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    except Exception as e:
        get_logger().warning(f"Failed to get memory usage: {e}")
        return 0.0

def log_memory_usage(context: str = "") -> float:
    """Log current memory usage and return the value in MB."""
    memory_mb = get_memory_usage_mb()
    if memory_mb > 0:
        get_logger().info(f"Memory usage{' ' + context if context else ''}: {memory_mb:.1f}MB")
        if memory_mb > MEMORY_WARNING_THRESHOLD_MB:
            get_logger().warning(
                f"Memory usage ({memory_mb:.1f}MB) exceeds warning threshold ({MEMORY_WARNING_THRESHOLD_MB}MB)"
            )
    return memory_mb

def force_memory_cleanup(context: str = "") -> None:
    """Force comprehensive memory cleanup."""
    get_logger().info(f"Forcing memory cleanup{' for ' + context if context else ''}")
    
    # Force garbage collection multiple times for thoroughness
    for i in range(3):
        collected = gc.collect()
        if collected > 0:
            get_logger().debug(f"GC cycle {i+1}: collected {collected} objects")
    
    # Clear any cached objects if possible
    if hasattr(gc, 'set_threshold'):
        # Temporarily lower GC thresholds to be more aggressive
        original_thresholds = gc.get_threshold()
        gc.set_threshold(100, 10, 10)
        gc.collect()
        gc.set_threshold(*original_thresholds)
    
    memory_after = log_memory_usage("after cleanup")
    
    if memory_after > FORCE_CLEANUP_THRESHOLD_MB:
        get_logger().warning(
            f"Memory still high ({memory_after:.1f}MB) after cleanup. "
            f"Consider reducing batch sizes or processing fewer files."
        )

def should_cleanup_memory(batch_count: int, force_check: bool = False) -> bool:
    """Determine if memory cleanup should be performed."""
    if force_check or batch_count % MEMORY_CLEANUP_INTERVAL == 0:
        memory_mb = get_memory_usage_mb()
        return memory_mb > MEMORY_WARNING_THRESHOLD_MB or batch_count % MEMORY_CLEANUP_INTERVAL == 0
    return False

def get_adaptive_batch_size(base_batch_size: int, memory_usage_mb: float) -> int:
    """Adjust batch size based on current memory usage."""
    if memory_usage_mb > FORCE_CLEANUP_THRESHOLD_MB:
        # Reduce batch size significantly under memory pressure
        adjusted_size = max(1, base_batch_size // 4)
        get_logger().warning(
            f"High memory pressure ({memory_usage_mb:.1f}MB), reducing batch size from {base_batch_size} to {adjusted_size}"
        )
        return adjusted_size
    elif memory_usage_mb > MEMORY_WARNING_THRESHOLD_MB:
        # Moderate reduction under memory warning
        adjusted_size = max(1, base_batch_size // 2)
        get_logger().info(
            f"Memory warning ({memory_usage_mb:.1f}MB), reducing batch size from {base_batch_size} to {adjusted_size}"
        )
        return adjusted_size
    else:
        return base_batch_size

def clear_processing_variables(*variables) -> None:
    """Explicitly clear variables and force garbage collection."""
    for var in variables:
        if var is not None:
            if hasattr(var, 'clear') and callable(getattr(var, 'clear')):
                var.clear()
            del var
    gc.collect()

def _retry_individual_points(client, collection_name: str, points: List[PointStruct]) -> Tuple[int, int]:
    """Retry individual points when batch insertion fails.
    
    Args:
        client: Qdrant client instance
        collection_name: Name of the collection
        points: List of points to retry individually
        
    Returns:
        Tuple of (successful_count, failed_count)
    """
    logger = get_logger()
    successful_count = 0
    failed_count = 0
    
    for point in points:
        try:
            client.upsert(collection_name=collection_name, points=[point])
            successful_count += 1
        except Exception as e:
            failed_count += 1
            logger.debug(f"Individual point insertion failed for {point.id}: {e}")
            # Don't retry individual points to avoid infinite loops
    
    return successful_count, failed_count

def retry_operation(func, max_attempts=3, delay=1.0):
    last_error = None
    for attempt in range(max_attempts):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_attempts - 1:
                time.sleep(delay * (attempt + 1))
                continue
    raise last_error if last_error else Exception("Unknown error during retry")

def get_qdrant_client():
    global _qdrant_client
    if _qdrant_client is None:
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", "6333"))
        try:
            _qdrant_client = QdrantClient(host=host, port=port)
            retry_operation(lambda: _qdrant_client.get_collections())
        except Exception as e:
            raise QdrantConnectionError(
                f"Failed to connect to Qdrant at {host}:{port}", details=str(e)
            )
    return _qdrant_client

def get_embeddings_manager_instance():
    global _embeddings_manager
    if _embeddings_manager is None:
        # This needs to be properly initialized with Ollama client
        # For now, a placeholder. Real implementation would use Ollama client.
        from services.embedding_service import EmbeddingService
        _embeddings_manager = EmbeddingService() # This is a simplification
    return _embeddings_manager


def get_current_project(client_directory: Optional[str] = None) -> Optional[Dict[str, str]]:
    global _current_project
    if client_directory:
        cwd = Path(client_directory).resolve()
    else:
        cwd = Path.cwd()

    for parent in [cwd] + list(cwd.parents):
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                project_name = parent.name.replace(" ", "_").replace("-", "_")
                _current_project = {
                    "name": project_name,
                    "root": str(parent),
                    "collection_prefix": f"project_{project_name}"
                }
                return _current_project
    _current_project = {
        "name": cwd.name,
        "root": str(cwd),
        "collection_prefix": f"dir_{cwd.name.replace(' ', '_').replace('-', '_')}"
    }
    return _current_project

def get_collection_name(file_path: str, file_type: str = "code") -> str:
    path = Path(file_path).resolve()
    current_project = get_current_project()
    if current_project:
        project_root = Path(current_project["root"])
        try:
            path.relative_to(project_root)
            return f"{current_project['collection_prefix']}_{file_type}"
        except ValueError:
            pass
    for parent in path.parents:
        for marker in PROJECT_MARKERS:
            if (parent / marker).exists():
                project_name = parent.name.replace(" ", "_").replace("-", "_")
                return f"project_{project_name}_{file_type}"
    return f"global_{file_type}"

def ensure_collection(collection_name: str, embedding_dimension: Optional[int] = None, embedding_model_name: Optional[str] = None, content_type: str = "general"):
    client = get_qdrant_client()
    def check_and_create():
        existing = [c.name for c in client.get_collections().collections]
        if collection_name not in existing:
            # Simplified dimension and model name for now
            final_dimension = embedding_dimension or 768 # Default dimension for nomic-embed-text
            final_model_name = embedding_model_name or os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=final_dimension,
                    distance=Distance.COSINE
                )
            )
            # Store metadata (simplified)
            metadata_payload = {
                "type": "collection_metadata",
                "embedding_model": final_model_name,
                "embedding_dimension": final_dimension,
                "content_type": content_type,
                "created_at": datetime.now().isoformat(),
            }
            metadata_point_id = hashlib.md5(f"{collection_name}_metadata".encode()).hexdigest()
            metadata_point = PointStruct(
                id=metadata_point_id,
                vector=[0.0] * final_dimension,
                payload=metadata_payload
            )
            try:
                client.upsert(
                    collection_name=collection_name,
                    points=[metadata_point]
                )
                get_logger().info(f"Created collection {collection_name} with model {final_model_name}")
            except Exception as e:
                get_logger().warning(f"Failed to store metadata for collection {collection_name}: {e}")
    retry_operation(check_and_create)

def _truncate_content(content: str, max_length: int = 1500) -> str:
    if len(content) <= max_length:
        return content
    return content[:max_length] + "\n... (truncated)"

def _expand_search_context(results: List[Dict[str, Any]], qdrant_client, search_collections: List[str], context_chunks: int = 1, embedding_dimension: int = 384) -> List[Dict[str, Any]]:
    # This is a simplified version. Full implementation is complex.
    # For now, just return original results.
    return results

def _perform_hybrid_search(
    qdrant_client,
    embedding_model,
    query: str,
    query_embedding: List[float],
    search_collections: List[str],
    n_results: int,
    search_mode: str = "hybrid",
    collection_filter: Optional[Dict] = None,
    result_processor: Optional[Callable] = None,
    metadata_extractor: Optional[Callable] = None
) -> List[Dict[str, Any]]:
    all_results = []
    for collection in search_collections:
        try:
            results = qdrant_client.search(
                collection_name=collection,
                query_vector=query_embedding,
                query_filter=collection_filter,
                limit=n_results
            )
            for result in results:
                if not isinstance(result.payload, dict):
                    continue
                base_result = {
                    "score": float(result.score),
                    "collection": collection,
                    "search_mode": search_mode,
                    "file_path": result.payload.get("file_path", ""),
                    "content": result.payload.get("content", ""),
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "project": result.payload.get("project", "unknown")
                }
                if metadata_extractor:
                    base_result.update(metadata_extractor(result.payload))
                if result_processor:
                    base_result = result_processor(base_result)
                all_results.append(base_result)
        except Exception as e:
            get_logger().debug(f"Error searching collection {collection}: {e}")
            pass
    return all_results

def clear_project_collections() -> Dict[str, Any]:
    current_project = get_current_project()
    if not current_project:
        return {"error": "No project context found", "cleared": []}
    client = get_qdrant_client()
    cleared = []
    errors = []
    existing_collections = [c.name for c in client.get_collections().collections]
    for collection_type in ['code', 'config', 'documentation']:
        collection_name = f"{current_project['collection_prefix']}_{collection_type}"
        if collection_name in existing_collections:
            try:
                client.delete_collection(collection_name)
                cleared.append(collection_name)
                get_logger().info(f"Cleared collection: {collection_name}")
            except Exception as e:
                error_msg = f"Failed to clear {collection_name}: {str(e)}"
                errors.append(error_msg)
                get_logger().error(error_msg)
    return {
        "project": current_project['name'],
        "cleared_collections": cleared,
        "errors": errors if errors else None
    }

def delete_file_chunks(file_path: str, collection_name: Optional[str] = None) -> Dict[str, Any]:
    logger = get_logger()
    try:
        if not file_path or not isinstance(file_path, str):
            return {"error": "Invalid file path"}
        abs_path = Path(file_path).resolve()
        qdrant_client = get_qdrant_client()
        if collection_name is None:
            suffix = abs_path.suffix.lower()
            if suffix in ['.json', '.yaml', '.yml', '.xml', '.toml', '.ini', '.env']:
                file_type = "config"
            elif suffix in ['.md', '.markdown', '.rst', '.txt', '.mdx']:
                file_type = "documentation"
            else:
                file_type = "code"
            collection_name = get_collection_name(str(abs_path), file_type)
        try:
            collections = [c.name for c in qdrant_client.get_collections().collections]
            if collection_name not in collections:
                return {"error": f"Collection '{collection_name}' does not exist"}
        except Exception:
            return {"error": f"Could not access collection '{collection_name}'"}
        filter_condition = Filter(
            must=[FieldCondition(key="file_path", match=MatchValue(value=str(abs_path)))]
        )
        count_response = qdrant_client.count(
            collection_name=collection_name,
            count_filter=filter_condition,
            exact=True
        )
        points_before = count_response.count
        delete_response = qdrant_client.delete(
            collection_name=collection_name,
            points_selector=filter_condition
        )
        return {
            "file_path": str(abs_path),
            "collection": collection_name,
            "deleted_points": points_before,
        }
    except Exception as e:
        return {"error": str(e), "file_path": file_path}

def check_qdrant_health(client) -> Dict[str, Any]:
    """Check Qdrant connection health and return status information."""
    try:
        start_time = time.time()
        collections = client.get_collections()
        response_time = time.time() - start_time
        
        return {
            "healthy": True,
            "response_time_ms": response_time * 1000,
            "collections_count": len(collections.collections),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "healthy": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

def retry_qdrant_operation(operation_func, operation_name: str, max_retries: int = None) -> Any:
    """Retry Qdrant operations with exponential backoff."""
    max_retries = max_retries or int(os.getenv("DB_RETRY_ATTEMPTS", "3"))
    retry_delay = float(os.getenv("DB_RETRY_DELAY", "1.0"))
    logger = get_logger()
    
    for attempt in range(max_retries + 1):
        try:
            return operation_func()
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"{operation_name} failed after {max_retries + 1} attempts: {e}")
                raise e
            
            delay = retry_delay * (2 ** attempt)  # Exponential backoff
            logger.warning(f"{operation_name} attempt {attempt + 1} failed: {e}. Retrying in {delay:.2f}s...")
            time.sleep(delay)

def log_database_metrics(stats: Dict[str, Any], operation_time: float, operation_type: str) -> None:
    """Log detailed database operation metrics."""
    logger = get_logger()
    
    points_per_second = stats["successful_insertions"] / operation_time if operation_time > 0 else 0
    success_rate = (stats["successful_insertions"] / stats["total_points"]) * 100 if stats["total_points"] > 0 else 0
    
    logger.info(
        f"Database {operation_type} metrics: "
        f"{stats['successful_insertions']}/{stats['total_points']} points "
        f"({success_rate:.1f}% success), "
        f"{points_per_second:.1f} points/sec, "
        f"{stats['batch_count']} batches, "
        f"total time: {operation_time:.2f}s"
    )
    
    if stats["failed_insertions"] > 0:
        logger.warning(f"Database operation had {stats['failed_insertions']} failed insertions")
    
    if stats["errors"]:
        logger.error(f"Database operation errors: {len(stats['errors'])} error(s)")
        for i, error in enumerate(stats["errors"][:3]):  # Log first 3 errors
            logger.error(f"  Error {i+1}: {error}")
        if len(stats["errors"]) > 3:
            logger.error(f"  ... and {len(stats['errors']) - 3} more errors")

def check_existing_index(project_info: Dict[str, str]) -> Dict[str, Any]:
    """Check if the project already has indexed data.
    
    Args:
        project_info: Project information from get_current_project()
        
    Returns:
        Dictionary with existing index information
    """
    try:
        client = get_qdrant_client()
        existing_collections = [c.name for c in client.get_collections().collections]
        
        project_collections = []
        total_points = 0
        collection_info = []
        
        if project_info:
            prefix = project_info['collection_prefix']
            for collection_type in ['code', 'config', 'documentation']:
                collection_name = f"{prefix}_{collection_type}"
                if collection_name in existing_collections:
                    try:
                        # Get collection info
                        collection_info_response = client.get_collection(collection_name)
                        points_count = collection_info_response.points_count
                        
                        project_collections.append(collection_name)
                        total_points += points_count
                        collection_info.append({
                            "name": collection_name,
                            "type": collection_type,
                            "points_count": points_count
                        })
                    except Exception as e:
                        get_logger().warning(f"Could not get info for collection {collection_name}: {e}")
        
        return {
            "has_existing_data": len(project_collections) > 0,
            "collections": project_collections,
            "total_points": total_points,
            "collection_details": collection_info,
            "project_name": project_info.get('name', 'unknown') if project_info else 'unknown'
        }
        
    except Exception as e:
        get_logger().error(f"Error checking existing index: {e}")
        return {
            "has_existing_data": False,
            "error": str(e)
        }

def estimate_indexing_time(file_count: int, existing_points: int = 0) -> Dict[str, Any]:
    """Estimate indexing time and provide recommendations.
    
    Args:
        file_count: Number of files to be indexed
        existing_points: Number of existing points in collections
        
    Returns:
        Dictionary with time estimates and recommendations
    """
    # Rough estimates based on typical performance
    seconds_per_file = 0.5  # Including file reading, embedding, and DB insertion
    estimated_seconds = file_count * seconds_per_file
    estimated_minutes = estimated_seconds / 60
    
    if existing_points > 0:
        # Estimate time saved by not re-indexing
        avg_points_per_file = existing_points / max(file_count, 1)
        time_saved_seconds = existing_points / avg_points_per_file * seconds_per_file if avg_points_per_file > 0 else 0
        time_saved_minutes = time_saved_seconds / 60
    else:
        time_saved_minutes = 0
    
    # Smart recommendation logic
    if existing_points > 0 and estimated_minutes > 5:
        recommendation = "keep_existing_recommended"
    elif estimated_minutes > 30:
        recommendation = "large_operation_confirm"
    elif estimated_minutes > 10:
        recommendation = "medium_operation"
    else:
        recommendation = "quick_operation"
    
    return {
        "estimated_time_minutes": round(estimated_minutes, 1),
        "time_saved_by_keeping_existing_minutes": round(time_saved_minutes, 1),
        "recommendation": recommendation,
        "file_count": file_count,
        "existing_points": existing_points
    }

def load_ragignore_patterns(directory: Path) -> Tuple[Set[str], Set[str]]:
    exclude_dirs = set()
    exclude_patterns = set()
    default_exclude_dirs = {
        'node_modules', '__pycache__', '.git', '.venv', 'venv', 'env', '.env',
        'dist', 'build', 'target', '.pytest_cache', '.mypy_cache', '.coverage', 'htmlcov', '.tox',
        'data', 'logs', 'tmp', 'temp', '.idea', '.vscode', '.vs', 'qdrant_storage', 'models', '.cache'
    }
    default_exclude_patterns = {
        '*.pyc', '*.pyo', '*.pyd', '.DS_Store', '*.so', '*.dylib', '*.dll', '*.class',
        '*.log', '*.lock', '*.swp', '*.swo', '*.bak', '*.tmp', '*.temp', '*.old', '*.orig', '*.rej',
        '.env*', '*.sqlite', '*.db', '*.pid'
    }
    ragignore_path = None
    for parent in [directory] + list(directory.parents):
        potential_path = parent / '.ragignore'
        if potential_path.exists():
            ragignore_path = potential_path
            break
    if not ragignore_path:
        return default_exclude_dirs, default_exclude_patterns
    try:
        with open(ragignore_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.endswith('/'):
                    exclude_dirs.add(line.rstrip('/'))
                else:
                    exclude_patterns.add(line)
    except Exception as e:
        get_logger().warning(f"Error reading .ragignore: {e}, using defaults")
        return default_exclude_dirs, default_exclude_patterns
    if not exclude_dirs and not exclude_patterns:
        return default_exclude_dirs, default_exclude_patterns
    return exclude_dirs, exclude_patterns

def _process_chunk_batch_for_streaming(
    chunks: List[Any], 
    embeddings_manager, 
    batch_size: int = 20
) -> Generator[Tuple[str, List[PointStruct]], None, None]:
    """
    Process chunks in batches and yield (collection_name, points) for streaming insertion.
    
    Args:
        chunks: List of chunk objects to process
        embeddings_manager: Embedding service instance
        batch_size: Number of chunks to process in each batch
        
    Yields:
        Tuple of (collection_name, points_list) ready for database insertion
    """
    logger = get_logger()
    
    # Group chunks by collection to enable efficient batch processing
    collection_batches = defaultdict(list)
    
    for i in range(0, len(chunks), batch_size):
        # Adaptive batch sizing based on memory pressure
        current_memory = get_memory_usage_mb()
        adaptive_batch_size = get_adaptive_batch_size(batch_size, current_memory)
        
        # Adjust batch bounds if needed
        if adaptive_batch_size != batch_size:
            batch_chunks = chunks[i:i + adaptive_batch_size]
            # Update loop increment for next iteration
            batch_size = adaptive_batch_size
        else:
            batch_chunks = chunks[i:i + batch_size]
            
        logger.info(f"Processing chunk batch {i//batch_size + 1}: {len(batch_chunks)} chunks (memory: {current_memory:.1f}MB)")
        
        # Group chunks by collection within this batch
        batch_collections = defaultdict(list)
        
        for chunk in batch_chunks:
            file_path = chunk.metadata["file_path"]
            language = chunk.metadata.get("language", "unknown")
            
            # Determine collection type
            if language in ["json", "yaml", "config"]:
                group_name = "config"
            elif language in ["markdown", "documentation"]:
                group_name = "documentation"
            else:
                group_name = "code"
            
            collection_name = get_collection_name(file_path, group_name)
            batch_collections[collection_name].append(chunk)
        
        # Process each collection batch
        for collection_name, collection_chunks in batch_collections.items():
            try:
                # Ensure collection exists
                ensure_collection(collection_name, content_type=collection_name.split('_')[-1])
                
                # Prepare texts for batch embedding generation
                texts = [chunk.content for chunk in collection_chunks]
                
                # Generate embeddings in batch
                start_time = time.time()
                embeddings = embeddings_manager.generate_embeddings(
                    os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text"), 
                    texts
                )
                embedding_time = time.time() - start_time
                
                if embeddings is None:
                    logger.error(f"Failed to generate embeddings for batch in collection {collection_name}")
                    continue
                
                # Create points for this collection batch
                points = []
                successful_embeddings = 0
                
                for chunk, embedding in zip(collection_chunks, embeddings):
                    if embedding is None:
                        logger.warning(f"Skipping chunk from {chunk.metadata['file_path']} due to failed embedding")
                        continue
                    
                    # Convert tensor to list if necessary
                    if hasattr(embedding, 'tolist'):
                        embedding_list = embedding.tolist()
                    else:
                        embedding_list = embedding
                    
                    chunk_id = hashlib.md5(
                        f"{chunk.metadata['file_path']}_{chunk.metadata.get('chunk_index', 0)}".encode()
                    ).hexdigest()
                    
                    payload = {
                        "file_path": chunk.metadata["file_path"],
                        "content": chunk.content,
                        "chunk_index": chunk.metadata.get("chunk_index", 0),
                        "line_start": chunk.metadata.get("line_start", 0),
                        "line_end": chunk.metadata.get("line_end", 0),
                        "language": chunk.metadata.get("language", "unknown"),
                        "chunk_type": "file_chunk",
                    }
                    
                    points.append(PointStruct(id=chunk_id, vector=embedding_list, payload=payload))
                    successful_embeddings += 1
                
                if points:
                    logger.info(
                        f"Prepared {len(points)} points for collection {collection_name} "
                        f"(embedding time: {embedding_time:.2f}s)"
                    )
                    yield collection_name, points
                    
                    # Clear points from memory after yielding
                    clear_processing_variables(points)
                else:
                    logger.warning(f"No valid points generated for collection {collection_name}")
                
                # Clear intermediate variables
                clear_processing_variables(texts, embeddings, collection_chunks)
                    
            except Exception as e:
                logger.error(f"Error processing batch for collection {collection_name}: {e}")
                # Clean up on error
                clear_processing_variables(texts if 'texts' in locals() else None, 
                                         embeddings if 'embeddings' in locals() else None,
                                         collection_chunks if 'collection_chunks' in locals() else None)
                continue
        
        # Memory cleanup between processing batches
        if should_cleanup_memory(i//batch_size + 1):
            force_memory_cleanup(f"chunk batch {i//batch_size + 1}")
        else:
            gc.collect()

def _stream_points_to_qdrant(
    collection_points_generator: Generator[Tuple[str, List[PointStruct]], None, None],
    qdrant_batch_size: int = 500
) -> Dict[str, Any]:
    """
    Stream points to Qdrant in configurable batches with comprehensive monitoring.
    
    Args:
        collection_points_generator: Generator yielding (collection_name, points) tuples
        qdrant_batch_size: Maximum number of points to insert in each Qdrant operation
        
    Returns:
        Dictionary with detailed insertion statistics, timing metrics, and error information
    """
    logger = get_logger()
    qdrant_client = get_qdrant_client()
    
    # Enhanced statistics tracking
    stats = {
        "total_points": 0,
        "successful_insertions": 0,
        "failed_insertions": 0,
        "collections_used": set(),
        "batch_count": 0,
        "errors": [],
        "timing_metrics": {
            "total_time": 0.0,
            "avg_batch_time": 0.0,
            "min_batch_time": float('inf'),
            "max_batch_time": 0.0
        },
        "health_checks": [],
        "retry_count": 0,
        "partial_failures": 0
    }
    
    operation_start_time = time.time()
    
    try:
        for collection_name, points in collection_points_generator:
            stats["collections_used"].add(collection_name)
            
            # Split points into Qdrant-sized batches
            for i in range(0, len(points), qdrant_batch_size):
                batch_points = points[i:i + qdrant_batch_size]
                stats["batch_count"] += 1
                stats["total_points"] += len(batch_points)
                
                # Periodic health check
                health_check_interval = int(os.getenv("DB_HEALTH_CHECK_INTERVAL", "50"))
                if stats["batch_count"] % health_check_interval == 0:
                    health_status = check_qdrant_health(qdrant_client)
                    stats["health_checks"].append(health_status)
                    if not health_status["healthy"]:
                        logger.warning(f"Qdrant health check failed: {health_status['error']}")
                        # Attempt to reconnect
                        try:
                            global _qdrant_client
                            _qdrant_client = None  # Force reconnection
                            qdrant_client = get_qdrant_client()
                        except Exception as reconnect_error:
                            error_msg = f"Failed to reconnect to Qdrant: {reconnect_error}"
                            logger.error(error_msg)
                            stats["errors"].append(error_msg)
                
                # Perform insertion with retry logic
                insertion_start = time.time()
                insertion_successful = False
                
                def insert_batch():
                    return qdrant_client.upsert(collection_name=collection_name, points=batch_points)
                
                try:
                    retry_qdrant_operation(
                        insert_batch, 
                        f"batch insertion to {collection_name}",
                        max_retries=int(os.getenv("DB_RETRY_ATTEMPTS", "3"))
                    )
                    insertion_time = time.time() - insertion_start
                    insertion_successful = True
                    
                    # Update timing metrics
                    stats["timing_metrics"]["min_batch_time"] = min(
                        stats["timing_metrics"]["min_batch_time"], insertion_time
                    )
                    stats["timing_metrics"]["max_batch_time"] = max(
                        stats["timing_metrics"]["max_batch_time"], insertion_time
                    )
                    
                    stats["successful_insertions"] += len(batch_points)
                    logger.info(
                        f"Inserted {len(batch_points)} points to {collection_name} "
                        f"(batch {stats['batch_count']}, time: {insertion_time:.2f}s)"
                    )
                    
                except Exception as e:
                    insertion_time = time.time() - insertion_start
                    
                    # Try individual point insertion for partial recovery
                    if len(batch_points) > 1:
                        logger.info(f"Attempting individual point recovery for failed batch {stats['batch_count']}")
                        individual_successes, individual_failures = _retry_individual_points(
                            qdrant_client, collection_name, batch_points
                        )
                        
                        stats["successful_insertions"] += individual_successes
                        stats["failed_insertions"] += individual_failures
                        stats["partial_failures"] += 1
                        
                        if individual_successes > 0:
                            logger.info(f"Recovered {individual_successes}/{len(batch_points)} points individually")
                    else:
                        stats["failed_insertions"] += len(batch_points)
                    
                    error_msg = f"Failed to insert batch {stats['batch_count']} to {collection_name}: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                    stats["retry_count"] += 1
                
                # Memory cleanup after each database batch
                del batch_points
                
                # Check if comprehensive cleanup is needed
                if should_cleanup_memory(stats['batch_count'], force_check=True):
                    force_memory_cleanup(f"database batch {stats['batch_count']}")
                else:
                    gc.collect()
                
                # Log memory usage and metrics periodically
                if stats['batch_count'] % 10 == 0:
                    log_memory_usage(f"after {stats['batch_count']} database batches")
                    
                    # Log intermediate metrics
                    current_time = time.time() - operation_start_time
                    points_per_second = stats["successful_insertions"] / current_time if current_time > 0 else 0
                    logger.info(
                        f"Progress: {stats['successful_insertions']} points inserted, "
                        f"{points_per_second:.1f} points/sec, "
                        f"{stats['batch_count']} batches processed"
                    )
        
        # Finalize timing metrics
        total_operation_time = time.time() - operation_start_time
        stats["timing_metrics"]["total_time"] = total_operation_time
        if stats["batch_count"] > 0:
            stats["timing_metrics"]["avg_batch_time"] = total_operation_time / stats["batch_count"]
        
        # Convert set to list for JSON serialization
        stats["collections_used"] = list(stats["collections_used"])
        
        # Log comprehensive final metrics
        log_database_metrics(stats, total_operation_time, "streaming insertion")
        
        # Additional detailed logging
        logger.info(
            f"Streaming insertion complete: {stats['successful_insertions']} points inserted, "
            f"{stats['failed_insertions']} failed, {stats['batch_count']} batches, "
            f"{stats['retry_count']} retries, {stats['partial_failures']} partial failures"
        )
        
        if stats["health_checks"]:
            healthy_checks = sum(1 for check in stats["health_checks"] if check["healthy"])
            logger.info(f"Health checks: {healthy_checks}/{len(stats['health_checks'])} passed")
        
        return stats
        
    except Exception as e:
        total_operation_time = time.time() - operation_start_time
        stats["timing_metrics"]["total_time"] = total_operation_time
        
        error_msg = f"Critical error during streaming insertion: {e}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        stats["collections_used"] = list(stats["collections_used"])
        
        # Log metrics even on failure
        if stats["total_points"] > 0:
            log_database_metrics(stats, total_operation_time, "failed streaming insertion")
        
        return stats

def register_mcp_tools(mcp_app: FastMCP):
    @mcp_app.tool()
    async def health_check():
        """Check the health of the MCP server."""
        return {"status": "ok"}

    @mcp_app.tool()
    def index_directory(directory: str = ".", patterns: List[str] = None, recursive: bool = True, clear_existing: bool = False, incremental: bool = False) -> Dict[str, Any]:
        """
        Index files in a directory with smart existing data detection and time estimation.
        
        Args:
            directory: Directory to index (default: current directory)
            patterns: File patterns to include (default: common code file types)
            recursive: Whether to index subdirectories (default: True)
            clear_existing: Whether to clear existing indexed data (default: False)
                          If False and existing data is found, returns recommendations instead of indexing
            incremental: Whether to use incremental indexing (only process changed files) (default: False)
        
        Returns:
            Dictionary with indexing results, time estimates, or recommendations for existing data
        """
        try:
            # Initialize memory monitoring
            memory_threshold = float(os.getenv('MEMORY_WARNING_THRESHOLD_MB', '1000'))
            memory_monitor = MemoryMonitor(warning_threshold_mb=memory_threshold)
            logger = get_logger()
            
            # Initial memory check
            initial_memory = memory_monitor.check_memory_usage(logger)
            logger.info(f"Index operation starting - Initial memory: {initial_memory['memory_mb']} MB")
            
            if patterns is None:
                patterns = ["*.py", "*.js", "*.ts", "*.jsx", "*.tsx", "*.java", "*.go", "*.rs",
                            "*.sh", "*.bash", "*.zsh", "*.fish",
                            "*.tf", "*.tfvars",
                            "*.json", "*.yaml", "*.yml", "*.toml", "*.ini",
                            "*.md", "*.markdown", "*.rst", "*.txt",
                            ".gitignore", ".dockerignore", ".prettierrc*", ".eslintrc*", ".editorconfig", ".npmrc", ".yarnrc", ".ragignore"]

            if not directory:
                return {"error": "Directory parameter is required"}

            dir_path = Path(directory).resolve()
            if not dir_path.exists():
                return {"error": f"Directory not found: {directory}"}

            current_project = get_current_project(client_directory=str(dir_path))
            
            # Check for existing indexed data BEFORE processing
            existing_index_info = check_existing_index(current_project)
            
            # Get file count estimation for better decision making
            from services.indexing_service import IndexingService
            indexing_service = IndexingService()
            
            # Quick file count without full processing
            get_logger().info("Analyzing directory structure...")
            try:
                # Use project analysis service for quick file count
                from services.project_analysis_service import ProjectAnalysisService
                analysis_service = ProjectAnalysisService()
                quick_analysis = analysis_service.analyze_repository(str(dir_path))
                estimated_file_count = quick_analysis.get('relevant_files', 0)
            except Exception as e:
                get_logger().warning(f"Could not get quick file count: {e}, proceeding with full analysis")
                estimated_file_count = 0
            
            # Generate intelligent time estimates and recommendations
            from services.time_estimator_service import TimeEstimatorService
            time_estimator = TimeEstimatorService()
            
            # Determine mode for estimation
            estimation_mode = 'incremental' if incremental else 'clear_existing'
            estimate = time_estimator.estimate_indexing_time(str(dir_path), estimation_mode)
            
            # Legacy time estimates for backward compatibility
            time_estimates = {
                'estimated_time_minutes': estimate.estimated_minutes,
                'time_saved_by_keeping_existing_minutes': 0.0 if not incremental else estimate.estimated_minutes * 0.8,
                'recommendation': estimate.recommendation,
                'file_count': estimate.file_count,
                'existing_points': existing_index_info.get('total_points', 0)
            }
            
            # Smart decision logic for clear_existing
            if existing_index_info.get('has_existing_data', False) and not clear_existing:
                # Found existing data but clear_existing is False
                return {
                    "success": False,
                    "action_required": True,
                    "message": "Existing indexed data found for this project",
                    "existing_data": {
                        "project_name": existing_index_info['project_name'],
                        "collections": existing_index_info['collections'],
                        "total_points": existing_index_info['total_points'],
                        "collection_details": existing_index_info.get('collection_details', [])
                    },
                    "estimates": time_estimates,
                    "recommendations": {
                        "keep_existing": {
                            "description": "Keep existing data to save time",
                            "time_saved_minutes": time_estimates['time_saved_by_keeping_existing_minutes'],
                            "action": "Use existing indexed data for searches"
                        },
                        "incremental_update": {
                            "description": "Add only new/modified files using incremental mode",
                            "action": "Call index_directory again with incremental=true",
                            "estimated_time_minutes": time_estimator.estimate_indexing_time(str(dir_path), 'incremental').estimated_minutes
                        },
                        "full_reindex": {
                            "description": "Clear existing data and reindex everything",
                            "estimated_time_minutes": time_estimates['estimated_time_minutes'],
                            "action": "Call index_directory again with clear_existing=true"
                        },
                        "manual_tool": {
                            "description": "Use standalone manual indexing tool for heavy operations",
                            "command": time_estimator.get_manual_tool_command(str(dir_path), estimation_mode),
                            "recommended": time_estimator.should_recommend_manual_tool(estimate),
                            "reason": "Recommended for operations exceeding 5 minutes"
                        }
                    },
                    "directory": str(dir_path)
                }
            
            # Handle clearing existing data if requested
            if clear_existing and existing_index_info.get('has_existing_data', False):
                get_logger().info(f"Clearing existing data for project {existing_index_info['project_name']}...")
                clear_result = clear_project_collections()
                if clear_result.get("errors"):
                    return {"error": "Failed to clear some collections", "clear_errors": clear_result["errors"]}
                get_logger().info(f"Cleared {len(clear_result.get('cleared_collections', []))} collections")

            exclude_dirs, exclude_patterns = load_ragignore_patterns(dir_path)

            indexed_files = []
            errors = []
            collections_used = set()

            # Now proceed with full indexing
            get_logger().info("Processing codebase for indexing...")
            
            # Memory check before heavy processing
            pre_index_memory = memory_monitor.check_memory_usage(logger)
            logger.info(f"Pre-indexing memory: {pre_index_memory['memory_mb']} MB")
            
            # Pass incremental mode and project name to indexing service
            project_name = current_project.get('name') if current_project else None
            processed_chunks = indexing_service.process_codebase_for_indexing(
                str(dir_path), 
                incremental_mode=incremental,
                project_name=project_name
            )
            
            # Get progress summary after processing
            progress_summary = indexing_service.get_progress_summary()
            
            # Memory check after processing
            post_index_memory = memory_monitor.check_memory_usage(logger)
            logger.info(f"Post-indexing memory: {post_index_memory['memory_mb']} MB")

            if not processed_chunks:
                return {
                "success": False,
                "summary": "No relevant files found to index",
                "total_files": 0,
                "total_points": 0,
                "collections": [],
                "project_context": current_project["name"] if current_project else "no project",
                "directory": str(dir_path),
                "existing_data": existing_index_info if existing_index_info.get('has_existing_data', False) else None
            }

            # Get configuration for streaming processing
            chunk_batch_size = int(os.getenv("INDEXING_BATCH_SIZE", "20"))
            qdrant_batch_size = int(os.getenv("QDRANT_BATCH_SIZE", "500"))
            
            # Check initial memory and adapt batch sizes if needed
            initial_memory = get_memory_usage_mb()
            chunk_batch_size = get_adaptive_batch_size(chunk_batch_size, initial_memory)
            qdrant_batch_size = get_adaptive_batch_size(qdrant_batch_size, initial_memory)
            
            get_logger().info(f"Starting streaming indexing with chunk batch size: {chunk_batch_size}, Qdrant batch size: {qdrant_batch_size}")
            
            # Log initial memory usage
            log_memory_usage("before indexing")
            
            embeddings_manager = get_embeddings_manager_instance()
            
            # Create streaming pipeline: chunks -> embeddings -> database
            collection_points_generator = _process_chunk_batch_for_streaming(
                processed_chunks, 
                embeddings_manager, 
                batch_size=chunk_batch_size
            )
            
            # Stream points to Qdrant and collect statistics
            streaming_stats = _stream_points_to_qdrant(
                collection_points_generator,
                qdrant_batch_size=qdrant_batch_size
            )
            
            # Extract results from streaming stats
            collections_used = set(streaming_stats["collections_used"])
            total_indexed_points = streaming_stats["successful_insertions"]
            
            # Get unique file paths without creating huge lists for large codebases
            unique_files = set()
            for chunk in processed_chunks:
                unique_files.add(chunk.metadata["file_path"])
            
            indexed_files_count = len(unique_files)
            
            # For MCP response size management, limit the files list
            MAX_FILES_IN_RESPONSE = int(os.getenv("MAX_FILES_IN_RESPONSE", "50"))
            if indexed_files_count <= MAX_FILES_IN_RESPONSE:
                indexed_files = list(unique_files)
            else:
                # Return only a sample of files for large codebases
                indexed_files = list(unique_files)[:MAX_FILES_IN_RESPONSE]
                get_logger().info(f"Truncating file list in response: showing {MAX_FILES_IN_RESPONSE} of {indexed_files_count} files")
            
            # Add any errors from streaming to existing errors list
            if streaming_stats["errors"]:
                errors.extend(streaming_stats["errors"])
            
            # Final memory cleanup and logging
            force_memory_cleanup("indexing complete")
            final_memory = log_memory_usage("final")

            # Create a compact result for MCP response
            actual_time = streaming_stats.get("timing_metrics", {}).get("total_time", 0)
            
            result = {
                "success": True,
                "summary": f"Successfully indexed {indexed_files_count} files into {len(collections_used)} collections",
                "total_files": indexed_files_count,
                "total_points": total_indexed_points,
                "collections": list(collections_used),
                "project_context": current_project["name"] if current_project else "no project",
                "directory": str(dir_path),
                "was_reindexed": clear_existing and existing_index_info.get('has_existing_data', False),
                "performance": {
                    "batches_processed": streaming_stats["batch_count"],
                    "processing_time_seconds": actual_time,
                    "estimated_time_minutes": time_estimates.get('estimated_time_minutes', 0),
                    "actual_time_minutes": round(actual_time / 60, 1) if actual_time > 0 else 0,
                    "points_per_second": round(total_indexed_points / actual_time, 1) if actual_time > 0 else 0,
                    "memory_efficient": True,
                    "final_memory_mb": final_memory if 'final_memory' in locals() else None,
                    "progress_tracking": progress_summary
                },
                "errors": {
                    "count": len(errors) if errors else 0,
                    "failed_insertions": streaming_stats["failed_insertions"],
                    "retry_count": streaming_stats.get("retry_count", 0),
                    "details": errors[:3] if errors else None  # Only first 3 errors
                }
            }
            
            # Add change summary for incremental mode
            if incremental and hasattr(indexing_service, '_change_summary'):
                result["change_summary"] = indexing_service._change_summary
            elif incremental:
                # If no change summary available, add basic info
                result["change_summary"] = {
                    "mode": "incremental",
                    "files_processed": indexed_files_count,
                    "note": "Change detection completed successfully"
                }
            else:
                result["change_summary"] = {
                    "mode": "full_reindex",
                    "files_processed": indexed_files_count
                }
            
            # Add sample files only if the list is manageable
            if indexed_files_count <= MAX_FILES_IN_RESPONSE:
                result["indexed_files"] = indexed_files
            else:
                result["sample_files"] = indexed_files  # Sample of files
                result["note"] = f"Showing {len(indexed_files)} sample files out of {indexed_files_count} total files indexed"
            return result

        except Exception as e:
            error_msg = str(e)
            # Truncate very long error messages
            if len(error_msg) > 1000:
                error_msg = error_msg[:1000] + "... (truncated)"
            
            return {
                "success": False,
                "error": error_msg,
                "directory": directory,
                "summary": "Indexing failed due to error"
            }

    @mcp_app.tool()
    def search(
        query: str,
        n_results: int = 5,
        cross_project: bool = False,
        search_mode: str = "hybrid",
        include_context: bool = True,
        context_chunks: int = 1,
    ) -> Dict[str, Any]:
        """
        Search indexed content (defaults to current project only)
        """
        try:
            if not query or not isinstance(query, str):
                return {"error": "Invalid query"}
            if n_results < 1 or n_results > 100:
                return {"error": "Invalid result count"}

            embeddings_manager = get_embeddings_manager_instance()
            qdrant_client = get_qdrant_client()

            query_embedding_tensor = embeddings_manager.generate_embeddings(os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text"), query)
            
            # Handle embedding generation failure
            if query_embedding_tensor is None:
                return {"error": "Failed to generate embedding for query (empty query or embedding error)", "query": query}
            
            query_embedding = query_embedding_tensor.tolist()

            all_collections = [c.name for c in qdrant_client.get_collections().collections]
            if cross_project:
                search_collections = all_collections
            else:
                current_project = get_current_project()
                if current_project:
                    search_collections = [
                        c for c in all_collections if c.startswith(current_project['collection_prefix'])
                    ]
                else:
                    search_collections = [
                        c for c in all_collections if c.startswith('global_')
                    ]

            def general_metadata_extractor(payload):
                collection = payload.get("collection", "")
                return {
                    "display_path": payload.get("file_path", ""),
                    "type": "code" if "_code" in collection else ("config" if "_config" in collection else "docs"),
                    "language": payload.get("language", ""),
                    "line_start": payload.get("line_start", 0),
                    "line_end": payload.get("line_end", 0)
                }

            all_results = _perform_hybrid_search(
                qdrant_client=qdrant_client,
                embedding_model=embeddings_manager,
                query=query,
                query_embedding=query_embedding,
                search_collections=search_collections,
                n_results=n_results,
                search_mode=search_mode,
                metadata_extractor=general_metadata_extractor
            )

            if include_context and all_results:
                all_results = _expand_search_context(all_results, qdrant_client, search_collections, context_chunks, 768) # Updated dimension

            for result in all_results:
                if "content" in result:
                    result["content"] = _truncate_content(result["content"], max_length=1500)
                if "expanded_content" in result:
                    result["expanded_content"] = _truncate_content(result["expanded_content"], max_length=2000)

            current_project = get_current_project()

            result = {
                "results": all_results,
                "query": query,
                "total": len(all_results),
                "project_context": current_project["name"] if current_project else "no project",
                "search_scope": "all projects" if cross_project else "current project",
                "search_mode": search_mode,
                "collections_searched": search_collections
            }
            return result

        except Exception as e:
            tb_str = traceback.format_exc()
            return {"error": str(e), "query": query}
    
    # Register additional tools that were defined outside the function
    @mcp_app.tool()
    def analyze_repository_tool(directory: str = ".") -> Dict[str, Any]:
        """
        Analyze repository structure and provide detailed statistics for indexing planning.
        """
        return analyze_repository(directory)
    
    @mcp_app.tool()
    def get_file_filtering_stats_tool(directory: str = ".") -> Dict[str, Any]:
        """
        Get detailed statistics about file filtering for debugging and optimization.
        """
        return get_file_filtering_stats(directory)
    
    @mcp_app.tool()
    def check_index_status(directory: str = ".") -> Dict[str, Any]:
        """
        Check if a directory already has indexed data and provide recommendations.
        
        This tool helps users understand the current indexing state and make informed
        decisions about whether to reindex or use existing data.
        """
        try:
            dir_path = Path(directory).resolve()
            if not dir_path.exists():
                return {"error": f"Directory not found: {directory}"}
            
            current_project = get_current_project(client_directory=str(dir_path))
            existing_index_info = check_existing_index(current_project)
            
            if not existing_index_info.get('has_existing_data', False):
                return {
                    "has_existing_data": False,
                    "message": "No existing indexed data found for this project",
                    "project_context": current_project.get('name', 'unknown') if current_project else 'unknown',
                    "directory": str(dir_path),
                    "recommendation": "Ready for initial indexing"
                }
            
            # Get file count estimation for recommendations
            try:
                from services.project_analysis_service import ProjectAnalysisService
                analysis_service = ProjectAnalysisService()
                quick_analysis = analysis_service.analyze_repository(str(dir_path))
                estimated_file_count = quick_analysis.get('relevant_files', 0)
            except Exception:
                estimated_file_count = 0
            
            time_estimates = estimate_indexing_time(
                estimated_file_count, 
                existing_index_info.get('total_points', 0)
            )
            
            return {
                "has_existing_data": True,
                "project_context": existing_index_info['project_name'],
                "directory": str(dir_path),
                "existing_data": {
                    "collections": existing_index_info['collections'],
                    "total_points": existing_index_info['total_points'],
                    "collection_details": existing_index_info.get('collection_details', [])
                },
                "estimates": time_estimates,
                "recommendations": {
                    "current_status": "Data already indexed and ready for search",
                    "reindex_time_estimate_minutes": time_estimates['estimated_time_minutes'],
                    "time_saved_by_keeping_existing_minutes": time_estimates['time_saved_by_keeping_existing_minutes'],
                    "actions": {
                        "use_existing": "Data is ready for search operations",
                        "full_reindex": "Call index_directory with clear_existing=true to reindex everything",
                        "incremental_update": "Feature coming soon - add only new/changed files"
                    }
                }
            }
            
        except Exception as e:
            return {
                "error": f"Failed to check index status: {str(e)}",
                "directory": directory
            }
    
    @mcp_app.tool()
    def get_indexing_progress() -> Dict[str, Any]:
        """
        Get current progress of any ongoing indexing operations.
        
        This tool provides real-time progress updates during long indexing operations,
        including ETA, processing rate, memory usage, and stage information.
        """
        try:
            # For now, this is a placeholder since we need to implement
            # a global progress tracking system for async operations
            # In the current synchronous implementation, indexing completes
            # before returning, so this would mainly be useful for debugging
            
            system_memory = MemoryMonitor().get_system_memory_info()
            
            return {
                "status": "no_active_indexing",
                "message": "No indexing operations currently in progress",
                "system_info": {
                    "memory": system_memory,
                    "timestamp": time.time()
                },
                "note": "Progress tracking is available during indexing operations and returned in indexing results"
            }
            
        except Exception as e:
            return {
                "error": f"Failed to get indexing progress: {str(e)}"
            }


def analyze_repository(directory: str = ".") -> Dict[str, Any]:
    """
    Analyze repository structure and provide detailed statistics for indexing planning.
    
    This tool helps assess repository complexity, file distribution, and provides
    recommendations for optimal indexing strategies.
    
    Args:
        directory: Path to the directory to analyze (default: current directory)
    
    Returns:
        Detailed analysis including file counts, size distribution, language breakdown,
        complexity assessment, and indexing recommendations.
    """
    try:
        from services.project_analysis_service import ProjectAnalysisService
        
        console_logger.info(f"Analyzing repository: {directory}")
        
        analysis_service = ProjectAnalysisService()
        analysis = analysis_service.analyze_repository(directory)
        
        if "error" in analysis:
            console_logger.error(f"Repository analysis failed: {analysis['error']}")
            return analysis
        
        # Log key statistics
        console_logger.info(f"Repository analysis complete:")
        console_logger.info(f"  - Total files: {analysis['total_files']:,}")
        console_logger.info(f"  - Relevant files: {analysis['relevant_files']:,}")
        console_logger.info(f"  - Exclusion rate: {analysis['exclusion_rate']}%")
        console_logger.info(f"  - Repository size: {analysis['size_analysis']['total_size_mb']}MB")
        console_logger.info(f"  - Complexity level: {analysis['indexing_complexity']['level']}")
        
        return analysis
        
    except Exception as e:
        error_msg = f"Repository analysis failed: {str(e)}"
        console_logger.error(error_msg)
        tb_str = traceback.format_exc()
        console_logger.error(f"Traceback: {tb_str}")
        return {"error": error_msg, "directory": directory}


def get_file_filtering_stats(directory: str = ".") -> Dict[str, Any]:
    """
    Get detailed statistics about file filtering for debugging and optimization.
    
    This tool shows how many files are excluded by different criteria,
    helping users understand and optimize their .ragignore patterns.
    
    Args:
        directory: Path to the directory to analyze (default: current directory)
    
    Returns:
        Detailed breakdown of file filtering statistics including exclusion reasons,
        configuration settings, and recommendations.
    """
    try:
        from services.project_analysis_service import ProjectAnalysisService
        
        console_logger.info(f"Analyzing file filtering for: {directory}")
        
        analysis_service = ProjectAnalysisService()
        stats = analysis_service.get_file_filtering_stats(directory)
        
        if "error" in stats:
            console_logger.error(f"File filtering analysis failed: {stats['error']}")
            return stats
        
        # Log filtering statistics
        console_logger.info(f"File filtering analysis complete:")
        console_logger.info(f"  - Total examined: {stats['total_examined']:,}")
        console_logger.info(f"  - Included: {stats['included']:,}")
        console_logger.info(f"  - Excluded by size: {stats['excluded_by_size']:,}")
        console_logger.info(f"  - Excluded by binary detection: {stats['excluded_by_binary_extension'] + stats['excluded_by_binary_header']:,}")
        console_logger.info(f"  - Excluded by .ragignore: {stats['excluded_by_ragignore']:,}")
        
        # Calculate percentages
        total = stats['total_examined']
        if total > 0:
            stats['inclusion_rate'] = round(stats['included'] / total * 100, 1)
            stats['size_exclusion_rate'] = round(stats['excluded_by_size'] / total * 100, 1)
            stats['binary_exclusion_rate'] = round((stats['excluded_by_binary_extension'] + stats['excluded_by_binary_header']) / total * 100, 1)
            stats['ragignore_exclusion_rate'] = round(stats['excluded_by_ragignore'] / total * 100, 1)
        
        return stats
        
    except Exception as e:
        error_msg = f"File filtering analysis failed: {str(e)}"
        console_logger.error(error_msg)
        tb_str = traceback.format_exc()
        console_logger.error(f"Traceback: {tb_str}")
        return {"error": error_msg, "directory": directory}

