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

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue
from mcp.server.fastmcp import FastMCP

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

# Helper functions (simplified from reference)
def get_logger():
    return console_logger

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
        batch_chunks = chunks[i:i + batch_size]
        logger.info(f"Processing chunk batch {i//batch_size + 1}: {len(batch_chunks)} chunks")
        
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
                else:
                    logger.warning(f"No valid points generated for collection {collection_name}")
                    
            except Exception as e:
                logger.error(f"Error processing batch for collection {collection_name}: {e}")
                continue
        
        # Force garbage collection between batches to manage memory
        gc.collect()

def _stream_points_to_qdrant(
    collection_points_generator: Generator[Tuple[str, List[PointStruct]], None, None],
    qdrant_batch_size: int = 500
) -> Dict[str, Any]:
    """
    Stream points to Qdrant in configurable batches.
    
    Args:
        collection_points_generator: Generator yielding (collection_name, points) tuples
        qdrant_batch_size: Maximum number of points to insert in each Qdrant operation
        
    Returns:
        Dictionary with insertion statistics and any errors
    """
    logger = get_logger()
    qdrant_client = get_qdrant_client()
    
    stats = {
        "total_points": 0,
        "successful_insertions": 0,
        "failed_insertions": 0,
        "collections_used": set(),
        "batch_count": 0,
        "errors": []
    }
    
    try:
        for collection_name, points in collection_points_generator:
            stats["collections_used"].add(collection_name)
            
            # Split points into Qdrant-sized batches
            for i in range(0, len(points), qdrant_batch_size):
                batch_points = points[i:i + qdrant_batch_size]
                stats["batch_count"] += 1
                stats["total_points"] += len(batch_points)
                
                try:
                    insertion_start = time.time()
                    qdrant_client.upsert(collection_name=collection_name, points=batch_points)
                    insertion_time = time.time() - insertion_start
                    
                    stats["successful_insertions"] += len(batch_points)
                    logger.info(
                        f"Inserted {len(batch_points)} points to {collection_name} "
                        f"(batch {stats['batch_count']}, time: {insertion_time:.2f}s)"
                    )
                    
                except Exception as e:
                    error_msg = f"Failed to insert batch {stats['batch_count']} to {collection_name}: {e}"
                    logger.error(error_msg)
                    stats["errors"].append(error_msg)
                    stats["failed_insertions"] += len(batch_points)
                    continue
                
                # Memory cleanup after each database batch
                del batch_points
                gc.collect()
        
        # Convert set to list for JSON serialization
        stats["collections_used"] = list(stats["collections_used"])
        
        logger.info(
            f"Streaming insertion complete: {stats['successful_insertions']} points inserted, "
            f"{stats['failed_insertions']} failed, {stats['batch_count']} batches"
        )
        
        return stats
        
    except Exception as e:
        error_msg = f"Critical error during streaming insertion: {e}"
        logger.error(error_msg)
        stats["errors"].append(error_msg)
        stats["collections_used"] = list(stats["collections_used"])
        return stats

def register_mcp_tools(mcp_app: FastMCP):
    @mcp_app.tool()
    async def health_check():
        """Check the health of the MCP server."""
        return {"status": "ok"}

    @mcp_app.tool()
    def index_directory(directory: str = ".", patterns: List[str] = None, recursive: bool = True, clear_existing: bool = False) -> Dict[str, Any]:
        """
        Index files in a directory.
        """
        try:
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

            if clear_existing and current_project:
                clear_result = clear_project_collections()
                if clear_result.get("errors"):
                    return {"error": "Failed to clear some collections", "clear_errors": clear_result["errors"]}

            exclude_dirs, exclude_patterns = load_ragignore_patterns(dir_path)

            indexed_files = []
            errors = []
            collections_used = set()

            from services.indexing_service import IndexingService
            indexing_service = IndexingService()
            processed_chunks = indexing_service.process_codebase_for_indexing(str(dir_path))

            if not processed_chunks:
                return {"message": "No relevant files found to index.", "indexed_files": [], "total": 0, "collections": []}

            # Get configuration for streaming processing
            chunk_batch_size = int(os.getenv("INDEXING_BATCH_SIZE", "20"))
            qdrant_batch_size = int(os.getenv("QDRANT_BATCH_SIZE", "500"))
            
            get_logger().info(f"Starting streaming indexing with chunk batch size: {chunk_batch_size}, Qdrant batch size: {qdrant_batch_size}")
            
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
            
            # Approximate indexed files (points may not equal files if chunking is complex)
            # For now, assume each chunk represents a file for the simple use case
            indexed_files = [chunk.metadata["file_path"] for chunk in processed_chunks]
            indexed_files = list(set(indexed_files))  # Remove duplicates
            
            # Add any errors from streaming to existing errors list
            if streaming_stats["errors"]:
                errors.extend(streaming_stats["errors"])

            result = {
                "indexed_files": indexed_files,
                "total": len(indexed_files),
                "total_points": total_indexed_points,
                "collections": list(collections_used),
                "project_context": current_project["name"] if current_project else "no project",
                "directory": str(dir_path),
                "streaming_stats": {
                    "batches_processed": streaming_stats["batch_count"],
                    "successful_insertions": streaming_stats["successful_insertions"],
                    "failed_insertions": streaming_stats["failed_insertions"],
                    "memory_efficient": True
                },
                "errors": errors if errors else None
            }
            return result

        except Exception as e:
            return {"error": str(e), "directory": directory}

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

