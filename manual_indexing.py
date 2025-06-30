#!/usr/bin/env python3
"""
Manual Indexing Tool for Codebase RAG MCP Server

This standalone script allows users to perform indexing operations independently
from the MCP server, particularly useful for large codebases that might take
several minutes to process.

Usage:
    python manual_indexing.py -d /path/to/repo/dir/ -m clear_existing
    python manual_indexing.py -d /path/to/repo/dir/ -m incremental
    python manual_indexing.py --directory ./src --mode incremental --verbose
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add src directory to path to import our modules
sys.path.insert(0, str(Path(__file__).parent / "src"))

from services.indexing_service import IndexingService
from services.qdrant_service import QdrantService
from services.embedding_service import EmbeddingService
from services.file_metadata_service import FileMetadataService
from services.change_detector_service import ChangeDetectorService
from services.project_analysis_service import ProjectAnalysisService
from utils.performance_monitor import ProgressTracker, MemoryMonitor
from utils import format_duration, format_memory_size


class ManualIndexingTool:
    """
    Manual indexing tool for standalone indexing operations.
    
    This tool provides the same functionality as the MCP server's indexing
    capabilities but can be run independently for heavy operations.
    """
    
    def __init__(self, verbose: bool = False):
        """
        Initialize the manual indexing tool.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        self.setup_logging()
        
        # Initialize services
        self.qdrant_service = QdrantService()
        self.embedding_service = EmbeddingService()
        self.indexing_service = IndexingService()
        self.metadata_service = FileMetadataService(self.qdrant_service)
        self.change_detector = ChangeDetectorService(self.metadata_service)
        self.project_analysis = ProjectAnalysisService()
        
        # Performance monitoring
        self.memory_monitor = MemoryMonitor()
        self.progress_tracker = None  # Will be initialized when we know total items
        
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        # Reduce noise from third-party libraries
        if not self.verbose:
            logging.getLogger('qdrant_client').setLevel(logging.WARNING)
            logging.getLogger('httpx').setLevel(logging.WARNING)
    
    def validate_arguments(self, directory: str, mode: str) -> tuple[bool, str]:
        """
        Validate command-line arguments.
        
        Args:
            directory: Target directory path
            mode: Indexing mode
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Validate directory
        dir_path = Path(directory)
        if not dir_path.exists():
            return False, f"Directory does not exist: {directory}"
        
        if not dir_path.is_dir():
            return False, f"Path is not a directory: {directory}"
        
        if not os.access(dir_path, os.R_OK):
            return False, f"Directory is not readable: {directory}"
        
        # Validate mode
        valid_modes = ['clear_existing', 'incremental']
        if mode not in valid_modes:
            return False, f"Invalid mode '{mode}'. Must be one of: {', '.join(valid_modes)}"
        
        return True, ""
    
    def check_dependencies(self) -> tuple[bool, list[str]]:
        """
        Check if required dependencies are available.
        
        Returns:
            Tuple of (all_available, missing_services)
        """
        missing = []
        
        # Check Qdrant connection
        try:
            collections = self.qdrant_service.client.get_collections()
            self.logger.debug(f"Qdrant connection successful, found {len(collections.collections)} collections")
        except Exception as e:
            missing.append(f"Qdrant (Error: {e})")
        
        # Check embedding service
        try:
            # Simple health check for embedding service
            if not hasattr(self.embedding_service, 'generate_embeddings'):
                missing.append("Embedding service not properly initialized")
        except Exception as e:
            missing.append(f"Embedding service (Error: {e})")
        
        return len(missing) == 0, missing
    
    def estimate_indexing_time(self, directory: str, mode: str = 'clear_existing') -> tuple[int, int, float]:
        """
        Estimate indexing time and provide statistics.
        
        Args:
            directory: Directory to analyze
            mode: Indexing mode ('clear_existing' or 'incremental')
            
        Returns:
            Tuple of (file_count, total_size_mb, estimated_minutes)
        """
        try:
            # Use existing project analysis service
            analysis_result = self.project_analysis.analyze_directory_structure(directory)
            
            total_files = analysis_result.get('relevant_files', 0)
            total_size_mb = analysis_result.get('size_analysis', {}).get('total_size_mb', 0)
            
            if mode == 'incremental':
                # For incremental mode, try to detect actual changes
                try:
                    # Get project context
                    project_context = self.project_analysis.get_project_context(directory)
                    project_name = project_context.get('project_name', 'unknown')
                    
                    # Get current files
                    relevant_files = self.project_analysis.get_relevant_files(directory)
                    
                    # Detect changes
                    changes = self.change_detector.detect_changes(
                        project_name=project_name,
                        current_files=relevant_files,
                        project_root=directory
                    )
                    
                    if not changes.has_changes:
                        return 0, 0, 0.0  # No changes to process
                    
                    # Get actual files that need processing
                    files_to_reindex = changes.get_files_to_reindex()
                    files_to_remove = changes.get_files_to_remove()
                    
                    actual_file_count = len(files_to_reindex)
                    
                    # Estimate size of changed files only
                    changed_size_mb = 0
                    for file_path in files_to_reindex:
                        try:
                            file_size = Path(file_path).stat().st_size / (1024 * 1024)
                            changed_size_mb += file_size
                        except:
                            pass
                    
                    self.logger.info(f"Incremental mode: {actual_file_count} files need processing (out of {total_files} total)")
                    
                    file_count = actual_file_count
                    total_size_mb = changed_size_mb
                    
                except Exception as e:
                    self.logger.warning(f"Could not detect changes for incremental estimation, using full count: {e}")
                    file_count = total_files
            else:
                file_count = total_files
            
            # Rough estimation: ~100 files per minute for average-sized files
            # This is a conservative estimate and will vary based on file size and system performance
            base_rate = 100  # files per minute
            size_factor = max(1.0, total_size_mb / 10)  # Slow down for larger files
            estimated_minutes = (file_count / base_rate) * size_factor
            
            return file_count, total_size_mb, estimated_minutes
            
        except Exception as e:
            self.logger.warning(f"Could not estimate indexing time: {e}")
            return 0, 0, 0.0
    
    def show_pre_indexing_summary(self, directory: str, mode: str):
        """
        Show summary before starting indexing.
        
        Args:
            directory: Target directory
            mode: Indexing mode
        """
        print("\n" + "="*60)
        print("MANUAL INDEXING TOOL - PRE-INDEXING SUMMARY")
        print("="*60)
        
        print(f"📁 Directory: {directory}")
        print(f"⚙️  Mode: {mode}")
        
        # Estimate time and show stats
        file_count, size_mb, estimated_minutes = self.estimate_indexing_time(directory, mode)
        
        if file_count > 0:
            if mode == 'incremental':
                print(f"📊 Files to process: {file_count:,} (changed files only)")
            else:
                print(f"📊 Files to process: {file_count:,}")
            print(f"💾 Total size: {size_mb:.1f} MB")
            print(f"⏱️  Estimated time: {estimated_minutes:.1f} minutes")
            
            if estimated_minutes > 5:
                print("\n⚠️  WARNING: This operation may take several minutes.")
                print("   Consider running this in a separate terminal.")
        elif mode == 'incremental':
            print("✅ No changes detected - all files are up to date!")
            print("⏱️  Estimated time: 0 minutes (no processing needed)")
        else:
            print("⚠️  No files found to process")
        
        print("\n" + "-"*60)
    
    async def perform_indexing(self, directory: str, mode: str) -> bool:
        """
        Perform the actual indexing operation.
        
        Args:
            directory: Target directory
            mode: Indexing mode ('clear_existing' or 'incremental')
            
        Returns:
            True if indexing was successful
        """
        start_time = time.time()
        
        try:
            # Initialize performance monitoring
            self.memory_monitor.start_monitoring()
            
            # Get project context
            project_context = self.project_analysis.get_project_context(directory)
            project_name = project_context.get('project_name', 'unknown')
            
            # For incremental mode, first check if there are any changes
            if mode == 'incremental':
                relevant_files = self.project_analysis.get_relevant_files(directory)
                changes = self.change_detector.detect_changes(
                    project_name=project_name,
                    current_files=relevant_files,
                    project_root=directory
                )
                
                if not changes.has_changes:
                    print(f"\n✅ No changes detected for project: {project_name}")
                    print("🎉 All files are already up to date!")
                    return True
            
            print(f"\n🚀 Starting {mode} indexing for project: {project_name}")
            
            if mode == 'clear_existing':
                result = await self.perform_full_indexing(directory, project_name)
            elif mode == 'incremental':
                result = await self.perform_incremental_indexing(directory, project_name)
            else:
                self.logger.error(f"Unknown indexing mode: {mode}")
                return False
            
            # Calculate final statistics
            duration = time.time() - start_time
            final_memory = self.memory_monitor.get_current_usage()
            
            if result:
                print(f"\n✅ Indexing completed successfully!")
                print(f"⏱️  Total time: {format_duration(duration)}")
                print(f"💾 Memory usage: {format_memory_size(final_memory)}")
            else:
                print(f"\n❌ Indexing failed!")
                print(f"⏱️  Time elapsed: {format_duration(duration)}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during indexing: {e}", exc_info=True)
            print(f"\n❌ Indexing failed with error: {e}")
            return False
        finally:
            # Clean up progress tracker if it exists
            if self.progress_tracker:
                # Progress tracker doesn't have a stop method, just clear reference
                self.progress_tracker = None
            try:
                self.memory_monitor.stop_monitoring()
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
    
    async def perform_full_indexing(self, directory: str, project_name: str) -> bool:
        """
        Perform full (clear_existing) indexing.
        
        Args:
            directory: Target directory
            project_name: Name of the project
            
        Returns:
            True if successful
        """
        print("📋 Mode: Full indexing (clear_existing)")
        
        # Clear existing metadata
        print("🗑️  Clearing existing metadata...")
        self.metadata_service.clear_project_metadata(project_name)
        
        # Process codebase
        print("📊 Processing codebase...")
        chunks = self.indexing_service.process_codebase_for_indexing(directory)
        
        if not chunks:
            print("⚠️  No files found to index")
            return False
        
        print(f"📄 Generated {len(chunks)} chunks from codebase")
        
        # Generate embeddings and store
        print("🧠 Generating and storing embeddings...")
        self._generate_and_store_embeddings(
            chunks=chunks,
            project_context={'project_name': project_name, 'source_path': directory}
        )
        
        # Store metadata for future incremental updates
        print("💾 Storing file metadata...")
        self.store_file_metadata(directory, project_name)
        
        return True
    
    async def perform_incremental_indexing(self, directory: str, project_name: str) -> bool:
        """
        Perform incremental indexing.
        
        Args:
            directory: Target directory
            project_name: Name of the project
            
        Returns:
            True if successful
        """
        print("📋 Mode: Incremental indexing")
        
        # Get current files
        print("🔍 Analyzing current files...")
        relevant_files = self.project_analysis.get_relevant_files(directory)
        
        # Detect changes
        print("🔎 Detecting changes...")
        changes = self.change_detector.detect_changes(
            project_name=project_name,
            current_files=relevant_files,
            project_root=directory
        )
        
        if not changes.has_changes:
            print("✅ No changes detected - all files are up to date!")
            return True
        
        # Show change summary
        summary = changes.get_summary()
        print(f"📊 Changes detected:")
        for change_type, count in summary.items():
            if count > 0 and change_type != 'total_changes':
                print(f"   {change_type}: {count}")
        
        # Process changed files
        files_to_reindex = changes.get_files_to_reindex()
        files_to_remove = changes.get_files_to_remove()
        
        if files_to_remove:
            print(f"🗑️  Removing {len(files_to_remove)} obsolete entries...")
            # TODO: Implement removal from vector database
            
        if files_to_reindex:
            print(f"🔄 Reindexing {len(files_to_reindex)} changed files...")
            
            # Process only changed files
            chunks = self.indexing_service.process_specific_files(files_to_reindex)
            
            if chunks:
                # Generate embeddings for changed files
                self._generate_and_store_embeddings(
                    chunks=chunks,
                    project_context={'project_name': project_name, 'source_path': directory}
                )
        
        # Update metadata
        print("💾 Updating file metadata...")
        self.store_file_metadata(directory, project_name)
        
        return True
    
    def store_file_metadata(self, directory: str, project_name: str):
        """
        Store file metadata for future change detection.
        
        Args:
            directory: Source directory
            project_name: Project name
        """
        try:
            # Get all relevant files
            relevant_files = self.project_analysis.get_relevant_files(directory)
            
            # Create metadata for each file
            from models.file_metadata import FileMetadata
            metadata_list = []
            
            for file_path in relevant_files:
                try:
                    metadata = FileMetadata.from_file_path(file_path, directory)
                    metadata_list.append(metadata)
                except Exception as e:
                    self.logger.warning(f"Failed to create metadata for {file_path}: {e}")
            
            # Store metadata
            success = self.metadata_service.store_file_metadata(project_name, metadata_list)
            
            if success:
                self.logger.info(f"Stored metadata for {len(metadata_list)} files")
            else:
                self.logger.error("Failed to store file metadata")
                
        except Exception as e:
            self.logger.error(f"Error storing file metadata: {e}")
    
    def _generate_and_store_embeddings(self, chunks, project_context):
        """
        Generate embeddings and store them using the same logic as MCP tools.
        
        Args:
            chunks: List of chunks to process
            project_context: Context information including project name
        """
        try:
            import os
            from collections import defaultdict
            from qdrant_client.http.models import PointStruct
            import uuid
            
            # Get embedding model name
            model_name = os.getenv("OLLAMA_DEFAULT_EMBEDDING_MODEL", "nomic-embed-text")
            
            # Initialize Qdrant client
            from services.qdrant_service import QdrantService
            qdrant_service = QdrantService()
            
            # Group chunks by collection type (similar to MCP tools logic)
            collection_chunks = defaultdict(list)
            
            for chunk in chunks:
                file_path = chunk.metadata.get('file_path', '')
                language = chunk.metadata.get('language', 'unknown')
                
                # Determine collection type based on file characteristics
                if language in ['python', 'javascript', 'typescript', 'java', 'go', 'rust']:
                    collection_type = 'code'
                elif any(file_path.endswith(ext) for ext in ['.json', '.yaml', '.yml', '.toml', '.ini']):
                    collection_type = 'config'
                else:
                    collection_type = 'documentation'
                
                project_name = project_context.get('project_name', 'unknown')
                collection_name = f"project_{project_name}_{collection_type}"
                collection_chunks[collection_name].append(chunk)
            
            print(f"📊 Processing {len(chunks)} chunks across {len(collection_chunks)} collections")
            
            total_points = 0
            
            # Process each collection
            for collection_name, collection_chunk_list in collection_chunks.items():
                print(f"🔄 Processing collection: {collection_name} ({len(collection_chunk_list)} chunks)")
                
                # Prepare texts for embedding generation
                texts = [chunk.content for chunk in collection_chunk_list]
                
                # Generate embeddings
                embeddings = self.embedding_service.generate_embeddings(model_name, texts)
                
                if embeddings is None:
                    self.logger.error(f"Failed to generate embeddings for collection {collection_name}")
                    continue
                
                # Create points for Qdrant
                points = []
                for chunk, embedding in zip(collection_chunk_list, embeddings):
                    if embedding is None:
                        continue
                    
                    point_id = str(uuid.uuid4())
                    
                    # Prepare metadata
                    metadata = chunk.metadata.copy()
                    metadata['collection'] = collection_name
                    
                    point = PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=metadata
                    )
                    points.append(point)
                
                if points:
                    # Ensure collection exists before inserting
                    print(f"🔧 Ensuring collection exists: {collection_name}")
                    self._ensure_collection_exists(collection_name, qdrant_service)
                    
                    # Store in Qdrant
                    stats = qdrant_service.batch_upsert_with_retry(collection_name, points)
                    total_points += stats.successful_insertions
                    print(f"✅ Stored {stats.successful_insertions} points in {collection_name}")
            
            print(f"🎉 Successfully processed {total_points} total points")
            
        except Exception as e:
            self.logger.error(f"Error generating and storing embeddings: {e}")
            raise
    
    def _ensure_collection_exists(self, collection_name: str, qdrant_service):
        """
        Ensure that a Qdrant collection exists before attempting to insert data.
        
        Args:
            collection_name: Name of the collection to check/create
            qdrant_service: QdrantService instance
        """
        try:
            if not qdrant_service.collection_exists(collection_name):
                # Import required classes
                from qdrant_client.http.models import VectorParams, Distance
                
                self.logger.info(f"Creating collection: {collection_name}")
                
                # Get embedding dimension (default to 768 for nomic-embed-text)
                embedding_dimension = 768
                
                # Create the collection
                qdrant_service.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=embedding_dimension,
                        distance=Distance.COSINE
                    )
                )
                
                self.logger.info(f"Successfully created collection: {collection_name}")
            else:
                self.logger.debug(f"Collection already exists: {collection_name}")
                
        except Exception as e:
            self.logger.error(f"Failed to ensure collection {collection_name} exists: {e}")
            raise


def main():
    """Main entry point for the manual indexing tool."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Manual Indexing Tool for Codebase RAG MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -d /path/to/repo -m clear_existing
  %(prog)s -d ./src -m incremental
  %(prog)s --directory /large/codebase --mode clear_existing --verbose
        """
    )
    
    parser.add_argument(
        '-d', '--directory',
        required=True,
        help='Target directory path to index'
    )
    
    parser.add_argument(
        '-m', '--mode',
        required=True,
        choices=['clear_existing', 'incremental'],
        help='Indexing mode: clear_existing (full reindex) or incremental (only changed files)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompts (use with caution)'
    )
    
    args = parser.parse_args()
    
    # Initialize tool
    tool = ManualIndexingTool(verbose=args.verbose)
    
    # Validate arguments
    is_valid, error_msg = tool.validate_arguments(args.directory, args.mode)
    if not is_valid:
        print(f"❌ Error: {error_msg}")
        sys.exit(1)
    
    # Check dependencies
    deps_ok, missing = tool.check_dependencies()
    if not deps_ok:
        print("❌ Missing dependencies:")
        for service in missing:
            print(f"   - {service}")
        print("\nPlease ensure Qdrant is running and services are properly configured.")
        sys.exit(1)
    
    # Show pre-indexing summary
    tool.show_pre_indexing_summary(args.directory, args.mode)
    
    # Confirmation prompt (unless --no-confirm)
    if not args.no_confirm:
        response = input("\nProceed with indexing? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Indexing cancelled.")
            sys.exit(0)
    
    # Run indexing
    try:
        success = asyncio.run(tool.perform_indexing(args.directory, args.mode))
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Indexing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()