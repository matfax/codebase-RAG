"""
File metadata service for managing file metadata in Qdrant.

This service handles storing and retrieving file metadata for incremental indexing,
using a dedicated metadata collection in Qdrant to track file states.
"""

import logging
from typing import Any, Union

from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.http.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

from src.models.file_metadata import FileMetadata
from src.utils.point_id_generator import generate_deterministic_uuid
from src.utils.point_id_validator import validate_point_id

from .qdrant_service import QdrantService


class FileMetadataService:
    """
    Service for managing file metadata in Qdrant vector database.

    This service uses a dedicated collection to store file metadata
    separately from the actual content embeddings, enabling efficient
    change detection for incremental indexing.
    """

    def __init__(self, qdrant_service: QdrantService | None = None):
        """
        Initialize the file metadata service.

        Args:
            qdrant_service: Optional QdrantService instance, creates new one if None
        """
        self.qdrant_service = qdrant_service or QdrantService()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        # Metadata collection naming convention
        self.METADATA_SUFFIX = "_file_metadata"

    def _get_metadata_collection_name(self, project_name: str) -> str:
        """
        Generate metadata collection name for a project.

        Args:
            project_name: Name of the project

        Returns:
            Metadata collection name
        """
        return f"project_{project_name}{self.METADATA_SUFFIX}"

    def _ensure_metadata_collection_exists(self, collection_name: str) -> bool:
        """
        Ensure metadata collection exists, create if it doesn't.

        Args:
            collection_name: Name of the metadata collection

        Returns:
            True if collection exists or was created successfully
        """
        try:
            # Check if collection exists
            collections = self.qdrant_service.client.get_collections()
            existing_names = [col.name for col in collections.collections]

            if collection_name in existing_names:
                self.logger.debug(f"Metadata collection '{collection_name}' already exists")
                return True

            # Create metadata collection with minimal vector configuration
            # We use a single-dimension vector since we only need the payload functionality
            self.qdrant_service.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=1,  # Minimal vector size since we only care about metadata
                    distance=Distance.COSINE,
                ),
            )

            self.logger.info(f"Created metadata collection: {collection_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to create metadata collection '{collection_name}': {e}")
            return False

    def store_file_metadata(self, project_name: str, metadata_list: list[FileMetadata]) -> bool:
        """
        Store file metadata for a project, updating existing entries.

        Args:
            project_name: Name of the project
            metadata_list: List of FileMetadata objects to store

        Returns:
            True if all metadata was stored successfully
        """
        if not metadata_list:
            self.logger.warning("No metadata provided to store")
            return True

        collection_name = self._get_metadata_collection_name(project_name)

        # Ensure collection exists
        if not self._ensure_metadata_collection_exists(collection_name):
            return False

        # First, remove existing entries for these files to avoid duplicates
        file_paths_to_update = [metadata.file_path for metadata in metadata_list]
        self._remove_existing_metadata(collection_name, file_paths_to_update)

        # Convert metadata to Qdrant points with consistent IDs
        points = []
        for metadata in metadata_list:
            # Use hash of file path as consistent point ID
            point_id = self._generate_point_id(metadata.file_path)

            # Create a dummy vector since Qdrant requires it
            vector = [0.0]

            # Store all metadata in the payload
            payload = metadata.to_dict()

            points.append(PointStruct(id=point_id, vector=vector, payload=payload))

        # Perform batch upsert
        try:
            stats = self.qdrant_service.batch_upsert_with_retry(collection_name=collection_name, points=points)

            success = stats.success_rate > 95  # Allow for small failure rate

            if success:
                self.logger.info(
                    f"Successfully updated {stats.successful_insertions}/{stats.total_points} "
                    f"file metadata entries for project '{project_name}'"
                )
            else:
                self.logger.error(f"Failed to store metadata: {stats.failed_insertions}/{stats.total_points} " f"failed insertions")

            return success

        except Exception as e:
            self.logger.error(f"Error storing file metadata: {e}")
            return False

    def get_project_file_metadata(self, project_name: str) -> dict[str, FileMetadata]:
        """
        Retrieve all file metadata for a project.

        Args:
            project_name: Name of the project

        Returns:
            Dictionary mapping file paths to FileMetadata objects
        """
        collection_name = self._get_metadata_collection_name(project_name)

        try:
            # Check if collection exists
            collections = self.qdrant_service.client.get_collections()
            existing_names = [col.name for col in collections.collections]

            if collection_name not in existing_names:
                self.logger.debug(f"Metadata collection '{collection_name}' does not exist")
                return {}

            # Scroll through all points in the collection
            result = {}
            offset = None

            while True:
                response = self.qdrant_service.client.scroll(
                    collection_name=collection_name,
                    limit=1000,  # Process in batches
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,  # We don't need the dummy vectors
                )

                if not response[0]:  # No more points
                    break

                # Process points in this batch
                for point in response[0]:
                    try:
                        metadata = FileMetadata.from_dict(point.payload)
                        result[metadata.file_path] = metadata
                    except Exception as e:
                        self.logger.warning(f"Failed to parse metadata for point {point.id}: {e}")

                # Update offset for next batch
                offset = response[1]  # Next offset

                if len(response[0]) < 1000:  # Last batch
                    break

            self.logger.info(f"Retrieved {len(result)} file metadata entries for project '{project_name}'")
            return result

        except Exception as e:
            self.logger.error(f"Error retrieving file metadata for project '{project_name}': {e}")
            return {}

    def update_file_metadata(self, project_name: str, metadata: FileMetadata) -> bool:
        """
        Update metadata for a specific file.

        Args:
            project_name: Name of the project
            metadata: Updated FileMetadata object

        Returns:
            True if update was successful
        """
        # For simplicity, we'll delete old entries and insert new one
        # This could be optimized to update in place, but would require tracking point IDs
        return self.store_file_metadata(project_name, [metadata])

    def remove_file_metadata(self, project_name: str, file_paths: list[str]) -> bool:
        """
        Remove metadata for specific files.

        Args:
            project_name: Name of the project
            file_paths: List of file paths to remove

        Returns:
            True if removal was successful
        """
        if not file_paths:
            return True

        collection_name = self._get_metadata_collection_name(project_name)

        try:
            # Find points to delete by filtering on file_path
            points_to_delete = []

            for file_path in file_paths:
                # Search for points with this file path
                search_result = self.qdrant_service.client.scroll(
                    collection_name=collection_name,
                    scroll_filter={"must": [{"key": "file_path", "match": {"value": file_path}}]},
                    limit=100,  # Should be few points per file
                    with_payload=False,
                    with_vectors=False,
                )

                # Collect point IDs
                for point in search_result[0]:
                    points_to_delete.append(point.id)

            if points_to_delete:
                # Delete the points
                self.qdrant_service.client.delete(collection_name=collection_name, points_selector=points_to_delete)

                self.logger.info(f"Removed {len(points_to_delete)} metadata entries for {len(file_paths)} files")

            return True

        except Exception as e:
            self.logger.error(f"Error removing file metadata: {e}")
            return False

    def clear_project_metadata(self, project_name: str) -> bool:
        """
        Clear all metadata for a project.

        Args:
            project_name: Name of the project

        Returns:
            True if clearing was successful
        """
        collection_name = self._get_metadata_collection_name(project_name)

        try:
            # Check if collection exists
            collections = self.qdrant_service.client.get_collections()
            existing_names = [col.name for col in collections.collections]

            if collection_name in existing_names:
                # Delete the entire collection
                self.qdrant_service.client.delete_collection(collection_name)
                self.logger.info(f"Cleared metadata collection: {collection_name}")
            else:
                self.logger.debug(f"Metadata collection '{collection_name}' does not exist")

            return True

        except Exception as e:
            self.logger.error(f"Error clearing project metadata: {e}")
            return False

    def get_project_metadata_stats(self, project_name: str) -> dict[str, Any]:
        """
        Get statistics about project metadata.

        Args:
            project_name: Name of the project

        Returns:
            Dictionary with metadata statistics
        """
        collection_name = self._get_metadata_collection_name(project_name)

        try:
            # Get collection info
            info = self.qdrant_service.client.get_collection(collection_name)

            return {
                "project_name": project_name,
                "collection_name": collection_name,
                "total_files": info.points_count,
                "collection_status": info.status.value if info.status else "unknown",
            }

        except UnexpectedResponse:
            # Collection doesn't exist
            return {
                "project_name": project_name,
                "collection_name": collection_name,
                "total_files": 0,
                "collection_status": "not_found",
            }
        except Exception as e:
            self.logger.error(f"Error getting metadata stats: {e}")
            return {
                "project_name": project_name,
                "collection_name": collection_name,
                "total_files": 0,
                "collection_status": "error",
                "error": str(e),
            }

    def _generate_point_id(self, file_path: str) -> str:
        """
        Generate a consistent point ID based on file path.

        Args:
            file_path: File path to generate ID for

        Returns:
            Consistent UUID string ID for the file
        """
        # Generate deterministic UUID using the new utility
        point_id = generate_deterministic_uuid(file_path)

        # Validate the generated Point ID
        validate_point_id(point_id, f"file_path: {file_path}")

        return point_id

    def _remove_existing_metadata(self, collection_name: str, file_paths: list[str]) -> None:
        """
        Remove existing metadata entries for specified files.

        Args:
            collection_name: Name of the metadata collection
            file_paths: List of file paths to remove metadata for
        """
        if not file_paths:
            return

        try:
            points_to_delete = []

            # Find existing points for these files
            for file_path in file_paths:
                try:
                    # Use consistent ID generation
                    point_id = self._generate_point_id(file_path)
                    points_to_delete.append(point_id)
                except Exception as e:
                    self.logger.warning(f"Failed to generate point ID for {file_path}: {e}")

            if points_to_delete:
                # Delete existing points
                self.qdrant_service.client.delete(collection_name=collection_name, points_selector=points_to_delete)
                self.logger.debug(f"Removed {len(points_to_delete)} existing metadata entries")

        except Exception as e:
            self.logger.warning(f"Failed to remove existing metadata: {e}")
            # Continue anyway - upsert should still work
