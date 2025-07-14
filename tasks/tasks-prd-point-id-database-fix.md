# Task List: Point ID Database Fix

## Relevant Files

- `src/services/file_metadata_service.py` - Contains the problematic _generate_point_id() method that generates invalid hex strings
- `src/services/file_metadata_service.test.py` - Unit tests for file metadata service
- `src/utils/point_id_generator.py` - New utility for deterministic UUID generation (to be created)
- `src/utils/point_id_generator.test.py` - Unit tests for Point ID generator
- `src/utils/point_id_validator.py` - New utility for Point ID format validation (to be created)
- `src/utils/point_id_validator.test.py` - Unit tests for Point ID validator
- `src/services/indexing_pipeline.py` - Contains pipeline logic where pre-indexing validation should be added
- `src/services/indexing_pipeline.test.py` - Unit tests for indexing pipeline
- `src/services/qdrant_service.py` - May need updates for Point ID validation before database operations
- `src/services/qdrant_service.test.py` - Unit tests for Qdrant service
- `tests/test_point_id_consistency.py` - Integration tests for Point ID generation consistency (to be created)
- `tests/test_indexing_performance.py` - Performance benchmarks to verify 50% speed requirement (to be created)

### Notes

- Unit tests should typically be placed alongside the code files they are testing
- Use `source .venv/bin/activate` to activate the virtual environment before running tests
- Use `uv run pytest [optional/path/to/test/file]` to run tests in this project
- Performance benchmarks should measure Point ID generation speed to ensure 50% performance requirement

## Tasks

- [ ] 1.0 Replace Hex String Point ID Generation with Deterministic UUID Generation
  - [ ] 1.1 Create new `point_id_generator.py` utility module in `src/utils/`
  - [ ] 1.2 Implement `generate_deterministic_uuid(file_path: str) -> str` using uuid5 with a consistent namespace
  - [ ] 1.3 Update `FileMetadataService._generate_point_id()` to use the new UUID generator
  - [ ] 1.4 Search codebase for other Point ID generation locations and update them
  - [ ] 1.5 Verify UUID format matches Qdrant requirements (standard UUID string format)
  - [ ] 1.6 Test deterministic behavior - same path always generates same UUID

- [ ] 2.0 Implement Point ID Validation System
  - [ ] 2.1 Create new `point_id_validator.py` utility module in `src/utils/`
  - [ ] 2.2 Implement `is_valid_point_id(point_id: str) -> bool` with UUID regex validation
  - [ ] 2.3 Implement `validate_point_id(point_id: str) -> None` that raises descriptive exceptions
  - [ ] 2.4 Add validation to `QdrantService` before any upsert operations
  - [ ] 2.5 Create unit tests covering valid UUIDs, invalid hex strings, and edge cases

- [ ] 3.0 Add Pre-indexing Validation with Early Detection
  - [ ] 3.1 Add `_validate_existing_point_ids()` method to `IndexingPipeline`
  - [ ] 3.2 Query Qdrant for a sample of existing Point IDs in metadata collection
  - [ ] 3.3 Validate sampled Point IDs using the validator
  - [ ] 3.4 If invalid IDs found, display warning with clear_existing suggestion
  - [ ] 3.5 Add validation check early in both `execute_full_indexing()` and `execute_incremental_indexing()`
  - [ ] 3.6 Implement `--skip-validation` flag for manual indexing if needed

- [ ] 4.0 Update Error Handling and User Messaging
  - [ ] 4.1 Create user-friendly error message for invalid Point ID detection
  - [ ] 4.2 Include specific remediation steps mentioning `clear_existing` mode
  - [ ] 4.3 Update `IndexingReporter` to track Point ID validation warnings
  - [ ] 4.4 Add validation status to indexing summary report
  - [ ] 4.5 Ensure error messages appear early before processing begins

- [ ] 5.0 Create Comprehensive Tests and Performance Verification
  - [ ] 5.1 Create `test_point_id_consistency.py` integration test
  - [ ] 5.2 Test that same file paths generate identical UUIDs across runs
  - [ ] 5.3 Create `test_indexing_performance.py` with timing benchmarks
  - [ ] 5.4 Measure Point ID generation speed (hex vs UUID) to verify 50% requirement
  - [ ] 5.5 Add regression tests for the original hex string issue
  - [ ] 5.6 Test end-to-end indexing with new Point ID generation
  - [ ] 5.7 Verify Qdrant accepts all generated Point IDs without errors
