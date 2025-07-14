"""
Unit Tests for MCP Prompts System

Tests the core functionality of the MCP Prompts guided workflows system.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add the src directory to the Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.prompts import base
from src.models.prompt_context import (
    DifficultyLevel,
    PromptContext,
    PromptType,
    TaskType,
    UserRole,
)
from src.utils.prompt_error_handler import (
    ErrorCategory,
    ErrorSeverity,
    PromptError,
    PromptErrorHandler,
)
from src.utils.prompt_validator import PromptValidator, ValidationResult
from src.utils.workflow_orchestrator import WorkflowOrchestrator


class TestPromptValidator:
    """Test cases for the PromptValidator class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = PromptValidator()

    def test_validate_directory_path_valid(self):
        """Test validation of valid directory paths."""
        with tempfile.TemporaryDirectory() as temp_dir:
            result = self.validator.validate_directory_path(temp_dir)
            assert result.valid
            assert result.value == str(Path(temp_dir).resolve())
            assert len(result.errors) == 0

    def test_validate_directory_path_invalid(self):
        """Test validation of invalid directory paths."""
        invalid_path = "/non/existent/directory"
        result = self.validator.validate_directory_path(invalid_path)
        assert not result.valid
        assert len(result.errors) > 0
        assert "does not exist" in result.errors[0]

    def test_validate_directory_path_dangerous_patterns(self):
        """Test detection of dangerous patterns in paths."""
        dangerous_paths = ["../../../etc/passwd", "~/../../root", "/etc/shadow"]

        for path in dangerous_paths:
            result = self.validator.validate_directory_path(path, must_exist=False)
            assert not result.valid
            assert any("dangerous patterns" in error for error in result.errors)

    def test_validate_component_name_valid(self):
        """Test validation of valid component names."""
        valid_names = ["MyClass", "my_function", "module.py", "package.module.Class"]

        for name in valid_names:
            result = self.validator.validate_component_name(name)
            assert result.valid
            assert result.value == name.strip()

    def test_validate_component_name_invalid(self):
        """Test validation of invalid component names."""
        result = self.validator.validate_component_name("")
        assert not result.valid
        assert "cannot be empty" in result.errors[0]

        result = self.validator.validate_component_name(123)
        assert not result.valid
        assert "must be a string" in result.errors[0]

    def test_validate_user_role(self):
        """Test validation of user role parameter."""
        # Valid role
        result = self.validator.validate_user_role("developer")
        assert result.valid
        assert result.value == "developer"

        # Invalid role
        result = self.validator.validate_user_role("invalid_role")
        assert result.valid  # Should default to valid role
        assert result.value == "developer"
        assert len(result.warnings) > 0

    def test_validate_boolean_param(self):
        """Test validation of boolean parameters."""
        # Valid boolean values
        test_cases = [
            (True, True),
            (False, False),
            ("true", True),
            ("false", False),
            ("yes", True),
            ("no", False),
            ("1", True),
            ("0", False),
        ]

        for input_val, expected in test_cases:
            result = self.validator.validate_boolean_param(input_val, "test_param")
            assert result.valid
            assert result.value == expected

    def test_validate_integer_param(self):
        """Test validation of integer parameters."""
        # Valid integer
        result = self.validator.validate_integer_param(5, "test_param", min_val=1, max_val=10)
        assert result.valid
        assert result.value == 5

        # Below minimum
        result = self.validator.validate_integer_param(0, "test_param", min_val=1, max_val=10)
        assert result.valid
        assert result.value == 1
        assert len(result.warnings) > 0

        # Above maximum
        result = self.validator.validate_integer_param(15, "test_param", min_val=1, max_val=10)
        assert result.valid
        assert result.value == 10
        assert len(result.warnings) > 0

    def test_validate_search_queries(self):
        """Test validation of search query parameters."""
        # Valid single query
        result = self.validator.validate_search_queries("test query")
        assert result.valid
        assert result.value == ["test query"]

        # Valid query list
        queries = ["query1", "query2", "query3"]
        result = self.validator.validate_search_queries(queries)
        assert result.valid
        assert result.value == queries

        # Empty query
        result = self.validator.validate_search_queries("")
        assert not result.valid
        assert "cannot be empty" in result.errors[0]

    def test_validate_prompt_parameters(self):
        """Test comprehensive parameter validation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            params = {
                "directory": temp_dir,
                "component_name": "TestClass",
                "component_type": "class",
                "user_role": "developer",
                "include_examples": True,
                "context_chunks": 2,
            }

            results = self.validator.validate_prompt_parameters("test_prompt", **params)

            # All validations should pass
            assert all(result.valid for result in results.values())
            assert "directory" in results
            assert "component_name" in results
            assert "component_type" in results

    def test_get_validation_summary(self):
        """Test validation summary generation."""
        results = {
            "param1": ValidationResult(valid=True, value="test"),
            "param2": ValidationResult(valid=False, errors=["error1"]),
            "param3": ValidationResult(valid=True, warnings=["warning1"]),
        }

        summary = self.validator.get_validation_summary(results)

        assert summary["total_parameters"] == 3
        assert summary["valid_parameters"] == 2
        assert summary["total_errors"] == 1
        assert summary["total_warnings"] == 1
        assert not summary["all_valid"]


class TestPromptErrorHandler:
    """Test cases for the PromptErrorHandler class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.error_handler = PromptErrorHandler("test_prompts")

    def test_handle_error_basic(self):
        """Test basic error handling."""
        exception = ValueError("Test error message")

        error = self.error_handler.handle_error(
            exception,
            prompt_type=PromptType.EXPLORE_PROJECT,
            parameters={"directory": "/test"},
            correlation_id="test-123",
        )

        assert error.error_id.startswith("error_")
        assert error.exception_type == "ValueError"
        assert error.message == "Test error message"
        assert error.prompt_type == PromptType.EXPLORE_PROJECT
        assert error.correlation_id == "test-123"
        assert error.category in ErrorCategory
        assert error.severity in ErrorSeverity

    def test_error_classification(self):
        """Test error classification logic."""
        # Validation error
        validation_error = ValueError("Invalid parameter format")
        error = self.error_handler.handle_error(validation_error)
        assert error.category == ErrorCategory.VALIDATION_ERROR
        assert error.severity == ErrorSeverity.WARNING

        # Timeout error
        timeout_error = TimeoutError("Operation timed out")
        error = self.error_handler.handle_error(timeout_error)
        assert error.category == ErrorCategory.TIMEOUT_ERROR
        assert error.severity == ErrorSeverity.ERROR

    def test_error_recovery_suggestions(self):
        """Test generation of recovery suggestions."""
        validation_error = ValueError("Invalid parameter")
        error = self.error_handler.handle_error(validation_error)

        assert len(error.suggested_fixes) > 0
        assert any("parameter" in fix.lower() for fix in error.suggested_fixes)
        assert error.recoverable

    def test_error_statistics(self):
        """Test error statistics tracking."""
        initial_count = self.error_handler.stats.total_errors

        # Generate a few errors
        for i in range(3):
            error = ValueError(f"Test error {i}")
            self.error_handler.handle_error(error, prompt_type=PromptType.EXPLORE_PROJECT)

        stats = self.error_handler.get_error_statistics()

        assert stats["total_errors"] == initial_count + 3
        assert PromptType.EXPLORE_PROJECT.value in stats["errors_by_prompt_type"]
        assert stats["errors_by_prompt_type"][PromptType.EXPLORE_PROJECT.value] == 3

    def test_get_user_friendly_error(self):
        """Test user-friendly error formatting."""
        error = PromptError(
            error_id="test-error",
            message="Technical error message",
            user_message="User-friendly message",
            severity=ErrorSeverity.WARNING,
            recoverable=True,
            suggested_fixes=["Fix 1", "Fix 2"],
        )

        friendly_error = self.error_handler.get_user_friendly_error(error)

        assert friendly_error["error_id"] == "test-error"
        assert friendly_error["message"] == "User-friendly message"
        assert friendly_error["severity"] == "warning"
        assert friendly_error["recoverable"] is True
        assert len(friendly_error["suggested_fixes"]) == 2

    def test_error_context_manager(self):
        """Test error context manager functionality."""
        with pytest.raises(ValueError):  # Should re-raise as PromptOperationError
            with self.error_handler.error_context(prompt_type=PromptType.EXPLORE_PROJECT, parameters={"test": "value"}):
                raise ValueError("Test error in context")


class TestWorkflowOrchestrator:
    """Test cases for the WorkflowOrchestrator class."""

    def setup_method(self):
        """Setup test fixtures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            self.orchestrator = WorkflowOrchestrator(storage_dir=temp_dir)
            self.temp_dir = temp_dir

    def test_create_workflow(self):
        """Test workflow creation."""
        workflow = self.orchestrator.create_workflow(
            workflow_type="project_exploration",
            user_id="test-user",
            session_id="test-session",
        )

        assert workflow.workflow_id.startswith("project_exploration")
        assert workflow.session_id == "test-session"
        assert workflow.user_id == "test-user"
        assert workflow.current_phase == "project_overview"
        assert workflow.workflow_id in self.orchestrator.active_workflows

    def test_execute_prompt_with_workflow(self):
        """Test prompt execution within workflow context."""
        # Create a workflow
        workflow = self.orchestrator.create_workflow("project_exploration")

        # Mock prompt function
        def mock_prompt_function(**kwargs):
            return [
                base.Message(
                    role="user",
                    content=base.TextContent(type="text", text=f"Test prompt with params: {kwargs}"),
                )
            ]

        # Execute prompt
        result = self.orchestrator.execute_prompt_with_workflow(
            workflow.workflow_id,
            PromptType.EXPLORE_PROJECT,
            mock_prompt_function,
            directory="/test",
            detail_level="overview",
        )

        assert result.prompt_type == PromptType.EXPLORE_PROJECT
        assert result.success
        assert result.execution_time_ms > 0
        assert result.context_used is not None
        assert "directory" in result.parameters_used

    def test_get_workflow_progress(self):
        """Test workflow progress tracking."""
        workflow = self.orchestrator.create_workflow("project_exploration")

        progress = self.orchestrator.get_workflow_progress(workflow.workflow_id)

        assert "workflow_id" in progress
        assert "current_phase" in progress
        assert "progress_percentage" in progress
        assert "timing" in progress
        assert "context" in progress
        assert progress["workflow_id"] == workflow.workflow_id

    def test_get_next_recommended_prompts(self):
        """Test next prompt recommendations."""
        workflow = self.orchestrator.create_workflow("project_exploration")

        recommendations = self.orchestrator.get_next_recommended_prompts(workflow.workflow_id)

        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3  # Limited to 3 recommendations

        if recommendations:  # If there are recommendations
            first_rec = recommendations[0]
            assert hasattr(first_rec, "action_id")
            assert hasattr(first_rec, "title")
            assert hasattr(first_rec, "description")

    def test_workflow_checkpoint(self):
        """Test workflow checkpointing."""
        workflow = self.orchestrator.create_workflow("project_exploration")

        # Save checkpoint
        success = self.orchestrator.save_workflow_checkpoint(workflow.workflow_id)
        assert success

        # Verify checkpoint file exists
        checkpoint_file = Path(self.temp_dir) / f"checkpoint_{workflow.workflow_id}.json"
        assert checkpoint_file.exists()

        # Load checkpoint
        # Clear active workflows first
        self.orchestrator.active_workflows.clear()
        self.orchestrator.active_contexts.clear()

        load_success = self.orchestrator.load_workflow_checkpoint(workflow.workflow_id)
        assert load_success
        assert workflow.workflow_id in self.orchestrator.active_workflows


class TestMCPPromptsSystem:
    """Test cases for the MCPPromptsSystem class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.mock_app = Mock(spec=FastMCP)

        # Mock the services to avoid dependency issues in tests
        with (
            patch("mcp_prompts.IndexingService"),
            patch("mcp_prompts.ProjectAnalysisService"),
            patch("mcp_prompts.EmbeddingService"),
        ):
            from mcp_prompts import MCPPromptsSystem

            self.prompts_system = MCPPromptsSystem(self.mock_app)

    def test_initialization(self):
        """Test system initialization."""
        assert self.prompts_system.mcp_app == self.mock_app
        assert self.prompts_system.indexing_service is not None
        assert self.prompts_system.analysis_service is not None
        assert self.prompts_system.embedding_service is not None

    def test_prompt_registration(self):
        """Test that prompts are registered with the MCP app."""
        # Check that the mcp_app.prompt decorator was called
        # This is a bit tricky to test directly, but we can verify
        # that the registration method was called during initialization
        assert self.mock_app.prompt.called

        # The exact number of calls depends on the number of prompts registered
        # We have 6 prompts: explore_project, understand_component, trace_functionality,
        # find_entry_points, suggest_next_steps, optimize_search
        assert self.mock_app.prompt.call_count >= 6


class TestPromptContext:
    """Test cases for PromptContext and related models."""

    def test_prompt_context_creation(self):
        """Test PromptContext creation and default values."""
        context = PromptContext()

        assert context.user_role == UserRole.DEVELOPER
        assert context.skill_level == DifficultyLevel.INTERMEDIATE
        assert context.current_task == TaskType.EXPLORATION
        assert context.working_directory == "."
        assert isinstance(context.previous_prompts, list)
        assert isinstance(context.search_history, list)
        assert isinstance(context.discovered_components, list)

    def test_prompt_context_customization(self):
        """Test PromptContext with custom values."""
        context = PromptContext(
            user_role=UserRole.ARCHITECT,
            skill_level=DifficultyLevel.ADVANCED,
            current_task=TaskType.REFACTORING,
            working_directory="/custom/path",
        )

        assert context.user_role == UserRole.ARCHITECT
        assert context.skill_level == DifficultyLevel.ADVANCED
        assert context.current_task == TaskType.REFACTORING
        assert context.working_directory == "/custom/path"


# Integration tests


class TestMCPPromptsIntegration:
    """Integration tests for the complete MCP Prompts system."""

    def setup_method(self):
        """Setup integration test fixtures."""
        self.mock_app = Mock(spec=FastMCP)

        # Create a more realistic mock that captures registered prompts
        self.registered_prompts = {}

        def mock_prompt_decorator():
            def decorator(func):
                prompt_name = func.__name__
                self.registered_prompts[prompt_name] = func
                return func

            return decorator

        self.mock_app.prompt = Mock(side_effect=mock_prompt_decorator)

    @patch("mcp_prompts.IndexingService")
    @patch("mcp_prompts.ProjectAnalysisService")
    @patch("mcp_prompts.EmbeddingService")
    def test_full_system_integration(self, mock_embedding, mock_analysis, mock_indexing):
        """Test full system integration."""
        # Initialize the system
        from mcp_prompts import MCPPromptsSystem

        MCPPromptsSystem(self.mock_app)

        # Verify all expected prompts were registered
        expected_prompts = {
            "explore_project",
            "understand_component",
            "trace_functionality",
            "find_entry_points",
            "suggest_next_steps",
            "optimize_search",
        }

        assert expected_prompts.issubset(set(self.registered_prompts.keys()))

        # Test that we can call a registered prompt
        explore_prompt = self.registered_prompts["explore_project"]

        # Mock the analysis service to return test data
        mock_analysis_instance = mock_analysis.return_value
        mock_analysis_instance.analyze_repository.return_value = {
            "total_files": 100,
            "relevant_files": 50,
            "language_breakdown": {"python": 30, "javascript": 20},
            "size_analysis": {"total_size_mb": 10},
            "indexing_complexity": {"level": "medium"},
        }

        # Call the prompt
        with tempfile.TemporaryDirectory() as temp_dir:
            result = explore_prompt(directory=temp_dir)

            assert isinstance(result, list)
            assert len(result) > 0
            assert isinstance(result[0], base.Message)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
