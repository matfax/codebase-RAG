"""
Base Classes and Interfaces for MCP Prompts

This module defines the core abstractions and base classes for the MCP Prompts system,
providing a consistent interface for prompt implementation and execution.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from mcp.server.fastmcp.prompts import base

from .prompt_context import (
    DifficultyLevel,
    PromptContext,
    PromptResult,
    PromptType,
    TaskType,
    UserRole,
)


class PromptExecutionMode(Enum):
    """Execution modes for prompts."""

    SYNCHRONOUS = "synchronous"
    ASYNCHRONOUS = "asynchronous"
    STREAMING = "streaming"


class PromptCapability(Enum):
    """Capabilities that prompts can have."""

    CONTEXT_AWARE = "context_aware"
    WORKFLOW_CHAINABLE = "workflow_chainable"
    PERSONALIZED = "personalized"
    RECOVERABLE = "recoverable"
    CACHEABLE = "cacheable"
    SEARCHABLE = "searchable"
    ANALYZABLE = "analyzable"


@dataclass
class PromptMetadata:
    """
    Metadata describing a prompt's characteristics and capabilities.

    Used for prompt discovery, optimization, and intelligent chaining.
    """

    # Basic identification
    name: str
    prompt_type: PromptType
    version: str = "1.0.0"

    # Description and documentation
    title: str = ""
    description: str = ""
    long_description: str = ""
    examples: list[str] = field(default_factory=list)

    # Capabilities and features
    capabilities: list[PromptCapability] = field(default_factory=list)
    execution_mode: PromptExecutionMode = PromptExecutionMode.SYNCHRONOUS

    # Target audience and context
    target_roles: list[UserRole] = field(default_factory=list)
    suitable_tasks: list[TaskType] = field(default_factory=list)
    skill_levels: list[DifficultyLevel] = field(default_factory=list)

    # Performance characteristics
    estimated_duration_seconds: int | None = None
    complexity_score: float = 0.5  # 0.0 to 1.0
    resource_requirements: dict[str, Any] = field(default_factory=dict)

    # Integration and dependencies
    required_services: list[str] = field(default_factory=list)
    optional_services: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)

    # Workflow integration
    typical_predecessors: list[PromptType] = field(default_factory=list)
    typical_successors: list[PromptType] = field(default_factory=list)
    chainable_with: list[PromptType] = field(default_factory=list)

    # Quality and reliability
    success_rate: float = 1.0
    error_rate: float = 0.0
    user_satisfaction: float = 0.8

    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptParameter:
    """
    Definition of a prompt parameter with validation and documentation.
    """

    # Parameter identification
    name: str
    param_type: type
    description: str = ""

    # Validation rules
    required: bool = True
    default_value: Any = None
    min_value: int | float | None = None
    max_value: int | float | None = None
    valid_values: list[Any] | None = None
    pattern: str | None = None  # Regex pattern for string validation

    # Documentation
    examples: list[str] = field(default_factory=list)
    help_text: str = ""

    # UI hints
    input_type: str = "text"  # text, number, boolean, select, etc.
    placeholder: str = ""
    group: str = "general"  # For organizing parameters in UI

    # Advanced features
    sensitive: bool = False  # Contains sensitive information
    affects_caching: bool = True  # Changes affect result caching
    validation_function: Callable | None = None


class BasePrompt(ABC):
    """
    Abstract base class for all MCP prompts.

    Defines the core interface that all prompts must implement,
    providing consistency and enabling advanced features like
    workflow orchestration and intelligent chaining.
    """

    def __init__(self):
        self._metadata = self._create_metadata()
        self._parameters = self._define_parameters()
        self._capabilities = self._get_capabilities()

        # Runtime state
        self._execution_count = 0
        self._success_count = 0
        self._total_execution_time = 0.0
        self._last_execution: datetime | None = None

    @property
    def metadata(self) -> PromptMetadata:
        """Get prompt metadata."""
        return self._metadata

    @property
    def parameters(self) -> list[PromptParameter]:
        """Get prompt parameters definition."""
        return self._parameters

    @property
    def capabilities(self) -> list[PromptCapability]:
        """Get prompt capabilities."""
        return self._capabilities

    @abstractmethod
    def _create_metadata(self) -> PromptMetadata:
        """Create and return prompt metadata."""
        pass

    @abstractmethod
    def _define_parameters(self) -> list[PromptParameter]:
        """Define prompt parameters."""
        pass

    @abstractmethod
    def _get_capabilities(self) -> list[PromptCapability]:
        """Get prompt capabilities."""
        pass

    @abstractmethod
    def execute(self, context: PromptContext | None = None, **parameters) -> list[base.Message]:
        """
        Execute the prompt with given parameters and context.

        Args:
            context: Optional prompt context for personalization
            **parameters: Prompt-specific parameters

        Returns:
            List of FastMCP prompt messages
        """
        pass

    def validate_parameters(self, **parameters) -> dict[str, Any]:
        """
        Validate prompt parameters.

        Args:
            **parameters: Parameters to validate

        Returns:
            Dictionary with validation results and cleaned parameters
        """
        validation_results = {}
        cleaned_parameters = {}
        errors = []
        warnings = []

        # Check required parameters
        required_params = {p.name for p in self._parameters if p.required}
        provided_params = set(parameters.keys())
        missing_params = required_params - provided_params

        if missing_params:
            errors.extend([f"Missing required parameter: {param}" for param in missing_params])

        # Validate each provided parameter
        for param_def in self._parameters:
            param_name = param_def.name
            if param_name not in parameters:
                if param_def.default_value is not None:
                    cleaned_parameters[param_name] = param_def.default_value
                continue

            param_value = parameters[param_name]

            # Type validation
            if not isinstance(param_value, param_def.param_type):
                try:
                    # Attempt type conversion
                    cleaned_parameters[param_name] = param_def.param_type(param_value)
                except (ValueError, TypeError):
                    errors.append(f"Parameter {param_name} must be of type {param_def.param_type.__name__}")
                    continue
            else:
                cleaned_parameters[param_name] = param_value

            # Value validation
            if param_def.valid_values and cleaned_parameters[param_name] not in param_def.valid_values:
                errors.append(f"Parameter {param_name} must be one of: {param_def.valid_values}")
                continue

            # Range validation
            if param_def.min_value is not None and cleaned_parameters[param_name] < param_def.min_value:
                warnings.append(f"Parameter {param_name} below minimum value {param_def.min_value}")

            if param_def.max_value is not None and cleaned_parameters[param_name] > param_def.max_value:
                warnings.append(f"Parameter {param_name} above maximum value {param_def.max_value}")

            # Custom validation
            if param_def.validation_function:
                try:
                    validation_result = param_def.validation_function(cleaned_parameters[param_name])
                    if not validation_result:
                        errors.append(f"Parameter {param_name} failed custom validation")
                except Exception as e:
                    errors.append(f"Parameter {param_name} validation error: {e}")

        validation_results = {
            "valid": len(errors) == 0,
            "parameters": cleaned_parameters,
            "errors": errors,
            "warnings": warnings,
        }

        return validation_results

    def can_chain_with(self, other_prompt: "BasePrompt") -> bool:
        """
        Check if this prompt can be chained with another prompt.

        Args:
            other_prompt: The prompt to check chaining compatibility with

        Returns:
            True if prompts can be chained, False otherwise
        """
        # Check if other prompt is in chainable list
        if other_prompt.metadata.prompt_type in self.metadata.chainable_with:
            return True

        # Check if other prompt is a typical successor
        if other_prompt.metadata.prompt_type in self.metadata.typical_successors:
            return True

        # Check if this prompt is a typical predecessor of the other
        if self.metadata.prompt_type in other_prompt.metadata.typical_predecessors:
            return True

        return False

    def is_suitable_for_context(self, context: PromptContext) -> bool:
        """
        Check if this prompt is suitable for the given context.

        Args:
            context: The context to check suitability for

        Returns:
            True if prompt is suitable, False otherwise
        """
        # Check user role compatibility
        if self.metadata.target_roles and context.user_role not in self.metadata.target_roles:
            return False

        # Check task type compatibility
        if self.metadata.suitable_tasks and context.current_task not in self.metadata.suitable_tasks:
            return False

        # Check skill level compatibility
        if self.metadata.skill_levels and context.skill_level not in self.metadata.skill_levels:
            return False

        return True

    def get_personalized_parameters(self, context: PromptContext) -> dict[str, Any]:
        """
        Get personalized parameter suggestions based on context.

        Args:
            context: The context to personalize for

        Returns:
            Dictionary of suggested parameter values
        """
        suggestions = {}

        # Default personalization based on user role
        if context.user_role == UserRole.NEWCOMER:
            suggestions.update(
                {
                    "detail_level": "comprehensive",
                    "include_examples": True,
                    "include_explanations": True,
                }
            )
        elif context.user_role == UserRole.ARCHITECT:
            suggestions.update(
                {
                    "detail_level": "overview",
                    "focus_on_architecture": True,
                    "include_dependencies": True,
                }
            )

        # Personalization based on skill level
        if context.skill_level == DifficultyLevel.BEGINNER:
            suggestions.update({"provide_guidance": True, "explain_terminology": True})
        elif context.skill_level == DifficultyLevel.ADVANCED:
            suggestions.update({"detail_level": "technical", "include_advanced_options": True})

        return suggestions

    def update_execution_stats(self, execution_time: float, success: bool):
        """Update execution statistics."""
        self._execution_count += 1
        self._total_execution_time += execution_time
        if success:
            self._success_count += 1
        self._last_execution = datetime.now()

        # Update metadata with current statistics
        self.metadata.success_rate = self._success_count / self._execution_count
        self.metadata.error_rate = 1.0 - self.metadata.success_rate

    def get_performance_stats(self) -> dict[str, Any]:
        """Get performance statistics for this prompt."""
        return {
            "execution_count": self._execution_count,
            "success_count": self._success_count,
            "success_rate": self.metadata.success_rate,
            "error_rate": self.metadata.error_rate,
            "average_execution_time": (self._total_execution_time / self._execution_count if self._execution_count > 0 else 0.0),
            "total_execution_time": self._total_execution_time,
            "last_execution": (self._last_execution.isoformat() if self._last_execution else None),
        }


class ContextAwarePrompt(BasePrompt):
    """
    Base class for context-aware prompts.

    Provides additional functionality for prompts that can
    adapt their behavior based on context and history.
    """

    def __init__(self):
        super().__init__()
        self._context_history: list[PromptContext] = []
        self._adaptation_strategies: dict[str, Callable] = {}

    def adapt_to_context(self, context: PromptContext) -> dict[str, Any]:
        """
        Adapt prompt behavior based on context.

        Args:
            context: The context to adapt to

        Returns:
            Dictionary of adaptations made
        """
        adaptations = {}

        # Store context for learning
        self._context_history.append(context)
        self._trim_context_history()

        # Apply adaptation strategies
        for strategy_name, strategy_func in self._adaptation_strategies.items():
            try:
                adaptation = strategy_func(context, self._context_history)
                if adaptation:
                    adaptations[strategy_name] = adaptation
            except Exception:
                # Log error but continue with other strategies
                pass

        return adaptations

    def learn_from_feedback(self, context: PromptContext, result: PromptResult, feedback: dict[str, Any]):
        """
        Learn from user feedback to improve future executions.

        Args:
            context: The context when prompt was executed
            result: The result of the prompt execution
            feedback: User feedback about the result
        """
        # Implement learning logic in subclasses
        pass

    def _trim_context_history(self, max_history: int = 100):
        """Trim context history to reasonable size."""
        if len(self._context_history) > max_history:
            self._context_history = self._context_history[-max_history:]


class ChainablePrompt(BasePrompt):
    """
    Base class for prompts that can be chained in workflows.

    Provides functionality for intelligent prompt chaining
    and workflow integration.
    """

    def __init__(self):
        super().__init__()
        self._workflow_state: dict[str, Any] = {}
        self._chain_outputs: list[Any] = []

    def prepare_for_chain(self, previous_results: list[PromptResult]) -> dict[str, Any]:
        """
        Prepare this prompt for execution in a chain.

        Args:
            previous_results: Results from previous prompts in chain

        Returns:
            Dictionary of prepared parameters
        """
        prepared_params = {}

        # Extract relevant information from previous results
        for result in previous_results:
            if result.analysis_results:
                # Use analysis results to inform parameters
                if "focus_area" in result.analysis_results:
                    prepared_params["focus_area"] = result.analysis_results["focus_area"]

                if "discovered_components" in result.analysis_results:
                    prepared_params["components"] = result.analysis_results["discovered_components"]

        return prepared_params

    def generate_chain_recommendations(self, context: PromptContext) -> list[PromptType]:
        """
        Generate recommendations for next prompts in chain.

        Args:
            context: Current context

        Returns:
            List of recommended next prompt types
        """
        recommendations = []

        # Use typical successors as base recommendations
        recommendations.extend(self.metadata.typical_successors)

        # Add context-specific recommendations
        if context.current_task == TaskType.EXPLORATION:
            if self.metadata.prompt_type == PromptType.EXPLORE_PROJECT:
                recommendations.extend([PromptType.FIND_ENTRY_POINTS, PromptType.UNDERSTAND_COMPONENT])

        return recommendations[:3]  # Limit to top 3 recommendations


# Factory function for creating prompt instances


def create_prompt_instance(prompt_type: PromptType) -> BasePrompt | None:
    """
    Factory function to create prompt instances.

    Args:
        prompt_type: Type of prompt to create

    Returns:
        Prompt instance or None if type not found
    """
    # This would be implemented with actual prompt classes
    # For now, return None as placeholder
    return None


# Utility functions for working with prompt base classes


def get_compatible_prompts(context: PromptContext, available_prompts: list[BasePrompt]) -> list[BasePrompt]:
    """Get prompts compatible with given context."""
    return [prompt for prompt in available_prompts if prompt.is_suitable_for_context(context)]


def find_chainable_prompts(source_prompt: BasePrompt, available_prompts: list[BasePrompt]) -> list[BasePrompt]:
    """Find prompts that can be chained with the source prompt."""
    return [prompt for prompt in available_prompts if source_prompt.can_chain_with(prompt)]


def get_prompt_recommendations(
    context: PromptContext,
    previous_prompts: list[PromptType],
    available_prompts: list[BasePrompt],
) -> list[BasePrompt]:
    """Get intelligent prompt recommendations based on context and history."""
    compatible = get_compatible_prompts(context, available_prompts)

    # Filter out already executed prompts
    executed_types = set(previous_prompts)
    candidates = [p for p in compatible if p.metadata.prompt_type not in executed_types]

    # Sort by relevance (simplified scoring)
    def relevance_score(prompt: BasePrompt) -> float:
        score = 0.0

        # Role compatibility
        if context.user_role in prompt.metadata.target_roles:
            score += 1.0

        # Task compatibility
        if context.current_task in prompt.metadata.suitable_tasks:
            score += 1.0

        # Success rate
        score += prompt.metadata.success_rate * 0.5

        return score

    candidates.sort(key=relevance_score, reverse=True)
    return candidates[:5]  # Return top 5 recommendations
