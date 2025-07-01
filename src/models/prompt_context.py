"""
Prompt Context Data Models

This module defines data structures for managing prompt execution context,
results, and state across workflow operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum


class PromptType(Enum):
    """Types of prompts supported by the system."""
    EXPLORE_PROJECT = "explore_project"
    UNDERSTAND_COMPONENT = "understand_component"
    TRACE_FUNCTIONALITY = "trace_functionality"
    FIND_ENTRY_POINTS = "find_entry_points"
    SUGGEST_NEXT_STEPS = "suggest_next_steps"
    OPTIMIZE_SEARCH = "optimize_search"


class UserRole(Enum):
    """User roles for personalized recommendations."""
    DEVELOPER = "developer"
    ARCHITECT = "architect"
    REVIEWER = "reviewer"
    NEWCOMER = "newcomer"
    DEBUGGER = "debugger"


class TaskType(Enum):
    """Types of tasks users are performing."""
    EXPLORATION = "exploration"
    DEVELOPMENT = "development"
    REFACTORING = "refactoring"
    DEBUGGING = "debugging"
    REVIEW = "review"


class DifficultyLevel(Enum):
    """Difficulty levels for recommendations."""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


@dataclass
class PromptContext:
    """
    Context information for prompt execution.
    
    Maintains state and context across prompt executions to enable
    intelligent workflow orchestration and personalized recommendations.
    """
    
    # Basic context
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Project context
    project_name: Optional[str] = None
    project_root: Optional[str] = None
    working_directory: str = "."
    
    # User context
    user_role: UserRole = UserRole.DEVELOPER
    skill_level: DifficultyLevel = DifficultyLevel.INTERMEDIATE
    current_task: TaskType = TaskType.EXPLORATION
    
    # Workflow state
    previous_prompts: List[str] = field(default_factory=list)
    search_history: List[str] = field(default_factory=list)
    discovered_components: List[str] = field(default_factory=list)
    explored_areas: List[str] = field(default_factory=list)
    
    # Preferences and configuration
    preferred_detail_level: str = "overview"
    include_examples: bool = True
    include_dependencies: bool = True
    max_recommendations: int = 5
    
    # Session metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PromptResult:
    """
    Result of prompt execution with analysis and recommendations.
    
    Contains the generated prompt content plus metadata about
    execution, analysis results, and suggested follow-up actions.
    """
    
    # Core result
    prompt_type: PromptType
    generated_prompt: str
    execution_time_ms: float
    success: bool = True
    
    # Context tracking
    context_used: Optional[PromptContext] = None
    parameters_used: Dict[str, Any] = field(default_factory=dict)
    
    # Analysis results (populated by services)
    analysis_results: Dict[str, Any] = field(default_factory=dict)
    search_results: List[Dict[str, Any]] = field(default_factory=list)
    component_matches: List[str] = field(default_factory=list)
    
    # Recommendations
    suggested_next_prompts: List[str] = field(default_factory=list)
    recommended_searches: List[str] = field(default_factory=list)
    learning_path: List[str] = field(default_factory=list)
    
    # Error handling
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowState:
    """
    State management for prompt workflow orchestration.
    
    Tracks the current state of a user's exploration workflow
    and enables intelligent chaining of prompts.
    """
    
    # Workflow identification
    workflow_id: str
    session_id: str
    user_id: Optional[str] = None
    
    # Current state
    current_phase: str = "initial"
    active_prompts: List[PromptType] = field(default_factory=list)
    completed_prompts: List[PromptType] = field(default_factory=list)
    
    # Progress tracking
    exploration_coverage: Dict[str, float] = field(default_factory=dict)
    component_understanding: Dict[str, float] = field(default_factory=dict)
    task_progress: float = 0.0
    
    # Context accumulation
    discovered_insights: List[str] = field(default_factory=list)
    key_findings: Dict[str, str] = field(default_factory=dict)
    potential_blockers: List[str] = field(default_factory=list)
    
    # Workflow metadata
    started_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    estimated_completion: Optional[datetime] = None
    
    # Persistence data
    state_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationCriteria:
    """
    Criteria for generating personalized recommendations.
    
    Defines the parameters used to customize recommendations
    based on user context, progress, and preferences.
    """
    
    # User factors
    user_role: UserRole
    skill_level: DifficultyLevel
    task_type: TaskType
    
    # Context factors
    current_understanding: float = 0.0  # 0-1 scale
    time_available: Optional[int] = None  # minutes
    complexity_preference: str = "adaptive"  # adaptive, simple, detailed
    
    # Content preferences
    include_code_examples: bool = True
    include_explanations: bool = True
    include_warnings: bool = True
    prefer_interactive_approach: bool = True
    
    # Scope preferences
    focus_areas: List[str] = field(default_factory=list)
    exclude_areas: List[str] = field(default_factory=list)
    max_depth: int = 3
    
    # Learning preferences
    learning_style: str = "hands_on"  # hands_on, conceptual, reference
    pace_preference: str = "moderate"  # slow, moderate, fast
    challenge_level: str = "appropriate"  # easy, appropriate, challenging


@dataclass
class ContextualRecommendation:
    """
    A contextual recommendation with rationale and metadata.
    
    Represents a single recommendation with explanation,
    priority, and context-specific rationale.
    """
    
    # Core recommendation
    action: str
    description: str
    rationale: str
    
    # Prioritization
    priority: int = 1  # 1=highest, 5=lowest
    confidence: float = 1.0  # 0-1 scale
    effort_estimate: str = "medium"  # low, medium, high
    
    # Context
    context_factors: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    related_prompts: List[PromptType] = field(default_factory=list)
    
    # Execution guidance
    suggested_parameters: Dict[str, Any] = field(default_factory=dict)
    expected_outcome: str = ""
    success_criteria: List[str] = field(default_factory=list)
    
    # Learning value
    learning_value: str = "medium"  # low, medium, high
    skill_building: List[str] = field(default_factory=list)
    knowledge_gained: List[str] = field(default_factory=list)


# Utility functions for working with context models

def create_default_context(
    working_directory: str = ".",
    user_role: UserRole = UserRole.DEVELOPER,
    task_type: TaskType = TaskType.EXPLORATION
) -> PromptContext:
    """Create a default prompt context with basic settings."""
    return PromptContext(
        working_directory=working_directory,
        user_role=user_role,
        current_task=task_type,
        timestamp=datetime.now()
    )


def update_context_from_result(context: PromptContext, result: PromptResult) -> PromptContext:
    """Update context based on prompt execution result."""
    # Add to prompt history
    context.previous_prompts.append(result.prompt_type.value)
    
    # Update discovered components
    if result.component_matches:
        context.discovered_components.extend(result.component_matches)
        # Remove duplicates while preserving order
        context.discovered_components = list(dict.fromkeys(context.discovered_components))
    
    # Update exploration areas
    if result.analysis_results.get("focus_area"):
        area = result.analysis_results["focus_area"]
        if area not in context.explored_areas:
            context.explored_areas.append(area)
    
    # Update search history if search terms were used
    if result.metadata.get("search_terms"):
        search_terms = result.metadata["search_terms"]
        if isinstance(search_terms, list):
            context.search_history.extend(search_terms)
        else:
            context.search_history.append(str(search_terms))
    
    return context


def create_recommendation_criteria_from_context(context: PromptContext) -> RecommendationCriteria:
    """Create recommendation criteria from prompt context."""
    return RecommendationCriteria(
        user_role=context.user_role,
        skill_level=context.skill_level,
        task_type=context.current_task,
        include_code_examples=context.include_examples,
        include_explanations=True,
        focus_areas=context.explored_areas,
        max_depth=3 if context.skill_level == DifficultyLevel.BEGINNER else 5
    )