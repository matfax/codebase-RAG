"""
Recommendation Data Models

This module defines data structures for the intelligent recommendation system
that provides contextual suggestions and next-step guidance.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
from enum import Enum

from .prompt_context import UserRole, TaskType, DifficultyLevel, PromptType


class RecommendationType(Enum):
    """Types of recommendations the system can provide."""
    NEXT_ACTION = "next_action"
    SEARCH_STRATEGY = "search_strategy"
    LEARNING_PATH = "learning_path"
    EXPLORATION_TARGET = "exploration_target"
    OPTIMIZATION_SUGGESTION = "optimization_suggestion"
    WORKFLOW_GUIDANCE = "workflow_guidance"


class PriorityLevel(Enum):
    """Priority levels for recommendations."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4
    OPTIONAL = 5


class ConfidenceLevel(Enum):
    """Confidence levels for recommendation quality."""
    VERY_HIGH = 0.9
    HIGH = 0.8
    MEDIUM = 0.6
    LOW = 0.4
    VERY_LOW = 0.2


@dataclass
class RecommendationAction:
    """
    A specific actionable recommendation.
    
    Represents a single recommended action with all necessary
    context and guidance for execution.
    """
    
    # Core action details
    action_id: str
    title: str
    description: str
    action_type: RecommendationType
    
    # Execution details
    command: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_duration: Optional[int] = None  # minutes
    
    # Prioritization
    priority: PriorityLevel = PriorityLevel.MEDIUM
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    urgency: bool = False
    
    # Context and rationale
    rationale: str = ""
    benefits: List[str] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    
    # Relationships
    related_prompts: List[PromptType] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)  # action_ids
    enables: List[str] = field(default_factory=list)  # action_ids
    
    # Success criteria
    success_indicators: List[str] = field(default_factory=list)
    completion_criteria: List[str] = field(default_factory=list)
    
    # Learning value
    skill_development: List[str] = field(default_factory=list)
    knowledge_gained: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    context_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecommendationSet:
    """
    A curated set of recommendations for a specific context.
    
    Groups related recommendations and provides ordering,
    prioritization, and workflow guidance.
    """
    
    # Set identification
    set_id: str
    title: str
    description: str
    
    # Recommendations
    actions: List[RecommendationAction] = field(default_factory=list)
    immediate_actions: List[str] = field(default_factory=list)  # action_ids
    long_term_actions: List[str] = field(default_factory=list)  # action_ids
    
    # Context
    target_user_role: UserRole
    task_context: TaskType
    skill_level: DifficultyLevel
    
    # Workflow guidance
    suggested_order: List[str] = field(default_factory=list)  # action_ids
    parallel_actions: List[List[str]] = field(default_factory=list)  # groups of action_ids
    decision_points: Dict[str, List[str]] = field(default_factory=dict)  # condition -> action_ids
    
    # Quality metrics
    overall_confidence: float = 0.8
    completeness_score: float = 0.8
    relevance_score: float = 0.8
    
    # Timing and effort
    estimated_total_time: Optional[int] = None  # minutes
    complexity_level: str = "medium"  # low, medium, high
    
    # Personalization
    personalization_factors: List[str] = field(default_factory=list)
    customization_options: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    expiry_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchOptimizationSuggestion:
    """
    Suggestions for optimizing search strategies.
    
    Provides specific guidance for improving search effectiveness
    based on analysis of previous search patterns and results.
    """
    
    # Analysis results
    current_strategy: str
    identified_gaps: List[str] = field(default_factory=list)
    ineffective_patterns: List[str] = field(default_factory=list)
    
    # Optimization suggestions
    improved_terms: List[str] = field(default_factory=list)
    alternative_approaches: List[str] = field(default_factory=list)
    search_techniques: List[str] = field(default_factory=list)
    
    # Specific recommendations
    exact_search_suggestions: List[str] = field(default_factory=list)
    wildcard_suggestions: List[str] = field(default_factory=list)
    contextual_suggestions: List[str] = field(default_factory=list)
    
    # Strategy guidance
    search_sequence: List[str] = field(default_factory=list)
    fallback_strategies: List[str] = field(default_factory=list)
    advanced_techniques: List[str] = field(default_factory=list)
    
    # Expected outcomes
    expected_improvements: List[str] = field(default_factory=list)
    success_indicators: List[str] = field(default_factory=list)
    
    # Metadata
    confidence_score: float = 0.8
    complexity_level: str = "medium"
    estimated_time_savings: Optional[int] = None  # minutes


@dataclass
class LearningPathRecommendation:
    """
    Structured learning path for understanding the codebase.
    
    Provides a step-by-step progression designed to build
    understanding systematically and efficiently.
    """
    
    # Path identification
    path_id: str
    title: str
    description: str
    
    # Learning structure
    phases: List[Dict[str, Any]] = field(default_factory=list)
    milestones: List[str] = field(default_factory=list)
    checkpoints: Dict[str, List[str]] = field(default_factory=dict)  # phase -> criteria
    
    # Content organization
    core_concepts: List[str] = field(default_factory=list)
    key_files: List[str] = field(default_factory=list)
    important_components: List[str] = field(default_factory=list)
    
    # Progression details
    estimated_duration: Optional[int] = None  # hours
    difficulty_curve: List[str] = field(default_factory=list)  # per phase
    prerequisite_knowledge: List[str] = field(default_factory=list)
    
    # Personalization
    adapted_for_role: UserRole
    skill_level_target: DifficultyLevel
    customization_notes: List[str] = field(default_factory=list)
    
    # Resources and guidance
    recommended_tools: List[str] = field(default_factory=list)
    helpful_prompts: List[PromptType] = field(default_factory=list)
    external_resources: List[str] = field(default_factory=list)
    
    # Quality metrics
    effectiveness_score: float = 0.8
    completeness: float = 0.8
    user_satisfaction_predicted: float = 0.8


@dataclass
class ContextualInsight:
    """
    An insight derived from context analysis.
    
    Represents understanding gained about the user's current
    situation and potential next steps.
    """
    
    # Insight details
    insight_type: str
    title: str
    description: str
    
    # Context factors
    derived_from: List[str] = field(default_factory=list)
    confidence: float = 0.8
    relevance: float = 0.8
    
    # Actionability
    actionable: bool = True
    suggested_actions: List[str] = field(default_factory=list)
    blocked_actions: List[str] = field(default_factory=list)
    
    # Impact assessment
    potential_impact: str = "medium"  # low, medium, high
    time_sensitivity: str = "normal"  # urgent, normal, flexible
    scope_affected: List[str] = field(default_factory=list)
    
    # Supporting evidence
    evidence: List[str] = field(default_factory=list)
    contradictory_evidence: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


# Factory functions for creating recommendations

def create_immediate_action(
    title: str,
    description: str,
    action_type: RecommendationType = RecommendationType.NEXT_ACTION,
    priority: PriorityLevel = PriorityLevel.HIGH,
    duration: Optional[int] = None
) -> RecommendationAction:
    """Create a high-priority immediate action recommendation."""
    return RecommendationAction(
        action_id=f"action_{datetime.now().timestamp()}",
        title=title,
        description=description,
        action_type=action_type,
        priority=priority,
        expected_duration=duration,
        urgency=True
    )


def create_search_optimization(
    current_strategy: str,
    improved_terms: List[str],
    alternative_approaches: List[str]
) -> SearchOptimizationSuggestion:
    """Create a search optimization suggestion."""
    return SearchOptimizationSuggestion(
        current_strategy=current_strategy,
        improved_terms=improved_terms,
        alternative_approaches=alternative_approaches,
        confidence_score=0.8
    )


def create_learning_path(
    title: str,
    description: str,
    phases: List[Dict[str, Any]],
    target_role: UserRole,
    skill_level: DifficultyLevel
) -> LearningPathRecommendation:
    """Create a structured learning path recommendation."""
    return LearningPathRecommendation(
        path_id=f"path_{datetime.now().timestamp()}",
        title=title,
        description=description,
        phases=phases,
        adapted_for_role=target_role,
        skill_level_target=skill_level
    )


def create_contextual_insight(
    insight_type: str,
    title: str,
    description: str,
    confidence: float = 0.8,
    actionable: bool = True
) -> ContextualInsight:
    """Create a contextual insight from analysis."""
    return ContextualInsight(
        insight_type=insight_type,
        title=title,
        description=description,
        confidence=confidence,
        actionable=actionable
    )


# Utility functions for working with recommendations

def prioritize_actions(actions: List[RecommendationAction]) -> List[RecommendationAction]:
    """Sort actions by priority, confidence, and urgency."""
    def priority_key(action: RecommendationAction) -> tuple:
        return (
            action.priority.value,  # Lower number = higher priority
            -action.confidence.value,  # Higher confidence first
            not action.urgency  # Urgent items first
        )
    
    return sorted(actions, key=priority_key)


def filter_by_context(
    actions: List[RecommendationAction],
    user_role: UserRole,
    task_type: TaskType,
    skill_level: DifficultyLevel
) -> List[RecommendationAction]:
    """Filter actions relevant to specific context."""
    filtered = []
    
    for action in actions:
        # Check if action is appropriate for the context
        if action.metadata.get("target_roles") and user_role not in action.metadata["target_roles"]:
            continue
        if action.metadata.get("task_types") and task_type not in action.metadata["task_types"]:
            continue
        if action.metadata.get("min_skill_level") and skill_level.value < action.metadata["min_skill_level"]:
            continue
            
        filtered.append(action)
    
    return filtered


def estimate_total_time(actions: List[RecommendationAction]) -> int:
    """Estimate total time for a set of actions."""
    total_minutes = 0
    for action in actions:
        if action.expected_duration:
            total_minutes += action.expected_duration
        else:
            # Default estimates based on action type
            if action.action_type == RecommendationType.SEARCH_STRATEGY:
                total_minutes += 5
            elif action.action_type == RecommendationType.EXPLORATION_TARGET:
                total_minutes += 15
            elif action.action_type == RecommendationType.LEARNING_PATH:
                total_minutes += 30
            else:
                total_minutes += 10
    
    return total_minutes