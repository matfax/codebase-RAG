"""
Workflow Orchestrator for MCP Prompts

This module manages prompt chaining, workflow state, and context persistence
for complex multi-step codebase exploration and analysis workflows.
"""

import json
import logging
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from models.prompt_context import (
    DifficultyLevel,
    PromptContext,
    PromptResult,
    PromptType,
    TaskType,
    UserRole,
    WorkflowState,
)
from models.recommendation import (
    ConfidenceLevel,
    PriorityLevel,
    RecommendationAction,
    RecommendationSet,
    RecommendationType,
)

logger = logging.getLogger(__name__)


class WorkflowOrchestrator:
    """
    Orchestrates complex prompt workflows and manages state persistence.

    Handles prompt chaining, context management, and workflow state tracking
    to enable intelligent multi-step codebase exploration workflows.
    """

    def __init__(self, storage_dir: str | None = None):
        self.logger = logger
        self.storage_dir = Path(storage_dir) if storage_dir else Path.cwd() / ".mcp_workflows"
        self.storage_dir.mkdir(exist_ok=True)

        # Active workflows and contexts
        self.active_workflows: dict[str, WorkflowState] = {}
        self.active_contexts: dict[str, PromptContext] = {}

        # Workflow templates and patterns
        self.workflow_templates = self._initialize_workflow_templates()
        self.prompt_chains = self._initialize_prompt_chains()

        # Performance tracking
        self.execution_metrics: dict[str, list[float]] = defaultdict(list)
        self.success_rates: dict[str, list[bool]] = defaultdict(list)

        self.logger.info(f"Workflow orchestrator initialized with storage: {self.storage_dir}")

    def create_workflow(
        self,
        workflow_type: str,
        user_id: str | None = None,
        session_id: str | None = None,
        initial_context: PromptContext | None = None,
    ) -> WorkflowState:
        """
        Create a new workflow instance.

        Args:
            workflow_type: Type of workflow to create
            user_id: Optional user identifier
            session_id: Optional session identifier
            initial_context: Optional initial context

        Returns:
            WorkflowState: Created workflow state
        """
        workflow_id = f"{workflow_type}_{int(time.time() * 1000)}"

        if session_id is None:
            session_id = f"session_{int(time.time() * 1000)}"

        workflow = WorkflowState(
            workflow_id=workflow_id,
            session_id=session_id,
            user_id=user_id,
            current_phase=self._get_initial_phase(workflow_type),
        )

        # Initialize from template if available
        if workflow_type in self.workflow_templates:
            template = self.workflow_templates[workflow_type]
            workflow.exploration_coverage = template.get("initial_coverage", {})
            workflow.state_data.update(template.get("initial_state", {}))

        # Create or update context
        if initial_context:
            context = initial_context
        else:
            context = PromptContext(session_id=session_id)

        # Store active workflow and context
        self.active_workflows[workflow_id] = workflow
        self.active_contexts[workflow_id] = context

        # Persist to storage
        self._save_workflow(workflow)
        self._save_context(workflow_id, context)

        self.logger.info(f"Created workflow {workflow_id} of type {workflow_type}")
        return workflow

    def execute_prompt_with_workflow(
        self,
        workflow_id: str,
        prompt_type: PromptType,
        prompt_function: Callable,
        **kwargs,
    ) -> PromptResult:
        """
        Execute a prompt within a workflow context.

        Args:
            workflow_id: ID of the workflow
            prompt_type: Type of prompt being executed
            prompt_function: The prompt function to execute
            **kwargs: Parameters for the prompt function

        Returns:
            PromptResult: Result of prompt execution with workflow context
        """
        start_time = time.time()

        # Get workflow and context
        workflow = self.active_workflows.get(workflow_id)
        context = self.active_contexts.get(workflow_id)

        if not workflow or not context:
            raise ValueError(f"Workflow {workflow_id} not found or not active")

        try:
            # Pre-execution workflow updates
            self._update_workflow_pre_execution(workflow, prompt_type)

            # Execute the prompt function
            prompt_messages = prompt_function(**kwargs)

            # Create result object
            execution_time = (time.time() - start_time) * 1000
            result = PromptResult(
                prompt_type=prompt_type,
                generated_prompt=self._extract_prompt_text(prompt_messages),
                execution_time_ms=execution_time,
                context_used=context,
                parameters_used=kwargs,
            )

            # Post-execution workflow updates
            self._update_workflow_post_execution(workflow, prompt_type, result)
            self._update_context_from_execution(context, result)

            # Generate recommendations
            recommendations = self._generate_workflow_recommendations(workflow, context, result)
            result.suggested_next_prompts = [rec.action for rec in recommendations.actions[:3]]

            # Update metrics
            self.execution_metrics[prompt_type.value].append(execution_time)
            self.success_rates[prompt_type.value].append(result.success)

            # Persist updates
            self._save_workflow(workflow)
            self._save_context(workflow_id, context)

            self.logger.info(f"Executed prompt {prompt_type.value} in workflow {workflow_id}, " f"took {execution_time:.1f}ms")

            return result

        except Exception as e:
            execution_time = (time.time() - start_time) * 1000
            self.logger.error(f"Error executing prompt {prompt_type.value} in workflow {workflow_id}: {e}")

            result = PromptResult(
                prompt_type=prompt_type,
                generated_prompt="",
                execution_time_ms=execution_time,
                success=False,
                context_used=context,
                parameters_used=kwargs,
            )
            result.errors.append(str(e))

            self.success_rates[prompt_type.value].append(False)
            return result

    def get_next_recommended_prompts(self, workflow_id: str, limit: int = 3) -> list[RecommendationAction]:
        """
        Get recommended next prompts for a workflow.

        Args:
            workflow_id: ID of the workflow
            limit: Maximum number of recommendations

        Returns:
            List of recommended actions
        """
        workflow = self.active_workflows.get(workflow_id)
        context = self.active_contexts.get(workflow_id)

        if not workflow or not context:
            return []

        recommendations = self._generate_workflow_recommendations(workflow, context)
        return recommendations.actions[:limit]

    def get_workflow_progress(self, workflow_id: str) -> dict[str, Any]:
        """
        Get comprehensive workflow progress information.

        Args:
            workflow_id: ID of the workflow

        Returns:
            Dictionary with progress information
        """
        workflow = self.active_workflows.get(workflow_id)
        context = self.active_contexts.get(workflow_id)

        if not workflow or not context:
            return {"error": f"Workflow {workflow_id} not found"}

        # Calculate progress metrics
        total_prompts = len(self.prompt_chains.get(workflow.current_phase, []))
        completed_prompts = len(workflow.completed_prompts)
        progress_percentage = (completed_prompts / total_prompts * 100) if total_prompts > 0 else 0

        # Get coverage metrics
        exploration_areas = list(workflow.exploration_coverage.keys())
        avg_coverage = sum(workflow.exploration_coverage.values()) / len(exploration_areas) if exploration_areas else 0

        # Calculate time metrics
        elapsed_time = datetime.now() - workflow.started_at
        estimated_remaining = self._estimate_remaining_time(workflow, context)

        return {
            "workflow_id": workflow_id,
            "current_phase": workflow.current_phase,
            "progress_percentage": round(progress_percentage, 1),
            "completed_prompts": completed_prompts,
            "total_prompts": total_prompts,
            "exploration_coverage": {
                "areas": exploration_areas,
                "average_coverage": round(avg_coverage * 100, 1),
            },
            "timing": {
                "elapsed_minutes": round(elapsed_time.total_seconds() / 60, 1),
                "estimated_remaining_minutes": estimated_remaining,
                "started_at": workflow.started_at.isoformat(),
            },
            "insights": {
                "discovered_insights": len(workflow.discovered_insights),
                "key_findings": len(workflow.key_findings),
                "potential_blockers": len(workflow.potential_blockers),
            },
            "context": {
                "user_role": context.user_role.value,
                "task_type": context.current_task.value,
                "skill_level": context.skill_level.value,
                "discovered_components": len(context.discovered_components),
            },
        }

    def save_workflow_checkpoint(self, workflow_id: str) -> bool:
        """
        Save a checkpoint of the current workflow state.

        Args:
            workflow_id: ID of the workflow

        Returns:
            True if successful, False otherwise
        """
        try:
            workflow = self.active_workflows.get(workflow_id)
            context = self.active_contexts.get(workflow_id)

            if not workflow or not context:
                return False

            checkpoint_data = {
                "workflow": asdict(workflow),
                "context": asdict(context),
                "timestamp": datetime.now().isoformat(),
                "metrics": {
                    "execution_times": dict(self.execution_metrics),
                    "success_rates": dict(self.success_rates),
                },
            }

            checkpoint_file = self.storage_dir / f"checkpoint_{workflow_id}.json"
            with open(checkpoint_file, "w") as f:
                json.dump(checkpoint_data, f, indent=2, default=str)

            self.logger.info(f"Saved checkpoint for workflow {workflow_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save checkpoint for workflow {workflow_id}: {e}")
            return False

    def load_workflow_checkpoint(self, workflow_id: str) -> bool:
        """
        Load a workflow from a checkpoint.

        Args:
            workflow_id: ID of the workflow

        Returns:
            True if successful, False otherwise
        """
        try:
            checkpoint_file = self.storage_dir / f"checkpoint_{workflow_id}.json"

            if not checkpoint_file.exists():
                self.logger.warning(f"Checkpoint file not found for workflow {workflow_id}")
                return False

            with open(checkpoint_file) as f:
                checkpoint_data = json.load(f)

            # Restore workflow state
            workflow_data = checkpoint_data["workflow"]
            workflow = WorkflowState(**workflow_data)
            self.active_workflows[workflow_id] = workflow

            # Restore context
            context_data = checkpoint_data["context"]
            # Convert enum strings back to enums
            if "user_role" in context_data:
                context_data["user_role"] = UserRole(context_data["user_role"])
            if "skill_level" in context_data:
                context_data["skill_level"] = DifficultyLevel(context_data["skill_level"])
            if "current_task" in context_data:
                context_data["current_task"] = TaskType(context_data["current_task"])

            context = PromptContext(**context_data)
            self.active_contexts[workflow_id] = context

            # Restore metrics if available
            if "metrics" in checkpoint_data:
                metrics = checkpoint_data["metrics"]
                if "execution_times" in metrics:
                    self.execution_metrics.update(metrics["execution_times"])
                if "success_rates" in metrics:
                    self.success_rates.update(metrics["success_rates"])

            self.logger.info(f"Loaded checkpoint for workflow {workflow_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to load checkpoint for workflow {workflow_id}: {e}")
            return False

    def _initialize_workflow_templates(self) -> dict[str, dict[str, Any]]:
        """Initialize workflow templates."""
        return {
            "project_exploration": {
                "initial_phase": "project_overview",
                "initial_coverage": {
                    "architecture": 0.0,
                    "components": 0.0,
                    "entry_points": 0.0,
                    "dependencies": 0.0,
                },
                "initial_state": {
                    "exploration_goals": [
                        "understand_architecture",
                        "identify_components",
                        "map_dependencies",
                    ]
                },
            },
            "component_deep_dive": {
                "initial_phase": "component_identification",
                "initial_coverage": {
                    "interface_analysis": 0.0,
                    "implementation_details": 0.0,
                    "usage_patterns": 0.0,
                    "dependencies": 0.0,
                },
                "initial_state": {
                    "analysis_goals": [
                        "understand_purpose",
                        "analyze_interface",
                        "trace_usage",
                    ]
                },
            },
            "functionality_tracing": {
                "initial_phase": "entry_point_discovery",
                "initial_coverage": {
                    "call_chain": 0.0,
                    "data_flow": 0.0,
                    "configuration": 0.0,
                    "error_handling": 0.0,
                },
                "initial_state": {
                    "tracing_goals": [
                        "map_call_chain",
                        "understand_data_flow",
                        "identify_config",
                    ]
                },
            },
        }

    def _initialize_prompt_chains(self) -> dict[str, list[PromptType]]:
        """Initialize prompt chain templates."""
        return {
            "project_overview": [
                PromptType.EXPLORE_PROJECT,
                PromptType.FIND_ENTRY_POINTS,
                PromptType.SUGGEST_NEXT_STEPS,
            ],
            "component_identification": [
                PromptType.UNDERSTAND_COMPONENT,
                PromptType.TRACE_FUNCTIONALITY,
                PromptType.SUGGEST_NEXT_STEPS,
            ],
            "entry_point_discovery": [
                PromptType.FIND_ENTRY_POINTS,
                PromptType.TRACE_FUNCTIONALITY,
                PromptType.UNDERSTAND_COMPONENT,
            ],
        }

    def _get_initial_phase(self, workflow_type: str) -> str:
        """Get initial phase for workflow type."""
        template = self.workflow_templates.get(workflow_type)
        return template.get("initial_phase", "initial") if template else "initial"

    def _update_workflow_pre_execution(self, workflow: WorkflowState, prompt_type: PromptType):
        """Update workflow state before prompt execution."""
        workflow.last_activity = datetime.now()

        if prompt_type not in workflow.active_prompts:
            workflow.active_prompts.append(prompt_type)

    def _update_workflow_post_execution(self, workflow: WorkflowState, prompt_type: PromptType, result: PromptResult):
        """Update workflow state after prompt execution."""
        # Move from active to completed
        if prompt_type in workflow.active_prompts:
            workflow.active_prompts.remove(prompt_type)

        if prompt_type not in workflow.completed_prompts:
            workflow.completed_prompts.append(prompt_type)

        # Update progress tracking
        workflow.task_progress = len(workflow.completed_prompts) / max(len(self.prompt_chains.get(workflow.current_phase, [])), 1)

        # Extract insights from result
        if result.analysis_results:
            insights = result.analysis_results.get("insights", [])
            workflow.discovered_insights.extend(insights)

        # Check for phase transitions
        self._check_phase_transition(workflow)

    def _update_context_from_execution(self, context: PromptContext, result: PromptResult):
        """Update context based on execution result."""
        # Add to prompt history
        context.previous_prompts.append(result.prompt_type.value)

        # Update search history
        if result.parameters_used.get("search_terms"):
            context.search_history.extend(result.parameters_used["search_terms"])

        # Update discovered components
        if result.component_matches:
            context.discovered_components.extend(result.component_matches)
            # Remove duplicates
            context.discovered_components = list(dict.fromkeys(context.discovered_components))

    def _generate_workflow_recommendations(
        self,
        workflow: WorkflowState,
        context: PromptContext,
        result: PromptResult | None = None,
    ) -> RecommendationSet:
        """Generate intelligent recommendations based on workflow state."""
        recommendations = RecommendationSet(
            set_id=f"rec_{workflow.workflow_id}_{int(time.time())}",
            title="Next Steps Recommendations",
            description="Intelligent recommendations based on current workflow progress",
            target_user_role=context.user_role,
            task_context=context.current_task,
            skill_level=context.skill_level,
        )

        # Get potential next prompts based on chain
        current_chain = self.prompt_chains.get(workflow.current_phase, [])
        completed = set(workflow.completed_prompts)
        remaining = [p for p in current_chain if p not in completed]

        # Create recommendations for remaining prompts
        for i, prompt_type in enumerate(remaining[:3]):
            priority = PriorityLevel.HIGH if i == 0 else PriorityLevel.MEDIUM

            action = RecommendationAction(
                action_id=f"action_{prompt_type.value}_{int(time.time())}",
                title=f"Execute {prompt_type.value.replace('_', ' ').title()}",
                description=self._get_prompt_description(prompt_type),
                action_type=RecommendationType.NEXT_ACTION,
                priority=priority,
                confidence=ConfidenceLevel.HIGH,
                related_prompts=[prompt_type],
            )

            recommendations.actions.append(action)
            if i == 0:
                recommendations.immediate_actions.append(action.action_id)
            else:
                recommendations.long_term_actions.append(action.action_id)

        # Add search optimization if search history exists
        if context.search_history:
            search_action = RecommendationAction(
                action_id=f"search_opt_{int(time.time())}",
                title="Optimize Search Strategy",
                description="Analyze and improve your search approach based on previous queries",
                action_type=RecommendationType.SEARCH_STRATEGY,
                priority=PriorityLevel.MEDIUM,
                confidence=ConfidenceLevel.MEDIUM,
                related_prompts=[PromptType.OPTIMIZE_SEARCH],
            )
            recommendations.actions.append(search_action)

        return recommendations

    def _get_prompt_description(self, prompt_type: PromptType) -> str:
        """Get user-friendly description for prompt type."""
        descriptions = {
            PromptType.EXPLORE_PROJECT: "Get a comprehensive overview of the project architecture and structure",
            PromptType.UNDERSTAND_COMPONENT: "Deep dive into a specific component to understand its purpose and interfaces",
            PromptType.TRACE_FUNCTIONALITY: "Follow the complete implementation path of specific functionality",
            PromptType.FIND_ENTRY_POINTS: "Discover all main entry points and application starting points",
            PromptType.SUGGEST_NEXT_STEPS: "Get intelligent recommendations for your next actions",
            PromptType.OPTIMIZE_SEARCH: "Improve your search strategy and find better approaches",
        }
        return descriptions.get(prompt_type, f"Execute {prompt_type.value}")

    def _check_phase_transition(self, workflow: WorkflowState):
        """Check if workflow should transition to next phase."""
        current_chain = self.prompt_chains.get(workflow.current_phase, [])
        completed = set(workflow.completed_prompts)

        # Simple transition logic: move to next phase when 80% complete
        completion_rate = len([p for p in current_chain if p in completed]) / len(current_chain) if current_chain else 1.0

        if completion_rate >= 0.8:
            # Determine next phase based on workflow type
            next_phase = self._get_next_phase(workflow.current_phase)
            if next_phase and next_phase != workflow.current_phase:
                self.logger.info(f"Transitioning workflow {workflow.workflow_id} from {workflow.current_phase} to {next_phase}")
                workflow.current_phase = next_phase

    def _get_next_phase(self, current_phase: str) -> str | None:
        """Get next phase in workflow."""
        phase_transitions = {
            "project_overview": "component_identification",
            "component_identification": "functionality_tracing",
            "entry_point_discovery": "component_identification",
        }
        return phase_transitions.get(current_phase)

    def _estimate_remaining_time(self, workflow: WorkflowState, context: PromptContext) -> int | None:
        """Estimate remaining time for workflow completion."""
        current_chain = self.prompt_chains.get(workflow.current_phase, [])
        completed = set(workflow.completed_prompts)
        remaining_prompts = len([p for p in current_chain if p not in completed])

        # Estimate based on historical execution times
        avg_time_per_prompt = 5  # Default 5 minutes

        if self.execution_metrics:
            total_time = sum(sum(times) for times in self.execution_metrics.values())
            total_executions = sum(len(times) for times in self.execution_metrics.values())
            if total_executions > 0:
                avg_time_per_prompt = (total_time / total_executions) / 60000  # Convert to minutes

        return int(remaining_prompts * avg_time_per_prompt)

    def _extract_prompt_text(self, prompt_messages) -> str:
        """Extract text from prompt messages."""
        if isinstance(prompt_messages, list):
            texts = []
            for message in prompt_messages:
                if hasattr(message, "content") and isinstance(message.content, list):
                    for content in message.content:
                        if hasattr(content, "text"):
                            texts.append(content.text)
            return "\n".join(texts)
        return str(prompt_messages)

    def _save_workflow(self, workflow: WorkflowState):
        """Save workflow state to storage."""
        try:
            workflow_file = self.storage_dir / f"workflow_{workflow.workflow_id}.json"
            with open(workflow_file, "w") as f:
                json.dump(asdict(workflow), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save workflow {workflow.workflow_id}: {e}")

    def _save_context(self, workflow_id: str, context: PromptContext):
        """Save context to storage."""
        try:
            context_file = self.storage_dir / f"context_{workflow_id}.json"
            with open(context_file, "w") as f:
                json.dump(asdict(context), f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Failed to save context for workflow {workflow_id}: {e}")


# Global orchestrator instance
workflow_orchestrator = WorkflowOrchestrator()
