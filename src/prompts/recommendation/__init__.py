"""Recommendation prompts for optimization and next steps.

This module contains prompts for suggesting improvements and next actions.
"""

from .optimize_search import OptimizeSearchPrompt
from .suggest_next_steps import SuggestNextStepsPrompt

__all__ = ["SuggestNextStepsPrompt", "OptimizeSearchPrompt"]
