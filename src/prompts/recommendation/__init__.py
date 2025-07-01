"""Recommendation prompts for optimization and next steps.

This module contains prompts for suggesting improvements and next actions.
"""

from .suggest_next_steps import SuggestNextStepsPrompt
from .optimize_search import OptimizeSearchPrompt

__all__ = [
    "SuggestNextStepsPrompt",
    "OptimizeSearchPrompt"
]