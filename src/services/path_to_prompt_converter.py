"""
Path-to-Prompt Conversion Service for Wave 2.0 Task 2.5 - LLM Context Generation.

This service converts relational paths into structured prompts optimized for LLM consumption.
It provides template generation, context structuring, and prompt optimization to maximize
the effectiveness of path information for code understanding tasks.
"""

import asyncio
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from ..models.relational_path import (
    AnyPath,
    DataFlowPath,
    DependencyPath,
    ExecutionPath,
    PathNode,
    PathType,
    RelationalPathCollection,
)


class PromptTemplate(Enum):
    """Available prompt templates for different use cases."""

    COMPREHENSIVE = "comprehensive"  # Detailed, full-context prompts
    CONCISE = "concise"  # Brief, focused prompts
    STRUCTURED = "structured"  # Highly structured with clear sections
    NARRATIVE = "narrative"  # Natural language narrative style
    TECHNICAL = "technical"  # Technical documentation style
    CONVERSATIONAL = "conversational"  # Natural conversation style


class ContextLevel(Enum):
    """Level of context detail to include."""

    MINIMAL = "minimal"  # Essential information only
    STANDARD = "standard"  # Standard level of detail
    DETAILED = "detailed"  # Comprehensive detail
    EXHAUSTIVE = "exhaustive"  # All available information


class PromptPurpose(Enum):
    """Purpose of the generated prompt."""

    ANALYSIS = "analysis"  # Code analysis tasks
    EXPLANATION = "explanation"  # Code explanation tasks
    DEBUGGING = "debugging"  # Debugging assistance
    DOCUMENTATION = "documentation"  # Documentation generation
    REFACTORING = "refactoring"  # Refactoring guidance
    LEARNING = "learning"  # Educational purposes


@dataclass
class PromptSection:
    """Represents a section of a generated prompt."""

    section_id: str  # Unique section identifier
    title: str  # Section title/header
    content: str  # Section content
    priority: int  # Priority level (1-10, higher = more important)
    context_level: ContextLevel  # Level of detail in this section

    # Metadata
    word_count: int = 0  # Approximate word count
    technical_density: float = 0.0  # Technical complexity (0-1)
    information_value: float = 0.0  # Information value score (0-1)

    def __post_init__(self):
        """Calculate derived properties."""
        self.word_count = len(self.content.split()) if self.content else 0
        self.technical_density = self._calculate_technical_density()
        self.information_value = self._calculate_information_value()

    def _calculate_technical_density(self) -> float:
        """Calculate technical complexity of the content."""
        if not self.content:
            return 0.0

        technical_keywords = {
            "function",
            "class",
            "method",
            "variable",
            "import",
            "dependency",
            "async",
            "await",
            "exception",
            "loop",
            "condition",
            "parameter",
            "return",
            "yield",
            "lambda",
            "decorator",
            "inheritance",
            "composition",
        }

        words = self.content.lower().split()
        technical_word_count = sum(1 for word in words if any(kw in word for kw in technical_keywords))

        return min(1.0, technical_word_count / max(1, len(words)))

    def _calculate_information_value(self) -> float:
        """Calculate information value of the content."""
        if not self.content:
            return 0.0

        # Simple heuristic based on content length, technical density, and priority
        length_score = min(1.0, len(self.content) / 500.0)  # Normalize to 500 chars
        priority_score = self.priority / 10.0

        return length_score * 0.4 + self.technical_density * 0.3 + priority_score * 0.3


@dataclass
class GeneratedPrompt:
    """Complete generated prompt with metadata."""

    # Core content
    prompt_id: str  # Unique prompt identifier
    full_prompt: str  # Complete generated prompt
    sections: list[PromptSection]  # Individual prompt sections

    # Configuration used
    template: PromptTemplate  # Template used for generation
    context_level: ContextLevel  # Context level used
    purpose: PromptPurpose  # Purpose of the prompt

    # Quality metrics
    total_word_count: int  # Total word count
    average_technical_density: float  # Average technical density
    information_completeness: float  # How complete the information is (0-1)
    readability_score: float  # Estimated readability (0-1)

    # Source information
    source_paths: list[str]  # IDs of source paths
    path_types_covered: set[PathType]  # Types of paths included

    # Processing metadata
    generation_time_ms: float  # Time taken to generate
    optimization_applied: bool  # Whether optimization was applied

    # Quality insights
    quality_insights: list[str] = field(default_factory=list)
    improvement_suggestions: list[str] = field(default_factory=list)

    def get_token_estimate(self) -> int:
        """Estimate token count for LLM consumption."""
        # Rough approximation: 1 token ≈ 4 characters for English text
        return max(1, len(self.full_prompt) // 4)

    def get_section_by_priority(self) -> list[PromptSection]:
        """Get sections ordered by priority (highest first)."""
        return sorted(self.sections, key=lambda s: s.priority, reverse=True)

    def is_high_quality(self) -> bool:
        """Check if this is a high-quality prompt."""
        return (
            self.information_completeness > 0.7
            and self.readability_score > 0.6
            and len(self.quality_insights) >= len(self.improvement_suggestions)
        )


@dataclass
class ConversionConfig:
    """Configuration for path-to-prompt conversion."""

    # Template and style
    default_template: PromptTemplate = PromptTemplate.STRUCTURED
    default_context_level: ContextLevel = ContextLevel.STANDARD
    default_purpose: PromptPurpose = PromptPurpose.ANALYSIS

    # Content control
    max_prompt_length: int = 4000  # Maximum prompt length in characters
    max_paths_per_prompt: int = 10  # Maximum paths to include in single prompt
    include_code_snippets: bool = True  # Include actual code snippets
    include_breadcrumbs: bool = True  # Include breadcrumb paths

    # Quality settings
    min_information_density: float = 0.3  # Minimum information density for inclusion
    prioritize_high_importance: bool = True  # Prioritize high-importance paths
    balance_path_types: bool = True  # Try to include diverse path types

    # Optimization settings
    enable_prompt_optimization: bool = True  # Apply optimization techniques
    remove_redundant_information: bool = True  # Remove redundant content
    optimize_for_token_efficiency: bool = True  # Optimize for token usage

    # Customization
    custom_templates: dict[str, str] = field(default_factory=dict)  # Custom templates
    section_priorities: dict[str, int] = field(default_factory=dict)  # Custom priorities
    terminology_mapping: dict[str, str] = field(default_factory=dict)  # Term mappings


class PathToPromptConverter:
    """
    Advanced service that converts relational paths into structured, optimized prompts
    for LLM consumption. Provides multiple templates, context levels, and optimization
    strategies to maximize the effectiveness of path information.

    Key features:
    - Multiple prompt templates and styles
    - Context level control (minimal to exhaustive)
    - Purpose-driven prompt generation
    - Token efficiency optimization
    - Quality assessment and improvement suggestions
    - Customizable templates and terminology
    """

    def __init__(self, config: ConversionConfig | None = None):
        """
        Initialize the path-to-prompt converter.

        Args:
            config: Conversion configuration options
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or ConversionConfig()

        # Performance tracking
        self._conversion_stats = {
            "total_conversions": 0,
            "successful_conversions": 0,
            "failed_conversions": 0,
            "average_generation_time_ms": 0.0,
            "average_prompt_length": 0.0,
            "total_prompts_generated": 0,
            "average_token_count": 0.0,
        }

        # Template cache
        self._template_cache: dict[str, str] = {}
        self._section_cache: dict[str, list[PromptSection]] = {}

    async def convert_path_collection(
        self,
        collection: RelationalPathCollection,
        template: PromptTemplate | None = None,
        context_level: ContextLevel | None = None,
        purpose: PromptPurpose | None = None,
    ) -> GeneratedPrompt:
        """
        Convert a path collection into a structured prompt.

        Args:
            collection: Path collection to convert
            template: Template to use for generation
            context_level: Level of context detail
            purpose: Purpose of the generated prompt

        Returns:
            GeneratedPrompt with structured context
        """
        start_time = time.time()

        try:
            self.logger.info(f"Converting path collection to prompt: {collection.collection_name}")

            # Use config defaults if not specified
            template = template or self.config.default_template
            context_level = context_level or self.config.default_context_level
            purpose = purpose or self.config.default_purpose

            # Get all paths from collection
            all_paths = collection.execution_paths + collection.data_flow_paths + collection.dependency_paths

            if not all_paths:
                return self._create_empty_prompt(collection, template, context_level, purpose, "No paths available")

            # Select and prioritize paths for inclusion
            selected_paths = await self._select_paths_for_prompt(all_paths, purpose)

            # Generate prompt sections
            sections = await self._generate_prompt_sections(selected_paths, collection, template, context_level, purpose)

            # Assemble final prompt
            full_prompt = await self._assemble_prompt(sections, template)

            # Apply optimization if enabled
            if self.config.enable_prompt_optimization:
                full_prompt, sections = await self._optimize_prompt(full_prompt, sections)
                optimization_applied = True
            else:
                optimization_applied = False

            # Calculate quality metrics
            processing_time_ms = (time.time() - start_time) * 1000
            quality_metrics = await self._calculate_quality_metrics(full_prompt, sections, selected_paths)

            # Create generated prompt
            generated_prompt = GeneratedPrompt(
                prompt_id=f"prompt_{uuid.uuid4().hex[:8]}",
                full_prompt=full_prompt,
                sections=sections,
                template=template,
                context_level=context_level,
                purpose=purpose,
                total_word_count=len(full_prompt.split()),
                average_technical_density=quality_metrics["avg_technical_density"],
                information_completeness=quality_metrics["information_completeness"],
                readability_score=quality_metrics["readability_score"],
                source_paths=[path.path_id for path in selected_paths],
                path_types_covered={path.path_type for path in selected_paths},
                generation_time_ms=processing_time_ms,
                optimization_applied=optimization_applied,
            )

            # Add quality insights and suggestions
            await self._add_quality_insights(generated_prompt, collection)

            # Update performance statistics
            self._update_performance_stats(processing_time_ms, len(full_prompt), generated_prompt.get_token_estimate(), True)

            self.logger.info(
                f"Prompt generation completed: {generated_prompt.total_word_count} words, "
                f"~{generated_prompt.get_token_estimate()} tokens in {processing_time_ms:.2f}ms"
            )

            return generated_prompt

        except Exception as e:
            self.logger.error(f"Prompt conversion failed: {str(e)}")
            processing_time_ms = (time.time() - start_time) * 1000
            self._update_performance_stats(processing_time_ms, 0, 0, False)

            return self._create_empty_prompt(
                collection,
                template or self.config.default_template,
                context_level or self.config.default_context_level,
                purpose or self.config.default_purpose,
                f"Conversion failed: {str(e)}",
            )

    async def convert_single_path(
        self,
        path: AnyPath,
        template: PromptTemplate | None = None,
        context_level: ContextLevel | None = None,
        purpose: PromptPurpose | None = None,
    ) -> GeneratedPrompt:
        """
        Convert a single path into a focused prompt.

        Args:
            path: Path to convert
            template: Template to use
            context_level: Context detail level
            purpose: Purpose of prompt

        Returns:
            Generated prompt for the single path
        """
        # Create minimal collection for the single path
        collection = RelationalPathCollection(collection_id=f"single_{path.path_id}", collection_name=f"Single Path: {path.path_id}")

        # Add path to appropriate collection based on type
        if isinstance(path, ExecutionPath):
            collection.execution_paths = [path]
        elif isinstance(path, DataFlowPath):
            collection.data_flow_paths = [path]
        elif isinstance(path, DependencyPath):
            collection.dependency_paths = [path]

        return await self.convert_path_collection(collection, template, context_level, purpose)

    async def generate_comparison_prompt(
        self, paths: list[AnyPath], comparison_focus: str = "similarities and differences"
    ) -> GeneratedPrompt:
        """
        Generate a prompt for comparing multiple paths.

        Args:
            paths: Paths to compare
            comparison_focus: What to focus on in comparison

        Returns:
            Generated comparison prompt
        """
        if len(paths) < 2:
            raise ValueError("At least 2 paths required for comparison")

        # Create comparison-focused collection
        collection = RelationalPathCollection(
            collection_id=f"comparison_{uuid.uuid4().hex[:8]}", collection_name=f"Path Comparison: {comparison_focus}"
        )

        # Group paths by type
        for path in paths:
            if isinstance(path, ExecutionPath):
                collection.execution_paths.append(path)
            elif isinstance(path, DataFlowPath):
                collection.data_flow_paths.append(path)
            elif isinstance(path, DependencyPath):
                collection.dependency_paths.append(path)

        # Use structured template for comparisons
        return await self.convert_path_collection(
            collection, template=PromptTemplate.STRUCTURED, context_level=ContextLevel.DETAILED, purpose=PromptPurpose.ANALYSIS
        )

    async def _select_paths_for_prompt(self, all_paths: list[AnyPath], purpose: PromptPurpose) -> list[AnyPath]:
        """Select and prioritize paths for inclusion in prompt."""
        if len(all_paths) <= self.config.max_paths_per_prompt:
            return all_paths

        # Score paths based on purpose and configuration
        path_scores = []
        for path in all_paths:
            score = await self._calculate_path_relevance_score(path, purpose)
            path_scores.append((path, score))

        # Sort by score (descending) and take top paths
        path_scores.sort(key=lambda x: x[1], reverse=True)

        selected_paths = []
        path_type_counts = defaultdict(int)

        for path, score in path_scores:
            # Check if we should include this path
            if len(selected_paths) >= self.config.max_paths_per_prompt:
                break

            # Balance path types if enabled
            if self.config.balance_path_types:
                if path_type_counts[path.path_type] >= 5:  # Max 5 per type
                    continue

            selected_paths.append(path)
            path_type_counts[path.path_type] += 1

        return selected_paths

    async def _calculate_path_relevance_score(self, path: AnyPath, purpose: PromptPurpose) -> float:
        """Calculate relevance score for a path given the prompt purpose."""
        base_score = 0.5

        # Get path importance
        if isinstance(path, ExecutionPath):
            importance = getattr(path, "criticality_score", 0.5)
        elif isinstance(path, DataFlowPath):
            importance = getattr(path, "data_quality_score", 0.5)
        elif isinstance(path, DependencyPath):
            importance = getattr(path, "stability_score", 0.5)
        else:
            importance = 0.5

        # Adjust based on purpose
        purpose_multiplier = 1.0
        if purpose == PromptPurpose.DEBUGGING:
            # Prefer execution paths for debugging
            if isinstance(path, ExecutionPath):
                purpose_multiplier = 1.5
        elif purpose == PromptPurpose.DOCUMENTATION:
            # Prefer comprehensive paths for documentation
            purpose_multiplier = 1.0 + (len(path.nodes) / 20.0)
        elif purpose == PromptPurpose.REFACTORING:
            # Prefer complex paths for refactoring
            complexity = getattr(path, "complexity_score", 0.0)
            purpose_multiplier = 1.0 + complexity

        # Calculate information density
        node_diversity = len({node.chunk_type for node in path.nodes})
        density_score = min(1.0, node_diversity / 5.0)

        # Combine factors
        final_score = (base_score + importance + density_score) * purpose_multiplier / 3.0

        return min(1.0, final_score)

    async def _generate_prompt_sections(
        self,
        paths: list[AnyPath],
        collection: RelationalPathCollection,
        template: PromptTemplate,
        context_level: ContextLevel,
        purpose: PromptPurpose,
    ) -> list[PromptSection]:
        """Generate individual sections of the prompt."""
        sections = []

        # Header section
        header_section = await self._create_header_section(collection, purpose)
        sections.append(header_section)

        # Overview section
        overview_section = await self._create_overview_section(paths, context_level)
        sections.append(overview_section)

        # Path-specific sections
        for path_type in {path.path_type for path in paths}:
            type_paths = [p for p in paths if p.path_type == path_type]
            type_section = await self._create_path_type_section(type_paths, path_type, context_level, template)
            sections.append(type_section)

        # Analysis section (for analysis purposes)
        if purpose == PromptPurpose.ANALYSIS:
            analysis_section = await self._create_analysis_section(paths, collection)
            sections.append(analysis_section)

        # Summary section
        summary_section = await self._create_summary_section(paths, purpose)
        sections.append(summary_section)

        return sections

    async def _create_header_section(self, collection: RelationalPathCollection, purpose: PromptPurpose) -> PromptSection:
        """Create header section with basic information."""
        content_parts = [
            f"# Code Path Analysis: {collection.collection_name}",
            "",
            f"**Purpose**: {purpose.value.title()}",
            f"**Collection ID**: {collection.collection_id}",
            f"**Total Paths**: {collection.get_total_path_count()}",
            f"**Files Covered**: {len(collection.covered_files)}",
            "",
        ]

        if collection.architectural_patterns:
            content_parts.extend([f"**Architectural Patterns Detected**: {', '.join(collection.architectural_patterns)}", ""])

        content = "\n".join(content_parts)

        return PromptSection(section_id="header", title="Header", content=content, priority=10, context_level=ContextLevel.MINIMAL)

    async def _create_overview_section(self, paths: list[AnyPath], context_level: ContextLevel) -> PromptSection:
        """Create overview section with path statistics."""
        path_type_counts = defaultdict(int)
        total_nodes = 0
        complexity_scores = []

        for path in paths:
            path_type_counts[path.path_type] += 1
            total_nodes += len(path.nodes)

            if hasattr(path, "complexity_score"):
                complexity_scores.append(getattr(path, "complexity_score", 0.0))

        content_parts = ["## Overview", "", f"This analysis covers {len(paths)} paths with {total_nodes} total nodes.", ""]

        # Path type breakdown
        content_parts.append("### Path Type Distribution")
        for path_type, count in path_type_counts.items():
            percentage = (count / len(paths)) * 100
            content_parts.append(f"- **{path_type.value.title()}**: {count} paths ({percentage:.1f}%)")

        content_parts.append("")

        # Complexity information if available
        if complexity_scores and context_level != ContextLevel.MINIMAL:
            avg_complexity = sum(complexity_scores) / len(complexity_scores)
            content_parts.extend(
                [
                    "### Complexity Metrics",
                    f"- **Average Complexity**: {avg_complexity:.2f}",
                    f"- **Complexity Range**: {min(complexity_scores):.2f} - {max(complexity_scores):.2f}",
                    "",
                ]
            )

        content = "\n".join(content_parts)

        return PromptSection(section_id="overview", title="Overview", content=content, priority=9, context_level=context_level)

    async def _create_path_type_section(
        self, type_paths: list[AnyPath], path_type: PathType, context_level: ContextLevel, template: PromptTemplate
    ) -> PromptSection:
        """Create section for a specific path type."""
        content_parts = [f"## {path_type.value.title()} Paths", "", f"Found {len(type_paths)} {path_type.value} path(s):", ""]

        for i, path in enumerate(type_paths, 1):
            content_parts.append(f"### Path {i}: {path.path_id}")

            # Basic path information
            content_parts.extend([f"- **Nodes**: {len(path.nodes)}", f"- **Path Length**: {len(path.nodes)}"])

            # Path-specific information
            if isinstance(path, ExecutionPath):
                content_parts.extend(await self._format_execution_path(path, context_level))
            elif isinstance(path, DataFlowPath):
                content_parts.extend(await self._format_data_flow_path(path, context_level))
            elif isinstance(path, DependencyPath):
                content_parts.extend(await self._format_dependency_path(path, context_level))

            # Node details for detailed context
            if context_level in {ContextLevel.DETAILED, ContextLevel.EXHAUSTIVE}:
                content_parts.extend(await self._format_path_nodes(path.nodes, context_level))

            content_parts.append("")

        content = "\n".join(content_parts)

        return PromptSection(
            section_id=f"paths_{path_type.value}",
            title=f"{path_type.value.title()} Paths",
            content=content,
            priority=8,
            context_level=context_level,
        )

    async def _format_execution_path(self, path: ExecutionPath, context_level: ContextLevel) -> list[str]:
        """Format execution path specific information."""
        details = []

        if hasattr(path, "entry_points") and path.entry_points:
            details.append(f"- **Entry Points**: {', '.join(path.entry_points)}")

        if hasattr(path, "exit_points") and path.exit_points:
            details.append(f"- **Exit Points**: {', '.join(path.exit_points)}")

        if hasattr(path, "max_depth"):
            details.append(f"- **Maximum Depth**: {path.max_depth}")

        if hasattr(path, "is_async") and path.is_async:
            details.append("- **Contains Async Operations**: Yes")

        if hasattr(path, "has_loops") and path.has_loops:
            details.append("- **Contains Loops**: Yes")

        if hasattr(path, "has_exceptions") and path.has_exceptions:
            details.append("- **Exception Handling**: Yes")

        if hasattr(path, "criticality_score"):
            details.append(f"- **Criticality Score**: {path.criticality_score:.2f}")

        if context_level == ContextLevel.EXHAUSTIVE and hasattr(path, "semantic_description"):
            if path.semantic_description:
                details.extend(
                    [
                        f"- **Description**: {path.semantic_description}",
                    ]
                )

        return details

    async def _format_data_flow_path(self, path: DataFlowPath, context_level: ContextLevel) -> list[str]:
        """Format data flow path specific information."""
        details = []

        if hasattr(path, "data_source") and path.data_source:
            details.append(f"- **Data Source**: {path.data_source}")

        if hasattr(path, "data_destinations") and path.data_destinations:
            details.append(f"- **Data Destinations**: {', '.join(path.data_destinations)}")

        if hasattr(path, "data_types") and path.data_types:
            details.append(f"- **Data Types**: {', '.join(path.data_types)}")

        if hasattr(path, "transformations") and path.transformations:
            details.append(f"- **Transformations**: {len(path.transformations)}")
            if context_level in {ContextLevel.DETAILED, ContextLevel.EXHAUSTIVE}:
                for i, transform in enumerate(path.transformations, 1):
                    details.append(f"  {i}. {transform}")

        if hasattr(path, "is_mutable"):
            details.append(f"- **Mutable Data**: {'Yes' if path.is_mutable else 'No'}")

        if hasattr(path, "has_side_effects") and path.has_side_effects:
            details.append("- **Has Side Effects**: Yes")

        if hasattr(path, "data_quality_score"):
            details.append(f"- **Data Quality Score**: {path.data_quality_score:.2f}")

        return details

    async def _format_dependency_path(self, path: DependencyPath, context_level: ContextLevel) -> list[str]:
        """Format dependency path specific information."""
        details = []

        if hasattr(path, "dependency_type"):
            details.append(f"- **Dependency Type**: {path.dependency_type}")

        if hasattr(path, "required_modules") and path.required_modules:
            details.append(f"- **Required Modules**: {', '.join(path.required_modules)}")

        if hasattr(path, "optional_modules") and path.optional_modules:
            details.append(f"- **Optional Modules**: {', '.join(path.optional_modules)}")

        if hasattr(path, "is_external") and path.is_external:
            details.append("- **External Dependency**: Yes")

        if hasattr(path, "is_circular") and path.is_circular:
            details.append("- **Circular Dependency**: Yes")

        if hasattr(path, "stability_score"):
            details.append(f"- **Stability Score**: {path.stability_score:.2f}")

        if hasattr(path, "coupling_strength"):
            details.append(f"- **Coupling Strength**: {path.coupling_strength:.2f}")

        return details

    async def _format_path_nodes(self, nodes: list[PathNode], context_level: ContextLevel) -> list[str]:
        """Format path node information."""
        if not nodes:
            return []

        details = ["- **Path Nodes**:"]

        for i, node in enumerate(nodes, 1):
            node_info = f"  {i}. **{node.name}** ({node.chunk_type.value})"

            if context_level == ContextLevel.EXHAUSTIVE:
                node_info += f" - {node.breadcrumb}"
                if node.role_in_path:
                    node_info += f" [{node.role_in_path}]"

            details.append(node_info)

            # Add importance score for detailed view
            if context_level in {ContextLevel.DETAILED, ContextLevel.EXHAUSTIVE}:
                if node.importance_score > 0:
                    details.append(f"     *Importance: {node.importance_score:.2f}*")

        return details

    async def _create_analysis_section(self, paths: list[AnyPath], collection: RelationalPathCollection) -> PromptSection:
        """Create analysis section with insights."""
        content_parts = [
            "## Analysis Insights",
            "",
        ]

        # Path complexity analysis
        if paths:
            complexity_scores = []
            for path in paths:
                if hasattr(path, "complexity_score"):
                    complexity_scores.append(getattr(path, "complexity_score", 0.0))

            if complexity_scores:
                avg_complexity = sum(complexity_scores) / len(complexity_scores)
                high_complexity_paths = len([s for s in complexity_scores if s > 0.7])

                content_parts.extend(
                    [
                        "### Complexity Analysis",
                        f"- Average complexity across all paths: {avg_complexity:.2f}",
                        f"- High complexity paths (>0.7): {high_complexity_paths}",
                        "",
                    ]
                )

        # Coverage analysis
        if collection.coverage_score > 0:
            content_parts.extend(
                [
                    "### Coverage Analysis",
                    f"- Collection coverage score: {collection.coverage_score:.2f}",
                    f"- Coherence score: {collection.coherence_score:.2f}",
                    f"- Completeness score: {collection.completeness_score:.2f}",
                    "",
                ]
            )

        # Key patterns
        if collection.architectural_patterns:
            content_parts.extend(["### Architectural Patterns", "The following patterns were identified:"])
            for pattern in collection.architectural_patterns:
                content_parts.append(f"- {pattern}")
            content_parts.append("")

        content = "\n".join(content_parts)

        return PromptSection(
            section_id="analysis", title="Analysis Insights", content=content, priority=7, context_level=ContextLevel.STANDARD
        )

    async def _create_summary_section(self, paths: list[AnyPath], purpose: PromptPurpose) -> PromptSection:
        """Create summary section."""
        content_parts = [
            "## Summary",
            "",
        ]

        # Purpose-specific summary
        if purpose == PromptPurpose.ANALYSIS:
            content_parts.extend(
                [
                    "This code analysis reveals the following key points:",
                    f"- {len(paths)} distinct paths covering various code relationships",
                    "- Mix of execution flows, data transformations, and dependencies",
                    "- Varying complexity levels requiring different analysis approaches",
                ]
            )
        elif purpose == PromptPurpose.DEBUGGING:
            content_parts.extend(
                [
                    "For debugging purposes, focus on:",
                    "- Execution paths showing call sequences",
                    "- Data flow paths revealing state changes",
                    "- Error-prone areas with high complexity",
                ]
            )
        elif purpose == PromptPurpose.DOCUMENTATION:
            content_parts.extend(
                [
                    "Documentation should cover:",
                    "- Key execution flows and their purposes",
                    "- Data transformations and their effects",
                    "- Dependencies and their relationships",
                ]
            )
        else:
            content_parts.extend(
                [
                    f"This analysis provides comprehensive path information for {purpose.value} tasks.",
                    "Review the detailed path information above for specific insights.",
                ]
            )

        content_parts.extend(["", "---", "*Generated by PathRAG Path-to-Prompt Converter*"])

        content = "\n".join(content_parts)

        return PromptSection(section_id="summary", title="Summary", content=content, priority=6, context_level=ContextLevel.STANDARD)

    async def _assemble_prompt(self, sections: list[PromptSection], template: PromptTemplate) -> str:
        """Assemble final prompt from sections."""
        if template == PromptTemplate.COMPREHENSIVE:
            # Include all sections in priority order
            sorted_sections = sorted(sections, key=lambda s: s.priority, reverse=True)
            return "\n\n".join(section.content for section in sorted_sections)

        elif template == PromptTemplate.CONCISE:
            # Include only high-priority sections
            high_priority_sections = [s for s in sections if s.priority >= 8]
            sorted_sections = sorted(high_priority_sections, key=lambda s: s.priority, reverse=True)
            return "\n\n".join(section.content for section in sorted_sections)

        elif template == PromptTemplate.STRUCTURED:
            # Well-structured with clear headers
            sorted_sections = sorted(sections, key=lambda s: s.priority, reverse=True)
            content_parts = []

            for section in sorted_sections:
                if section.content.strip():
                    content_parts.append(section.content)

            return "\n\n".join(content_parts)

        elif template == PromptTemplate.NARRATIVE:
            # More natural language flow
            return await self._create_narrative_prompt(sections)

        elif template == PromptTemplate.TECHNICAL:
            # Technical documentation style
            return await self._create_technical_prompt(sections)

        else:  # CONVERSATIONAL
            return await self._create_conversational_prompt(sections)

    async def _create_narrative_prompt(self, sections: list[PromptSection]) -> str:
        """Create narrative-style prompt."""
        # This would create a more flowing, natural language version
        # For now, return structured version with narrative connectors
        sorted_sections = sorted(sections, key=lambda s: s.priority, reverse=True)

        content_parts = []
        for i, section in enumerate(sorted_sections):
            if i > 0:
                content_parts.append("Additionally, ")
            content_parts.append(section.content)

        return "\n\n".join(content_parts)

    async def _create_technical_prompt(self, sections: list[PromptSection]) -> str:
        """Create technical documentation style prompt."""
        sorted_sections = sorted(sections, key=lambda s: s.priority, reverse=True)

        # Add technical formatting
        content_parts = ["# Technical Analysis Report", ""]

        for section in sorted_sections:
            if section.section_id != "header":  # Skip duplicate header
                content_parts.append(section.content)

        return "\n\n".join(content_parts)

    async def _create_conversational_prompt(self, sections: list[PromptSection]) -> str:
        """Create conversational style prompt."""
        sorted_sections = sorted(sections, key=lambda s: s.priority, reverse=True)

        # Make it more conversational
        content_parts = ["Let me walk you through this code analysis:", ""]

        for section in sorted_sections:
            # Convert technical headers to conversational language
            conversational_content = section.content.replace("##", "Looking at")
            conversational_content = conversational_content.replace("###", "For")
            content_parts.append(conversational_content)

        return "\n\n".join(content_parts)

    async def _optimize_prompt(self, full_prompt: str, sections: list[PromptSection]) -> tuple[str, list[PromptSection]]:
        """Optimize prompt for token efficiency and quality."""
        if not self.config.enable_prompt_optimization:
            return full_prompt, sections

        # Check if prompt exceeds length limit
        if len(full_prompt) <= self.config.max_prompt_length:
            return full_prompt, sections

        # Remove least important sections until under limit
        sorted_sections = sorted(sections, key=lambda s: s.priority, reverse=True)
        optimized_sections = []
        current_length = 0

        for section in sorted_sections:
            section_length = len(section.content)
            if current_length + section_length <= self.config.max_prompt_length:
                optimized_sections.append(section)
                current_length += section_length
            elif section.priority >= 9:  # Always include critical sections
                optimized_sections.append(section)
                current_length += section_length

        # Reassemble optimized prompt
        optimized_prompt = "\n\n".join(section.content for section in optimized_sections)

        # Apply additional optimizations
        if self.config.remove_redundant_information:
            optimized_prompt = await self._remove_redundant_information(optimized_prompt)

        if self.config.optimize_for_token_efficiency:
            optimized_prompt = await self._optimize_for_tokens(optimized_prompt)

        return optimized_prompt, optimized_sections

    async def _remove_redundant_information(self, prompt: str) -> str:
        """Remove redundant information from prompt."""
        # Simple deduplication - remove repeated phrases
        lines = prompt.split("\n")
        seen_content = set()
        deduplicated_lines = []

        for line in lines:
            content = line.strip().lower()
            # Skip empty lines and headers
            if not content or content.startswith("#") or content.startswith("**"):
                deduplicated_lines.append(line)
            elif content not in seen_content:
                seen_content.add(content)
                deduplicated_lines.append(line)

        return "\n".join(deduplicated_lines)

    async def _optimize_for_tokens(self, prompt: str) -> str:
        """Optimize prompt for token efficiency."""
        # Simple token optimization - remove unnecessary words
        optimizations = [
            ("  ", " "),  # Multiple spaces to single space
            (" the the ", " the "),  # Duplicate articles
            (" and and ", " and "),  # Duplicate conjunctions
            ("**Path ", "**P"),  # Shorten common prefixes
            ("**Nodes**: ", "**N**: "),  # Abbreviate common terms
            ("Complexity Score", "Complexity"),  # Shorten terms
        ]

        optimized_prompt = prompt
        for old, new in optimizations:
            optimized_prompt = optimized_prompt.replace(old, new)

        return optimized_prompt

    async def _calculate_quality_metrics(self, full_prompt: str, sections: list[PromptSection], paths: list[AnyPath]) -> dict[str, float]:
        """Calculate quality metrics for the generated prompt."""
        # Technical density
        technical_densities = [s.technical_density for s in sections if s.technical_density > 0]
        avg_technical_density = sum(technical_densities) / len(technical_densities) if technical_densities else 0.0

        # Information completeness (based on path coverage)
        total_nodes = sum(len(path.nodes) for path in paths)
        nodes_mentioned = full_prompt.count("**") + full_prompt.count("- ")  # Rough approximation
        information_completeness = min(1.0, nodes_mentioned / max(1, total_nodes))

        # Readability score (based on sentence length and structure)
        sentences = full_prompt.split(".")
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, len(sentences))
        readability_score = max(0.0, min(1.0, 1.0 - (avg_sentence_length - 15) / 30))  # Optimal ~15 words/sentence

        return {
            "avg_technical_density": avg_technical_density,
            "information_completeness": information_completeness,
            "readability_score": readability_score,
        }

    async def _add_quality_insights(self, generated_prompt: GeneratedPrompt, collection: RelationalPathCollection):
        """Add quality insights and improvement suggestions."""
        insights = []
        suggestions = []

        # Token count analysis
        token_count = generated_prompt.get_token_estimate()
        if token_count > 3000:
            suggestions.append("Consider using CONCISE template to reduce token usage")
        elif token_count < 500:
            insights.append("Compact prompt suitable for focused queries")

        # Information completeness analysis
        if generated_prompt.information_completeness > 0.8:
            insights.append("High information completeness - comprehensive coverage")
        elif generated_prompt.information_completeness < 0.5:
            suggestions.append("Consider including more path details for better completeness")

        # Readability analysis
        if generated_prompt.readability_score > 0.7:
            insights.append("Good readability score - well-structured content")
        elif generated_prompt.readability_score < 0.4:
            suggestions.append("Consider breaking down complex sentences for better readability")

        # Path type coverage
        if len(generated_prompt.path_types_covered) == 3:
            insights.append("Comprehensive path type coverage (execution, data flow, dependencies)")
        elif len(generated_prompt.path_types_covered) == 1:
            insights.append(f"Focused on {list(generated_prompt.path_types_covered)[0].value} paths only")

        generated_prompt.quality_insights = insights
        generated_prompt.improvement_suggestions = suggestions

    def _create_empty_prompt(
        self,
        collection: RelationalPathCollection,
        template: PromptTemplate,
        context_level: ContextLevel,
        purpose: PromptPurpose,
        error_message: str,
    ) -> GeneratedPrompt:
        """Create empty prompt for error cases."""
        empty_content = f"# Error: {error_message}\n\nCollection: {collection.collection_name}\nNo paths available for conversion."

        return GeneratedPrompt(
            prompt_id=f"empty_{uuid.uuid4().hex[:8]}",
            full_prompt=empty_content,
            sections=[],
            template=template,
            context_level=context_level,
            purpose=purpose,
            total_word_count=len(empty_content.split()),
            average_technical_density=0.0,
            information_completeness=0.0,
            readability_score=0.0,
            source_paths=[],
            path_types_covered=set(),
            generation_time_ms=0.0,
            optimization_applied=False,
            improvement_suggestions=[error_message],
        )

    def _update_performance_stats(self, processing_time_ms: float, prompt_length: int, token_count: int, success: bool):
        """Update internal performance statistics."""
        self._conversion_stats["total_conversions"] += 1

        if success:
            self._conversion_stats["successful_conversions"] += 1

            # Update averages
            operations = self._conversion_stats["successful_conversions"]

            # Average generation time
            current_avg_time = self._conversion_stats["average_generation_time_ms"]
            self._conversion_stats["average_generation_time_ms"] = (current_avg_time * (operations - 1) + processing_time_ms) / operations

            # Average prompt length
            current_avg_length = self._conversion_stats["average_prompt_length"]
            self._conversion_stats["average_prompt_length"] = (current_avg_length * (operations - 1) + prompt_length) / operations

            # Average token count
            current_avg_tokens = self._conversion_stats["average_token_count"]
            self._conversion_stats["average_token_count"] = (current_avg_tokens * (operations - 1) + token_count) / operations

            self._conversion_stats["total_prompts_generated"] += 1
        else:
            self._conversion_stats["failed_conversions"] += 1

    def get_performance_stats(self) -> dict[str, Any]:
        """Get current performance statistics."""
        stats = dict(self._conversion_stats)

        # Add derived metrics
        total_conversions = stats["total_conversions"]
        if total_conversions > 0:
            stats["success_rate"] = stats["successful_conversions"] / total_conversions
            stats["failure_rate"] = stats["failed_conversions"] / total_conversions
        else:
            stats["success_rate"] = 0.0
            stats["failure_rate"] = 0.0

        # Add cache statistics
        stats["template_cache_size"] = len(self._template_cache)
        stats["section_cache_size"] = len(self._section_cache)

        return stats

    def clear_caches(self):
        """Clear internal caches to free memory."""
        self._template_cache.clear()
        self._section_cache.clear()
        self.logger.info("Path-to-prompt converter caches cleared")


# Factory function
def create_path_to_prompt_converter(config: ConversionConfig | None = None) -> PathToPromptConverter:
    """
    Factory function to create a PathToPromptConverter instance.

    Args:
        config: Optional conversion configuration

    Returns:
        Configured PathToPromptConverter instance
    """
    return PathToPromptConverter(config)
