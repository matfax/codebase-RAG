"""
Graph Analysis Report Service

This service provides comprehensive report generation functionality for graph analysis tools,
including statistical summaries, recommendations, and detailed insights.
"""

import asyncio
import logging
import statistics
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Union

from src.models.code_chunk import ChunkType
from src.services.embedding_service import EmbeddingService
from src.services.graph_rag_service import GraphRAGService
from src.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)


class ReportType(Enum):
    """Types of graph analysis reports."""

    STRUCTURE_SUMMARY = "structure_summary"
    PATTERN_ANALYSIS = "pattern_analysis"
    CONNECTIVITY_INSIGHTS = "connectivity_insights"
    PERFORMANCE_ANALYSIS = "performance_analysis"
    COMPREHENSIVE = "comprehensive"
    PROJECT_HEALTH = "project_health"


class SeverityLevel(Enum):
    """Severity levels for recommendations."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    OPTIMIZATION = "optimization"


@dataclass
class Recommendation:
    """A single recommendation with context and impact."""

    title: str
    description: str
    severity: SeverityLevel
    category: str
    impact: str
    suggested_actions: list[str]
    affected_components: list[str] = field(default_factory=list)
    confidence: float = 1.0
    estimated_effort: str = "medium"


@dataclass
class StatisticalSummary:
    """Statistical summary for graph analysis metrics."""

    metric_name: str
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    percentiles: dict[int, float] = field(default_factory=dict)
    distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class GraphAnalysisReport:
    """Comprehensive graph analysis report."""

    report_type: ReportType
    project_name: str
    generated_at: datetime
    execution_time_ms: float

    # Core content
    summary: dict[str, Any]
    statistics: list[StatisticalSummary]
    recommendations: list[Recommendation]
    insights: list[str]

    # Detailed analysis
    structural_metrics: dict[str, Any] = field(default_factory=dict)
    performance_metrics: dict[str, Any] = field(default_factory=dict)
    pattern_metrics: dict[str, Any] = field(default_factory=dict)
    connectivity_metrics: dict[str, Any] = field(default_factory=dict)

    # Metadata
    scope: str = "project"
    analysis_depth: str = "standard"
    data_quality_score: float = 1.0
    confidence_score: float = 1.0


class GraphAnalysisReportService:
    """Service for generating comprehensive graph analysis reports."""

    def __init__(self, graph_rag_service: GraphRAGService):
        """
        Initialize the report service.

        Args:
            graph_rag_service: Graph RAG service for data analysis
        """
        self.graph_rag_service = graph_rag_service
        self.logger = logging.getLogger(__name__)

        # Performance tracking
        self._report_generation_stats = {
            "reports_generated": 0,
            "avg_generation_time_ms": 0.0,
            "cache_hits": 0,
            "cache_misses": 0,
        }

    async def generate_structure_analysis_report(
        self,
        breadcrumb: str,
        project_name: str,
        analysis_type: str = "comprehensive",
        include_performance_insights: bool = True,
        include_optimization_suggestions: bool = True,
    ) -> GraphAnalysisReport:
        """
        Generate a comprehensive structure analysis report.

        Args:
            breadcrumb: The breadcrumb to analyze
            project_name: Project name
            analysis_type: Type of analysis performed
            include_performance_insights: Include performance analysis
            include_optimization_suggestions: Include optimization recommendations

        Returns:
            Comprehensive GraphAnalysisReport
        """
        start_time = time.time()

        try:
            self.logger.info(f"Generating structure analysis report for {breadcrumb} in {project_name}")

            # Get the structure graph for analysis
            graph = await self.graph_rag_service.build_structure_graph(project_name)

            if not graph or breadcrumb not in graph.nodes:
                return self._create_error_report(
                    ReportType.STRUCTURE_SUMMARY, project_name, f"Component '{breadcrumb}' not found in project structure"
                )

            # Collect analysis data
            hierarchy_data = await self.graph_rag_service.get_component_hierarchy(breadcrumb, project_name, include_siblings=True)

            connectivity_data = await self.graph_rag_service.analyze_component_connectivity(breadcrumb, project_name)

            related_components = await self.graph_rag_service.find_related_components(breadcrumb, project_name, max_depth=5)

            # Generate statistical summaries
            statistics = await self._generate_structure_statistics(graph, breadcrumb, hierarchy_data, connectivity_data, related_components)

            # Generate insights
            insights = self._generate_structure_insights(breadcrumb, hierarchy_data, connectivity_data, related_components, graph)

            # Generate recommendations
            recommendations = []
            if include_optimization_suggestions:
                recommendations.extend(
                    self._generate_structure_recommendations(breadcrumb, hierarchy_data, connectivity_data, related_components, graph)
                )

            # Performance insights
            performance_metrics = {}
            if include_performance_insights:
                performance_metrics = self._analyze_structure_performance(breadcrumb, graph, related_components)

            # Create summary
            summary = {
                "target_component": {
                    "breadcrumb": breadcrumb,
                    "type": graph.nodes[breadcrumb].chunk_type.value,
                    "depth": graph.nodes[breadcrumb].depth,
                    "file_path": graph.nodes[breadcrumb].file_path,
                },
                "hierarchy_depth": len(hierarchy_data.get("ancestors", [])),
                "children_count": len(hierarchy_data.get("descendants", [])),
                "siblings_count": len(hierarchy_data.get("siblings", [])),
                "related_components_count": len(related_components.related_components),
                "connectivity_score": connectivity_data.get("influence_score", 0.0),
                "analysis_type": analysis_type,
            }

            execution_time = (time.time() - start_time) * 1000

            report = GraphAnalysisReport(
                report_type=ReportType.STRUCTURE_SUMMARY,
                project_name=project_name,
                generated_at=datetime.now(),
                execution_time_ms=execution_time,
                summary=summary,
                statistics=statistics,
                recommendations=recommendations,
                insights=insights,
                structural_metrics=self._extract_structural_metrics(hierarchy_data, connectivity_data),
                performance_metrics=performance_metrics,
                scope=f"component:{breadcrumb}",
                analysis_depth=analysis_type,
                data_quality_score=self._calculate_data_quality_score(graph, breadcrumb),
                confidence_score=self._calculate_confidence_score(hierarchy_data, connectivity_data),
            )

            self._update_generation_stats(execution_time)
            return report

        except Exception as e:
            self.logger.error(f"Error generating structure analysis report: {e}")
            return self._create_error_report(ReportType.STRUCTURE_SUMMARY, project_name, f"Error during report generation: {str(e)}")

    async def generate_pattern_analysis_report(
        self, project_name: str, pattern_types: list[str] = None, min_confidence: float = 0.6, include_comparisons: bool = True
    ) -> GraphAnalysisReport:
        """
        Generate a pattern analysis report.

        Args:
            project_name: Project to analyze
            pattern_types: Types of patterns to analyze
            min_confidence: Minimum confidence threshold
            include_comparisons: Include pattern comparisons

        Returns:
            Pattern analysis report
        """
        start_time = time.time()

        try:
            self.logger.info(f"Generating pattern analysis report for {project_name}")

            # Get project structure
            graph = await self.graph_rag_service.build_structure_graph(project_name)
            project_overview = await self.graph_rag_service.get_project_structure_overview(project_name)

            # Analyze patterns
            pattern_insights = self._analyze_architectural_patterns(graph, project_overview)
            design_patterns = self._identify_design_patterns(graph)
            naming_patterns = self._analyze_naming_patterns(graph)

            # Generate statistics
            statistics = await self._generate_pattern_statistics(pattern_insights, design_patterns, naming_patterns)

            # Generate insights
            insights = self._generate_pattern_insights(pattern_insights, design_patterns, naming_patterns, project_overview)

            # Generate recommendations
            recommendations = self._generate_pattern_recommendations(pattern_insights, design_patterns, naming_patterns, min_confidence)

            summary = {
                "total_patterns_identified": len(pattern_insights) + len(design_patterns) + len(naming_patterns),
                "architectural_patterns": len(pattern_insights),
                "design_patterns": len(design_patterns),
                "naming_patterns": len(naming_patterns),
                "confidence_threshold": min_confidence,
                "pattern_quality_score": self._calculate_pattern_quality_score(pattern_insights, design_patterns, naming_patterns),
            }

            execution_time = (time.time() - start_time) * 1000

            report = GraphAnalysisReport(
                report_type=ReportType.PATTERN_ANALYSIS,
                project_name=project_name,
                generated_at=datetime.now(),
                execution_time_ms=execution_time,
                summary=summary,
                statistics=statistics,
                recommendations=recommendations,
                insights=insights,
                pattern_metrics={
                    "architectural_patterns": pattern_insights,
                    "design_patterns": design_patterns,
                    "naming_patterns": naming_patterns,
                },
                scope="project",
                analysis_depth="comprehensive",
                confidence_score=self._calculate_pattern_confidence(pattern_insights, design_patterns),
            )

            self._update_generation_stats(execution_time)
            return report

        except Exception as e:
            self.logger.error(f"Error generating pattern analysis report: {e}")
            return self._create_error_report(ReportType.PATTERN_ANALYSIS, project_name, f"Error during pattern analysis: {str(e)}")

    async def generate_project_health_report(
        self, project_name: str, include_performance_analysis: bool = True, include_recommendations: bool = True
    ) -> GraphAnalysisReport:
        """
        Generate a comprehensive project health report.

        Args:
            project_name: Project to analyze
            include_performance_analysis: Include performance metrics
            include_recommendations: Include health recommendations

        Returns:
            Project health report
        """
        start_time = time.time()

        try:
            self.logger.info(f"Generating project health report for {project_name}")

            # Get comprehensive project data
            graph = await self.graph_rag_service.build_structure_graph(project_name)
            project_overview = await self.graph_rag_service.get_project_structure_overview(project_name)
            performance_stats = self.graph_rag_service.get_performance_stats()

            # Analyze project health metrics
            health_metrics = self._analyze_project_health(graph, project_overview)
            structure_quality = self._assess_structure_quality(graph, project_overview)
            maintainability_score = self._calculate_maintainability_score(graph, project_overview)

            # Generate statistics
            statistics = await self._generate_health_statistics(graph, project_overview, health_metrics)

            # Generate insights
            insights = self._generate_health_insights(health_metrics, structure_quality, maintainability_score, project_overview)

            # Generate recommendations
            recommendations = []
            if include_recommendations:
                recommendations = self._generate_health_recommendations(health_metrics, structure_quality, maintainability_score)

            summary = {
                "overall_health_score": health_metrics.get("overall_score", 0.0),
                "structure_quality_score": structure_quality.get("score", 0.0),
                "maintainability_score": maintainability_score,
                "total_components": len(graph.nodes),
                "total_relationships": len(graph.edges),
                "critical_issues": len([r for r in recommendations if r.severity == SeverityLevel.CRITICAL]),
                "warnings": len([r for r in recommendations if r.severity == SeverityLevel.WARNING]),
                "optimization_opportunities": len([r for r in recommendations if r.severity == SeverityLevel.OPTIMIZATION]),
            }

            execution_time = (time.time() - start_time) * 1000

            report = GraphAnalysisReport(
                report_type=ReportType.PROJECT_HEALTH,
                project_name=project_name,
                generated_at=datetime.now(),
                execution_time_ms=execution_time,
                summary=summary,
                statistics=statistics,
                recommendations=recommendations,
                insights=insights,
                structural_metrics=health_metrics,
                performance_metrics=performance_stats if include_performance_analysis else {},
                scope="project",
                analysis_depth="comprehensive",
                data_quality_score=structure_quality.get("data_quality", 1.0),
                confidence_score=structure_quality.get("confidence", 1.0),
            )

            self._update_generation_stats(execution_time)
            return report

        except Exception as e:
            self.logger.error(f"Error generating project health report: {e}")
            return self._create_error_report(ReportType.PROJECT_HEALTH, project_name, f"Error during health analysis: {str(e)}")

    async def generate_comprehensive_report(
        self, project_name: str, target_breadcrumb: str = None, include_all_metrics: bool = True
    ) -> GraphAnalysisReport:
        """
        Generate a comprehensive report combining all analysis types.

        Args:
            project_name: Project to analyze
            target_breadcrumb: Optional specific component to focus on
            include_all_metrics: Include all available metrics

        Returns:
            Comprehensive analysis report
        """
        start_time = time.time()

        try:
            self.logger.info(f"Generating comprehensive analysis report for {project_name}")

            # Generate individual reports in parallel
            tasks = []

            if target_breadcrumb:
                tasks.append(self.generate_structure_analysis_report(target_breadcrumb, project_name, "comprehensive"))

            tasks.extend(
                [
                    self.generate_pattern_analysis_report(project_name),
                    self.generate_project_health_report(project_name, include_all_metrics),
                ]
            )

            individual_reports = await asyncio.gather(*tasks, return_exceptions=True)

            # Combine results
            all_statistics = []
            all_recommendations = []
            all_insights = []
            combined_metrics = {}

            for report in individual_reports:
                if isinstance(report, GraphAnalysisReport):
                    all_statistics.extend(report.statistics)
                    all_recommendations.extend(report.recommendations)
                    all_insights.extend(report.insights)
                    combined_metrics.update(
                        {
                            f"{report.report_type.value}_metrics": {
                                "structural": report.structural_metrics,
                                "performance": report.performance_metrics,
                                "pattern": report.pattern_metrics,
                                "connectivity": report.connectivity_metrics,
                            }
                        }
                    )

            # Generate summary insights
            comprehensive_insights = self._generate_comprehensive_insights(individual_reports)
            all_insights.extend(comprehensive_insights)

            # Prioritize recommendations
            prioritized_recommendations = self._prioritize_recommendations(all_recommendations)

            summary = {
                "analysis_scope": "comprehensive",
                "target_component": target_breadcrumb,
                "total_insights": len(all_insights),
                "total_recommendations": len(prioritized_recommendations),
                "critical_issues": len([r for r in prioritized_recommendations if r.severity == SeverityLevel.CRITICAL]),
                "reports_combined": len([r for r in individual_reports if isinstance(r, GraphAnalysisReport)]),
                "analysis_quality": self._assess_comprehensive_quality(individual_reports),
            }

            execution_time = (time.time() - start_time) * 1000

            report = GraphAnalysisReport(
                report_type=ReportType.COMPREHENSIVE,
                project_name=project_name,
                generated_at=datetime.now(),
                execution_time_ms=execution_time,
                summary=summary,
                statistics=all_statistics,
                recommendations=prioritized_recommendations,
                insights=all_insights,
                structural_metrics=combined_metrics,
                scope=f"comprehensive:{target_breadcrumb or 'project'}",
                analysis_depth="comprehensive",
                confidence_score=self._calculate_comprehensive_confidence(individual_reports),
            )

            self._update_generation_stats(execution_time)
            return report

        except Exception as e:
            self.logger.error(f"Error generating comprehensive report: {e}")
            return self._create_error_report(ReportType.COMPREHENSIVE, project_name, f"Error during comprehensive analysis: {str(e)}")

    def get_report_statistics(self) -> dict[str, Any]:
        """Get report generation statistics."""
        return {
            "generation_stats": self._report_generation_stats.copy(),
            "service_health": "operational",
            "total_reports": self._report_generation_stats["reports_generated"],
        }

    # =================== Private Helper Methods ===================

    def _create_error_report(self, report_type: ReportType, project_name: str, error_message: str) -> GraphAnalysisReport:
        """Create an error report when analysis fails."""
        return GraphAnalysisReport(
            report_type=report_type,
            project_name=project_name,
            generated_at=datetime.now(),
            execution_time_ms=0.0,
            summary={"error": error_message},
            statistics=[],
            recommendations=[
                Recommendation(
                    title="Analysis Error",
                    description=error_message,
                    severity=SeverityLevel.CRITICAL,
                    category="system",
                    impact="Cannot perform analysis",
                    suggested_actions=["Check project indexing", "Verify component exists", "Check system logs"],
                )
            ],
            insights=[f"Analysis failed: {error_message}"],
            confidence_score=0.0,
        )

    async def _generate_structure_statistics(
        self, graph, breadcrumb: str, hierarchy_data: dict, connectivity_data: dict, related_components
    ) -> list[StatisticalSummary]:
        """Generate statistical summaries for structure analysis."""
        statistics = []

        # Hierarchy depth statistics
        if hierarchy_data.get("ancestors"):
            ancestors = hierarchy_data["ancestors"]
            statistics.append(
                StatisticalSummary(
                    metric_name="hierarchy_depth",
                    count=len(ancestors),
                    mean=len(ancestors),
                    median=len(ancestors),
                    std_dev=0.0,
                    min_value=len(ancestors),
                    max_value=len(ancestors),
                )
            )

        # Related components statistics
        if related_components.related_components:
            similarity_scores = [getattr(comp, "similarity_score", 0.0) for comp in related_components.related_components]
            if similarity_scores:
                statistics.append(
                    StatisticalSummary(
                        metric_name="component_similarity",
                        count=len(similarity_scores),
                        mean=statistics.mean(similarity_scores),
                        median=statistics.median(similarity_scores),
                        std_dev=statistics.stdev(similarity_scores) if len(similarity_scores) > 1 else 0.0,
                        min_value=min(similarity_scores),
                        max_value=max(similarity_scores),
                    )
                )

        # Connectivity statistics
        if connectivity_data:
            influence_score = connectivity_data.get("influence_score", 0.0)
            statistics.append(
                StatisticalSummary(
                    metric_name="connectivity_influence",
                    count=1,
                    mean=influence_score,
                    median=influence_score,
                    std_dev=0.0,
                    min_value=influence_score,
                    max_value=influence_score,
                )
            )

        return statistics

    def _generate_structure_insights(
        self, breadcrumb: str, hierarchy_data: dict, connectivity_data: dict, related_components, graph
    ) -> list[str]:
        """Generate insights for structure analysis."""
        insights = []

        # Hierarchy insights
        ancestors = hierarchy_data.get("ancestors", [])
        descendants = hierarchy_data.get("descendants", [])
        siblings = hierarchy_data.get("siblings", [])

        if len(ancestors) > 3:
            insights.append(f"Component '{breadcrumb}' is deeply nested with {len(ancestors)} ancestor levels")

        if len(descendants) > 10:
            insights.append(f"Component '{breadcrumb}' has high complexity with {len(descendants)} child components")
        elif len(descendants) == 0:
            insights.append(f"Component '{breadcrumb}' is a leaf node with no child components")

        if len(siblings) > 5:
            insights.append(f"Component '{breadcrumb}' is part of a large sibling group ({len(siblings)} siblings)")

        # Connectivity insights
        influence_score = connectivity_data.get("influence_score", 0.0)
        if influence_score > 0.8:
            insights.append(f"Component '{breadcrumb}' has high connectivity influence (score: {influence_score:.2f})")
        elif influence_score < 0.2:
            insights.append(f"Component '{breadcrumb}' has low connectivity influence (score: {influence_score:.2f})")

        # Related components insights
        if related_components and len(related_components.related_components) > 20:
            insights.append(
                f"Component '{breadcrumb}' has extensive relationships with {len(related_components.related_components)} related components"
            )

        return insights

    def _generate_structure_recommendations(
        self, breadcrumb: str, hierarchy_data: dict, connectivity_data: dict, related_components, graph
    ) -> list[Recommendation]:
        """Generate recommendations for structure optimization."""
        recommendations = []

        # Deep nesting recommendation
        ancestors = hierarchy_data.get("ancestors", [])
        if len(ancestors) > 4:
            recommendations.append(
                Recommendation(
                    title="Deep Component Nesting Detected",
                    description=f"Component '{breadcrumb}' is nested {len(ancestors)} levels deep, which may impact maintainability",
                    severity=SeverityLevel.WARNING,
                    category="structure",
                    impact="Reduced code maintainability and navigation difficulty",
                    suggested_actions=[
                        "Consider refactoring to reduce nesting levels",
                        "Extract intermediate abstraction layers",
                        "Review component hierarchy design",
                    ],
                    affected_components=[breadcrumb],
                    confidence=0.8,
                    estimated_effort="medium",
                )
            )

        # High complexity recommendation
        descendants = hierarchy_data.get("descendants", [])
        if len(descendants) > 15:
            recommendations.append(
                Recommendation(
                    title="High Component Complexity",
                    description=f"Component '{breadcrumb}' manages {len(descendants)} child components",
                    severity=SeverityLevel.OPTIMIZATION,
                    category="complexity",
                    impact="Potential difficulty in understanding and maintaining the component",
                    suggested_actions=[
                        "Consider breaking down into smaller components",
                        "Group related child components",
                        "Apply composition patterns",
                    ],
                    affected_components=[breadcrumb],
                    confidence=0.7,
                    estimated_effort="high",
                )
            )

        # Low connectivity recommendation
        influence_score = connectivity_data.get("influence_score", 0.0)
        if influence_score < 0.1:
            recommendations.append(
                Recommendation(
                    title="Low Component Connectivity",
                    description=f"Component '{breadcrumb}' has very low connectivity (score: {influence_score:.2f})",
                    severity=SeverityLevel.INFO,
                    category="connectivity",
                    impact="Component may be isolated or underutilized",
                    suggested_actions=[
                        "Review if component is still needed",
                        "Consider integrating with related components",
                        "Verify component documentation",
                    ],
                    affected_components=[breadcrumb],
                    confidence=0.6,
                    estimated_effort="low",
                )
            )

        return recommendations

    def _analyze_architectural_patterns(self, graph, project_overview) -> list[dict[str, Any]]:
        """Analyze architectural patterns in the project."""
        patterns = []

        # Analyze layered architecture
        depth_distribution = project_overview.get("breakdown", {}).get("by_depth", {})
        if len(depth_distribution) > 3:
            patterns.append(
                {
                    "name": "Layered Architecture",
                    "confidence": 0.8,
                    "description": f"Project shows {len(depth_distribution)} distinct architectural layers",
                    "components_involved": len(graph.nodes),
                    "pattern_type": "architectural",
                }
            )

        # Analyze modular structure
        type_breakdown = project_overview.get("breakdown", {}).get("by_type", {})
        if len(type_breakdown) > 3 and "class" in type_breakdown and "function" in type_breakdown:
            patterns.append(
                {
                    "name": "Modular Design",
                    "confidence": 0.7,
                    "description": "Project demonstrates good separation between classes and functions",
                    "components_involved": sum(type_breakdown.values()),
                    "pattern_type": "structural",
                }
            )

        return patterns

    def _identify_design_patterns(self, graph) -> list[dict[str, Any]]:
        """Identify design patterns in the codebase."""
        patterns = []

        # Look for factory patterns (classes with create/build methods)
        factory_candidates = []
        for node in graph.nodes.values():
            if node.chunk_type == ChunkType.CLASS:
                children_names = [graph.nodes[child].name.lower() for child in node.children_breadcrumbs if child in graph.nodes]
                if any(name in ["create", "build", "make", "factory"] for name in children_names):
                    factory_candidates.append(node.breadcrumb)

        if factory_candidates:
            patterns.append(
                {
                    "name": "Factory Pattern",
                    "confidence": 0.6,
                    "description": f"Identified {len(factory_candidates)} potential factory patterns",
                    "components_involved": len(factory_candidates),
                    "pattern_type": "creational",
                }
            )

        # Look for observer patterns (classes with add/remove listener methods)
        observer_candidates = []
        for node in graph.nodes.values():
            if node.chunk_type == ChunkType.CLASS:
                children_names = [graph.nodes[child].name.lower() for child in node.children_breadcrumbs if child in graph.nodes]
                if any("listener" in name or "observer" in name for name in children_names):
                    observer_candidates.append(node.breadcrumb)

        if observer_candidates:
            patterns.append(
                {
                    "name": "Observer Pattern",
                    "confidence": 0.7,
                    "description": f"Identified {len(observer_candidates)} potential observer patterns",
                    "components_involved": len(observer_candidates),
                    "pattern_type": "behavioral",
                }
            )

        return patterns

    def _analyze_naming_patterns(self, graph) -> list[dict[str, Any]]:
        """Analyze naming patterns in the codebase."""
        patterns = []

        # Analyze naming consistency
        function_names = [node.name for node in graph.nodes.values() if node.chunk_type == ChunkType.FUNCTION and node.name]
        class_names = [node.name for node in graph.nodes.values() if node.chunk_type == ChunkType.CLASS and node.name]

        # Check for snake_case in functions
        snake_case_functions = sum(1 for name in function_names if "_" in name and name.islower())
        if snake_case_functions > len(function_names) * 0.7:
            patterns.append(
                {
                    "name": "Snake Case Naming",
                    "confidence": 0.9,
                    "description": f"{snake_case_functions}/{len(function_names)} functions use snake_case naming",
                    "components_involved": snake_case_functions,
                    "pattern_type": "naming",
                }
            )

        # Check for PascalCase in classes
        pascal_case_classes = sum(1 for name in class_names if name and name[0].isupper())
        if pascal_case_classes > len(class_names) * 0.7:
            patterns.append(
                {
                    "name": "Pascal Case Naming",
                    "confidence": 0.9,
                    "description": f"{pascal_case_classes}/{len(class_names)} classes use PascalCase naming",
                    "components_involved": pascal_case_classes,
                    "pattern_type": "naming",
                }
            )

        return patterns

    def _analyze_project_health(self, graph, project_overview) -> dict[str, Any]:
        """Analyze overall project health metrics."""
        health_metrics = {}

        # Structure health
        total_nodes = len(graph.nodes)
        total_edges = len(graph.edges)

        # Calculate connectivity density
        max_possible_edges = total_nodes * (total_nodes - 1)
        connectivity_density = total_edges / max_possible_edges if max_possible_edges > 0 else 0

        # Calculate orphaned nodes
        orphaned_nodes = project_overview.get("structure_health", {}).get("orphaned_nodes", 0)
        orphaned_ratio = orphaned_nodes / total_nodes if total_nodes > 0 else 0

        # Calculate depth distribution balance
        depth_distribution = project_overview.get("breakdown", {}).get("by_depth", {})
        depth_variance = statistics.variance(depth_distribution.values()) if len(depth_distribution) > 1 else 0

        # Overall health score
        connectivity_score = min(connectivity_density * 10, 1.0)  # Normalize to 0-1
        structure_score = 1.0 - orphaned_ratio
        balance_score = 1.0 / (1.0 + depth_variance / 100)  # Normalized balance score

        overall_score = (connectivity_score + structure_score + balance_score) / 3

        health_metrics.update(
            {
                "overall_score": overall_score,
                "connectivity_density": connectivity_density,
                "orphaned_ratio": orphaned_ratio,
                "depth_variance": depth_variance,
                "structure_score": structure_score,
                "balance_score": balance_score,
                "total_components": total_nodes,
                "total_relationships": total_edges,
            }
        )

        return health_metrics

    def _assess_structure_quality(self, graph, project_overview) -> dict[str, Any]:
        """Assess the quality of the project structure."""
        quality_metrics = {}

        # Type distribution quality
        type_breakdown = project_overview.get("breakdown", {}).get("by_type", {})
        type_diversity = len(type_breakdown)
        type_balance = (
            1.0 - statistics.variance(type_breakdown.values()) / statistics.mean(type_breakdown.values())
            if len(type_breakdown) > 1
            else 1.0
        )

        # Relationship quality
        relationship_breakdown = project_overview.get("breakdown", {}).get("by_relationship", {})
        relationship_diversity = len(relationship_breakdown)

        # Documentation quality (estimated from docstrings)
        documented_nodes = sum(1 for node in graph.nodes.values() if hasattr(node, "docstring") and node.docstring)
        documentation_ratio = documented_nodes / len(graph.nodes) if graph.nodes else 0

        # Calculate overall quality score
        quality_score = (type_balance + (relationship_diversity / 10) + documentation_ratio) / 3

        quality_metrics.update(
            {
                "score": quality_score,
                "type_diversity": type_diversity,
                "type_balance": type_balance,
                "relationship_diversity": relationship_diversity,
                "documentation_ratio": documentation_ratio,
                "data_quality": min(quality_score + 0.2, 1.0),
                "confidence": quality_score,
            }
        )

        return quality_metrics

    def _calculate_maintainability_score(self, graph, project_overview) -> float:
        """Calculate maintainability score for the project."""
        # Component size distribution
        largest_components = project_overview.get("largest_components", [])
        max_children = max([comp["children_count"] for comp in largest_components], default=0)

        # Depth distribution
        depth_distribution = project_overview.get("breakdown", {}).get("by_depth", {})
        max_depth = max(depth_distribution.keys(), default=0)

        # Calculate maintainability factors
        size_factor = 1.0 - min(max_children / 50, 1.0)  # Penalize large components
        depth_factor = 1.0 - min(max_depth / 10, 1.0)  # Penalize deep nesting

        # Relationship density (moderate density is good)
        structure_health = project_overview.get("structure_health", {})
        density = structure_health.get("relationship_density", 0)
        density_factor = 1.0 - abs(density - 0.3)  # Optimal density around 0.3

        maintainability_score = (size_factor + depth_factor + density_factor) / 3
        return max(0.0, min(1.0, maintainability_score))

    async def _generate_pattern_statistics(self, architectural_patterns, design_patterns, naming_patterns) -> list[StatisticalSummary]:
        """Generate statistics for pattern analysis."""
        statistics = []

        # Pattern confidence distribution
        all_patterns = architectural_patterns + design_patterns + naming_patterns
        if all_patterns:
            confidence_scores = [pattern["confidence"] for pattern in all_patterns]
            statistics.append(
                StatisticalSummary(
                    metric_name="pattern_confidence",
                    count=len(confidence_scores),
                    mean=statistics.mean(confidence_scores),
                    median=statistics.median(confidence_scores),
                    std_dev=statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0,
                    min_value=min(confidence_scores),
                    max_value=max(confidence_scores),
                )
            )

        # Components involved distribution
        if all_patterns:
            component_counts = [pattern["components_involved"] for pattern in all_patterns]
            statistics.append(
                StatisticalSummary(
                    metric_name="pattern_component_involvement",
                    count=len(component_counts),
                    mean=statistics.mean(component_counts),
                    median=statistics.median(component_counts),
                    std_dev=statistics.stdev(component_counts) if len(component_counts) > 1 else 0.0,
                    min_value=min(component_counts),
                    max_value=max(component_counts),
                )
            )

        return statistics

    async def _generate_health_statistics(self, graph, project_overview, health_metrics) -> list[StatisticalSummary]:
        """Generate statistics for health analysis."""
        statistics = []

        # Component distribution by type
        type_breakdown = project_overview.get("breakdown", {}).get("by_type", {})
        if type_breakdown:
            type_counts = list(type_breakdown.values())
            statistics.append(
                StatisticalSummary(
                    metric_name="component_type_distribution",
                    count=len(type_counts),
                    mean=statistics.mean(type_counts),
                    median=statistics.median(type_counts),
                    std_dev=statistics.stdev(type_counts) if len(type_counts) > 1 else 0.0,
                    min_value=min(type_counts),
                    max_value=max(type_counts),
                    distribution=type_breakdown,
                )
            )

        # Depth distribution
        depth_distribution = project_overview.get("breakdown", {}).get("by_depth", {})
        if depth_distribution:
            depth_counts = list(depth_distribution.values())
            statistics.append(
                StatisticalSummary(
                    metric_name="hierarchy_depth_distribution",
                    count=len(depth_counts),
                    mean=statistics.mean(depth_counts),
                    median=statistics.median(depth_counts),
                    std_dev=statistics.stdev(depth_counts) if len(depth_counts) > 1 else 0.0,
                    min_value=min(depth_counts),
                    max_value=max(depth_counts),
                    distribution=depth_distribution,
                )
            )

        return statistics

    def _generate_pattern_insights(self, architectural_patterns, design_patterns, naming_patterns, project_overview) -> list[str]:
        """Generate insights from pattern analysis."""
        insights = []

        total_patterns = len(architectural_patterns) + len(design_patterns) + len(naming_patterns)

        if total_patterns == 0:
            insights.append("No clear patterns detected - consider implementing consistent architectural patterns")
        else:
            insights.append(f"Identified {total_patterns} patterns across architectural, design, and naming categories")

        # Architectural insights
        if architectural_patterns:
            high_confidence_arch = [p for p in architectural_patterns if p["confidence"] > 0.7]
            if high_confidence_arch:
                insights.append(f"Strong architectural patterns detected: {', '.join(p['name'] for p in high_confidence_arch)}")

        # Design pattern insights
        if design_patterns:
            insights.append(f"Design patterns identified: {', '.join(p['name'] for p in design_patterns)}")

        # Naming pattern insights
        if naming_patterns:
            consistent_naming = [p for p in naming_patterns if p["confidence"] > 0.8]
            if consistent_naming:
                insights.append("Consistent naming conventions detected across the project")
            else:
                insights.append("Naming conventions could be more consistent")

        return insights

    def _generate_health_insights(self, health_metrics, structure_quality, maintainability_score, project_overview) -> list[str]:
        """Generate insights from health analysis."""
        insights = []

        overall_score = health_metrics.get("overall_score", 0.0)

        if overall_score > 0.8:
            insights.append("Project demonstrates excellent structural health")
        elif overall_score > 0.6:
            insights.append("Project shows good structural health with room for improvement")
        else:
            insights.append("Project structure needs attention to improve maintainability")

        # Connectivity insights
        connectivity_density = health_metrics.get("connectivity_density", 0.0)
        if connectivity_density > 0.5:
            insights.append("High connectivity density may indicate tight coupling")
        elif connectivity_density < 0.1:
            insights.append("Low connectivity density may indicate isolated components")

        # Orphaned components insight
        orphaned_ratio = health_metrics.get("orphaned_ratio", 0.0)
        if orphaned_ratio > 0.1:
            insights.append(f"Found {orphaned_ratio:.1%} orphaned components that may need attention")

        # Maintainability insight
        if maintainability_score > 0.7:
            insights.append("Code structure supports good maintainability")
        else:
            insights.append("Code structure may benefit from refactoring for better maintainability")

        return insights

    def _generate_pattern_recommendations(
        self, architectural_patterns, design_patterns, naming_patterns, min_confidence
    ) -> list[Recommendation]:
        """Generate recommendations from pattern analysis."""
        recommendations = []

        # Low pattern confidence recommendations
        low_confidence_patterns = []
        for pattern_group in [architectural_patterns, design_patterns, naming_patterns]:
            low_confidence_patterns.extend([p for p in pattern_group if p["confidence"] < min_confidence])

        if low_confidence_patterns:
            recommendations.append(
                Recommendation(
                    title="Inconsistent Pattern Implementation",
                    description=f"Found {len(low_confidence_patterns)} patterns with low confidence scores",
                    severity=SeverityLevel.WARNING,
                    category="patterns",
                    impact="Reduced code consistency and maintainability",
                    suggested_actions=[
                        "Review and standardize pattern implementations",
                        "Add documentation for pattern usage",
                        "Consider refactoring inconsistent implementations",
                    ],
                    confidence=0.8,
                    estimated_effort="medium",
                )
            )

        # Missing architectural patterns
        if not architectural_patterns:
            recommendations.append(
                Recommendation(
                    title="No Clear Architectural Patterns",
                    description="Project lacks identifiable architectural patterns",
                    severity=SeverityLevel.OPTIMIZATION,
                    category="architecture",
                    impact="Potential difficulty in understanding project structure",
                    suggested_actions=[
                        "Implement consistent architectural patterns",
                        "Consider layered or modular architecture",
                        "Document architectural decisions",
                    ],
                    confidence=0.7,
                    estimated_effort="high",
                )
            )

        return recommendations

    def _generate_health_recommendations(self, health_metrics, structure_quality, maintainability_score) -> list[Recommendation]:
        """Generate recommendations from health analysis."""
        recommendations = []

        overall_score = health_metrics.get("overall_score", 0.0)

        # Overall health recommendations
        if overall_score < 0.5:
            recommendations.append(
                Recommendation(
                    title="Poor Project Health",
                    description=f"Project health score is {overall_score:.2f}, indicating structural issues",
                    severity=SeverityLevel.CRITICAL,
                    category="health",
                    impact="Significant impact on maintainability and development velocity",
                    suggested_actions=[
                        "Conduct comprehensive code review",
                        "Refactor problematic components",
                        "Improve component relationships",
                        "Add documentation and tests",
                    ],
                    confidence=0.9,
                    estimated_effort="high",
                )
            )

        # Orphaned components
        orphaned_ratio = health_metrics.get("orphaned_ratio", 0.0)
        if orphaned_ratio > 0.15:
            recommendations.append(
                Recommendation(
                    title="High Number of Orphaned Components",
                    description=f"{orphaned_ratio:.1%} of components are orphaned",
                    severity=SeverityLevel.WARNING,
                    category="structure",
                    impact="Potential dead code and maintenance overhead",
                    suggested_actions=[
                        "Review orphaned components for necessity",
                        "Remove unused components",
                        "Integrate isolated components where appropriate",
                    ],
                    confidence=0.8,
                    estimated_effort="medium",
                )
            )

        # Maintainability
        if maintainability_score < 0.5:
            recommendations.append(
                Recommendation(
                    title="Low Maintainability Score",
                    description=f"Maintainability score is {maintainability_score:.2f}",
                    severity=SeverityLevel.WARNING,
                    category="maintainability",
                    impact="Increased development time and difficulty making changes",
                    suggested_actions=[
                        "Reduce component complexity",
                        "Limit nesting depth",
                        "Improve component relationships",
                        "Add comprehensive documentation",
                    ],
                    confidence=0.8,
                    estimated_effort="high",
                )
            )

        return recommendations

    def _prioritize_recommendations(self, recommendations: list[Recommendation]) -> list[Recommendation]:
        """Prioritize recommendations by severity and impact."""
        # Define severity order
        severity_order = {
            SeverityLevel.CRITICAL: 4,
            SeverityLevel.WARNING: 3,
            SeverityLevel.OPTIMIZATION: 2,
            SeverityLevel.INFO: 1,
        }

        # Sort by severity (descending) and confidence (descending)
        return sorted(recommendations, key=lambda r: (severity_order.get(r.severity, 0), r.confidence), reverse=True)

    def _generate_comprehensive_insights(self, individual_reports: list) -> list[str]:
        """Generate insights from comprehensive analysis."""
        insights = []

        successful_reports = [r for r in individual_reports if isinstance(r, GraphAnalysisReport)]

        if len(successful_reports) > 1:
            insights.append(f"Comprehensive analysis completed across {len(successful_reports)} analysis dimensions")

        # Cross-analysis insights
        all_recommendations = []
        for report in successful_reports:
            all_recommendations.extend(report.recommendations)

        critical_issues = [r for r in all_recommendations if r.severity == SeverityLevel.CRITICAL]
        if critical_issues:
            insights.append(f"Identified {len(critical_issues)} critical issues requiring immediate attention")

        # Pattern consistency insights
        pattern_reports = [r for r in successful_reports if r.report_type == ReportType.PATTERN_ANALYSIS]
        health_reports = [r for r in successful_reports if r.report_type == ReportType.PROJECT_HEALTH]

        if pattern_reports and health_reports:
            insights.append("Cross-analysis reveals correlation between pattern consistency and project health")

        return insights

    def _assess_comprehensive_quality(self, individual_reports: list) -> float:
        """Assess the quality of comprehensive analysis."""
        successful_reports = [r for r in individual_reports if isinstance(r, GraphAnalysisReport)]
        if not successful_reports:
            return 0.0

        avg_confidence = statistics.mean([r.confidence_score for r in successful_reports])
        completeness = len(successful_reports) / 3  # Expecting 3 types of reports

        return (avg_confidence + completeness) / 2

    def _calculate_comprehensive_confidence(self, individual_reports: list) -> float:
        """Calculate confidence for comprehensive analysis."""
        successful_reports = [r for r in individual_reports if isinstance(r, GraphAnalysisReport)]
        if not successful_reports:
            return 0.0

        return statistics.mean([r.confidence_score for r in successful_reports])

    def _extract_structural_metrics(self, hierarchy_data: dict, connectivity_data: dict) -> dict[str, Any]:
        """Extract structural metrics from analysis data."""
        return {
            "hierarchy_metrics": {
                "ancestor_count": len(hierarchy_data.get("ancestors", [])),
                "descendant_count": len(hierarchy_data.get("descendants", [])),
                "sibling_count": len(hierarchy_data.get("siblings", [])),
            },
            "connectivity_metrics": {
                "influence_score": connectivity_data.get("influence_score", 0.0),
                "connection_count": connectivity_data.get("connection_count", 0),
            },
        }

    def _analyze_structure_performance(self, breadcrumb: str, graph, related_components) -> dict[str, Any]:
        """Analyze performance aspects of structure."""
        return {
            "graph_size": len(graph.nodes),
            "traversal_depth": related_components.traversal_depth if related_components else 0,
            "execution_time_ms": related_components.execution_time_ms if related_components else 0.0,
            "related_components_count": len(related_components.related_components) if related_components else 0,
        }

    def _calculate_data_quality_score(self, graph, breadcrumb: str) -> float:
        """Calculate data quality score."""
        node = graph.nodes.get(breadcrumb)
        if not node:
            return 0.0

        quality_factors = []

        # Check if node has complete metadata
        if node.name:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.5)

        if hasattr(node, "docstring") and node.docstring:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.7)

        if node.file_path:
            quality_factors.append(1.0)
        else:
            quality_factors.append(0.3)

        return statistics.mean(quality_factors)

    def _calculate_confidence_score(self, hierarchy_data: dict, connectivity_data: dict) -> float:
        """Calculate confidence score for analysis."""
        confidence_factors = []

        # Hierarchy data confidence
        if hierarchy_data and not hierarchy_data.get("error"):
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.3)

        # Connectivity data confidence
        if connectivity_data and not connectivity_data.get("error"):
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.3)

        return statistics.mean(confidence_factors)

    def _calculate_pattern_quality_score(self, architectural_patterns, design_patterns, naming_patterns) -> float:
        """Calculate pattern quality score."""
        all_patterns = architectural_patterns + design_patterns + naming_patterns
        if not all_patterns:
            return 0.0

        avg_confidence = statistics.mean([p["confidence"] for p in all_patterns])
        pattern_diversity = len({p["pattern_type"] for p in all_patterns})
        diversity_score = min(pattern_diversity / 3, 1.0)  # Max 3 pattern types

        return (avg_confidence + diversity_score) / 2

    def _calculate_pattern_confidence(self, architectural_patterns, design_patterns) -> float:
        """Calculate pattern analysis confidence."""
        all_patterns = architectural_patterns + design_patterns
        if not all_patterns:
            return 0.5  # Neutral confidence when no patterns found

        return statistics.mean([p["confidence"] for p in all_patterns])

    def _update_generation_stats(self, execution_time_ms: float):
        """Update report generation statistics."""
        self._report_generation_stats["reports_generated"] += 1

        current_avg = self._report_generation_stats["avg_generation_time_ms"]
        reports_count = self._report_generation_stats["reports_generated"]

        if reports_count == 1:
            self._report_generation_stats["avg_generation_time_ms"] = execution_time_ms
        else:
            self._report_generation_stats["avg_generation_time_ms"] = (
                current_avg * (reports_count - 1) + execution_time_ms
            ) / reports_count


# Singleton instance for global access
_report_service_instance: GraphAnalysisReportService | None = None


def get_graph_analysis_report_service(graph_rag_service: GraphRAGService | None = None) -> GraphAnalysisReportService:
    """
    Get the global Graph Analysis Report Service instance.

    Args:
        graph_rag_service: GraphRAGService instance (required for first call)

    Returns:
        GraphAnalysisReportService singleton instance
    """
    global _report_service_instance

    if _report_service_instance is None:
        if graph_rag_service is None:
            raise ValueError("graph_rag_service is required for first initialization")
        _report_service_instance = GraphAnalysisReportService(graph_rag_service)

    return _report_service_instance
