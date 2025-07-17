"""
Pattern Recognition Service for Graph RAG enhancement.

This service identifies common architectural patterns, design patterns, and code organization
patterns across codebases using structural analysis and semantic understanding.

Built on top of Wave 2's Graph RAG infrastructure and relationship analysis capabilities.
"""

import asyncio
import logging
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from ..models.code_chunk import ChunkType, CodeChunk
from .cross_project_search_service import CrossProjectSearchFilter
from .graph_rag_service import GraphRAGService
from .structure_relationship_builder import GraphNode, StructureGraph


class PatternType(Enum):
    """Types of patterns that can be recognized."""

    # Design Patterns (Gang of Four)
    SINGLETON = "singleton"
    FACTORY = "factory"
    OBSERVER = "observer"
    DECORATOR = "decorator"
    STRATEGY = "strategy"
    COMMAND = "command"
    ADAPTER = "adapter"
    FACADE = "facade"
    BUILDER = "builder"
    TEMPLATE_METHOD = "template_method"

    # Architectural Patterns
    MVC = "mvc"
    MVP = "mvp"
    MVVM = "mvvm"
    REPOSITORY = "repository"
    SERVICE_LAYER = "service_layer"
    LAYERED_ARCHITECTURE = "layered_architecture"
    MICROSERVICES = "microservices"
    EVENT_DRIVEN = "event_driven"

    # Code Organization Patterns
    MODULE_PATTERN = "module_pattern"
    NAMESPACE_PATTERN = "namespace_pattern"
    PLUGIN_ARCHITECTURE = "plugin_architecture"
    DEPENDENCY_INJECTION = "dependency_injection"
    INVERSION_OF_CONTROL = "inversion_of_control"

    # Data Access Patterns
    ACTIVE_RECORD = "active_record"
    DATA_MAPPER = "data_mapper"
    DAO = "dao"
    ORM_PATTERN = "orm_pattern"

    # Concurrency Patterns
    PRODUCER_CONSUMER = "producer_consumer"
    THREAD_POOL = "thread_pool"
    ACTOR_MODEL = "actor_model"

    # API Patterns
    REST_API = "rest_api"
    GRAPHQL_API = "graphql_api"
    RPC_PATTERN = "rpc_pattern"

    # Testing Patterns
    TEST_FIXTURE = "test_fixture"
    MOCK_OBJECT = "mock_object"
    PAGE_OBJECT = "page_object"

    # Custom/Unknown Patterns
    CUSTOM = "custom"
    UNKNOWN = "unknown"


@dataclass
class PatternSignature:
    """Signature for identifying a specific pattern."""

    pattern_type: PatternType
    required_components: list[str]  # Component types that must be present
    structural_indicators: list[str]  # Structural relationships to look for
    naming_patterns: list[str]  # Naming conventions or keywords
    behavioral_indicators: list[str]  # Behavioral characteristics
    confidence_threshold: float = 0.7  # Minimum confidence to consider a match


@dataclass
class PatternMatch:
    """Represents a detected pattern in the codebase."""

    pattern_type: PatternType
    confidence: float
    components: list[GraphNode]  # Components that form this pattern
    center_component: GraphNode  # Main/central component of the pattern

    # Evidence for the pattern
    structural_evidence: dict[str, Any]
    naming_evidence: list[str]
    behavioral_evidence: list[str]

    # Context information
    project_name: str
    file_paths: list[str]
    breadcrumb_scope: str  # Common breadcrumb prefix for pattern components

    # Metrics
    pattern_complexity: float  # How complex this pattern implementation is
    pattern_completeness: float  # How complete the pattern implementation is
    pattern_quality: float  # Overall quality score for this pattern

    def __post_init__(self):
        """Initialize default values for mutable fields."""
        if not hasattr(self, "file_paths") or self.file_paths is None:
            self.file_paths = []
        if not hasattr(self, "naming_evidence") or self.naming_evidence is None:
            self.naming_evidence = []
        if not hasattr(self, "behavioral_evidence") or self.behavioral_evidence is None:
            self.behavioral_evidence = []
        if not hasattr(self, "structural_evidence") or self.structural_evidence is None:
            self.structural_evidence = {}


@dataclass
class PatternAnalysisResult:
    """Result of pattern analysis for a project or scope."""

    project_name: str
    scope: str  # Project-wide, module-specific, etc.
    patterns_found: list[PatternMatch]

    # Analysis statistics
    total_components_analyzed: int
    analysis_time_ms: float
    coverage_percentage: float  # Percentage of code covered by recognized patterns

    # Pattern distribution
    patterns_by_type: dict[PatternType, int] = None
    patterns_by_confidence: dict[str, int] = None  # High/Medium/Low confidence buckets
    complexity_distribution: dict[str, int] = None

    # Quality metrics
    avg_pattern_quality: float = 0.0
    avg_pattern_completeness: float = 0.0
    pattern_consistency_score: float = 0.0  # How consistent patterns are across the codebase

    def __post_init__(self):
        """Initialize default values and compute statistics."""
        if self.patterns_by_type is None:
            self.patterns_by_type = {}
        if self.patterns_by_confidence is None:
            self.patterns_by_confidence = {}
        if self.complexity_distribution is None:
            self.complexity_distribution = {}

        # Compute statistics from found patterns
        if self.patterns_found:
            # Count patterns by type
            for pattern in self.patterns_found:
                self.patterns_by_type[pattern.pattern_type] = self.patterns_by_type.get(pattern.pattern_type, 0) + 1

                # Categorize by confidence
                if pattern.confidence >= 0.8:
                    confidence_category = "high"
                elif pattern.confidence >= 0.6:
                    confidence_category = "medium"
                else:
                    confidence_category = "low"
                self.patterns_by_confidence[confidence_category] = self.patterns_by_confidence.get(confidence_category, 0) + 1

                # Categorize by complexity
                if pattern.pattern_complexity >= 0.7:
                    complexity_category = "complex"
                elif pattern.pattern_complexity >= 0.4:
                    complexity_category = "moderate"
                else:
                    complexity_category = "simple"
                self.complexity_distribution[complexity_category] = self.complexity_distribution.get(complexity_category, 0) + 1

            # Calculate average metrics
            self.avg_pattern_quality = sum(p.pattern_quality for p in self.patterns_found) / len(self.patterns_found)
            self.avg_pattern_completeness = sum(p.pattern_completeness for p in self.patterns_found) / len(self.patterns_found)


class PatternRecognitionService:
    """
    Service for identifying architectural and design patterns in codebases.

    This service uses structural analysis, naming conventions, and behavioral
    characteristics to identify common patterns and architectural approaches.
    """

    def __init__(self, graph_rag_service: GraphRAGService):
        """Initialize the pattern recognition service.

        Args:
            graph_rag_service: Service for graph operations and structural analysis
        """
        self.graph_rag_service = graph_rag_service
        self.logger = logging.getLogger(__name__)

        # Initialize pattern signatures
        self.pattern_signatures = self._initialize_pattern_signatures()

        # Cache for pattern analysis results
        self._analysis_cache = {}

    async def analyze_project_patterns(
        self,
        project_name: str,
        scope_breadcrumb: str | None = None,
        min_confidence: float = 0.6,
    ) -> PatternAnalysisResult:
        """
        Analyze patterns in a project or specific scope.

        Args:
            project_name: Name of the project to analyze
            scope_breadcrumb: Optional breadcrumb to limit analysis scope
            min_confidence: Minimum confidence threshold for pattern detection

        Returns:
            PatternAnalysisResult with detected patterns and analysis
        """
        start_time = time.time()

        try:
            self.logger.info(f"Starting pattern analysis for project: {project_name}")
            if scope_breadcrumb:
                self.logger.info(f"Analysis scope limited to: {scope_breadcrumb}")

            # Step 1: Get project structure
            project_graph = await self.graph_rag_service.get_project_structure_graph(project_name)

            if not project_graph or not project_graph.nodes:
                self.logger.warning(f"No structure graph found for project: {project_name}")
                return PatternAnalysisResult(
                    project_name=project_name,
                    scope=scope_breadcrumb or "project-wide",
                    patterns_found=[],
                    total_components_analyzed=0,
                    analysis_time_ms=(time.time() - start_time) * 1000,
                    coverage_percentage=0.0,
                )

            # Step 2: Filter nodes by scope if specified
            analysis_nodes = self._filter_nodes_by_scope(project_graph.nodes, scope_breadcrumb)

            # Step 3: Analyze patterns using different strategies
            detected_patterns = []

            # Strategy 1: Structural pattern analysis
            structural_patterns = await self._analyze_structural_patterns(analysis_nodes, project_name, min_confidence)
            detected_patterns.extend(structural_patterns)

            # Strategy 2: Naming convention analysis
            naming_patterns = await self._analyze_naming_patterns(analysis_nodes, project_name, min_confidence)
            detected_patterns.extend(naming_patterns)

            # Strategy 3: Behavioral pattern analysis
            behavioral_patterns = await self._analyze_behavioral_patterns(analysis_nodes, project_name, min_confidence)
            detected_patterns.extend(behavioral_patterns)

            # Strategy 4: Cross-component relationship analysis
            relationship_patterns = await self._analyze_relationship_patterns(analysis_nodes, project_graph, project_name, min_confidence)
            detected_patterns.extend(relationship_patterns)

            # Step 4: Merge and deduplicate patterns
            merged_patterns = self._merge_overlapping_patterns(detected_patterns)

            # Step 5: Calculate coverage and quality metrics
            coverage_percentage = self._calculate_pattern_coverage(analysis_nodes, merged_patterns)
            consistency_score = self._calculate_pattern_consistency(merged_patterns)

            analysis_time_ms = (time.time() - start_time) * 1000

            result = PatternAnalysisResult(
                project_name=project_name,
                scope=scope_breadcrumb or "project-wide",
                patterns_found=merged_patterns,
                total_components_analyzed=len(analysis_nodes),
                analysis_time_ms=analysis_time_ms,
                coverage_percentage=coverage_percentage,
                pattern_consistency_score=consistency_score,
            )

            self.logger.info(
                f"Pattern analysis completed in {analysis_time_ms:.2f}ms. "
                f"Found {len(merged_patterns)} patterns with {coverage_percentage:.1f}% coverage."
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in pattern analysis: {e}")
            analysis_time_ms = (time.time() - start_time) * 1000
            return PatternAnalysisResult(
                project_name=project_name,
                scope=scope_breadcrumb or "project-wide",
                patterns_found=[],
                total_components_analyzed=0,
                analysis_time_ms=analysis_time_ms,
                coverage_percentage=0.0,
            )

    async def find_similar_patterns(
        self,
        reference_pattern: PatternMatch,
        target_projects: list[str],
        similarity_threshold: float = 0.7,
    ) -> list[PatternMatch]:
        """
        Find similar pattern implementations across projects.

        Args:
            reference_pattern: The pattern to find similar implementations of
            target_projects: Projects to search in
            similarity_threshold: Minimum similarity threshold

        Returns:
            List of similar pattern matches
        """
        try:
            self.logger.info(f"Finding similar patterns to {reference_pattern.pattern_type.value}")

            similar_patterns = []

            for project_name in target_projects:
                # Analyze patterns in target project
                project_analysis = await self.analyze_project_patterns(project_name)

                # Find patterns of the same type
                for pattern in project_analysis.patterns_found:
                    if pattern.pattern_type == reference_pattern.pattern_type:
                        similarity = self._calculate_pattern_similarity(reference_pattern, pattern)

                        if similarity >= similarity_threshold:
                            # Add similarity information to the pattern
                            pattern.structural_evidence["similarity_to_reference"] = similarity
                            similar_patterns.append(pattern)

            # Sort by similarity (highest first)
            similar_patterns.sort(key=lambda p: p.structural_evidence.get("similarity_to_reference", 0.0), reverse=True)

            return similar_patterns

        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {e}")
            return []

    async def suggest_pattern_improvements(
        self,
        pattern: PatternMatch,
        reference_projects: list[str],
    ) -> dict[str, Any]:
        """
        Suggest improvements for a pattern implementation based on reference projects.

        Args:
            pattern: The pattern to suggest improvements for
            reference_projects: Projects to use as references for best practices

        Returns:
            Dictionary with improvement suggestions
        """
        try:
            improvements = {
                "pattern_type": pattern.pattern_type.value,
                "current_quality": pattern.pattern_quality,
                "suggestions": [],
                "reference_examples": [],
            }

            # Find high-quality examples of the same pattern type
            high_quality_examples = []
            for project_name in reference_projects:
                project_analysis = await self.analyze_project_patterns(project_name)

                for ref_pattern in project_analysis.patterns_found:
                    if ref_pattern.pattern_type == pattern.pattern_type and ref_pattern.pattern_quality > pattern.pattern_quality + 0.1:
                        high_quality_examples.append(ref_pattern)

            # Analyze differences and generate suggestions
            if high_quality_examples:
                # Sort by quality (highest first)
                high_quality_examples.sort(key=lambda p: p.pattern_quality, reverse=True)
                improvements["reference_examples"] = high_quality_examples[:3]  # Top 3 examples

                # Generate specific suggestions based on differences
                for ref_example in high_quality_examples[:3]:
                    suggestions = self._generate_improvement_suggestions(pattern, ref_example)
                    improvements["suggestions"].extend(suggestions)

            # Remove duplicate suggestions
            improvements["suggestions"] = list(set(improvements["suggestions"]))

            return improvements

        except Exception as e:
            self.logger.error(f"Error generating pattern improvement suggestions: {e}")
            return {"pattern_type": pattern.pattern_type.value, "suggestions": [], "reference_examples": []}

    def _initialize_pattern_signatures(self) -> dict[PatternType, PatternSignature]:
        """Initialize pattern signatures for detection."""
        signatures = {}

        # Singleton Pattern
        signatures[PatternType.SINGLETON] = PatternSignature(
            pattern_type=PatternType.SINGLETON,
            required_components=["class"],
            structural_indicators=["private_constructor", "static_instance"],
            naming_patterns=["singleton", "instance", "get_instance"],
            behavioral_indicators=["single_instance", "global_access"],
        )

        # Factory Pattern
        signatures[PatternType.FACTORY] = PatternSignature(
            pattern_type=PatternType.FACTORY,
            required_components=["class", "method"],
            structural_indicators=["factory_method", "product_creation"],
            naming_patterns=["factory", "create", "build", "make"],
            behavioral_indicators=["object_creation", "type_selection"],
        )

        # Observer Pattern
        signatures[PatternType.OBSERVER] = PatternSignature(
            pattern_type=PatternType.OBSERVER,
            required_components=["class", "interface"],
            structural_indicators=["observer_list", "notification_method"],
            naming_patterns=["observer", "listener", "subscriber", "notify", "update"],
            behavioral_indicators=["event_notification", "subscription"],
        )

        # MVC Pattern
        signatures[PatternType.MVC] = PatternSignature(
            pattern_type=PatternType.MVC,
            required_components=["class", "class", "class"],  # Model, View, Controller
            structural_indicators=["model_view_separation", "controller_mediation"],
            naming_patterns=["model", "view", "controller", "mvc"],
            behavioral_indicators=["user_input_handling", "data_presentation"],
        )

        # Repository Pattern
        signatures[PatternType.REPOSITORY] = PatternSignature(
            pattern_type=PatternType.REPOSITORY,
            required_components=["class", "interface"],
            structural_indicators=["data_access_abstraction", "query_methods"],
            naming_patterns=["repository", "dao", "data_access", "store"],
            behavioral_indicators=["data_persistence", "query_encapsulation"],
        )

        # Service Layer Pattern
        signatures[PatternType.SERVICE_LAYER] = PatternSignature(
            pattern_type=PatternType.SERVICE_LAYER,
            required_components=["class"],
            structural_indicators=["business_logic_layer", "transaction_boundary"],
            naming_patterns=["service", "manager", "handler", "processor"],
            behavioral_indicators=["business_operations", "transaction_management"],
        )

        # Decorator Pattern
        signatures[PatternType.DECORATOR] = PatternSignature(
            pattern_type=PatternType.DECORATOR,
            required_components=["class", "interface"],
            structural_indicators=["wrapper_composition", "behavior_extension"],
            naming_patterns=["decorator", "wrapper", "enhance", "extend"],
            behavioral_indicators=["behavior_modification", "composition_over_inheritance"],
        )

        # Strategy Pattern
        signatures[PatternType.STRATEGY] = PatternSignature(
            pattern_type=PatternType.STRATEGY,
            required_components=["interface", "class"],
            structural_indicators=["algorithm_family", "runtime_selection"],
            naming_patterns=["strategy", "algorithm", "policy", "behavior"],
            behavioral_indicators=["algorithm_switching", "behavior_selection"],
        )

        # Add more patterns as needed...

        return signatures

    def _filter_nodes_by_scope(self, nodes: list[GraphNode], scope_breadcrumb: str | None) -> list[GraphNode]:
        """Filter nodes by scope breadcrumb."""
        if not scope_breadcrumb:
            return nodes

        filtered_nodes = []
        for node in nodes:
            if node.breadcrumb and node.breadcrumb.startswith(scope_breadcrumb):
                filtered_nodes.append(node)

        return filtered_nodes

    async def _analyze_structural_patterns(
        self,
        nodes: list[GraphNode],
        project_name: str,
        min_confidence: float,
    ) -> list[PatternMatch]:
        """Analyze structural patterns in the code."""
        patterns = []

        try:
            # Group nodes by their structural characteristics
            class_nodes = [n for n in nodes if n.chunk_type == ChunkType.CLASS]
            interface_nodes = [n for n in nodes if n.chunk_type == ChunkType.INTERFACE]
            method_nodes = [n for n in nodes if n.chunk_type in [ChunkType.METHOD, ChunkType.FUNCTION]]

            # Look for structural pattern indicators
            for signature in self.pattern_signatures.values():
                if signature.confidence_threshold > min_confidence:
                    continue

                # Check if required components are present
                pattern_candidates = self._find_structural_candidates(signature, class_nodes, interface_nodes, method_nodes)

                for candidate in pattern_candidates:
                    confidence = self._calculate_structural_confidence(candidate, signature)

                    if confidence >= min_confidence:
                        pattern = PatternMatch(
                            pattern_type=signature.pattern_type,
                            confidence=confidence,
                            components=candidate["components"],
                            center_component=candidate["center"],
                            structural_evidence=candidate["evidence"],
                            naming_evidence=[],
                            behavioral_evidence=[],
                            project_name=project_name,
                            breadcrumb_scope=candidate.get("scope", ""),
                            pattern_complexity=self._calculate_pattern_complexity(candidate),
                            pattern_completeness=self._calculate_pattern_completeness(candidate, signature),
                            pattern_quality=0.0,  # Will be calculated later
                        )

                        # Calculate overall quality
                        pattern.pattern_quality = (
                            pattern.confidence * 0.4
                            + pattern.pattern_completeness * 0.4
                            + (1.0 - pattern.pattern_complexity) * 0.2  # Lower complexity = higher quality
                        )

                        patterns.append(pattern)

            return patterns

        except Exception as e:
            self.logger.error(f"Error in structural pattern analysis: {e}")
            return []

    async def _analyze_naming_patterns(
        self,
        nodes: list[GraphNode],
        project_name: str,
        min_confidence: float,
    ) -> list[PatternMatch]:
        """Analyze naming convention patterns."""
        patterns = []

        try:
            for signature in self.pattern_signatures.values():
                if signature.confidence_threshold > min_confidence:
                    continue

                naming_matches = []

                for node in nodes:
                    # Check node name against pattern naming conventions
                    node_name = node.name.lower() if node.name else ""
                    breadcrumb_lower = node.breadcrumb.lower() if node.breadcrumb else ""

                    matches_count = 0
                    matched_patterns = []

                    for naming_pattern in signature.naming_patterns:
                        pattern_lower = naming_pattern.lower()

                        if (
                            pattern_lower in node_name
                            or pattern_lower in breadcrumb_lower
                            or re.search(rf"\b{re.escape(pattern_lower)}\b", node_name)
                            or re.search(rf"\b{re.escape(pattern_lower)}\b", breadcrumb_lower)
                        ):
                            matches_count += 1
                            matched_patterns.append(naming_pattern)

                    if matches_count > 0:
                        naming_matches.append(
                            {
                                "node": node,
                                "matches": matches_count,
                                "patterns": matched_patterns,
                                "confidence": matches_count / len(signature.naming_patterns),
                            }
                        )

                # Group related naming matches into pattern candidates
                if naming_matches:
                    # Sort by confidence and group by breadcrumb scope
                    naming_matches.sort(key=lambda m: m["confidence"], reverse=True)

                    grouped_matches = defaultdict(list)
                    for match in naming_matches:
                        # Group by breadcrumb prefix (up to 2 levels)
                        breadcrumb_parts = match["node"].breadcrumb.split(".") if match["node"].breadcrumb else [""]
                        scope = ".".join(breadcrumb_parts[:2]) if len(breadcrumb_parts) > 1 else breadcrumb_parts[0]
                        grouped_matches[scope].append(match)

                    # Create pattern matches for each group
                    for scope, group_matches in grouped_matches.items():
                        if len(group_matches) >= 1:  # At least one strong naming match
                            avg_confidence = sum(m["confidence"] for m in group_matches) / len(group_matches)

                            if avg_confidence >= min_confidence:
                                # Find the most central/important node as center
                                center_node = max(group_matches, key=lambda m: m["confidence"])["node"]

                                pattern = PatternMatch(
                                    pattern_type=signature.pattern_type,
                                    confidence=avg_confidence,
                                    components=[m["node"] for m in group_matches],
                                    center_component=center_node,
                                    structural_evidence={"naming_based": True},
                                    naming_evidence=[p for m in group_matches for p in m["patterns"]],
                                    behavioral_evidence=[],
                                    project_name=project_name,
                                    breadcrumb_scope=scope,
                                    pattern_complexity=0.5,  # Default complexity for naming-based patterns
                                    pattern_completeness=avg_confidence,
                                    pattern_quality=avg_confidence * 0.8,  # Naming patterns have moderate quality
                                )

                                patterns.append(pattern)

            return patterns

        except Exception as e:
            self.logger.error(f"Error in naming pattern analysis: {e}")
            return []

    async def _analyze_behavioral_patterns(
        self,
        nodes: list[GraphNode],
        project_name: str,
        min_confidence: float,
    ) -> list[PatternMatch]:
        """Analyze behavioral patterns based on method signatures and interactions."""
        patterns = []

        try:
            # This is a simplified behavioral analysis
            # In a full implementation, you would analyze method signatures,
            # parameter types, return types, and interaction patterns

            method_nodes = [n for n in nodes if n.chunk_type in [ChunkType.METHOD, ChunkType.FUNCTION]]

            for signature in self.pattern_signatures.values():
                if signature.confidence_threshold > min_confidence:
                    continue

                behavioral_indicators = []

                for node in method_nodes:
                    node_name = node.name.lower() if node.name else ""

                    # Look for behavioral indicators in method names
                    matches = 0
                    for behavior in signature.behavioral_indicators:
                        behavior_lower = behavior.lower().replace("_", "")
                        if behavior_lower in node_name.replace("_", ""):
                            matches += 1

                    if matches > 0:
                        behavioral_indicators.append(
                            {
                                "node": node,
                                "behavior_score": matches / len(signature.behavioral_indicators),
                            }
                        )

                # Create pattern if we have sufficient behavioral evidence
                if behavioral_indicators:
                    avg_behavior_score = sum(bi["behavior_score"] for bi in behavioral_indicators) / len(behavioral_indicators)

                    if avg_behavior_score >= min_confidence:
                        # Find center component
                        center_node = max(behavioral_indicators, key=lambda bi: bi["behavior_score"])["node"]

                        pattern = PatternMatch(
                            pattern_type=signature.pattern_type,
                            confidence=avg_behavior_score,
                            components=[bi["node"] for bi in behavioral_indicators],
                            center_component=center_node,
                            structural_evidence={"behavioral_based": True},
                            naming_evidence=[],
                            behavioral_evidence=signature.behavioral_indicators,
                            project_name=project_name,
                            breadcrumb_scope=center_node.breadcrumb or "",
                            pattern_complexity=0.6,  # Default complexity for behavioral patterns
                            pattern_completeness=avg_behavior_score,
                            pattern_quality=avg_behavior_score * 0.7,  # Behavioral patterns have good quality
                        )

                        patterns.append(pattern)

            return patterns

        except Exception as e:
            self.logger.error(f"Error in behavioral pattern analysis: {e}")
            return []

    async def _analyze_relationship_patterns(
        self,
        nodes: list[GraphNode],
        project_graph: StructureGraph,
        project_name: str,
        min_confidence: float,
    ) -> list[PatternMatch]:
        """Analyze patterns based on component relationships."""
        patterns = []

        try:
            # Analyze relationships between components
            # Look for common architectural relationship patterns

            # Pattern 1: Layered architecture (components at different depths with clear dependencies)
            layered_evidence = self._detect_layered_architecture(nodes, project_graph)
            if layered_evidence["confidence"] >= min_confidence:
                pattern = PatternMatch(
                    pattern_type=PatternType.LAYERED_ARCHITECTURE,
                    confidence=layered_evidence["confidence"],
                    components=layered_evidence["components"],
                    center_component=layered_evidence["center"],
                    structural_evidence=layered_evidence,
                    naming_evidence=[],
                    behavioral_evidence=["layered_dependencies", "separation_of_concerns"],
                    project_name=project_name,
                    breadcrumb_scope="",
                    pattern_complexity=layered_evidence.get("complexity", 0.7),
                    pattern_completeness=layered_evidence.get("completeness", 0.8),
                    pattern_quality=layered_evidence["confidence"] * 0.9,
                )
                patterns.append(pattern)

            # Pattern 2: Module pattern (clear module boundaries)
            module_evidence = self._detect_module_pattern(nodes, project_graph)
            if module_evidence["confidence"] >= min_confidence:
                pattern = PatternMatch(
                    pattern_type=PatternType.MODULE_PATTERN,
                    confidence=module_evidence["confidence"],
                    components=module_evidence["components"],
                    center_component=module_evidence["center"],
                    structural_evidence=module_evidence,
                    naming_evidence=[],
                    behavioral_evidence=["module_encapsulation", "clear_interfaces"],
                    project_name=project_name,
                    breadcrumb_scope="",
                    pattern_complexity=module_evidence.get("complexity", 0.5),
                    pattern_completeness=module_evidence.get("completeness", 0.7),
                    pattern_quality=module_evidence["confidence"] * 0.8,
                )
                patterns.append(pattern)

            return patterns

        except Exception as e:
            self.logger.error(f"Error in relationship pattern analysis: {e}")
            return []

    def _detect_layered_architecture(self, nodes: list[GraphNode], graph: StructureGraph) -> dict[str, Any]:
        """Detect layered architecture pattern."""
        try:
            # Group nodes by depth (breadcrumb depth)
            depth_groups = defaultdict(list)
            for node in nodes:
                depth_groups[node.depth].append(node)

            # Check if we have multiple layers with reasonable distribution
            if len(depth_groups) < 3:  # Need at least 3 layers
                return {"confidence": 0.0, "components": [], "center": None}

            # Analyze dependencies between layers
            layer_dependencies = 0
            total_possible_deps = 0

            for depth in sorted(depth_groups.keys()):
                current_layer = depth_groups[depth]
                for higher_depth in [d for d in depth_groups.keys() if d > depth]:
                    higher_layer = depth_groups[higher_depth]

                    # Check for dependencies from higher to lower layers
                    for higher_node in higher_layer:
                        for current_node in current_layer:
                            total_possible_deps += 1
                            # In a real implementation, check actual dependencies from graph
                            # For now, assume some dependencies exist based on naming
                            if self._nodes_likely_related(higher_node, current_node):
                                layer_dependencies += 1

            dependency_ratio = layer_dependencies / max(total_possible_deps, 1)

            # Calculate confidence based on layer structure and dependencies
            layer_balance = 1.0 - abs(0.5 - (len(depth_groups[1]) / len(nodes))) if 1 in depth_groups else 0.5
            confidence = dependency_ratio * 0.6 + layer_balance * 0.4

            if confidence > 0.0:
                # Select representative components from each layer
                representative_components = []
                for depth in sorted(depth_groups.keys())[:3]:  # Top 3 layers
                    if depth_groups[depth]:
                        representative_components.append(depth_groups[depth][0])

                center = representative_components[len(representative_components) // 2] if representative_components else None

                return {
                    "confidence": confidence,
                    "components": representative_components,
                    "center": center,
                    "layers": len(depth_groups),
                    "dependency_ratio": dependency_ratio,
                    "complexity": min(1.0, len(depth_groups) / 10.0),
                    "completeness": confidence,
                }

            return {"confidence": 0.0, "components": [], "center": None}

        except Exception as e:
            self.logger.error(f"Error detecting layered architecture: {e}")
            return {"confidence": 0.0, "components": [], "center": None}

    def _detect_module_pattern(self, nodes: list[GraphNode], graph: StructureGraph) -> dict[str, Any]:
        """Detect module pattern based on breadcrumb organization."""
        try:
            # Group nodes by module (first part of breadcrumb)
            module_groups = defaultdict(list)
            for node in nodes:
                if node.breadcrumb:
                    module_name = node.breadcrumb.split(".")[0]
                    module_groups[module_name].append(node)

            # Check module organization quality
            if len(module_groups) < 2:  # Need at least 2 modules
                return {"confidence": 0.0, "components": [], "center": None}

            # Analyze module balance and organization
            module_sizes = [len(group) for group in module_groups.values()]
            avg_module_size = sum(module_sizes) / len(module_sizes)
            size_variance = sum((size - avg_module_size) ** 2 for size in module_sizes) / len(module_sizes)

            # Lower variance indicates better module balance
            balance_score = 1.0 / (1.0 + size_variance / max(avg_module_size, 1))

            # Check for clear module separation (different chunk types in modules)
            separation_score = 0.0
            for module_nodes in module_groups.values():
                chunk_types = {node.chunk_type for node in module_nodes}
                if len(chunk_types) > 1:  # Module has different types of components
                    separation_score += 1.0

            separation_score /= len(module_groups)

            confidence = balance_score * 0.5 + separation_score * 0.5

            if confidence > 0.0:
                # Select representative components from modules
                representative_components = []
                for module_nodes in list(module_groups.values())[:5]:  # Top 5 modules
                    if module_nodes:
                        # Select the most "central" component (class if available)
                        class_nodes = [n for n in module_nodes if n.chunk_type == ChunkType.CLASS]
                        if class_nodes:
                            representative_components.append(class_nodes[0])
                        else:
                            representative_components.append(module_nodes[0])

                center = representative_components[0] if representative_components else None

                return {
                    "confidence": confidence,
                    "components": representative_components,
                    "center": center,
                    "modules": len(module_groups),
                    "balance_score": balance_score,
                    "separation_score": separation_score,
                    "complexity": min(1.0, len(module_groups) / 20.0),
                    "completeness": confidence,
                }

            return {"confidence": 0.0, "components": [], "center": None}

        except Exception as e:
            self.logger.error(f"Error detecting module pattern: {e}")
            return {"confidence": 0.0, "components": [], "center": None}

    def _nodes_likely_related(self, node1: GraphNode, node2: GraphNode) -> bool:
        """Check if two nodes are likely related based on naming and structure."""
        try:
            # Check breadcrumb relationship
            if node1.breadcrumb and node2.breadcrumb:
                breadcrumb1_parts = node1.breadcrumb.split(".")
                breadcrumb2_parts = node2.breadcrumb.split(".")

                # Check if they share common prefixes
                common_parts = 0
                for i in range(min(len(breadcrumb1_parts), len(breadcrumb2_parts))):
                    if breadcrumb1_parts[i] == breadcrumb2_parts[i]:
                        common_parts += 1
                    else:
                        break

                if common_parts >= 1:  # Share at least one breadcrumb level
                    return True

            # Check naming similarity
            if node1.name and node2.name:
                name1_lower = node1.name.lower()
                name2_lower = node2.name.lower()

                # Check for common word roots
                if any(word in name2_lower for word in name1_lower.split("_")) or any(
                    word in name1_lower for word in name2_lower.split("_")
                ):
                    return True

            return False

        except Exception:
            return False

    def _find_structural_candidates(
        self,
        signature: PatternSignature,
        class_nodes: list[GraphNode],
        interface_nodes: list[GraphNode],
        method_nodes: list[GraphNode],
    ) -> list[dict[str, Any]]:
        """Find structural candidates for a pattern signature."""
        candidates = []

        try:
            # Simple candidate detection based on required components
            required_classes = signature.required_components.count("class")
            required_interfaces = signature.required_components.count("interface")
            required_methods = signature.required_components.count("method")

            # Find groups of components that could form the pattern
            if required_classes > 0 and len(class_nodes) >= required_classes:
                for i, class_node in enumerate(class_nodes):
                    candidate_components = [class_node]

                    # Add related interfaces if needed
                    if required_interfaces > 0:
                        related_interfaces = [iface for iface in interface_nodes if self._nodes_likely_related(class_node, iface)][
                            :required_interfaces
                        ]
                        candidate_components.extend(related_interfaces)

                    # Add related methods if needed
                    if required_methods > 0:
                        related_methods = [method for method in method_nodes if self._nodes_likely_related(class_node, method)][
                            :required_methods
                        ]
                        candidate_components.extend(related_methods)

                    if len(candidate_components) >= len(signature.required_components):
                        candidates.append(
                            {
                                "center": class_node,
                                "components": candidate_components,
                                "evidence": {"structural_match": True},
                                "scope": class_node.breadcrumb or "",
                            }
                        )

            return candidates

        except Exception as e:
            self.logger.error(f"Error finding structural candidates: {e}")
            return []

    def _calculate_structural_confidence(self, candidate: dict[str, Any], signature: PatternSignature) -> float:
        """Calculate confidence for a structural pattern candidate."""
        try:
            confidence_factors = []

            # Factor 1: Component type match
            required_types = signature.required_components
            actual_types = [comp.chunk_type.value for comp in candidate["components"]]

            type_matches = 0
            for req_type in required_types:
                if req_type in actual_types:
                    type_matches += 1

            type_match_ratio = type_matches / len(required_types) if required_types else 0.0
            confidence_factors.append(type_match_ratio)

            # Factor 2: Structural indicator presence
            # This would check for specific structural patterns in the code
            # For now, use a default based on component organization
            organization_score = 0.7  # Default organization score
            confidence_factors.append(organization_score)

            # Factor 3: Naming alignment
            naming_alignment = 0.0
            for component in candidate["components"]:
                if component.name:
                    component_name_lower = component.name.lower()
                    for naming_pattern in signature.naming_patterns:
                        if naming_pattern.lower() in component_name_lower:
                            naming_alignment += 1.0
                            break

            naming_alignment /= len(candidate["components"])
            confidence_factors.append(naming_alignment)

            # Calculate weighted average
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating structural confidence: {e}")
            return 0.0

    def _calculate_pattern_complexity(self, candidate: dict[str, Any]) -> float:
        """Calculate complexity score for a pattern candidate."""
        try:
            # Complexity based on number of components and their relationships
            num_components = len(candidate["components"])

            # More components = higher complexity
            component_complexity = min(1.0, num_components / 10.0)

            # Depth-based complexity
            depths = [comp.depth for comp in candidate["components"] if hasattr(comp, "depth")]
            depth_variance = 0.0
            if depths:
                avg_depth = sum(depths) / len(depths)
                depth_variance = sum((d - avg_depth) ** 2 for d in depths) / len(depths)

            depth_complexity = min(1.0, depth_variance / 10.0)

            return component_complexity * 0.7 + depth_complexity * 0.3

        except Exception as e:
            self.logger.error(f"Error calculating pattern complexity: {e}")
            return 0.5

    def _calculate_pattern_completeness(self, candidate: dict[str, Any], signature: PatternSignature) -> float:
        """Calculate how complete a pattern implementation is."""
        try:
            completeness_factors = []

            # Factor 1: All required components present
            required_count = len(signature.required_components)
            actual_count = len(candidate["components"])
            component_completeness = min(1.0, actual_count / required_count) if required_count > 0 else 1.0
            completeness_factors.append(component_completeness)

            # Factor 2: Structural indicators present
            # This would check for specific structural completeness
            # For now, use a default based on evidence
            structural_completeness = 0.8 if candidate.get("evidence", {}).get("structural_match") else 0.5
            completeness_factors.append(structural_completeness)

            # Factor 3: Naming convention adherence
            naming_completeness = 0.0
            for component in candidate["components"]:
                if component.name:
                    for pattern in signature.naming_patterns:
                        if pattern.lower() in component.name.lower():
                            naming_completeness += 1.0
                            break
            naming_completeness /= len(candidate["components"]) if candidate["components"] else 1.0
            completeness_factors.append(naming_completeness)

            return sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating pattern completeness: {e}")
            return 0.0

    def _merge_overlapping_patterns(self, patterns: list[PatternMatch]) -> list[PatternMatch]:
        """Merge overlapping pattern matches."""
        try:
            if not patterns:
                return []

            # Sort patterns by confidence (highest first)
            sorted_patterns = sorted(patterns, key=lambda p: p.confidence, reverse=True)

            merged_patterns = []
            used_components = set()

            for pattern in sorted_patterns:
                # Check if this pattern overlaps significantly with already selected patterns
                pattern_component_ids = {comp.chunk_id for comp in pattern.components}
                overlap = len(pattern_component_ids & used_components)
                overlap_ratio = overlap / len(pattern_component_ids) if pattern_component_ids else 0.0

                # If overlap is less than 50%, include this pattern
                if overlap_ratio < 0.5:
                    merged_patterns.append(pattern)
                    used_components.update(pattern_component_ids)

            return merged_patterns

        except Exception as e:
            self.logger.error(f"Error merging overlapping patterns: {e}")
            return patterns

    def _calculate_pattern_coverage(self, nodes: list[GraphNode], patterns: list[PatternMatch]) -> float:
        """Calculate what percentage of code is covered by recognized patterns."""
        try:
            if not nodes:
                return 0.0

            covered_nodes = set()
            for pattern in patterns:
                for component in pattern.components:
                    covered_nodes.add(component.chunk_id)

            total_nodes = {node.chunk_id for node in nodes}
            coverage = len(covered_nodes & total_nodes) / len(total_nodes)

            return coverage * 100.0  # Return as percentage

        except Exception as e:
            self.logger.error(f"Error calculating pattern coverage: {e}")
            return 0.0

    def _calculate_pattern_consistency(self, patterns: list[PatternMatch]) -> float:
        """Calculate consistency score across detected patterns."""
        try:
            if not patterns:
                return 0.0

            # Group patterns by type
            pattern_groups = defaultdict(list)
            for pattern in patterns:
                pattern_groups[pattern.pattern_type].append(pattern)

            # Calculate consistency within each pattern type
            consistency_scores = []

            for pattern_type, type_patterns in pattern_groups.items():
                if len(type_patterns) > 1:
                    # Calculate variance in quality scores
                    qualities = [p.pattern_quality for p in type_patterns]
                    avg_quality = sum(qualities) / len(qualities)
                    variance = sum((q - avg_quality) ** 2 for q in qualities) / len(qualities)

                    # Lower variance = higher consistency
                    consistency = 1.0 / (1.0 + variance)
                    consistency_scores.append(consistency)
                else:
                    # Single pattern of this type = perfect consistency
                    consistency_scores.append(1.0)

            return sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating pattern consistency: {e}")
            return 0.0

    def _calculate_pattern_similarity(self, pattern1: PatternMatch, pattern2: PatternMatch) -> float:
        """Calculate similarity between two patterns."""
        try:
            if pattern1.pattern_type != pattern2.pattern_type:
                return 0.0

            similarity_factors = []

            # Factor 1: Structural similarity
            struct1 = pattern1.structural_evidence
            struct2 = pattern2.structural_evidence

            common_keys = set(struct1.keys()) & set(struct2.keys())
            structural_similarity = len(common_keys) / max(len(struct1), len(struct2), 1)
            similarity_factors.append(structural_similarity)

            # Factor 2: Component count similarity
            count1 = len(pattern1.components)
            count2 = len(pattern2.components)
            count_similarity = 1.0 - abs(count1 - count2) / max(count1, count2, 1)
            similarity_factors.append(count_similarity)

            # Factor 3: Quality similarity
            quality_similarity = 1.0 - abs(pattern1.pattern_quality - pattern2.pattern_quality)
            similarity_factors.append(quality_similarity)

            # Factor 4: Naming similarity
            naming1 = set(pattern1.naming_evidence)
            naming2 = set(pattern2.naming_evidence)
            if naming1 or naming2:
                naming_overlap = len(naming1 & naming2) / len(naming1 | naming2)
                similarity_factors.append(naming_overlap)

            return sum(similarity_factors) / len(similarity_factors) if similarity_factors else 0.0

        except Exception as e:
            self.logger.error(f"Error calculating pattern similarity: {e}")
            return 0.0

    def _generate_improvement_suggestions(self, pattern: PatternMatch, reference: PatternMatch) -> list[str]:
        """Generate improvement suggestions by comparing patterns."""
        suggestions = []

        try:
            # Compare completeness
            if reference.pattern_completeness > pattern.pattern_completeness + 0.1:
                suggestions.append(
                    f"Consider implementing missing components to improve pattern completeness "
                    f"(reference: {reference.pattern_completeness:.2f} vs current: {pattern.pattern_completeness:.2f})"
                )

            # Compare quality
            if reference.pattern_quality > pattern.pattern_quality + 0.1:
                suggestions.append(
                    f"Overall pattern quality can be improved "
                    f"(reference: {reference.pattern_quality:.2f} vs current: {pattern.pattern_quality:.2f})"
                )

            # Compare naming conventions
            ref_naming = set(reference.naming_evidence)
            current_naming = set(pattern.naming_evidence)
            missing_naming = ref_naming - current_naming

            if missing_naming:
                suggestions.append(f"Consider adopting naming conventions: {', '.join(list(missing_naming)[:3])}")

            # Compare component count
            if len(reference.components) > len(pattern.components):
                suggestions.append(
                    f"Pattern might benefit from additional components "
                    f"(reference has {len(reference.components)} vs {len(pattern.components)})"
                )

            # Compare structural evidence
            ref_structural = set(reference.structural_evidence.keys())
            current_structural = set(pattern.structural_evidence.keys())
            missing_structural = ref_structural - current_structural

            if missing_structural:
                suggestions.append(f"Consider implementing structural aspects: {', '.join(list(missing_structural)[:2])}")

            return suggestions

        except Exception as e:
            self.logger.error(f"Error generating improvement suggestions: {e}")
            return ["Unable to generate specific suggestions due to analysis error"]


# Factory function for dependency injection
_pattern_recognition_service_instance = None


def get_pattern_recognition_service(graph_rag_service: GraphRAGService = None) -> PatternRecognitionService:
    """
    Get or create a PatternRecognitionService instance.

    Args:
        graph_rag_service: Graph RAG service instance (optional, will be created if not provided)

    Returns:
        PatternRecognitionService instance
    """
    global _pattern_recognition_service_instance

    if _pattern_recognition_service_instance is None:
        if graph_rag_service is None:
            from .graph_rag_service import get_graph_rag_service

            graph_rag_service = get_graph_rag_service()

        _pattern_recognition_service_instance = PatternRecognitionService(graph_rag_service=graph_rag_service)

    return _pattern_recognition_service_instance
