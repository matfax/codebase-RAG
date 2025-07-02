"""
Unit tests for Enhanced Project Exploration Service

Tests the RAG-enhanced project exploration capabilities.
"""

import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from services.project_exploration_service import (
    ProjectExplorationResult,
    ProjectExplorationService,
)


class TestEnhancedProjectExplorationService(unittest.TestCase):
    """Test cases for enhanced Project Exploration Service with RAG integration."""

    def setUp(self):
        """Set up test fixtures."""
        # Create service with RAG enabled
        self.service_with_rag = ProjectExplorationService(enable_rag=True)

        # Create service with RAG disabled for comparison
        self.service_without_rag = ProjectExplorationService(enable_rag=False)

        # Mock RAG results
        self.mock_rag_results = {
            "architecture_analysis": {
                "primary_pattern": {
                    "pattern_name": "MVC",
                    "confidence": 0.85,
                    "evidence_count": 5,
                },
                "all_patterns": {"mvc": {"pattern_name": "MVC", "confidence": 0.85, "detected": True}},
            },
            "entry_points": {
                "primary_entry_points": [
                    {
                        "file_path": "src/main.py",
                        "function_name": "main",
                        "function_signature": "main()",
                        "confidence": 0.95,
                        "entry_type": "Main Functions",
                        "is_executable": True,
                        "framework_detected": "Flask",
                    },
                    {
                        "file_path": "src/cli.py",
                        "function_name": "cli_main",
                        "function_signature": "cli_main(args)",
                        "confidence": 0.80,
                        "entry_type": "CLI Entry Points",
                        "has_cli_interface": True,
                        "framework_detected": "Click",
                    },
                ]
            },
            "component_relationships": {
                "core_components": [
                    {
                        "name": "UserService",
                        "file_path": "src/services/user_service.py",
                        "chunk_type": "class",
                        "importance_score": 0.9,
                    },
                    {
                        "name": "UserRepository",
                        "file_path": "src/repositories/user_repository.py",
                        "chunk_type": "class",
                        "importance_score": 0.8,
                    },
                ],
                "relationships": [
                    {
                        "component1": {
                            "name": "UserService",
                            "file_path": "src/services/user_service.py",
                        },
                        "component2": {
                            "name": "UserRepository",
                            "file_path": "src/repositories/user_repository.py",
                        },
                        "similarity_score": 0.85,
                        "relationship_type": "service_repository",
                        "relationship_strength": "strong",
                    }
                ],
                "architectural_insights": [
                    "Service-Repository pattern detected in component relationships",
                    "Found 2 strong component relationships indicating tight coupling",
                ],
                "analysis_summary": {
                    "components_analyzed": 2,
                    "relationships_found": 1,
                    "analysis_method": "Vector similarity with clustering analysis",
                },
            },
            "analysis_duration": 1.5,
        }

    def test_initialization_with_rag_enabled(self):
        """Test service initialization with RAG enabled."""
        self.assertTrue(self.service_with_rag.enable_rag)
        self.assertIsNotNone(self.service_with_rag.rag_search_strategy)

    def test_initialization_with_rag_disabled(self):
        """Test service initialization with RAG disabled."""
        self.assertFalse(self.service_without_rag.enable_rag)
        self.assertIsNone(self.service_without_rag.rag_search_strategy)

    @patch.object(ProjectExplorationService, "_perform_basic_analysis")
    @patch.object(ProjectExplorationService, "_detect_architecture_pattern")
    @patch.object(ProjectExplorationService, "_identify_key_components")
    @patch.object(ProjectExplorationService, "_perform_rag_analysis")
    def test_explore_project_with_rag(self, mock_rag_analysis, mock_components, mock_architecture, mock_basic):
        """Test project exploration with RAG integration."""
        # Mock traditional analysis results
        mock_basic.return_value = {"total_files": 50, "relevant_files": 30}
        mock_architecture.return_value = {"pattern": "unknown", "confidence": 0.0}
        mock_components.return_value = {"core_modules": [], "entry_points": []}

        # Mock RAG analysis results
        mock_rag_analysis.return_value = self.mock_rag_results

        # Execute exploration
        result = self.service_with_rag.explore_project(project_path="/test/project", detail_level="comprehensive")

        # Verify RAG analysis was called
        mock_rag_analysis.assert_called_once()

        # Verify RAG insights are integrated
        self.assertTrue(result.rag_insights_enabled)
        self.assertEqual(result.architecture_pattern, "MVC")  # Enhanced by RAG
        self.assertEqual(len(result.rag_entry_points), 2)
        self.assertIn("src/services/user_service.py", result.core_modules)  # Enhanced by RAG

    @patch.object(ProjectExplorationService, "_perform_basic_analysis")
    @patch.object(ProjectExplorationService, "_detect_architecture_pattern")
    @patch.object(ProjectExplorationService, "_identify_key_components")
    def test_explore_project_without_rag(self, mock_components, mock_architecture, mock_basic):
        """Test project exploration without RAG integration."""
        # Mock traditional analysis results
        mock_basic.return_value = {"total_files": 50, "relevant_files": 30}
        mock_architecture.return_value = {"pattern": "unknown", "confidence": 0.0}
        mock_components.return_value = {"core_modules": [], "entry_points": []}

        # Execute exploration
        result = self.service_without_rag.explore_project(project_path="/test/project", detail_level="comprehensive")

        # Verify RAG insights are not enabled
        self.assertFalse(result.rag_insights_enabled)
        self.assertEqual(result.architecture_pattern, "unknown")  # Not enhanced by RAG
        self.assertEqual(len(result.rag_entry_points), 0)

    def test_perform_rag_analysis(self):
        """Test RAG analysis execution."""
        # Mock the RAG search strategy
        mock_rag_strategy = Mock()

        # Mock architecture detection
        mock_rag_strategy.detect_architecture_patterns.return_value = self.mock_rag_results["architecture_analysis"]

        # Mock entry point discovery
        mock_rag_strategy.discover_entry_points.return_value = self.mock_rag_results["entry_points"]

        # Mock component relationship analysis
        mock_rag_strategy.analyze_component_relationships.return_value = self.mock_rag_results["component_relationships"]

        self.service_with_rag.rag_search_strategy = mock_rag_strategy

        # Execute RAG analysis
        result = self.service_with_rag._perform_rag_analysis(
            project_path=Path("/test/project"),
            focus_area=None,
            detail_level="comprehensive",
        )

        # Verify all RAG methods were called
        mock_rag_strategy.detect_architecture_patterns.assert_called_once()
        mock_rag_strategy.discover_entry_points.assert_called_once()
        mock_rag_strategy.analyze_component_relationships.assert_called_once()

        # Verify results structure
        self.assertIn("architecture_analysis", result)
        self.assertIn("entry_points", result)
        self.assertIn("component_relationships", result)
        self.assertIn("analysis_duration", result)

    def test_integrate_rag_results(self):
        """Test integration of RAG results into exploration result."""
        # Create a basic exploration result
        result = ProjectExplorationResult(
            project_name="test_project",
            project_root="/test/project",
            architecture_pattern="unknown",
            entry_points=[],
            core_modules=[],
        )

        # Integrate RAG results
        self.service_with_rag._integrate_rag_results(result, self.mock_rag_results)

        # Verify integration
        self.assertTrue(result.rag_insights_enabled)

        # Verify architecture pattern enhancement
        self.assertEqual(result.architecture_pattern, "MVC")

        # Verify entry points enhancement
        self.assertIn("src/main.py", result.entry_points)
        self.assertEqual(len(result.rag_entry_points), 2)

        # Verify core modules enhancement
        self.assertIn("src/services/user_service.py", result.core_modules)

        # Verify RAG data storage
        self.assertEqual(
            result.rag_architecture_analysis,
            self.mock_rag_results["architecture_analysis"],
        )
        self.assertEqual(
            result.rag_component_relationships,
            self.mock_rag_results["component_relationships"],
        )

    def test_enhance_learning_recommendations_with_rag(self):
        """Test enhancement of learning recommendations using RAG insights."""
        result = ProjectExplorationResult(
            project_name="test_project",
            project_root="/test/project",
            key_concepts_to_understand=[],
            recommended_reading_order=[],
            coding_patterns=[],
        )

        # Enhance with RAG insights
        self.service_with_rag._enhance_learning_recommendations_with_rag(result, self.mock_rag_results)

        # Verify architecture pattern recommendation
        self.assertIn(
            "Study MVC architecture pattern implementation",
            result.key_concepts_to_understand,
        )

        # Verify entry point recommendations
        self.assertTrue(any("src/main.py" in item for item in result.recommended_reading_order))

        # Verify architectural insights
        self.assertIn(
            "Service-Repository pattern detected in component relationships",
            result.coding_patterns,
        )

    def test_format_exploration_summary_with_rag(self):
        """Test formatting of exploration summary with RAG insights."""
        result = ProjectExplorationResult(
            project_name="test_project",
            project_root="/test/project",
            architecture_pattern="MVC",
            rag_insights_enabled=True,
            rag_architecture_analysis=self.mock_rag_results["architecture_analysis"],
            rag_entry_points=self.mock_rag_results["entry_points"]["primary_entry_points"],
            rag_component_relationships=self.mock_rag_results["component_relationships"],
        )

        summary = self.service_with_rag.format_exploration_summary(result, detail_level="comprehensive")

        # Verify RAG insights section is included
        self.assertIn("🔍 **RAG-Enhanced Insights**", summary)
        self.assertIn("Architecture Pattern Detected**: MVC", summary)
        self.assertIn("Function-Level Entry Points Found**: 2", summary)
        self.assertIn("Component Analysis**: 2 components, 1 relationships", summary)
        self.assertIn("RAG Analysis**: Enabled", summary)

    def test_format_exploration_summary_without_rag(self):
        """Test formatting of exploration summary without RAG insights."""
        result = ProjectExplorationResult(
            project_name="test_project",
            project_root="/test/project",
            architecture_pattern="unknown",
            rag_insights_enabled=False,
        )

        summary = self.service_without_rag.format_exploration_summary(result, detail_level="overview")

        # Verify RAG insights section is not included
        self.assertNotIn("🔍 **RAG-Enhanced Insights**", summary)
        self.assertNotIn("RAG Analysis**: Enabled", summary)

    def test_rag_analysis_error_handling(self):
        """Test error handling in RAG analysis."""
        # Mock RAG strategy that raises exception
        mock_rag_strategy = Mock()
        mock_rag_strategy.detect_architecture_patterns.side_effect = Exception("RAG analysis failed")

        self.service_with_rag.rag_search_strategy = mock_rag_strategy

        # Execute RAG analysis
        result = self.service_with_rag._perform_rag_analysis(
            project_path=Path("/test/project"),
            focus_area=None,
            detail_level="comprehensive",
        )

        # Verify error handling
        self.assertIn("error", result)
        self.assertEqual(result["error"], "RAG analysis failed")

        # Verify empty results structure is returned
        self.assertEqual(result["architecture_analysis"], {})
        self.assertEqual(result["entry_points"], [])
        self.assertEqual(result["component_relationships"], {})

    def test_rag_analysis_detail_levels(self):
        """Test RAG analysis behavior with different detail levels."""
        mock_rag_strategy = Mock()
        mock_rag_strategy.discover_entry_points.return_value = self.mock_rag_results["entry_points"]
        mock_rag_strategy.detect_architecture_patterns.return_value = self.mock_rag_results["architecture_analysis"]
        mock_rag_strategy.analyze_component_relationships.return_value = self.mock_rag_results["component_relationships"]

        self.service_with_rag.rag_search_strategy = mock_rag_strategy

        # Test overview level (should only do entry points)
        self.service_with_rag._perform_rag_analysis(project_path=Path("/test/project"), focus_area=None, detail_level="overview")

        # Entry points should always be called
        mock_rag_strategy.discover_entry_points.assert_called()

        # Architecture detection should not be called for overview
        mock_rag_strategy.detect_architecture_patterns.assert_not_called()

        # Component analysis should not be called for overview
        mock_rag_strategy.analyze_component_relationships.assert_not_called()

        # Reset mocks
        mock_rag_strategy.reset_mock()

        # Test comprehensive level (should do all analysis)
        self.service_with_rag._perform_rag_analysis(
            project_path=Path("/test/project"),
            focus_area=None,
            detail_level="comprehensive",
        )

        # All methods should be called for comprehensive
        mock_rag_strategy.discover_entry_points.assert_called()
        mock_rag_strategy.detect_architecture_patterns.assert_called()
        mock_rag_strategy.analyze_component_relationships.assert_called()


class TestProjectExplorationResult(unittest.TestCase):
    """Test cases for ProjectExplorationResult dataclass with RAG enhancements."""

    def test_project_exploration_result_creation(self):
        """Test creation of ProjectExplorationResult with RAG fields."""
        result = ProjectExplorationResult(project_name="test_project", project_root="/test/project")

        # Verify basic fields
        self.assertEqual(result.project_name, "test_project")
        self.assertEqual(result.project_root, "/test/project")

        # Verify RAG fields are initialized
        self.assertEqual(result.rag_architecture_analysis, {})
        self.assertEqual(result.rag_entry_points, [])
        self.assertEqual(result.rag_component_relationships, {})
        self.assertFalse(result.rag_insights_enabled)

    def test_project_exploration_result_with_rag_data(self):
        """Test ProjectExplorationResult with RAG data."""
        rag_architecture = {"primary_pattern": {"pattern_name": "MVC", "confidence": 0.8}}
        rag_entry_points = [{"file_path": "main.py", "function_name": "main"}]
        rag_relationships = {"core_components": [{"name": "UserService"}]}

        result = ProjectExplorationResult(
            project_name="test_project",
            project_root="/test/project",
            rag_architecture_analysis=rag_architecture,
            rag_entry_points=rag_entry_points,
            rag_component_relationships=rag_relationships,
            rag_insights_enabled=True,
        )

        # Verify RAG data is stored correctly
        self.assertEqual(result.rag_architecture_analysis, rag_architecture)
        self.assertEqual(result.rag_entry_points, rag_entry_points)
        self.assertEqual(result.rag_component_relationships, rag_relationships)
        self.assertTrue(result.rag_insights_enabled)


if __name__ == "__main__":
    unittest.main()
