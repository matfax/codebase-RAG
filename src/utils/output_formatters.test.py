"""
Unit tests for output formatting utilities.

This module tests the comprehensive formatting functions for converting function paths
and chains into various output formats including arrow format and Mermaid diagrams.
"""

import pytest

from src.utils.output_formatters import (
    MermaidStyle,
    OutputFormat,
    _clean_mermaid_text,
    _get_quality_style,
    _wrap_arrow_path,
    format_arrow_path,
    format_comprehensive_output,
    format_mermaid_path,
    format_path_comparison,
)


class TestArrowFormatting:
    """Test arrow format generation."""

    def test_format_arrow_path_basic(self):
        """Test basic arrow path formatting."""
        path_steps = ["func_a", "func_b", "func_c"]
        result = format_arrow_path(path_steps)
        assert result == "func_a => func_b => func_c"

    def test_format_arrow_path_single_step(self):
        """Test single step path formatting."""
        path_steps = ["func_a"]
        result = format_arrow_path(path_steps)
        assert result == "func_a"

    def test_format_arrow_path_empty(self):
        """Test empty path formatting."""
        path_steps = []
        result = format_arrow_path(path_steps)
        assert result == ""

    def test_format_arrow_path_with_relationships(self):
        """Test arrow path with relationship annotations."""
        path_steps = ["func_a", "func_b", "func_c"]
        relationships = ["calls", "inherits"]
        result = format_arrow_path(path_steps, relationships, include_relationships=True)
        assert "--[calls]-->" in result
        assert "--[inherits]-->" in result

    def test_format_arrow_path_custom_separator(self):
        """Test arrow path with custom separator."""
        path_steps = ["func_a", "func_b", "func_c"]
        result = format_arrow_path(path_steps, custom_separator=" -> ")
        assert result == "func_a -> func_b -> func_c"

    def test_format_arrow_path_wrapping(self):
        """Test arrow path wrapping for long paths."""
        path_steps = ["very_long_function_name_a", "very_long_function_name_b", "very_long_function_name_c"]
        result = format_arrow_path(path_steps, max_line_length=30)
        assert "\n" in result  # Should wrap

    def test_wrap_arrow_path(self):
        """Test arrow path wrapping function."""
        path = "func_a => func_b => func_c"
        result = _wrap_arrow_path(path, 15, " => ")
        lines = result.split("\n")
        assert len(lines) > 1


class TestMermaidFormatting:
    """Test Mermaid format generation."""

    def test_format_mermaid_path_basic(self):
        """Test basic Mermaid path formatting."""
        path_steps = ["func_a", "func_b", "func_c"]
        result = format_mermaid_path(path_steps)
        assert "flowchart TD" in result
        assert "func_a" in result
        assert "func_b" in result
        assert "func_c" in result
        assert "-->" in result

    def test_format_mermaid_path_single_step(self):
        """Test single step Mermaid formatting."""
        path_steps = ["func_a"]
        result = format_mermaid_path(path_steps)
        assert "graph TD" in result
        assert "func_a" in result

    def test_format_mermaid_path_empty(self):
        """Test empty Mermaid formatting."""
        path_steps = []
        result = format_mermaid_path(path_steps)
        assert result == ""

    def test_format_mermaid_path_with_relationships(self):
        """Test Mermaid path with relationship annotations."""
        path_steps = ["func_a", "func_b", "func_c"]
        relationships = ["calls", "inherits"]
        result = format_mermaid_path(path_steps, relationships)
        assert "|calls|" in result
        assert "|inherits|" in result

    def test_format_mermaid_path_with_quality(self):
        """Test Mermaid path with quality information."""
        path_steps = ["func_a", "func_b"]
        quality_scores = {"overall_score": 0.85}
        result = format_mermaid_path(path_steps, include_quality_info=True, quality_scores=quality_scores)
        assert "highQuality" in result

    def test_format_mermaid_path_graph_style(self):
        """Test Mermaid path with graph style."""
        path_steps = ["func_a", "func_b"]
        result = format_mermaid_path(path_steps, style=MermaidStyle.GRAPH)
        assert "graph TD" in result

    def test_format_mermaid_path_sequence_style(self):
        """Test Mermaid path with sequence style."""
        path_steps = ["func_a", "func_b"]
        result = format_mermaid_path(path_steps, style=MermaidStyle.SEQUENCE)
        assert "sequenceDiagram" in result
        assert "->>" in result

    def test_clean_mermaid_text(self):
        """Test text cleaning for Mermaid."""
        text = 'func_with_"quotes"_and_[brackets]'
        result = _clean_mermaid_text(text)
        assert '"' not in result
        assert "[" not in result
        assert "]" not in result

    def test_clean_mermaid_text_long(self):
        """Test text cleaning for very long names."""
        text = "a" * 50
        result = _clean_mermaid_text(text)
        assert len(result) <= 40
        assert result.endswith("...")

    def test_get_quality_style(self):
        """Test quality-based styling."""
        # High quality
        high_quality = _get_quality_style({"overall_score": 0.9}, 0)
        assert high_quality == ":::highQuality"

        # Medium quality
        medium_quality = _get_quality_style({"overall_score": 0.7}, 0)
        assert medium_quality == ":::mediumQuality"

        # Low quality
        low_quality = _get_quality_style({"overall_score": 0.3}, 0)
        assert low_quality == ":::lowQuality"


class TestPathComparison:
    """Test path comparison formatting."""

    def test_format_path_comparison_basic(self):
        """Test basic path comparison."""
        paths = [
            {"path_steps": ["func_a", "func_b"], "quality": {"overall_score": 0.8, "reliability_score": 0.9}, "relationships": ["calls"]},
            {
                "path_steps": ["func_a", "func_c", "func_b"],
                "quality": {"overall_score": 0.6, "reliability_score": 0.7},
                "relationships": ["calls", "inherits"],
            },
        ]
        result = format_path_comparison(paths)
        assert "paths" in result
        assert "comparison" in result
        assert "recommendations" in result
        assert len(result["paths"]) == 2

    def test_format_path_comparison_empty(self):
        """Test empty path comparison."""
        result = format_path_comparison([])
        assert result["paths"] == []
        assert result["comparison"] == {}
        assert result["recommendations"] == {}

    def test_format_path_comparison_custom_metrics(self):
        """Test path comparison with custom metrics."""
        paths = [
            {"path_steps": ["func_a", "func_b"], "quality": {"overall_score": 0.8, "complexity_score": 0.3}, "relationships": ["calls"]}
        ]
        result = format_path_comparison(paths, comparison_metrics=["overall_score", "complexity_score"])
        assert "overall_score" in result["comparison"]
        assert "complexity_score" in result["comparison"]

    def test_format_path_comparison_no_recommendations(self):
        """Test path comparison without recommendations."""
        paths = [{"path_steps": ["func_a", "func_b"], "quality": {"overall_score": 0.8}, "relationships": ["calls"]}]
        result = format_path_comparison(paths, include_recommendations=False)
        assert result["recommendations"] == {}


class TestComprehensiveOutput:
    """Test comprehensive output formatting."""

    def test_format_comprehensive_output_basic(self):
        """Test basic comprehensive output formatting."""
        paths = [
            {
                "path_id": "path_1",
                "start_breadcrumb": "func_a",
                "end_breadcrumb": "func_b",
                "path_steps": ["func_a", "func_b"],
                "quality": {"overall_score": 0.8, "reliability_score": 0.9},
                "path_type": "execution",
                "relationships": ["calls"],
                "evidence": [],
            }
        ]
        result = format_comprehensive_output(paths)
        assert "paths" in result
        assert "summary" in result
        assert len(result["paths"]) == 1
        assert result["summary"]["total_paths"] == 1

    def test_format_comprehensive_output_empty(self):
        """Test empty comprehensive output."""
        result = format_comprehensive_output([])
        assert result["paths"] == []
        assert result["summary"]["total_paths"] == 0

    def test_format_comprehensive_output_arrow_only(self):
        """Test comprehensive output with arrow format only."""
        paths = [
            {
                "path_id": "path_1",
                "start_breadcrumb": "func_a",
                "end_breadcrumb": "func_b",
                "path_steps": ["func_a", "func_b"],
                "quality": {"overall_score": 0.8},
                "path_type": "execution",
                "relationships": ["calls"],
                "evidence": [],
            }
        ]
        result = format_comprehensive_output(paths, output_format=OutputFormat.ARROW)
        assert "arrow_format" in result["paths"][0]
        assert "mermaid_format" not in result["paths"][0]

    def test_format_comprehensive_output_mermaid_only(self):
        """Test comprehensive output with Mermaid format only."""
        paths = [
            {
                "path_id": "path_1",
                "start_breadcrumb": "func_a",
                "end_breadcrumb": "func_b",
                "path_steps": ["func_a", "func_b"],
                "quality": {"overall_score": 0.8},
                "path_type": "execution",
                "relationships": ["calls"],
                "evidence": [],
            }
        ]
        result = format_comprehensive_output(paths, output_format=OutputFormat.MERMAID)
        assert "mermaid_format" in result["paths"][0]
        assert "arrow_format" not in result["paths"][0]

    def test_format_comprehensive_output_both_formats(self):
        """Test comprehensive output with both formats."""
        paths = [
            {
                "path_id": "path_1",
                "start_breadcrumb": "func_a",
                "end_breadcrumb": "func_b",
                "path_steps": ["func_a", "func_b"],
                "quality": {"overall_score": 0.8},
                "path_type": "execution",
                "relationships": ["calls"],
                "evidence": [],
            }
        ]
        result = format_comprehensive_output(paths, output_format=OutputFormat.BOTH)
        assert "arrow_format" in result["paths"][0]
        assert "mermaid_format" in result["paths"][0]

    def test_format_comprehensive_output_with_comparison(self):
        """Test comprehensive output with comparison."""
        paths = [
            {
                "path_id": "path_1",
                "start_breadcrumb": "func_a",
                "end_breadcrumb": "func_b",
                "path_steps": ["func_a", "func_b"],
                "quality": {"overall_score": 0.8, "reliability_score": 0.9},
                "path_type": "execution",
                "relationships": ["calls"],
                "evidence": [],
            },
            {
                "path_id": "path_2",
                "start_breadcrumb": "func_a",
                "end_breadcrumb": "func_b",
                "path_steps": ["func_a", "func_c", "func_b"],
                "quality": {"overall_score": 0.6, "reliability_score": 0.7},
                "path_type": "execution",
                "relationships": ["calls", "inherits"],
                "evidence": [],
            },
        ]
        result = format_comprehensive_output(paths, include_comparison=True)
        assert "comparison" in result
        assert "overall_score" in result["comparison"]

    def test_format_comprehensive_output_with_recommendations(self):
        """Test comprehensive output with recommendations."""
        paths = [
            {
                "path_id": "path_1",
                "start_breadcrumb": "func_a",
                "end_breadcrumb": "func_b",
                "path_steps": ["func_a", "func_b"],
                "quality": {"overall_score": 0.8, "reliability_score": 0.9, "directness_score": 0.8},
                "path_type": "execution",
                "relationships": ["calls"],
                "evidence": [],
            },
            {
                "path_id": "path_2",
                "start_breadcrumb": "func_a",
                "end_breadcrumb": "func_b",
                "path_steps": ["func_a", "func_c", "func_b"],
                "quality": {"overall_score": 0.6, "reliability_score": 0.7, "directness_score": 0.5},
                "path_type": "execution",
                "relationships": ["calls", "inherits"],
                "evidence": [],
            },
        ]
        result = format_comprehensive_output(paths, include_recommendations=True)
        assert "recommendations" in result
        assert "best_overall" in result["recommendations"]
        assert "shortest" in result["recommendations"]

    def test_format_comprehensive_output_mermaid_styles(self):
        """Test comprehensive output with different Mermaid styles."""
        paths = [
            {
                "path_id": "path_1",
                "start_breadcrumb": "func_a",
                "end_breadcrumb": "func_b",
                "path_steps": ["func_a", "func_b"],
                "quality": {"overall_score": 0.8},
                "path_type": "execution",
                "relationships": ["calls"],
                "evidence": [],
            }
        ]

        # Test flowchart style
        result_flowchart = format_comprehensive_output(paths, output_format=OutputFormat.MERMAID, mermaid_style=MermaidStyle.FLOWCHART)
        assert "flowchart TD" in result_flowchart["paths"][0]["mermaid_format"]

        # Test sequence style
        result_sequence = format_comprehensive_output(paths, output_format=OutputFormat.MERMAID, mermaid_style=MermaidStyle.SEQUENCE)
        assert "sequenceDiagram" in result_sequence["paths"][0]["mermaid_format"]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_format_arrow_path_none_relationships(self):
        """Test arrow formatting with None relationships."""
        path_steps = ["func_a", "func_b"]
        result = format_arrow_path(path_steps, None, include_relationships=True)
        assert result == "func_a => func_b"

    def test_format_mermaid_path_none_relationships(self):
        """Test Mermaid formatting with None relationships."""
        path_steps = ["func_a", "func_b"]
        result = format_mermaid_path(path_steps, None)
        assert "flowchart TD" in result
        assert "func_a" in result

    def test_format_path_comparison_missing_quality(self):
        """Test path comparison with missing quality data."""
        paths = [{"path_steps": ["func_a", "func_b"], "relationships": ["calls"]}]
        result = format_path_comparison(paths)
        assert len(result["paths"]) == 1
        assert result["paths"][0]["quality"] == {}

    def test_comprehensive_output_custom_styling(self):
        """Test comprehensive output with custom styling."""
        paths = [
            {
                "path_id": "path_1",
                "start_breadcrumb": "func_a",
                "end_breadcrumb": "func_b",
                "path_steps": ["func_a", "func_b"],
                "quality": {"overall_score": 0.8},
                "path_type": "execution",
                "relationships": ["calls"],
                "evidence": [],
            }
        ]
        custom_styling = {"customClass": "fill:#ff9999"}
        result = format_comprehensive_output(paths, output_format=OutputFormat.MERMAID, custom_styling=custom_styling)
        assert "customClass" in result["paths"][0]["mermaid_format"]
