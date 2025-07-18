"""
Advanced frequency analysis for function calls within files and across projects.

This module provides sophisticated frequency analysis capabilities to enhance
weight calculation with detailed frequency patterns and distribution metrics.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from models.function_call import CallType, FunctionCall

logger = logging.getLogger(__name__)


@dataclass
class FrequencyAnalysisResult:
    """
    Result of frequency analysis for function calls.

    Contains detailed frequency metrics and distribution patterns.
    """

    # Basic frequency metrics
    total_calls: int  # Total number of calls analyzed
    unique_targets: int  # Number of unique target functions
    unique_sources: int  # Number of unique source functions

    # Call frequency distributions
    target_frequency: dict[str, int]  # Target breadcrumb -> call count
    source_frequency: dict[str, int]  # Source breadcrumb -> call count
    call_type_frequency: dict[str, int]  # Call type -> count

    # Pattern analysis
    most_frequent_targets: list[tuple[str, int]]  # Top called functions
    most_frequent_sources: list[tuple[str, int]]  # Most calling functions
    hotspot_files: list[tuple[str, int]]  # Files with most calls

    # Frequency distribution metrics
    frequency_distribution: dict[int, int]  # Frequency -> number of targets with that frequency
    frequency_percentiles: dict[str, float]  # Percentile analysis of frequencies

    # Advanced patterns
    call_chains: list[list[str]]  # Detected call chains (A->B->C)
    circular_calls: list[list[str]]  # Detected circular call patterns
    hotspot_patterns: dict[str, list[str]]  # Pattern type -> list of examples

    def get_frequency_factor_recommendations(self) -> dict[str, float]:
        """
        Generate recommended frequency factors based on analysis.

        Returns:
            Dictionary mapping target breadcrumbs to recommended frequency factors
        """
        recommendations = {}

        if not self.target_frequency:
            return recommendations

        # Calculate statistical thresholds
        frequencies = list(self.target_frequency.values())
        if not frequencies:
            return recommendations

        median_freq = sorted(frequencies)[len(frequencies) // 2]
        q75_freq = sorted(frequencies)[int(len(frequencies) * 0.75)]
        q90_freq = sorted(frequencies)[int(len(frequencies) * 0.90)]

        # Assign frequency factors based on distribution
        for target, freq in self.target_frequency.items():
            if freq >= q90_freq:
                # Very frequent calls - high frequency factor
                recommendations[target] = 1.5 + min(0.5, (freq - q90_freq) / max(1, q90_freq))
            elif freq >= q75_freq:
                # Frequent calls - moderate frequency factor
                recommendations[target] = 1.2 + min(0.3, (freq - q75_freq) / max(1, q75_freq))
            elif freq >= median_freq:
                # Above average calls - slight bonus
                recommendations[target] = 1.1 + min(0.1, (freq - median_freq) / max(1, median_freq))
            else:
                # Below average calls - standard factor
                recommendations[target] = 1.0

        return recommendations


class CallFrequencyAnalyzer:
    """
    Advanced frequency analyzer for function calls.

    This service provides comprehensive frequency analysis capabilities including:
    - Per-file frequency analysis
    - Cross-file frequency patterns
    - Call chain detection
    - Hotspot identification
    - Frequency-based recommendations
    """

    def __init__(self):
        """Initialize the frequency analyzer."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def analyze_file_frequencies(self, calls: list[FunctionCall]) -> FrequencyAnalysisResult:
        """
        Analyze frequency patterns within a single file.

        Args:
            calls: List of function calls from a single file

        Returns:
            Comprehensive frequency analysis result
        """
        if not calls:
            return FrequencyAnalysisResult(
                total_calls=0,
                unique_targets=0,
                unique_sources=0,
                target_frequency={},
                source_frequency={},
                call_type_frequency={},
                most_frequent_targets=[],
                most_frequent_sources=[],
                hotspot_files=[],
                frequency_distribution={},
                frequency_percentiles={},
                call_chains=[],
                circular_calls=[],
                hotspot_patterns={},
            )

        # Basic frequency calculations
        target_freq = defaultdict(int)
        source_freq = defaultdict(int)
        call_type_freq = defaultdict(int)

        for call in calls:
            target_freq[call.target_breadcrumb] += 1
            source_freq[call.source_breadcrumb] += 1
            call_type_freq[call.call_type.value] += 1

        # Calculate frequency distribution
        freq_dist = defaultdict(int)
        for freq in target_freq.values():
            freq_dist[freq] += 1

        # Calculate percentiles
        frequencies = sorted(target_freq.values())
        percentiles = {}
        if frequencies:
            percentiles["p50"] = frequencies[len(frequencies) // 2]
            percentiles["p75"] = frequencies[int(len(frequencies) * 0.75)]
            percentiles["p90"] = frequencies[int(len(frequencies) * 0.90)]
            percentiles["p95"] = frequencies[int(len(frequencies) * 0.95)]
            percentiles["p99"] = frequencies[int(len(frequencies) * 0.99)]

        # Find most frequent patterns
        most_frequent_targets = sorted(target_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        most_frequent_sources = sorted(source_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        # Detect call chains and circular patterns
        call_chains = self._detect_call_chains(calls)
        circular_calls = self._detect_circular_calls(calls)

        # Identify hotspot patterns
        hotspot_patterns = self._identify_hotspot_patterns(calls, target_freq)

        # File-level analysis (single file in this case)
        file_path = calls[0].file_path if calls else ""
        hotspot_files = [(file_path, len(calls))] if calls else []

        result = FrequencyAnalysisResult(
            total_calls=len(calls),
            unique_targets=len(target_freq),
            unique_sources=len(source_freq),
            target_frequency=dict(target_freq),
            source_frequency=dict(source_freq),
            call_type_frequency=dict(call_type_freq),
            most_frequent_targets=most_frequent_targets,
            most_frequent_sources=most_frequent_sources,
            hotspot_files=hotspot_files,
            frequency_distribution=dict(freq_dist),
            frequency_percentiles=percentiles,
            call_chains=call_chains,
            circular_calls=circular_calls,
            hotspot_patterns=hotspot_patterns,
        )

        self.logger.info(
            f"Analyzed {len(calls)} calls: {len(target_freq)} unique targets, "
            f"{len(source_freq)} unique sources, {len(call_chains)} call chains"
        )

        return result

    def analyze_cross_file_frequencies(self, calls_by_file: dict[str, list[FunctionCall]]) -> FrequencyAnalysisResult:
        """
        Analyze frequency patterns across multiple files.

        Args:
            calls_by_file: Dictionary mapping file paths to lists of calls

        Returns:
            Cross-file frequency analysis result
        """
        # Flatten all calls
        all_calls = []
        for file_calls in calls_by_file.values():
            all_calls.extend(file_calls)

        if not all_calls:
            return FrequencyAnalysisResult(
                total_calls=0,
                unique_targets=0,
                unique_sources=0,
                target_frequency={},
                source_frequency={},
                call_type_frequency={},
                most_frequent_targets=[],
                most_frequent_sources=[],
                hotspot_files=[],
                frequency_distribution={},
                frequency_percentiles={},
                call_chains=[],
                circular_calls=[],
                hotspot_patterns={},
            )

        # Aggregate frequencies across files
        global_target_freq = defaultdict(int)
        global_source_freq = defaultdict(int)
        global_call_type_freq = defaultdict(int)

        for call in all_calls:
            global_target_freq[call.target_breadcrumb] += 1
            global_source_freq[call.source_breadcrumb] += 1
            global_call_type_freq[call.call_type.value] += 1

        # Calculate cross-file frequency distribution
        freq_dist = defaultdict(int)
        for freq in global_target_freq.values():
            freq_dist[freq] += 1

        # Calculate percentiles for global distribution
        frequencies = sorted(global_target_freq.values())
        percentiles = {}
        if frequencies:
            percentiles["p50"] = frequencies[len(frequencies) // 2]
            percentiles["p75"] = frequencies[int(len(frequencies) * 0.75)]
            percentiles["p90"] = frequencies[int(len(frequencies) * 0.90)]
            percentiles["p95"] = frequencies[int(len(frequencies) * 0.95)]
            percentiles["p99"] = frequencies[int(len(frequencies) * 0.99)]

        # Find most frequent patterns globally
        most_frequent_targets = sorted(global_target_freq.items(), key=lambda x: x[1], reverse=True)[:20]
        most_frequent_sources = sorted(global_source_freq.items(), key=lambda x: x[1], reverse=True)[:20]

        # Identify hotspot files
        file_call_counts = [(file_path, len(calls)) for file_path, calls in calls_by_file.items()]
        hotspot_files = sorted(file_call_counts, key=lambda x: x[1], reverse=True)[:10]

        # Detect cross-file call chains
        call_chains = self._detect_cross_file_call_chains(calls_by_file)
        circular_calls = self._detect_cross_file_circular_calls(calls_by_file)

        # Identify cross-file hotspot patterns
        hotspot_patterns = self._identify_cross_file_hotspot_patterns(calls_by_file, global_target_freq)

        result = FrequencyAnalysisResult(
            total_calls=len(all_calls),
            unique_targets=len(global_target_freq),
            unique_sources=len(global_source_freq),
            target_frequency=dict(global_target_freq),
            source_frequency=dict(global_source_freq),
            call_type_frequency=dict(global_call_type_freq),
            most_frequent_targets=most_frequent_targets,
            most_frequent_sources=most_frequent_sources,
            hotspot_files=hotspot_files,
            frequency_distribution=dict(freq_dist),
            frequency_percentiles=percentiles,
            call_chains=call_chains,
            circular_calls=circular_calls,
            hotspot_patterns=hotspot_patterns,
        )

        self.logger.info(
            f"Cross-file analysis: {len(all_calls)} total calls across {len(calls_by_file)} files, "
            f"{len(global_target_freq)} unique targets, {len(call_chains)} call chains"
        )

        return result

    def calculate_enhanced_frequency_factors(self, calls: list[FunctionCall], analysis: FrequencyAnalysisResult) -> dict[str, float]:
        """
        Calculate enhanced frequency factors using comprehensive analysis.

        Args:
            calls: List of function calls
            analysis: Frequency analysis result

        Returns:
            Dictionary mapping call identifiers to frequency factors
        """
        factors = {}

        if not calls or not analysis.target_frequency:
            return factors

        # Get frequency factor recommendations from analysis
        recommendations = analysis.get_frequency_factor_recommendations()

        # Calculate factors for each call
        for call in calls:
            call_id = f"{call.target_breadcrumb}@{call.line_number}"
            target = call.target_breadcrumb

            # Base frequency factor from recommendations
            base_factor = recommendations.get(target, 1.0)

            # Apply additional adjustments
            factor = base_factor

            # Bonus for cross-module calls (they're more architecturally significant)
            if call.is_cross_module_call():
                factor *= 1.1

            # Penalty for recursive calls (to avoid infinite loops in analysis)
            if call.is_recursive_call():
                factor *= 0.8

            # Bonus for calls from highly active sources
            source_freq = analysis.source_frequency.get(call.source_breadcrumb, 1)
            if source_freq > analysis.frequency_percentiles.get("p75", 1):
                factor *= 1.05

            # Apply call type specific frequency adjustments
            if call.call_type in [CallType.DIRECT, CallType.METHOD]:
                # Core call types get standard treatment
                pass
            elif call.call_type in [CallType.ASYNC, CallType.ASYNC_METHOD]:
                # Async calls are typically more significant
                factor *= 1.1
            elif call.call_type == CallType.DYNAMIC:
                # Dynamic calls are harder to analyze, reduce factor
                factor *= 0.9

            factors[call_id] = factor

        self.logger.debug(f"Calculated enhanced frequency factors for {len(factors)} calls")

        return factors

    def _detect_call_chains(self, calls: list[FunctionCall]) -> list[list[str]]:
        """
        Detect call chains within a file (A calls B, B calls C, etc.).

        Args:
            calls: List of function calls

        Returns:
            List of call chains (each chain is a list of breadcrumbs)
        """
        # Build call graph
        call_graph = defaultdict(set)
        for call in calls:
            call_graph[call.source_breadcrumb].add(call.target_breadcrumb)

        # Find chains using DFS
        chains = []
        visited = set()

        def dfs(node: str, current_chain: list[str], depth: int = 0):
            if depth > 10 or node in current_chain:  # Avoid infinite recursion
                return

            current_chain.append(node)

            if node in call_graph:
                has_children = False
                for target in call_graph[node]:
                    if target not in current_chain:  # Avoid cycles in individual chains
                        has_children = True
                        dfs(target, current_chain.copy(), depth + 1)

                if not has_children and len(current_chain) > 2:
                    # This is a terminal node with a meaningful chain
                    chains.append(current_chain.copy())
            elif len(current_chain) > 2:
                # Terminal node with meaningful chain
                chains.append(current_chain.copy())

        # Start DFS from all potential root nodes
        for source in call_graph.keys():
            if source not in visited:
                dfs(source, [], 0)

        # Filter and deduplicate chains
        unique_chains = []
        seen_chains = set()
        for chain in chains:
            chain_tuple = tuple(chain)
            if chain_tuple not in seen_chains:
                seen_chains.add(chain_tuple)
                unique_chains.append(chain)

        return unique_chains[:20]  # Limit to top 20 chains

    def _detect_circular_calls(self, calls: list[FunctionCall]) -> list[list[str]]:
        """
        Detect circular call patterns within a file.

        Args:
            calls: List of function calls

        Returns:
            List of circular call patterns
        """
        # Build call graph
        call_graph = defaultdict(set)
        for call in calls:
            call_graph[call.source_breadcrumb].add(call.target_breadcrumb)

        # Find cycles using DFS
        cycles = []
        visited = set()
        rec_stack = set()

        def dfs_cycles(node: str, path: list[str]):
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                if len(cycle) > 2:  # Meaningful cycles only
                    cycles.append(cycle)
                return

            if node in visited:
                return

            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            if node in call_graph:
                for target in call_graph[node]:
                    dfs_cycles(target, path.copy())

            rec_stack.remove(node)

        # Start DFS from all nodes
        for source in call_graph.keys():
            if source not in visited:
                dfs_cycles(source, [])

        # Deduplicate cycles
        unique_cycles = []
        seen_cycles = set()
        for cycle in cycles:
            # Normalize cycle (start from lexicographically smallest element)
            min_idx = cycle.index(min(cycle[:-1]))  # Exclude last element (duplicate)
            normalized = cycle[min_idx:-1] + cycle[:min_idx] + [cycle[min_idx]]
            cycle_tuple = tuple(normalized)

            if cycle_tuple not in seen_cycles:
                seen_cycles.add(cycle_tuple)
                unique_cycles.append(cycle)

        return unique_cycles[:10]  # Limit to top 10 cycles

    def _identify_hotspot_patterns(self, calls: list[FunctionCall], target_freq: dict[str, int]) -> dict[str, list[str]]:
        """
        Identify hotspot patterns in function calls.

        Args:
            calls: List of function calls
            target_freq: Target frequency distribution

        Returns:
            Dictionary mapping pattern types to examples
        """
        patterns = defaultdict(list)

        # Identify high-frequency targets
        if target_freq:
            max_freq = max(target_freq.values())
            high_freq_threshold = max(3, max_freq * 0.7)

            for target, freq in target_freq.items():
                if freq >= high_freq_threshold:
                    patterns["high_frequency_targets"].append(f"{target} ({freq} calls)")

        # Identify call type patterns
        call_type_counts = defaultdict(int)
        for call in calls:
            call_type_counts[call.call_type.value] += 1

        for call_type, count in call_type_counts.items():
            if count >= 5:  # Significant usage
                patterns["frequent_call_types"].append(f"{call_type} ({count} calls)")

        # Identify async patterns
        async_calls = [call for call in calls if "async" in call.call_type.value.lower()]
        if len(async_calls) >= 3:
            async_targets = {call.target_breadcrumb for call in async_calls}
            patterns["async_patterns"].extend(list(async_targets)[:5])

        # Identify cross-module patterns
        cross_module_calls = [call for call in calls if call.is_cross_module_call()]
        if len(cross_module_calls) >= 2:
            cross_module_targets = {call.target_breadcrumb for call in cross_module_calls}
            patterns["cross_module_patterns"].extend(list(cross_module_targets)[:5])

        return dict(patterns)

    def _detect_cross_file_call_chains(self, calls_by_file: dict[str, list[FunctionCall]]) -> list[list[str]]:
        """Detect call chains across multiple files."""
        # For cross-file analysis, we need to be more careful about chains
        # This is a simplified implementation - in practice, you'd want more sophisticated analysis
        all_calls = []
        for file_calls in calls_by_file.values():
            all_calls.extend(file_calls)

        return self._detect_call_chains(all_calls)[:10]  # Limit cross-file chains

    def _detect_cross_file_circular_calls(self, calls_by_file: dict[str, list[FunctionCall]]) -> list[list[str]]:
        """Detect circular calls across multiple files."""
        all_calls = []
        for file_calls in calls_by_file.values():
            all_calls.extend(file_calls)

        return self._detect_circular_calls(all_calls)[:5]  # Limit cross-file cycles

    def _identify_cross_file_hotspot_patterns(
        self, calls_by_file: dict[str, list[FunctionCall]], global_target_freq: dict[str, int]
    ) -> dict[str, list[str]]:
        """Identify hotspot patterns across multiple files."""
        patterns = defaultdict(list)

        # File-level hotspots
        file_call_counts = [(file_path, len(calls)) for file_path, calls in calls_by_file.items()]
        top_files = sorted(file_call_counts, key=lambda x: x[1], reverse=True)[:5]

        for file_path, count in top_files:
            patterns["hotspot_files"].append(f"{file_path} ({count} calls)")

        # Global high-frequency targets
        if global_target_freq:
            max_freq = max(global_target_freq.values())
            high_freq_threshold = max(5, max_freq * 0.6)

            for target, freq in global_target_freq.items():
                if freq >= high_freq_threshold:
                    patterns["global_high_frequency"].append(f"{target} ({freq} calls)")

        return dict(patterns)


# Utility functions for frequency analysis


def calculate_frequency_statistics(target_frequency: dict[str, int]) -> dict[str, float]:
    """
    Calculate statistical metrics for frequency distribution.

    Args:
        target_frequency: Dictionary mapping targets to frequencies

    Returns:
        Dictionary containing statistical metrics
    """
    if not target_frequency:
        return {}

    frequencies = list(target_frequency.values())
    frequencies.sort()

    n = len(frequencies)
    return {
        "min": frequencies[0],
        "max": frequencies[-1],
        "mean": sum(frequencies) / n,
        "median": frequencies[n // 2],
        "q25": frequencies[n // 4],
        "q75": frequencies[3 * n // 4],
        "std": (sum((f - sum(frequencies) / n) ** 2 for f in frequencies) / n) ** 0.5,
    }


def identify_frequency_outliers(target_frequency: dict[str, int], threshold: float = 2.0) -> list[tuple[str, int]]:
    """
    Identify frequency outliers using statistical thresholds.

    Args:
        target_frequency: Dictionary mapping targets to frequencies
        threshold: Number of standard deviations for outlier detection

    Returns:
        List of (target, frequency) tuples for outliers
    """
    if not target_frequency:
        return []

    frequencies = list(target_frequency.values())
    mean_freq = sum(frequencies) / len(frequencies)
    std_freq = (sum((f - mean_freq) ** 2 for f in frequencies) / len(frequencies)) ** 0.5

    if std_freq == 0:
        return []

    outliers = []
    for target, freq in target_frequency.items():
        z_score = abs(freq - mean_freq) / std_freq
        if z_score > threshold:
            outliers.append((target, freq))

    return sorted(outliers, key=lambda x: x[1], reverse=True)
