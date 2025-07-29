"""
Query Preprocessor for Wave 4.0 Query Analysis and Routing System

This service implements comprehensive query preprocessing to standardize, clean,
and optimize query inputs before analysis and routing.
"""

import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

# from ..models.query_features import QueryFeatures  # Commented out to avoid import issues


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Result of query preprocessing with detailed transformation information."""

    # Processed query versions
    original_query: str
    normalized_query: str
    cleaned_query: str
    enhanced_query: str
    standardized_query: str

    # Preprocessing operations applied
    operations_applied: list[str] = field(default_factory=list)
    transformations: dict[str, str] = field(default_factory=dict)

    # Quality metrics
    preprocessing_quality_score: float = 0.0
    confidence_improvement: float = 0.0

    # Detected issues and fixes
    issues_detected: list[str] = field(default_factory=list)
    fixes_applied: list[str] = field(default_factory=list)

    # Processing metadata
    processing_time_ms: float = 0.0
    preprocessing_version: str = "4.0.0"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def get_best_query(self) -> str:
        """Get the best processed query version."""
        # Return the most enhanced version that's available
        for query in [self.enhanced_query, self.standardized_query, self.cleaned_query, self.normalized_query]:
            if query.strip():
                return query
        return self.original_query


@dataclass
class PreprocessingConfig:
    """Configuration for query preprocessing operations."""

    # Normalization options
    enable_case_normalization: bool = True
    enable_whitespace_normalization: bool = True
    enable_punctuation_normalization: bool = True
    enable_contraction_expansion: bool = True

    # Cleaning options
    enable_typo_correction: bool = True
    enable_stop_word_removal: bool = False  # Usually False for queries
    enable_noise_removal: bool = True

    # Enhancement options
    enable_synonym_expansion: bool = False  # Usually False to avoid ambiguity
    enable_abbreviation_expansion: bool = True
    enable_context_enhancement: bool = True

    # Standardization options
    enable_term_standardization: bool = True
    enable_format_standardization: bool = True
    enable_encoding_standardization: bool = True

    # Quality assurance
    min_query_quality_score: float = 0.5
    max_preprocessing_time_ms: float = 1000.0
    preserve_original_intent: bool = True

    # Language-specific options
    target_language: str = "en"
    enable_language_detection: bool = True
    enable_translation: bool = False


class QueryPreprocessor:
    """
    Comprehensive query preprocessor that standardizes, cleans, and optimizes
    query inputs for better analysis and routing performance.
    """

    def __init__(self, config: PreprocessingConfig | None = None):
        self.config = config or PreprocessingConfig()
        self.logger = logging.getLogger(__name__)

        # Initialize preprocessing components
        self._initialize_normalization_rules()
        self._initialize_cleaning_rules()
        self._initialize_enhancement_rules()
        self._initialize_standardization_rules()

        # Statistics tracking
        self.preprocessing_stats = {
            "total_processed": 0,
            "average_processing_time_ms": 0.0,
            "average_quality_improvement": 0.0,
            "common_issues": {},
            "operations_frequency": {},
        }

    def _initialize_normalization_rules(self):
        """Initialize normalization rules and patterns."""

        # Case normalization (usually lowercase for consistency)
        self.case_normalization = {"preserve_acronyms": True, "preserve_proper_nouns": True}

        # Whitespace normalization
        self.whitespace_patterns = [
            (r"\s+", " "),  # Multiple spaces to single
            (r"^\s+|\s+$", ""),  # Leading/trailing spaces
            (r"\s+([.!?])", r"\1"),  # Space before punctuation
            (r"([.!?])\s+([.!?])", r"\1\2"),  # Multiple punctuation
        ]

        # Punctuation normalization
        self.punctuation_normalization = {
            # Smart quotes to regular
            '"': '"',
            '"': '"',
            """: "'", """: "'",
            # En/em dashes to hyphens
            "–": "-",
            "—": "-",
            # Other special characters
            "…": "...",
            "′": "'",
            "″": '"',
        }

        # Contraction expansion
        self.contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "couldn't": "could not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "doesn't": "does not",
            "didn't": "did not",
            "what's": "what is",
            "how's": "how is",
            "where's": "where is",
            "when's": "when is",
            "why's": "why is",
            "who's": "who is",
            "it's": "it is",
            "that's": "that is",
            "there's": "there is",
            "here's": "here is",
            "you're": "you are",
            "we're": "we are",
            "they're": "they are",
            "i'm": "i am",
            "you'll": "you will",
            "we'll": "we will",
            "they'll": "they will",
            "i'll": "i will",
            "you'd": "you would",
            "we'd": "we would",
            "they'd": "they would",
            "i'd": "i would",
            "you've": "you have",
            "we've": "we have",
            "they've": "they have",
            "i've": "i have",
        }

    def _initialize_cleaning_rules(self):
        """Initialize cleaning rules for noise removal and typo correction."""

        # Common typos and corrections
        self.typo_corrections = {
            # Programming-related typos
            "funciton": "function",
            "fucntion": "function",
            "functoin": "function",
            "calss": "class",
            "clas": "class",
            "clss": "class",
            "varialbe": "variable",
            "varible": "variable",
            "variabel": "variable",
            "methdo": "method",
            "metod": "method",
            "metohd": "method",
            "databse": "database",
            "datbase": "database",
            "databas": "database",
            "algorith": "algorithm",
            "algorithim": "algorithm",
            "algoritm": "algorithm",
            # General typos
            "teh": "the",
            "hte": "the",
            "adn": "and",
            "nad": "and",
            "whta": "what",
            "waht": "what",
            "hwat": "what",
            "hwo": "how",
            "woh": "how",
            "whih": "which",
            "wich": "which",
            "woudl": "would",
            "shuold": "should",
            "coudl": "could",
            "recieve": "receive",
            "seperate": "separate",
            "definately": "definitely",
        }

        # Noise patterns to remove
        self.noise_patterns = [
            r"\bum+\b",  # Filler words
            r"\buh+\b",
            r"\blike\s+you\s+know\b",  # Verbal fillers
            r"\bI\s+mean\b",
            r"[\(\[].*[Tt]rying.*[\)\]]",  # Meta comments
            r"[\(\[].*[Hh]elp.*[\)\]]",
            r"\bplease\s+help\b",  # Politeness (preserve intent but simplify)
        ]

        # Patterns for excessive punctuation
        self.excessive_punctuation = [
            (r"[!]{2,}", "!"),  # Multiple exclamations
            (r"[?]{2,}", "?"),  # Multiple questions
            (r"[.]{3,}", "..."),  # Multiple dots
            (r"[-]{2,}", "-"),  # Multiple dashes
        ]

    def _initialize_enhancement_rules(self):
        """Initialize enhancement rules for query improvement."""

        # Abbreviation expansions (technical domain)
        self.abbreviation_expansions = {
            # Programming abbreviations
            "js": "javascript",
            "ts": "typescript",
            "py": "python",
            "cpp": "c++",
            "cs": "c#",
            "php": "php",
            "html": "html",
            "css": "css",
            "sql": "sql",
            # Framework abbreviations
            "react": "react",
            "vue": "vue",
            "ng": "angular",
            "django": "django",
            "flask": "flask",
            "express": "express",
            # Database abbreviations
            "db": "database",
            "mysql": "mysql",
            "postgres": "postgresql",
            "mongo": "mongodb",
            "redis": "redis",
            # Architecture abbreviations
            "api": "api",
            "rest": "rest",
            "crud": "crud",
            "mvc": "mvc",
            "mvp": "mvp",
            "mvvm": "mvvm",
            # Development abbreviations
            "ci": "continuous integration",
            "cd": "continuous deployment",
            "tdd": "test driven development",
            "bdd": "behavior driven development",
            "orm": "object relational mapping",
            "ide": "integrated development environment",
        }

        # Context enhancement patterns
        self.context_patterns = [
            # Add implicit context
            (r"\b(show|find|get)\s+(\w+)\b", r"\1 \2 information"),
            (r"\bhow\s+to\s+(\w+)\b", r"how to \1 implementation"),
            (r"\bwhat\s+is\s+(\w+)\b", r"what is \1 definition"),
        ]

    def _initialize_standardization_rules(self):
        """Initialize standardization rules for consistent formatting."""

        # Term standardization (normalize to preferred forms)
        self.term_standardization = {
            # Prefer full forms over abbreviated
            "func": "function",
            "meth": "method",
            "var": "variable",
            "obj": "object",
            "arr": "array",
            "str": "string",
            "num": "number",
            "bool": "boolean",
            "int": "integer",
            # Standardize naming conventions
            "camelCase": "camel case",
            "PascalCase": "pascal case",
            "snake_case": "snake case",
            "kebab-case": "kebab case",
            # Standardize concepts
            "async": "asynchronous",
            "sync": "synchronous",
            "config": "configuration",
            "param": "parameter",
            "arg": "argument",
            "ref": "reference",
        }

        # Format standardization patterns
        self.format_patterns = [
            # Standardize code references
            (r"`([^`]+)`", r"\1"),  # Remove backticks for analysis
            (r"\b([a-zA-Z_]\w*)\(\)", r"\1 function"),  # Function calls
            (r"\b([A-Z][a-zA-Z]*)\.([a-z]\w*)", r"\1 \2 method"),  # Method calls
        ]

    async def preprocess_query(self, query: str) -> PreprocessingResult:
        """Perform comprehensive query preprocessing."""
        start_time = time.time()

        try:
            self.logger.debug(f"Preprocessing query: '{query[:50]}...'")

            # Initialize result
            result = PreprocessingResult(original_query=query)

            # Stage 1: Input validation and initial assessment
            quality_issues = self._assess_query_quality(query)
            result.issues_detected.extend(quality_issues)

            # Stage 2: Normalization
            normalized = await self._normalize_query(query, result)
            result.normalized_query = normalized

            # Stage 3: Cleaning
            cleaned = await self._clean_query(normalized, result)
            result.cleaned_query = cleaned

            # Stage 4: Enhancement (if enabled)
            enhanced = await self._enhance_query(cleaned, result)
            result.enhanced_query = enhanced

            # Stage 5: Standardization
            standardized = await self._standardize_query(enhanced, result)
            result.standardized_query = standardized

            # Stage 6: Quality assessment and validation
            result.preprocessing_quality_score = self._calculate_quality_score(result)
            result.confidence_improvement = self._estimate_confidence_improvement(result)

            # Record processing time
            result.processing_time_ms = (time.time() - start_time) * 1000

            # Update statistics
            self._update_preprocessing_stats(result)

            self.logger.debug(
                f"Preprocessing complete: quality={result.preprocessing_quality_score:.2f}, "
                f"improvement={result.confidence_improvement:.2f}, "
                f"time={result.processing_time_ms:.2f}ms"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error in query preprocessing: {e}")
            # Return minimal result on error
            result = PreprocessingResult(
                original_query=query,
                normalized_query=query,
                cleaned_query=query,
                enhanced_query=query,
                standardized_query=query,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            result.issues_detected.append(f"Preprocessing error: {str(e)}")
            return result

    def _assess_query_quality(self, query: str) -> list[str]:
        """Assess initial query quality and identify potential issues."""
        issues = []

        # Length checks
        if len(query.strip()) == 0:
            issues.append("Empty query")
        elif len(query) < 3:
            issues.append("Very short query")
        elif len(query) > 500:
            issues.append("Very long query")

        # Character encoding issues
        if not query.isascii():
            issues.append("Non-ASCII characters detected")

        # Excessive repetition
        words = query.lower().split()
        if len(words) != len(set(words)) and len(words) > 5:
            issues.append("Excessive word repetition")

        # Excessive punctuation
        punct_ratio = sum(1 for c in query if not c.isalnum() and not c.isspace()) / max(1, len(query))
        if punct_ratio > 0.3:
            issues.append("Excessive punctuation")

        # Case inconsistency
        if query != query.lower() and query != query.upper():
            has_mixed_case = any(word.islower() and word.isupper() for word in words)
            if has_mixed_case:
                issues.append("Inconsistent case usage")

        return issues

    async def _normalize_query(self, query: str, result: PreprocessingResult) -> str:
        """Apply normalization operations."""
        normalized = query

        if self.config.enable_case_normalization:
            normalized = self._normalize_case(normalized)
            if normalized != query:
                result.operations_applied.append("case_normalization")
                result.transformations["case_normalization"] = f"{query[:20]}... -> {normalized[:20]}..."

        if self.config.enable_whitespace_normalization:
            normalized = self._normalize_whitespace(normalized)
            if normalized != result.transformations.get("case_normalization", query):
                result.operations_applied.append("whitespace_normalization")

        if self.config.enable_punctuation_normalization:
            normalized = self._normalize_punctuation(normalized)
            if "punctuation_normalization" not in result.operations_applied:
                result.operations_applied.append("punctuation_normalization")

        if self.config.enable_contraction_expansion:
            normalized = self._expand_contractions(normalized)
            if "contraction_expansion" not in result.operations_applied:
                result.operations_applied.append("contraction_expansion")

        return normalized

    def _normalize_case(self, query: str) -> str:
        """Normalize case while preserving important capitalization."""
        if not self.config.enable_case_normalization:
            return query

        # Preserve acronyms and proper nouns if configured
        if self.case_normalization["preserve_acronyms"]:
            # Find acronyms (2+ consecutive uppercase letters)
            acronyms = re.findall(r"\b[A-Z]{2,}\b", query)

            # Convert to lowercase
            normalized = query.lower()

            # Restore acronyms
            for acronym in acronyms:
                normalized = re.sub(r"\b" + acronym.lower() + r"\b", acronym, normalized)

            return normalized
        else:
            return query.lower()

    def _normalize_whitespace(self, query: str) -> str:
        """Normalize whitespace patterns."""
        normalized = query

        for pattern, replacement in self.whitespace_patterns:
            normalized = re.sub(pattern, replacement, normalized)

        return normalized.strip()

    def _normalize_punctuation(self, query: str) -> str:
        """Normalize punctuation characters."""
        normalized = query

        # Apply character-level normalization
        for char, replacement in self.punctuation_normalization.items():
            normalized = normalized.replace(char, replacement)

        # Apply pattern-based normalization
        for pattern, replacement in self.excessive_punctuation:
            normalized = re.sub(pattern, replacement, normalized)

        return normalized

    def _expand_contractions(self, query: str) -> str:
        """Expand contractions to full forms."""
        normalized = query

        # Apply contraction expansions (case-insensitive)
        for contraction, expansion in self.contractions.items():
            pattern = r"\b" + re.escape(contraction) + r"\b"
            normalized = re.sub(pattern, expansion, normalized, flags=re.IGNORECASE)

        return normalized

    async def _clean_query(self, query: str, result: PreprocessingResult) -> str:
        """Apply cleaning operations."""
        cleaned = query

        if self.config.enable_typo_correction:
            cleaned = self._correct_typos(cleaned)
            if cleaned != query:
                result.operations_applied.append("typo_correction")
                result.fixes_applied.append("Corrected spelling errors")

        if self.config.enable_noise_removal:
            cleaned = self._remove_noise(cleaned)
            if cleaned != result.operations_applied and "noise_removal" not in result.operations_applied:
                result.operations_applied.append("noise_removal")
                result.fixes_applied.append("Removed noise patterns")

        return cleaned

    def _correct_typos(self, query: str) -> str:
        """Correct common typos."""
        corrected = query

        # Apply typo corrections (case-insensitive)
        for typo, correction in self.typo_corrections.items():
            pattern = r"\b" + re.escape(typo) + r"\b"
            corrected = re.sub(pattern, correction, corrected, flags=re.IGNORECASE)

        return corrected

    def _remove_noise(self, query: str) -> str:
        """Remove noise patterns from query."""
        cleaned = query

        # Apply noise removal patterns
        for pattern in self.noise_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # Clean up extra whitespace introduced by removals
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        return cleaned

    async def _enhance_query(self, query: str, result: PreprocessingResult) -> str:
        """Apply enhancement operations."""
        enhanced = query

        if self.config.enable_abbreviation_expansion:
            enhanced = self._expand_abbreviations(enhanced)
            if enhanced != query:
                result.operations_applied.append("abbreviation_expansion")

        if self.config.enable_context_enhancement:
            enhanced = self._enhance_context(enhanced)
            if "context_enhancement" not in result.operations_applied and enhanced != query:
                result.operations_applied.append("context_enhancement")

        return enhanced

    def _expand_abbreviations(self, query: str) -> str:
        """Expand technical abbreviations."""
        expanded = query

        # Apply abbreviation expansions (word boundaries)
        for abbrev, expansion in self.abbreviation_expansions.items():
            pattern = r"\b" + re.escape(abbrev) + r"\b"
            expanded = re.sub(pattern, expansion, expanded, flags=re.IGNORECASE)

        return expanded

    def _enhance_context(self, query: str) -> str:
        """Enhance query with implicit context."""
        enhanced = query

        # Apply context enhancement patterns
        for pattern, replacement in self.context_patterns:
            enhanced = re.sub(pattern, replacement, enhanced, flags=re.IGNORECASE)

        return enhanced

    async def _standardize_query(self, query: str, result: PreprocessingResult) -> str:
        """Apply standardization operations."""
        standardized = query

        if self.config.enable_term_standardization:
            standardized = self._standardize_terms(standardized)
            if standardized != query:
                result.operations_applied.append("term_standardization")

        if self.config.enable_format_standardization:
            standardized = self._standardize_format(standardized)
            if "format_standardization" not in result.operations_applied and standardized != query:
                result.operations_applied.append("format_standardization")

        return standardized

    def _standardize_terms(self, query: str) -> str:
        """Standardize terminology to preferred forms."""
        standardized = query

        # Apply term standardizations
        for term, standard_form in self.term_standardization.items():
            pattern = r"\b" + re.escape(term) + r"\b"
            standardized = re.sub(pattern, standard_form, standardized, flags=re.IGNORECASE)

        return standardized

    def _standardize_format(self, query: str) -> str:
        """Standardize formatting patterns."""
        standardized = query

        # Apply format standardizations
        for pattern, replacement in self.format_patterns:
            standardized = re.sub(pattern, replacement, standardized)

        return standardized

    def _calculate_quality_score(self, result: PreprocessingResult) -> float:
        """Calculate overall preprocessing quality score."""
        score = 0.5  # Base score

        # Bonus for successful operations
        score += len(result.operations_applied) * 0.05

        # Bonus for fixes applied
        score += len(result.fixes_applied) * 0.1

        # Penalty for unresolved issues
        unresolved_issues = [issue for issue in result.issues_detected if not any(fix in issue for fix in result.fixes_applied)]
        score -= len(unresolved_issues) * 0.1

        # Bonus for query improvement
        if result.standardized_query != result.original_query:
            improvement_ratio = 1 - (len(result.original_query) / max(1, len(result.standardized_query)))
            score += abs(improvement_ratio) * 0.2

        return min(max(score, 0.0), 1.0)

    def _estimate_confidence_improvement(self, result: PreprocessingResult) -> float:
        """Estimate how much preprocessing improved query analysis confidence."""
        improvement = 0.0

        # Improvement from normalization
        if "case_normalization" in result.operations_applied:
            improvement += 0.05
        if "whitespace_normalization" in result.operations_applied:
            improvement += 0.03
        if "punctuation_normalization" in result.operations_applied:
            improvement += 0.02

        # Improvement from cleaning
        if "typo_correction" in result.operations_applied:
            improvement += 0.1
        if "noise_removal" in result.operations_applied:
            improvement += 0.08

        # Improvement from enhancement
        if "abbreviation_expansion" in result.operations_applied:
            improvement += 0.12
        if "context_enhancement" in result.operations_applied:
            improvement += 0.06

        # Improvement from standardization
        if "term_standardization" in result.operations_applied:
            improvement += 0.07
        if "format_standardization" in result.operations_applied:
            improvement += 0.04

        return min(improvement, 0.5)  # Cap at 50% improvement

    def _update_preprocessing_stats(self, result: PreprocessingResult) -> None:
        """Update preprocessing statistics."""
        self.preprocessing_stats["total_processed"] += 1

        # Update average processing time
        total = self.preprocessing_stats["total_processed"]
        current_avg = self.preprocessing_stats["average_processing_time_ms"]
        new_avg = ((current_avg * (total - 1)) + result.processing_time_ms) / total
        self.preprocessing_stats["average_processing_time_ms"] = new_avg

        # Update average quality improvement
        current_quality_avg = self.preprocessing_stats["average_quality_improvement"]
        new_quality_avg = ((current_quality_avg * (total - 1)) + result.confidence_improvement) / total
        self.preprocessing_stats["average_quality_improvement"] = new_quality_avg

        # Update common issues tracking
        for issue in result.issues_detected:
            if issue not in self.preprocessing_stats["common_issues"]:
                self.preprocessing_stats["common_issues"][issue] = 0
            self.preprocessing_stats["common_issues"][issue] += 1

        # Update operations frequency
        for operation in result.operations_applied:
            if operation not in self.preprocessing_stats["operations_frequency"]:
                self.preprocessing_stats["operations_frequency"][operation] = 0
            self.preprocessing_stats["operations_frequency"][operation] += 1

    def get_preprocessing_statistics(self) -> dict[str, Any]:
        """Get current preprocessing statistics."""
        return self.preprocessing_stats.copy()

    def update_configuration(self, new_config: PreprocessingConfig) -> None:
        """Update preprocessing configuration."""
        self.config = new_config
        self.logger.info("Preprocessing configuration updated")

    def validate_query(self, query: str) -> tuple[bool, list[str]]:
        """Validate a query for basic requirements."""
        issues = []

        if not query.strip():
            issues.append("Query is empty")

        if len(query) > 1000:
            issues.append("Query is too long")

        if not re.search(r"[a-zA-Z]", query):
            issues.append("Query contains no alphabetic characters")

        # Check for potentially harmful patterns
        harmful_patterns = [
            r"<script[^>]*>",  # Script tags
            r"javascript:",  # JavaScript URLs
            r"eval\s*\(",  # Eval functions
            r"exec\s*\(",  # Exec functions
        ]

        for pattern in harmful_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                issues.append("Query contains potentially harmful content")
                break

        return len(issues) == 0, issues


# Factory function
_query_preprocessor_instance = None


def get_query_preprocessor(config: PreprocessingConfig | None = None) -> QueryPreprocessor:
    """Get or create a QueryPreprocessor instance."""
    global _query_preprocessor_instance
    if _query_preprocessor_instance is None or config is not None:
        _query_preprocessor_instance = QueryPreprocessor(config)
    return _query_preprocessor_instance
