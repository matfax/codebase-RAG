"""
High/Low-Level Keyword Extraction Utility

This module implements intelligent keyword extraction for the multi-modal retrieval system,
separating high-level conceptual keywords from low-level entity-specific keywords.
"""

import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

try:
    from ..models.query_features import KeywordExtraction, KeywordLevel
except ImportError:
    from models.query_features import KeywordExtraction, KeywordLevel


logger = logging.getLogger(__name__)


@dataclass
class KeywordPatterns:
    """Patterns for identifying different types of keywords."""

    # Low-level patterns (specific entities)
    function_patterns = [
        r"\b\w+\(\)",  # function calls
        r"\bdef\s+\w+",  # function definitions
        r"\bfunction\s+\w+",  # function declarations
        r"\b[a-z_]\w*[A-Z]\w*\b",  # camelCase identifiers
        r"\b[a-z]+_[a-z_]+\b",  # snake_case identifiers
    ]

    class_patterns = [
        r"\bclass\s+\w+",  # class definitions
        r"\b[A-Z][a-z]*[A-Z]\w*\b",  # PascalCase class names
        r"\b[A-Z][a-z]+\b",  # Simple class names
    ]

    variable_patterns = [
        r"\b[a-z_]\w*\b",  # simple variables
        r"\bself\.\w+",  # instance variables
        r"\bthis\.\w+",  # JavaScript this references
    ]

    file_patterns = [
        r"\b\w+\.(py|js|ts|java|cpp|c|rs|go|php|rb)\b",  # file extensions
        r"\b\w+/\w+",  # path components
        r"\b__\w+__\b",  # Python special methods/attributes
    ]

    # High-level patterns (concepts and relationships)
    concept_patterns = [
        r"\b(pattern|design|architecture|framework|structure)\b",
        r"\b(algorithm|approach|method|technique|strategy)\b",
        r"\b(implementation|solution|system|process)\b",
        r"\b(relationship|connection|dependency|inheritance)\b",
        r"\b(interface|abstraction|encapsulation|polymorphism)\b",
    ]

    relationship_patterns = [
        r"\b(inherits?|extends?|implements?|uses?|calls?)\b",
        r"\b(connects?|links?|relates?|associates?)\b",
        r"\b(depends?|requires?|imports?|includes?)\b",
        r"\b(parent|child|ancestor|descendant)\b",
        r"\b(base|derived|super|sub)\b",
    ]

    technical_patterns = [
        r"\b(api|sdk|library|module|package|namespace)\b",
        r"\b(database|query|index|schema|model)\b",
        r"\b(service|component|middleware|handler)\b",
        r"\b(configuration|settings|parameters|options)\b",
        r"\b(error|exception|bug|issue|problem)\b",
    ]


class KeywordExtractor:
    """
    Intelligent keyword extractor that separates high-level conceptual keywords
    from low-level entity-specific keywords for optimal retrieval strategy selection.
    """

    def __init__(self):
        self.patterns = KeywordPatterns()
        self.logger = logging.getLogger(__name__)

        # Pre-compiled regex patterns for better performance
        self._compile_patterns()

        # Stop words and common terms to filter out
        self.stop_words = {
            "a",
            "an",
            "and",
            "are",
            "as",
            "at",
            "be",
            "by",
            "for",
            "from",
            "has",
            "he",
            "in",
            "is",
            "it",
            "its",
            "of",
            "on",
            "that",
            "the",
            "to",
            "was",
            "will",
            "with",
            "how",
            "what",
            "where",
            "when",
            "why",
            "can",
            "could",
            "should",
            "would",
            "do",
            "does",
            "did",
            "have",
            "had",
            "this",
            "these",
            "those",
            "i",
            "you",
            "we",
            "they",
        }

        # Technical vocabulary for classification
        self.technical_vocabulary = self._load_technical_vocabulary()
        self.concept_vocabulary = self._load_concept_vocabulary()

    def _compile_patterns(self):
        """Pre-compile regex patterns for better performance."""
        self.compiled_patterns = {
            "function": [re.compile(p, re.IGNORECASE) for p in self.patterns.function_patterns],
            "class": [re.compile(p, re.IGNORECASE) for p in self.patterns.class_patterns],
            "variable": [re.compile(p, re.IGNORECASE) for p in self.patterns.variable_patterns],
            "file": [re.compile(p, re.IGNORECASE) for p in self.patterns.file_patterns],
            "concept": [re.compile(p, re.IGNORECASE) for p in self.patterns.concept_patterns],
            "relationship": [re.compile(p, re.IGNORECASE) for p in self.patterns.relationship_patterns],
            "technical": [re.compile(p, re.IGNORECASE) for p in self.patterns.technical_patterns],
        }

    def _load_technical_vocabulary(self) -> set[str]:
        """Load technical vocabulary for keyword classification."""
        return {
            # Programming languages
            "python",
            "javascript",
            "typescript",
            "java",
            "cpp",
            "c++",
            "rust",
            "go",
            "php",
            "ruby",
            "swift",
            "kotlin",
            "scala",
            "clojure",
            "haskell",
            "erlang",
            "elixir",
            # Frameworks and libraries
            "react",
            "vue",
            "angular",
            "django",
            "flask",
            "fastapi",
            "spring",
            "express",
            "laravel",
            "rails",
            "tornado",
            "asyncio",
            "pandas",
            "numpy",
            "tensorflow",
            "pytorch",
            "keras",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "plotly",
            # Technologies
            "docker",
            "kubernetes",
            "terraform",
            "ansible",
            "jenkins",
            "github",
            "gitlab",
            "aws",
            "azure",
            "gcp",
            "redis",
            "mongodb",
            "postgresql",
            "mysql",
            "sqlite",
            "elasticsearch",
            "kafka",
            "rabbitmq",
            "celery",
            "nginx",
            "apache",
            # Patterns and concepts
            "mvc",
            "mvp",
            "mvvm",
            "singleton",
            "factory",
            "observer",
            "decorator",
            "adapter",
            "facade",
            "proxy",
            "strategy",
            "command",
            "state",
            "template",
            "repository",
            "service",
            "dao",
            "dto",
            "api",
            "rest",
            "graphql",
            "rpc",
        }

    def _load_concept_vocabulary(self) -> set[str]:
        """Load conceptual vocabulary for high-level keyword classification."""
        return {
            # Architecture and design
            "architecture",
            "design",
            "pattern",
            "structure",
            "framework",
            "system",
            "approach",
            "methodology",
            "paradigm",
            "principle",
            "concept",
            "model",
            # Relationships and connections
            "relationship",
            "connection",
            "dependency",
            "association",
            "composition",
            "aggregation",
            "inheritance",
            "polymorphism",
            "encapsulation",
            "abstraction",
            # Process and workflow
            "process",
            "workflow",
            "pipeline",
            "sequence",
            "flow",
            "lifecycle",
            "strategy",
            "algorithm",
            "technique",
            "method",
            "procedure",
            "protocol",
            # Quality and performance
            "performance",
            "optimization",
            "efficiency",
            "scalability",
            "reliability",
            "maintainability",
            "testability",
            "usability",
            "security",
            "quality",
            # Problem-solving
            "problem",
            "solution",
            "issue",
            "challenge",
            "requirement",
            "specification",
            "implementation",
            "integration",
            "deployment",
            "configuration",
            "setup",
        }

    def extract_keywords(self, query: str) -> KeywordExtraction:
        """
        Enhanced multi-level keyword extraction with advanced analysis.

        Args:
            query: The search query to analyze

        Returns:
            KeywordExtraction with categorized keywords
        """
        try:
            # Normalize the query
            normalized_query = self._normalize_query(query)

            # Perform comprehensive multi-level keyword analysis
            keyword_analysis = self._analyze_keywords_multilevel(normalized_query)

            # Build the enhanced result
            extraction = KeywordExtraction(
                low_level_keywords=keyword_analysis["low_level_keywords"],
                high_level_keywords=keyword_analysis["high_level_keywords"],
                entity_names=keyword_analysis["entity_names"],
                concept_terms=keyword_analysis["concept_terms"],
                technical_terms=keyword_analysis["technical_terms"],
                relationship_indicators=keyword_analysis["relationship_indicators"],
            )

            # Store advanced analysis in extraction metadata if available
            if hasattr(extraction, "metadata"):
                extraction.metadata = keyword_analysis.get("analysis_metadata", {})

            self.logger.debug(
                f"Enhanced keyword extraction from query '{query[:50]}...' complete: "
                f"{len(extraction.get_all_keywords())} total keywords across "
                f"{keyword_analysis['analysis_metadata']['keyword_levels']} levels"
            )

            return extraction

        except Exception as e:
            self.logger.error(f"Error in enhanced keyword extraction from query '{query}': {e}")
            return KeywordExtraction()

    def _analyze_keywords_multilevel(self, query: str) -> dict[str, any]:
        """Comprehensive multi-level keyword analysis."""
        # Stage 1: Basic extraction
        raw_keywords = self._extract_raw_keywords(query)

        # Stage 2: Advanced classification with multiple levels
        classified_keywords = self._classify_keywords_advanced(raw_keywords, query)

        # Stage 3: Specialized entity extraction
        entity_analysis = self._extract_entities_advanced(query)

        # Stage 4: Enhanced relationship analysis
        relationship_analysis = self._extract_relationships_advanced(query)

        # Stage 5: Semantic level analysis
        semantic_levels = self._analyze_semantic_levels(query, classified_keywords)

        # Stage 6: Contextual keyword enhancement
        contextual_enhancement = self._enhance_keywords_contextually(query, classified_keywords)

        # Combine all analysis results
        result = {
            "low_level_keywords": classified_keywords.get("low_level", []),
            "high_level_keywords": classified_keywords.get("high_level", []),
            "entity_names": entity_analysis["entities"],
            "concept_terms": classified_keywords.get("concepts", []),
            "technical_terms": classified_keywords.get("technical", []),
            "relationship_indicators": relationship_analysis["relationships"],
            "analysis_metadata": {
                "keyword_levels": len(semantic_levels),
                "semantic_levels": semantic_levels,
                "entity_analysis": entity_analysis,
                "relationship_analysis": relationship_analysis,
                "contextual_enhancement": contextual_enhancement,
                "extraction_quality": self._calculate_extraction_quality(classified_keywords, entity_analysis, relationship_analysis),
            },
        }

        return result

    def _classify_keywords_advanced(self, keywords: list[str], full_query: str) -> dict[str, list[str]]:
        """Advanced keyword classification with enhanced multi-level analysis."""
        classified = {
            "high_level": [],
            "low_level": [],
            "concepts": [],
            "technical": [],
            "micro_level": [],  # Very specific identifiers
            "macro_level": [],  # Broad architectural terms
            "domain_specific": [],  # Domain-specific terminology
            "contextual": [],  # Context-dependent terms
        }

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Multi-level classification
            levels = self._determine_keyword_levels(keyword, full_query)

            # Primary classification
            if keyword_lower in self.concept_vocabulary:
                classified["concepts"].append(keyword)
                classified["high_level"].append(keyword)
                if "macro" in levels:
                    classified["macro_level"].append(keyword)
            elif keyword_lower in self.technical_vocabulary:
                classified["technical"].append(keyword)
                classified["high_level"].append(keyword)
                if "domain" in levels:
                    classified["domain_specific"].append(keyword)
            else:
                # Pattern-based classification with level analysis
                classification_result = self._classify_keyword_by_patterns(keyword, full_query, levels)

                for category, should_include in classification_result.items():
                    if should_include and category in classified:
                        classified[category].append(keyword)

        # Remove duplicates while preserving order
        for category in classified:
            classified[category] = list(dict.fromkeys(classified[category]))

        return classified

    def _determine_keyword_levels(self, keyword: str, full_query: str) -> list[str]:
        """Determine the semantic levels of a keyword."""
        levels = []

        # Micro-level indicators (very specific)
        if (
            len(keyword) <= 3  # Short abbreviations
            or keyword.count("_") > 1  # Multi-underscore identifiers
            or any(c.isdigit() for c in keyword)  # Contains numbers
            or keyword.isupper()
            and len(keyword) > 1  # All caps (likely acronym)
        ):
            levels.append("micro")

        # Macro-level indicators (architectural/broad)
        macro_terms = {"architecture", "system", "framework", "platform", "infrastructure", "ecosystem"}
        if keyword.lower() in macro_terms or any(term in keyword.lower() for term in macro_terms):
            levels.append("macro")

        # Domain-specific indicators
        if keyword.lower() in self.technical_vocabulary:
            levels.append("domain")

        # Contextual indicators (depends on surrounding words)
        contextual_patterns = [
            (r"\b" + re.escape(keyword) + r"\s+(pattern|design|approach)\b", "contextual"),
            (r"\b(in|on|with|for)\s+" + re.escape(keyword) + r"\b", "contextual"),
            (r"\b" + re.escape(keyword) + r"\s+(class|function|method)\b", "contextual"),
        ]

        for pattern, level_type in contextual_patterns:
            if re.search(pattern, full_query, re.IGNORECASE):
                levels.append(level_type)
                break

        return levels if levels else ["standard"]

    def _classify_keyword_by_patterns(self, keyword: str, full_query: str, levels: list[str]) -> dict[str, bool]:
        """Classify a keyword using pattern analysis and level information."""
        classification = {
            "low_level": False,
            "high_level": False,
            "micro_level": False,
            "macro_level": False,
            "domain_specific": False,
            "contextual": False,
        }

        # Apply level-based classification
        if "micro" in levels:
            classification["micro_level"] = True
            classification["low_level"] = True

        if "macro" in levels:
            classification["macro_level"] = True
            classification["high_level"] = True

        if "domain" in levels:
            classification["domain_specific"] = True
            classification["high_level"] = True

        if "contextual" in levels:
            classification["contextual"] = True

        # Pattern-based classification
        is_low_level = self._is_low_level_keyword(keyword, full_query)
        is_high_level = self._is_high_level_keyword(keyword, full_query)

        if is_low_level and not any(classification.values()):
            classification["low_level"] = True
        elif is_high_level and not any(classification.values()):
            classification["high_level"] = True
        elif not any(classification.values()):
            # Default classification based on characteristics
            if self._has_specific_characteristics(keyword):
                classification["low_level"] = True
            else:
                classification["high_level"] = True

        return classification

    def _extract_entities_advanced(self, query: str) -> dict[str, any]:
        """Advanced entity extraction with categorization and confidence scoring."""
        entities = {"entities": [], "entity_types": {}, "entity_confidence": {}, "entity_contexts": {}}

        # Extract different types of entities
        entity_extractors = {
            "quoted_entities": self._extract_quoted_entities,
            "camel_case_entities": self._extract_camel_case_entities,
            "snake_case_entities": self._extract_snake_case_entities,
            "function_entities": self._extract_function_entities,
            "file_path_entities": self._extract_file_path_entities,
            "url_entities": self._extract_url_entities,
        }

        all_entities = []
        entity_sources = {}

        for extractor_name, extractor_func in entity_extractors.items():
            extracted = extractor_func(query)
            for entity in extracted:
                all_entities.append(entity)
                entity_sources[entity] = extractor_name

        # Remove duplicates and score confidence
        unique_entities = list(dict.fromkeys(all_entities))

        for entity in unique_entities:
            # Determine entity type and confidence
            entity_type = self._classify_entity_type(entity, query)
            confidence = self._calculate_entity_confidence(entity, query, entity_sources.get(entity, "unknown"))
            context = self._extract_entity_context(entity, query)

            entities["entities"].append(entity)
            entities["entity_types"][entity] = entity_type
            entities["entity_confidence"][entity] = confidence
            entities["entity_contexts"][entity] = context

        return entities

    def _extract_relationships_advanced(self, query: str) -> dict[str, any]:
        """Advanced relationship extraction with type classification and strength scoring."""
        relationships = {"relationships": [], "relationship_types": {}, "relationship_strengths": {}, "relationship_contexts": {}}

        # Extract basic relationship indicators
        basic_relationships = self._extract_relationship_indicators(query)

        # Advanced relationship pattern extraction
        advanced_patterns = {
            "hierarchical": r"\b(inherit|extend|implement|derive|parent|child|base|super|sub)\b",
            "compositional": r"\b(contain|include|compose|aggregate|part\s+of|member\s+of)\b",
            "dependency": r"\b(depend|require|need|use|import|reference|call)\b",
            "association": r"\b(associate|connect|link|relate|interact|communicate)\b",
            "temporal": r"\b(before|after|during|while|then|next|sequence|follow)\b",
            "causal": r"\b(cause|effect|result|trigger|lead|produce|generate)\b",
            "comparative": r"\b(similar|different|like|unlike|compare|contrast|versus)\b",
        }

        all_relationships = set(basic_relationships)
        relationship_sources = dict.fromkeys(basic_relationships, "basic")

        # Extract advanced relationship patterns
        for rel_type, pattern in advanced_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            for match in matches:
                all_relationships.add(match.lower())
                relationship_sources[match.lower()] = rel_type

        # Analyze each relationship
        for relationship in all_relationships:
            rel_type = relationship_sources.get(relationship, "unknown")
            strength = self._calculate_relationship_strength(relationship, query)
            context = self._extract_relationship_context(relationship, query)

            relationships["relationships"].append(relationship)
            relationships["relationship_types"][relationship] = rel_type
            relationships["relationship_strengths"][relationship] = strength
            relationships["relationship_contexts"][relationship] = context

        return relationships

    def _analyze_semantic_levels(self, query: str, classified_keywords: dict[str, list[str]]) -> dict[str, any]:
        """Analyze semantic levels in the query for hierarchical understanding."""
        levels = {
            "surface_level": [],  # Direct, explicit terms
            "intermediate_level": [],  # Inferred, contextual terms
            "deep_level": [],  # Abstract, conceptual terms
            "meta_level": [],  # Meta-cognitive, reflective terms
        }

        all_keywords = []
        for keyword_list in classified_keywords.values():
            all_keywords.extend(keyword_list)

        for keyword in set(all_keywords):
            semantic_depth = self._calculate_semantic_depth(keyword, query)

            if semantic_depth <= 0.25:
                levels["surface_level"].append(keyword)
            elif semantic_depth <= 0.50:
                levels["intermediate_level"].append(keyword)
            elif semantic_depth <= 0.75:
                levels["deep_level"].append(keyword)
            else:
                levels["meta_level"].append(keyword)

        return levels

    def _enhance_keywords_contextually(self, query: str, classified_keywords: dict[str, list[str]]) -> dict[str, any]:
        """Enhance keywords with contextual information and relationships."""
        enhancement = {"keyword_clusters": {}, "contextual_boosting": {}, "semantic_expansion": {}, "domain_alignment": {}}

        # Identify keyword clusters (related terms that appear together)
        all_keywords = []
        for keyword_list in classified_keywords.values():
            all_keywords.extend(keyword_list)

        clusters = self._identify_keyword_clusters(all_keywords, query)
        enhancement["keyword_clusters"] = clusters

        # Contextual boosting (keywords that gain importance from context)
        for keyword in set(all_keywords):
            boost_score = self._calculate_contextual_boost(keyword, query)
            if boost_score > 0.3:
                enhancement["contextual_boosting"][keyword] = boost_score

        # Semantic expansion (related terms that could be inferred)
        expansion = self._generate_semantic_expansion(all_keywords, query)
        enhancement["semantic_expansion"] = expansion

        # Domain alignment (how well keywords align with technical domains)
        domain_scores = self._calculate_domain_alignment(all_keywords)
        enhancement["domain_alignment"] = domain_scores

        return enhancement

    # Helper methods for advanced extraction
    def _extract_quoted_entities(self, query: str) -> list[str]:
        """Extract entities from quoted strings."""
        return re.findall(r'["\']([^"\']+)["\']', query)

    def _extract_camel_case_entities(self, query: str) -> list[str]:
        """Extract camelCase identifiers."""
        return re.findall(r"\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b", query)

    def _extract_snake_case_entities(self, query: str) -> list[str]:
        """Extract snake_case identifiers."""
        return re.findall(r"\b[a-z]+(?:_[a-z0-9]+)+\b", query)

    def _extract_function_entities(self, query: str) -> list[str]:
        """Extract function call patterns."""
        return re.findall(r"\b(\w+)\(\)", query)

    def _extract_file_path_entities(self, query: str) -> list[str]:
        """Extract file paths and extensions."""
        paths = re.findall(r"\b\w+\.[a-z]{2,4}\b", query)  # Simple file extensions
        paths.extend(re.findall(r"\b\w+/\w+(?:/\w+)*\b", query))  # Path patterns
        return paths

    def _extract_url_entities(self, query: str) -> list[str]:
        """Extract URL patterns."""
        return re.findall(r"https?://[^\s]+", query)

    def _classify_entity_type(self, entity: str, query: str) -> str:
        """Classify the type of an extracted entity."""
        if re.match(r"^[a-z][a-zA-Z0-9]*[A-Z]", entity):
            return "camelCase_identifier"
        elif "_" in entity and entity.islower():
            return "snake_case_identifier"
        elif entity.endswith("()"):
            return "function_call"
        elif "." in entity and any(entity.endswith(ext) for ext in [".py", ".js", ".ts", ".java"]):
            return "file_name"
        elif entity.startswith("http"):
            return "url"
        elif entity.isupper() and len(entity) > 1:
            return "acronym"
        else:
            return "generic_entity"

    def _calculate_entity_confidence(self, entity: str, query: str, source: str) -> float:
        """Calculate confidence score for entity extraction."""
        base_confidence = 0.5

        # Source-based confidence
        source_weights = {
            "quoted_entities": 0.9,
            "function_entities": 0.8,
            "camel_case_entities": 0.7,
            "snake_case_entities": 0.7,
            "file_path_entities": 0.6,
            "url_entities": 0.8,
        }

        confidence = source_weights.get(source, base_confidence)

        # Length-based adjustment
        if len(entity) <= 2:
            confidence *= 0.6  # Very short entities are less reliable
        elif len(entity) > 15:
            confidence *= 0.8  # Very long entities might be noise

        # Context-based adjustment
        if entity.lower() in query.lower():
            confidence *= 1.1  # Mentioned explicitly

        return min(confidence, 1.0)

    def _extract_entity_context(self, entity: str, query: str) -> dict[str, any]:
        """Extract contextual information around an entity."""
        words = query.split()
        context = {"preceding_words": [], "following_words": [], "position": -1}

        for i, word in enumerate(words):
            if entity.lower() in word.lower():
                context["position"] = i
                context["preceding_words"] = words[max(0, i - 2) : i]
                context["following_words"] = words[i + 1 : min(len(words), i + 3)]
                break

        return context

    def _calculate_relationship_strength(self, relationship: str, query: str) -> float:
        """Calculate the strength/importance of a relationship indicator."""
        # Count occurrences
        occurrences = query.lower().count(relationship.lower())

        # Base strength by relationship type
        strong_relationships = {"inherit", "extend", "implement", "depend", "require"}
        moderate_relationships = {"use", "call", "connect", "relate"}

        if relationship.lower() in strong_relationships:
            base_strength = 0.8
        elif relationship.lower() in moderate_relationships:
            base_strength = 0.6
        else:
            base_strength = 0.4

        # Adjust for frequency
        strength = base_strength * min(1.0 + (occurrences - 1) * 0.2, 1.5)

        return min(strength, 1.0)

    def _extract_relationship_context(self, relationship: str, query: str) -> dict[str, any]:
        """Extract context around relationship indicators."""
        # Find the relationship in the query and get surrounding context
        pattern = r"(\S+\s+)?" + re.escape(relationship) + r"(\s+\S+)?"
        matches = re.finditer(pattern, query, re.IGNORECASE)

        contexts = []
        for match in matches:
            contexts.append(
                {
                    "full_match": match.group(0),
                    "before": match.group(1) if match.group(1) else "",
                    "after": match.group(2) if match.group(2) else "",
                    "position": match.start(),
                }
            )

        return {"contexts": contexts, "count": len(contexts)}

    def _calculate_semantic_depth(self, keyword: str, query: str) -> float:
        """Calculate the semantic depth of a keyword (0=surface, 1=deep)."""
        depth_indicators = {
            "surface": ["show", "get", "find", "list", "display"],
            "intermediate": ["how", "what", "where", "when", "which"],
            "deep": ["why", "understand", "explain", "analyze", "design"],
            "meta": ["optimize", "improve", "architect", "evaluate", "assess"],
        }

        keyword_lower = keyword.lower()

        for level, indicators in depth_indicators.items():
            if keyword_lower in indicators or any(ind in keyword_lower for ind in indicators):
                if level == "surface":
                    return 0.2
                elif level == "intermediate":
                    return 0.4
                elif level == "deep":
                    return 0.7
                elif level == "meta":
                    return 0.9

        # Default depth based on abstraction level
        if keyword_lower in self.concept_vocabulary:
            return 0.6
        elif keyword_lower in self.technical_vocabulary:
            return 0.4
        else:
            return 0.3

    def _identify_keyword_clusters(self, keywords: list[str], query: str) -> dict[str, list[str]]:
        """Identify clusters of related keywords."""
        clusters = {}

        # Simple clustering based on co-occurrence and semantic similarity
        for i, keyword1 in enumerate(keywords):
            for j, keyword2 in enumerate(keywords[i + 1 :], i + 1):
                similarity = self._calculate_keyword_similarity(keyword1, keyword2, query)
                if similarity > 0.6:
                    cluster_key = f"cluster_{min(i,j)}"
                    if cluster_key not in clusters:
                        clusters[cluster_key] = []
                    if keyword1 not in clusters[cluster_key]:
                        clusters[cluster_key].append(keyword1)
                    if keyword2 not in clusters[cluster_key]:
                        clusters[cluster_key].append(keyword2)

        return clusters

    def _calculate_contextual_boost(self, keyword: str, query: str) -> float:
        """Calculate how much a keyword's importance is boosted by context."""
        boost = 0.0

        # Boost for keywords that appear with qualifiers
        qualifiers = ["main", "primary", "key", "important", "core", "central"]
        for qualifier in qualifiers:
            if re.search(f"\b{qualifier}\\s+{re.escape(keyword)}\b", query, re.IGNORECASE):
                boost += 0.3

        # Boost for keywords in questions
        if re.search(f"\b(what|how|why).*{re.escape(keyword)}\b", query, re.IGNORECASE):
            boost += 0.2

        # Boost for emphasized keywords (caps, quotes)
        if keyword.isupper() or f'"{keyword}"' in query or f"'{keyword}'" in query:
            boost += 0.4

        return boost

    def _generate_semantic_expansion(self, keywords: list[str], query: str) -> dict[str, list[str]]:
        """Generate semantically related terms that could expand the query."""
        expansion = {}

        # Simple expansion based on known relationships
        expansion_rules = {
            "function": ["method", "procedure", "routine"],
            "class": ["object", "type", "structure"],
            "variable": ["field", "attribute", "property"],
            "database": ["table", "schema", "query"],
            "api": ["endpoint", "service", "interface"],
        }

        for keyword in keywords:
            keyword_lower = keyword.lower()
            for base_term, related_terms in expansion_rules.items():
                if base_term in keyword_lower:
                    expansion[keyword] = related_terms
                    break

        return expansion

    def _calculate_domain_alignment(self, keywords: list[str]) -> dict[str, dict[str, float]]:
        """Calculate how well keywords align with different technical domains."""
        domains = {
            "web_development": ["html", "css", "javascript", "react", "vue", "angular", "frontend", "backend"],
            "data_science": ["data", "analysis", "machine", "learning", "model", "pandas", "numpy"],
            "system_architecture": ["architecture", "design", "pattern", "system", "microservice", "api"],
            "database": ["database", "sql", "query", "table", "schema", "index"],
            "mobile_development": ["mobile", "app", "android", "ios", "react", "native"],
        }

        alignment = {}

        for keyword in keywords:
            keyword_lower = keyword.lower()
            keyword_alignment = {}

            for domain, domain_terms in domains.items():
                score = sum(1 for term in domain_terms if term in keyword_lower)
                keyword_alignment[domain] = score / len(domain_terms)

            alignment[keyword] = keyword_alignment

        return alignment

    def _calculate_keyword_similarity(self, keyword1: str, keyword2: str, query: str) -> float:
        """Calculate similarity between two keywords based on various factors."""
        # Lexical similarity (simple character overlap)
        common_chars = set(keyword1.lower()) & set(keyword2.lower())
        lexical_sim = len(common_chars) / max(len(set(keyword1.lower())), len(set(keyword2.lower())), 1)

        # Contextual similarity (appear near each other)
        words = query.lower().split()
        pos1 = next((i for i, word in enumerate(words) if keyword1.lower() in word), -1)
        pos2 = next((i for i, word in enumerate(words) if keyword2.lower() in word), -1)

        if pos1 != -1 and pos2 != -1:
            distance = abs(pos1 - pos2)
            contextual_sim = max(0, 1 - distance / 5)  # Closer words have higher similarity
        else:
            contextual_sim = 0

        # Semantic similarity (same category)
        semantic_sim = 0
        if (keyword1.lower() in self.technical_vocabulary and keyword2.lower() in self.technical_vocabulary) or (
            keyword1.lower() in self.concept_vocabulary and keyword2.lower() in self.concept_vocabulary
        ):
            semantic_sim = 0.5

        return lexical_sim * 0.3 + contextual_sim * 0.4 + semantic_sim * 0.3

    def _calculate_extraction_quality(
        self, classified_keywords: dict[str, list[str]], entity_analysis: dict[str, any], relationship_analysis: dict[str, any]
    ) -> dict[str, float]:
        """Calculate quality metrics for the keyword extraction."""
        total_keywords = sum(len(keywords) for keywords in classified_keywords.values())
        total_entities = len(entity_analysis["entities"])
        total_relationships = len(relationship_analysis["relationships"])

        quality = {
            "keyword_diversity": len(classified_keywords) / max(1, total_keywords),
            "entity_confidence_avg": sum(entity_analysis["entity_confidence"].values()) / max(1, total_entities),
            "relationship_strength_avg": sum(relationship_analysis["relationship_strengths"].values()) / max(1, total_relationships),
            "extraction_completeness": min(1.0, (total_keywords + total_entities + total_relationships) / 10),
            "classification_balance": 1.0
            - abs(len(classified_keywords.get("high_level", [])) - len(classified_keywords.get("low_level", []))) / max(1, total_keywords),
        }

        return quality

    def _normalize_query(self, query: str) -> str:
        """Normalize the query for consistent processing."""
        # Convert to lowercase
        normalized = query.lower().strip()

        # Remove extra whitespace
        normalized = re.sub(r"\s+", " ", normalized)

        # Handle contractions
        contractions = {
            "don't": "do not",
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
        }

        for contraction, expansion in contractions.items():
            normalized = normalized.replace(contraction, expansion)

        return normalized

    def _extract_raw_keywords(self, query: str) -> list[str]:
        """Extract raw keywords using basic tokenization."""
        # Split on whitespace and punctuation
        tokens = re.findall(r"\b\w+\b", query)

        # Filter out stop words and very short words
        keywords = [token for token in tokens if len(token) > 2 and token.lower() not in self.stop_words]

        return keywords

    def _classify_keywords(self, keywords: list[str], full_query: str) -> dict[str, list[str]]:
        """Classify keywords into different categories."""
        classified = {
            "high_level": [],
            "low_level": [],
            "concepts": [],
            "technical": [],
        }

        for keyword in keywords:
            keyword_lower = keyword.lower()

            # Check if it's a concept keyword
            if keyword_lower in self.concept_vocabulary:
                classified["concepts"].append(keyword)
                classified["high_level"].append(keyword)
                continue

            # Check if it's a technical term
            if keyword_lower in self.technical_vocabulary:
                classified["technical"].append(keyword)
                classified["high_level"].append(keyword)
                continue

            # Check pattern-based classification
            is_low_level = self._is_low_level_keyword(keyword, full_query)
            is_high_level = self._is_high_level_keyword(keyword, full_query)

            if is_low_level:
                classified["low_level"].append(keyword)
            elif is_high_level:
                classified["high_level"].append(keyword)
            else:
                # Default classification based on characteristics
                if self._has_specific_characteristics(keyword):
                    classified["low_level"].append(keyword)
                else:
                    classified["high_level"].append(keyword)

        return classified

    def _is_low_level_keyword(self, keyword: str, full_query: str) -> bool:
        """Check if a keyword is low-level (entity-specific)."""
        # Check against compiled patterns
        for pattern_list in [
            self.compiled_patterns["function"],
            self.compiled_patterns["class"],
            self.compiled_patterns["variable"],
            self.compiled_patterns["file"],
        ]:
            for pattern in pattern_list:
                if pattern.search(keyword) or pattern.search(full_query):
                    return True

        # Check for specific naming conventions
        if (
            # CamelCase or PascalCase
            re.match(r"^[a-z][a-zA-Z0-9]*[A-Z]", keyword)
            or re.match(r"^[A-Z][a-z]*[A-Z]", keyword)
            or
            # snake_case
            "_" in keyword
            or
            # Has numbers (likely identifiers)
            re.search(r"\d", keyword)
            or
            # Contains specific characters
            any(char in keyword for char in ["$", "@", "#"])
        ):
            return True

        return False

    def _is_high_level_keyword(self, keyword: str, full_query: str) -> bool:
        """Check if a keyword is high-level (conceptual)."""
        # Check against compiled patterns
        for pattern_list in [
            self.compiled_patterns["concept"],
            self.compiled_patterns["relationship"],
            self.compiled_patterns["technical"],
        ]:
            for pattern in pattern_list:
                if pattern.search(keyword) or pattern.search(full_query):
                    return True

        return False

    def _has_specific_characteristics(self, keyword: str) -> bool:
        """Check if keyword has characteristics of specific entities."""
        return (
            # Very short (likely abbreviations or specific names)
            len(keyword) <= 3
            or
            # Contains underscores (likely code identifiers)
            "_" in keyword
            or
            # Mixed case (likely identifiers)
            keyword != keyword.lower()
            and keyword != keyword.upper()
            or
            # Ends with common code suffixes
            keyword.endswith(("er", "or", "ing", "ed"))
            or
            # Contains numbers
            any(c.isdigit() for c in keyword)
        )

    def _extract_entity_names(self, query: str) -> list[str]:
        """Extract specific entity names from the query."""
        entity_names = []

        # Look for quoted strings (explicit entities)
        quoted_entities = re.findall(r'["\']([^"\']+)["\']', query)
        entity_names.extend(quoted_entities)

        # Look for camelCase/PascalCase identifiers
        camel_case = re.findall(r"\b[a-z][a-zA-Z0-9]*[A-Z][a-zA-Z0-9]*\b", query)
        entity_names.extend(camel_case)

        pascal_case = re.findall(r"\b[A-Z][a-z]*[A-Z][a-zA-Z0-9]*\b", query)
        entity_names.extend(pascal_case)

        # Look for snake_case identifiers
        snake_case = re.findall(r"\b[a-z]+_[a-z_]+\b", query)
        entity_names.extend(snake_case)

        # Look for function calls
        function_calls = re.findall(r"\b(\w+)\(\)", query)
        entity_names.extend(function_calls)

        # Remove duplicates and filter
        unique_entities = list(set(entity_names))
        filtered_entities = [entity for entity in unique_entities if len(entity) > 1 and entity.lower() not in self.stop_words]

        return filtered_entities

    def _extract_relationship_indicators(self, query: str) -> list[str]:
        """Extract words that indicate relationships or connections."""
        relationship_words = []

        for pattern in self.compiled_patterns["relationship"]:
            matches = pattern.findall(query)
            relationship_words.extend(matches)

        # Additional relationship indicators
        relationship_indicators = [
            "between",
            "among",
            "connects",
            "links",
            "relates",
            "associates",
            "inherits",
            "extends",
            "implements",
            "uses",
            "calls",
            "invokes",
            "depends",
            "requires",
            "imports",
            "includes",
            "contains",
            "parent",
            "child",
            "ancestor",
            "descendant",
            "base",
            "derived",
            "similar",
            "different",
            "compare",
            "contrast",
            "like",
            "unlike",
        ]

        query_words = set(query.lower().split())
        found_indicators = [indicator for indicator in relationship_indicators if indicator in query_words]

        relationship_words.extend(found_indicators)

        return list(set(relationship_words))

    def analyze_query_characteristics(self, query: str) -> dict[str, any]:
        """
        Analyze query to determine its characteristics for retrieval mode selection.

        Args:
            query: The search query to analyze

        Returns:
            Dictionary with query characteristics
        """
        keywords = self.extract_keywords(query)
        normalized_query = self._normalize_query(query)

        characteristics = {
            "has_specific_entities": len(keywords.entity_names) > 0,
            "has_low_level_focus": len(keywords.low_level_keywords) > len(keywords.high_level_keywords),
            "has_high_level_focus": len(keywords.high_level_keywords) > len(keywords.low_level_keywords),
            "has_relationships": len(keywords.relationship_indicators) > 0,
            "has_technical_terms": len(keywords.technical_terms) > 0,
            "has_concepts": len(keywords.concept_terms) > 0,
            "entity_density": len(keywords.entity_names) / max(1, len(normalized_query.split())),
            "concept_density": len(keywords.concept_terms) / max(1, len(normalized_query.split())),
            "relationship_density": len(keywords.relationship_indicators) / max(1, len(normalized_query.split())),
            "query_length": len(query),
            "word_count": len(normalized_query.split()),
            "complexity_score": self._calculate_complexity_score(keywords, normalized_query),
        }

        return characteristics

    def _calculate_complexity_score(self, keywords: KeywordExtraction, query: str) -> float:
        """Calculate a complexity score for the query."""
        # Base complexity factors
        word_count = len(query.split())
        entity_count = len(keywords.entity_names)
        concept_count = len(keywords.concept_terms)
        relationship_count = len(keywords.relationship_indicators)

        # Calculate complexity score (0.0 to 1.0)
        complexity = 0.0

        # Word count contribution (20%)
        complexity += min(word_count / 20.0, 1.0) * 0.2

        # Entity complexity (30%)
        complexity += min(entity_count / 5.0, 1.0) * 0.3

        # Concept complexity (25%)
        complexity += min(concept_count / 3.0, 1.0) * 0.25

        # Relationship complexity (25%)
        complexity += min(relationship_count / 3.0, 1.0) * 0.25

        return min(complexity, 1.0)

    def recommend_retrieval_mode(self, query: str) -> tuple[str, float]:
        """
        Recommend the best retrieval mode for a given query.

        Args:
            query: The search query to analyze

        Returns:
            Tuple of (recommended_mode, confidence_score)
        """
        characteristics = self.analyze_query_characteristics(query)

        # Decision logic based on query characteristics
        entity_focused = (
            characteristics["has_specific_entities"] and characteristics["has_low_level_focus"] and characteristics["entity_density"] > 0.3
        )

        relationship_focused = characteristics["has_relationships"] and characteristics["relationship_density"] > 0.2

        concept_focused = (
            characteristics["has_high_level_focus"] and characteristics["has_concepts"] and characteristics["concept_density"] > 0.2
        )

        # Mode selection logic
        if entity_focused and not relationship_focused:
            return "local", 0.8
        elif relationship_focused and not entity_focused:
            return "global", 0.8
        elif concept_focused and relationship_focused:
            return "global", 0.7
        elif entity_focused and relationship_focused:
            return "hybrid", 0.7
        elif characteristics["complexity_score"] > 0.7:
            return "mix", 0.6
        else:
            return "hybrid", 0.5


# Factory function
_keyword_extractor_instance = None


def get_keyword_extractor() -> KeywordExtractor:
    """Get or create a KeywordExtractor instance."""
    global _keyword_extractor_instance
    if _keyword_extractor_instance is None:
        _keyword_extractor_instance = KeywordExtractor()
    return _keyword_extractor_instance
