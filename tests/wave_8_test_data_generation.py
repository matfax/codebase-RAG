#!/usr/bin/env python3
"""
Wave 8.0 Task 8.10: Test Data Generation Framework

This module generates synthetic test datasets for comprehensive validation,
creating realistic code repositories, query patterns, and edge cases
to thoroughly test the system under various conditions.
"""

import asyncio
import json
import logging
import random
import shutil
import string
import sys
import tempfile
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class CodeTemplate:
    """Template for generating code files"""

    language: str
    file_extension: str
    template: str
    variables: dict[str, list[str]]
    complexity_levels: dict[str, dict[str, Any]]


@dataclass
class QueryPattern:
    """Pattern for generating test queries"""

    category: str
    complexity: str
    template: str
    variables: dict[str, list[str]]
    expected_result_types: list[str]


@dataclass
class TestDataset:
    """Generated test dataset"""

    dataset_id: str
    name: str
    description: str
    creation_time: str
    file_count: int
    query_count: int
    languages: list[str]
    complexity_distribution: dict[str, int]
    files: list[str]
    queries: list[dict[str, Any]]
    ground_truth: dict[str, list[str]]
    metadata: dict[str, Any]


class TestDataGenerator:
    """Comprehensive test data generation framework"""

    def __init__(self, output_dir: str = "test_datasets"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()

        # Code templates for different languages
        self.code_templates = self._define_code_templates()

        # Query patterns for testing
        self.query_patterns = self._define_query_patterns()

        # Dataset configurations
        self.dataset_configs = self._define_dataset_configs()

    def _setup_logging(self) -> logging.Logger:
        """Setup logging for test data generation"""
        logger = logging.getLogger("test_data_generator")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _define_code_templates(self) -> dict[str, CodeTemplate]:
        """Define code templates for different languages"""
        return {
            "python": CodeTemplate(
                language="python",
                file_extension="py",
                template='''#!/usr/bin/env python3
"""
{module_description}
"""

import {imports}
from typing import {type_imports}
from dataclasses import dataclass

@dataclass
class {class_name}:
    """Class for {class_purpose}"""
    {class_attributes}
    
    def __init__(self, {init_params}):
        {init_body}
        
    def {method_name}(self, {method_params}) -> {return_type}:
        """Method for {method_purpose}"""
        {method_body}
        
    async def {async_method_name}(self, {async_params}) -> {async_return_type}:
        """Async method for {async_purpose}"""
        {async_body}
        
def {function_name}({function_params}) -> {function_return_type}:
    """Function for {function_purpose}"""
    {function_body}

async def {async_function_name}({async_function_params}) -> {async_function_return_type}:
    """Async function for {async_function_purpose}"""
    {async_function_body}

if __name__ == "__main__":
    {main_body}
''',
                variables={
                    "module_description": [
                        "Data processing utilities",
                        "User management module",
                        "File handling operations",
                        "Network communication tools",
                        "Database interface layer",
                        "Caching mechanisms",
                    ],
                    "imports": [
                        "os, sys, json",
                        "asyncio, logging",
                        "time, datetime",
                        "collections, itertools",
                        "pathlib, tempfile",
                        "random, uuid",
                    ],
                    "type_imports": ["Dict, List, Optional", "Any, Union, Tuple", "Callable, Iterator", "Protocol, TypeVar"],
                    "class_name": ["DataProcessor", "UserManager", "FileHandler", "NetworkClient", "DatabaseManager", "CacheService"],
                    "class_purpose": [
                        "handling data operations",
                        "managing user accounts",
                        "processing file operations",
                        "network communications",
                    ],
                    "method_name": [
                        "process_data",
                        "validate_input",
                        "transform_data",
                        "filter_results",
                        "calculate_metrics",
                        "format_output",
                    ],
                    "function_name": ["utility_function", "helper_method", "validator", "transformer", "calculator", "formatter"],
                },
                complexity_levels={
                    "simple": {"methods": 2, "functions": 1, "lines": 50},
                    "medium": {"methods": 5, "functions": 3, "lines": 150},
                    "complex": {"methods": 10, "functions": 5, "lines": 300},
                },
            ),
            "javascript": CodeTemplate(
                language="javascript",
                file_extension="js",
                template="""/**
 * {module_description}
 */

const {imports} = require('{import_path}');

class {class_name} {{
    constructor({constructor_params}) {{
        {constructor_body}
    }}
    
    {method_name}({method_params}) {{
        {method_body}
    }}
    
    async {async_method_name}({async_params}) {{
        {async_body}
    }}
}}

function {function_name}({function_params}) {{
    {function_body}
}}

async function {async_function_name}({async_function_params}) {{
    {async_function_body}
}}

module.exports = {{
    {class_name},
    {function_name},
    {async_function_name}
}};
""",
                variables={
                    "module_description": [
                        "Data processing utilities",
                        "User interface components",
                        "API client library",
                        "Utility functions",
                    ],
                    "class_name": ["DataManager", "ApiClient", "EventHandler", "Validator"],
                    "function_name": ["processData", "validateInput", "formatOutput", "calculateHash"],
                },
                complexity_levels={
                    "simple": {"methods": 2, "functions": 1, "lines": 40},
                    "medium": {"methods": 4, "functions": 2, "lines": 100},
                    "complex": {"methods": 8, "functions": 4, "lines": 200},
                },
            ),
            "typescript": CodeTemplate(
                language="typescript",
                file_extension="ts",
                template="""/**
 * {module_description}
 */

import {{ {imports} }} from '{import_path}';

interface {interface_name} {{
    {interface_properties}
}}

class {class_name} implements {interface_name} {{
    private {private_property}: {property_type};
    
    constructor({constructor_params}: {constructor_types}) {{
        {constructor_body}
    }}
    
    public {method_name}({method_params}: {method_types}): {return_type} {{
        {method_body}
    }}
    
    public async {async_method_name}({async_params}: {async_types}): Promise<{async_return_type}> {{
        {async_body}
    }}
}}

export function {function_name}({function_params}: {function_types}): {function_return_type} {{
    {function_body}
}}

export async function {async_function_name}({async_function_params}: {async_function_types}): Promise<{async_function_return_type}> {{
    {async_function_body}
}}

export {{ {class_name}, {interface_name} }};
""",
                variables={
                    "interface_name": ["IDataProcessor", "IUserManager", "IApiClient", "IValidator"],
                    "class_name": ["DataProcessor", "UserManager", "ApiClient", "Validator"],
                },
                complexity_levels={
                    "simple": {"methods": 2, "functions": 1, "lines": 60},
                    "medium": {"methods": 4, "functions": 2, "lines": 120},
                    "complex": {"methods": 8, "functions": 4, "lines": 250},
                },
            ),
        }

    def _define_query_patterns(self) -> dict[str, QueryPattern]:
        """Define query patterns for testing"""
        return {
            "simple_keyword": QueryPattern(
                category="keyword",
                complexity="simple",
                template="find {keyword}",
                variables={
                    "keyword": [
                        "function",
                        "class",
                        "method",
                        "variable",
                        "import",
                        "def",
                        "async",
                        "await",
                        "return",
                        "if",
                        "for",
                        "while",
                    ]
                },
                expected_result_types=["function", "class", "variable"],
            ),
            "semantic_search": QueryPattern(
                category="semantic",
                complexity="medium",
                template="find {concept} in {domain}",
                variables={
                    "concept": [
                        "error handling",
                        "data validation",
                        "user authentication",
                        "file processing",
                        "database operations",
                        "API endpoints",
                        "logging mechanisms",
                        "caching strategies",
                    ],
                    "domain": ["user management", "data processing", "file operations", "network communication", "authentication system"],
                },
                expected_result_types=["function", "class", "module"],
            ),
            "graph_rag_query": QueryPattern(
                category="graph_rag",
                complexity="complex",
                template="trace {operation} from {start_point} to {end_point}",
                variables={
                    "operation": ["data flow", "function calls", "dependency chain", "error propagation", "authentication flow"],
                    "start_point": ["user login", "API request", "data input", "file upload"],
                    "end_point": ["database storage", "response generation", "file processing", "error handling", "user notification"],
                },
                expected_result_types=["function", "class", "module", "flow"],
            ),
            "architectural_analysis": QueryPattern(
                category="architecture",
                complexity="complex",
                template="analyze {pattern_type} patterns in {scope}",
                variables={
                    "pattern_type": ["design", "architectural", "structural", "behavioral"],
                    "scope": ["entire codebase", "user module", "data layer", "service layer", "API layer"],
                },
                expected_result_types=["pattern", "structure", "relationship"],
            ),
            "edge_case_query": QueryPattern(
                category="edge_case",
                complexity="high",
                template="find {edge_case_type} with {constraint}",
                variables={
                    "edge_case_type": [
                        "recursive functions",
                        "circular dependencies",
                        "complex inheritance",
                        "multiple decorators",
                        "nested async operations",
                    ],
                    "constraint": ["deep nesting", "multiple parameters", "complex signatures", "error conditions", "edge cases"],
                },
                expected_result_types=["function", "class", "structure"],
            ),
        }

    def _define_dataset_configs(self) -> dict[str, dict[str, Any]]:
        """Define configurations for different dataset types"""
        return {
            "small_project": {
                "file_count": 50,
                "query_count": 100,
                "languages": ["python", "javascript"],
                "complexity_distribution": {"simple": 0.6, "medium": 0.3, "complex": 0.1},
            },
            "medium_project": {
                "file_count": 200,
                "query_count": 300,
                "languages": ["python", "javascript", "typescript"],
                "complexity_distribution": {"simple": 0.4, "medium": 0.4, "complex": 0.2},
            },
            "large_project": {
                "file_count": 1000,
                "query_count": 500,
                "languages": ["python", "javascript", "typescript"],
                "complexity_distribution": {"simple": 0.3, "medium": 0.5, "complex": 0.2},
            },
            "multilang_project": {
                "file_count": 300,
                "query_count": 400,
                "languages": ["python", "javascript", "typescript"],
                "complexity_distribution": {"simple": 0.3, "medium": 0.4, "complex": 0.3},
            },
            "stress_test_project": {
                "file_count": 2000,
                "query_count": 1000,
                "languages": ["python", "javascript", "typescript"],
                "complexity_distribution": {"simple": 0.2, "medium": 0.3, "complex": 0.5},
            },
        }

    def _generate_code_content(self, template: CodeTemplate, complexity: str) -> str:
        """Generate code content from template"""
        try:
            # Select random values for template variables
            template_vars = {}
            for var_name, options in template.variables.items():
                template_vars[var_name] = random.choice(options)

            # Add complexity-specific content
            complexity_config = template.complexity_levels.get(complexity, template.complexity_levels["medium"])

            # Generate additional content based on complexity
            if template.language == "python":
                template_vars.update(
                    {
                        "class_attributes": self._generate_class_attributes(complexity_config["methods"]),
                        "init_params": self._generate_function_params(3),
                        "init_body": self._generate_function_body(5),
                        "method_params": self._generate_function_params(2),
                        "method_body": self._generate_function_body(complexity_config["lines"] // 10),
                        "return_type": random.choice(["str", "int", "Dict[str, Any]", "List[str]"]),
                        "method_purpose": "processing data and returning results",
                        "async_method_name": f"async_{random.choice(['process', 'fetch', 'update'])}",
                        "async_params": self._generate_function_params(2),
                        "async_return_type": random.choice(["str", "Dict[str, Any]", "List[Any]"]),
                        "async_purpose": "asynchronous data processing",
                        "async_body": self._generate_async_function_body(8),
                        "function_params": self._generate_function_params(3),
                        "function_return_type": random.choice(["str", "bool", "int", "Dict"]),
                        "function_purpose": "utility operations",
                        "function_body": self._generate_function_body(6),
                        "async_function_name": f"async_{random.choice(['helper', 'processor', 'validator'])}",
                        "async_function_params": self._generate_function_params(2),
                        "async_function_return_type": random.choice(["str", "bool", "Dict"]),
                        "async_function_purpose": "asynchronous utility operations",
                        "async_function_body": self._generate_async_function_body(6),
                        "main_body": self._generate_main_body(),
                    }
                )
            elif template.language in ["javascript", "typescript"]:
                template_vars.update(
                    {
                        "import_path": random.choice(["./utils", "../services", "./types"]),
                        "constructor_params": self._generate_js_params(2),
                        "constructor_body": self._generate_js_function_body(3),
                        "method_params": self._generate_js_params(2),
                        "method_body": self._generate_js_function_body(complexity_config["lines"] // 15),
                        "async_params": self._generate_js_params(1),
                        "async_body": self._generate_js_async_body(5),
                        "function_params": self._generate_js_params(2),
                        "function_body": self._generate_js_function_body(4),
                        "async_function_params": self._generate_js_params(1),
                        "async_function_body": self._generate_js_async_body(4),
                    }
                )

                if template.language == "typescript":
                    template_vars.update(
                        {
                            "interface_properties": self._generate_interface_properties(3),
                            "property_type": random.choice(["string", "number", "boolean", "any[]"]),
                            "constructor_types": self._generate_ts_types(2),
                            "method_types": self._generate_ts_types(2),
                            "return_type": random.choice(["string", "number", "boolean", "object"]),
                            "async_types": self._generate_ts_types(1),
                            "async_return_type": random.choice(["string", "object", "boolean"]),
                            "function_types": self._generate_ts_types(2),
                            "function_return_type": random.choice(["string", "number", "boolean"]),
                            "async_function_types": self._generate_ts_types(1),
                            "async_function_return_type": random.choice(["string", "object"]),
                        }
                    )

            # Fill template
            return template.template.format(**template_vars)

        except KeyError as e:
            self.logger.warning(f"Missing template variable: {e}")
            return f"// Generated {template.language} file\n// Template variable missing: {e}\n"
        except Exception as e:
            self.logger.error(f"Error generating code content: {e}")
            return f"// Generated {template.language} file\n// Error in generation: {e}\n"

    def _generate_class_attributes(self, count: int) -> str:
        """Generate class attributes for Python"""
        attributes = []
        types = ["str", "int", "float", "bool", "List[str]", "Dict[str, Any]"]

        for i in range(count):
            attr_name = f"attr_{i + 1}"
            attr_type = random.choice(types)
            attributes.append(f"{attr_name}: {attr_type}")

        return "\n    ".join(attributes)

    def _generate_function_params(self, count: int) -> str:
        """Generate function parameters"""
        params = []
        types = ["str", "int", "Dict[str, Any]", "List[str]", "bool"]

        for i in range(count):
            param_name = f"param_{i + 1}"
            param_type = random.choice(types)
            params.append(f"{param_name}: {param_type}")

        return ", ".join(params)

    def _generate_function_body(self, lines: int) -> str:
        """Generate function body content"""
        statements = [
            "result = process_input(data)",
            "if not validation_check():\n        raise ValueError('Invalid input')",
            "logger.info('Processing started')",
            "temp_data = transform_data(input_data)",
            "for item in data_list:\n        process_item(item)",
            "try:\n        operation_result = perform_operation()\nexcept Exception as e:\n        handle_error(e)",
            "return formatted_result",
        ]

        body_lines = []
        for _ in range(min(lines, len(statements))):
            body_lines.append(f"        {random.choice(statements)}")

        return "\n".join(body_lines)

    def _generate_async_function_body(self, lines: int) -> str:
        """Generate async function body content"""
        statements = [
            "result = await async_operation()",
            "await asyncio.sleep(0.1)",
            "async with aiofiles.open('file.txt') as f:\n            data = await f.read()",
            "response = await http_client.get(url)",
            "await database.execute(query)",
            "return await process_async_result(data)",
        ]

        body_lines = []
        for _ in range(min(lines, len(statements))):
            body_lines.append(f"        {random.choice(statements)}")

        return "\n".join(body_lines)

    def _generate_main_body(self) -> str:
        """Generate main function body"""
        return """    processor = DataProcessor()
    result = processor.process_data({"test": "data"})
    print(f"Result: {result}")"""

    def _generate_js_params(self, count: int) -> str:
        """Generate JavaScript parameters"""
        params = [f"param{i + 1}" for i in range(count)]
        return ", ".join(params)

    def _generate_js_function_body(self, lines: int) -> str:
        """Generate JavaScript function body"""
        statements = [
            "const result = processData(input);",
            "if (!isValid(data)) throw new Error('Invalid data');",
            "console.log('Processing started');",
            "const transformed = data.map(item => transform(item));",
            "return formatResult(result);",
            "await delay(100);",
        ]

        body_lines = []
        for _ in range(min(lines, len(statements))):
            body_lines.append(f"        {random.choice(statements)}")

        return "\n".join(body_lines)

    def _generate_js_async_body(self, lines: int) -> str:
        """Generate JavaScript async function body"""
        statements = [
            "const result = await fetchData();",
            "await new Promise(resolve => setTimeout(resolve, 100));",
            "const response = await fetch(url);",
            "const data = await response.json();",
            "return await processAsyncResult(data);",
        ]

        body_lines = []
        for _ in range(min(lines, len(statements))):
            body_lines.append(f"        {random.choice(statements)}")

        return "\n".join(body_lines)

    def _generate_interface_properties(self, count: int) -> str:
        """Generate TypeScript interface properties"""
        properties = []
        types = ["string", "number", "boolean", "string[]", "object"]

        for i in range(count):
            prop_name = f"property{i + 1}"
            prop_type = random.choice(types)
            properties.append(f"{prop_name}: {prop_type};")

        return "\n    ".join(properties)

    def _generate_ts_types(self, count: int) -> str:
        """Generate TypeScript type annotations"""
        types = ["string", "number", "boolean", "object", "any[]"]
        return ", ".join([random.choice(types) for _ in range(count)])

    def _generate_query(self, pattern: QueryPattern) -> dict[str, Any]:
        """Generate a query from pattern"""
        try:
            # Select random values for query variables
            query_vars = {}
            for var_name, options in pattern.variables.items():
                query_vars[var_name] = random.choice(options)

            # Fill query template
            query_text = pattern.template.format(**query_vars)

            # Generate expected results
            expected_results = []
            for _ in range(random.randint(1, 5)):
                result_type = random.choice(pattern.expected_result_types)
                result_id = f"{result_type}_{uuid.uuid4().hex[:8]}"
                expected_results.append(result_id)

            return {
                "query_id": f"query_{uuid.uuid4().hex[:8]}",
                "query_text": query_text,
                "category": pattern.category,
                "complexity": pattern.complexity,
                "expected_results": expected_results,
                "variables": query_vars,
            }

        except Exception as e:
            self.logger.error(f"Error generating query: {e}")
            return {
                "query_id": f"query_{uuid.uuid4().hex[:8]}",
                "query_text": "test query",
                "category": pattern.category,
                "complexity": pattern.complexity,
                "expected_results": [],
                "variables": {},
            }

    def generate_dataset(self, dataset_type: str, custom_config: dict[str, Any] | None = None) -> TestDataset:
        """Generate a complete test dataset"""
        self.logger.info(f"Generating dataset: {dataset_type}")

        # Get configuration
        config = self.dataset_configs.get(dataset_type, self.dataset_configs["medium_project"])
        if custom_config:
            config.update(custom_config)

        # Create dataset directory
        dataset_id = f"{dataset_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        dataset_dir = self.output_dir / dataset_id
        dataset_dir.mkdir(exist_ok=True)

        # Generate files
        files = []
        complexity_counts = {}

        for complexity, ratio in config["complexity_distribution"].items():
            count = int(config["file_count"] * ratio)
            complexity_counts[complexity] = count

            for i in range(count):
                # Select random language
                language = random.choice(config["languages"])
                template = self.code_templates[language]

                # Generate file content
                content = self._generate_code_content(template, complexity)

                # Create file
                filename = f"{language}_file_{complexity}_{i}.{template.file_extension}"
                file_path = dataset_dir / filename

                with open(file_path, "w") as f:
                    f.write(content)

                files.append(str(file_path.relative_to(dataset_dir)))

        # Generate queries
        queries = []
        ground_truth = {}

        for i in range(config["query_count"]):
            # Select random query pattern
            pattern = random.choice(list(self.query_patterns.values()))
            query = self._generate_query(pattern)
            queries.append(query)

            # Store ground truth
            ground_truth[query["query_id"]] = query["expected_results"]

        # Create dataset metadata
        dataset = TestDataset(
            dataset_id=dataset_id,
            name=f"{dataset_type.title()} Test Dataset",
            description=f"Synthetic test dataset for {dataset_type} testing",
            creation_time=datetime.now().isoformat(),
            file_count=len(files),
            query_count=len(queries),
            languages=config["languages"],
            complexity_distribution=complexity_counts,
            files=files,
            queries=queries,
            ground_truth=ground_truth,
            metadata={"config": config, "generation_method": "template_based", "version": "1.0"},
        )

        # Save dataset metadata
        metadata_file = dataset_dir / "dataset.json"
        with open(metadata_file, "w") as f:
            json.dump(asdict(dataset), f, indent=2)

        # Save ground truth separately
        ground_truth_file = dataset_dir / "ground_truth.json"
        with open(ground_truth_file, "w") as f:
            json.dump(ground_truth, f, indent=2)

        # Save queries separately
        queries_file = dataset_dir / "queries.json"
        with open(queries_file, "w") as f:
            json.dump(queries, f, indent=2)

        self.logger.info(f"Generated dataset: {dataset_id} with {len(files)} files and {len(queries)} queries")

        return dataset

    def generate_all_datasets(self) -> list[TestDataset]:
        """Generate all predefined datasets"""
        datasets = []

        for dataset_type in self.dataset_configs.keys():
            try:
                dataset = self.generate_dataset(dataset_type)
                datasets.append(dataset)
            except Exception as e:
                self.logger.error(f"Failed to generate dataset {dataset_type}: {e}")

        return datasets

    def generate_edge_case_dataset(self) -> TestDataset:
        """Generate dataset with edge cases and challenging scenarios"""
        edge_case_config = {
            "file_count": 100,
            "query_count": 200,
            "languages": ["python", "javascript", "typescript"],
            "complexity_distribution": {"simple": 0.1, "medium": 0.3, "complex": 0.6},
        }

        return self.generate_dataset("edge_cases", edge_case_config)

    def generate_benchmark_dataset(self) -> TestDataset:
        """Generate dataset specifically for benchmarking"""
        benchmark_config = {
            "file_count": 500,
            "query_count": 1000,
            "languages": ["python", "javascript", "typescript"],
            "complexity_distribution": {"simple": 0.33, "medium": 0.34, "complex": 0.33},
        }

        return self.generate_dataset("benchmark", benchmark_config)

    def export_dataset_summary(self, datasets: list[TestDataset], output_file: str):
        """Export summary of all generated datasets"""
        summary = {
            "generation_time": datetime.now().isoformat(),
            "total_datasets": len(datasets),
            "total_files": sum(d.file_count for d in datasets),
            "total_queries": sum(d.query_count for d in datasets),
            "datasets": [
                {
                    "dataset_id": d.dataset_id,
                    "name": d.name,
                    "file_count": d.file_count,
                    "query_count": d.query_count,
                    "languages": d.languages,
                    "complexity_distribution": d.complexity_distribution,
                }
                for d in datasets
            ],
        }

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2)

        self.logger.info(f"Exported dataset summary: {output_file}")


async def main():
    """Main function to generate test datasets"""
    generator = TestDataGenerator()

    print("Generating comprehensive test datasets...")

    # Generate all standard datasets
    datasets = generator.generate_all_datasets()

    # Generate special datasets
    edge_case_dataset = generator.generate_edge_case_dataset()
    datasets.append(edge_case_dataset)

    benchmark_dataset = generator.generate_benchmark_dataset()
    datasets.append(benchmark_dataset)

    # Export summary
    generator.export_dataset_summary(datasets, "test_dataset_summary.json")

    print("\n=== Test Data Generation Complete ===")
    print(f"Generated {len(datasets)} datasets")
    print(f"Total files: {sum(d.file_count for d in datasets)}")
    print(f"Total queries: {sum(d.query_count for d in datasets)}")
    print(f"Languages: {list(set(lang for d in datasets for lang in d.languages))}")

    # Print dataset details
    for dataset in datasets:
        print(f"\nDataset: {dataset.name}")
        print(f"  ID: {dataset.dataset_id}")
        print(f"  Files: {dataset.file_count}")
        print(f"  Queries: {dataset.query_count}")
        print(f"  Languages: {', '.join(dataset.languages)}")
        print(f"  Complexity: {dataset.complexity_distribution}")

    return datasets


if __name__ == "__main__":
    asyncio.run(main())
