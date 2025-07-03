# Sample Code Files for Testing Intelligent Chunking

This directory contains comprehensive sample code files in various programming languages to test the intelligent chunking capabilities of the CodeParser service.

## File Overview

### Programming Languages

| File | Language | Description | Key Features |
|------|----------|-------------|--------------|
| `sample_python.py` | Python | User management system | Classes, functions, async methods, decorators, context managers |
| `sample_javascript.js` | JavaScript | User API client with React components | ES6 classes, arrow functions, async/await, event handling |
| `sample_typescript.ts` | TypeScript | Type-safe user service | Interfaces, generics, type definitions, enums |
| `sample_go.go` | Go | HTTP server with user management | Structs, interfaces, methods, goroutines, error handling |
| `sample_rust.rs` | Rust | Memory-safe user service | Traits, generics, ownership, error handling, async |
| `sample_java.java` | Java | Enterprise user service | Classes, inheritance, generics, annotations, streams |
| `sample_cpp.cpp` | C++ | Template-based user service | Classes, templates, namespaces, smart pointers, STL |

### Configuration and Documentation

| File | Type | Description | Key Features |
|------|------|-------------|--------------|
| `sample_config.json` | JSON | Comprehensive application configuration | Nested objects, arrays, environment variables |
| `sample_config.yaml` | YAML | Multi-environment configuration | Complex hierarchies, lists, mappings |
| `sample_documentation.md` | Markdown | API documentation | Headers, code blocks, tables, lists |
| `sample_syntax_errors.py` | Python | Code with intentional syntax errors | Error recovery testing |

## Testing Coverage

### Code Constructs Tested

- **Functions/Methods**: Regular functions, async functions, methods, lambdas
- **Classes/Objects**: Classes, inheritance, constructors, destructors
- **Interfaces/Traits**: Abstract interfaces, trait implementations
- **Data Types**: Structs, enums, unions, type aliases
- **Modules**: Imports, exports, namespaces, packages
- **Constants/Variables**: Module-level constants, static variables
- **Templates/Generics**: Template classes, generic functions
- **Error Handling**: Try-catch blocks, error types, result types
- **Concurrency**: Async/await, promises, channels, threads

### Language-Specific Features

#### Python (`sample_python.py`)
- Dataclasses and type hints
- Context managers (`__enter__`, `__exit__`)
- Decorators and closures
- Generator functions
- Property methods
- Static and class methods

#### JavaScript (`sample_javascript.js`)
- ES6 classes and inheritance
- Arrow functions and function expressions
- Promise-based async operations
- Event emitters and callbacks
- Object methods and prototypes
- Import/export statements

#### TypeScript (`sample_typescript.ts`)
- Interface definitions and implementations
- Type aliases and union types
- Generic types and constraints
- Optional and nullable types
- Enum definitions
- Method signatures

#### Go (`sample_go.go`)
- Struct methods and interfaces
- Goroutines and channels
- Error handling patterns
- Package-level functions
- Constant and variable declarations
- HTTP handlers and middleware

#### Rust (`sample_rust.rs`)
- Trait definitions and implementations
- Ownership and borrowing
- Error handling with Result types
- Generic functions and structs
- Async functions and futures
- Pattern matching

#### Java (`sample_java.java`)
- Class hierarchies and inheritance
- Interface implementations
- Generic types and wildcards
- Annotation usage
- Exception handling
- Static and instance methods
- Enum definitions

#### C++ (`sample_cpp.cpp`)
- Template classes and functions
- Namespace declarations
- Smart pointers and RAII
- Operator overloading
- STL containers and algorithms
- Constructor/destructor patterns

### Configuration Files

#### JSON (`sample_config.json`)
- Nested object structures
- Array configurations
- Environment variable placeholders
- Complex application settings

#### YAML (`sample_config.yaml`)
- Multi-level hierarchies
- List and mapping structures
- Environment-specific overrides
- Comments and documentation

#### Markdown (`sample_documentation.md`)
- Multiple heading levels
- Code blocks with syntax highlighting
- Tables and lists
- Links and references

## Testing Objectives

### Chunking Quality
- **Semantic Boundaries**: Proper identification of logical code units
- **Context Preservation**: Maintaining relationships between related code
- **Metadata Extraction**: Accurate extraction of names, signatures, and types
- **Hierarchical Understanding**: Proper nesting of classes, methods, and modules

### Error Handling
- **Syntax Error Recovery**: Graceful handling of malformed code
- **Partial Parsing**: Extracting valid chunks from files with errors
- **Error Classification**: Proper categorization of syntax errors

### Performance
- **Processing Speed**: Efficient parsing of large files
- **Memory Usage**: Reasonable memory consumption during parsing
- **Scalability**: Handling multiple files concurrently

### Language Support
- **Parser Availability**: Ensuring Tree-sitter parsers are available
- **Feature Coverage**: Supporting language-specific constructs
- **Quality Metrics**: Consistent chunking quality across languages

## Usage in Tests

These sample files are used by:

1. **Unit Tests**: `test_code_parser_service.py` for basic functionality
2. **Integration Tests**: `test_intelligent_chunking.py` for end-to-end workflows
3. **Performance Tests**: `test_performance_benchmarks.py` for speed and memory usage
4. **Error Handling Tests**: `test_syntax_error_handling.py` for robustness

## Maintenance

When updating sample files:

1. **Preserve Complexity**: Maintain realistic code patterns
2. **Add Documentation**: Include comments explaining features
3. **Test Coverage**: Ensure new constructs are properly chunked
4. **Update Tests**: Modify corresponding test expectations
5. **Validate Parsing**: Run parser tests after changes

## Quality Metrics

Expected chunking results for each file:

- **Python**: ~15-25 chunks (classes, functions, constants)
- **JavaScript**: ~20-30 chunks (classes, functions, exports)
- **TypeScript**: ~25-35 chunks (interfaces, types, classes)
- **Go**: ~30-40 chunks (structs, functions, methods)
- **Rust**: ~40-80 chunks (traits, impls, functions)
- **Java**: ~50-100 chunks (classes, methods, interfaces)
- **C++**: ~60-120 chunks (classes, templates, namespaces)

These numbers may vary based on chunking algorithm improvements and configuration changes.
