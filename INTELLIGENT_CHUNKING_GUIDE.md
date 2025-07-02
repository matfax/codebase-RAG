# Intelligent Code Chunking Guide

## Overview

This guide provides comprehensive instructions and best practices for using the intelligent code chunking system in the Codebase RAG MCP Server. The intelligent chunking system uses Tree-sitter parsers to break down source code into semantically meaningful chunks (functions, classes, methods) instead of processing entire files as single units.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Supported Languages](#supported-languages)
3. [Understanding Chunk Types](#understanding-chunk-types)
4. [Best Practices](#best-practices)
5. [Error Handling](#error-handling)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)
8. [Advanced Usage](#advanced-usage)

## Quick Start

### Basic Indexing with Intelligent Chunking

Intelligent chunking is enabled by default when you index a directory:

```bash
# Using MCP tools (automatic chunking)
python -c "
import asyncio
from src.main import app

async def index_project():
    result = await app.call_tool('index_directory', {'directory': '.'})
    print(result)

asyncio.run(index_project())
"

# Using manual indexing tool (recommended for large projects)
python manual_indexing.py -d /path/to/your/project -m clear_existing
```

### Searching with Function-Level Precision

```python
import asyncio
from src.main import app

async def search_functions():
    # Search for specific functions
    result = await app.call_tool('search', {
        'query': 'user authentication function',
        'n_results': 5,
        'include_context': True
    })

    for chunk in result['results']:
        print(f"Found: {chunk['metadata']['name']} in {chunk['metadata']['file_path']}")
        print(f"Type: {chunk['metadata']['chunk_type']}")
        print(f"Signature: {chunk['metadata']['signature']}")
        print(f"Lines: {chunk['metadata']['start_line']}-{chunk['metadata']['end_line']}")
        print("---")

asyncio.run(search_functions())
```

## Supported Languages

### Phase 1 (Fully Supported)

#### Python (.py)
- **Functions**: Regular functions, async functions, lambdas
- **Classes**: Class definitions with methods and properties
- **Methods**: Instance methods, class methods, static methods
- **Constants**: Module-level constants and complex objects
- **Docstrings**: Automatically extracted and associated

```python
# Example chunks extracted from Python code
class UserService:  # -> Class chunk
    """User management service."""

    def __init__(self, db):  # -> Method chunk
        self.db = db

    async def authenticate_user(self, username, password):  # -> Method chunk
        """Authenticate user credentials."""
        return await self.db.verify_credentials(username, password)

def calculate_hash(data):  # -> Function chunk
    """Calculate SHA256 hash."""
    return hashlib.sha256(data.encode()).hexdigest()
```

#### JavaScript (.js, .jsx)
- **Functions**: Function declarations, expressions, arrow functions
- **Objects**: Complex object literals and configuration objects
- **Classes**: ES6 classes with methods
- **Modules**: Export/import statements and module patterns

```javascript
// Example chunks extracted from JavaScript code
const config = {  // -> Constant chunk
    api: {
        baseUrl: 'https://api.example.com',
        timeout: 5000
    }
};

class APIClient {  // -> Class chunk
    constructor(config) {  // -> Method chunk
        this.config = config;
    }

    async fetchUser(id) {  // -> Method chunk
        return await fetch(`${this.config.api.baseUrl}/users/${id}`);
    }
}

const validateEmail = (email) => {  // -> Function chunk
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
};
```

#### TypeScript (.ts, .tsx)
- **Interfaces**: Type definitions and contracts
- **Types**: Type aliases and generic definitions
- **Classes**: Classes with TypeScript annotations
- **Functions**: Typed functions and methods

```typescript
// Example chunks extracted from TypeScript code
interface User {  // -> Interface chunk
    id: number;
    username: string;
    email: string;
}

type ApiResponse<T> = {  // -> Type chunk
    data: T;
    status: number;
    message: string;
};

class UserRepository {  // -> Class chunk
    async findById(id: number): Promise<User | null> {  // -> Method chunk
        // Implementation
    }
}
```

### Phase 2 (Extended Support)

#### Go (.go)
- **Functions**: Function definitions with receivers
- **Structs**: Type definitions and methods
- **Interfaces**: Interface declarations
- **Methods**: Struct methods and function methods

#### Rust (.rs)
- **Functions**: Function definitions
- **Structs**: Data structures
- **Impl blocks**: Implementation blocks
- **Traits**: Trait definitions

#### Java (.java)
- **Classes**: Class definitions with annotations
- **Methods**: Method definitions with modifiers
- **Interfaces**: Interface declarations
- **Annotations**: Custom annotations

### Structured Files

#### JSON/YAML Configuration
```json
{
    "scripts": {  // -> Config chunk
        "build": "npm run build",
        "test": "jest"
    },
    "dependencies": {  // -> Config chunk
        "express": "^4.18.0",
        "lodash": "^4.17.21"
    }
}
```

#### Markdown Documentation
```markdown
## Installation  # -> Documentation chunk (section)
Instructions for installing...

### Prerequisites  # -> Documentation chunk (subsection)
You need to have...

## Configuration  # -> Documentation chunk (new section)
Configure the system...
```

## Understanding Chunk Types

### Chunk Type Hierarchy

```
ChunkType
├── FUNCTION          # Standalone functions
├── CLASS             # Class definitions (without methods)
├── METHOD            # Methods within classes
├── INTERFACE         # TypeScript/Java interfaces
├── TYPE_ALIAS        # Type definitions
├── CONSTANT          # Important constants and configs
├── CONFIG_OBJECT     # Configuration sections
└── IMPORT_BLOCK      # Import/require statements
```

### Chunk Metadata

Each chunk includes rich metadata:

```python
{
    "content": "def authenticate_user(username, password):\n    ...",
    "file_path": "src/auth/user_service.py",
    "chunk_type": "function",
    "name": "authenticate_user",
    "signature": "authenticate_user(username, password)",
    "start_line": 15,
    "end_line": 25,
    "language": "python",
    "docstring": "Authenticate user with username and password",
    "access_modifier": "public",
    "parent_class": null,
    "has_syntax_errors": false,
    "chunk_id": "550e8400-e29b-41d4-a716-446655440000",
    "content_hash": "abc123..."
}
```

## Best Practices

### 1. Code Organization for Optimal Chunking

#### ✅ Good Practices

```python
# Clear, well-documented functions
def calculate_user_score(user_data: dict, weights: dict) -> float:
    """
    Calculate user score based on activity data.

    Args:
        user_data: Dictionary containing user activity metrics
        weights: Dictionary containing scoring weights

    Returns:
        Calculated score as float
    """
    score = 0.0
    for metric, value in user_data.items():
        if metric in weights:
            score += value * weights[metric]
    return score

# Well-structured classes with clear methods
class UserAnalytics:
    """Analytics service for user behavior analysis."""

    def __init__(self, database_client):
        self.db = database_client

    def track_event(self, user_id: str, event_type: str, metadata: dict):
        """Track user event with metadata."""
        # Implementation

    def generate_report(self, user_id: str, date_range: tuple) -> dict:
        """Generate analytics report for user."""
        # Implementation
```

#### ❌ Practices to Avoid

```python
# Avoid overly complex nested functions
def complex_processor():
    def inner1():
        def inner2():
            def inner3():  # Too deeply nested
                pass
    # This creates confusing chunks

# Avoid mixing multiple concerns in one function
def do_everything(data):
    # Validates data
    # Processes data
    # Saves to database
    # Sends notifications
    # Generates reports
    # This should be multiple functions
```

### 2. Documentation Best Practices

#### For Functions
```python
def process_payment(amount: float, payment_method: str) -> dict:
    """
    Process payment transaction.

    This function handles payment processing including validation,
    gateway communication, and transaction recording.

    Args:
        amount: Payment amount in USD
        payment_method: Payment method identifier

    Returns:
        Transaction result with status and transaction_id

    Raises:
        PaymentError: When payment processing fails
        ValidationError: When input validation fails
    """
```

#### For Classes
```python
class PaymentProcessor:
    """
    Payment processing service.

    Handles various payment methods including credit cards,
    digital wallets, and bank transfers. Provides transaction
    tracking and error handling.

    Attributes:
        gateway: Payment gateway client
        config: Processor configuration
    """
```

### 3. File Structure Recommendations

```
src/
├── services/           # Business logic services
│   ├── user_service.py
│   ├── payment_service.py
│   └── notification_service.py
├── models/            # Data models
│   ├── user.py
│   ├── payment.py
│   └── transaction.py
├── utils/             # Utility functions
│   ├── validators.py
│   ├── formatters.py
│   └── helpers.py
└── config/            # Configuration
    ├── database.py
    ├── api_config.py
    └── settings.py
```

### 4. Language-Specific Recommendations

#### Python
- Use type hints for better metadata extraction
- Include comprehensive docstrings
- Follow PEP 8 for consistent naming
- Use dataclasses for simple data structures

#### JavaScript/TypeScript
- Use JSDoc comments for documentation
- Prefer named functions over anonymous ones
- Use TypeScript for better type information
- Group related functions in classes or modules

#### Go
- Follow Go naming conventions
- Use package-level documentation
- Group related functions logically
- Use struct methods for object behavior

## Error Handling

### Understanding Syntax Errors

The intelligent chunking system can handle various types of syntax errors:

#### Minor Errors (Recoverable)
```python
# Missing comma - system can still extract function
def calculate_total(items):
    return sum(item.price item.quantity for item in items)  # Missing *
    #                   ^ Minor syntax error
```

#### Major Errors (Fallback to Whole File)
```python
# Completely malformed code
def broken_function(
    # Missing closing parenthesis and function body
    # System falls back to whole-file processing
```

### Error Recovery Strategies

1. **Partial Content Preservation**: Correct code sections are preserved
2. **Context Inclusion**: Surrounding correct code provides context
3. **Graceful Fallback**: Falls back to whole-file processing when needed
4. **Error Reporting**: Detailed error statistics in logs

### Viewing Error Reports

```bash
# Run manual indexing with verbose error reporting
python manual_indexing.py -d /path/to/project -m clear_existing --verbose

# Check error statistics in output
# Example output:
# Syntax Error Statistics:
# - Files with minor errors: 3
# - Files with major errors: 1
# - Total chunks recovered: 45/50 (90%)
```

## Performance Optimization

### 1. Large Codebase Strategies

For repositories with 1000+ files:

```bash
# Use manual indexing tool for heavy operations
python manual_indexing.py -d /large/repository -m clear_existing

# Use incremental indexing for updates
python manual_indexing.py -d /large/repository -m incremental
```

### 2. Memory Optimization

Configure environment variables for optimal performance:

```bash
# .env file settings for large projects
INDEXING_CONCURRENCY=8          # Increase for more CPU cores
INDEXING_BATCH_SIZE=50          # Larger batches for better throughput
EMBEDDING_BATCH_SIZE=20         # Balance API calls vs memory
MEMORY_WARNING_THRESHOLD_MB=2000 # Higher threshold for large projects
```

### 3. File Filtering

Use `.ragignore` to exclude unnecessary files:

```
# .ragignore
node_modules/
.git/
dist/
build/
*.log
*.tmp
__pycache__/
.pytest_cache/
coverage/
.nyc_output/
```

### 4. Language-Specific Optimizations

#### Python Projects
```bash
# Exclude common Python build artifacts
echo "__pycache__/
*.pyc
*.pyo
dist/
build/
*.egg-info/" >> .ragignore
```

#### Node.js Projects
```bash
# Exclude Node.js artifacts
echo "node_modules/
npm-debug.log
.npm/
dist/
coverage/" >> .ragignore
```

## Troubleshooting

### Common Issues and Solutions

#### 1. "No chunks extracted from file"

**Cause**: Syntax errors or unsupported language
**Solution**:
```bash
# Check if language is supported
python -c "
from src.services.code_parser_service import CodeParserService
parser = CodeParserService()
print('Supported languages:', list(parser.supported_languages.keys()))
"

# Run with verbose logging to see specific errors
python manual_indexing.py -d . -m clear_existing --verbose
```

#### 2. "Tree-sitter parser not found"

**Cause**: Missing language parser dependency
**Solution**:
```bash
# Reinstall dependencies
.venv/bin/poetry install

# Check if specific parser is available
python -c "
import tree_sitter_python
import tree_sitter_javascript
import tree_sitter_typescript
print('Parsers installed successfully')
"
```

#### 3. "Memory usage too high"

**Cause**: Processing very large files or too many files concurrently
**Solution**:
```bash
# Reduce batch sizes in .env
INDEXING_BATCH_SIZE=10
EMBEDDING_BATCH_SIZE=5
INDEXING_CONCURRENCY=2

# Skip large files
MAX_FILE_SIZE_MB=2
```

#### 4. "Search results not precise enough"

**Cause**: Indexing completed but chunks not properly extracted
**Solution**:
```bash
# Verify chunking worked correctly
python -c "
import asyncio
from src.main import app

async def check_chunks():
    result = await app.call_tool('search', {
        'query': 'chunk_type:function',
        'n_results': 5
    })
    print(f'Found {len(result[\"results\"])} function chunks')

asyncio.run(check_chunks())
"
```

### Debugging Chunking Issues

#### Enable Debug Logging

```bash
# Set environment variable for detailed logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python manual_indexing.py -d . -m clear_existing --verbose
```

#### Check Individual File Processing

```python
from src.services.code_parser_service import CodeParserService

# Test specific file
parser = CodeParserService()
chunks = parser.parse_file('path/to/problematic/file.py')

for chunk in chunks:
    print(f"Chunk: {chunk.name} ({chunk.chunk_type})")
    print(f"Lines: {chunk.start_line}-{chunk.end_line}")
    print(f"Errors: {chunk.has_syntax_errors}")
    print("---")
```

## Advanced Usage

### 1. Custom Project Names

```python
# Use custom project names for organization
result = await app.call_tool('index_directory', {
    'directory': '/path/to/frontend',
    'project_name': 'my_frontend_app'
})

result = await app.call_tool('index_directory', {
    'directory': '/path/to/backend',
    'project_name': 'my_backend_api'
})

# Search across specific projects
result = await app.call_tool('search', {
    'query': 'authentication',
    'cross_project': True  # Search all projects
})
```

### 2. Context-Enhanced Searches

```python
# Get rich context around search results
result = await app.call_tool('search', {
    'query': 'database connection',
    'include_context': True,
    'context_chunks': 3,  # Include 3 surrounding chunks
    'n_results': 10
})

# Results include:
# - Main matching chunk
# - Surrounding functions/classes
# - Import statements
# - Related docstrings
```

### 3. Cross-Language Project Analysis

```python
# Analyze multi-language projects
result = await app.call_tool('analyze_repository_tool', {
    'directory': '/path/to/fullstack/project'
})

# Returns language breakdown:
# - Python: 150 files, 1200 functions
# - TypeScript: 80 files, 800 functions
# - Go: 30 files, 200 functions
```

### 4. Performance Monitoring

```python
# Get indexing progress for long operations
result = await app.call_tool('get_indexing_progress')

# Returns:
# - Files processed: 150/500
# - Current stage: "Embedding generation"
# - ETA: "5 minutes"
# - Memory usage: "850 MB"
# - Syntax errors: 3
```

### 5. Custom Filtering Patterns

```python
# Index only specific file types
result = await app.call_tool('index_directory', {
    'directory': '/mixed/project',
    'patterns': ['*.py', '*.ts', '*.go'],  # Only these languages
    'recursive': True
})
```

## Migration from Whole-File Indexing

If you're upgrading from a previous version that used whole-file indexing:

### 1. Clear Existing Index

```bash
# Remove old collections (they're incompatible)
python manual_indexing.py -d . -m clear_existing
```

### 2. Re-index with Intelligent Chunking

```bash
# Full re-indexing (this is now the default)
python manual_indexing.py -d . -m clear_existing --no-confirm
```

### 3. Verify Migration

```python
# Check that chunks are function-level
result = await app.call_tool('search', {
    'query': 'your_function_name',
    'n_results': 5
})

# Should return specific functions, not whole files
for chunk in result['results']:
    assert chunk['metadata']['chunk_type'] in ['function', 'method', 'class']
```

## Conclusion

The intelligent code chunking system provides significant improvements in code search precision and relevance. By following these best practices and guidelines, you can maximize the benefits of function-level code understanding and retrieval.

For additional support or questions:
- Check the troubleshooting section above
- Review error logs with `--verbose` flag
- Examine the `CLAUDE.md` file for architectural details
- Open an issue in the project repository

Remember that intelligent chunking is designed to gracefully handle imperfect code, so don't worry about minor syntax errors - the system will work around them while preserving as much semantic information as possible.
