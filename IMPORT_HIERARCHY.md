# Import Hierarchy Design - MCP System Refactoring

## Overview
This document defines the import hierarchy for the refactored MCP system to prevent circular dependencies and establish clear dependency relationships.

## Import Hierarchy Levels

### Level 1: Core Infrastructure (No Dependencies)
```
src/tools/core/
├── errors.py           # Custom exception classes
├── retry_utils.py      # Retry logic utilities  
└── memory_utils.py     # Memory management utilities
```

**Dependencies**: None (only standard library)
**Purpose**: Foundational utilities used by all other modules

### Level 2: External Service Clients (Depends on Level 1)
```
src/tools/core/
└── client_manager.py   # Qdrant, Ollama client management
```

**Dependencies**: Level 1 (errors, retry_utils)
**Purpose**: External service connectivity and client management

### Level 3: Database Operations (Depends on Level 1-2)
```
src/tools/database/
├── collection_manager.py  # Collection CRUD operations
├── streaming.py           # Streaming database operations  
└── operations.py          # General database utilities
```

**Dependencies**: Level 1-2 (client_manager, errors, retry_utils)
**Purpose**: Database-specific operations and utilities

### Level 4: Project Management (Depends on Level 1)
```
src/tools/project/
├── project_manager.py    # Project detection and management
├── file_operations.py    # File filtering and operations
└── status_tools.py       # Index status checking
```

**Dependencies**: Level 1 (errors, memory_utils)
**Purpose**: Project-level operations independent of database

### Level 5: Indexing Operations (Depends on Level 1-4)
```
src/tools/indexing/
├── analysis_tools.py     # Repository analysis
├── chunking_tools.py     # Chunking metrics and diagnostics
├── parser_tools.py       # Tree-sitter parser diagnostics
├── progress_tools.py     # Progress monitoring
├── search_tools.py       # Search functionality
└── indexing_tools.py     # Main indexing functionality
```

**Dependencies**: Level 1-4 (all previous levels)
**Purpose**: High-level indexing and search operations

### Level 6: Health Monitoring (Depends on Level 1-2)
```
src/tools/core/
└── health.py             # Health check implementation
```

**Dependencies**: Level 1-2 (client_manager, errors)
**Purpose**: System health monitoring

### Level 7: Tool Registry (Depends on All Previous)
```
src/tools/
└── registry.py           # Tool registration with FastMCP
```

**Dependencies**: All previous levels
**Purpose**: Central registration point for all tools

## Prompts Hierarchy

### Level 1: Base Prompt Infrastructure
```
src/prompts/base/
├── prompt_base.py        # Base prompt classes
└── prompt_builder.py     # Common prompt building utilities
```

**Dependencies**: None (only standard library and models)
**Purpose**: Foundational prompt utilities

### Level 2: Specific Prompt Categories
```
src/prompts/exploration/    # Project exploration prompts
src/prompts/recommendation/ # Recommendation prompts
```

**Dependencies**: Level 1 (base prompt utilities)
**Purpose**: Specific prompt implementations

### Level 3: Prompts Registry
```
src/prompts/
└── registry.py           # Prompt registration with FastMCP
```

**Dependencies**: All prompt modules
**Purpose**: Central registration point for all prompts

## Key Design Principles

### 1. Dependency Direction
- **Upward Only**: Higher levels can depend on lower levels, never the reverse
- **No Horizontal**: Modules at the same level should not depend on each other
- **Skip Levels**: Higher levels can skip intermediate levels if needed

### 2. Shared State Management
- **Client Singletons**: Managed in `client_manager.py` (Level 2)
- **Project State**: Managed in `project_manager.py` (Level 4)
- **No Global Variables**: All shared state accessed through proper interfaces

### 3. Configuration Management
- **Environment Variables**: Loaded once in client_manager.py
- **Configuration Objects**: Passed as parameters, not accessed globally
- **No Side Effects**: Modules don't modify global configuration on import

### 4. Error Propagation
- **Custom Exceptions**: Defined in `errors.py` (Level 1)
- **Error Context**: Each level adds appropriate context
- **Graceful Degradation**: Higher levels handle lower-level failures

### 5. Testing Strategy
- **Unit Tests**: Each level can be tested independently
- **Mock Dependencies**: Lower levels can be mocked for testing higher levels
- **Integration Tests**: Full stack testing at registry level

## Import Examples

### Correct Import Patterns
```python
# Level 5 importing from Level 1-4 (ALLOWED)
from tools.core.errors import IndexingError
from tools.database.collection_manager import ensure_collection
from tools.project.project_manager import get_current_project

# Level 2 importing from Level 1 (ALLOWED)
from tools.core.retry_utils import retry_operation
from tools.core.errors import QdrantConnectionError
```

### Forbidden Import Patterns
```python
# Level 1 importing from Level 2+ (FORBIDDEN)
from tools.database.operations import some_function  # WRONG

# Horizontal imports at same level (FORBIDDEN)  
from tools.indexing.search_tools import search  # From indexing_tools.py - WRONG

# Registry importing specific implementations (MINIMIZE)
from tools.indexing.indexing_tools import index_directory  # Prefer dynamic imports
```

## Migration Strategy

### Phase 1: Create Base Infrastructure (Level 1-2)
1. Extract error classes and utilities
2. Create client manager with proper initialization
3. Test basic functionality

### Phase 2: Database and Project Layers (Level 3-4)
1. Extract database operations with proper client dependency
2. Extract project management utilities
3. Test database and project operations independently

### Phase 3: High-Level Operations (Level 5-6)
1. Extract indexing and search tools
2. Extract health monitoring
3. Test all high-level operations

### Phase 4: Registry Integration (Level 7)
1. Create dynamic tool registration
2. Update main.py imports
3. End-to-end testing

## Validation Checklist

- [ ] No circular imports (tools can detect this)
- [ ] Each level only depends on lower levels
- [ ] Shared state properly managed through interfaces
- [ ] Configuration loaded once and passed as parameters
- [ ] Error handling follows the hierarchy
- [ ] Tests can run independently for each level
- [ ] Registry uses dynamic imports to reduce coupling