# Manual Indexing Tool Examples

This document provides comprehensive examples and use cases for the manual indexing tool with intelligent code chunking capabilities.

## Table of Contents

1. [Basic Usage](#basic-usage)
2. [Language-Specific Examples](#language-specific-examples)
3. [Performance Optimization](#performance-optimization)
4. [Error Handling and Debugging](#error-handling-and-debugging)
5. [CI/CD Integration](#cicd-integration)
6. [Real-World Scenarios](#real-world-scenarios)

## Basic Usage

### First-Time Indexing

For initial indexing of a new project with intelligent chunking:

```bash
# Basic full indexing
python manual_indexing.py -d /path/to/your/project -m clear_existing

# With verbose output to see chunking details
python manual_indexing.py -d /path/to/your/project -m clear_existing --verbose
```

**Expected Output:**
```
🔍 Pre-indexing Analysis for /path/to/your/project
├── Total files found: 250
├── Code files (Python, JS, TS): 180
├── Config files (JSON, YAML): 15
├── Documentation files: 25
├── Estimated processing time: 3-5 minutes
└── Intelligent chunking: ✓ Enabled

Languages detected:
├── Python: 120 files (functions, classes, methods)
├── TypeScript: 45 files (interfaces, types, functions)
├── JavaScript: 15 files (functions, objects)
└── JSON/YAML: 15 files (structured sections)

Proceed with indexing? [y/N]:
```

### Incremental Updates

For daily development updates (much faster):

```bash
# Process only changed files
python manual_indexing.py -d /path/to/your/project -m incremental

# Skip confirmation for automated updates
python manual_indexing.py -d /path/to/your/project -m incremental --no-confirm
```

**Expected Output:**
```
🔍 Incremental Analysis for /path/to/your/project
├── Files scanned: 250
├── Files changed since last index: 8
├── New files: 2
├── Modified files: 6
├── Estimated processing time: 30-60 seconds
└── Expected time savings: 85%

Files to process:
├── src/auth/user_service.py (modified)
├── src/api/routes.py (modified)
├── src/utils/helpers.ts (new)
└── ...

Proceed with incremental indexing? [y/N]:
```

## Language-Specific Examples

### Python Projects

```bash
# Index Python web application
python manual_indexing.py -d /django/project -m clear_existing --verbose

# Index data science project with notebooks
python manual_indexing.py -d /ml/research -m clear_existing --verbose
```

**Python Chunking Results:**
```json
{
  "chunks_extracted": {
    "functions": 145,
    "classes": 32,
    "methods": 178,
    "constants": 23
  },
  "metadata_extracted": {
    "docstrings": 89,
    "type_hints": 156,
    "decorators": 45
  },
  "syntax_errors": {
    "minor_errors": 2,
    "major_errors": 0,
    "recovery_rate": "100%"
  }
}
```

### JavaScript/TypeScript Projects

```bash
# Index React frontend application
python manual_indexing.py -d /react/frontend -m clear_existing --verbose

# Index Node.js backend API
python manual_indexing.py -d /node/backend -m incremental
```

**TypeScript Chunking Results:**
```json
{
  "chunks_extracted": {
    "functions": 89,
    "interfaces": 34,
    "types": 56,
    "classes": 12,
    "constants": 18
  },
  "metadata_extracted": {
    "jsdoc_comments": 67,
    "type_annotations": 134,
    "generic_types": 23
  }
}
```

### Go Microservices

```bash
# Index Go microservice
python manual_indexing.py -d /go/microservice -m clear_existing --verbose
```

**Go Chunking Results:**
```json
{
  "chunks_extracted": {
    "functions": 78,
    "structs": 25,
    "interfaces": 12,
    "methods": 92
  },
  "metadata_extracted": {
    "doc_comments": 45,
    "struct_tags": 67,
    "receiver_types": 92
  }
}
```

### Multi-Language Monorepo

```bash
# Index full-stack monorepo
python manual_indexing.py -d /monorepo -m clear_existing --verbose

# Update specific service
python manual_indexing.py -d /monorepo/services/auth -m incremental
```

## Performance Optimization

### Large Codebase Configuration

For repositories with 1000+ files, optimize with environment variables:

```bash
# Set optimal concurrency for your system
export INDEXING_CONCURRENCY=8
export INDEXING_BATCH_SIZE=50
export EMBEDDING_BATCH_SIZE=20
export MEMORY_WARNING_THRESHOLD_MB=2000

# Run indexing
python manual_indexing.py -d /large/codebase -m clear_existing --no-confirm
```

### Memory-Constrained Environments

```bash
# Conservative settings for limited memory
export INDEXING_CONCURRENCY=2
export INDEXING_BATCH_SIZE=10
export EMBEDDING_BATCH_SIZE=5
export MAX_FILE_SIZE_MB=2

python manual_indexing.py -d /project -m clear_existing --verbose
```

### SSD vs HDD Optimization

```bash
# SSD optimization (faster I/O)
export INDEXING_CONCURRENCY=12
export INDEXING_BATCH_SIZE=100

# HDD optimization (less concurrent I/O)
export INDEXING_CONCURRENCY=4
export INDEXING_BATCH_SIZE=20
```

## Error Handling and Debugging

### Syntax Error Analysis

```bash
# Run with verbose output to see syntax errors
python manual_indexing.py -d /project/with/errors -m clear_existing --verbose

# Save detailed error report
python manual_indexing.py -d /project -m clear_existing --error-report-dir ./reports
```

**Error Report Example:**
```json
{
  "operation_type": "clear_existing",
  "directory": "/project/with/errors",
  "syntax_errors": [
    {
      "error_type": "syntax",
      "file_path": "src/broken_file.py",
      "line_number": 45,
      "error_message": "Unexpected token: missing closing parenthesis",
      "severity": "warning",
      "context": "def calculate_score(items\n    return sum(...",
      "suggestion": "Add missing closing parenthesis in function definition"
    }
  ],
  "recovery_statistics": {
    "total_chunks_attempted": 156,
    "successful_chunks": 152,
    "fallback_to_whole_file": 4,
    "recovery_rate": "97.4%"
  }
}
```

### Debugging Tree-sitter Issues

```bash
# Check if language parsers are available
python -c "
from src.services.code_parser_service import CodeParserService
parser = CodeParserService()
print('Supported languages:', list(parser.supported_languages.keys()))
"

# Test specific file parsing
python -c "
from src.services.code_parser_service import CodeParserService
parser = CodeParserService()
chunks = parser.parse_file('problematic_file.py')
for chunk in chunks:
    print(f'{chunk.name}: {chunk.chunk_type} (errors: {chunk.has_syntax_errors})')
"
```

### Performance Debugging

```bash
# Monitor memory usage and processing rates
python manual_indexing.py -d /large/project -m clear_existing --verbose 2>&1 | tee indexing.log

# Analyze log for bottlenecks
grep -E "(Memory usage|Processing rate|ETA)" indexing.log
```

## CI/CD Integration

### GitHub Actions

```yaml
name: Update Code Index
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  index:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
          
      - name: Install dependencies
        run: |
          python -m pip install poetry
          poetry install
          
      - name: Start Qdrant
        run: |
          docker run -d -p 6333:6333 qdrant/qdrant
          
      - name: Incremental indexing
        run: |
          poetry run python manual_indexing.py -d . -m incremental --no-confirm --verbose
          
      - name: Upload error reports
        uses: actions/upload-artifact@v3
        if: always()
        with:
          name: indexing-reports
          path: indexing_report_*.json
```

### GitLab CI

```yaml
stages:
  - index

update-index:
  stage: index
  image: python:3.10
  services:
    - qdrant/qdrant:latest
  variables:
    QDRANT_HOST: qdrant
  script:
    - pip install poetry
    - poetry install
    - poetry run python manual_indexing.py -d . -m incremental --no-confirm
  artifacts:
    reports:
      junit: indexing_report_*.json
    when: always
    expire_in: 1 week
  only:
    - main
    - develop
```

### Jenkins Pipeline

```groovy
pipeline {
    agent any
    
    environment {
        INDEXING_CONCURRENCY = '4'
        MEMORY_WARNING_THRESHOLD_MB = '1000'
    }
    
    stages {
        stage('Setup') {
            steps {
                sh 'docker run -d -p 6333:6333 --name qdrant qdrant/qdrant'
                sh 'pip install poetry && poetry install'
            }
        }
        
        stage('Incremental Index') {
            steps {
                sh '''
                    poetry run python manual_indexing.py \
                        -d . \
                        -m incremental \
                        --no-confirm \
                        --verbose \
                        --error-report-dir ./reports
                '''
            }
            post {
                always {
                    archiveArtifacts artifacts: 'reports/*.json', fingerprint: true
                }
            }
        }
        
        stage('Full Index Weekly') {
            when {
                cron('0 2 * * 0') // Sunday 2 AM
            }
            steps {
                sh '''
                    poetry run python manual_indexing.py \
                        -d . \
                        -m clear_existing \
                        --no-confirm \
                        --verbose
                '''
            }
        }
    }
    
    post {
        cleanup {
            sh 'docker stop qdrant && docker rm qdrant'
        }
    }
}
```

## Real-World Scenarios

### Scenario 1: Large Enterprise Monorepo

**Challenge:** 5000+ files, multiple languages, tight memory constraints

```bash
# Configuration
export INDEXING_CONCURRENCY=6
export INDEXING_BATCH_SIZE=25
export EMBEDDING_BATCH_SIZE=10
export MEMORY_WARNING_THRESHOLD_MB=1500
export MAX_FILE_SIZE_MB=3

# Initial indexing (run during off-hours)
python manual_indexing.py -d /enterprise/monorepo -m clear_existing --no-confirm --verbose

# Daily incremental updates
python manual_indexing.py -d /enterprise/monorepo -m incremental --no-confirm
```

**Results:**
- Initial indexing: 45 minutes
- Daily updates: 2-5 minutes
- Chunk extraction: 15,000+ functions/classes
- Memory usage: ~1.2GB peak

### Scenario 2: AI/ML Research Project

**Challenge:** Mixed Python/Jupyter notebooks, frequent experimentation

```bash
# Handle notebooks and Python files
python manual_indexing.py -d /ml/research -m incremental --verbose

# Focus on specific experiment directory
python manual_indexing.py -d /ml/research/experiments/transformer-v2 -m clear_existing
```

**Benefits:**
- Function-level search for model architectures
- Easy retrieval of specific training loops
- Documentation chunking for research notes

### Scenario 3: Microservices Architecture

**Challenge:** Multiple repositories, different languages per service

```bash
# Index each microservice separately
for service in auth payment user notification; do
    python manual_indexing.py -d /microservices/$service -m incremental --no-confirm
done

# Or use custom project names
python manual_indexing.py -d /microservices/auth -m clear_existing --verbose
python manual_indexing.py -d /microservices/payment -m clear_existing --verbose
```

### Scenario 4: Legacy Code Modernization

**Challenge:** Old codebase with syntax errors, mixed coding styles

```bash
# Use verbose mode to identify problematic files
python manual_indexing.py -d /legacy/codebase -m clear_existing --verbose --error-report-dir ./legacy-reports

# Analyze error patterns
python -c "
import json
with open('legacy-reports/indexing_report_*.json') as f:
    report = json.load(f)
    print(f'Syntax errors: {len(report[\"syntax_errors\"])}')
    print(f'Recovery rate: {report[\"recovery_statistics\"][\"recovery_rate\"]}')
"
```

**Typical Results:**
- 15-20% files with minor syntax errors
- 95%+ recovery rate through intelligent fallback
- Detailed error reports guide modernization efforts

### Scenario 5: Open Source Project Analysis

**Challenge:** Unknown codebase, need to understand structure quickly

```bash
# Quick analysis with detailed reporting
python manual_indexing.py -d /opensource/project -m clear_existing --verbose

# Focus on core modules
python manual_indexing.py -d /opensource/project/src -m clear_existing --verbose
```

**Analysis Output:**
```
📊 Project Analysis Summary:
├── Languages: Python (70%), JavaScript (25%), Shell (5%)
├── Functions extracted: 450
├── Classes extracted: 67
├── Interfaces/Types: 23
├── Documentation coverage: 67%
└── Code quality indicators: High (low syntax error rate)

🔍 Top-level functions by complexity:
├── main_processor() - 89 lines (high complexity)
├── data_transformer() - 67 lines (medium complexity)
└── config_loader() - 45 lines (medium complexity)
```

## Best Practices Summary

### When to Use Each Mode

**Use `clear_existing` for:**
- First-time project indexing
- Major refactoring or restructuring
- Weekly/monthly full rebuilds
- After significant dependency updates
- When chunk structure has changed

**Use `incremental` for:**
- Daily development workflow
- Small feature additions
- Bug fixes and minor changes
- CI/CD automated updates
- Performance-sensitive environments

### Performance Tips

1. **Optimize for your hardware:**
   - CPU cores: Set `INDEXING_CONCURRENCY` to CPU count
   - Memory: Adjust batch sizes based on available RAM
   - Storage: Higher concurrency for SSDs, lower for HDDs

2. **Use `.ragignore` effectively:**
   ```
   node_modules/
   .git/
   dist/
   build/
   __pycache__/
   .pytest_cache/
   coverage/
   *.log
   *.tmp
   ```

3. **Monitor and tune:**
   - Use `--verbose` to identify bottlenecks
   - Save error reports to track improvements
   - Adjust batch sizes based on memory warnings

### Error Handling Strategy

1. **Minor syntax errors:** Let intelligent chunking handle them
2. **Major syntax errors:** Fix critical issues, let tool fallback gracefully
3. **Persistent errors:** Use error reports to prioritize fixes
4. **Performance issues:** Tune environment variables and batch sizes

This comprehensive guide should help you effectively use the manual indexing tool for any project size or complexity. The intelligent chunking system is designed to be robust and handle real-world code with grace while providing detailed feedback for optimization.