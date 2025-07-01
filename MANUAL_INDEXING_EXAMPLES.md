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
2025-07-01 14:45:08,341 - services.embedding_service.EmbeddingService - INFO - MPS is available. Using Metal for acceleration.
2025-07-01 14:45:08,342 - utils.tree_sitter_manager - INFO - Successfully initialized python parser (version 14)
2025-07-01 14:45:08,343 - utils.tree_sitter_manager - INFO - Successfully initialized javascript parser (version 14)
2025-07-01 14:45:08,345 - utils.tree_sitter_manager - INFO - Successfully initialized typescript parser (version 14)
2025-07-01 14:45:08,345 - utils.tree_sitter_manager - INFO - Successfully initialized tsx parser (version 14)
2025-07-01 14:45:08,348 - utils.tree_sitter_manager - INFO - Successfully initialized go parser (version 14)
2025-07-01 14:45:08,349 - utils.tree_sitter_manager - INFO - Successfully initialized rust parser (version 14)
2025-07-01 14:45:08,350 - utils.tree_sitter_manager - INFO - Successfully initialized java parser (version 14)
2025-07-01 14:45:08,352 - utils.tree_sitter_manager - INFO - Successfully initialized cpp parser (version 14)
2025-07-01 14:45:08,352 - services.code_parser_service - INFO - Initialized Tree-sitter parsers: 8/8 languages successful

============================================================
MANUAL INDEXING TOOL - PRE-INDEXING SUMMARY
============================================================
📁 Directory: /Users/jeff/Documents/code_repo/test_project
⚙️  Mode: clear_existing
📊 Files to process: 4,309
💾 Total size: 23.1 MB
⏱️  Estimated time: 99.6 minutes

⚠️  WARNING: This operation may take several minutes.
   Consider running this in a separate terminal.

------------------------------------------------------------

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
2025-07-01 14:45:08,352 - services.code_parser_service - INFO - Initialized Tree-sitter parsers: 8/8 languages successful

============================================================
MANUAL INDEXING TOOL - PRE-INDEXING SUMMARY
============================================================
📁 Directory: /path/to/your/project
⚙️  Mode: incremental
📊 Files to process: 8 (changed from 250 total)
💾 Total size: 145.2 KB
⏱️  Estimated time: 1.2 minutes

Files changed since last index:
├── src/auth/user_service.py (modified)
├── src/api/routes.py (modified) 
├── src/utils/helpers.ts (new)
├── tests/test_auth.py (modified)
└── ... 4 more files

⚡ Expected time savings: 85% (vs full reindex)

------------------------------------------------------------

Proceed with indexing? [y/N]:
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
```
2025-07-01 14:47:23,156 - services.indexing_service - INFO - Processing Python files: 120/250
2025-07-01 14:47:23,157 - services.code_parser_service - INFO - Extracted 145 functions, 32 classes, 178 methods from Python files
2025-07-01 14:47:23,158 - services.code_parser_service - INFO - Found 89 docstrings, 156 type hints
2025-07-01 14:47:23,159 - services.code_parser_service - WARNING - 2 files had minor syntax errors, gracefully handled
2025-07-01 14:47:23,160 - services.indexing_service - INFO - Python processing complete: 99.2% success rate

Final Statistics:
├── Functions extracted: 145
├── Classes extracted: 32  
├── Methods extracted: 178
├── Constants/Variables: 23
├── Syntax error recovery: 100%
└── Processing success rate: 99.2%
```

### JavaScript/TypeScript Projects

```bash
# Index React frontend application
python manual_indexing.py -d /react/frontend -m clear_existing --verbose

# Index Node.js backend API
python manual_indexing.py -d /node/backend -m incremental
```

**TypeScript Chunking Results:**
```
2025-07-01 14:48:15,234 - services.code_parser_service - INFO - Processing TypeScript files: 45/60
2025-07-01 14:48:15,235 - services.code_parser_service - INFO - Extracted 89 functions, 34 interfaces, 56 types
2025-07-01 14:48:15,236 - services.code_parser_service - INFO - Found 67 JSDoc comments, 134 type annotations
2025-07-01 14:48:15,237 - services.indexing_service - INFO - TypeScript processing complete: 100% success rate

Final Statistics:
├── Functions extracted: 89
├── Interfaces extracted: 34
├── Type definitions: 56
├── Classes extracted: 12
├── Constants/Objects: 18
└── JSDoc coverage: 75%
```

### Go Microservices

```bash
# Index Go microservice
python manual_indexing.py -d /go/microservice -m clear_existing --verbose
```

**Go Chunking Results:**
```
2025-07-01 14:49:02,123 - services.code_parser_service - INFO - Processing Go files: 30/30
2025-07-01 14:49:02,124 - services.code_parser_service - INFO - Extracted 78 functions, 25 structs, 12 interfaces
2025-07-01 14:49:02,125 - services.code_parser_service - INFO - Found 45 doc comments, 67 struct tags, 92 methods
2025-07-01 14:49:02,126 - services.indexing_service - INFO - Go processing complete: 100% success rate

Final Statistics:
├── Functions extracted: 78
├── Structs extracted: 25
├── Interfaces extracted: 12
├── Methods (with receivers): 92
├── Doc comment coverage: 58%
└── Processing success rate: 100%
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

**Expected Output with Optimization:**
```
2025-07-01 14:45:08,341 - services.embedding_service.EmbeddingService - INFO - MPS is available. Using Metal for acceleration.
2025-07-01 14:45:08,352 - services.code_parser_service - INFO - Initialized Tree-sitter parsers: 8/8 languages successful

============================================================
MANUAL INDEXING TOOL - PRE-INDEXING SUMMARY
============================================================
📁 Directory: /large/codebase
⚙️  Mode: clear_existing
📊 Files to process: 12,450
💾 Total size: 156.7 MB
⏱️  Estimated time: 8.2 hours

⚠️  WARNING: Large repository detected.
   This operation will take several hours.
   Consider running overnight or in a dedicated session.

Performance Settings Detected:
├── Concurrency: 8 workers
├── Batch size: 50 files/batch
├── Embedding batch: 20 texts/batch
├── Memory threshold: 2000 MB
└── GPU acceleration: Metal (MPS) ✓

------------------------------------------------------------

Proceed with indexing? [y/N]:
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
```
2025-07-01 15:23:45,123 - services.code_parser_service - WARNING - Syntax error in src/broken_file.py:45 - Missing closing parenthesis
2025-07-01 15:23:45,124 - services.code_parser_service - INFO - Recovered 3/4 chunks from src/broken_file.py, using fallback for 1 chunk
2025-07-01 15:23:45,125 - services.indexing_service - WARNING - File src/legacy_code.js has multiple syntax errors, using whole-file fallback

============================================================
INDEXING COMPLETE - ERROR SUMMARY
============================================================
📊 Processing Statistics:
├── Total files processed: 4,309
├── Successful files: 4,305 (99.9%)
├── Files with minor errors: 3
├── Files requiring fallback: 1
└── Overall success rate: 99.9%

🔧 Syntax Error Details:
├── src/broken_file.py:45 - Missing closing parenthesis (recovered)
├── src/utils/helper.py:23 - Invalid indentation (recovered) 
├── src/legacy_code.js:multiple - Multiple parse errors (fallback used)
└── Recovery rate: 97.4% of chunks successfully extracted

📝 Error Report Saved: indexing_report_project_20250701_152345.json

💡 Recommendations:
├── Fix syntax errors in src/broken_file.py and src/utils/helper.py
├── Consider refactoring src/legacy_code.js for better parsing
└── Overall indexing very successful - minor issues don't affect functionality
```

### Debugging Tree-sitter Issues

```bash
# Check if language parsers are available
python -c "
from src.services.code_parser_service import CodeParserService
parser = CodeParserService()
print('Supported languages:', list(parser.supported_languages.keys()))
"

# Expected output:
# 2025-07-01 14:45:08,342 - utils.tree_sitter_manager - INFO - Successfully initialized python parser (version 14)
# 2025-07-01 14:45:08,343 - utils.tree_sitter_manager - INFO - Successfully initialized javascript parser (version 14)
# 2025-07-01 14:45:08,345 - utils.tree_sitter_manager - INFO - Successfully initialized typescript parser (version 14)
# 2025-07-01 14:45:08,345 - utils.tree_sitter_manager - INFO - Successfully initialized tsx parser (version 14)
# 2025-07-01 14:45:08,348 - utils.tree_sitter_manager - INFO - Successfully initialized go parser (version 14)
# 2025-07-01 14:45:08,349 - utils.tree_sitter_manager - INFO - Successfully initialized rust parser (version 14)
# 2025-07-01 14:45:08,350 - utils.tree_sitter_manager - INFO - Successfully initialized java parser (version 14)
# 2025-07-01 14:45:08,352 - utils.tree_sitter_manager - INFO - Successfully initialized cpp parser (version 14)
# 2025-07-01 14:45:08,352 - services.code_parser_service - INFO - Initialized Tree-sitter parsers: 8/8 languages successful
# Supported languages: ['.py', '.js', '.jsx', '.ts', '.tsx', '.go', '.rs', '.java', '.cpp', '.c', '.h']

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
```
============================================================
INDEXING COMPLETE - ENTERPRISE MONOREPO SUMMARY
============================================================
📁 Directory: /enterprise/monorepo
📊 Total files processed: 5,247
💾 Total size: 287.3 MB
⏱️  Total time: 42.3 minutes

🎯 Intelligent Chunking Results:
├── Functions extracted: 8,456
├── Classes extracted: 1,204  
├── Methods extracted: 6,743
├── Interfaces/Types: 892
├── Constants/Configs: 567
└── Total semantic chunks: 16,862

🔧 Language Breakdown:
├── Python: 3,245 files → 7,234 chunks
├── TypeScript: 1,456 files → 5,891 chunks  
├── Go: 334 files → 2,456 chunks
├── Java: 212 files → 1,281 chunks
└── Others: 0 files → 0 chunks

⚡ Performance Metrics:
├── Processing rate: 124 files/minute
├── Peak memory usage: 1.2 GB
├── GPU acceleration: Metal (MPS) ✓
└── Syntax error recovery: 99.1%

💡 Daily incremental updates typically take 2-5 minutes
```

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
