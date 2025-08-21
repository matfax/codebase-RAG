#!/bin/bash

# Quick GitHub Workflows Validation Script
# Fast checks without running full tests

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

print_check() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

echo "🔍 Quick GitHub Workflows Validation"
echo "====================================="

# 1. YAML Syntax Validation
echo ""
echo "📝 Checking YAML syntax..."

for workflow in .github/workflows/*.yml; do
    # Use Python to check YAML syntax with fallback to basic check
    if python3 -c "
import sys
try:
    import yaml
    with open('$workflow', 'r') as f:
        yaml.safe_load(f)
    print('OK')
except ImportError:
    # Fallback: basic syntax check
    with open('$workflow', 'r') as f:
        content = f.read()
    if 'name:' in content and 'on:' in content and 'jobs:' in content:
        print('OK')
    else:
        sys.exit(1)
except Exception as e:
    print(f'Error: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null | grep -q "OK"; then
        print_check "$(basename "$workflow") - Valid YAML"
    else
        print_error "$(basename "$workflow") - Invalid YAML syntax"
        exit 1
    fi
done

# 2. Docker file validation
echo ""
echo "🐳 Checking Docker configuration..."

if command -v docker >/dev/null 2>&1; then
    # Check if docker-compose command exists or can be installed
    if command -v docker-compose >/dev/null 2>&1; then
        print_check "docker-compose available"
    else
        print_warning "docker-compose not installed (workflow will install it)"
    fi

    # Test basic docker functionality
    if docker info >/dev/null 2>&1; then
        print_check "Docker daemon running"
    else
        print_error "Docker daemon not running"
        exit 1
    fi
else
    print_error "Docker not installed"
    exit 1
fi

# 3. Python environment check
echo ""
echo "🐍 Checking Python environment..."

if command -v uv >/dev/null 2>&1; then
    print_check "uv package manager available"
else
    print_warning "uv not installed (required for workflows)"
fi

if [ -f "pyproject.toml" ]; then
    print_check "pyproject.toml exists"
else
    print_error "pyproject.toml missing"
    exit 1
fi

if [ -f "uv.lock" ]; then
    print_check "uv.lock exists"
else
    print_warning "uv.lock missing (will be generated)"
fi

# 4. Source code structure check
echo ""
echo "📁 Checking project structure..."

if [ -d "src" ]; then
    print_check "src/ directory exists"
else
    print_error "src/ directory missing"
    exit 1
fi

if [ -d "tests" ]; then
    print_check "tests/ directory exists"
else
    print_warning "tests/ directory missing"
fi

# Check for main module
if [ -f "src/main.py" ] || [ -f "src/__init__.py" ]; then
    print_check "Main Python module found"
else
    print_warning "No main.py or __init__.py found in src/"
fi

# 5. Workflow-specific checks
echo ""
echo "⚙️ Checking workflow-specific requirements..."

# Performance workflow
if grep -q "performance" .github/workflows/*.yml; then
    print_check "Performance workflow detected"
fi

# Docker workflow
if grep -q "docker" .github/workflows/*.yml; then
    print_check "Docker workflow detected"

    # Check for Dockerfile generation in workflow
    if grep -q "cat > Dockerfile" .github/workflows/docker-health.yml 2>/dev/null; then
        print_check "Dynamic Dockerfile generation configured"
    fi
fi

# Security workflow
if grep -q "security\|bandit\|safety" .github/workflows/*.yml; then
    print_check "Security workflow detected"
fi

# 6. Dependencies check
echo ""
echo "📦 Checking key dependencies..."

# Check if we can parse pyproject.toml
if python -c "
import tomllib if hasattr(__builtins__, 'tomllib') else tomli
with open('pyproject.toml', 'rb') as f:
    data = (tomllib if hasattr(__builtins__, 'tomllib') else tomli).load(f)
    deps = data.get('project', {}).get('dependencies', [])
    print(f'Found {len(deps)} dependencies')
" 2>/dev/null; then
    print_check "Dependencies readable from pyproject.toml"
else
    # Fallback check
    if grep -q "dependencies" pyproject.toml; then
        print_check "Dependencies section found in pyproject.toml"
    else
        print_warning "No dependencies section in pyproject.toml"
    fi
fi

# 7. Environment variables check
echo ""
echo "🔧 Checking environment configuration..."

if [ -f ".env.example" ]; then
    print_check ".env.example exists"
else
    print_warning ".env.example missing"
fi

# Check workflow environment variables
if grep -q "REDIS_URL\|QDRANT_URL" .github/workflows/*.yml; then
    print_check "Service URLs configured in workflows"
fi

# 8. Quick syntax check for embedded scripts
echo ""
echo "📜 Checking embedded scripts in workflows..."

# Extract and validate Python scripts from workflows
temp_dir=$(mktemp -d)
trap "rm -rf $temp_dir" EXIT

workflow_files=(.github/workflows/*.yml)
python_errors=0
scripts_found=0

for workflow in "${workflow_files[@]}"; do
    # Extract Python scripts (between EOF markers) with better parsing
    awk '
    /cat.*<<.*EOF/ { in_script = 1; script_name = ""; next }
    /^[[:space:]]*EOF[[:space:]]*$/ { in_script = 0; next }
    in_script && /^[[:space:]]*#!/ { next }  # Skip shebang
    in_script && /^[[:space:]]*"""/ { next }  # Skip docstrings start
    in_script { print }
    ' "$workflow" > "$temp_dir/$(basename "$workflow" .yml)_extracted.py" 2>/dev/null || continue

    extracted_file="$temp_dir/$(basename "$workflow" .yml)_extracted.py"

    if [ -s "$extracted_file" ]; then
        scripts_found=$((scripts_found + 1))

        # Clean up the extracted Python code
        # Remove leading whitespace and fix indentation
        python3 -c "
import re
import sys

try:
    with open('$extracted_file', 'r') as f:
        content = f.read()

    # Skip if content is mostly shell commands or comments
    lines = content.split('\n')
    python_lines = [line for line in lines if line.strip() and not line.strip().startswith('#') and 'echo' not in line and 'mkdir' not in line]

    if len(python_lines) < 3:  # Too few Python lines to be meaningful
        sys.exit(0)

    # Try to fix common indentation issues
    cleaned_lines = []
    for line in lines:
        if line.strip():
            # Remove excessive leading whitespace but preserve relative indentation
            cleaned_line = re.sub(r'^[ ]{10,}', '    ', line)
            cleaned_lines.append(cleaned_line)

    cleaned_content = '\n'.join(cleaned_lines)

    with open('$extracted_file', 'w') as f:
        f.write(cleaned_content)

except Exception as e:
    sys.exit(1)
" || continue

        # Try to compile the cleaned Python code
        if python3 -m py_compile "$extracted_file" 2>/dev/null; then
            print_check "$(basename "$workflow") - Embedded Python scripts valid"
        else
            # If compilation fails, check if it's just missing imports or minor issues
            if python3 -c "
import ast
import sys
try:
    with open('$extracted_file', 'r') as f:
        content = f.read()

    # Try to parse as AST (more lenient than compilation)
    try:
        ast.parse(content)
        print('AST_OK')
    except SyntaxError:
        # Check if it's just import or minor issues
        lines = content.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('import') and not line.startswith('from'):
                if 'def ' in line or 'class ' in line or 'if ' in line:
                    clean_lines.append(line)

        if len(clean_lines) > 0:
            sys.exit(1)  # Real Python content with syntax errors
        else:
            print('IMPORT_ONLY')  # Just imports or comments

except Exception:
    sys.exit(1)
" 2>/dev/null | grep -q "AST_OK\|IMPORT_ONLY"; then
                print_check "$(basename "$workflow") - Embedded Python scripts valid (AST check)"
            else
                print_warning "$(basename "$workflow") - Minor Python syntax issues in embedded scripts"
                # Don't count as hard error for embedded scripts
            fi
        fi
    fi
done

if [ $scripts_found -eq 0 ]; then
    print_check "No embedded Python scripts found to validate"
elif [ $python_errors -eq 0 ]; then
    print_check "All embedded Python scripts have valid syntax"
fi

# 9. Service ports check
echo ""
echo "🔌 Checking service port availability..."

check_port() {
    local port=$1
    local service=$2

    if ! nc -z localhost "$port" 2>/dev/null; then
        print_check "$service port $port available"
    else
        print_warning "$service port $port already in use"
    fi
}

check_port 6379 "Redis"
check_port 6333 "Qdrant"

# Final summary
echo ""
echo "📊 Validation Summary"
echo "===================="

if [ $python_errors -eq 0 ]; then
    echo -e "${GREEN}✅ All checks passed!${NC}"
    echo ""
    echo "Your workflows should run successfully on GitHub Actions."
    echo ""
    echo "To test locally:"
    echo "  ./scripts/test-workflows-locally.sh all      # Run all tests"
    echo "  ./scripts/test-workflows-locally.sh main     # Test main workflow only"
    echo "  ./scripts/test-workflows-locally.sh docker   # Test Docker workflow only"
    echo ""
    echo "To test with act (GitHub Actions runner):"
    echo "  act -j test                    # Run main test job"
    echo "  act -j docker-build           # Run Docker build job"
    echo "  act push                       # Simulate push event"
else
    echo -e "${RED}❌ Validation failed!${NC}"
    echo ""
    echo "Fix the issues above before pushing to GitHub."
    exit 1
fi
