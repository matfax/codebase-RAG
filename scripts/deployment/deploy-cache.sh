#!/bin/bash

# Cache Deployment Automation Script
# Handles deployment of the Query Caching Layer with health checks and rollback capability

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.cache.yml"
ENV_FILE="$PROJECT_ROOT/.env"
LOG_FILE="/tmp/cache-deployment-$(date +%Y%m%d-%H%M%S).log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" | tee -a "$LOG_FILE" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*" | tee -a "$LOG_FILE"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        return 1
    fi

    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed"
        return 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        error "Docker daemon is not running"
        return 1
    fi

    # Check environment file
    if [[ ! -f "$ENV_FILE" ]]; then
        warning "Environment file not found at $ENV_FILE"
        warning "Creating from example..."
        if [[ -f "$PROJECT_ROOT/.env.example" ]]; then
            cp "$PROJECT_ROOT/.env.example" "$ENV_FILE"
            log "Created .env file from .env.example"
        else
            error "No .env.example file found"
            return 1
        fi
    fi

    log "Prerequisites check passed"
    return 0
}

# Validate environment configuration
validate_configuration() {
    log "Validating configuration..."

    # Source environment file
    if [[ -f "$ENV_FILE" ]]; then
        set -a
        source "$ENV_FILE"
        set +a
    fi

    # Check required variables
    local required_vars=(
        "REDIS_HOST"
        "REDIS_PORT"
        "REDIS_PASSWORD"
        "CACHE_ENABLED"
    )

    local missing_vars=()
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            missing_vars+=("$var")
        fi
    done

    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        error "Missing required environment variables: ${missing_vars[*]}"
        return 1
    fi

    # Validate Redis password strength
    if [[ ${#REDIS_PASSWORD} -lt 8 ]]; then
        warning "Redis password should be at least 8 characters long"
    fi

    log "Configuration validation passed"
    return 0
}

# Backup existing data
backup_data() {
    log "Creating backup of existing data..."

    local backup_dir="$PROJECT_ROOT/backups/$(date +%Y%m%d-%H%M%S)"
    mkdir -p "$backup_dir"

    # Check if Redis container is running
    if docker ps --format '{{.Names}}' | grep -q "codebase_rag_redis"; then
        log "Backing up Redis data..."
        docker exec codebase_rag_redis redis-cli --rdb "$backup_dir/dump.rdb" BGSAVE

        # Wait for backup to complete
        while docker exec codebase_rag_redis redis-cli LASTSAVE | grep -q "in progress"; do
            sleep 1
        done

        # Copy backup file
        docker cp "codebase_rag_redis:/data/dump.rdb" "$backup_dir/dump.rdb"
        log "Redis backup completed: $backup_dir/dump.rdb"
    else
        warning "Redis container not running, skipping backup"
    fi

    # Save deployment metadata
    cat > "$backup_dir/metadata.json" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "version": "$(git describe --tags --always 2>/dev/null || echo 'unknown')",
    "environment": {
        "REDIS_HOST": "${REDIS_HOST:-localhost}",
        "REDIS_PORT": "${REDIS_PORT:-6379}",
        "CACHE_LEVEL": "${CACHE_LEVEL:-BOTH}"
    }
}
EOF

    log "Backup completed: $backup_dir"
    echo "$backup_dir"
}

# Deploy cache services
deploy_services() {
    log "Deploying cache services..."

    # Pull latest images
    log "Pulling latest Docker images..."
    docker-compose -f "$COMPOSE_FILE" pull

    # Start services
    log "Starting cache services..."
    docker-compose -f "$COMPOSE_FILE" up -d --remove-orphans

    # Wait for services to be ready
    log "Waiting for services to be ready..."
    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if docker-compose -f "$COMPOSE_FILE" ps | grep -q "healthy"; then
            log "Services are healthy"
            return 0
        fi

        attempt=$((attempt + 1))
        log "Waiting for services to be healthy... (attempt $attempt/$max_attempts)"
        sleep 2
    done

    error "Services failed to become healthy within timeout"
    return 1
}

# Run health checks
run_health_checks() {
    log "Running health checks..."

    # Check Redis connectivity
    log "Checking Redis connectivity..."
    if docker exec codebase_rag_redis redis-cli -a "$REDIS_PASSWORD" ping | grep -q "PONG"; then
        log "Redis is responding"
    else
        error "Redis health check failed"
        return 1
    fi

    # Check Redis memory
    local used_memory=$(docker exec codebase_rag_redis redis-cli -a "$REDIS_PASSWORD" INFO memory | grep "used_memory_human" | cut -d: -f2 | tr -d '\r')
    log "Redis memory usage: $used_memory"

    # Check Redis persistence
    local last_save=$(docker exec codebase_rag_redis redis-cli -a "$REDIS_PASSWORD" LASTSAVE)
    log "Redis last save: $(date -d @$last_save 2>/dev/null || echo $last_save)"

    # Test cache operations
    log "Testing cache operations..."
    docker exec codebase_rag_redis redis-cli -a "$REDIS_PASSWORD" SET "health_check_key" "test_value" EX 60
    local test_value=$(docker exec codebase_rag_redis redis-cli -a "$REDIS_PASSWORD" GET "health_check_key")

    if [[ "$test_value" == "test_value" ]]; then
        log "Cache operations test passed"
        docker exec codebase_rag_redis redis-cli -a "$REDIS_PASSWORD" DEL "health_check_key"
    else
        error "Cache operations test failed"
        return 1
    fi

    log "All health checks passed"
    return 0
}

# Rollback deployment
rollback_deployment() {
    local backup_dir="$1"

    error "Rolling back deployment..."

    # Stop current services
    docker-compose -f "$COMPOSE_FILE" down

    # Restore from backup if available
    if [[ -d "$backup_dir" ]] && [[ -f "$backup_dir/dump.rdb" ]]; then
        log "Restoring from backup: $backup_dir"

        # Start Redis temporarily
        docker-compose -f "$COMPOSE_FILE" up -d redis
        sleep 5

        # Restore data
        docker cp "$backup_dir/dump.rdb" "codebase_rag_redis:/data/dump.rdb"
        docker exec codebase_rag_redis redis-cli -a "$REDIS_PASSWORD" SHUTDOWN SAVE

        # Restart services
        docker-compose -f "$COMPOSE_FILE" up -d

        log "Rollback completed"
    else
        warning "No backup available for rollback"
    fi
}

# Generate deployment report
generate_report() {
    local status="$1"
    local report_file="$PROJECT_ROOT/deployment-report-$(date +%Y%m%d-%H%M%S).json"

    log "Generating deployment report..."

    local container_status=""
    if command -v docker &> /dev/null; then
        container_status=$(docker-compose -f "$COMPOSE_FILE" ps --format json 2>/dev/null || echo "{}")
    fi

    cat > "$report_file" <<EOF
{
    "deployment_status": "$status",
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "environment": {
        "REDIS_HOST": "${REDIS_HOST:-localhost}",
        "REDIS_PORT": "${REDIS_PORT:-6379}",
        "CACHE_ENABLED": "${CACHE_ENABLED:-true}",
        "CACHE_LEVEL": "${CACHE_LEVEL:-BOTH}"
    },
    "containers": $container_status,
    "log_file": "$LOG_FILE"
}
EOF

    log "Deployment report saved: $report_file"
}

# Main deployment function
main() {
    log "Starting cache deployment..."
    log "Log file: $LOG_FILE"

    # Initialize deployment status
    local deployment_status="failed"
    local backup_dir=""

    # Check prerequisites
    if ! check_prerequisites; then
        error "Prerequisites check failed"
        generate_report "failed"
        exit 1
    fi

    # Validate configuration
    if ! validate_configuration; then
        error "Configuration validation failed"
        generate_report "failed"
        exit 1
    fi

    # Create backup
    backup_dir=$(backup_data)

    # Deploy services
    if deploy_services; then
        # Run health checks
        if run_health_checks; then
            deployment_status="success"
            log "Deployment completed successfully!"
        else
            error "Health checks failed"
            rollback_deployment "$backup_dir"
        fi
    else
        error "Service deployment failed"
        rollback_deployment "$backup_dir"
    fi

    # Generate final report
    generate_report "$deployment_status"

    # Exit with appropriate code
    if [[ "$deployment_status" == "success" ]]; then
        exit 0
    else
        exit 1
    fi
}

# Handle script arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    rollback)
        if [[ -n "${2:-}" ]]; then
            rollback_deployment "$2"
        else
            error "Backup directory required for rollback"
            echo "Usage: $0 rollback <backup-directory>"
            exit 1
        fi
        ;;
    health)
        run_health_checks
        ;;
    *)
        echo "Usage: $0 [deploy|rollback|health]"
        exit 1
        ;;
esac
