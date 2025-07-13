#!/bin/bash

# Cache Backup and Restore Script
# Provides comprehensive backup and restore functionality for the cache system

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BACKUP_DIR="${BACKUP_DIR:-$PROJECT_ROOT/backups}"
COMPOSE_FILE="$PROJECT_ROOT/docker-compose.cache.yml"
REDIS_CONTAINER="codebase_rag_redis"

# Source environment
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $*"
}

error() {
    echo -e "${RED}[ERROR]${NC} $*" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $*"
}

info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

# Check if Redis is running
check_redis() {
    if ! docker ps --format '{{.Names}}' | grep -q "$REDIS_CONTAINER"; then
        error "Redis container is not running"
        return 1
    fi
    return 0
}

# Create backup
create_backup() {
    local backup_name="${1:-}"
    local backup_timestamp=$(date +%Y%m%d-%H%M%S)
    local backup_path="$BACKUP_DIR/$backup_timestamp"

    if [[ -n "$backup_name" ]]; then
        backup_path="$BACKUP_DIR/${backup_name}_$backup_timestamp"
    fi

    log "Creating backup at: $backup_path"
    mkdir -p "$backup_path"

    # Check Redis
    if ! check_redis; then
        error "Cannot create backup - Redis not running"
        return 1
    fi

    # Get Redis info
    info "Collecting Redis information..."
    docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" INFO > "$backup_path/redis-info.txt"

    # Get cache statistics
    info "Collecting cache statistics..."
    if command -v python3 &> /dev/null; then
        python3 "$SCRIPT_DIR/cache-stats.py" > "$backup_path/cache-stats.json" 2>/dev/null || true
    fi

    # Trigger background save
    log "Initiating Redis backup..."
    docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" BGSAVE

    # Wait for backup to complete
    info "Waiting for backup to complete..."
    while true; do
        if docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" LASTSAVE | grep -q "$(date +%s)"; then
            break
        fi
        sleep 1
    done

    # Copy Redis dump file
    log "Copying Redis data..."
    docker cp "$REDIS_CONTAINER:/data/dump.rdb" "$backup_path/dump.rdb"

    # Copy AOF file if exists
    if docker exec "$REDIS_CONTAINER" test -f /data/appendonly.aof; then
        log "Copying AOF file..."
        docker cp "$REDIS_CONTAINER:/data/appendonly.aof" "$backup_path/appendonly.aof"
    fi

    # Export cache keys for analysis
    info "Exporting cache key patterns..."
    docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" --scan --pattern "codebase_rag:*" > "$backup_path/cache-keys.txt"

    # Calculate key statistics
    local total_keys=$(wc -l < "$backup_path/cache-keys.txt")
    local embedding_keys=$(grep -c "embedding:" "$backup_path/cache-keys.txt" || true)
    local search_keys=$(grep -c "search:" "$backup_path/cache-keys.txt" || true)
    local project_keys=$(grep -c "project:" "$backup_path/cache-keys.txt" || true)
    local file_keys=$(grep -c "file:" "$backup_path/cache-keys.txt" || true)

    # Create metadata file
    cat > "$backup_path/metadata.json" <<EOF
{
    "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "backup_name": "${backup_name:-default}",
    "redis_version": "$(docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" INFO server | grep redis_version | cut -d: -f2 | tr -d '\r')",
    "environment": "${ENVIRONMENT:-development}",
    "statistics": {
        "total_keys": $total_keys,
        "embedding_keys": $embedding_keys,
        "search_keys": $search_keys,
        "project_keys": $project_keys,
        "file_keys": $file_keys
    },
    "backup_files": {
        "dump_rdb": "$(ls -lh "$backup_path/dump.rdb" 2>/dev/null | awk '{print $5}' || echo 'N/A')",
        "aof": "$(ls -lh "$backup_path/appendonly.aof" 2>/dev/null | awk '{print $5}' || echo 'N/A')"
    }
}
EOF

    # Compress backup
    if command -v tar &> /dev/null && command -v gzip &> /dev/null; then
        log "Compressing backup..."
        cd "$BACKUP_DIR"
        tar -czf "${backup_timestamp}.tar.gz" "$(basename "$backup_path")"
        local compressed_size=$(ls -lh "${backup_timestamp}.tar.gz" | awk '{print $5}')
        info "Compressed backup size: $compressed_size"
    fi

    log "Backup completed successfully"
    info "Backup location: $backup_path"
    info "Total keys backed up: $total_keys"

    # Cleanup old backups (keep last 10)
    cleanup_old_backups

    echo "$backup_path"
}

# Restore from backup
restore_backup() {
    local backup_path="$1"

    if [[ ! -d "$backup_path" ]]; then
        # Check if it's a compressed backup
        if [[ -f "$backup_path" ]] && [[ "$backup_path" == *.tar.gz ]]; then
            log "Extracting compressed backup..."
            local temp_dir="/tmp/cache-restore-$(date +%s)"
            mkdir -p "$temp_dir"
            tar -xzf "$backup_path" -C "$temp_dir"
            backup_path="$temp_dir/$(ls "$temp_dir")"
        else
            error "Backup not found: $backup_path"
            return 1
        fi
    fi

    # Check required files
    if [[ ! -f "$backup_path/dump.rdb" ]]; then
        error "Redis dump file not found in backup"
        return 1
    fi

    log "Restoring from backup: $backup_path"

    # Show backup metadata
    if [[ -f "$backup_path/metadata.json" ]]; then
        info "Backup metadata:"
        cat "$backup_path/metadata.json" | python3 -m json.tool
    fi

    # Confirm restore
    warning "This will replace all current cache data!"
    read -p "Are you sure you want to restore? (yes/no): " confirm

    if [[ "$confirm" != "yes" ]]; then
        log "Restore cancelled"
        return 0
    fi

    # Create safety backup of current data
    log "Creating safety backup of current data..."
    local safety_backup=$(create_backup "pre-restore-safety")

    # Stop Redis
    log "Stopping Redis..."
    docker-compose -f "$COMPOSE_FILE" stop redis

    # Copy backup files
    log "Restoring Redis data..."
    docker cp "$backup_path/dump.rdb" "$REDIS_CONTAINER:/data/dump.rdb"

    if [[ -f "$backup_path/appendonly.aof" ]]; then
        log "Restoring AOF file..."
        docker cp "$backup_path/appendonly.aof" "$REDIS_CONTAINER:/data/appendonly.aof"
    fi

    # Set correct permissions
    docker exec "$REDIS_CONTAINER" chown redis:redis /data/dump.rdb
    if [[ -f "$backup_path/appendonly.aof" ]]; then
        docker exec "$REDIS_CONTAINER" chown redis:redis /data/appendonly.aof
    fi

    # Start Redis
    log "Starting Redis..."
    docker-compose -f "$COMPOSE_FILE" start redis

    # Wait for Redis to be ready
    info "Waiting for Redis to be ready..."
    local max_attempts=30
    local attempt=0

    while [[ $attempt -lt $max_attempts ]]; do
        if docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" ping 2>/dev/null | grep -q PONG; then
            break
        fi
        attempt=$((attempt + 1))
        sleep 1
    done

    if [[ $attempt -eq $max_attempts ]]; then
        error "Redis failed to start after restore"
        warning "Safety backup available at: $safety_backup"
        return 1
    fi

    # Verify restore
    log "Verifying restore..."
    local restored_keys=$(docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" DBSIZE | awk '{print $1}')
    info "Restored keys: $restored_keys"

    # Run health check
    if [[ -f "$SCRIPT_DIR/health-check.py" ]]; then
        log "Running health check..."
        python3 "$SCRIPT_DIR/health-check.py" || warning "Health check reported issues"
    fi

    log "Restore completed successfully"
    info "Safety backup available at: $safety_backup"
}

# List available backups
list_backups() {
    log "Available backups:"
    echo

    if [[ ! -d "$BACKUP_DIR" ]]; then
        warning "No backups directory found"
        return 0
    fi

    # List directories
    local count=0
    for backup in "$BACKUP_DIR"/*; do
        if [[ -d "$backup" ]] && [[ -f "$backup/metadata.json" ]]; then
            count=$((count + 1))
            local name=$(basename "$backup")
            local timestamp=$(jq -r '.timestamp' "$backup/metadata.json" 2>/dev/null || echo "unknown")
            local keys=$(jq -r '.statistics.total_keys' "$backup/metadata.json" 2>/dev/null || echo "unknown")
            local size=$(du -sh "$backup" | cut -f1)

            printf "%-30s | %s | %8s keys | %s\n" "$name" "$timestamp" "$keys" "$size"
        fi
    done

    # List compressed backups
    for backup in "$BACKUP_DIR"/*.tar.gz; do
        if [[ -f "$backup" ]]; then
            count=$((count + 1))
            local name=$(basename "$backup")
            local size=$(ls -lh "$backup" | awk '{print $5}')

            printf "%-30s | (compressed) | %s\n" "$name" "$size"
        fi
    done

    if [[ $count -eq 0 ]]; then
        warning "No backups found"
    else
        echo
        info "Total backups: $count"
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    local keep_count="${BACKUP_RETENTION:-10}"

    log "Cleaning up old backups (keeping last $keep_count)..."

    # Get list of backups sorted by age
    local backups=($(ls -1dt "$BACKUP_DIR"/*/ 2>/dev/null | head -n -$keep_count))

    if [[ ${#backups[@]} -gt 0 ]]; then
        for backup in "${backups[@]}"; do
            if [[ -d "$backup" ]]; then
                warning "Removing old backup: $(basename "$backup")"
                rm -rf "$backup"

                # Also remove compressed version if exists
                local compressed="${backup%.tar.gz}.tar.gz"
                if [[ -f "$compressed" ]]; then
                    rm -f "$compressed"
                fi
            fi
        done
        info "Removed ${#backups[@]} old backups"
    fi
}

# Export specific cache data
export_cache_data() {
    local export_type="$1"
    local output_file="${2:-cache-export-$(date +%Y%m%d-%H%M%S).json}"

    log "Exporting $export_type cache data..."

    if ! check_redis; then
        error "Cannot export - Redis not running"
        return 1
    fi

    case "$export_type" in
        embeddings)
            pattern="codebase_rag:embedding:*"
            ;;
        search)
            pattern="codebase_rag:search:*"
            ;;
        projects)
            pattern="codebase_rag:project:*"
            ;;
        files)
            pattern="codebase_rag:file:*"
            ;;
        all)
            pattern="codebase_rag:*"
            ;;
        *)
            error "Unknown export type: $export_type"
            echo "Valid types: embeddings, search, projects, files, all"
            return 1
            ;;
    esac

    # Export data
    info "Scanning for keys matching: $pattern"
    local keys=$(docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" --scan --pattern "$pattern")
    local count=$(echo "$keys" | wc -l)

    info "Found $count keys to export"

    # Create export file
    echo "{" > "$output_file"
    echo "  \"export_type\": \"$export_type\"," >> "$output_file"
    echo "  \"timestamp\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"," >> "$output_file"
    echo "  \"key_count\": $count," >> "$output_file"
    echo "  \"data\": {" >> "$output_file"

    # Export each key
    local first=true
    while IFS= read -r key; do
        if [[ -n "$key" ]]; then
            if [[ "$first" != "true" ]]; then
                echo "," >> "$output_file"
            fi
            first=false

            # Get value and type
            local type=$(docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" TYPE "$key" | tr -d '\r\n')
            local value=""

            case "$type" in
                string)
                    value=$(docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" GET "$key")
                    ;;
                hash)
                    value=$(docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" HGETALL "$key" | paste -d: - -)
                    ;;
                list)
                    value=$(docker exec "$REDIS_CONTAINER" redis-cli -a "$REDIS_PASSWORD" LRANGE "$key" 0 -1 | tr '\n' ',')
                    ;;
                *)
                    value="<unsupported type: $type>"
                    ;;
            esac

            # Write to file (escape JSON)
            printf '    "%s": %s' "$key" "$(echo "$value" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')" >> "$output_file"
        fi
    done <<< "$keys"

    echo "" >> "$output_file"
    echo "  }" >> "$output_file"
    echo "}" >> "$output_file"

    log "Export completed: $output_file"
    info "Exported $count keys"
}

# Main function
main() {
    case "${1:-help}" in
        backup)
            create_backup "${2:-}"
            ;;
        restore)
            if [[ -z "${2:-}" ]]; then
                error "Backup path required"
                echo "Usage: $0 restore <backup-path>"
                exit 1
            fi
            restore_backup "$2"
            ;;
        list)
            list_backups
            ;;
        export)
            if [[ -z "${2:-}" ]]; then
                error "Export type required"
                echo "Usage: $0 export <type> [output-file]"
                echo "Types: embeddings, search, projects, files, all"
                exit 1
            fi
            export_cache_data "$2" "${3:-}"
            ;;
        cleanup)
            cleanup_old_backups
            ;;
        help|*)
            echo "Cache Backup and Restore Tool"
            echo
            echo "Usage: $0 <command> [options]"
            echo
            echo "Commands:"
            echo "  backup [name]      Create a new backup"
            echo "  restore <path>     Restore from a backup"
            echo "  list              List available backups"
            echo "  export <type>     Export specific cache data"
            echo "  cleanup           Remove old backups"
            echo "  help              Show this help message"
            echo
            echo "Environment Variables:"
            echo "  BACKUP_DIR        Backup directory (default: ./backups)"
            echo "  BACKUP_RETENTION  Number of backups to keep (default: 10)"
            ;;
    esac
}

# Run main function
main "$@"
