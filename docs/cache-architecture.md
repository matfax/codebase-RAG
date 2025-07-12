# Query Caching Layer Architecture

## Overview

The Query Caching Layer is a comprehensive multi-tier caching system designed to enhance the Codebase RAG MCP Server by implementing intelligent caching across the entire data pipeline. This system addresses critical performance bottlenecks while maintaining data consistency and security.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                   MCP Tool Layer                                │
│  ┌───────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ │
│  │  Search   │ │  Indexing   │ │   Project   │ │    Cache     │ │
│  │   Tools   │ │    Tools    │ │    Tools    │ │ Management   │ │
│  └───────────┘ └─────────────┘ └─────────────┘ └──────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Multi-Tier Cache Service Layer                    │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                L1 Memory Cache                         │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │   │
│  │  │    LRU      │ │   TTL       │ │   Size-based    │   │   │
│  │  │  Eviction   │ │  Expiry     │ │   Eviction      │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                               │                                 │
│                               ▼                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                L2 Redis Cache                          │   │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────┐   │   │
│  │  │ Connection  │ │ Persistence │ │   Distributed   │   │   │
│  │  │   Pooling   │ │   Storage   │ │    Scaling      │   │   │
│  │  └─────────────┘ └─────────────┘ └─────────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│              Specialized Cache Services                        │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐  │
│  │  Embedding  │ │   Search    │ │   Project   │ │   File   │  │
│  │    Cache    │ │   Cache     │ │    Cache    │ │  Cache   │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                Core Service Layer                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌──────────┐  │
│  │ Embedding   │ │  Qdrant     │ │   Code      │ │ Project  │  │
│  │  Service    │ │  Service    │ │   Parser    │ │ Analysis │  │
│  └─────────────┘ └─────────────┘ └─────────────┘ └──────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Multi-Tier Cache Architecture

#### L1 Memory Cache (Fast Access)
- **Implementation**: LRU (Least Recently Used) in-memory cache
- **Purpose**: Ultra-fast access for frequently used data
- **Capacity**: Configurable (default: 1000 entries, 256MB)
- **TTL**: Configurable per cache type (default: 1 hour)
- **Eviction**: LRU policy with size and memory-based eviction

#### L2 Redis Cache (Persistent Storage)
- **Implementation**: Redis-based distributed cache
- **Purpose**: Persistent storage and cross-session caching
- **Features**: Connection pooling, SSL support, clustering ready
- **Persistence**: Configurable RDB/AOF persistence
- **Scalability**: Horizontal scaling support

### 2. Cache Write Strategies

#### Write-Through (Default)
- Writes to both L1 and L2 simultaneously
- Ensures data consistency
- Higher write latency but guaranteed consistency

#### Write-Back (Performance Optimized)
- Writes to L1 immediately, L2 asynchronously
- Lower write latency
- Periodic flush to L2 with dirty tracking

#### Write-Around (Cache Bypass)
- Writes directly to L2, bypasses L1
- Used for large data that won't be re-read soon

### 3. Specialized Cache Services

#### Embedding Cache Service
```python
# Cache Structure
{
    "key": "embedding:sha256:{content_hash}:{model_version}",
    "value": {
        "embedding": [0.1, 0.2, ...],
        "model": "nomic-embed-text",
        "version": "1.0",
        "timestamp": "2024-01-01T00:00:00Z"
    },
    "ttl": 7200  # 2 hours
}
```

**Features:**
- Content-based key generation using SHA-256
- Model version awareness for cache invalidation
- Compression for storage efficiency
- Batch operations for bulk embedding

#### Search Cache Service
```python
# Cache Structure
{
    "key": "search:{query_hash}:{params_hash}:{project_hash}",
    "value": {
        "results": [...],
        "total_count": 42,
        "search_params": {...},
        "timestamp": "2024-01-01T00:00:00Z"
    },
    "ttl": 1800  # 30 minutes
}
```

**Features:**
- Composite key generation for search parameters
- Result ranking preservation
- Contextual search result caching
- Automatic invalidation on content changes

#### Project Cache Service
```python
# Cache Structure
{
    "key": "project:{project_name}:{operation}",
    "value": {
        "project_info": {...},
        "collections": [...],
        "file_stats": {...},
        "last_modified": "2024-01-01T00:00:00Z"
    },
    "ttl": 3600  # 1 hour
}
```

**Features:**
- Project metadata caching
- Collection mapping and statistics
- File filtering result caching
- Project-wide invalidation

#### File Cache Service
```python
# Cache Structure
{
    "key": "file:{file_path_hash}:{modification_time}",
    "value": {
        "parsed_content": {...},
        "chunks": [...],
        "metadata": {...},
        "syntax_tree": {...}
    },
    "ttl": 1800  # 30 minutes
}
```

**Features:**
- Tree-sitter parsing result caching
- Intelligent chunking result storage
- Incremental parsing for changed files
- Language-specific optimization

## Cache Key Management

### Hierarchical Key Structure
```
{prefix}:{cache_type}:{identifier}:{version}:{context}
```

Examples:
- `codebase_rag:embedding:sha256:abc123:v1.0`
- `codebase_rag:search:query456:params789:project123`
- `codebase_rag:project:myproject:info:v2`

### Key Generation Strategy
- **Content-based**: SHA-256 hashing for deterministic keys
- **Hierarchical**: Namespace separation for cache organization
- **Versioning**: Version-aware keys for cache invalidation
- **Collision detection**: Automatic key collision resolution

## Cache Invalidation System

### Automatic Invalidation Triggers
1. **File System Changes**: File modification/deletion
2. **Project Updates**: Project structure changes
3. **Model Updates**: Embedding model version changes
4. **Manual Operations**: User-triggered cache clearing

### Invalidation Strategies
- **Selective Invalidation**: Target specific cache entries
- **Cascade Invalidation**: Propagate invalidation to dependent caches
- **Partial Invalidation**: Invalidate subsets of cached data
- **Time-based Expiry**: TTL-based automatic expiration

## Security and Encryption

### Data Protection
- **AES-256 Encryption**: Optional encryption for sensitive data
- **Key Management**: Secure key generation and rotation
- **Project Isolation**: Strict separation between projects
- **Access Control**: User session-based cache isolation

### Security Features
- **Encrypted Storage**: Sensitive cache data encryption
- **Secure Transmission**: SSL/TLS for Redis connections
- **Audit Logging**: Cache access and operation logging
- **Cross-project Isolation**: Prevent data leakage

## Performance Optimization

### Memory Management
- **Adaptive Sizing**: Dynamic cache size adjustment
- **Memory Pressure Handling**: Intelligent eviction under pressure
- **Garbage Collection**: Coordinated cache cleanup
- **Memory Profiling**: Continuous memory usage monitoring

### Network Optimization
- **Connection Pooling**: Efficient Redis connection management
- **Batch Operations**: Bulk cache operations
- **Compression**: Data compression for network efficiency
- **Pipelining**: Redis pipeline operations

### Cache Warming
- **Predictive Loading**: Preload frequently accessed data
- **Background Processing**: Asynchronous cache population
- **Pattern-based Warming**: Load based on usage patterns

## Monitoring and Metrics

### Performance Metrics
- **Hit/Miss Rates**: Cache effectiveness measurement
- **Response Times**: Cache operation latency
- **Memory Usage**: Cache memory consumption
- **Error Rates**: Cache operation failures

### Health Monitoring
- **Redis Connectivity**: Connection health checks
- **Memory Pressure**: Memory usage alerts
- **Performance Degradation**: Automatic performance monitoring
- **Alert System**: Configurable threshold alerts

### Telemetry Integration
- **OpenTelemetry**: Distributed tracing support
- **Metrics Export**: Prometheus-compatible metrics
- **Dashboard Integration**: Real-time monitoring dashboards

## Configuration Management

### Environment-based Configuration
```python
# Core Settings
CACHE_ENABLED=true
CACHE_LEVEL=BOTH  # L1_MEMORY, L2_REDIS, BOTH
CACHE_WRITE_STRATEGY=WRITE_THROUGH

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_password
REDIS_MAX_CONNECTIONS=10

# Memory Cache Configuration
MEMORY_CACHE_MAX_SIZE=1000
MEMORY_CACHE_TTL=3600
MEMORY_CACHE_MAX_MEMORY_MB=256

# Security Settings
CACHE_ENCRYPTION_ENABLED=false
CACHE_ENCRYPTION_KEY=your_key
```

### Cache Type Specific Configuration
- **Embedding Cache**: Extended TTL for stability
- **Search Cache**: Short TTL for freshness
- **Project Cache**: Medium TTL for balance
- **File Cache**: Context-dependent TTL

## Error Handling and Resilience

### Graceful Degradation
- **Cache Failure Handling**: Continue operation without cache
- **Fallback Strategies**: Alternative data sources
- **Circuit Breaker**: Prevent cascade failures
- **Retry Logic**: Exponential backoff for transient failures

### Self-Healing Mechanisms
- **Automatic Recovery**: Self-healing cache corruption
- **Health Monitoring**: Continuous health assessment
- **Failover Support**: Automatic failover mechanisms
- **Backup and Restore**: Cache data protection

## Integration Points

### Service Integration
- **EmbeddingService**: Transparent cache integration
- **QdrantService**: Vector database caching
- **CodeParserService**: Parsing result caching
- **ProjectAnalysisService**: Metadata caching

### MCP Tool Integration
- **Search Tools**: Cached search results
- **Indexing Tools**: Cached processing results
- **Project Tools**: Cached project information
- **Cache Management Tools**: Administrative operations

## Best Practices

### Development Guidelines
- **Async Operations**: Non-blocking cache operations
- **Error Handling**: Comprehensive error management
- **Testing**: Extensive test coverage
- **Documentation**: Clear API documentation

### Operational Guidelines
- **Monitoring**: Continuous performance monitoring
- **Maintenance**: Regular cache maintenance
- **Backup**: Regular cache backup procedures
- **Security**: Regular security assessments

## Future Enhancements

### Planned Features
- **Advanced Eviction Policies**: LFU, Time-aware LRU
- **Distributed Caching**: Multi-node cache clusters
- **ML-based Prediction**: Intelligent cache warming
- **Advanced Compression**: Context-aware compression

### Scalability Roadmap
- **Horizontal Scaling**: Multi-Redis deployment
- **Cache Sharding**: Distributed cache partitioning
- **Load Balancing**: Intelligent load distribution
- **Global Cache**: Cross-region cache synchronization
