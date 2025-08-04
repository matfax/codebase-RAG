# MCP Codebase RAG Server - Logging Guide

## Overview

The MCP Codebase RAG Server uses a comprehensive file-based logging system designed for effective debugging, monitoring, and troubleshooting of multi-modal search operations and service interactions.

## Log Files Structure

All logs are written to the `logs/` directory by default. The following log files are automatically created:

### Core Log Files

1. **`mcp_server.log`** - Main application log (10MB, 5 backups)
   - General application flow and important events
   - Service initialization and configuration
   - Tool execution summaries

2. **`mcp_debug.log`** - Detailed debug information (20MB, 3 backups)
   - Only created when `MCP_DEBUG_LEVEL=DEBUG`
   - Verbose debugging information for all components
   - Internal state changes and detailed execution traces

3. **`mcp_errors.log`** - Error tracking (5MB, 10 backups)
   - All ERROR and CRITICAL level messages
   - Exception traces and failure scenarios
   - Service failure recovery attempts

### Specialized Log Files

4. **`mcp_performance.log`** - Performance monitoring (15MB, 7 backups)
   - Response times and throughput metrics
   - Resource utilization tracking
   - Performance bottleneck identification

5. **`mcp_multimodal.log`** - Multi-modal search debugging (10MB, 3 backups)
   - Mode selection decisions and reasoning
   - Search fallback scenarios
   - Query analysis and mode execution details

6. **`mcp_service_calls.log`** - Service interaction tracking (5MB, 5 backups)
   - Inter-service communication
   - API call timing and success rates
   - Service dependency mapping

7. **`mcp_configuration.log`** - Configuration decisions (5MB, 3 backups)
   - Auto-configuration outcomes
   - Feature toggle decisions
   - Runtime configuration changes

## Environment Configuration

Configure logging behavior using environment variables:

```bash
# Basic Configuration
export MCP_DEBUG_LEVEL=INFO          # DEBUG, INFO, WARNING, ERROR
export MCP_LOG_DIR=logs              # Custom log directory
export MCP_CONSOLE_LOGGING=false     # Also log to console
export MCP_ENABLE_REQUEST_TRACKING=false  # Track request chains
```

## Debugging Scenarios

### 1. Multi-Modal Search Issues

When debugging why multi-modal search modes return identical results:

```bash
# Enable detailed multi-modal debugging
export MCP_DEBUG_LEVEL=DEBUG
export MCP_ENABLE_REQUEST_TRACKING=true

# Check these log files:
tail -f logs/mcp_multimodal.log
tail -f logs/mcp_service_calls.log
```

**Key log markers to look for:**
- `[MULTI_MODAL_START]` - Search initiation
- `[MODAL_DECISION]` - Mode selection logic
- `[MODAL_FALLBACK]` - Fallback triggers
- `[SERVICE_CALL]` - Service interaction points

### 2. Performance Issues

For analyzing response time problems:

```bash
# Enable performance tracking
export MCP_DEBUG_LEVEL=DEBUG

# Monitor performance logs
tail -f logs/mcp_performance.log
tail -f logs/mcp_server.log | grep -E "(ms|seconds|timeout)"
```

**Performance indicators:**
- Function execution times
- Graph building phases
- Service call latencies
- Memory usage spikes

### 3. Service Communication Problems

For debugging MCP tool chains and service interactions:

```bash
# Enable request tracking
export MCP_ENABLE_REQUEST_TRACKING=true
export MCP_DEBUG_LEVEL=DEBUG

# Follow service call chains
tail -f logs/mcp_service_calls.log
```

**Service call patterns:**
- `service_a → service_b.method(params)` - Outbound calls
- `service_a ← service_b.method ✓ (45.2ms)` - Successful responses
- `service_a ← service_b.method ✗ (timeout)` - Failed calls

## Log Analysis Examples

### Finding Multi-Modal Fallback Issues

```bash
# Search for fallback scenarios
grep "MODAL_FALLBACK" logs/mcp_multimodal.log

# Check service initialization failures
grep "SERVICE_INIT.*error" logs/mcp_service_calls.log

# Look for timeout patterns
grep -E "(timeout|TimeoutError)" logs/mcp_*.log
```

### Performance Bottleneck Analysis

```bash
# Find slowest operations
grep -E "([0-9]+\.[0-9]+ms)" logs/mcp_performance.log | sort -k3 -n

# Identify memory pressure
grep -i "memory" logs/mcp_performance.log

# Track graph building performance
grep "graph.*phase" logs/mcp_debug.log
```

### Request Tracing

When `MCP_ENABLE_REQUEST_TRACKING=true`, each request gets a unique ID:

```bash
# Follow a specific request chain
grep "REQ-a1b2c3d4" logs/mcp_*.log

# Find all requests for a specific tool
grep "trace_function_chain" logs/mcp_*.log | grep "REQ-"
```

## Log Rotation and Maintenance

### Automatic Rotation
- Logs automatically rotate when they reach their size limits
- Backup files are numbered (e.g., `mcp_server.log.1`, `mcp_server.log.2`)
- Old backups are automatically deleted when backup count is exceeded

### Manual Cleanup
```bash
# Remove old log files
find logs/ -name "*.log.*" -mtime +7 -delete

# Archive current logs
tar -czf logs_backup_$(date +%Y%m%d).tar.gz logs/
```

## Integration with Existing Code

### Using the Centralized Logger

```python
from src.config.logging_config import get_search_logger, get_multimodal_logger

# In search tools
logger = get_search_logger()
logger.info("Starting search operation")

# In multi-modal components
mm_logger = get_multimodal_logger()
mm_logger.debug("Mode selection: local → global", mode="hybrid")
```

### Enhanced Debug Logging

```python
from src.tools.core.debug_logger import (
    with_request_tracking, 
    multi_modal_logger, 
    service_call_logger
)

@with_request_tracking
async def my_mcp_tool(query: str):
    # Request tracking automatically added
    multi_modal_logger.log_mode_decision(query, "local", "hybrid", "auto_selected")
    
    service_call_logger.log_call_start("search_tools", "qdrant_service", "search", {})
    # ... service call ...
    service_call_logger.log_call_complete("search_tools", "qdrant_service", "search", True, 125.0)
```

## Log File Locations by Component

| Component | Primary Log | Debug Log | Performance |
|-----------|-------------|-----------|-------------|
| Search Tools | `mcp_server.log` | `mcp_debug.log` | `mcp_performance.log` |
| Multi-Modal Search | `mcp_multimodal.log` | `mcp_debug.log` | `mcp_performance.log` |
| Graph RAG | `mcp_server.log` | `mcp_debug.log` | `mcp_performance.log` |
| Service Calls | `mcp_service_calls.log` | `mcp_debug.log` | - |
| Configuration | `mcp_configuration.log` | `mcp_debug.log` | - |
| Errors | `mcp_errors.log` | - | - |

## Troubleshooting Common Issues

### Issue: No Log Files Created
```bash
# Check permissions
ls -la logs/
chmod 755 logs/

# Check environment
echo $MCP_LOG_DIR
```

### Issue: Logs Not Detailed Enough
```bash
# Enable debug mode
export MCP_DEBUG_LEVEL=DEBUG

# Restart MCP server
./mcp_server
```

### Issue: Too Many Log Files
```bash
# Reduce backup counts in logging_config.py
# Or clean up manually
find logs/ -name "*.log.*" -delete
```

## Best Practices

1. **Always check multiple log files** - Issues often span multiple components
2. **Use request tracking** - Essential for debugging complex MCP tool chains
3. **Monitor performance logs** - Proactively identify bottlenecks
4. **Set appropriate debug levels** - DEBUG for troubleshooting, INFO for monitoring
5. **Regular log cleanup** - Prevent disk space issues
6. **Correlate timestamps** - Use timestamps to understand event sequences

## Advanced Usage

### Custom Log Analysis Scripts

```bash
#!/bin/bash
# analyze_multimodal_performance.sh

echo "Multi-Modal Search Performance Analysis"
echo "======================================="

echo "Fallback Rate:"
grep -c "MODAL_FALLBACK" logs/mcp_multimodal.log

echo "Average Response Time:"
grep -E "([0-9]+\.[0-9]+ms)" logs/mcp_multimodal.log | \
    awk '{print $NF}' | sed 's/ms//' | \
    awk '{sum+=$1; count++} END {print sum/count "ms"}'

echo "Most Common Fallback Reasons:"
grep "MODAL_FALLBACK" logs/mcp_multimodal.log | \
    cut -d'"' -f2 | sort | uniq -c | sort -nr
```

This logging system provides comprehensive visibility into the MCP Codebase RAG Server's operation, enabling effective debugging of complex multi-modal search scenarios and performance optimization.