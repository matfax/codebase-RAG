# Cache Deployment and Setup Guide

## Overview

This comprehensive guide covers the deployment and setup of the Query Caching Layer system across different environments, from local development to production clusters. Follow these step-by-step instructions to deploy a robust and scalable cache system.

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB total (2GB for Redis, 2GB for application)
- **Storage**: 10GB SSD
- **Network**: 100 Mbps

#### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 16GB total (8GB for Redis, 8GB for application)
- **Storage**: 50GB+ SSD with IOPS > 3000
- **Network**: 1 Gbps

#### Production Requirements
- **CPU**: 8+ cores
- **RAM**: 32GB+ total (16GB+ for Redis, 16GB+ for application)
- **Storage**: 200GB+ NVMe SSD
- **Network**: 10 Gbps
- **High Availability**: Multi-node setup

### Software Dependencies

```bash
# Core dependencies
Docker >= 20.10
Docker Compose >= 2.0
Python >= 3.9
uv (Python package manager)

# Optional monitoring tools
Prometheus
Grafana
Redis Insight
```

### Network Requirements

```bash
# Required ports
6379  # Redis (default)
6380  # Redis SSL (optional)
8000  # Application API (default)
5000  # Metrics endpoint (optional)
3000  # Grafana (optional)
9090  # Prometheus (optional)

# Firewall configuration
sudo ufw allow 6379/tcp
sudo ufw allow 6380/tcp
sudo ufw allow 8000/tcp
```

## Local Development Setup

### Quick Start

```bash
# 1. Clone repository
git clone <repository-url>
cd query-caching-layer

# 2. Install Python dependencies
uv sync

# 3. Create environment configuration
cp .env.example .env.development

# 4. Start Redis with Docker Compose
docker-compose -f docker-compose.cache.yml up -d

# 5. Verify installation
python -c "
import asyncio
from src.services.cache_service import get_cache_service

async def test():
    cache = await get_cache_service()
    await cache.set('test', 'hello')
    result = await cache.get('test')
    print(f'Cache test: {result}')

asyncio.run(test())
"
```

### Development Environment Configuration

```bash
# .env.development
ENVIRONMENT=development

# Cache settings
CACHE_ENABLED=true
CACHE_LEVEL=BOTH
CACHE_DEBUG_MODE=true
CACHE_LOG_LEVEL=DEBUG

# Redis settings
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=dev_password
REDIS_MAX_CONNECTIONS=10

# Memory cache settings
MEMORY_CACHE_MAX_SIZE=1000
MEMORY_CACHE_MAX_MEMORY_MB=256

# Security (relaxed for development)
CACHE_ENCRYPTION_ENABLED=false
CACHE_PROJECT_ISOLATION=false

# Monitoring
CACHE_METRICS_ENABLED=true
CACHE_HEALTH_CHECK_INTERVAL=60
```

### Docker Compose for Development

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  redis-dev:
    image: redis:7-alpine
    container_name: redis-dev
    ports:
      - "6379:6379"
    volumes:
      - redis_dev_data:/data
      - ./configs/redis-dev.conf:/usr/local/etc/redis/redis.conf
    command: redis-server /usr/local/etc/redis/redis.conf --requirepass dev_password
    networks:
      - dev_network
    healthcheck:
      test: ["CMD", "redis-cli", "auth", "dev_password", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis-insight:
    image: redisinsight/redisinsight:latest
    container_name: redis-insight-dev
    ports:
      - "8001:8001"
    volumes:
      - redis_insight_data:/db
    depends_on:
      - redis-dev
    networks:
      - dev_network

networks:
  dev_network:
    driver: bridge

volumes:
  redis_dev_data:
  redis_insight_data:
```

### Development Redis Configuration

```conf
# configs/redis-dev.conf
port 6379
bind 0.0.0.0
protected-mode yes

# Memory settings for development
maxmemory 512mb
maxmemory-policy allkeys-lru

# Persistence (disabled for development)
save ""
appendonly no

# Logging
loglevel debug
logfile ""

# Performance
tcp-keepalive 300
timeout 0
databases 16
```

## Testing Environment Setup

### Test Environment Configuration

```bash
# .env.testing
ENVIRONMENT=testing

# Cache settings for testing
CACHE_ENABLED=true
CACHE_LEVEL=L1_MEMORY  # Memory only for tests
CACHE_DEBUG_MODE=true
CACHE_LOG_LEVEL=DEBUG

# No Redis for unit tests
REDIS_HOST=localhost
REDIS_PORT=6380  # Different port for test Redis

# Small memory footprint
MEMORY_CACHE_MAX_SIZE=100
MEMORY_CACHE_MAX_MEMORY_MB=64

# Short TTLs for testing
CACHE_DEFAULT_TTL=60
EMBEDDING_CACHE_TTL=120

# Security testing
CACHE_ENCRYPTION_ENABLED=false
CACHE_PROJECT_ISOLATION=true
```

### Test Docker Setup

```yaml
# docker-compose.test.yml
version: '3.8'

services:
  redis-test:
    image: redis:7-alpine
    container_name: redis-test
    ports:
      - "6380:6379"
    volumes:
      - redis_test_data:/data
    command: redis-server --save "" --appendonly no --maxmemory 256mb
    networks:
      - test_network
    tmpfs:
      - /data:noexec,nosuid,size=256m

  test-runner:
    build:
      context: .
      dockerfile: Dockerfile.test
    container_name: cache-test-runner
    environment:
      - REDIS_HOST=redis-test
      - REDIS_PORT=6379
      - CACHE_LEVEL=BOTH
    depends_on:
      - redis-test
    networks:
      - test_network
    volumes:
      - .:/app
      - test_cache:/tmp/cache

networks:
  test_network:
    driver: bridge

volumes:
  redis_test_data:
  test_cache:
```

### Automated Testing Setup

```bash
#!/bin/bash
# scripts/setup-testing.sh

set -e

echo "Setting up testing environment..."

# Start test services
docker-compose -f docker-compose.test.yml up -d redis-test

# Wait for Redis to be ready
echo "Waiting for Redis to be ready..."
timeout 30 bash -c 'until docker exec redis-test redis-cli ping > /dev/null 2>&1; do sleep 1; done'

# Run tests
echo "Running cache tests..."
uv run pytest tests/ -v --tb=short

# Integration tests
echo "Running integration tests..."
uv run pytest tests/test_cache_integration.py -v

# Performance tests
echo "Running performance tests..."
uv run python run_performance_tests.py

# Cleanup
echo "Cleaning up..."
docker-compose -f docker-compose.test.yml down -v

echo "Testing setup complete!"
```

## Staging Environment Setup

### Staging Infrastructure

```yaml
# docker-compose.staging.yml
version: '3.8'

services:
  redis-staging:
    image: redis:7-alpine
    container_name: redis-staging
    ports:
      - "6379:6379"
    volumes:
      - redis_staging_data:/data
      - ./configs/redis-staging.conf:/usr/local/etc/redis/redis.conf
      - ./certs:/tls:ro
    command: redis-server /usr/local/etc/redis/redis.conf
    environment:
      - REDIS_PASSWORD_FILE=/run/secrets/redis_password
    secrets:
      - redis_password
    networks:
      - staging_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "auth", "$$(cat /run/secrets/redis_password)", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
        reservations:
          memory: 2G
          cpus: '1'

  cache-app:
    build:
      context: .
      dockerfile: Dockerfile.staging
    container_name: cache-app-staging
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=staging
      - REDIS_HOST=redis-staging
    env_file:
      - .env.staging
    depends_on:
      redis-staging:
        condition: service_healthy
    networks:
      - staging_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  monitoring:
    image: prom/prometheus:latest
    container_name: prometheus-staging
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - staging_network
    restart: unless-stopped

secrets:
  redis_password:
    file: ./secrets/redis_password.txt

networks:
  staging_network:
    driver: bridge

volumes:
  redis_staging_data:
  prometheus_data:
```

### Staging Configuration

```bash
# .env.staging
ENVIRONMENT=staging

# Cache settings
CACHE_ENABLED=true
CACHE_LEVEL=BOTH
CACHE_WRITE_STRATEGY=WRITE_THROUGH
CACHE_DEBUG_MODE=false
CACHE_LOG_LEVEL=INFO

# Redis settings
REDIS_HOST=redis-staging
REDIS_PORT=6379
REDIS_PASSWORD_FILE=/run/secrets/redis_password
REDIS_MAX_CONNECTIONS=25
REDIS_SSL_ENABLED=false

# Memory settings
MEMORY_CACHE_MAX_SIZE=5000
MEMORY_CACHE_MAX_MEMORY_MB=1024

# TTL settings
CACHE_DEFAULT_TTL=1800
EMBEDDING_CACHE_TTL=3600
SEARCH_CACHE_TTL=900

# Security
CACHE_ENCRYPTION_ENABLED=true
CACHE_ENCRYPTION_KEY_FILE=/run/secrets/encryption_key
CACHE_PROJECT_ISOLATION=true
CACHE_ACCESS_LOGGING=true

# Monitoring
CACHE_METRICS_ENABLED=true
CACHE_HEALTH_CHECK_INTERVAL=30
CACHE_PERFORMANCE_MONITORING=true
```

### Staging Redis Configuration

```conf
# configs/redis-staging.conf
port 6379
bind 0.0.0.0
protected-mode yes

# Authentication
requirepass $(cat /run/secrets/redis_password)

# Memory management
maxmemory 3gb
maxmemory-policy allkeys-lru
maxmemory-samples 5

# Persistence
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data

# Performance
tcp-keepalive 300
timeout 0
tcp-nodelay yes
databases 16

# Logging
loglevel notice
logfile /data/redis.log

# Security
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""
```

## Production Deployment

### Production Architecture

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  redis-primary:
    image: redis:7-alpine
    container_name: redis-primary
    ports:
      - "6380:6380"  # SSL port
    volumes:
      - redis_primary_data:/data
      - ./configs/redis-prod.conf:/usr/local/etc/redis/redis.conf:ro
      - ./certs:/tls:ro
      - ./secrets:/run/secrets:ro
    command: redis-server /usr/local/etc/redis/redis.conf
    networks:
      - prod_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "--tls", "--cert", "/tls/redis-client.crt", "--key", "/tls/redis-client.key", "--cacert", "/tls/ca.crt", "-p", "6380", "ping"]
      interval: 30s
      timeout: 10s
      retries: 5
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'
        reservations:
          memory: 8G
          cpus: '4'
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

  redis-replica:
    image: redis:7-alpine
    container_name: redis-replica
    ports:
      - "6381:6380"
    volumes:
      - redis_replica_data:/data
      - ./configs/redis-replica.conf:/usr/local/etc/redis/redis.conf:ro
      - ./certs:/tls:ro
      - ./secrets:/run/secrets:ro
    command: redis-server /usr/local/etc/redis/redis.conf
    depends_on:
      - redis-primary
    networks:
      - prod_network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 16G
          cpus: '8'
        reservations:
          memory: 8G
          cpus: '4'

  haproxy:
    image: haproxy:2.8-alpine
    container_name: redis-loadbalancer
    ports:
      - "6379:6379"  # Load balancer port
    volumes:
      - ./configs/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg:ro
    depends_on:
      - redis-primary
      - redis-replica
    networks:
      - prod_network
    restart: unless-stopped

  cache-app-1:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: cache-app-1
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT=production
      - INSTANCE_ID=1
      - REDIS_HOST=haproxy
    env_file:
      - .env.production
    depends_on:
      - redis-primary
      - haproxy
    networks:
      - prod_network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
        reservations:
          memory: 4G
          cpus: '2'

  cache-app-2:
    build:
      context: .
      dockerfile: Dockerfile.prod
    container_name: cache-app-2
    ports:
      - "8001:8000"
    environment:
      - ENVIRONMENT=production
      - INSTANCE_ID=2
      - REDIS_HOST=haproxy
    env_file:
      - .env.production
    depends_on:
      - redis-primary
      - haproxy
    networks:
      - prod_network
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'

  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus-prod
    ports:
      - "9090:9090"
    volumes:
      - ./configs/prometheus-prod.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--web.enable-lifecycle'
    networks:
      - prod_network
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    container_name: grafana-prod
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD_FILE=/run/secrets/grafana_password
    volumes:
      - grafana_data:/var/lib/grafana
      - ./configs/grafana:/etc/grafana/provisioning
      - ./secrets:/run/secrets:ro
    depends_on:
      - prometheus
    networks:
      - prod_network
    restart: unless-stopped

networks:
  prod_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16

volumes:
  redis_primary_data:
  redis_replica_data:
  prometheus_data:
  grafana_data:
```

### Production Redis Configuration

```conf
# configs/redis-prod.conf
# Network and Security
port 0
tls-port 6380
bind 0.0.0.0
protected-mode yes

# TLS Configuration
tls-cert-file /tls/redis-server.crt
tls-key-file /tls/redis-server.key
tls-ca-cert-file /tls/ca.crt
tls-dh-params-file /tls/redis.dh
tls-protocols "TLSv1.2 TLSv1.3"
tls-ciphers "ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-AES128-GCM-SHA256"
tls-ciphersuites "TLS_AES_256_GCM_SHA384:TLS_AES_128_GCM_SHA256"

# Authentication
requirepass $(cat /run/secrets/redis_password)

# Memory Management
maxmemory 14gb
maxmemory-policy allkeys-lru
maxmemory-samples 10

# Persistence
save 900 1
save 300 10
save 60 10000
rdbcompression yes
rdbchecksum yes
dbfilename dump.rdb
dir /data
appendonly yes
appendfilename "appendonly.aof"
appendfsync everysec
no-appendfsync-on-rewrite no
auto-aof-rewrite-percentage 100
auto-aof-rewrite-min-size 64mb

# Performance
tcp-keepalive 300
timeout 0
tcp-nodelay yes
databases 16
lua-time-limit 5000

# Logging
loglevel notice
logfile /data/redis.log
syslog-enabled yes
syslog-ident redis

# Security
rename-command FLUSHDB ""
rename-command FLUSHALL ""
rename-command DEBUG ""
rename-command SHUTDOWN SHUTDOWN_PROD
rename-command CONFIG CONFIG_PROD

# Client Management
maxclients 10000
```

### HAProxy Configuration for Redis

```conf
# configs/haproxy.cfg
global
    daemon
    log stdout local0
    maxconn 4096

defaults
    mode tcp
    timeout connect 5s
    timeout client 30s
    timeout server 30s
    option tcplog
    log global

frontend redis_frontend
    bind *:6379
    default_backend redis_backend

backend redis_backend
    balance roundrobin
    option tcp-check
    tcp-check send AUTH\ $(cat\ /run/secrets/redis_password)\r\n
    tcp-check expect string +OK
    tcp-check send PING\r\n
    tcp-check expect string +PONG

    server redis-primary redis-primary:6380 check port 6380 ssl verify none weight 100
    server redis-replica redis-replica:6380 check port 6380 ssl verify none weight 50 backup
```

### SSL Certificate Generation

```bash
#!/bin/bash
# scripts/generate-ssl-certs.sh

set -e

CERT_DIR="./certs"
VALIDITY_DAYS=365

echo "Generating SSL certificates for Redis..."

# Create certificate directory
mkdir -p $CERT_DIR

# Generate CA private key
openssl genrsa -out $CERT_DIR/ca.key 4096

# Generate CA certificate
openssl req -new -x509 -days $VALIDITY_DAYS -key $CERT_DIR/ca.key \
    -out $CERT_DIR/ca.crt \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=Redis-CA"

# Generate server private key
openssl genrsa -out $CERT_DIR/redis-server.key 2048

# Generate server certificate signing request
openssl req -new -key $CERT_DIR/redis-server.key \
    -out $CERT_DIR/redis-server.csr \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=redis-server"

# Generate server certificate
openssl x509 -req -days $VALIDITY_DAYS \
    -in $CERT_DIR/redis-server.csr \
    -CA $CERT_DIR/ca.crt \
    -CAkey $CERT_DIR/ca.key \
    -CAcreateserial \
    -out $CERT_DIR/redis-server.crt

# Generate client private key
openssl genrsa -out $CERT_DIR/redis-client.key 2048

# Generate client certificate signing request
openssl req -new -key $CERT_DIR/redis-client.key \
    -out $CERT_DIR/redis-client.csr \
    -subj "/C=US/ST=State/L=City/O=Organization/CN=redis-client"

# Generate client certificate
openssl x509 -req -days $VALIDITY_DAYS \
    -in $CERT_DIR/redis-client.csr \
    -CA $CERT_DIR/ca.crt \
    -CAkey $CERT_DIR/ca.key \
    -CAcreateserial \
    -out $CERT_DIR/redis-client.crt

# Generate DH parameters
openssl dhparam -out $CERT_DIR/redis.dh 2048

# Set proper permissions
chmod 600 $CERT_DIR/*.key
chmod 644 $CERT_DIR/*.crt
chmod 644 $CERT_DIR/*.dh

echo "SSL certificates generated successfully!"
echo "Certificates location: $CERT_DIR"
```

## Kubernetes Deployment

### Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: cache-system
  labels:
    name: cache-system

---
# k8s/redis-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: redis-config
  namespace: cache-system
data:
  redis.conf: |
    port 6379
    bind 0.0.0.0
    protected-mode yes
    maxmemory 8gb
    maxmemory-policy allkeys-lru
    save 900 1
    save 300 10
    save 60 10000
    appendonly yes
    tcp-keepalive 300
    timeout 0

---
# k8s/redis-secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: redis-secret
  namespace: cache-system
type: Opaque
data:
  password: <base64-encoded-password>

---
# k8s/redis-statefulset.yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-primary
  namespace: cache-system
spec:
  serviceName: redis-primary-service
  replicas: 1
  selector:
    matchLabels:
      app: redis-primary
  template:
    metadata:
      labels:
        app: redis-primary
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        ports:
        - containerPort: 6379
          name: redis
        command:
        - redis-server
        - /etc/redis/redis.conf
        - --requirepass
        - $(REDIS_PASSWORD)
        env:
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: password
        volumeMounts:
        - name: redis-config
          mountPath: /etc/redis
        - name: redis-data
          mountPath: /data
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
        livenessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - redis-cli
            - ping
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: redis-config
        configMap:
          name: redis-config
  volumeClaimTemplates:
  - metadata:
      name: redis-data
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 100Gi
      storageClassName: fast-ssd

---
# k8s/redis-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: redis-service
  namespace: cache-system
spec:
  selector:
    app: redis-primary
  ports:
  - port: 6379
    targetPort: 6379
    name: redis
  type: ClusterIP

---
# k8s/cache-app-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cache-app
  namespace: cache-system
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cache-app
  template:
    metadata:
      labels:
        app: cache-app
    spec:
      containers:
      - name: cache-app
        image: cache-app:latest
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: REDIS_HOST
          value: "redis-service"
        - name: REDIS_PORT
          value: "6379"
        - name: REDIS_PASSWORD
          valueFrom:
            secretKeyRef:
              name: redis-secret
              key: password
        envFrom:
        - configMapRef:
            name: cache-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
# k8s/cache-app-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: cache-app-service
  namespace: cache-system
spec:
  selector:
    app: cache-app
  ports:
  - port: 80
    targetPort: 8000
    name: http
  type: LoadBalancer
```

### Helm Chart Structure

```
charts/cache-system/
├── Chart.yaml
├── values.yaml
├── values-dev.yaml
├── values-staging.yaml
├── values-prod.yaml
└── templates/
    ├── namespace.yaml
    ├── configmap.yaml
    ├── secret.yaml
    ├── redis-statefulset.yaml
    ├── redis-service.yaml
    ├── app-deployment.yaml
    ├── app-service.yaml
    ├── ingress.yaml
    ├── hpa.yaml
    └── monitoring/
        ├── servicemonitor.yaml
        └── prometheusrule.yaml
```

```yaml
# charts/cache-system/Chart.yaml
apiVersion: v2
name: cache-system
description: Query Caching Layer Helm Chart
type: application
version: 1.0.0
appVersion: "1.0.0"
keywords:
  - cache
  - redis
  - performance
home: https://github.com/your-org/cache-system
sources:
  - https://github.com/your-org/cache-system
maintainers:
  - name: Cache Team
    email: cache-team@yourorg.com
dependencies:
  - name: redis
    version: "17.3.7"
    repository: "https://charts.bitnami.com/bitnami"
    condition: redis.enabled
```

```yaml
# charts/cache-system/values.yaml
global:
  imageRegistry: ""
  imagePullSecrets: []

replicaCount: 3

image:
  repository: cache-app
  pullPolicy: IfNotPresent
  tag: "latest"

nameOverride: ""
fullnameOverride: ""

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: false
  className: ""
  annotations: {}
  hosts:
    - host: cache.local
      paths:
        - path: /
          pathType: Prefix
  tls: []

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

redis:
  enabled: true
  auth:
    enabled: true
    password: "your-redis-password"
  master:
    persistence:
      enabled: true
      size: 100Gi
      storageClass: "fast-ssd"
    resources:
      limits:
        memory: 8Gi
        cpu: 4
      requests:
        memory: 4Gi
        cpu: 1

cache:
  config:
    enabled: true
    level: "BOTH"
    writeStrategy: "WRITE_THROUGH"
    defaultTTL: 3600
    encryptionEnabled: true
    projectIsolation: true

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
  prometheusRule:
    enabled: true
```

## Deployment Automation

### Deployment Scripts

```bash
#!/bin/bash
# scripts/deploy.sh

set -e

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}

echo "Deploying cache system to $ENVIRONMENT environment..."

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(development|staging|production)$ ]]; then
    echo "Error: Invalid environment. Use development, staging, or production."
    exit 1
fi

# Set environment-specific variables
case $ENVIRONMENT in
    development)
        COMPOSE_FILE="docker-compose.dev.yml"
        ENV_FILE=".env.development"
        ;;
    staging)
        COMPOSE_FILE="docker-compose.staging.yml"
        ENV_FILE=".env.staging"
        ;;
    production)
        COMPOSE_FILE="docker-compose.prod.yml"
        ENV_FILE=".env.production"
        ;;
esac

# Pre-deployment validation
echo "Running pre-deployment validation..."
./scripts/validate-config.sh $ENV_FILE

# Build images if needed
if [[ "$ENVIRONMENT" != "development" ]]; then
    echo "Building application image..."
    docker build -t cache-app:$VERSION -f Dockerfile.$ENVIRONMENT .
fi

# Generate SSL certificates for production
if [[ "$ENVIRONMENT" == "production" ]]; then
    echo "Generating SSL certificates..."
    ./scripts/generate-ssl-certs.sh
fi

# Deploy services
echo "Deploying services..."
docker-compose -f $COMPOSE_FILE --env-file $ENV_FILE up -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
./scripts/wait-for-services.sh $ENVIRONMENT

# Run health checks
echo "Running health checks..."
./scripts/health-check.sh $ENVIRONMENT

# Run smoke tests
echo "Running smoke tests..."
./scripts/smoke-tests.sh $ENVIRONMENT

echo "Deployment to $ENVIRONMENT completed successfully!"
```

### Health Check Script

```bash
#!/bin/bash
# scripts/health-check.sh

ENVIRONMENT=${1:-staging}

echo "Running health checks for $ENVIRONMENT environment..."

# Set environment-specific endpoints
case $ENVIRONMENT in
    development)
        REDIS_HOST="localhost"
        REDIS_PORT="6379"
        APP_URL="http://localhost:8000"
        ;;
    staging)
        REDIS_HOST="localhost"
        REDIS_PORT="6379"
        APP_URL="http://localhost:8000"
        ;;
    production)
        REDIS_HOST="localhost"
        REDIS_PORT="6379"
        APP_URL="http://localhost:8000"
        ;;
esac

# Check Redis connectivity
echo "Checking Redis connectivity..."
if docker exec redis-$ENVIRONMENT redis-cli ping > /dev/null 2>&1; then
    echo "✓ Redis is responding"
else
    echo "✗ Redis is not responding"
    exit 1
fi

# Check application health
echo "Checking application health..."
if curl -f $APP_URL/health > /dev/null 2>&1; then
    echo "✓ Application is healthy"
else
    echo "✗ Application health check failed"
    exit 1
fi

# Check cache functionality
echo "Testing cache functionality..."
if curl -f -X POST $APP_URL/test/cache > /dev/null 2>&1; then
    echo "✓ Cache functionality working"
else
    echo "✗ Cache functionality test failed"
    exit 1
fi

echo "All health checks passed!"
```

### Monitoring Setup Script

```bash
#!/bin/bash
# scripts/setup-monitoring.sh

ENVIRONMENT=${1:-staging}

echo "Setting up monitoring for $ENVIRONMENT environment..."

# Create monitoring directory
mkdir -p monitoring/dashboards
mkdir -p monitoring/alerts

# Deploy Prometheus configuration
cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts/*.yml"

scrape_configs:
  - job_name: 'cache-app'
    static_configs:
      - targets: ['cache-app:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

# Deploy Grafana dashboards
curl -X POST \
  http://admin:admin@localhost:3000/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/dashboards/cache-dashboard.json

echo "Monitoring setup completed!"
```

## Troubleshooting Deployment Issues

### Common Deployment Problems

#### Docker Issues

```bash
# Check Docker daemon
sudo systemctl status docker

# Check Docker Compose version
docker-compose --version

# Check container logs
docker-compose logs redis-cache
docker-compose logs cache-app

# Check container resource usage
docker stats

# Clean up failed deployments
docker-compose down -v
docker system prune -f
```

#### Network Connectivity

```bash
# Test Redis connectivity
redis-cli -h localhost -p 6379 ping

# Test application endpoints
curl -v http://localhost:8000/health

# Check port availability
netstat -tuln | grep 6379

# Test SSL connections
openssl s_client -connect localhost:6380 -tls1_2
```

#### Performance Issues

```bash
# Monitor Redis performance
redis-cli --latency-history -h localhost -p 6379

# Check memory usage
redis-cli info memory

# Monitor cache hit rates
curl http://localhost:8000/metrics | grep cache_hit_rate

# Check application metrics
curl http://localhost:8000/stats/cache
```

### Recovery Procedures

#### Data Recovery

```bash
# Backup Redis data
docker exec redis-cache redis-cli --rdb /data/backup.rdb

# Restore Redis data
docker cp backup.rdb redis-cache:/data/dump.rdb
docker restart redis-cache
```

#### Configuration Rollback

```bash
# Rollback to previous version
git checkout HEAD~1 -- docker-compose.prod.yml
docker-compose -f docker-compose.prod.yml up -d

# Rollback environment configuration
cp .env.production.backup .env.production
```

## Maintenance Procedures

### Regular Maintenance Tasks

```bash
#!/bin/bash
# scripts/maintenance.sh

echo "Running cache system maintenance..."

# Update system packages
sudo apt update && sudo apt upgrade -y

# Clean up Docker resources
docker system prune -f

# Rotate Redis logs
docker exec redis-cache logrotate /etc/logrotate.d/redis

# Update SSL certificates (if needed)
./scripts/renew-certificates.sh

# Backup Redis data
./scripts/backup-redis.sh

# Check and optimize Redis memory
./scripts/optimize-redis-memory.sh

echo "Maintenance completed!"
```

### Backup Procedures

```bash
#!/bin/bash
# scripts/backup-redis.sh

BACKUP_DIR="/var/backups/redis"
DATE=$(date +%Y%m%d_%H%M%S)

mkdir -p $BACKUP_DIR

# Create Redis backup
docker exec redis-cache redis-cli BGSAVE
docker exec redis-cache redis-cli LASTSAVE

# Copy backup file
docker cp redis-cache:/data/dump.rdb $BACKUP_DIR/redis_backup_$DATE.rdb

# Compress backup
gzip $BACKUP_DIR/redis_backup_$DATE.rdb

# Clean old backups (keep last 7 days)
find $BACKUP_DIR -name "*.rdb.gz" -mtime +7 -delete

echo "Redis backup completed: $BACKUP_DIR/redis_backup_$DATE.rdb.gz"
```

This deployment guide provides comprehensive coverage of setting up the cache system across all environments with proper security, monitoring, and maintenance procedures.
