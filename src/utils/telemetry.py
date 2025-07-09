"""
OpenTelemetry integration for distributed tracing and observability.

This module provides comprehensive telemetry support for cache operations,
including distributed tracing, metrics, and observability platform integration.
"""

import functools
import logging
import os
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional, Union

try:
    # OpenTelemetry imports
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.zipkin.json import ZipkinExporter
    from opentelemetry.instrumentation.redis import RedisInstrumentor
    from opentelemetry.instrumentation.requests import RequestsInstrumentor
    from opentelemetry.instrumentation.urllib3 import URLLib3Instrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import ConsoleMetricExporter, PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.util.types import AttributeValue

    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    # Fallback types
    trace = None
    metrics = None
    AttributeValue = str | int | float | bool


logger = logging.getLogger(__name__)


@dataclass
class TelemetryConfig:
    """Configuration for OpenTelemetry integration."""

    # Service identification
    service_name: str = "codebase-rag-mcp"
    service_version: str = "1.0.0"
    service_namespace: str = "mcp"

    # Tracing configuration
    tracing_enabled: bool = True
    trace_exporter: str = "console"  # "console", "jaeger", "zipkin", "otlp"
    jaeger_endpoint: str = "http://localhost:14268/api/traces"
    zipkin_endpoint: str = "http://localhost:9411/api/v2/spans"
    otlp_endpoint: str = "http://localhost:4317"

    # Metrics configuration
    metrics_enabled: bool = True
    metric_exporter: str = "console"  # "console", "otlp"
    metrics_interval: int = 30  # seconds

    # Instrumentation configuration
    auto_instrument_requests: bool = True
    auto_instrument_redis: bool = True

    # Sampling configuration
    trace_sample_rate: float = 1.0  # 100% sampling by default

    # Resource attributes
    deployment_environment: str = "development"
    resource_attributes: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_environment(cls) -> "TelemetryConfig":
        """Create configuration from environment variables."""
        return cls(
            service_name=os.getenv("OTEL_SERVICE_NAME", "codebase-rag-mcp"),
            service_version=os.getenv("OTEL_SERVICE_VERSION", "1.0.0"),
            service_namespace=os.getenv("OTEL_SERVICE_NAMESPACE", "mcp"),
            tracing_enabled=os.getenv("OTEL_TRACING_ENABLED", "true").lower() == "true",
            trace_exporter=os.getenv("OTEL_TRACE_EXPORTER", "console"),
            jaeger_endpoint=os.getenv("OTEL_JAEGER_ENDPOINT", "http://localhost:14268/api/traces"),
            zipkin_endpoint=os.getenv("OTEL_ZIPKIN_ENDPOINT", "http://localhost:9411/api/v2/spans"),
            otlp_endpoint=os.getenv("OTEL_OTLP_ENDPOINT", "http://localhost:4317"),
            metrics_enabled=os.getenv("OTEL_METRICS_ENABLED", "true").lower() == "true",
            metric_exporter=os.getenv("OTEL_METRIC_EXPORTER", "console"),
            metrics_interval=int(os.getenv("OTEL_METRICS_INTERVAL", "30")),
            auto_instrument_requests=os.getenv("OTEL_AUTO_INSTRUMENT_REQUESTS", "true").lower() == "true",
            auto_instrument_redis=os.getenv("OTEL_AUTO_INSTRUMENT_REDIS", "true").lower() == "true",
            trace_sample_rate=float(os.getenv("OTEL_TRACE_SAMPLE_RATE", "1.0")),
            deployment_environment=os.getenv("OTEL_DEPLOYMENT_ENVIRONMENT", "development"),
        )


class TelemetryManager:
    """Manages OpenTelemetry integration for the cache system."""

    def __init__(self, config: TelemetryConfig | None = None):
        self.config = config or TelemetryConfig.from_environment()
        self.tracer = None
        self.meter = None
        self.initialized = False
        self.telemetry_enabled = OPENTELEMETRY_AVAILABLE and (self.config.tracing_enabled or self.config.metrics_enabled)

        # Cache-specific metrics
        self.cache_operation_counter = None
        self.cache_hit_counter = None
        self.cache_miss_counter = None
        self.cache_error_counter = None
        self.cache_response_time_histogram = None
        self.cache_size_gauge = None
        self.cache_eviction_counter = None

        if self.telemetry_enabled:
            self.initialize()

    def initialize(self) -> None:
        """Initialize OpenTelemetry components."""
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("OpenTelemetry is not available. Telemetry will be disabled.")
            self.telemetry_enabled = False
            return

        try:
            # Create resource
            resource = Resource.create(
                {
                    ResourceAttributes.SERVICE_NAME: self.config.service_name,
                    ResourceAttributes.SERVICE_VERSION: self.config.service_version,
                    ResourceAttributes.SERVICE_NAMESPACE: self.config.service_namespace,
                    ResourceAttributes.DEPLOYMENT_ENVIRONMENT: self.config.deployment_environment,
                    **self.config.resource_attributes,
                }
            )

            # Initialize tracing
            if self.config.tracing_enabled:
                self._initialize_tracing(resource)

            # Initialize metrics
            if self.config.metrics_enabled:
                self._initialize_metrics(resource)

            # Auto-instrument libraries
            self._setup_auto_instrumentation()

            self.initialized = True
            logger.info(f"OpenTelemetry initialized successfully for service: {self.config.service_name}")

        except Exception as e:
            logger.error(f"Failed to initialize OpenTelemetry: {e}")
            self.telemetry_enabled = False

    def _initialize_tracing(self, resource: Resource) -> None:
        """Initialize tracing components."""
        tracer_provider = TracerProvider(resource=resource)

        # Configure span exporter
        if self.config.trace_exporter == "console":
            span_exporter = ConsoleSpanExporter()
        elif self.config.trace_exporter == "jaeger":
            span_exporter = JaegerExporter(
                collector_endpoint=self.config.jaeger_endpoint,
            )
        elif self.config.trace_exporter == "zipkin":
            span_exporter = ZipkinExporter(
                endpoint=self.config.zipkin_endpoint,
            )
        elif self.config.trace_exporter == "otlp":
            span_exporter = OTLPSpanExporter(
                endpoint=self.config.otlp_endpoint,
            )
        else:
            logger.warning(f"Unknown trace exporter: {self.config.trace_exporter}, using console")
            span_exporter = ConsoleSpanExporter()

        # Add span processor
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))

        # Set tracer provider
        trace.set_tracer_provider(tracer_provider)
        self.tracer = trace.get_tracer(__name__)

        logger.info(f"Tracing initialized with {self.config.trace_exporter} exporter")

    def _initialize_metrics(self, resource: Resource) -> None:
        """Initialize metrics components."""
        # Configure metric exporter
        if self.config.metric_exporter == "console":
            metric_exporter = ConsoleMetricExporter()
        elif self.config.metric_exporter == "otlp":
            metric_exporter = OTLPMetricExporter(
                endpoint=self.config.otlp_endpoint,
            )
        else:
            logger.warning(f"Unknown metric exporter: {self.config.metric_exporter}, using console")
            metric_exporter = ConsoleMetricExporter()

        # Create metric reader
        metric_reader = PeriodicExportingMetricReader(
            exporter=metric_exporter,
            export_interval_millis=self.config.metrics_interval * 1000,
        )

        # Set meter provider
        meter_provider = MeterProvider(
            resource=resource,
            metric_readers=[metric_reader],
        )
        metrics.set_meter_provider(meter_provider)

        # Get meter
        self.meter = metrics.get_meter(__name__)

        # Create cache-specific metrics
        self._create_cache_metrics()

        logger.info(f"Metrics initialized with {self.config.metric_exporter} exporter")

    def _create_cache_metrics(self) -> None:
        """Create cache-specific metrics."""
        if not self.meter:
            return

        # Counter for cache operations
        self.cache_operation_counter = self.meter.create_counter(
            name="cache_operations_total",
            description="Total number of cache operations",
            unit="1",
        )

        # Counter for cache hits
        self.cache_hit_counter = self.meter.create_counter(
            name="cache_hits_total",
            description="Total number of cache hits",
            unit="1",
        )

        # Counter for cache misses
        self.cache_miss_counter = self.meter.create_counter(
            name="cache_misses_total",
            description="Total number of cache misses",
            unit="1",
        )

        # Counter for cache errors
        self.cache_error_counter = self.meter.create_counter(
            name="cache_errors_total",
            description="Total number of cache errors",
            unit="1",
        )

        # Histogram for response times
        self.cache_response_time_histogram = self.meter.create_histogram(
            name="cache_response_time_seconds",
            description="Cache operation response time",
            unit="s",
        )

        # Gauge for cache size
        self.cache_size_gauge = self.meter.create_up_down_counter(
            name="cache_size_bytes",
            description="Current cache size in bytes",
            unit="By",
        )

        # Counter for cache evictions
        self.cache_eviction_counter = self.meter.create_counter(
            name="cache_evictions_total",
            description="Total number of cache evictions",
            unit="1",
        )

    def _setup_auto_instrumentation(self) -> None:
        """Setup automatic instrumentation for common libraries."""
        try:
            if self.config.auto_instrument_requests:
                RequestsInstrumentor().instrument()
                URLLib3Instrumentor().instrument()
                logger.debug("HTTP requests instrumentation enabled")

            if self.config.auto_instrument_redis:
                RedisInstrumentor().instrument()
                logger.debug("Redis instrumentation enabled")

        except Exception as e:
            logger.warning(f"Failed to setup auto-instrumentation: {e}")

    def create_span(
        self,
        name: str,
        attributes: dict[str, AttributeValue] | None = None,
        kind: trace.SpanKind | None = None,
    ) -> Union[trace.Span, "NoOpSpan"]:
        """Create a new span."""
        if not self.telemetry_enabled or not self.tracer:
            return NoOpSpan()

        span = self.tracer.start_span(
            name=name,
            attributes=attributes or {},
            kind=kind or trace.SpanKind.INTERNAL,
        )
        return span

    def record_cache_operation(
        self,
        operation: str,
        cache_name: str,
        hit: bool | None = None,
        error: bool | None = None,
        response_time: float | None = None,
        cache_size: int | None = None,
        additional_attributes: dict[str, AttributeValue] | None = None,
    ) -> None:
        """Record cache operation metrics."""
        if not self.telemetry_enabled:
            return

        attributes = {
            "cache.operation": operation,
            "cache.name": cache_name,
            **(additional_attributes or {}),
        }

        # Record operation counter
        if self.cache_operation_counter:
            self.cache_operation_counter.add(1, attributes)

        # Record hit/miss
        if hit is not None:
            if hit and self.cache_hit_counter:
                self.cache_hit_counter.add(1, attributes)
            elif not hit and self.cache_miss_counter:
                self.cache_miss_counter.add(1, attributes)

        # Record error
        if error and self.cache_error_counter:
            self.cache_error_counter.add(1, attributes)

        # Record response time
        if response_time is not None and self.cache_response_time_histogram:
            self.cache_response_time_histogram.record(response_time, attributes)

        # Record cache size
        if cache_size is not None and self.cache_size_gauge:
            self.cache_size_gauge.add(cache_size, attributes)

    def record_cache_eviction(
        self,
        cache_name: str,
        eviction_reason: str,
        items_evicted: int = 1,
        additional_attributes: dict[str, AttributeValue] | None = None,
    ) -> None:
        """Record cache eviction metrics."""
        if not self.telemetry_enabled or not self.cache_eviction_counter:
            return

        attributes = {
            "cache.name": cache_name,
            "cache.eviction.reason": eviction_reason,
            **(additional_attributes or {}),
        }

        self.cache_eviction_counter.add(items_evicted, attributes)

    def is_enabled(self) -> bool:
        """Check if telemetry is enabled."""
        return self.telemetry_enabled and self.initialized


class NoOpSpan:
    """No-op span implementation when telemetry is disabled."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        pass

    def set_status(self, status, description: str = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def add_event(self, name: str, attributes: dict[str, AttributeValue] = None) -> None:
        pass

    def end(self) -> None:
        pass


# Global telemetry manager
_telemetry_manager: TelemetryManager | None = None


def get_telemetry_manager() -> TelemetryManager:
    """Get the global telemetry manager instance."""
    global _telemetry_manager
    if _telemetry_manager is None:
        _telemetry_manager = TelemetryManager()
    return _telemetry_manager


def initialize_telemetry(config: TelemetryConfig | None = None) -> TelemetryManager:
    """Initialize the global telemetry manager."""
    global _telemetry_manager
    _telemetry_manager = TelemetryManager(config)
    return _telemetry_manager


@contextmanager
def trace_cache_operation(
    operation: str,
    cache_name: str,
    cache_key: str | None = None,
    additional_attributes: dict[str, AttributeValue] | None = None,
):
    """Context manager for tracing cache operations."""
    telemetry = get_telemetry_manager()

    attributes = {
        "cache.operation": operation,
        "cache.name": cache_name,
        **(additional_attributes or {}),
    }

    if cache_key:
        attributes["cache.key"] = cache_key

    span = telemetry.create_span(
        name=f"cache.{operation}",
        attributes=attributes,
        kind=trace.SpanKind.INTERNAL if OPENTELEMETRY_AVAILABLE else None,
    )

    start_time = time.time()
    hit = None
    error = None

    try:
        with span:
            yield span
    except Exception as e:
        error = True
        if hasattr(span, "record_exception"):
            span.record_exception(e)
        if hasattr(span, "set_status"):
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        raise
    finally:
        response_time = time.time() - start_time

        # Record metrics
        telemetry.record_cache_operation(
            operation=operation,
            cache_name=cache_name,
            hit=hit,
            error=error,
            response_time=response_time,
            additional_attributes=additional_attributes,
        )


def trace_cache_method(
    operation: str | None = None,
    cache_name_attr: str = "cache_name",
    cache_key_attr: str | None = None,
):
    """Decorator for tracing cache methods."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Determine operation name
            op_name = operation or func.__name__

            # Get cache name from instance or arguments
            cache_name = "unknown"
            if args and hasattr(args[0], cache_name_attr):
                cache_name = getattr(args[0], cache_name_attr)
            elif cache_name_attr in kwargs:
                cache_name = kwargs[cache_name_attr]

            # Get cache key if specified
            cache_key = None
            if cache_key_attr:
                if args and hasattr(args[0], cache_key_attr):
                    cache_key = getattr(args[0], cache_key_attr)
                elif cache_key_attr in kwargs:
                    cache_key = kwargs[cache_key_attr]

            # Additional attributes
            additional_attributes = {
                "code.function": func.__name__,
                "code.namespace": func.__module__,
            }

            with trace_cache_operation(
                operation=op_name,
                cache_name=cache_name,
                cache_key=cache_key,
                additional_attributes=additional_attributes,
            ) as span:
                try:
                    result = func(*args, **kwargs)

                    # Try to determine if it was a hit/miss from result
                    if hasattr(result, "hit"):
                        hit = result.hit
                    elif isinstance(result, tuple) and len(result) == 2:
                        # Common pattern: (value, hit)
                        _, hit = result
                    else:
                        hit = result is not None

                    # Update span attributes
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("cache.hit", hit)

                    return result

                except Exception:
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("cache.error", True)
                    raise

        return wrapper

    return decorator


async def trace_async_cache_method(
    operation: str | None = None,
    cache_name_attr: str = "cache_name",
    cache_key_attr: str | None = None,
):
    """Decorator for tracing async cache methods."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Determine operation name
            op_name = operation or func.__name__

            # Get cache name from instance or arguments
            cache_name = "unknown"
            if args and hasattr(args[0], cache_name_attr):
                cache_name = getattr(args[0], cache_name_attr)
            elif cache_name_attr in kwargs:
                cache_name = kwargs[cache_name_attr]

            # Get cache key if specified
            cache_key = None
            if cache_key_attr:
                if args and hasattr(args[0], cache_key_attr):
                    cache_key = getattr(args[0], cache_key_attr)
                elif cache_key_attr in kwargs:
                    cache_key = kwargs[cache_key_attr]

            # Additional attributes
            additional_attributes = {
                "code.function": func.__name__,
                "code.namespace": func.__module__,
            }

            with trace_cache_operation(
                operation=op_name,
                cache_name=cache_name,
                cache_key=cache_key,
                additional_attributes=additional_attributes,
            ) as span:
                try:
                    result = await func(*args, **kwargs)

                    # Try to determine if it was a hit/miss from result
                    if hasattr(result, "hit"):
                        hit = result.hit
                    elif isinstance(result, tuple) and len(result) == 2:
                        # Common pattern: (value, hit)
                        _, hit = result
                    else:
                        hit = result is not None

                    # Update span attributes
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("cache.hit", hit)

                    return result

                except Exception:
                    if hasattr(span, "set_attribute"):
                        span.set_attribute("cache.error", True)
                    raise

        return wrapper

    return decorator


def get_telemetry_status() -> dict[str, Any]:
    """Get current telemetry status and configuration."""
    telemetry = get_telemetry_manager()

    return {
        "telemetry_enabled": telemetry.is_enabled(),
        "opentelemetry_available": OPENTELEMETRY_AVAILABLE,
        "initialized": telemetry.initialized,
        "configuration": {
            "service_name": telemetry.config.service_name,
            "service_version": telemetry.config.service_version,
            "service_namespace": telemetry.config.service_namespace,
            "tracing_enabled": telemetry.config.tracing_enabled,
            "trace_exporter": telemetry.config.trace_exporter,
            "metrics_enabled": telemetry.config.metrics_enabled,
            "metric_exporter": telemetry.config.metric_exporter,
            "metrics_interval": telemetry.config.metrics_interval,
            "deployment_environment": telemetry.config.deployment_environment,
            "trace_sample_rate": telemetry.config.trace_sample_rate,
        },
        "instrumentation": {
            "auto_instrument_requests": telemetry.config.auto_instrument_requests,
            "auto_instrument_redis": telemetry.config.auto_instrument_redis,
        },
    }


def configure_telemetry_from_dict(config_dict: dict[str, Any]) -> TelemetryManager:
    """Configure telemetry from a dictionary."""
    config = TelemetryConfig(**config_dict)
    return initialize_telemetry(config)


# Example usage in cache services
class TracedCacheService:
    """Example of how to integrate telemetry with a cache service."""

    def __init__(self, cache_name: str):
        self.cache_name = cache_name
        self.telemetry = get_telemetry_manager()

    @trace_cache_method(operation="get", cache_name_attr="cache_name")
    def get(self, key: str) -> Any | None:
        """Get value from cache with tracing."""
        # Simulate cache operation
        return None

    @trace_cache_method(operation="set", cache_name_attr="cache_name")
    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with tracing."""
        # Simulate cache operation
        return True

    def clear(self) -> None:
        """Clear cache with manual tracing."""
        with trace_cache_operation(
            operation="clear",
            cache_name=self.cache_name,
            additional_attributes={"cache.operation.type": "bulk"},
        ):
            # Simulate cache clear
            pass

    def evict(self, reason: str = "size_limit") -> int:
        """Evict cache entries with telemetry."""
        items_evicted = 5  # Simulate eviction

        # Record eviction metrics
        self.telemetry.record_cache_eviction(
            cache_name=self.cache_name,
            eviction_reason=reason,
            items_evicted=items_evicted,
        )

        return items_evicted
