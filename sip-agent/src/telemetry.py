"""
OpenTelemetry Instrumentation for SIP AI Assistant
===================================================
Provides distributed tracing, metrics, and logging via OpenTelemetry.

Environment Variables:
    OTEL_ENABLED: Set to 'true' to enable OpenTelemetry (default: false)
    OTEL_EXPORTER_OTLP_ENDPOINT: OTLP endpoint (e.g., http://otel-collector:4317)
    OTEL_SERVICE_NAME: Service name for traces (default: sip-agent)
    OTEL_RESOURCE_ATTRIBUTES: Additional resource attributes
"""

import os
import logging
from contextlib import contextmanager
from typing import Optional, Dict, Any, Callable
from functools import wraps

logger = logging.getLogger(__name__)

# Check if OpenTelemetry is enabled
OTEL_ENABLED = os.environ.get("OTEL_ENABLED", "false").lower() == "true"

# Lazy initialization flags
_initialized = False
_tracer = None
_meter = None

def is_enabled() -> bool:
    """Check if OpenTelemetry is enabled."""
    return OTEL_ENABLED


def init_telemetry(service_name: str = "sip-agent") -> bool:
    """
    Initialize OpenTelemetry instrumentation.
    
    Returns True if initialization was successful, False otherwise.
    """
    global _initialized, _tracer, _meter
    
    if _initialized:
        return True
        
    if not OTEL_ENABLED:
        logger.info("OpenTelemetry disabled (OTEL_ENABLED != 'true')")
        return False
    
    try:
        from opentelemetry import trace, metrics
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
        from opentelemetry.instrumentation.redis import RedisInstrumentor
        from opentelemetry.instrumentation.logging import LoggingInstrumentor
        
        # Build resource attributes
        resource_attrs = {SERVICE_NAME: service_name}
        
        # Parse additional attributes from environment
        extra_attrs = os.environ.get("OTEL_RESOURCE_ATTRIBUTES", "")
        if extra_attrs:
            for attr in extra_attrs.split(","):
                if "=" in attr:
                    key, value = attr.split("=", 1)
                    resource_attrs[key.strip()] = value.strip()
        
        resource = Resource.create(resource_attrs)
        
        # Get OTLP endpoint
        endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
        
        # Initialize Tracer
        tracer_provider = TracerProvider(resource=resource)
        span_exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
        trace.set_tracer_provider(tracer_provider)
        _tracer = trace.get_tracer(__name__)
        
        # Initialize Meter
        metric_exporter = OTLPMetricExporter(endpoint=endpoint, insecure=True)
        metric_reader = PeriodicExportingMetricReader(metric_exporter, export_interval_millis=15000)
        meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
        metrics.set_meter_provider(meter_provider)
        _meter = metrics.get_meter(__name__)
        
        # Auto-instrument libraries
        try:
            HTTPXClientInstrumentor().instrument()
            logger.info("Instrumented HTTPX client")
        except Exception as e:
            logger.debug(f"HTTPX instrumentation skipped: {e}")
            
        try:
            RedisInstrumentor().instrument()
            logger.info("Instrumented Redis client")
        except Exception as e:
            logger.debug(f"Redis instrumentation skipped: {e}")
            
        try:
            LoggingInstrumentor().instrument(set_logging_format=True)
            logger.info("Instrumented logging")
        except Exception as e:
            logger.debug(f"Logging instrumentation skipped: {e}")
        
        _initialized = True
        logger.info(f"OpenTelemetry initialized: service={service_name}, endpoint={endpoint}")
        return True
        
    except ImportError as e:
        logger.warning(f"OpenTelemetry packages not installed: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to initialize OpenTelemetry: {e}")
        return False


def instrument_fastapi(app):
    """Instrument a FastAPI application."""
    if not OTEL_ENABLED:
        return
        
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI application instrumented")
    except Exception as e:
        logger.warning(f"Failed to instrument FastAPI: {e}")


def get_tracer():
    """Get the OpenTelemetry tracer."""
    global _tracer
    if not _initialized:
        init_telemetry()
    return _tracer


def get_meter():
    """Get the OpenTelemetry meter."""
    global _meter
    if not _initialized:
        init_telemetry()
    return _meter


@contextmanager
def create_span(name: str, attributes: Optional[Dict[str, Any]] = None):
    """
    Create a traced span context manager.
    
    Usage:
        with create_span("my_operation", {"key": "value"}) as span:
            # do work
            span.set_attribute("result", "success")
    """
    tracer = get_tracer()
    if tracer is None:
        # No-op context manager when OTEL is disabled
        class NoOpSpan:
            def set_attribute(self, key, value): pass
            def set_status(self, status): pass
            def record_exception(self, exc): pass
            def add_event(self, name, attributes=None): pass
        yield NoOpSpan()
        return
    
    from opentelemetry import trace
    
    with tracer.start_as_current_span(name) as span:
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)
        try:
            yield span
        except Exception as e:
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


def traced(name: Optional[str] = None, attributes: Optional[Dict[str, Any]] = None):
    """
    Decorator to trace a function.
    
    Usage:
        @traced("my_function")
        def my_function():
            pass
            
        @traced(attributes={"component": "audio"})
        async def process_audio():
            pass
    """
    def decorator(func: Callable) -> Callable:
        span_name = name or func.__name__
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with create_span(span_name, attributes) as span:
                return func(*args, **kwargs)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            with create_span(span_name, attributes) as span:
                return await func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


# ===================
# Metrics Helpers
# ===================

class Metrics:
    """Pre-defined metrics for the SIP agent."""
    
    _counters = {}
    _histograms = {}
    _gauges = {}
    
    @classmethod
    def _get_or_create_counter(cls, name: str, description: str, unit: str = "1"):
        if name not in cls._counters:
            meter = get_meter()
            if meter:
                cls._counters[name] = meter.create_counter(
                    name=name,
                    description=description,
                    unit=unit
                )
        return cls._counters.get(name)
    
    @classmethod
    def _get_or_create_histogram(cls, name: str, description: str, unit: str = "ms"):
        if name not in cls._histograms:
            meter = get_meter()
            if meter:
                cls._histograms[name] = meter.create_histogram(
                    name=name,
                    description=description,
                    unit=unit
                )
        return cls._histograms.get(name)
    
    @classmethod
    def _get_or_create_gauge(cls, name: str, description: str, unit: str = "1"):
        if name not in cls._gauges:
            meter = get_meter()
            if meter:
                cls._gauges[name] = meter.create_up_down_counter(
                    name=name,
                    description=description,
                    unit=unit
                )
        return cls._gauges.get(name)
    
    # Call metrics
    @classmethod
    def record_call_started(cls, call_type: str = "inbound"):
        counter = cls._get_or_create_counter(
            "sip.calls.started",
            "Number of calls started"
        )
        if counter:
            counter.add(1, {"call.type": call_type})
    
    @classmethod
    def record_call_ended(cls, call_type: str = "inbound", status: str = "completed"):
        counter = cls._get_or_create_counter(
            "sip.calls.ended",
            "Number of calls ended"
        )
        if counter:
            counter.add(1, {"call.type": call_type, "call.status": status})
    
    @classmethod
    def record_call_duration(cls, duration_ms: float, call_type: str = "inbound"):
        histogram = cls._get_or_create_histogram(
            "sip.calls.duration",
            "Call duration in milliseconds"
        )
        if histogram:
            histogram.record(duration_ms, {"call.type": call_type})
    
    # Audio pipeline metrics
    @classmethod
    def record_tts_latency(cls, latency_ms: float, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.tts.latency",
            "TTS synthesis latency in milliseconds"
        )
        if histogram:
            histogram.record(latency_ms, {"tts.model": model})
    
    @classmethod
    def record_stt_latency(cls, latency_ms: float, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.stt.latency",
            "STT transcription latency in milliseconds"
        )
        if histogram:
            histogram.record(latency_ms, {"stt.model": model})
    
    @classmethod
    def record_llm_latency(cls, latency_ms: float, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.llm.latency",
            "LLM inference latency in milliseconds"
        )
        if histogram:
            histogram.record(latency_ms, {"llm.model": model})
    
    # Queue metrics
    @classmethod
    def record_queue_depth(cls, depth: int):
        gauge = cls._get_or_create_gauge(
            "sip.queue.depth",
            "Number of calls in queue"
        )
        if gauge:
            # Reset and set new value (up_down_counter pattern)
            gauge.add(depth)
    
    @classmethod
    def record_queue_wait_time(cls, wait_time_ms: float):
        histogram = cls._get_or_create_histogram(
            "sip.queue.wait_time",
            "Time spent waiting in queue in milliseconds"
        )
        if histogram:
            histogram.record(wait_time_ms)


# ===================
# Logging Integration
# ===================

def add_trace_context_to_log(record: logging.LogRecord) -> logging.LogRecord:
    """Add trace context to log record for correlation."""
    if not OTEL_ENABLED:
        return record
        
    try:
        from opentelemetry import trace
        
        span = trace.get_current_span()
        if span:
            ctx = span.get_span_context()
            if ctx.is_valid:
                record.trace_id = format(ctx.trace_id, '032x')
                record.span_id = format(ctx.span_id, '016x')
                record.trace_flags = ctx.trace_flags
    except Exception:
        pass
    
    return record


class TraceContextFilter(logging.Filter):
    """Logging filter that adds trace context to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        add_trace_context_to_log(record)
        return True
