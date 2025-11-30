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
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
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
        
        # Initialize Logger Provider for OTLP log export
        try:
            # Try different import paths for OTLPLogExporter (varies by version)
            try:
                from opentelemetry.exporter.otlp.proto.grpc._log_exporter import OTLPLogExporter
            except ImportError:
                try:
                    from opentelemetry.exporter.otlp.proto.grpc.log_exporter import OTLPLogExporter
                except ImportError:
                    raise ImportError("OTLPLogExporter not available in this opentelemetry version")
            
            log_exporter = OTLPLogExporter(endpoint=endpoint, insecure=True)
            logger_provider = LoggerProvider(resource=resource)
            logger_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
            set_logger_provider(logger_provider)
            
            # Attach OTEL handler to root logger to capture all logs
            otel_handler = LoggingHandler(
                level=logging.DEBUG,
                logger_provider=logger_provider
            )
            logging.getLogger().addHandler(otel_handler)
            logger.info("OTLP log export configured")
        except ImportError as e:
            logger.warning(f"OTLP log export not available: {e}")
        except Exception as e:
            logger.warning(f"OTLP log export setup failed: {e}")
        
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
            logger.info("Instrumented logging with trace context")
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
    _observable_gauges = {}
    
    # Track active calls for gauge
    _active_calls = 0
    
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
    
    # ===================
    # Call Quality & Reliability
    # ===================
    
    @classmethod
    def record_call_started(cls, call_type: str = "inbound"):
        counter = cls._get_or_create_counter(
            "sip.calls.started",
            "Number of calls started"
        )
        if counter:
            counter.add(1, {"call.type": call_type})
        cls._active_calls += 1
        cls._update_active_calls()
    
    @classmethod
    def record_call_ended(cls, call_type: str = "inbound", status: str = "completed"):
        counter = cls._get_or_create_counter(
            "sip.calls.ended",
            "Number of calls ended"
        )
        if counter:
            counter.add(1, {"call.type": call_type, "call.status": status})
        cls._active_calls = max(0, cls._active_calls - 1)
        cls._update_active_calls()
    
    @classmethod
    def _update_active_calls(cls):
        gauge = cls._get_or_create_gauge(
            "sip.calls.active",
            "Currently active calls"
        )
        # Note: Using up_down_counter, we track delta. For true gauge, we'd need observable gauge
    
    @classmethod
    def record_call_failed(cls, call_type: str = "inbound", reason: str = "error"):
        counter = cls._get_or_create_counter(
            "sip.calls.failed",
            "Calls that failed"
        )
        if counter:
            counter.add(1, {"call.type": call_type, "failure.reason": reason})
    
    @classmethod
    def record_call_abandoned(cls, call_type: str = "inbound"):
        counter = cls._get_or_create_counter(
            "sip.calls.abandoned",
            "Calls where caller hung up before response"
        )
        if counter:
            counter.add(1, {"call.type": call_type})
    
    @classmethod
    def record_barge_in(cls):
        counter = cls._get_or_create_counter(
            "sip.calls.barge_in",
            "User interruptions during playback"
        )
        if counter:
            counter.add(1)
    
    @classmethod
    def record_silence_timeout(cls):
        counter = cls._get_or_create_counter(
            "sip.calls.silence_timeout",
            "Calls ended due to silence"
        )
        if counter:
            counter.add(1)
    
    @classmethod
    def record_call_duration(cls, duration_ms: float, call_type: str = "inbound"):
        histogram = cls._get_or_create_histogram(
            "sip.calls.duration",
            "Call duration in milliseconds"
        )
        if histogram:
            histogram.record(duration_ms, {"call.type": call_type})
    
    # ===================
    # Conversation Quality
    # ===================
    
    @classmethod
    def record_conversation_turns(cls, turns: int):
        histogram = cls._get_or_create_histogram(
            "sip.conversation.turns",
            "Number of turns per conversation",
            unit="1"
        )
        if histogram:
            histogram.record(turns)
    
    @classmethod
    def record_user_utterance(cls, word_count: int):
        histogram = cls._get_or_create_histogram(
            "sip.conversation.user_words",
            "Words per user utterance",
            unit="1"
        )
        if histogram:
            histogram.record(word_count)
    
    @classmethod
    def record_assistant_response(cls, word_count: int):
        histogram = cls._get_or_create_histogram(
            "sip.conversation.assistant_words",
            "Words per assistant response",
            unit="1"
        )
        if histogram:
            histogram.record(word_count)
    
    @classmethod
    def record_tool_call(cls, tool_name: str):
        counter = cls._get_or_create_counter(
            "sip.tools.calls",
            "Tool/function invocations"
        )
        if counter:
            counter.add(1, {"tool.name": tool_name})
    
    @classmethod
    def record_tool_error(cls, tool_name: str, error_type: str = "unknown"):
        counter = cls._get_or_create_counter(
            "sip.tools.errors",
            "Tool execution failures"
        )
        if counter:
            counter.add(1, {"tool.name": tool_name, "error.type": error_type})
    
    @classmethod
    def record_tool_latency(cls, latency_ms: float, tool_name: str):
        histogram = cls._get_or_create_histogram(
            "sip.tools.latency",
            "Tool execution time in milliseconds"
        )
        if histogram:
            histogram.record(latency_ms, {"tool.name": tool_name})
    
    # ===================
    # Audio Pipeline - STT
    # ===================
    
    @classmethod
    def record_stt_latency(cls, latency_ms: float, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.stt.latency",
            "STT transcription latency in milliseconds"
        )
        if histogram:
            histogram.record(latency_ms, {"stt.model": model})
    
    @classmethod
    def record_stt_confidence(cls, confidence: float, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.stt.confidence",
            "STT confidence scores",
            unit="1"
        )
        if histogram:
            histogram.record(confidence, {"stt.model": model})
    
    @classmethod
    def record_stt_audio_duration(cls, duration_s: float):
        histogram = cls._get_or_create_histogram(
            "sip.stt.audio_duration",
            "Input audio duration in seconds",
            unit="s"
        )
        if histogram:
            histogram.record(duration_s)
    
    @classmethod
    def record_stt_error(cls, model: str = "unknown", error_type: str = "unknown"):
        counter = cls._get_or_create_counter(
            "sip.stt.errors",
            "STT transcription failures"
        )
        if counter:
            counter.add(1, {"stt.model": model, "error.type": error_type})
    
    # ===================
    # Audio Pipeline - TTS
    # ===================
    
    @classmethod
    def record_tts_latency(cls, latency_ms: float, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.tts.latency",
            "TTS synthesis latency in milliseconds"
        )
        if histogram:
            histogram.record(latency_ms, {"tts.model": model})
    
    @classmethod
    def record_tts_characters(cls, char_count: int, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.tts.characters",
            "Characters synthesized",
            unit="1"
        )
        if histogram:
            histogram.record(char_count, {"tts.model": model})
    
    @classmethod
    def record_tts_audio_duration(cls, duration_s: float, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.tts.audio_duration",
            "Output audio duration in seconds",
            unit="s"
        )
        if histogram:
            histogram.record(duration_s, {"tts.model": model})
    
    @classmethod
    def record_tts_error(cls, model: str = "unknown", error_type: str = "unknown"):
        counter = cls._get_or_create_counter(
            "sip.tts.errors",
            "TTS synthesis failures"
        )
        if counter:
            counter.add(1, {"tts.model": model, "error.type": error_type})
    
    # ===================
    # Audio Pipeline - VAD
    # ===================
    
    @classmethod
    def record_vad_speech_segment(cls):
        counter = cls._get_or_create_counter(
            "sip.vad.speech_segments",
            "VAD speech segment detections"
        )
        if counter:
            counter.add(1)
    
    @classmethod
    def record_audio_buffer_size(cls, size_bytes: int):
        gauge = cls._get_or_create_gauge(
            "sip.audio.buffer_size",
            "Audio buffer size in bytes",
            unit="By"
        )
        if gauge:
            gauge.add(size_bytes)
    
    # ===================
    # LLM Performance
    # ===================
    
    @classmethod
    def record_llm_latency(cls, latency_ms: float, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.llm.latency",
            "LLM inference latency in milliseconds"
        )
        if histogram:
            histogram.record(latency_ms, {"llm.model": model})
    
    @classmethod
    def record_llm_tokens_input(cls, tokens: int, model: str = "unknown"):
        counter = cls._get_or_create_counter(
            "sip.llm.tokens_input",
            "Input tokens consumed"
        )
        if counter:
            counter.add(tokens, {"llm.model": model})
    
    @classmethod
    def record_llm_tokens_output(cls, tokens: int, model: str = "unknown"):
        counter = cls._get_or_create_counter(
            "sip.llm.tokens_output",
            "Output tokens generated"
        )
        if counter:
            counter.add(tokens, {"llm.model": model})
    
    @classmethod
    def record_llm_ttft(cls, ttft_ms: float, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.llm.ttft",
            "Time to first token in milliseconds"
        )
        if histogram:
            histogram.record(ttft_ms, {"llm.model": model})
    
    @classmethod
    def record_llm_tokens_per_second(cls, tps: float, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.llm.tokens_per_second",
            "Token generation throughput",
            unit="1"
        )
        if histogram:
            histogram.record(tps, {"llm.model": model})
    
    @classmethod
    def record_llm_context_tokens(cls, tokens: int, model: str = "unknown"):
        histogram = cls._get_or_create_histogram(
            "sip.llm.context_tokens",
            "Context window token usage",
            unit="1"
        )
        if histogram:
            histogram.record(tokens, {"llm.model": model})
    
    @classmethod
    def record_llm_error(cls, model: str = "unknown", error_type: str = "unknown"):
        counter = cls._get_or_create_counter(
            "sip.llm.errors",
            "LLM API failures"
        )
        if counter:
            counter.add(1, {"llm.model": model, "error.type": error_type})
    
    # ===================
    # Queue & Capacity
    # ===================
    
    @classmethod
    def record_queue_depth(cls, depth: int):
        gauge = cls._get_or_create_gauge(
            "sip.queue.depth",
            "Number of calls in queue"
        )
        if gauge:
            gauge.add(depth)
    
    @classmethod
    def record_queue_enqueued(cls):
        counter = cls._get_or_create_counter(
            "sip.queue.enqueued",
            "Calls added to queue"
        )
        if counter:
            counter.add(1)
    
    @classmethod
    def record_queue_rejected(cls, reason: str = "queue_full"):
        counter = cls._get_or_create_counter(
            "sip.queue.rejected",
            "Calls rejected"
        )
        if counter:
            counter.add(1, {"rejection.reason": reason})
    
    @classmethod
    def record_queue_timeout(cls):
        counter = cls._get_or_create_counter(
            "sip.queue.timeout",
            "Calls that timed out in queue"
        )
        if counter:
            counter.add(1)
    
    @classmethod
    def record_queue_wait_time(cls, wait_time_ms: float):
        histogram = cls._get_or_create_histogram(
            "sip.queue.wait_time",
            "Time spent waiting in queue in milliseconds"
        )
        if histogram:
            histogram.record(wait_time_ms)
    
    # ===================
    # Callbacks
    # ===================
    
    @classmethod
    def record_callback_success(cls):
        counter = cls._get_or_create_counter(
            "sip.callback.success",
            "Successful callbacks"
        )
        if counter:
            counter.add(1)
    
    @classmethod
    def record_callback_failed(cls, reason: str = "unknown"):
        counter = cls._get_or_create_counter(
            "sip.callback.failed",
            "Failed callbacks"
        )
        if counter:
            counter.add(1, {"failure.reason": reason})
    
    @classmethod
    def record_callback_retry(cls):
        counter = cls._get_or_create_counter(
            "sip.callback.retries",
            "Callback retry attempts"
        )
        if counter:
            counter.add(1)


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


def get_otel_log_handler():
    """
    Attach the OTEL LoggingHandler to the root logger.
    
    Call this AFTER logging.basicConfig() to ensure the handler isn't removed.
    """
    if not OTEL_ENABLED or not _initialized:
        return False
    
    try:
        from opentelemetry._logs import get_logger_provider
        from opentelemetry.sdk._logs import LoggingHandler
        
        logger_provider = get_logger_provider()
        if logger_provider is None:
            logger.warning("No OTEL logger provider available")
            return False
        
        # # Check if handler already attached
        # root_logger = logging.getLogger()
        # for handler in root_logger.handlers:
        #     if isinstance(handler, LoggingHandler):
        #         return True  # Already attached
        
        # Attach handler
        otel_handler = LoggingHandler(
            level=logging.DEBUG,
            logger_provider=logger_provider
        )
        return otel_handler
        
    except Exception as e:
        logger.warning(f"Failed to attach OTEL log handler: {e}")
        return False