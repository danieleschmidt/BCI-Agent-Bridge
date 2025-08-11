"""
Distributed tracing support with trace correlation for BCI system monitoring.
Provides end-to-end request tracking across neural processing, decoding, and AI integration.
"""

import time
import asyncio
import uuid
import json
import logging
import threading
import weakref
from typing import Dict, Any, List, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import contextvars
from contextlib import asynccontextmanager, contextmanager
import inspect


class SpanKind(Enum):
    """Types of spans in distributed tracing."""
    SERVER = "server"          # Incoming request
    CLIENT = "client"          # Outgoing request
    PRODUCER = "producer"      # Message/event producer
    CONSUMER = "consumer"      # Message/event consumer
    INTERNAL = "internal"      # Internal operation


class SpanStatus(Enum):
    """Status of span execution."""
    OK = "ok"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class SpanEvent:
    """Event within a span."""
    name: str
    timestamp: float
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceContext:
    """Context information for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    trace_flags: int = 0
    baggage: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "trace_flags": self.trace_flags,
            "baggage": self.baggage
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TraceContext':
        """Create from dictionary."""
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
            trace_flags=data.get("trace_flags", 0),
            baggage=data.get("baggage", {})
        )
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation."""
        headers = {
            "X-Trace-Id": self.trace_id,
            "X-Span-Id": self.span_id,
            "X-Trace-Flags": str(self.trace_flags)
        }
        
        if self.parent_span_id:
            headers["X-Parent-Span-Id"] = self.parent_span_id
        
        if self.baggage:
            headers["X-Trace-Baggage"] = json.dumps(self.baggage)
        
        return headers
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional['TraceContext']:
        """Create from HTTP headers."""
        trace_id = headers.get("X-Trace-Id")
        span_id = headers.get("X-Span-Id")
        
        if not trace_id or not span_id:
            return None
        
        baggage = {}
        baggage_header = headers.get("X-Trace-Baggage")
        if baggage_header:
            try:
                baggage = json.loads(baggage_header)
            except json.JSONDecodeError:
                pass
        
        return cls(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=headers.get("X-Parent-Span-Id"),
            trace_flags=int(headers.get("X-Trace-Flags", "0")),
            baggage=baggage
        )


@dataclass
class Span:
    """Individual span in distributed trace."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: SpanStatus = SpanStatus.OK
    tags: Dict[str, Any] = field(default_factory=dict)
    events: List[SpanEvent] = field(default_factory=list)
    correlations: Set[str] = field(default_factory=set)
    
    # BCI-specific fields
    neural_session_id: Optional[str] = None
    patient_id: Optional[str] = None  # Anonymized patient identifier
    device_id: Optional[str] = None
    paradigm: Optional[str] = None  # P300, SSVEP, Motor Imagery, etc.
    
    def __post_init__(self):
        """Post-initialization processing."""
        if not self.end_time and self.duration_ms:
            self.end_time = self.start_time + (self.duration_ms / 1000.0)
        elif self.end_time and not self.duration_ms:
            self.duration_ms = (self.end_time - self.start_time) * 1000.0

    def add_event(self, name: str, attributes: Dict[str, Any] = None) -> None:
        """Add an event to this span."""
        event = SpanEvent(
            name=name,
            timestamp=time.time(),
            attributes=attributes or {}
        )
        self.events.append(event)

    def set_tag(self, key: str, value: Any) -> None:
        """Set a tag on this span."""
        self.tags[key] = value

    def set_status(self, status: SpanStatus, error_message: str = None) -> None:
        """Set the span status."""
        self.status = status
        if error_message:
            self.set_tag("error.message", error_message)

    def add_correlation(self, correlation_id: str) -> None:
        """Add a correlation ID to this span."""
        self.correlations.add(correlation_id)

    def finish(self, status: SpanStatus = SpanStatus.OK) -> None:
        """Finish this span."""
        if not self.end_time:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000.0
        
        self.status = status

    def to_dict(self) -> Dict[str, Any]:
        """Convert span to dictionary for export."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "operation_name": self.operation_name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "tags": self.tags,
            "events": [
                {
                    "name": event.name,
                    "timestamp": event.timestamp,
                    "attributes": event.attributes
                } for event in self.events
            ],
            "correlations": list(self.correlations),
            "neural_session_id": self.neural_session_id,
            "patient_id": self.patient_id,
            "device_id": self.device_id,
            "paradigm": self.paradigm
        }


# Context variables for trace propagation
_current_trace_context: contextvars.ContextVar[Optional[TraceContext]] = contextvars.ContextVar(
    'current_trace_context', default=None
)
_current_span: contextvars.ContextVar[Optional[Span]] = contextvars.ContextVar(
    'current_span', default=None
)


class DistributedTracer:
    """
    Main distributed tracing system for BCI applications.
    """
    
    def __init__(self, service_name: str = "bci-agent-bridge",
                 sampling_rate: float = 1.0,
                 max_spans_per_trace: int = 1000,
                 trace_retention_hours: int = 24):
        
        self.service_name = service_name
        self.sampling_rate = sampling_rate
        self.max_spans_per_trace = max_spans_per_trace
        self.trace_retention_hours = trace_retention_hours
        
        self.logger = logging.getLogger(__name__)
        
        # Storage for traces and spans
        self.active_traces: Dict[str, List[Span]] = defaultdict(list)
        self.completed_traces: Dict[str, List[Span]] = {}
        self.span_storage: Dict[str, Span] = {}
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Exporters and processors
        self.span_processors: List[Callable[[Span], None]] = []
        self.trace_exporters: List[Callable[[str, List[Span]], None]] = []
        
        # Sampling configuration
        self.sampling_rules: List[Callable[[str, Dict[str, Any]], bool]] = []
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'traces_started': 0,
            'traces_completed': 0,
            'spans_created': 0,
            'spans_exported': 0,
            'sampling_decisions': {'sampled': 0, 'not_sampled': 0}
        }
        
        # Setup default BCI sampling rules
        self._setup_bci_sampling_rules()

    def _setup_bci_sampling_rules(self):
        """Setup BCI-specific sampling rules."""
        
        def always_sample_medical_operations(operation_name: str, tags: Dict[str, Any]) -> bool:
            """Always sample medical safety operations."""
            medical_keywords = ['medical', 'safety', 'emergency', 'critical', 'patient']
            return any(keyword in operation_name.lower() for keyword in medical_keywords)
        
        def sample_neural_processing(operation_name: str, tags: Dict[str, Any]) -> bool:
            """Sample neural processing operations at higher rate."""
            neural_keywords = ['neural', 'signal', 'decode', 'bci', 'eeg']
            if any(keyword in operation_name.lower() for keyword in neural_keywords):
                return True  # 100% sampling for neural operations
            return False
        
        def sample_claude_operations(operation_name: str, tags: Dict[str, Any]) -> bool:
            """Sample Claude operations based on safety mode."""
            if 'claude' in operation_name.lower():
                safety_mode = tags.get('safety_mode', 'moderate')
                if safety_mode == 'strict':
                    return True  # Always sample strict safety mode
                elif safety_mode == 'moderate':
                    return time.time() % 2 == 0  # 50% sampling
                else:
                    return time.time() % 4 == 0  # 25% sampling
            return False
        
        self.sampling_rules.extend([
            always_sample_medical_operations,
            sample_neural_processing,
            sample_claude_operations
        ])

    def should_sample(self, operation_name: str, tags: Dict[str, Any] = None) -> bool:
        """Determine if operation should be sampled."""
        tags = tags or {}
        
        # Check custom sampling rules first
        for rule in self.sampling_rules:
            try:
                if rule(operation_name, tags):
                    self.stats['sampling_decisions']['sampled'] += 1
                    return True
            except Exception as e:
                self.logger.warning(f"Sampling rule error: {e}")
        
        # Default probabilistic sampling
        should_sample = time.time() % (1.0 / self.sampling_rate) < 1.0
        
        if should_sample:
            self.stats['sampling_decisions']['sampled'] += 1
        else:
            self.stats['sampling_decisions']['not_sampled'] += 1
        
        return should_sample

    def start_trace(self, operation_name: str, kind: SpanKind = SpanKind.SERVER,
                   tags: Dict[str, Any] = None, 
                   neural_session_id: str = None,
                   patient_id: str = None,
                   device_id: str = None,
                   paradigm: str = None) -> TraceContext:
        """Start a new distributed trace."""
        
        tags = tags or {}
        
        # Check sampling decision
        if not self.should_sample(operation_name, tags):
            # Return a non-recording trace context
            return TraceContext(
                trace_id=str(uuid.uuid4()),
                span_id=str(uuid.uuid4()),
                trace_flags=0  # Not sampled
            )
        
        # Create new trace
        trace_id = str(uuid.uuid4())
        span_id = str(uuid.uuid4())
        
        # Create root span
        span = Span(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=None,
            operation_name=operation_name,
            kind=kind,
            start_time=time.time(),
            tags=tags.copy(),
            neural_session_id=neural_session_id,
            patient_id=patient_id,
            device_id=device_id,
            paradigm=paradigm
        )
        
        # Store span
        with self._lock:
            self.active_traces[trace_id].append(span)
            self.span_storage[span_id] = span
            self.stats['traces_started'] += 1
            self.stats['spans_created'] += 1
        
        # Create trace context
        context = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=1  # Sampled
        )
        
        # Add BCI-specific baggage
        if neural_session_id:
            context.baggage['neural_session_id'] = neural_session_id
        if patient_id:
            context.baggage['patient_id'] = patient_id
        if paradigm:
            context.baggage['paradigm'] = paradigm
        
        self.logger.debug(f"Started trace {trace_id} for operation: {operation_name}")
        
        return context

    def start_span(self, operation_name: str, kind: SpanKind = SpanKind.INTERNAL,
                  parent_context: TraceContext = None,
                  tags: Dict[str, Any] = None) -> Optional[Span]:
        """Start a new span within existing trace or create new trace."""
        
        # Get current context if not provided
        if parent_context is None:
            parent_context = _current_trace_context.get()
        
        if parent_context is None or parent_context.trace_flags == 0:
            # No active trace or not sampled
            return None
        
        tags = tags or {}
        
        # Create child span
        span_id = str(uuid.uuid4())
        span = Span(
            trace_id=parent_context.trace_id,
            span_id=span_id,
            parent_span_id=parent_context.span_id,
            operation_name=operation_name,
            kind=kind,
            start_time=time.time(),
            tags=tags.copy()
        )
        
        # Inherit BCI context from parent
        if parent_context.baggage:
            span.neural_session_id = parent_context.baggage.get('neural_session_id')
            span.patient_id = parent_context.baggage.get('patient_id')
            span.paradigm = parent_context.baggage.get('paradigm')
        
        # Store span
        with self._lock:
            self.active_traces[parent_context.trace_id].append(span)
            self.span_storage[span_id] = span
            self.stats['spans_created'] += 1
        
        self.logger.debug(f"Started span {span_id} for operation: {operation_name}")
        
        return span

    def finish_span(self, span: Span, status: SpanStatus = SpanStatus.OK,
                   error: Exception = None) -> None:
        """Finish a span."""
        if span is None:
            return
        
        span.finish(status)
        
        if error:
            span.set_tag("error", True)
            span.set_tag("error.type", type(error).__name__)
            span.set_tag("error.message", str(error))
        
        # Process span through processors
        for processor in self.span_processors:
            try:
                processor(span)
            except Exception as e:
                self.logger.error(f"Span processor error: {e}")
        
        self.logger.debug(f"Finished span {span.span_id} with status: {status.value}")

    def finish_trace(self, trace_id: str) -> None:
        """Finish a trace and export it."""
        with self._lock:
            if trace_id not in self.active_traces:
                return
            
            spans = self.active_traces.pop(trace_id)
            self.completed_traces[trace_id] = spans
            self.stats['traces_completed'] += 1
        
        # Export trace
        for exporter in self.trace_exporters:
            try:
                exporter(trace_id, spans)
                self.stats['spans_exported'] += len(spans)
            except Exception as e:
                self.logger.error(f"Trace exporter error: {e}")
        
        self.logger.debug(f"Finished trace {trace_id} with {len(spans)} spans")

    @contextmanager
    def trace(self, operation_name: str, kind: SpanKind = SpanKind.INTERNAL,
             tags: Dict[str, Any] = None, **kwargs):
        """Context manager for tracing operations."""
        # Check if we have an active trace
        current_context = _current_trace_context.get()
        
        if current_context is None:
            # Start new trace
            context = self.start_trace(operation_name, SpanKind.SERVER, tags, **kwargs)
            span = self.span_storage.get(context.span_id)
        else:
            # Start child span
            span = self.start_span(operation_name, kind, current_context, tags)
            context = TraceContext(
                trace_id=span.trace_id if span else current_context.trace_id,
                span_id=span.span_id if span else current_context.span_id,
                parent_span_id=current_context.span_id,
                trace_flags=current_context.trace_flags,
                baggage=current_context.baggage.copy()
            )
        
        # Set context variables
        context_token = _current_trace_context.set(context)
        span_token = _current_span.set(span)
        
        try:
            yield span
        except Exception as e:
            if span:
                self.finish_span(span, SpanStatus.ERROR, e)
            raise
        else:
            if span:
                self.finish_span(span, SpanStatus.OK)
        finally:
            # Reset context variables
            _current_trace_context.reset(context_token)
            _current_span.reset(span_token)
            
            # Finish trace if this was the root span
            if span and span.parent_span_id is None:
                self.finish_trace(span.trace_id)

    @asynccontextmanager
    async def trace_async(self, operation_name: str, kind: SpanKind = SpanKind.INTERNAL,
                         tags: Dict[str, Any] = None, **kwargs):
        """Async context manager for tracing operations."""
        with self.trace(operation_name, kind, tags, **kwargs) as span:
            yield span

    def add_span_processor(self, processor: Callable[[Span], None]) -> None:
        """Add a span processor."""
        self.span_processors.append(processor)
        self.logger.info(f"Added span processor")

    def add_trace_exporter(self, exporter: Callable[[str, List[Span]], None]) -> None:
        """Add a trace exporter."""
        self.trace_exporters.append(exporter)
        self.logger.info(f"Added trace exporter")

    def get_current_trace_context(self) -> Optional[TraceContext]:
        """Get current trace context."""
        return _current_trace_context.get()

    def get_current_span(self) -> Optional[Span]:
        """Get current span."""
        return _current_span.get()

    def add_event_to_current_span(self, event_name: str, attributes: Dict[str, Any] = None) -> None:
        """Add event to current span if available."""
        span = _current_span.get()
        if span:
            span.add_event(event_name, attributes)

    def set_tag_on_current_span(self, key: str, value: Any) -> None:
        """Set tag on current span if available."""
        span = _current_span.get()
        if span:
            span.set_tag(key, value)

    def get_trace_analysis(self, trace_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed analysis of a trace."""
        spans = self.completed_traces.get(trace_id) or self.active_traces.get(trace_id)
        
        if not spans:
            return None
        
        # Sort spans by start time
        sorted_spans = sorted(spans, key=lambda s: s.start_time)
        root_span = next((s for s in sorted_spans if s.parent_span_id is None), None)
        
        if not root_span:
            return None
        
        # Calculate trace statistics
        trace_duration = max(s.end_time or time.time() for s in spans) - root_span.start_time
        total_spans = len(spans)
        error_spans = sum(1 for s in spans if s.status == SpanStatus.ERROR)
        
        # Build span hierarchy
        span_tree = self._build_span_tree(spans)
        
        # Calculate critical path
        critical_path = self._calculate_critical_path(spans)
        
        # BCI-specific analysis
        bci_analysis = self._analyze_bci_trace(spans)
        
        return {
            "trace_id": trace_id,
            "root_operation": root_span.operation_name,
            "total_duration_ms": trace_duration * 1000,
            "total_spans": total_spans,
            "error_spans": error_spans,
            "success_rate": (total_spans - error_spans) / total_spans if total_spans > 0 else 0,
            "span_tree": span_tree,
            "critical_path": critical_path,
            "bci_analysis": bci_analysis,
            "neural_session_id": root_span.neural_session_id,
            "patient_id": root_span.patient_id,
            "paradigm": root_span.paradigm
        }

    def _build_span_tree(self, spans: List[Span]) -> Dict[str, Any]:
        """Build hierarchical span tree."""
        span_map = {span.span_id: span for span in spans}
        root_spans = [span for span in spans if span.parent_span_id is None]
        
        def build_node(span: Span) -> Dict[str, Any]:
            children = [build_node(child) for child in spans 
                       if child.parent_span_id == span.span_id]
            
            return {
                "span_id": span.span_id,
                "operation_name": span.operation_name,
                "duration_ms": span.duration_ms,
                "status": span.status.value,
                "tags": span.tags,
                "children": children
            }
        
        return [build_node(root) for root in root_spans]

    def _calculate_critical_path(self, spans: List[Span]) -> List[Dict[str, Any]]:
        """Calculate critical path through the trace."""
        # Simple critical path: longest sequential path
        span_map = {span.span_id: span for span in spans}
        
        def find_longest_path(span: Span, visited: Set[str]) -> List[Span]:
            if span.span_id in visited:
                return []
            
            visited.add(span.span_id)
            
            children = [s for s in spans if s.parent_span_id == span.span_id]
            if not children:
                return [span]
            
            longest_child_path = []
            for child in children:
                child_path = find_longest_path(child, visited.copy())
                if len(child_path) > len(longest_child_path):
                    longest_child_path = child_path
            
            return [span] + longest_child_path
        
        root_spans = [span for span in spans if span.parent_span_id is None]
        if not root_spans:
            return []
        
        critical_path = find_longest_path(root_spans[0], set())
        
        return [
            {
                "span_id": span.span_id,
                "operation_name": span.operation_name,
                "duration_ms": span.duration_ms,
                "cumulative_ms": sum(s.duration_ms or 0 for s in critical_path[:i+1])
            }
            for i, span in enumerate(critical_path)
        ]

    def _analyze_bci_trace(self, spans: List[Span]) -> Dict[str, Any]:
        """Analyze BCI-specific aspects of the trace."""
        neural_spans = [s for s in spans if 'neural' in s.operation_name.lower()]
        decoding_spans = [s for s in spans if 'decode' in s.operation_name.lower()]
        claude_spans = [s for s in spans if 'claude' in s.operation_name.lower()]
        
        analysis = {
            "neural_processing": {
                "span_count": len(neural_spans),
                "total_duration_ms": sum(s.duration_ms or 0 for s in neural_spans),
                "average_duration_ms": sum(s.duration_ms or 0 for s in neural_spans) / max(1, len(neural_spans))
            },
            "decoding": {
                "span_count": len(decoding_spans),
                "total_duration_ms": sum(s.duration_ms or 0 for s in decoding_spans),
                "average_duration_ms": sum(s.duration_ms or 0 for s in decoding_spans) / max(1, len(decoding_spans))
            },
            "claude_integration": {
                "span_count": len(claude_spans),
                "total_duration_ms": sum(s.duration_ms or 0 for s in claude_spans),
                "average_duration_ms": sum(s.duration_ms or 0 for s in claude_spans) / max(1, len(claude_spans))
            }
        }
        
        # Extract BCI-specific metrics from span tags
        signal_qualities = []
        decoding_confidences = []
        
        for span in spans:
            if 'signal_quality' in span.tags:
                signal_qualities.append(span.tags['signal_quality'])
            if 'decoding_confidence' in span.tags:
                decoding_confidences.append(span.tags['decoding_confidence'])
        
        if signal_qualities:
            analysis['signal_quality'] = {
                "min": min(signal_qualities),
                "max": max(signal_qualities),
                "average": sum(signal_qualities) / len(signal_qualities)
            }
        
        if decoding_confidences:
            analysis['decoding_confidence'] = {
                "min": min(decoding_confidences),
                "max": max(decoding_confidences),
                "average": sum(decoding_confidences) / len(decoding_confidences)
            }
        
        return analysis

    async def start_cleanup_task(self):
        """Start background cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self):
        """Background cleanup of old traces."""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_traces()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

    async def _cleanup_old_traces(self):
        """Clean up old completed traces."""
        current_time = time.time()
        retention_seconds = self.trace_retention_hours * 3600
        cutoff_time = current_time - retention_seconds
        
        traces_to_remove = []
        
        with self._lock:
            for trace_id, spans in self.completed_traces.items():
                if not spans:
                    continue
                
                # Check if trace is old
                latest_span_time = max(span.end_time or span.start_time for span in spans)
                if latest_span_time < cutoff_time:
                    traces_to_remove.append(trace_id)
            
            # Remove old traces
            for trace_id in traces_to_remove:
                del self.completed_traces[trace_id]
            
            # Clean up span storage
            spans_to_remove = []
            for span_id, span in self.span_storage.items():
                if (span.end_time or span.start_time) < cutoff_time:
                    spans_to_remove.append(span_id)
            
            for span_id in spans_to_remove:
                del self.span_storage[span_id]
        
        if traces_to_remove:
            self.logger.info(f"Cleaned up {len(traces_to_remove)} old traces")

    def get_tracer_stats(self) -> Dict[str, Any]:
        """Get tracer statistics."""
        with self._lock:
            active_traces_count = len(self.active_traces)
            completed_traces_count = len(self.completed_traces)
            total_spans_count = len(self.span_storage)
        
        return {
            "service_name": self.service_name,
            "sampling_rate": self.sampling_rate,
            "traces": {
                "active": active_traces_count,
                "completed": completed_traces_count,
                "total_started": self.stats['traces_started'],
                "total_completed": self.stats['traces_completed']
            },
            "spans": {
                "total_created": self.stats['spans_created'],
                "total_exported": self.stats['spans_exported'],
                "in_memory": total_spans_count
            },
            "sampling": self.stats['sampling_decisions'],
            "processors": len(self.span_processors),
            "exporters": len(self.trace_exporters)
        }


# Decorators for easy tracing
def trace_function(operation_name: str = None, kind: SpanKind = SpanKind.INTERNAL,
                  tags: Dict[str, Any] = None):
    """Decorator to trace function calls."""
    def decorator(func):
        nonlocal operation_name
        if operation_name is None:
            operation_name = f"{func.__module__}.{func.__name__}"
        
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                tracer = get_global_tracer()
                async with tracer.trace_async(operation_name, kind, tags):
                    return await func(*args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                tracer = get_global_tracer()
                with tracer.trace(operation_name, kind, tags):
                    return func(*args, **kwargs)
            return sync_wrapper
    
    return decorator


def trace_bci_operation(paradigm: str = None, neural_session_id: str = None):
    """Decorator specifically for BCI operations."""
    def decorator(func):
        operation_name = f"bci.{func.__name__}"
        tags = {}
        if paradigm:
            tags['bci.paradigm'] = paradigm
        
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            tracer = get_global_tracer()
            async with tracer.trace_async(
                operation_name, 
                SpanKind.INTERNAL, 
                tags,
                neural_session_id=neural_session_id,
                paradigm=paradigm
            ):
                return await func(*args, **kwargs)
        
        return wrapper
    
    return decorator


# Global tracer instance
_global_tracer: Optional[DistributedTracer] = None


def initialize_global_tracer(service_name: str = "bci-agent-bridge", **kwargs) -> DistributedTracer:
    """Initialize the global tracer."""
    global _global_tracer
    _global_tracer = DistributedTracer(service_name, **kwargs)
    return _global_tracer


def get_global_tracer() -> DistributedTracer:
    """Get the global tracer instance."""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = initialize_global_tracer()
    return _global_tracer


# Export functions for trace context propagation
def inject_trace_context(headers: Dict[str, str]) -> None:
    """Inject current trace context into headers."""
    context = _current_trace_context.get()
    if context:
        headers.update(context.to_headers())


def extract_trace_context(headers: Dict[str, str]) -> Optional[TraceContext]:
    """Extract trace context from headers."""
    return TraceContext.from_headers(headers)


def set_trace_context(context: TraceContext) -> None:
    """Set the current trace context."""
    _current_trace_context.set(context)


# Example exporters
class ConsoleTraceExporter:
    """Simple console exporter for development."""
    
    def __init__(self, include_spans: bool = True):
        self.include_spans = include_spans
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, trace_id: str, spans: List[Span]) -> None:
        """Export trace to console."""
        root_span = next((s for s in spans if s.parent_span_id is None), None)
        if not root_span:
            return
        
        trace_duration = max(s.end_time or time.time() for s in spans) - root_span.start_time
        
        print(f"\n--- TRACE {trace_id} ---")
        print(f"Operation: {root_span.operation_name}")
        print(f"Duration: {trace_duration * 1000:.1f}ms")
        print(f"Spans: {len(spans)}")
        print(f"Errors: {sum(1 for s in spans if s.status == SpanStatus.ERROR)}")
        
        if root_span.neural_session_id:
            print(f"Neural Session: {root_span.neural_session_id}")
        if root_span.paradigm:
            print(f"Paradigm: {root_span.paradigm}")
        
        if self.include_spans:
            print("\nSpans:")
            for span in sorted(spans, key=lambda s: s.start_time):
                indent = "  " * self._get_span_depth(span, spans)
                status_symbol = "✓" if span.status == SpanStatus.OK else "✗"
                print(f"{indent}{status_symbol} {span.operation_name} ({span.duration_ms:.1f}ms)")
        print()
    
    def _get_span_depth(self, span: Span, all_spans: List[Span]) -> int:
        """Calculate depth of span in trace hierarchy."""
        if span.parent_span_id is None:
            return 0
        
        parent = next((s for s in all_spans if s.span_id == span.parent_span_id), None)
        if parent:
            return 1 + self._get_span_depth(parent, all_spans)
        
        return 0


class JSONFileTraceExporter:
    """Export traces to JSON files."""
    
    def __init__(self, file_path: str = "traces.jsonl"):
        self.file_path = file_path
        self.logger = logging.getLogger(__name__)
    
    def __call__(self, trace_id: str, spans: List[Span]) -> None:
        """Export trace to JSON file."""
        try:
            trace_data = {
                "trace_id": trace_id,
                "timestamp": time.time(),
                "spans": [span.to_dict() for span in spans]
            }
            
            with open(self.file_path, 'a') as f:
                f.write(json.dumps(trace_data) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to export trace to file: {e}")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import random
    
    async def demo_distributed_tracing():
        print("BCI Distributed Tracing Demo")
        print("=" * 50)
        
        # Initialize tracer
        tracer = initialize_global_tracer("bci-demo-service", sampling_rate=1.0)
        
        # Add exporters
        tracer.add_trace_exporter(ConsoleTraceExporter(include_spans=True))
        tracer.add_trace_exporter(JSONFileTraceExporter("demo_traces.jsonl"))
        
        # Start cleanup task
        await tracer.start_cleanup_task()
        
        # Simulate BCI operations
        
        @trace_bci_operation(paradigm="P300", neural_session_id="session_123")
        async def neural_data_acquisition():
            """Simulate neural data acquisition."""
            await asyncio.sleep(0.1)
            
            # Add events and tags
            tracer.set_tag_on_current_span("channels", 64)
            tracer.set_tag_on_current_span("sampling_rate", 250)
            tracer.add_event_to_current_span("electrodes_attached")
            
            # Simulate signal quality check
            signal_quality = random.uniform(0.6, 1.0)
            tracer.set_tag_on_current_span("signal_quality", signal_quality)
            
            if signal_quality < 0.7:
                tracer.add_event_to_current_span("signal_quality_warning", 
                                                {"quality": signal_quality})
            
            return {"data": "neural_samples", "quality": signal_quality}
        
        @trace_function("signal_processing.filter")
        async def signal_filtering(neural_data):
            """Simulate signal filtering."""
            await asyncio.sleep(0.05)
            tracer.set_tag_on_current_span("filter_type", "bandpass")
            tracer.set_tag_on_current_span("frequency_range", "0.5-30Hz")
            return {"filtered_data": neural_data["data"]}
        
        @trace_function("bci.decode_intention")
        async def decode_intention(filtered_data):
            """Simulate intention decoding."""
            await asyncio.sleep(0.08)
            
            confidence = random.uniform(0.4, 1.0)
            tracer.set_tag_on_current_span("decoding_confidence", confidence)
            tracer.set_tag_on_current_span("paradigm", "P300")
            
            if confidence > 0.8:
                intention = "move_cursor_right"
                tracer.add_event_to_current_span("high_confidence_detection",
                                                {"intention": intention, "confidence": confidence})
            else:
                intention = "uncertain"
                tracer.add_event_to_current_span("low_confidence_detection",
                                                {"confidence": confidence})
            
            return {"intention": intention, "confidence": confidence}
        
        @trace_function("claude.process_intention", SpanKind.CLIENT)
        async def process_with_claude(intention_data):
            """Simulate Claude processing."""
            await asyncio.sleep(random.uniform(0.5, 1.5))  # Variable Claude latency
            
            tracer.set_tag_on_current_span("safety_mode", "strict")
            tracer.set_tag_on_current_span("model", "claude-3-sonnet")
            
            # Simulate occasional errors
            if random.random() < 0.1:  # 10% error rate
                tracer.add_event_to_current_span("api_error", {"error": "rate_limited"})
                raise Exception("Claude API rate limited")
            
            tracer.add_event_to_current_span("safety_check_passed")
            
            return {
                "response": f"Executing {intention_data['intention']} with confidence {intention_data['confidence']:.2f}",
                "safety_validated": True
            }
        
        # Main BCI processing pipeline
        async def bci_processing_pipeline():
            """Complete BCI processing pipeline."""
            with tracer.trace("bci.processing_pipeline", SpanKind.SERVER,
                            neural_session_id="session_123",
                            patient_id="patient_456",
                            paradigm="P300") as span:
                
                span.add_event("pipeline_started")
                
                try:
                    # Neural data acquisition
                    neural_data = await neural_data_acquisition()
                    span.add_event("neural_data_acquired")
                    
                    # Signal processing
                    filtered_data = await signal_filtering(neural_data)
                    span.add_event("signal_filtered")
                    
                    # Intention decoding
                    intention = await decode_intention(filtered_data)
                    span.add_event("intention_decoded")
                    
                    # Claude processing
                    response = await process_with_claude(intention)
                    span.add_event("claude_processing_complete")
                    
                    span.set_tag("pipeline_success", True)
                    span.set_tag("final_confidence", intention["confidence"])
                    
                    return response
                    
                except Exception as e:
                    span.add_event("pipeline_error", {"error": str(e)})
                    span.set_tag("pipeline_success", False)
                    raise
        
        # Run multiple pipeline iterations
        print("Running BCI processing pipeline iterations...")
        
        for i in range(5):
            try:
                print(f"\nIteration {i+1}:")
                result = await bci_processing_pipeline()
                print(f"✓ Success: {result['response']}")
            except Exception as e:
                print(f"✗ Error: {e}")
            
            await asyncio.sleep(0.5)
        
        # Show tracer statistics
        print(f"\n--- Tracer Statistics ---")
        stats = tracer.get_tracer_stats()
        print(f"Traces started: {stats['traces']['total_started']}")
        print(f"Spans created: {stats['spans']['total_created']}")
        print(f"Sampling decisions: {stats['sampling']}")
        
        # Analyze a trace
        if tracer.completed_traces:
            trace_id = list(tracer.completed_traces.keys())[0]
            analysis = tracer.get_trace_analysis(trace_id)
            if analysis:
                print(f"\n--- Trace Analysis: {trace_id[:8]}... ---")
                print(f"Total duration: {analysis['total_duration_ms']:.1f}ms")
                print(f"Success rate: {analysis['success_rate']:.1%}")
                if analysis['bci_analysis']['signal_quality']:
                    print(f"Average signal quality: {analysis['bci_analysis']['signal_quality']['average']:.2f}")
        
        # Cleanup
        await tracer.stop_cleanup_task()
        print("\nDistributed tracing demo completed")
    
    asyncio.run(demo_distributed_tracing())