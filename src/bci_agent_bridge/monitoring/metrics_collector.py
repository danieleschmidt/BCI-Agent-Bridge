"""
Metrics collection and performance monitoring.
"""

import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging
import json


@dataclass
class Metric:
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""


@dataclass
class MetricSummary:
    name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    last_value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """
    Collects and aggregates performance metrics for BCI components.
    """
    
    def __init__(self, retention_period: float = 3600.0, max_metrics_per_name: int = 1000):
        self.retention_period = retention_period  # seconds
        self.max_metrics_per_name = max_metrics_per_name
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe storage
        self._lock = threading.Lock()
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_metrics_per_name))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        
        # Custom metric processors
        self.processors: Dict[str, Callable] = {}
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = "") -> None:
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {},
            unit=unit
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            
        # Process through any registered processors
        if name in self.processors:
            try:
                self.processors[name](metric)
            except Exception as e:
                self.logger.error(f"Metric processor error for '{name}': {e}")
    
    def increment_counter(self, name: str, delta: int = 1, tags: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += delta
        
        self.record_metric(f"{name}_total", self.counters[name], tags, "count")
    
    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = "") -> None:
        """Set a gauge metric value."""
        with self._lock:
            self.gauges[name] = value
        
        self.record_metric(name, value, tags, unit)
    
    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = "") -> None:
        """Record a value in a histogram."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only recent values for histogram
            if len(self.histograms[name]) > self.max_metrics_per_name:
                self.histograms[name] = self.histograms[name][-self.max_metrics_per_name:]
        
        self.record_metric(name, value, tags, unit)
    
    def register_processor(self, metric_name: str, processor: Callable[[Metric], None]) -> None:
        """Register a custom metric processor."""
        self.processors[metric_name] = processor
        self.logger.info(f"Registered processor for metric: {metric_name}")
    
    def get_metric_summary(self, name: str) -> Optional[MetricSummary]:
        """Get summary statistics for a metric."""
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
            
            values = [m.value for m in self.metrics[name]]
            last_metric = self.metrics[name][-1]
            
            return MetricSummary(
                name=name,
                count=len(values),
                min_value=min(values),
                max_value=max(values),
                avg_value=sum(values) / len(values),
                last_value=values[-1],
                unit=last_metric.unit,
                tags=last_metric.tags
            )
    
    def get_all_summaries(self) -> Dict[str, MetricSummary]:
        """Get summaries for all metrics."""
        summaries = {}
        with self._lock:
            for name in self.metrics.keys():
                summary = self.get_metric_summary(name)
                if summary:
                    summaries[name] = summary
        return summaries
    
    def get_recent_metrics(self, name: str, seconds: float = 60.0) -> List[Metric]:
        """Get metrics from the last N seconds."""
        cutoff_time = time.time() - seconds
        
        with self._lock:
            if name not in self.metrics:
                return []
            
            return [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
    
    def cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - self.retention_period
        
        with self._lock:
            for name, metric_deque in self.metrics.items():
                # Remove old metrics from the left side
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()
    
    def export_metrics(self, format: str = "json") -> str:
        """Export metrics in specified format."""
        if format.lower() == "json":
            return self._export_json()
        elif format.lower() == "prometheus":
            return self._export_prometheus()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _export_json(self) -> str:
        """Export metrics as JSON."""
        summaries = self.get_all_summaries()
        export_data = {
            "timestamp": time.time(),
            "metrics": {
                name: {
                    "count": summary.count,
                    "min": summary.min_value,
                    "max": summary.max_value,
                    "avg": summary.avg_value,
                    "last": summary.last_value,
                    "unit": summary.unit,
                    "tags": summary.tags
                }
                for name, summary in summaries.items()
            },
            "counters": dict(self.counters),
            "gauges": dict(self.gauges)
        }
        return json.dumps(export_data, indent=2)
    
    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        lines.append("# BCI Agent Bridge Metrics")
        lines.append(f"# Exported at {time.time()}")
        
        summaries = self.get_all_summaries()
        
        for name, summary in summaries.items():
            # Clean name for Prometheus
            prom_name = name.replace("-", "_").replace(".", "_")
            
            # Add help text
            lines.append(f"# HELP {prom_name} {summary.unit or 'value'}")
            lines.append(f"# TYPE {prom_name} gauge")
            
            # Add tags
            tag_str = ""
            if summary.tags:
                tag_pairs = [f'{k}="{v}"' for k, v in summary.tags.items()]
                tag_str = "{" + ",".join(tag_pairs) + "}"
            
            lines.append(f"{prom_name}{tag_str} {summary.last_value}")
        
        # Add counters
        for name, value in self.counters.items():
            prom_name = name.replace("-", "_").replace(".", "_")
            lines.append(f"# TYPE {prom_name} counter")
            lines.append(f"{prom_name} {value}")
        
        return "\n".join(lines)
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
        
        self.logger.info("All metrics reset")


class BCIMetricsCollector(MetricsCollector):
    """
    Specialized metrics collector for BCI bridge components.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_bci_processors()
    
    def _setup_bci_processors(self) -> None:
        """Setup BCI-specific metric processors."""
        
        def process_neural_data_rate(metric: Metric) -> None:
            """Process neural data rate metrics."""
            if metric.value < 200:  # Below 200 Hz
                self.logger.warning(f"Low neural data rate: {metric.value} Hz")
        
        def process_decoding_latency(metric: Metric) -> None:
            """Process decoding latency metrics."""
            if metric.value > 100:  # Above 100ms
                self.logger.warning(f"High decoding latency: {metric.value}ms")
        
        def process_claude_response_time(metric: Metric) -> None:
            """Process Claude response time metrics."""
            if metric.value > 2000:  # Above 2 seconds
                self.logger.warning(f"Slow Claude response: {metric.value}ms")
        
        self.register_processor("neural_data_rate", process_neural_data_rate)
        self.register_processor("decoding_latency", process_decoding_latency)
        self.register_processor("claude_response_time", process_claude_response_time)
    
    def record_neural_sample(self, channels: int, sampling_rate: int, quality_score: float = None) -> None:
        """Record neural data sample metrics."""
        self.increment_counter("neural_samples")
        self.set_gauge("neural_channels", channels, unit="count")
        self.record_metric("neural_data_rate", sampling_rate, unit="Hz")
        
        if quality_score is not None:
            self.record_histogram("signal_quality", quality_score, unit="score")
    
    def record_decoding_event(self, paradigm: str, confidence: float, latency_ms: float) -> None:
        """Record neural decoding event."""
        tags = {"paradigm": paradigm}
        
        self.increment_counter("decoding_attempts", tags=tags)
        self.record_histogram("decoding_confidence", confidence, tags, "score")
        self.record_histogram("decoding_latency", latency_ms, tags, "ms")
        
        if confidence >= 0.8:
            self.increment_counter("high_confidence_decodings", tags=tags)
    
    def record_claude_interaction(self, safety_mode: str, tokens_used: int, response_time_ms: float, 
                                 safety_flags: List[str] = None) -> None:
        """Record Claude interaction metrics."""
        tags = {"safety_mode": safety_mode}
        
        self.increment_counter("claude_requests", tags=tags)
        self.record_histogram("claude_response_time", response_time_ms, tags, "ms")
        self.record_histogram("claude_tokens_used", tokens_used, tags, "tokens")
        
        if safety_flags:
            for flag in safety_flags:
                self.increment_counter("safety_flags", tags={"flag": flag})
    
    def record_system_performance(self, cpu_percent: float, memory_mb: float, 
                                 buffer_size: int, active_streams: int) -> None:
        """Record system performance metrics."""
        self.set_gauge("cpu_usage", cpu_percent, unit="percent")
        self.set_gauge("memory_usage", memory_mb, unit="MB")
        self.set_gauge("buffer_size", buffer_size, unit="samples")
        self.set_gauge("active_streams", active_streams, unit="count")
    
    def get_bci_performance_summary(self) -> Dict[str, Any]:
        """Get BCI-specific performance summary."""
        summaries = self.get_all_summaries()
        
        return {
            "neural_processing": {
                "samples_processed": self.counters.get("neural_samples", 0),
                "data_rate": summaries.get("neural_data_rate"),
                "signal_quality": summaries.get("signal_quality")
            },
            "decoding_performance": {
                "attempts": self.counters.get("decoding_attempts", 0),
                "high_confidence": self.counters.get("high_confidence_decodings", 0),
                "confidence": summaries.get("decoding_confidence"),
                "latency": summaries.get("decoding_latency")
            },
            "claude_integration": {
                "requests": self.counters.get("claude_requests", 0),
                "response_time": summaries.get("claude_response_time"),
                "tokens_used": summaries.get("claude_tokens_used"),
                "safety_flags": self.counters.get("safety_flags", 0)
            },
            "system_health": {
                "cpu_usage": self.gauges.get("cpu_usage"),
                "memory_usage": self.gauges.get("memory_usage"),
                "buffer_size": self.gauges.get("buffer_size"),
                "active_streams": self.gauges.get("active_streams")
            }
        }