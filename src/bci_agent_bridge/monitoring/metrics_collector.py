"""
Enterprise-grade metrics collection and performance monitoring with time-series data and anomaly detection.
"""

import time
import threading
import statistics
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import logging
import json
import uuid
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import pickle
import os


class MetricType(Enum):
    GAUGE = "gauge"
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"
    RATE = "rate"


class AnomalyType(Enum):
    SPIKE = "spike"
    DIP = "dip"
    DRIFT = "drift"
    OSCILLATION = "oscillation"
    FLATLINE = "flatline"
    OUTLIER = "outlier"


@dataclass
class Metric:
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    unit: str = ""
    metric_type: MetricType = MetricType.GAUGE
    correlation_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "unit": self.unit,
            "type": self.metric_type.value,
            "correlation_id": self.correlation_id
        }


@dataclass
class MetricSummary:
    name: str
    count: int
    min_value: float
    max_value: float
    avg_value: float
    median_value: float
    std_deviation: float
    percentiles: Dict[str, float]
    last_value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)
    anomaly_score: float = 0.0
    trend_direction: str = "stable"
    trend_confidence: float = 0.0


@dataclass
class AnomalyDetection:
    metric_name: str
    anomaly_type: AnomalyType
    severity: float  # 0.0 to 1.0
    timestamp: float
    value: float
    expected_range: Tuple[float, float]
    confidence: float
    description: str
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert anomaly detection to dictionary."""
        return {
            "metric_name": self.metric_name,
            "anomaly_type": self.anomaly_type.value,
            "severity": self.severity,
            "timestamp": self.timestamp,
            "value": self.value,
            "expected_range": self.expected_range,
            "confidence": self.confidence,
            "description": self.description,
            "correlation_id": self.correlation_id,
            "tags": self.tags
        }


class TimeSeriesStorage:
    """Efficient time-series storage with SQLite backend."""
    
    def __init__(self, db_path: str = "metrics.db", retention_days: int = 30):
        self.db_path = db_path
        self.retention_days = retention_days
        self.logger = logging.getLogger(__name__)
        self._init_database()

    def _init_database(self):
        """Initialize SQLite database with proper schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    value REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    tags TEXT,
                    unit TEXT,
                    metric_type TEXT,
                    correlation_id TEXT
                )
            """)
            
            # Create indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_name_timestamp ON metrics(name, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON metrics(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_correlation_id ON metrics(correlation_id)")
            
            # Create anomalies table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT NOT NULL,
                    anomaly_type TEXT NOT NULL,
                    severity REAL NOT NULL,
                    timestamp REAL NOT NULL,
                    value REAL NOT NULL,
                    expected_min REAL,
                    expected_max REAL,
                    confidence REAL NOT NULL,
                    description TEXT,
                    correlation_id TEXT,
                    tags TEXT
                )
            """)
            
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomalies(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_metric ON anomalies(metric_name)")
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")

    def store_metric(self, metric: Metric):
        """Store a metric in time-series database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO metrics (name, value, timestamp, tags, unit, metric_type, correlation_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                metric.name,
                metric.value,
                metric.timestamp,
                json.dumps(metric.tags),
                metric.unit,
                metric.metric_type.value,
                metric.correlation_id
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store metric {metric.name}: {e}")

    def store_anomaly(self, anomaly: AnomalyDetection):
        """Store an anomaly detection in database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO anomalies (metric_name, anomaly_type, severity, timestamp, value,
                                     expected_min, expected_max, confidence, description, 
                                     correlation_id, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                anomaly.metric_name,
                anomaly.anomaly_type.value,
                anomaly.severity,
                anomaly.timestamp,
                anomaly.value,
                anomaly.expected_range[0],
                anomaly.expected_range[1],
                anomaly.confidence,
                anomaly.description,
                anomaly.correlation_id,
                json.dumps(anomaly.tags)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Failed to store anomaly for {anomaly.metric_name}: {e}")

    def query_metrics(self, metric_name: str, start_time: float, end_time: float,
                     tags: Optional[Dict[str, str]] = None) -> List[Metric]:
        """Query metrics from time-series database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = """
                SELECT name, value, timestamp, tags, unit, metric_type, correlation_id
                FROM metrics 
                WHERE name = ? AND timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            """
            
            cursor.execute(query, (metric_name, start_time, end_time))
            results = []
            
            for row in cursor.fetchall():
                stored_tags = json.loads(row[3]) if row[3] else {}
                
                # Filter by tags if specified
                if tags:
                    if not all(stored_tags.get(k) == v for k, v in tags.items()):
                        continue
                
                metric = Metric(
                    name=row[0],
                    value=row[1],
                    timestamp=row[2],
                    tags=stored_tags,
                    unit=row[4] or "",
                    metric_type=MetricType(row[5]) if row[5] else MetricType.GAUGE,
                    correlation_id=row[6]
                )
                results.append(metric)
            
            conn.close()
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to query metrics for {metric_name}: {e}")
            return []

    def cleanup_old_data(self):
        """Remove data older than retention period."""
        try:
            cutoff_time = time.time() - (self.retention_days * 24 * 3600)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Delete old metrics
            cursor.execute("DELETE FROM metrics WHERE timestamp < ?", (cutoff_time,))
            metrics_deleted = cursor.rowcount
            
            # Delete old anomalies
            cursor.execute("DELETE FROM anomalies WHERE timestamp < ?", (cutoff_time,))
            anomalies_deleted = cursor.rowcount
            
            # Vacuum database to reclaim space
            cursor.execute("VACUUM")
            
            conn.commit()
            conn.close()
            
            self.logger.info(f"Cleaned up {metrics_deleted} old metrics and {anomalies_deleted} old anomalies")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old data: {e}")


class AnomalyDetector:
    """Advanced anomaly detection for metrics."""
    
    def __init__(self, sensitivity: float = 0.7, min_samples: int = 30):
        self.sensitivity = sensitivity  # 0.0 to 1.0, higher = more sensitive
        self.min_samples = min_samples
        self.logger = logging.getLogger(__name__)

    def detect_anomalies(self, values: List[float], timestamps: List[float],
                        metric_name: str, tags: Dict[str, str] = None) -> List[AnomalyDetection]:
        """Detect anomalies in a time series."""
        if len(values) < self.min_samples:
            return []

        anomalies = []
        current_time = timestamps[-1] if timestamps else time.time()
        current_value = values[-1] if values else 0.0
        tags = tags or {}

        try:
            # Convert to numpy arrays for efficient computation
            values_array = np.array(values)
            timestamps_array = np.array(timestamps)

            # Statistical baseline
            mean_val = np.mean(values_array)
            std_val = np.std(values_array)
            median_val = np.median(values_array)
            
            # Robust statistics using IQR
            q25, q75 = np.percentile(values_array, [25, 75])
            iqr = q75 - q25
            
            # Expected range based on IQR
            expected_min = q25 - 1.5 * iqr
            expected_max = q75 + 1.5 * iqr
            expected_range = (expected_min, expected_max)

            # 1. Outlier detection (current value)
            if current_value < expected_min or current_value > expected_max:
                severity = min(1.0, abs(current_value - median_val) / (std_val + 1e-8))
                anomaly_type = AnomalyType.SPIKE if current_value > expected_max else AnomalyType.DIP
                
                anomalies.append(AnomalyDetection(
                    metric_name=metric_name,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    timestamp=current_time,
                    value=current_value,
                    expected_range=expected_range,
                    confidence=0.8,
                    description=f"Value {current_value:.3f} is outside expected range [{expected_min:.3f}, {expected_max:.3f}]",
                    tags=tags
                ))

            # 2. Trend-based anomalies
            if len(values) >= 10:
                trend_anomalies = self._detect_trend_anomalies(
                    values_array, timestamps_array, metric_name, tags, expected_range
                )
                anomalies.extend(trend_anomalies)

            # 3. Pattern-based anomalies
            if len(values) >= 20:
                pattern_anomalies = self._detect_pattern_anomalies(
                    values_array, timestamps_array, metric_name, tags, expected_range
                )
                anomalies.extend(pattern_anomalies)

        except Exception as e:
            self.logger.error(f"Anomaly detection failed for {metric_name}: {e}")

        return anomalies

    def _detect_trend_anomalies(self, values: np.ndarray, timestamps: np.ndarray,
                               metric_name: str, tags: Dict[str, str],
                               expected_range: Tuple[float, float]) -> List[AnomalyDetection]:
        """Detect trend-based anomalies."""
        anomalies = []
        
        try:
            # Linear regression to detect drift
            x = timestamps - timestamps[0]  # Normalize timestamps
            coeffs = np.polyfit(x, values, 1)
            slope = coeffs[0]
            
            # Calculate trend strength
            predicted = np.polyval(coeffs, x)
            r_squared = 1 - np.sum((values - predicted) ** 2) / np.sum((values - np.mean(values)) ** 2)
            
            # Significant trend detection
            trend_threshold = self.sensitivity * 0.1  # Adjust based on sensitivity
            if abs(slope) > trend_threshold and r_squared > 0.5:
                severity = min(1.0, abs(slope) / trend_threshold)
                
                anomalies.append(AnomalyDetection(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.DRIFT,
                    severity=severity,
                    timestamp=timestamps[-1],
                    value=values[-1],
                    expected_range=expected_range,
                    confidence=r_squared,
                    description=f"Significant trend detected: slope={slope:.6f}, R²={r_squared:.3f}",
                    tags=tags
                ))
                
        except Exception as e:
            self.logger.warning(f"Trend anomaly detection failed: {e}")
            
        return anomalies

    def _detect_pattern_anomalies(self, values: np.ndarray, timestamps: np.ndarray,
                                 metric_name: str, tags: Dict[str, str],
                                 expected_range: Tuple[float, float]) -> List[AnomalyDetection]:
        """Detect pattern-based anomalies."""
        anomalies = []
        
        try:
            # 1. Flatline detection
            recent_values = values[-10:]  # Last 10 values
            if len(set(recent_values.round(6))) <= 2:  # Values are essentially the same
                flatline_duration = len(recent_values)
                severity = min(1.0, flatline_duration / 20)  # Max severity at 20 identical values
                
                anomalies.append(AnomalyDetection(
                    metric_name=metric_name,
                    anomaly_type=AnomalyType.FLATLINE,
                    severity=severity,
                    timestamp=timestamps[-1],
                    value=values[-1],
                    expected_range=expected_range,
                    confidence=0.9,
                    description=f"Flatline detected: {flatline_duration} identical values",
                    tags=tags
                ))

            # 2. Oscillation detection
            if len(values) >= 20:
                # Calculate rate of change
                rate_changes = np.diff(values)
                sign_changes = np.diff(np.sign(rate_changes))
                oscillation_count = np.sum(np.abs(sign_changes) > 1)
                
                # High frequency oscillations are anomalous
                oscillation_rate = oscillation_count / len(rate_changes)
                if oscillation_rate > 0.3:  # More than 30% sign changes
                    severity = min(1.0, oscillation_rate)
                    
                    anomalies.append(AnomalyDetection(
                        metric_name=metric_name,
                        anomaly_type=AnomalyType.OSCILLATION,
                        severity=severity,
                        timestamp=timestamps[-1],
                        value=values[-1],
                        expected_range=expected_range,
                        confidence=0.7,
                        description=f"High-frequency oscillation detected: {oscillation_rate:.2f} oscillation rate",
                        tags=tags
                    ))

        except Exception as e:
            self.logger.warning(f"Pattern anomaly detection failed: {e}")
            
        return anomalies


class MetricsCollector:
    """
    Enterprise-grade metrics collector with time-series storage and anomaly detection.
    """
    
    def __init__(self, retention_period: float = 3600.0, max_metrics_per_name: int = 10000,
                 enable_time_series: bool = True, enable_anomaly_detection: bool = True,
                 db_path: Optional[str] = None):
        self.retention_period = retention_period  # seconds
        self.max_metrics_per_name = max_metrics_per_name
        self.enable_time_series = enable_time_series
        self.enable_anomaly_detection = enable_anomaly_detection
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe storage
        self._lock = threading.Lock()
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_metrics_per_name))
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.rates: Dict[str, List[Tuple[float, float]]] = defaultdict(list)  # (timestamp, value) pairs
        
        # Time-series storage
        if self.enable_time_series:
            db_path = db_path or "metrics.db"
            self.time_series = TimeSeriesStorage(db_path)
        else:
            self.time_series = None
        
        # Anomaly detection
        if self.enable_anomaly_detection:
            self.anomaly_detector = AnomalyDetector()
            self.detected_anomalies: List[AnomalyDetection] = []
            self.anomaly_callbacks: List[Callable[[AnomalyDetection], None]] = []
        else:
            self.anomaly_detector = None
            self.detected_anomalies = []
            self.anomaly_callbacks = []
        
        # Custom metric processors
        self.processors: Dict[str, Callable] = {}
        
        # Background tasks
        self.background_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="metrics")
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.collector_metrics = {
            'metrics_processed': 0,
            'anomalies_detected': 0,
            'storage_operations': 0,
            'processing_errors': 0,
            'last_cleanup': 0.0
        }
        
        # Start background cleanup
        self._start_cleanup_task()

    def _start_cleanup_task(self):
        """Start background cleanup task."""
        try:
            if not self._cleanup_task or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            # No event loop running, cleanup will be handled manually
            self._cleanup_task = None

    async def _cleanup_loop(self):
        """Background cleanup task."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._perform_cleanup()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")

    async def _perform_cleanup(self):
        """Perform cleanup operations."""
        try:
            # Clean up old in-memory metrics
            self.cleanup_old_metrics()
            
            # Clean up time-series storage
            if self.time_series:
                await asyncio.get_event_loop().run_in_executor(
                    self.background_executor, self.time_series.cleanup_old_data
                )
            
            # Clean up old anomalies (keep last 1000)
            if len(self.detected_anomalies) > 1000:
                self.detected_anomalies = self.detected_anomalies[-1000:]
            
            self.collector_metrics['last_cleanup'] = time.time()
            self.logger.debug("Cleanup completed successfully")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, 
                     unit: str = "", metric_type: MetricType = MetricType.GAUGE,
                     correlation_id: Optional[str] = None) -> None:
        """Record a metric value with enhanced features."""
        current_time = time.time()
        tags = tags or {}
        
        metric = Metric(
            name=name,
            value=value,
            timestamp=current_time,
            tags=tags,
            unit=unit,
            metric_type=metric_type,
            correlation_id=correlation_id
        )
        
        try:
            # Store in memory
            with self._lock:
                self.metrics[name].append(metric)
            
            # Store in time-series database
            if self.time_series:
                self.background_executor.submit(self.time_series.store_metric, metric)
                self.collector_metrics['storage_operations'] += 1
            
            # Process through any registered processors
            if name in self.processors:
                try:
                    self.processors[name](metric)
                except Exception as e:
                    self.logger.error(f"Metric processor error for '{name}': {e}")
                    self.collector_metrics['processing_errors'] += 1
            
            # Perform anomaly detection
            if self.anomaly_detector and len(self.metrics[name]) >= 30:
                self._detect_anomalies_async(name, tags)
            
            self.collector_metrics['metrics_processed'] += 1
            
        except Exception as e:
            self.logger.error(f"Failed to record metric {name}: {e}")
            self.collector_metrics['processing_errors'] += 1

    def _detect_anomalies_async(self, metric_name: str, tags: Dict[str, str]):
        """Asynchronously detect anomalies for a metric."""
        def detect_and_process():
            try:
                with self._lock:
                    recent_metrics = list(self.metrics[metric_name])[-100:]  # Last 100 metrics
                
                if len(recent_metrics) < 30:
                    return
                
                values = [m.value for m in recent_metrics]
                timestamps = [m.timestamp for m in recent_metrics]
                
                anomalies = self.anomaly_detector.detect_anomalies(
                    values, timestamps, metric_name, tags
                )
                
                for anomaly in anomalies:
                    self.detected_anomalies.append(anomaly)
                    
                    # Store in database
                    if self.time_series:
                        self.time_series.store_anomaly(anomaly)
                    
                    # Notify callbacks
                    for callback in self.anomaly_callbacks:
                        try:
                            callback(anomaly)
                        except Exception as e:
                            self.logger.error(f"Anomaly callback failed: {e}")
                    
                    self.logger.warning(
                        f"ANOMALY DETECTED: {anomaly.anomaly_type.value} in {metric_name} "
                        f"(severity: {anomaly.severity:.2f}, confidence: {anomaly.confidence:.2f})"
                    )
                
                if anomalies:
                    self.collector_metrics['anomalies_detected'] += len(anomalies)
                
            except Exception as e:
                self.logger.error(f"Anomaly detection failed for {metric_name}: {e}")
        
        self.background_executor.submit(detect_and_process)

    def increment_counter(self, name: str, delta: int = 1, tags: Dict[str, str] = None,
                         correlation_id: Optional[str] = None) -> None:
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] += delta
        
        self.record_metric(f"{name}_total", self.counters[name], tags, "count", 
                          MetricType.COUNTER, correlation_id)

    def set_gauge(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = "",
                 correlation_id: Optional[str] = None) -> None:
        """Set a gauge metric value."""
        with self._lock:
            self.gauges[name] = value
        
        self.record_metric(name, value, tags, unit, MetricType.GAUGE, correlation_id)

    def record_histogram(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = "",
                        correlation_id: Optional[str] = None) -> None:
        """Record a value in a histogram."""
        with self._lock:
            self.histograms[name].append(value)
            # Keep only recent values for histogram
            if len(self.histograms[name]) > self.max_metrics_per_name:
                self.histograms[name] = self.histograms[name][-self.max_metrics_per_name:]
        
        self.record_metric(name, value, tags, unit, MetricType.HISTOGRAM, correlation_id)

    def record_rate(self, name: str, value: float, tags: Dict[str, str] = None, unit: str = "",
                   correlation_id: Optional[str] = None) -> None:
        """Record a rate metric (value per second)."""
        current_time = time.time()
        
        with self._lock:
            self.rates[name].append((current_time, value))
            # Keep only recent values
            if len(self.rates[name]) > self.max_metrics_per_name:
                self.rates[name] = self.rates[name][-self.max_metrics_per_name:]
        
        # Calculate rate (per second) if we have previous values
        rate_value = 0.0
        if len(self.rates[name]) >= 2:
            prev_time, prev_value = self.rates[name][-2]
            time_diff = current_time - prev_time
            if time_diff > 0:
                rate_value = (value - prev_value) / time_diff
        
        self.record_metric(f"{name}_rate", rate_value, tags, f"{unit}/sec", 
                          MetricType.RATE, correlation_id)

    def register_processor(self, metric_name: str, processor: Callable[[Metric], None]) -> None:
        """Register a custom metric processor."""
        self.processors[metric_name] = processor
        self.logger.info(f"Registered processor for metric: {metric_name}")

    def register_anomaly_callback(self, callback: Callable[[AnomalyDetection], None]) -> None:
        """Register a callback for anomaly detection."""
        self.anomaly_callbacks.append(callback)
        self.logger.info(f"Registered anomaly callback")

    def get_metric_summary(self, name: str, include_anomaly_score: bool = True) -> Optional[MetricSummary]:
        """Get enhanced summary statistics for a metric."""
        with self._lock:
            if name not in self.metrics or not self.metrics[name]:
                return None
            
            values = [m.value for m in self.metrics[name]]
            last_metric = self.metrics[name][-1]
        
        try:
            # Calculate comprehensive statistics
            count = len(values)
            min_value = min(values)
            max_value = max(values)
            avg_value = statistics.mean(values)
            median_value = statistics.median(values)
            std_deviation = statistics.stdev(values) if count > 1 else 0.0
            
            # Calculate percentiles
            percentiles = {}
            if count >= 5:
                percentiles = {
                    "p5": np.percentile(values, 5),
                    "p25": np.percentile(values, 25),
                    "p50": np.percentile(values, 50),
                    "p75": np.percentile(values, 75),
                    "p95": np.percentile(values, 95),
                    "p99": np.percentile(values, 99)
                }
            
            # Calculate trend
            trend_direction = "stable"
            trend_confidence = 0.0
            
            if count >= 10:
                timestamps = [m.timestamp for m in self.metrics[name]]
                x = np.array(timestamps) - timestamps[0]
                y = np.array(values)
                
                try:
                    coeffs = np.polyfit(x, y, 1)
                    slope = coeffs[0]
                    
                    # Calculate R-squared for trend confidence
                    predicted = np.polyval(coeffs, x)
                    r_squared = 1 - np.sum((y - predicted) ** 2) / np.sum((y - np.mean(y)) ** 2)
                    trend_confidence = max(0.0, r_squared)
                    
                    if abs(slope) > std_deviation * 0.1:  # Significant trend
                        trend_direction = "increasing" if slope > 0 else "decreasing"
                    
                except Exception:
                    pass  # Keep defaults
            
            # Calculate anomaly score
            anomaly_score = 0.0
            if include_anomaly_score and self.detected_anomalies:
                recent_anomalies = [
                    a for a in self.detected_anomalies 
                    if a.metric_name == name and (time.time() - a.timestamp) < 3600  # Last hour
                ]
                if recent_anomalies:
                    anomaly_score = max(a.severity for a in recent_anomalies)
            
            return MetricSummary(
                name=name,
                count=count,
                min_value=min_value,
                max_value=max_value,
                avg_value=avg_value,
                median_value=median_value,
                std_deviation=std_deviation,
                percentiles=percentiles,
                last_value=values[-1],
                unit=last_metric.unit,
                tags=last_metric.tags,
                anomaly_score=anomaly_score,
                trend_direction=trend_direction,
                trend_confidence=trend_confidence
            )
            
        except Exception as e:
            self.logger.error(f"Failed to calculate summary for {name}: {e}")
            return None

    def get_time_series_data(self, metric_name: str, start_time: float, end_time: float,
                           tags: Optional[Dict[str, str]] = None) -> List[Metric]:
        """Get time-series data from storage."""
        if not self.time_series:
            # Fallback to in-memory data
            with self._lock:
                if metric_name not in self.metrics:
                    return []
                
                return [
                    m for m in self.metrics[metric_name]
                    if start_time <= m.timestamp <= end_time and
                    (not tags or all(m.tags.get(k) == v for k, v in tags.items()))
                ]
        
        return self.time_series.query_metrics(metric_name, start_time, end_time, tags)

    def get_anomalies(self, metric_name: Optional[str] = None, 
                     start_time: Optional[float] = None,
                     end_time: Optional[float] = None) -> List[AnomalyDetection]:
        """Get detected anomalies with optional filtering."""
        filtered_anomalies = self.detected_anomalies.copy()
        
        if metric_name:
            filtered_anomalies = [a for a in filtered_anomalies if a.metric_name == metric_name]
        
        if start_time:
            filtered_anomalies = [a for a in filtered_anomalies if a.timestamp >= start_time]
        
        if end_time:
            filtered_anomalies = [a for a in filtered_anomalies if a.timestamp <= end_time]
        
        return sorted(filtered_anomalies, key=lambda a: a.timestamp, reverse=True)

    def cleanup_old_metrics(self) -> None:
        """Remove metrics older than retention period."""
        cutoff_time = time.time() - self.retention_period
        
        with self._lock:
            for name, metric_deque in self.metrics.items():
                # Remove old metrics from the left side
                while metric_deque and metric_deque[0].timestamp < cutoff_time:
                    metric_deque.popleft()

    def export_metrics(self, format: str = "json", include_anomalies: bool = True,
                      start_time: Optional[float] = None, end_time: Optional[float] = None) -> str:
        """Export metrics with enhanced formatting options."""
        if format.lower() == "json":
            return self._export_json(include_anomalies, start_time, end_time)
        elif format.lower() == "prometheus":
            return self._export_prometheus()
        elif format.lower() == "influxdb":
            return self._export_influxdb(start_time, end_time)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def _export_json(self, include_anomalies: bool, start_time: Optional[float], end_time: Optional[float]) -> str:
        """Export metrics as JSON with time-series data."""
        summaries = {}
        for name in self.metrics.keys():
            summary = self.get_metric_summary(name)
            if summary:
                summaries[name] = {
                    "count": summary.count,
                    "min": summary.min_value,
                    "max": summary.max_value,
                    "avg": summary.avg_value,
                    "median": summary.median_value,
                    "std_dev": summary.std_deviation,
                    "percentiles": summary.percentiles,
                    "last": summary.last_value,
                    "unit": summary.unit,
                    "tags": summary.tags,
                    "anomaly_score": summary.anomaly_score,
                    "trend": summary.trend_direction,
                    "trend_confidence": summary.trend_confidence
                }
        
        export_data = {
            "timestamp": time.time(),
            "retention_period": self.retention_period,
            "metrics": summaries,
            "counters": dict(self.counters),
            "gauges": dict(self.gauges),
            "collector_stats": self.collector_metrics
        }
        
        if include_anomalies:
            export_data["anomalies"] = [
                anomaly.to_dict() for anomaly in self.get_anomalies(start_time=start_time, end_time=end_time)
            ]
        
        return json.dumps(export_data, indent=2, default=str)

    def _export_prometheus(self) -> str:
        """Export metrics in Prometheus format with enhanced metadata."""
        lines = []
        lines.append("# BCI Agent Bridge Metrics (Enhanced)")
        lines.append(f"# Exported at {time.time()}")
        
        summaries = {}
        for name in self.metrics.keys():
            summary = self.get_metric_summary(name)
            if summary:
                summaries[name] = summary
        
        for name, summary in summaries.items():
            # Clean name for Prometheus
            prom_name = name.replace("-", "_").replace(".", "_")
            
            # Add help text
            lines.append(f"# HELP {prom_name} {summary.unit or 'value'} (anomaly_score: {summary.anomaly_score:.2f})")
            
            # Determine metric type
            metric_type = "gauge"  # Default
            if name.endswith("_total"):
                metric_type = "counter"
            elif name in self.histograms:
                metric_type = "histogram"
            
            lines.append(f"# TYPE {prom_name} {metric_type}")
            
            # Add tags including trend information
            tag_pairs = []
            for k, v in summary.tags.items():
                tag_pairs.append(f'{k}="{v}"')
            
            # Add trend as a tag
            tag_pairs.append(f'trend="{summary.trend_direction}"')
            tag_pairs.append(f'anomaly_level="{self._categorize_anomaly_score(summary.anomaly_score)}"')
            
            tag_str = "{" + ",".join(tag_pairs) + "}" if tag_pairs else ""
            
            lines.append(f"{prom_name}{tag_str} {summary.last_value}")
            
            # Add additional metrics for histograms
            if metric_type == "histogram":
                for percentile, value in summary.percentiles.items():
                    lines.append(f"{prom_name}_{percentile}{tag_str} {value}")
        
        return "\n".join(lines)

    def _export_influxdb(self, start_time: Optional[float], end_time: Optional[float]) -> str:
        """Export metrics in InfluxDB line protocol format."""
        lines = []
        current_time = time.time()
        
        # Get time range
        if not start_time:
            start_time = current_time - self.retention_period
        if not end_time:
            end_time = current_time
        
        for name in self.metrics.keys():
            time_series_data = self.get_time_series_data(name, start_time, end_time)
            
            for metric in time_series_data:
                # Format: measurement,tag1=value1,tag2=value2 field1=value1,field2=value2 timestamp
                measurement = metric.name.replace(" ", "_").replace(",", "_")
                
                # Tags
                tag_parts = []
                for k, v in metric.tags.items():
                    tag_parts.append(f"{k}={v}")
                
                tag_str = "," + ",".join(tag_parts) if tag_parts else ""
                
                # Fields
                field_str = f"value={metric.value}"
                if metric.unit:
                    field_str += f',unit="{metric.unit}"'
                
                # Timestamp in nanoseconds
                timestamp_ns = int(metric.timestamp * 1e9)
                
                lines.append(f"{measurement}{tag_str} {field_str} {timestamp_ns}")
        
        return "\n".join(lines)

    def _categorize_anomaly_score(self, score: float) -> str:
        """Categorize anomaly score into levels."""
        if score >= 0.8:
            return "critical"
        elif score >= 0.6:
            return "high"
        elif score >= 0.4:
            return "medium"
        elif score >= 0.2:
            return "low"
        else:
            return "normal"

    def reset_metrics(self) -> None:
        """Reset all metrics."""
        with self._lock:
            self.metrics.clear()
            self.counters.clear()
            self.gauges.clear()
            self.histograms.clear()
            self.rates.clear()
        
        self.detected_anomalies.clear()
        self.collector_metrics = {
            'metrics_processed': 0,
            'anomalies_detected': 0,
            'storage_operations': 0,
            'processing_errors': 0,
            'last_cleanup': time.time()
        }
        
        self.logger.info("All metrics reset")

    async def cleanup(self):
        """Cleanup collector resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.background_executor.shutdown(wait=True)
        self.logger.info("Metrics collector cleanup complete")


class BCIMetricsCollector(MetricsCollector):
    """
    Specialized metrics collector for BCI bridge components with enhanced monitoring.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._setup_bci_processors()
        self._setup_bci_anomaly_callbacks()

    def _setup_bci_processors(self) -> None:
        """Setup BCI-specific metric processors with enhanced logic."""
        
        def process_neural_data_rate(metric: Metric) -> None:
            """Process neural data rate metrics with trend analysis."""
            if metric.value < 200:  # Below 200 Hz
                self.logger.warning(f"Low neural data rate: {metric.value} Hz")
                
                # Check if this is a trend
                recent_values = [m.value for m in list(self.metrics['neural_data_rate'])[-10:]]
                if len(recent_values) >= 5 and all(v < 200 for v in recent_values[-5:]):
                    self.logger.error("Sustained low neural data rate - potential hardware issue")
        
        def process_decoding_latency(metric: Metric) -> None:
            """Process decoding latency metrics with performance tracking."""
            if metric.value > 100:  # Above 100ms
                self.logger.warning(f"High decoding latency: {metric.value}ms")
                
                # Track performance degradation
                recent_values = [m.value for m in list(self.metrics['decoding_latency'])[-20:]]
                if len(recent_values) >= 10:
                    avg_recent = statistics.mean(recent_values[-10:])
                    avg_older = statistics.mean(recent_values[-20:-10]) if len(recent_values) >= 20 else avg_recent
                    
                    if avg_recent > avg_older * 1.5:  # 50% increase
                        self.logger.error("Decoding latency is degrading - system performance issue")
        
        def process_claude_response_time(metric: Metric) -> None:
            """Process Claude response time metrics with API health tracking."""
            if metric.value > 2000:  # Above 2 seconds
                self.logger.warning(f"Slow Claude response: {metric.value}ms")
                
                # Check for API health issues
                recent_values = [m.value for m in list(self.metrics['claude_response_time'])[-10:]]
                if len(recent_values) >= 5 and statistics.mean(recent_values[-5:]) > 3000:
                    self.logger.error("Claude API performance is consistently poor")
        
        def process_signal_quality(metric: Metric) -> None:
            """Process signal quality with medical safety considerations."""
            if metric.value < 0.5:  # Below 50% quality
                self.logger.warning(f"Signal quality degraded: {metric.value}")
                
                # Medical safety check
                if metric.value < 0.3:  # Critical quality threshold
                    self.logger.critical("MEDICAL ALERT: Signal quality critically low - patient safety may be compromised")
        
        # Register processors
        self.register_processor("neural_data_rate", process_neural_data_rate)
        self.register_processor("decoding_latency", process_decoding_latency)
        self.register_processor("claude_response_time", process_claude_response_time)
        self.register_processor("signal_quality", process_signal_quality)

    def _setup_bci_anomaly_callbacks(self) -> None:
        """Setup BCI-specific anomaly detection callbacks."""
        
        def handle_critical_anomaly(anomaly: AnomalyDetection) -> None:
            """Handle critical anomalies with appropriate alerting."""
            # Medical safety metrics require immediate attention
            medical_metrics = ['signal_quality', 'neural_data_rate', 'device_connection']
            
            if (anomaly.metric_name in medical_metrics and 
                anomaly.severity >= 0.8 and
                anomaly.anomaly_type in [AnomalyType.SPIKE, AnomalyType.DIP, AnomalyType.FLATLINE]):
                
                self.logger.critical(
                    f"CRITICAL MEDICAL ANOMALY: {anomaly.description} "
                    f"- Immediate intervention required"
                )
                
                # Here you would integrate with medical alerting systems
                # For now, we'll just log with high visibility
            
            elif anomaly.severity >= 0.7:
                self.logger.error(
                    f"HIGH SEVERITY ANOMALY: {anomaly.description} "
                    f"- System monitoring required"
                )
        
        self.register_anomaly_callback(handle_critical_anomaly)

    def record_neural_sample(self, channels: int, sampling_rate: int, quality_score: float = None,
                           correlation_id: Optional[str] = None) -> None:
        """Record neural data sample metrics with enhanced tracking."""
        self.increment_counter("neural_samples", correlation_id=correlation_id)
        self.set_gauge("neural_channels", channels, unit="count", correlation_id=correlation_id)
        self.record_metric("neural_data_rate", sampling_rate, unit="Hz", 
                          metric_type=MetricType.GAUGE, correlation_id=correlation_id)
        
        if quality_score is not None:
            self.record_histogram("signal_quality", quality_score, unit="score",
                                correlation_id=correlation_id)
            
            # Also record as gauge for real-time monitoring
            self.set_gauge("current_signal_quality", quality_score, unit="score",
                         correlation_id=correlation_id)

    def record_decoding_event(self, paradigm: str, confidence: float, latency_ms: float,
                            success: bool = True, correlation_id: Optional[str] = None) -> None:
        """Record neural decoding event with comprehensive metrics."""
        tags = {"paradigm": paradigm}
        
        self.increment_counter("decoding_attempts", tags=tags, correlation_id=correlation_id)
        self.record_histogram("decoding_confidence", confidence, tags, "score", correlation_id=correlation_id)
        self.record_histogram("decoding_latency", latency_ms, tags, "ms", correlation_id=correlation_id)
        
        if success:
            self.increment_counter("successful_decodings", tags=tags, correlation_id=correlation_id)
        else:
            self.increment_counter("failed_decodings", tags=tags, correlation_id=correlation_id)
        
        # Track confidence categories
        if confidence >= 0.8:
            self.increment_counter("high_confidence_decodings", tags=tags, correlation_id=correlation_id)
        elif confidence >= 0.6:
            self.increment_counter("medium_confidence_decodings", tags=tags, correlation_id=correlation_id)
        else:
            self.increment_counter("low_confidence_decodings", tags=tags, correlation_id=correlation_id)
        
        # Real-time success rate
        total_attempts = self.counters.get("decoding_attempts", 1)
        successful = self.counters.get("successful_decodings", 0)
        success_rate = successful / total_attempts
        self.set_gauge("decoding_success_rate", success_rate, unit="ratio", correlation_id=correlation_id)

    def record_claude_interaction(self, safety_mode: str, tokens_used: int, response_time_ms: float, 
                                 safety_flags: List[str] = None, error: bool = False,
                                 correlation_id: Optional[str] = None) -> None:
        """Record Claude interaction metrics with safety tracking."""
        tags = {"safety_mode": safety_mode}
        
        self.increment_counter("claude_requests", tags=tags, correlation_id=correlation_id)
        self.record_histogram("claude_response_time", response_time_ms, tags, "ms", correlation_id=correlation_id)
        self.record_histogram("claude_tokens_used", tokens_used, tags, "tokens", correlation_id=correlation_id)
        
        if error:
            self.increment_counter("claude_errors", tags=tags, correlation_id=correlation_id)
        else:
            self.increment_counter("claude_successes", tags=tags, correlation_id=correlation_id)
        
        # Track error rate
        total_requests = self.counters.get("claude_requests", 1)
        errors = self.counters.get("claude_errors", 0)
        error_rate = errors / total_requests
        self.set_gauge("claude_error_rate", error_rate, unit="ratio", correlation_id=correlation_id)
        
        # Safety flags tracking
        if safety_flags:
            for flag in safety_flags:
                self.increment_counter("safety_flags", tags={"flag": flag, **tags}, correlation_id=correlation_id)
            
            # Record number of flags per request
            self.record_histogram("safety_flags_per_request", len(safety_flags), 
                                tags, "count", correlation_id=correlation_id)

    def record_system_performance(self, cpu_percent: float, memory_mb: float, 
                                 buffer_size: int, active_streams: int,
                                 disk_usage_percent: float = None,
                                 network_latency_ms: float = None,
                                 correlation_id: Optional[str] = None) -> None:
        """Record comprehensive system performance metrics."""
        self.set_gauge("cpu_usage", cpu_percent, unit="percent", correlation_id=correlation_id)
        self.set_gauge("memory_usage", memory_mb, unit="MB", correlation_id=correlation_id)
        self.set_gauge("buffer_size", buffer_size, unit="samples", correlation_id=correlation_id)
        self.set_gauge("active_streams", active_streams, unit="count", correlation_id=correlation_id)
        
        # Optional metrics
        if disk_usage_percent is not None:
            self.set_gauge("disk_usage", disk_usage_percent, unit="percent", correlation_id=correlation_id)
        
        if network_latency_ms is not None:
            self.record_histogram("network_latency", network_latency_ms, unit="ms", correlation_id=correlation_id)
        
        # Derived metrics
        memory_gb = memory_mb / 1024
        self.set_gauge("memory_usage_gb", memory_gb, unit="GB", correlation_id=correlation_id)
        
        # System health score (0-100)
        health_score = 100
        if cpu_percent > 90:
            health_score -= 30
        elif cpu_percent > 70:
            health_score -= 15
        
        if memory_mb > 2000:  # Above 2GB
            health_score -= 20
        elif memory_mb > 1500:  # Above 1.5GB
            health_score -= 10
        
        if disk_usage_percent and disk_usage_percent > 90:
            health_score -= 20
        elif disk_usage_percent and disk_usage_percent > 80:
            health_score -= 10
        
        self.set_gauge("system_health_score", max(0, health_score), unit="score", correlation_id=correlation_id)

    def get_bci_performance_summary(self, include_anomalies: bool = True) -> Dict[str, Any]:
        """Get comprehensive BCI-specific performance summary."""
        base_summaries = {}
        for name in self.metrics.keys():
            summary = self.get_metric_summary(name, include_anomaly_score=include_anomalies)
            if summary:
                base_summaries[name] = summary
        
        # Calculate derived metrics
        current_time = time.time()
        
        # Neural processing metrics
        neural_processing = {
            "samples_processed": self.counters.get("neural_samples", 0),
            "data_rate": self._get_summary_dict(base_summaries.get("neural_data_rate")),
            "signal_quality": self._get_summary_dict(base_summaries.get("signal_quality")),
            "current_quality": self.gauges.get("current_signal_quality", 0.0),
            "channels_active": self.gauges.get("neural_channels", 0)
        }
        
        # Decoding performance metrics
        total_attempts = self.counters.get("decoding_attempts", 0)
        successful_decodings = self.counters.get("successful_decodings", 0)
        
        decoding_performance = {
            "total_attempts": total_attempts,
            "successful_decodings": successful_decodings,
            "success_rate": successful_decodings / max(total_attempts, 1),
            "high_confidence": self.counters.get("high_confidence_decodings", 0),
            "medium_confidence": self.counters.get("medium_confidence_decodings", 0),
            "low_confidence": self.counters.get("low_confidence_decodings", 0),
            "confidence_distribution": self._get_summary_dict(base_summaries.get("decoding_confidence")),
            "latency": self._get_summary_dict(base_summaries.get("decoding_latency")),
            "current_success_rate": self.gauges.get("decoding_success_rate", 0.0)
        }
        
        # Claude integration metrics
        total_requests = self.counters.get("claude_requests", 0)
        claude_errors = self.counters.get("claude_errors", 0)
        
        claude_integration = {
            "total_requests": total_requests,
            "successful_requests": self.counters.get("claude_successes", 0),
            "error_rate": claude_errors / max(total_requests, 1),
            "current_error_rate": self.gauges.get("claude_error_rate", 0.0),
            "response_time": self._get_summary_dict(base_summaries.get("claude_response_time")),
            "tokens_used": self._get_summary_dict(base_summaries.get("claude_tokens_used")),
            "safety_flags_triggered": self.counters.get("safety_flags", 0)
        }
        
        # System health metrics
        system_health = {
            "cpu_usage": self.gauges.get("cpu_usage", 0.0),
            "memory_usage_mb": self.gauges.get("memory_usage", 0.0),
            "memory_usage_gb": self.gauges.get("memory_usage_gb", 0.0),
            "disk_usage": self.gauges.get("disk_usage", 0.0),
            "buffer_size": self.gauges.get("buffer_size", 0),
            "active_streams": self.gauges.get("active_streams", 0),
            "health_score": self.gauges.get("system_health_score", 100.0),
            "network_latency": self._get_summary_dict(base_summaries.get("network_latency"))
        }
        
        # Anomaly summary
        anomaly_summary = {}
        if include_anomalies:
            recent_anomalies = self.get_anomalies(start_time=current_time - 3600)  # Last hour
            anomaly_summary = {
                "total_anomalies_1h": len(recent_anomalies),
                "by_type": defaultdict(int),
                "by_severity": {"low": 0, "medium": 0, "high": 0, "critical": 0},
                "by_metric": defaultdict(int)
            }
            
            for anomaly in recent_anomalies:
                anomaly_summary["by_type"][anomaly.anomaly_type.value] += 1
                anomaly_summary["by_metric"][anomaly.metric_name] += 1
                
                if anomaly.severity >= 0.8:
                    anomaly_summary["by_severity"]["critical"] += 1
                elif anomaly.severity >= 0.6:
                    anomaly_summary["by_severity"]["high"] += 1
                elif anomaly.severity >= 0.4:
                    anomaly_summary["by_severity"]["medium"] += 1
                else:
                    anomaly_summary["by_severity"]["low"] += 1
        
        return {
            "timestamp": current_time,
            "neural_processing": neural_processing,
            "decoding_performance": decoding_performance,
            "claude_integration": claude_integration,
            "system_health": system_health,
            "anomalies": dict(anomaly_summary) if anomaly_summary else {},
            "collector_stats": self.collector_metrics
        }

    def _get_summary_dict(self, summary: Optional[MetricSummary]) -> Dict[str, Any]:
        """Convert MetricSummary to dictionary for JSON serialization."""
        if not summary:
            return {}
        
        return {
            "count": summary.count,
            "min": summary.min_value,
            "max": summary.max_value,
            "avg": summary.avg_value,
            "median": summary.median_value,
            "std_dev": summary.std_deviation,
            "last": summary.last_value,
            "percentiles": summary.percentiles,
            "unit": summary.unit,
            "anomaly_score": summary.anomaly_score,
            "trend": summary.trend_direction,
            "trend_confidence": summary.trend_confidence
        }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import random
    
    async def demo_enhanced_metrics_collector():
        print("Enhanced BCI Metrics Collector Demo")
        print("=" * 50)
        
        # Create enhanced metrics collector
        collector = BCIMetricsCollector(
            retention_period=3600.0,
            enable_time_series=True,
            enable_anomaly_detection=True
        )
        
        # Simulate BCI operations for a period
        print("Simulating BCI operations...")
        
        for i in range(100):
            # Simulate neural data
            quality = max(0.1, min(1.0, 0.8 + random.normalvariate(0, 0.1)))
            if i > 50 and i < 60:  # Inject some poor quality data
                quality *= 0.3
            
            sampling_rate = random.normalvariate(250, 10)
            channels = random.choice([32, 64, 128])
            
            correlation_id = str(uuid.uuid4())
            collector.record_neural_sample(channels, sampling_rate, quality, correlation_id)
            
            # Simulate decoding events
            if i % 5 == 0:  # Every 5th iteration
                paradigm = random.choice(["P300", "SSVEP", "Motor Imagery"])
                confidence = max(0.3, min(1.0, random.normalvariate(0.75, 0.15)))
                latency = max(20, random.normalvariate(80, 20))
                success = confidence > 0.6
                
                collector.record_decoding_event(paradigm, confidence, latency, success, correlation_id)
            
            # Simulate Claude interactions
            if i % 10 == 0:  # Every 10th iteration
                safety_mode = random.choice(["strict", "moderate", "permissive"])
                tokens = random.randint(100, 2000)
                response_time = max(500, random.normalvariate(1500, 500))
                error = random.random() < 0.05  # 5% error rate
                
                safety_flags = []
                if random.random() < 0.1:  # 10% chance of safety flags
                    safety_flags = random.sample(["inappropriate_request", "medical_concern", "safety_override"], 
                                                random.randint(1, 2))
                
                collector.record_claude_interaction(safety_mode, tokens, response_time, 
                                                  safety_flags, error, correlation_id)
            
            # Simulate system performance
            cpu_usage = max(10, min(100, random.normalvariate(45, 15)))
            memory_mb = max(500, random.normalvariate(1200, 200))
            buffer_size = random.randint(1000, 5000)
            active_streams = random.randint(1, 5)
            disk_usage = max(30, min(100, random.normalvariate(60, 10)))
            network_latency = max(10, random.normalvariate(50, 15))
            
            collector.record_system_performance(cpu_usage, memory_mb, buffer_size, 
                                              active_streams, disk_usage, network_latency, correlation_id)
            
            await asyncio.sleep(0.1)  # 100ms interval
        
        print("Data collection complete. Analyzing...")
        
        # Wait for background processing
        await asyncio.sleep(2)
        
        # Get comprehensive summary
        summary = collector.get_bci_performance_summary()
        print("\n--- BCI Performance Summary ---")
        print(json.dumps(summary, indent=2, default=str))
        
        # Show recent anomalies
        anomalies = collector.get_anomalies()
        if anomalies:
            print(f"\n--- Detected Anomalies ({len(anomalies)}) ---")
            for anomaly in anomalies[:5]:  # Show first 5
                print(f"🚨 {anomaly.anomaly_type.value.upper()}: {anomaly.description}")
                print(f"   Metric: {anomaly.metric_name}, Severity: {anomaly.severity:.2f}, Confidence: {anomaly.confidence:.2f}")
                print(f"   Time: {time.strftime('%H:%M:%S', time.localtime(anomaly.timestamp))}")
                print()
        
        # Export metrics in different formats
        print("\n--- Export Examples ---")
        
        # JSON export
        json_export = collector.export_metrics("json", include_anomalies=True)
        print(f"JSON export size: {len(json_export)} characters")
        
        # Prometheus export
        prom_export = collector.export_metrics("prometheus")
        print(f"Prometheus export size: {len(prom_export)} characters")
        print("First few lines of Prometheus export:")
        print("\n".join(prom_export.split("\n")[:10]))
        
        # Cleanup
        await collector.cleanup()
    
    asyncio.run(demo_enhanced_metrics_collector())