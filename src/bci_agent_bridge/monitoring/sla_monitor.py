"""
Service Level Agreement (SLA) monitoring and performance tracking for BCI system.
Monitors performance metrics against defined SLA thresholds and tracks violations.
"""

import time
import asyncio
import logging
import statistics
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import json
import uuid
import threading


class SLAStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    VIOLATED = "violated"
    CRITICAL = "critical"


class SLAMetricType(Enum):
    LATENCY = "latency"              # Response time metrics
    AVAILABILITY = "availability"    # Uptime/downtime metrics  
    THROUGHPUT = "throughput"       # Requests per second
    ERROR_RATE = "error_rate"       # Error percentage
    QUALITY = "quality"             # Signal quality, confidence scores
    CUSTOM = "custom"               # Custom business metrics


@dataclass
class SLAThreshold:
    """Defines SLA threshold parameters."""
    metric_name: str
    metric_type: SLAMetricType
    warning_threshold: float
    violation_threshold: float
    critical_threshold: float
    time_window_seconds: int = 300  # 5 minutes default
    minimum_samples: int = 10
    higher_is_better: bool = False  # True for metrics like availability, quality
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass  
class SLAViolation:
    """Represents an SLA violation event."""
    id: str
    metric_name: str
    violation_type: SLAStatus
    threshold_value: float
    actual_value: float
    time_window_start: float
    time_window_end: float
    duration_seconds: float
    sample_count: int
    severity: float  # 0.0 to 1.0
    description: str
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    tags: Dict[str, str] = field(default_factory=dict)
    resolved_at: Optional[float] = None
    impact_assessment: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert violation to dictionary."""
        return {
            "id": self.id,
            "metric_name": self.metric_name,
            "violation_type": self.violation_type.value,
            "threshold_value": self.threshold_value,
            "actual_value": self.actual_value,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "duration_seconds": self.duration_seconds,
            "sample_count": self.sample_count,
            "severity": self.severity,
            "description": self.description,
            "correlation_id": self.correlation_id,
            "tags": self.tags,
            "resolved_at": self.resolved_at,
            "impact_assessment": self.impact_assessment
        }


@dataclass
class SLAReport:
    """SLA compliance report for a metric."""
    metric_name: str
    report_period_start: float
    report_period_end: float
    total_samples: int
    availability_percentage: float
    mean_value: float
    p50_value: float
    p95_value: float
    p99_value: float
    violations: List[SLAViolation]
    compliance_percentage: float
    sla_status: SLAStatus
    trend_direction: str
    recommendations: List[str]
    tags: Dict[str, str] = field(default_factory=dict)


class SLAMonitor:
    """
    Service Level Agreement monitor for BCI system performance tracking.
    """
    
    def __init__(self, 
                 check_interval: float = 60.0,
                 history_retention_hours: int = 24,
                 enable_auto_remediation: bool = True):
        
        self.check_interval = check_interval
        self.history_retention_hours = history_retention_hours
        self.enable_auto_remediation = enable_auto_remediation
        
        self.logger = logging.getLogger(__name__)
        
        # SLA configuration
        self.sla_thresholds: Dict[str, SLAThreshold] = {}
        
        # Data storage
        self.metric_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=10000)  # Keep last 10k samples per metric
        )
        
        # Violation tracking
        self.active_violations: Dict[str, SLAViolation] = {}
        self.violation_history: List[SLAViolation] = []
        self.violation_callbacks: List[Callable[[SLAViolation], None]] = []
        
        # Monitoring state
        self.monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.monitor_stats = {
            'checks_performed': 0,
            'violations_detected': 0,
            'violations_resolved': 0,
            'auto_remediations_attempted': 0,
            'auto_remediations_successful': 0,
            'last_check_time': 0.0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Auto-remediation strategies
        self.remediation_strategies: Dict[str, Callable[[SLAViolation], bool]] = {}
        
        # Setup default BCI SLA thresholds
        self._setup_default_sla_thresholds()

    def _setup_default_sla_thresholds(self):
        """Setup default SLA thresholds for BCI system."""
        # Neural signal processing SLAs
        self.register_sla_threshold(SLAThreshold(
            metric_name="neural_data_rate",
            metric_type=SLAMetricType.THROUGHPUT,
            warning_threshold=200.0,  # Hz
            violation_threshold=150.0,
            critical_threshold=100.0,
            time_window_seconds=300,
            minimum_samples=50,
            higher_is_better=True,
            description="Neural data sampling rate must maintain minimum frequency for reliable BCI operation"
        ))
        
        self.register_sla_threshold(SLAThreshold(
            metric_name="signal_quality",
            metric_type=SLAMetricType.QUALITY,
            warning_threshold=0.7,
            violation_threshold=0.5,
            critical_threshold=0.3,
            time_window_seconds=300,
            minimum_samples=20,
            higher_is_better=True,
            description="Neural signal quality must remain above minimum thresholds for accurate decoding"
        ))
        
        # Decoding performance SLAs
        self.register_sla_threshold(SLAThreshold(
            metric_name="decoding_latency",
            metric_type=SLAMetricType.LATENCY,
            warning_threshold=100.0,  # ms
            violation_threshold=200.0,
            critical_threshold=500.0,
            time_window_seconds=300,
            minimum_samples=30,
            higher_is_better=False,
            description="Decoding latency must stay below thresholds for real-time BCI control"
        ))
        
        self.register_sla_threshold(SLAThreshold(
            metric_name="decoding_success_rate",
            metric_type=SLAMetricType.QUALITY,
            warning_threshold=0.8,
            violation_threshold=0.6,
            critical_threshold=0.4,
            time_window_seconds=600,
            minimum_samples=20,
            higher_is_better=True,
            description="Decoding success rate must maintain acceptable accuracy for user satisfaction"
        ))
        
        # Claude API integration SLAs
        self.register_sla_threshold(SLAThreshold(
            metric_name="claude_response_time",
            metric_type=SLAMetricType.LATENCY,
            warning_threshold=2000.0,  # ms
            violation_threshold=5000.0,
            critical_threshold=10000.0,
            time_window_seconds=300,
            minimum_samples=10,
            higher_is_better=False,
            description="Claude API response times must be reasonable for interactive BCI applications"
        ))
        
        self.register_sla_threshold(SLAThreshold(
            metric_name="claude_error_rate",
            metric_type=SLAMetricType.ERROR_RATE,
            warning_threshold=0.05,  # 5%
            violation_threshold=0.15,  # 15%
            critical_threshold=0.30,  # 30%
            time_window_seconds=600,
            minimum_samples=10,
            higher_is_better=False,
            description="Claude API error rate must stay low for reliable BCI-AI integration"
        ))
        
        # System resource SLAs
        self.register_sla_threshold(SLAThreshold(
            metric_name="cpu_usage",
            metric_type=SLAMetricType.CUSTOM,
            warning_threshold=70.0,  # %
            violation_threshold=85.0,
            critical_threshold=95.0,
            time_window_seconds=300,
            minimum_samples=30,
            higher_is_better=False,
            description="CPU usage must stay below limits to prevent performance degradation"
        ))
        
        self.register_sla_threshold(SLAThreshold(
            metric_name="memory_usage",
            metric_type=SLAMetricType.CUSTOM,
            warning_threshold=1000.0,  # MB
            violation_threshold=1500.0,
            critical_threshold=2000.0,
            time_window_seconds=300,
            minimum_samples=30,
            higher_is_better=False,
            description="Memory usage must stay within limits to prevent system instability"
        ))

    def register_sla_threshold(self, threshold: SLAThreshold) -> None:
        """Register an SLA threshold for monitoring."""
        with self._lock:
            self.sla_thresholds[threshold.metric_name] = threshold
        
        self.logger.info(f"Registered SLA threshold for {threshold.metric_name}: "
                        f"warning={threshold.warning_threshold}, "
                        f"violation={threshold.violation_threshold}")

    def register_violation_callback(self, callback: Callable[[SLAViolation], None]) -> None:
        """Register callback for SLA violations."""
        self.violation_callbacks.append(callback)
        self.logger.info("Registered SLA violation callback")

    def register_remediation_strategy(self, metric_name: str, 
                                    strategy: Callable[[SLAViolation], bool]) -> None:
        """Register auto-remediation strategy for a metric."""
        self.remediation_strategies[metric_name] = strategy
        self.logger.info(f"Registered remediation strategy for {metric_name}")

    def record_metric_value(self, metric_name: str, value: float, 
                           timestamp: Optional[float] = None,
                           tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value for SLA monitoring."""
        if timestamp is None:
            timestamp = time.time()
        
        tags = tags or {}
        
        with self._lock:
            self.metric_history[metric_name].append({
                'value': value,
                'timestamp': timestamp,
                'tags': tags
            })

    async def start_monitoring(self) -> None:
        """Start SLA monitoring."""
        if self.monitoring_active:
            self.logger.warning("SLA monitoring already active")
            return
        
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("SLA monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop SLA monitoring."""
        self.monitoring_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("SLA monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Main SLA monitoring loop."""
        while self.monitoring_active:
            try:
                await self._check_all_slas()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"SLA monitoring error: {e}", exc_info=True)
                await asyncio.sleep(self.check_interval)

    async def _cleanup_loop(self) -> None:
        """Cleanup old data and resolved violations."""
        while self.monitoring_active:
            try:
                await asyncio.sleep(3600)  # Run every hour
                await self._cleanup_old_data()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"SLA cleanup error: {e}")

    async def _check_all_slas(self) -> None:
        """Check all registered SLA thresholds."""
        current_time = time.time()
        
        for metric_name, threshold in self.sla_thresholds.items():
            try:
                await self._check_metric_sla(metric_name, threshold, current_time)
                self.monitor_stats['checks_performed'] += 1
            except Exception as e:
                self.logger.error(f"Error checking SLA for {metric_name}: {e}")
        
        self.monitor_stats['last_check_time'] = current_time

    async def _check_metric_sla(self, metric_name: str, threshold: SLAThreshold, 
                               current_time: float) -> None:
        """Check SLA for a specific metric."""
        with self._lock:
            if metric_name not in self.metric_history:
                return
            
            history = list(self.metric_history[metric_name])
        
        # Filter to time window
        window_start = current_time - threshold.time_window_seconds
        recent_data = [
            entry for entry in history 
            if entry['timestamp'] >= window_start
        ]
        
        if len(recent_data) < threshold.minimum_samples:
            # Not enough data to evaluate SLA
            return
        
        # Extract values for analysis
        values = [entry['value'] for entry in recent_data]
        
        # Calculate statistics
        mean_value = statistics.mean(values)
        p95_value = self._calculate_percentile(values, 95)
        p99_value = self._calculate_percentile(values, 99)
        
        # Determine which value to use for SLA check based on metric type
        check_value = mean_value
        if threshold.metric_type == SLAMetricType.LATENCY:
            check_value = p95_value  # Use 95th percentile for latency
        elif threshold.metric_type == SLAMetricType.ERROR_RATE:
            check_value = mean_value  # Use mean for error rates
        
        # Determine SLA status
        sla_status = self._determine_sla_status(check_value, threshold)
        
        # Handle violations
        if sla_status in [SLAStatus.WARNING, SLAStatus.VIOLATED, SLAStatus.CRITICAL]:
            await self._handle_sla_violation(
                metric_name, threshold, sla_status, check_value, 
                recent_data, current_time
            )
        else:
            # Check if we can resolve existing violations
            await self._check_violation_resolution(metric_name, current_time)

    def _determine_sla_status(self, value: float, threshold: SLAThreshold) -> SLAStatus:
        """Determine SLA status based on value and thresholds."""
        if threshold.higher_is_better:
            # For metrics where higher values are better (availability, quality)
            if value <= threshold.critical_threshold:
                return SLAStatus.CRITICAL
            elif value <= threshold.violation_threshold:
                return SLAStatus.VIOLATED
            elif value <= threshold.warning_threshold:
                return SLAStatus.WARNING
            else:
                return SLAStatus.HEALTHY
        else:
            # For metrics where lower values are better (latency, error rate)
            if value >= threshold.critical_threshold:
                return SLAStatus.CRITICAL
            elif value >= threshold.violation_threshold:
                return SLAStatus.VIOLATED
            elif value >= threshold.warning_threshold:
                return SLAStatus.WARNING
            else:
                return SLAStatus.HEALTHY

    async def _handle_sla_violation(self, metric_name: str, threshold: SLAThreshold,
                                   sla_status: SLAStatus, actual_value: float,
                                   recent_data: List[Dict[str, Any]], current_time: float) -> None:
        """Handle SLA violation detection."""
        
        # Get threshold value based on status
        threshold_value = threshold.warning_threshold
        if sla_status == SLAStatus.VIOLATED:
            threshold_value = threshold.violation_threshold
        elif sla_status == SLAStatus.CRITICAL:
            threshold_value = threshold.critical_threshold
        
        # Check if this is a new violation or continuation of existing
        violation_key = f"{metric_name}_{sla_status.value}"
        
        if violation_key in self.active_violations:
            # Update existing violation
            violation = self.active_violations[violation_key]
            violation.time_window_end = current_time
            violation.duration_seconds = current_time - violation.time_window_start
            violation.actual_value = actual_value
            violation.sample_count = len(recent_data)
        else:
            # Create new violation
            window_start = current_time - threshold.time_window_seconds
            severity = self._calculate_violation_severity(
                actual_value, threshold_value, threshold.higher_is_better
            )
            
            violation = SLAViolation(
                id=str(uuid.uuid4()),
                metric_name=metric_name,
                violation_type=sla_status,
                threshold_value=threshold_value,
                actual_value=actual_value,
                time_window_start=window_start,
                time_window_end=current_time,
                duration_seconds=threshold.time_window_seconds,
                sample_count=len(recent_data),
                severity=severity,
                description=self._generate_violation_description(
                    metric_name, sla_status, actual_value, threshold_value, threshold
                ),
                tags=threshold.tags.copy()
            )
            
            # Add impact assessment
            violation.impact_assessment = self._assess_violation_impact(
                metric_name, sla_status, actual_value, threshold
            )
            
            self.active_violations[violation_key] = violation
            self.violation_history.append(violation)
            
            self.monitor_stats['violations_detected'] += 1
            
            # Log violation
            self.logger.warning(
                f"SLA {sla_status.value.upper()} for {metric_name}: "
                f"{actual_value} {('below' if threshold.higher_is_better else 'above')} "
                f"threshold {threshold_value}"
            )
            
            # Notify callbacks
            for callback in self.violation_callbacks:
                try:
                    callback(violation)
                except Exception as e:
                    self.logger.error(f"Violation callback failed: {e}")
            
            # Attempt auto-remediation
            if (self.enable_auto_remediation and 
                metric_name in self.remediation_strategies and
                sla_status in [SLAStatus.VIOLATED, SLAStatus.CRITICAL]):
                
                await self._attempt_auto_remediation(violation)

    async def _check_violation_resolution(self, metric_name: str, current_time: float) -> None:
        """Check if violations for a metric can be resolved."""
        violations_to_resolve = []
        
        for violation_key, violation in self.active_violations.items():
            if violation.metric_name == metric_name:
                violations_to_resolve.append(violation_key)
        
        for violation_key in violations_to_resolve:
            violation = self.active_violations[violation_key]
            violation.resolved_at = current_time
            
            del self.active_violations[violation_key]
            self.monitor_stats['violations_resolved'] += 1
            
            self.logger.info(
                f"SLA violation resolved for {violation.metric_name}: "
                f"{violation.violation_type.value} (duration: {violation.duration_seconds:.1f}s)"
            )

    async def _attempt_auto_remediation(self, violation: SLAViolation) -> None:
        """Attempt auto-remediation for an SLA violation."""
        try:
            strategy = self.remediation_strategies.get(violation.metric_name)
            if not strategy:
                return
            
            self.logger.info(f"Attempting auto-remediation for {violation.metric_name}")
            self.monitor_stats['auto_remediations_attempted'] += 1
            
            # Run remediation strategy
            success = await asyncio.to_thread(strategy, violation)
            
            if success:
                self.monitor_stats['auto_remediations_successful'] += 1
                self.logger.info(f"Auto-remediation successful for {violation.metric_name}")
            else:
                self.logger.warning(f"Auto-remediation failed for {violation.metric_name}")
                
        except Exception as e:
            self.logger.error(f"Auto-remediation error for {violation.metric_name}: {e}")

    def _calculate_violation_severity(self, actual_value: float, threshold_value: float,
                                    higher_is_better: bool) -> float:
        """Calculate violation severity (0.0 to 1.0)."""
        try:
            if higher_is_better:
                # For metrics where higher is better
                if actual_value >= threshold_value:
                    return 0.0  # No violation
                
                # Calculate severity based on how far below threshold
                severity = (threshold_value - actual_value) / max(threshold_value, 0.01)
                return min(1.0, max(0.0, severity))
            else:
                # For metrics where lower is better
                if actual_value <= threshold_value:
                    return 0.0  # No violation
                
                # Calculate severity based on how far above threshold
                severity = (actual_value - threshold_value) / max(threshold_value, 0.01)
                return min(1.0, max(0.0, severity))
                
        except Exception:
            return 0.5  # Default moderate severity

    def _generate_violation_description(self, metric_name: str, sla_status: SLAStatus,
                                       actual_value: float, threshold_value: float,
                                       threshold: SLAThreshold) -> str:
        """Generate human-readable violation description."""
        direction = "below" if threshold.higher_is_better else "above"
        unit = getattr(threshold, 'unit', '')
        
        return (f"{metric_name} is {actual_value:.3f}{unit} "
                f"({direction} {sla_status.value} threshold of {threshold_value:.3f}{unit})")

    def _assess_violation_impact(self, metric_name: str, sla_status: SLAStatus,
                                actual_value: float, threshold: SLAThreshold) -> Dict[str, Any]:
        """Assess the impact of an SLA violation."""
        impact_level = "low"
        if sla_status == SLAStatus.CRITICAL:
            impact_level = "critical"
        elif sla_status == SLAStatus.VIOLATED:
            impact_level = "high"
        elif sla_status == SLAStatus.WARNING:
            impact_level = "medium"
        
        # Medical safety assessment for BCI metrics
        medical_impact = False
        if metric_name in ["signal_quality", "neural_data_rate", "decoding_success_rate"]:
            medical_impact = sla_status in [SLAStatus.VIOLATED, SLAStatus.CRITICAL]
        
        # User experience impact
        ux_impact = "minimal"
        if metric_name in ["decoding_latency", "claude_response_time"]:
            if sla_status == SLAStatus.CRITICAL:
                ux_impact = "severe"
            elif sla_status == SLAStatus.VIOLATED:
                ux_impact = "significant"
            elif sla_status == SLAStatus.WARNING:
                ux_impact = "noticeable"
        
        return {
            "impact_level": impact_level,
            "medical_safety_concern": medical_impact,
            "user_experience_impact": ux_impact,
            "business_impact": self._assess_business_impact(metric_name, sla_status),
            "recommended_actions": self._get_recommended_actions(metric_name, sla_status)
        }

    def _assess_business_impact(self, metric_name: str, sla_status: SLAStatus) -> str:
        """Assess business impact of violation."""
        if sla_status == SLAStatus.CRITICAL:
            return "high"
        elif sla_status == SLAStatus.VIOLATED:
            return "medium"
        elif sla_status == SLAStatus.WARNING:
            return "low"
        else:
            return "minimal"

    def _get_recommended_actions(self, metric_name: str, sla_status: SLAStatus) -> List[str]:
        """Get recommended actions for violation."""
        actions = []
        
        if metric_name == "signal_quality":
            if sla_status == SLAStatus.CRITICAL:
                actions.extend([
                    "Check electrode connections immediately",
                    "Verify amplifier functionality",
                    "Consider switching to backup system",
                    "Alert medical personnel"
                ])
            else:
                actions.extend([
                    "Check electrode connections",
                    "Review signal processing parameters",
                    "Monitor for trend continuation"
                ])
        
        elif metric_name == "neural_data_rate":
            actions.extend([
                "Check BCI hardware connections",
                "Verify data acquisition system",
                "Review network connectivity",
                "Consider system restart if persistent"
            ])
        
        elif metric_name == "decoding_latency":
            actions.extend([
                "Review processing algorithms",
                "Check system resource usage",
                "Consider algorithm optimization",
                "Scale computing resources if needed"
            ])
        
        elif metric_name in ["cpu_usage", "memory_usage"]:
            actions.extend([
                "Identify resource-intensive processes",
                "Consider scaling system resources",
                "Optimize algorithms and data structures",
                "Review memory leaks and cleanup procedures"
            ])
        
        elif metric_name.startswith("claude_"):
            actions.extend([
                "Check API connectivity",
                "Review request patterns",
                "Consider request optimization",
                "Monitor Claude service status"
            ])
        
        # Add severity-specific actions
        if sla_status == SLAStatus.CRITICAL:
            actions.append("Escalate to on-call engineer immediately")
        elif sla_status == SLAStatus.VIOLATED:
            actions.append("Investigate root cause within 1 hour")
        
        return actions

    def _calculate_percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile value."""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = (percentile / 100.0) * (len(sorted_values) - 1)
        
        if index.is_integer():
            return sorted_values[int(index)]
        else:
            lower_index = int(index)
            upper_index = min(lower_index + 1, len(sorted_values) - 1)
            weight = index - lower_index
            
            return (sorted_values[lower_index] * (1 - weight) + 
                   sorted_values[upper_index] * weight)

    async def _cleanup_old_data(self) -> None:
        """Clean up old metric history and resolved violations."""
        current_time = time.time()
        retention_seconds = self.history_retention_hours * 3600
        cutoff_time = current_time - retention_seconds
        
        # Clean metric history
        metrics_cleaned = 0
        with self._lock:
            for metric_name, history in self.metric_history.items():
                initial_size = len(history)
                
                # Remove old entries
                while history and history[0]['timestamp'] < cutoff_time:
                    history.popleft()
                
                metrics_cleaned += initial_size - len(history)
        
        # Clean old resolved violations
        initial_violations = len(self.violation_history)
        self.violation_history = [
            v for v in self.violation_history
            if v.resolved_at is None or v.resolved_at >= cutoff_time
        ]
        violations_cleaned = initial_violations - len(self.violation_history)
        
        if metrics_cleaned > 0 or violations_cleaned > 0:
            self.logger.info(f"Cleaned up {metrics_cleaned} old metrics and "
                           f"{violations_cleaned} old violations")

    def generate_sla_report(self, metric_name: str, 
                           start_time: Optional[float] = None,
                           end_time: Optional[float] = None) -> Optional[SLAReport]:
        """Generate SLA compliance report for a metric."""
        current_time = time.time()
        
        if end_time is None:
            end_time = current_time
        if start_time is None:
            start_time = end_time - 24 * 3600  # Last 24 hours
        
        with self._lock:
            if metric_name not in self.metric_history:
                return None
            
            history = list(self.metric_history[metric_name])
        
        # Filter to report period
        period_data = [
            entry for entry in history
            if start_time <= entry['timestamp'] <= end_time
        ]
        
        if not period_data:
            return None
        
        threshold = self.sla_thresholds.get(metric_name)
        if not threshold:
            return None
        
        # Calculate statistics
        values = [entry['value'] for entry in period_data]
        mean_value = statistics.mean(values)
        p50_value = self._calculate_percentile(values, 50)
        p95_value = self._calculate_percentile(values, 95)
        p99_value = self._calculate_percentile(values, 99)
        
        # Calculate compliance
        compliant_count = 0
        for value in values:
            status = self._determine_sla_status(value, threshold)
            if status == SLAStatus.HEALTHY:
                compliant_count += 1
        
        compliance_percentage = (compliant_count / len(values)) * 100 if values else 0
        availability_percentage = compliance_percentage  # For now, same as compliance
        
        # Get violations for this period
        period_violations = [
            v for v in self.violation_history
            if (v.metric_name == metric_name and
                start_time <= v.time_window_start <= end_time)
        ]
        
        # Determine overall SLA status
        if compliance_percentage >= 99:
            sla_status = SLAStatus.HEALTHY
        elif compliance_percentage >= 95:
            sla_status = SLAStatus.WARNING
        else:
            sla_status = SLAStatus.VIOLATED
        
        # Calculate trend
        trend_direction = "stable"
        if len(values) >= 10:
            recent_mean = statistics.mean(values[-5:])
            older_mean = statistics.mean(values[:5])
            
            if abs(recent_mean - older_mean) > (statistics.stdev(values) * 0.5):
                if threshold.higher_is_better:
                    trend_direction = "improving" if recent_mean > older_mean else "degrading"
                else:
                    trend_direction = "improving" if recent_mean < older_mean else "degrading"
        
        # Generate recommendations
        recommendations = self._generate_sla_recommendations(
            metric_name, compliance_percentage, period_violations, trend_direction
        )
        
        return SLAReport(
            metric_name=metric_name,
            report_period_start=start_time,
            report_period_end=end_time,
            total_samples=len(period_data),
            availability_percentage=availability_percentage,
            mean_value=mean_value,
            p50_value=p50_value,
            p95_value=p95_value,
            p99_value=p99_value,
            violations=period_violations,
            compliance_percentage=compliance_percentage,
            sla_status=sla_status,
            trend_direction=trend_direction,
            recommendations=recommendations,
            tags=threshold.tags.copy()
        )

    def _generate_sla_recommendations(self, metric_name: str, compliance_percentage: float,
                                     violations: List[SLAViolation], trend_direction: str) -> List[str]:
        """Generate recommendations for SLA improvement."""
        recommendations = []
        
        if compliance_percentage < 95:
            recommendations.append(f"SLA compliance is below target (95%). Current: {compliance_percentage:.1f}%")
        
        if trend_direction == "degrading":
            recommendations.append("Performance trend is degrading. Investigate root causes.")
        
        if len(violations) > 5:
            recommendations.append("High number of violations detected. Review threshold appropriateness.")
        
        # Metric-specific recommendations
        if metric_name == "signal_quality" and compliance_percentage < 98:
            recommendations.extend([
                "Signal quality issues may affect BCI accuracy",
                "Review electrode maintenance procedures",
                "Consider hardware upgrade if persistent"
            ])
        
        if metric_name == "decoding_latency" and trend_direction == "degrading":
            recommendations.extend([
                "Consider algorithm optimization",
                "Review computational resource allocation",
                "Monitor for memory leaks or resource contention"
            ])
        
        if not recommendations:
            recommendations.append("SLA performance is meeting expectations")
        
        return recommendations

    def get_all_active_violations(self) -> List[SLAViolation]:
        """Get all currently active SLA violations."""
        return list(self.active_violations.values())

    def get_sla_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive SLA data for dashboard."""
        current_time = time.time()
        
        # Generate reports for all metrics
        reports = {}
        for metric_name in self.sla_thresholds.keys():
            report = self.generate_sla_report(metric_name)
            if report:
                reports[metric_name] = {
                    "compliance_percentage": report.compliance_percentage,
                    "sla_status": report.sla_status.value,
                    "trend_direction": report.trend_direction,
                    "total_samples": report.total_samples,
                    "violations_count": len(report.violations),
                    "mean_value": report.mean_value,
                    "p95_value": report.p95_value
                }
        
        # Active violations summary
        violations_by_severity = {"warning": 0, "violated": 0, "critical": 0}
        violations_by_metric = defaultdict(int)
        
        for violation in self.active_violations.values():
            violations_by_severity[violation.violation_type.value] += 1
            violations_by_metric[violation.metric_name] += 1
        
        return {
            "timestamp": current_time,
            "overall_sla_health": self._calculate_overall_sla_health(),
            "metric_reports": reports,
            "active_violations": {
                "total": len(self.active_violations),
                "by_severity": dict(violations_by_severity),
                "by_metric": dict(violations_by_metric),
                "details": [v.to_dict() for v in self.active_violations.values()]
            },
            "monitoring_stats": self.monitor_stats,
            "registered_metrics": list(self.sla_thresholds.keys()),
            "monitoring_active": self.monitoring_active
        }

    def _calculate_overall_sla_health(self) -> str:
        """Calculate overall SLA health status."""
        if not self.sla_thresholds:
            return "unknown"
        
        critical_violations = sum(1 for v in self.active_violations.values() 
                                 if v.violation_type == SLAStatus.CRITICAL)
        violated_violations = sum(1 for v in self.active_violations.values() 
                                 if v.violation_type == SLAStatus.VIOLATED)
        warning_violations = sum(1 for v in self.active_violations.values() 
                               if v.violation_type == SLAStatus.WARNING)
        
        total_metrics = len(self.sla_thresholds)
        
        if critical_violations > 0:
            return "critical"
        elif violated_violations > total_metrics * 0.3:  # More than 30% in violation
            return "poor"
        elif violated_violations > 0 or warning_violations > total_metrics * 0.5:
            return "degraded"
        else:
            return "healthy"


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import random
    
    async def demo_sla_monitor():
        print("BCI SLA Monitor Demo")
        print("=" * 40)
        
        # Create SLA monitor
        sla_monitor = SLAMonitor(check_interval=5.0)
        
        # Register violation callback
        def violation_callback(violation: SLAViolation):
            print(f"🚨 SLA VIOLATION: {violation.description}")
            print(f"   Severity: {violation.severity:.2f}, Impact: {violation.impact_assessment['impact_level']}")
            if violation.impact_assessment.get('medical_safety_concern'):
                print("   ⚠️  MEDICAL SAFETY CONCERN!")
            print()
        
        sla_monitor.register_violation_callback(violation_callback)
        
        # Register remediation strategy example
        def cpu_remediation_strategy(violation: SLAViolation) -> bool:
            print(f"🔧 Attempting remediation for {violation.metric_name}")
            # In real implementation, this would restart services, scale resources, etc.
            return random.choice([True, False])
        
        sla_monitor.register_remediation_strategy("cpu_usage", cpu_remediation_strategy)
        
        # Start monitoring
        await sla_monitor.start_monitoring()
        
        print("SLA monitoring started. Simulating metrics...")
        
        # Simulate metrics for demonstration
        for i in range(60):  # Run for 5 minutes at 5-second intervals
            # Simulate various metrics with occasional violations
            
            # Signal quality - occasionally drops
            signal_quality = max(0.1, min(1.0, 0.85 + random.normalvariate(0, 0.1)))
            if i > 20 and i < 35:  # Inject quality degradation
                signal_quality *= 0.6
            
            sla_monitor.record_metric_value("signal_quality", signal_quality)
            
            # Neural data rate - usually stable
            data_rate = max(50, random.normalvariate(250, 15))
            if i > 40 and i < 50:  # Inject data rate drop
                data_rate *= 0.4
            
            sla_monitor.record_metric_value("neural_data_rate", data_rate)
            
            # Decoding latency - gradually increases
            base_latency = 80 + (i * 2)  # Gradual increase
            latency = max(20, random.normalvariate(base_latency, 15))
            sla_monitor.record_metric_value("decoding_latency", latency)
            
            # CPU usage - spikes occasionally
            cpu_usage = max(10, min(100, random.normalvariate(45, 10)))
            if i % 15 == 0:  # Periodic spikes
                cpu_usage = min(100, cpu_usage * 2)
            
            sla_monitor.record_metric_value("cpu_usage", cpu_usage)
            
            # Claude response time
            claude_time = max(500, random.normalvariate(1500, 400))
            if random.random() < 0.1:  # 10% chance of slow response
                claude_time *= 3
            
            sla_monitor.record_metric_value("claude_response_time", claude_time)
            
            await asyncio.sleep(0.1)  # Speed up simulation
        
        print("\nGenerating SLA reports...")
        
        # Generate reports for key metrics
        for metric_name in ["signal_quality", "decoding_latency", "cpu_usage"]:
            report = sla_monitor.generate_sla_report(metric_name)
            if report:
                print(f"\n--- SLA Report: {metric_name} ---")
                print(f"Compliance: {report.compliance_percentage:.1f}%")
                print(f"Status: {report.sla_status.value}")
                print(f"Trend: {report.trend_direction}")
                print(f"Mean: {report.mean_value:.2f}")
                print(f"P95: {report.p95_value:.2f}")
                print(f"Violations: {len(report.violations)}")
                print(f"Recommendations:")
                for rec in report.recommendations:
                    print(f"  - {rec}")
        
        # Get dashboard data
        dashboard_data = sla_monitor.get_sla_dashboard_data()
        print(f"\n--- Overall SLA Health: {dashboard_data['overall_sla_health'].upper()} ---")
        print(f"Active violations: {dashboard_data['active_violations']['total']}")
        print(f"Monitoring stats: {dashboard_data['monitoring_stats']}")
        
        # Stop monitoring
        await sla_monitor.stop_monitoring()
        print("\nSLA monitoring stopped")
    
    asyncio.run(demo_sla_monitor())