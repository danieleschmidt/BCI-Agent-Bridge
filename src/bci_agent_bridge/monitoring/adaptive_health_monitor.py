"""
Advanced adaptive health monitoring system with machine learning-based anomaly detection.
Implements predictive maintenance and self-healing capabilities for BCI systems.
"""

import asyncio
import time
import logging
import threading
import statistics
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import uuid
from datetime import datetime, timedelta

# Security and audit imports
try:
    from ..security.audit_logger import security_logger, SecurityEvent
    _SECURITY_AVAILABLE = True
except ImportError:
    _SECURITY_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthSeverity(Enum):
    """Health issue severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AdaptiveThreshold:
    """Adaptive threshold that learns from historical data."""
    
    def __init__(self, initial_value: float, adaptation_rate: float = 0.1, 
                 max_deviation: float = 0.5):
        self.value = initial_value
        self.initial_value = initial_value
        self.adaptation_rate = adaptation_rate
        self.max_deviation = max_deviation
        self.history = deque(maxlen=1000)
        self.last_update = time.time()
        
    def update(self, measurement: float, timestamp: float = None) -> None:
        """Update threshold based on new measurement."""
        if timestamp is None:
            timestamp = time.time()
            
        self.history.append((measurement, timestamp))
        
        # Adapt threshold if enough data points
        if len(self.history) >= 50:
            recent_values = [val for val, _ in list(self.history)[-50:]]
            baseline = np.percentile(recent_values, 85)  # 85th percentile
            
            # Gradual adaptation with bounds
            target_threshold = baseline * 1.2  # 20% above baseline
            max_change = self.initial_value * self.max_deviation
            
            new_threshold = self.value + (target_threshold - self.value) * self.adaptation_rate
            
            # Bound the threshold change
            self.value = max(
                self.initial_value - max_change,
                min(self.initial_value + max_change, new_threshold)
            )
            
        self.last_update = timestamp
    
    def is_exceeded(self, value: float) -> bool:
        """Check if value exceeds adaptive threshold."""
        return value > self.value
    
    def get_status(self) -> Dict[str, Any]:
        """Get threshold status and statistics."""
        return {
            "current_threshold": self.value,
            "initial_threshold": self.initial_value,
            "history_size": len(self.history),
            "last_update": self.last_update,
            "adaptation_rate": self.adaptation_rate
        }


@dataclass
class HealthAlert:
    """Enhanced health alert with context and recommendations."""
    id: str
    component: str
    metric: str
    severity: HealthSeverity
    value: float
    threshold: float
    message: str
    timestamp: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    auto_remediation: Optional[str] = None
    acknowledged: bool = False
    resolved: bool = False
    resolution_time: Optional[float] = None


class AnomalyDetector:
    """Statistical anomaly detection for health metrics."""
    
    def __init__(self, window_size: int = 100, sensitivity: float = 2.5):
        self.window_size = window_size
        self.sensitivity = sensitivity  # Number of standard deviations
        self.data_buffer = deque(maxlen=window_size)
        self.baseline_mean = 0.0
        self.baseline_std = 1.0
        
    def add_sample(self, value: float) -> bool:
        """Add sample and return True if anomaly detected."""
        self.data_buffer.append(value)
        
        if len(self.data_buffer) < 10:  # Need minimum samples
            return False
            
        # Update baseline statistics
        values = list(self.data_buffer)
        self.baseline_mean = np.mean(values[:-1])  # Exclude current sample
        self.baseline_std = np.std(values[:-1]) + 1e-8  # Avoid division by zero
        
        # Check if current value is anomalous
        z_score = abs(value - self.baseline_mean) / self.baseline_std
        return z_score > self.sensitivity
    
    def get_z_score(self, value: float) -> float:
        """Get z-score for a value."""
        return abs(value - self.baseline_mean) / self.baseline_std if self.baseline_std > 0 else 0.0


class AdaptiveHealthMonitor:
    """
    Advanced health monitoring system with adaptive thresholds, 
    anomaly detection, and predictive maintenance capabilities.
    """
    
    def __init__(self, check_interval: int = 30, 
                 base_thresholds: Optional[Dict[str, float]] = None,
                 enable_auto_remediation: bool = True):
        self.check_interval = check_interval
        self.enable_auto_remediation = enable_auto_remediation
        
        # Initialize adaptive thresholds
        default_thresholds = {
            'neural_processing_latency': 100.0,  # ms
            'signal_quality': 0.3,
            'buffer_utilization': 90.0,  # %
            'decoder_confidence': 0.5,
            'memory_usage': 85.0,  # %
            'cpu_usage': 80.0,  # %
            'error_rate': 5.0,  # %
            'response_time': 1000.0  # ms
        }
        
        threshold_config = base_thresholds or default_thresholds
        self.adaptive_thresholds = {
            name: AdaptiveThreshold(value, adaptation_rate=0.05)
            for name, value in threshold_config.items()
        }
        
        # Anomaly detectors for each metric
        self.anomaly_detectors = {
            name: AnomalyDetector(window_size=100, sensitivity=2.0)
            for name in threshold_config.keys()
        }
        
        # State management
        self.is_monitoring = False
        self.metrics_history = deque(maxlen=10000)
        self.active_alerts = {}  # alert_id -> HealthAlert
        self.resolved_alerts = deque(maxlen=1000)
        self.health_callbacks = []
        
        # Performance tracking
        self.component_health = defaultdict(lambda: HealthSeverity.INFO)
        self.last_health_check = {}
        self.system_baseline = {}
        
        # Auto-remediation registry
        self.remediation_actions = {}
        self._register_default_remediations()
        
        # Monitoring thread
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        logger.info("Adaptive Health Monitor initialized with enhanced capabilities")
    
    def _register_default_remediations(self) -> None:
        """Register default auto-remediation actions."""
        self.remediation_actions = {
            'high_memory_usage': {
                'action': 'clear_buffers',
                'description': 'Clear non-essential data buffers',
                'max_attempts': 3,
                'cooldown': 300  # seconds
            },
            'high_buffer_utilization': {
                'action': 'increase_buffer_size',
                'description': 'Increase buffer capacity if possible',
                'max_attempts': 2,
                'cooldown': 600
            },
            'low_signal_quality': {
                'action': 'recalibrate_filters',
                'description': 'Adjust signal processing filters',
                'max_attempts': 5,
                'cooldown': 120
            },
            'high_error_rate': {
                'action': 'restart_components',
                'description': 'Restart problematic components',
                'max_attempts': 2,
                'cooldown': 900
            }
        }
    
    def add_health_callback(self, callback: Callable[[HealthAlert], None]) -> None:
        """Add callback function for health alerts."""
        self.health_callbacks.append(callback)
    
    async def start_monitoring(self) -> None:
        """Start the adaptive health monitoring system."""
        if self.is_monitoring:
            logger.warning("Health monitoring already active")
            return
        
        self.is_monitoring = True
        self.stop_event.clear()
        
        # Start monitoring in background thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info("Adaptive health monitoring started")
        
        # Log monitoring start
        if _SECURITY_AVAILABLE:
            security_logger.log_security_event(
                event_type=SecurityEvent.CONFIGURATION_CHANGE,
                resource="health_monitoring",
                action="start",
                details={"adaptive_thresholds": True, "auto_remediation": self.enable_auto_remediation},
                risk_score=2
            )
    
    def stop_monitoring(self) -> None:
        """Stop the health monitoring system."""
        if not self.is_monitoring:
            return
        
        self.is_monitoring = False
        self.stop_event.set()
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5.0)
        
        logger.info("Adaptive health monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop running in background thread."""
        while not self.stop_event.is_set():
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                
                # Process each metric
                for metric_name, value in metrics.items():
                    self._process_metric(metric_name, value)
                
                # Store metrics in history
                self.metrics_history.append({
                    'timestamp': time.time(),
                    'metrics': metrics.copy()
                })
                
                # Check for patterns and trends
                self._analyze_health_trends()
                
                # Perform auto-remediation if enabled
                if self.enable_auto_remediation:
                    self._perform_auto_remediation()
                
                # Update adaptive thresholds
                self._update_adaptive_thresholds(metrics)
                
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")
                if _SECURITY_AVAILABLE:
                    security_logger.log_system_error(
                        component="health_monitor",
                        error_type="monitoring_error",
                        error_message=str(e)
                    )
            
            # Wait for next check
            self.stop_event.wait(self.check_interval)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect comprehensive system metrics."""
        import psutil
        import os
        
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'disk_usage': disk.percent,
                'memory_available_mb': memory.available / (1024 * 1024),
                'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
            }
            
            # Add BCI-specific metrics (would be provided by actual BCI system)
            metrics.update({
                'neural_processing_latency': self._get_neural_processing_latency(),
                'signal_quality': self._get_signal_quality(),
                'buffer_utilization': self._get_buffer_utilization(),
                'decoder_confidence': self._get_decoder_confidence(),
                'error_rate': self._get_error_rate(),
                'response_time': self._get_response_time()
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
            return {}
    
    def _get_neural_processing_latency(self) -> float:
        """Get current neural processing latency (simulated)."""
        # In real implementation, this would query the BCI bridge
        import random
        base_latency = 45.0  # ms
        return base_latency + random.gauss(0, 10)
    
    def _get_signal_quality(self) -> float:
        """Get current signal quality (simulated)."""
        import random
        return max(0.0, min(1.0, random.gauss(0.7, 0.15)))
    
    def _get_buffer_utilization(self) -> float:
        """Get buffer utilization percentage (simulated)."""
        import random
        return max(0.0, min(100.0, random.gauss(45.0, 15.0)))
    
    def _get_decoder_confidence(self) -> float:
        """Get decoder confidence (simulated)."""
        import random
        return max(0.0, min(1.0, random.gauss(0.75, 0.1)))
    
    def _get_error_rate(self) -> float:
        """Get current error rate percentage (simulated)."""
        import random
        return max(0.0, random.exponential(2.0))
    
    def _get_response_time(self) -> float:
        """Get current response time in ms (simulated)."""
        import random
        return max(0.0, random.gauss(250.0, 50.0))
    
    def _process_metric(self, metric_name: str, value: float) -> None:
        """Process individual metric with adaptive thresholds and anomaly detection."""
        timestamp = time.time()
        
        # Update adaptive threshold
        if metric_name in self.adaptive_thresholds:
            threshold = self.adaptive_thresholds[metric_name]
            threshold.update(value, timestamp)
            
            # Check threshold breach
            if threshold.is_exceeded(value):
                self._create_threshold_alert(metric_name, value, threshold.value)
        
        # Anomaly detection
        if metric_name in self.anomaly_detectors:
            detector = self.anomaly_detectors[metric_name]
            is_anomaly = detector.add_sample(value)
            
            if is_anomaly:
                z_score = detector.get_z_score(value)
                self._create_anomaly_alert(metric_name, value, z_score)
        
        # Update component health status
        self._update_component_health(metric_name, value)
    
    def _create_threshold_alert(self, metric_name: str, value: float, threshold: float) -> None:
        """Create alert for threshold breach."""
        severity = self._determine_severity(metric_name, value, threshold)
        
        alert = HealthAlert(
            id=str(uuid.uuid4()),
            component="system",
            metric=metric_name,
            severity=severity,
            value=value,
            threshold=threshold,
            message=f"{metric_name} exceeded threshold: {value:.2f} > {threshold:.2f}",
            context={
                "metric_type": "threshold",
                "percentage_over": ((value - threshold) / threshold) * 100
            },
            recommendations=self._get_recommendations(metric_name, "threshold"),
            auto_remediation=self._get_auto_remediation(metric_name)
        )
        
        self._register_alert(alert)
    
    def _create_anomaly_alert(self, metric_name: str, value: float, z_score: float) -> None:
        """Create alert for statistical anomaly."""
        severity = HealthSeverity.WARNING if z_score < 3.0 else HealthSeverity.ERROR
        
        alert = HealthAlert(
            id=str(uuid.uuid4()),
            component="system",
            metric=metric_name,
            severity=severity,
            value=value,
            threshold=0.0,  # No threshold for anomalies
            message=f"Statistical anomaly detected in {metric_name}: z-score {z_score:.2f}",
            context={
                "metric_type": "anomaly",
                "z_score": z_score,
                "baseline_mean": self.anomaly_detectors[metric_name].baseline_mean,
                "baseline_std": self.anomaly_detectors[metric_name].baseline_std
            },
            recommendations=self._get_recommendations(metric_name, "anomaly"),
            auto_remediation=self._get_auto_remediation(metric_name)
        )
        
        self._register_alert(alert)
    
    def _determine_severity(self, metric_name: str, value: float, threshold: float) -> HealthSeverity:
        """Determine alert severity based on metric and breach magnitude."""
        breach_percentage = ((value - threshold) / threshold) * 100
        
        critical_metrics = ['neural_processing_latency', 'error_rate']
        
        if metric_name in critical_metrics:
            if breach_percentage > 50:
                return HealthSeverity.CRITICAL
            elif breach_percentage > 25:
                return HealthSeverity.ERROR
            else:
                return HealthSeverity.WARNING
        else:
            if breach_percentage > 100:
                return HealthSeverity.ERROR
            elif breach_percentage > 50:
                return HealthSeverity.WARNING
            else:
                return HealthSeverity.INFO
    
    def _get_recommendations(self, metric_name: str, alert_type: str) -> List[str]:
        """Get context-aware recommendations for metric issues."""
        recommendations = {
            'neural_processing_latency': [
                "Check BCI device connection quality",
                "Reduce signal processing complexity",
                "Optimize decoder algorithms",
                "Increase system priority for BCI processes"
            ],
            'signal_quality': [
                "Check electrode connections",
                "Reduce environmental electromagnetic interference",
                "Recalibrate signal processing filters",
                "Check for motion artifacts"
            ],
            'buffer_utilization': [
                "Increase buffer size if memory allows",
                "Optimize data processing pipeline",
                "Check for memory leaks",
                "Reduce data retention period"
            ],
            'decoder_confidence': [
                "Recalibrate the neural decoder",
                "Check signal quality",
                "Verify training data quality",
                "Consider adaptive threshold adjustment"
            ],
            'memory_usage': [
                "Clear unnecessary data buffers",
                "Check for memory leaks",
                "Optimize data structures",
                "Restart memory-intensive components"
            ],
            'cpu_usage': [
                "Reduce processing complexity",
                "Optimize algorithms",
                "Check for infinite loops",
                "Scale to additional processing units"
            ],
            'error_rate': [
                "Check system logs for error patterns",
                "Verify input data quality",
                "Restart failing components",
                "Update error handling logic"
            ]
        }
        
        return recommendations.get(metric_name, ["Monitor the situation", "Check system logs"])
    
    def _get_auto_remediation(self, metric_name: str) -> Optional[str]:
        """Get auto-remediation action for metric."""
        remediation_map = {
            'memory_usage': 'high_memory_usage',
            'buffer_utilization': 'high_buffer_utilization',
            'signal_quality': 'low_signal_quality',
            'error_rate': 'high_error_rate'
        }
        
        return remediation_map.get(metric_name)
    
    def _register_alert(self, alert: HealthAlert) -> None:
        """Register and process new health alert."""
        self.active_alerts[alert.id] = alert
        
        # Log alert
        logger.log(
            getattr(logging, alert.severity.value.upper(), logging.INFO),
            f"Health Alert [{alert.severity.value}]: {alert.message}"
        )
        
        # Security logging for critical alerts
        if _SECURITY_AVAILABLE and alert.severity in [HealthSeverity.CRITICAL, HealthSeverity.EMERGENCY]:
            security_logger.log_suspicious_activity(
                activity_type="critical_health_alert",
                details={
                    "alert_id": alert.id,
                    "metric": alert.metric,
                    "value": alert.value,
                    "severity": alert.severity.value
                },
                risk_score=8 if alert.severity == HealthSeverity.CRITICAL else 10
            )
        
        # Notify callbacks
        for callback in self.health_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Health callback error: {e}")
    
    def _update_component_health(self, metric_name: str, value: float) -> None:
        """Update overall component health status."""
        component_map = {
            'neural_processing_latency': 'neural_processor',
            'signal_quality': 'signal_acquisition',
            'buffer_utilization': 'data_management',
            'decoder_confidence': 'neural_decoder',
            'memory_usage': 'system_resources',
            'cpu_usage': 'system_resources',
            'error_rate': 'system_stability'
        }
        
        component = component_map.get(metric_name, 'unknown')
        self.last_health_check[component] = time.time()
        
        # Determine component health based on metric value and thresholds
        if metric_name in self.adaptive_thresholds:
            threshold = self.adaptive_thresholds[metric_name]
            if threshold.is_exceeded(value):
                if ((value - threshold.value) / threshold.value) > 0.5:
                    self.component_health[component] = HealthSeverity.ERROR
                else:
                    self.component_health[component] = HealthSeverity.WARNING
            else:
                self.component_health[component] = HealthSeverity.INFO
    
    def _analyze_health_trends(self) -> None:
        """Analyze health trends for predictive maintenance."""
        if len(self.metrics_history) < 20:  # Need sufficient data
            return
        
        # Get recent metrics for trend analysis
        recent_metrics = list(self.metrics_history)[-20:]
        
        for metric_name in self.adaptive_thresholds.keys():
            values = [m['metrics'].get(metric_name, 0) for m in recent_metrics]
            timestamps = [m['timestamp'] for m in recent_metrics]
            
            if len(values) >= 10:
                trend_direction, confidence = self._calculate_trend(values, timestamps)
                
                if trend_direction == "degrading" and confidence > 0.7:
                    self._create_predictive_alert(metric_name, values, trend_direction, confidence)
    
    def _calculate_trend(self, values: List[float], timestamps: List[float]) -> Tuple[str, float]:
        """Calculate trend direction and confidence."""
        if len(values) < 5:
            return "stable", 0.0
        
        try:
            # Linear regression for trend
            x = np.array(timestamps) - timestamps[0]  # Normalize
            y = np.array(values)
            
            # Calculate slope
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            
            # Calculate correlation for confidence
            correlation = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
            confidence = abs(correlation)
            
            # Determine trend direction
            if abs(slope) < 0.001:
                return "stable", confidence
            elif slope > 0:
                return "improving" if slope > 0 else "degrading", confidence
            else:
                return "degrading", confidence
                
        except Exception as e:
            logger.error(f"Trend calculation error: {e}")
            return "stable", 0.0
    
    def _create_predictive_alert(self, metric_name: str, values: List[float], 
                               trend_direction: str, confidence: float) -> None:
        """Create predictive maintenance alert."""
        current_value = values[-1]
        
        alert = HealthAlert(
            id=str(uuid.uuid4()),
            component="predictive_maintenance",
            metric=metric_name,
            severity=HealthSeverity.WARNING,
            value=current_value,
            threshold=0.0,
            message=f"Predictive alert: {metric_name} trending {trend_direction} (confidence: {confidence:.2f})",
            context={
                "metric_type": "predictive",
                "trend_direction": trend_direction,
                "confidence": confidence,
                "prediction_window": "next 30 minutes"
            },
            recommendations=[
                f"Monitor {metric_name} closely",
                "Consider preventive maintenance",
                "Review system configuration"
            ]
        )
        
        self._register_alert(alert)
    
    def _update_adaptive_thresholds(self, metrics: Dict[str, float]) -> None:
        """Update all adaptive thresholds with current metrics."""
        for metric_name, value in metrics.items():
            if metric_name in self.adaptive_thresholds:
                self.adaptive_thresholds[metric_name].update(value)
    
    def _perform_auto_remediation(self) -> None:
        """Perform auto-remediation actions for active alerts."""
        if not self.enable_auto_remediation:
            return
        
        for alert in list(self.active_alerts.values()):
            if (alert.auto_remediation and 
                not alert.acknowledged and 
                not alert.resolved and
                alert.auto_remediation in self.remediation_actions):
                
                self._execute_remediation(alert)
    
    def _execute_remediation(self, alert: HealthAlert) -> None:
        """Execute auto-remediation action for an alert."""
        remediation_config = self.remediation_actions[alert.auto_remediation]
        action = remediation_config['action']
        
        try:
            logger.info(f"Executing auto-remediation '{action}' for alert {alert.id}")
            
            # Execute the remediation action (these would be real implementations)
            success = False
            if action == 'clear_buffers':
                success = self._clear_buffers()
            elif action == 'increase_buffer_size':
                success = self._increase_buffer_size()
            elif action == 'recalibrate_filters':
                success = self._recalibrate_filters()
            elif action == 'restart_components':
                success = self._restart_components()
            
            if success:
                logger.info(f"Auto-remediation '{action}' completed successfully")
                alert.resolved = True
                alert.resolution_time = time.time()
                self.resolved_alerts.append(alert)
                del self.active_alerts[alert.id]
                
                # Log successful remediation
                if _SECURITY_AVAILABLE:
                    security_logger.log_security_event(
                        event_type=SecurityEvent.CONFIGURATION_CHANGE,
                        resource="auto_remediation",
                        action="execute",
                        details={
                            "alert_id": alert.id,
                            "remediation_action": action,
                            "success": True
                        },
                        risk_score=3
                    )
            else:
                logger.warning(f"Auto-remediation '{action}' failed")
                
        except Exception as e:
            logger.error(f"Auto-remediation execution failed: {e}")
    
    def _clear_buffers(self) -> bool:
        """Clear non-essential data buffers."""
        # Implementation would clear actual system buffers
        logger.info("Clearing non-essential data buffers")
        return True
    
    def _increase_buffer_size(self) -> bool:
        """Increase buffer capacity if possible."""
        # Implementation would increase actual buffer sizes
        logger.info("Increasing buffer capacity")
        return True
    
    def _recalibrate_filters(self) -> bool:
        """Recalibrate signal processing filters."""
        # Implementation would recalibrate actual filters
        logger.info("Recalibrating signal processing filters")
        return True
    
    def _restart_components(self) -> bool:
        """Restart problematic components."""
        # Implementation would restart actual components
        logger.info("Restarting system components")
        return True
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        active_alert_count = len(self.active_alerts)
        critical_alerts = sum(1 for alert in self.active_alerts.values() 
                            if alert.severity == HealthSeverity.CRITICAL)
        
        overall_health = HealthSeverity.INFO
        if critical_alerts > 0:
            overall_health = HealthSeverity.CRITICAL
        elif active_alert_count > 5:
            overall_health = HealthSeverity.ERROR
        elif active_alert_count > 0:
            overall_health = HealthSeverity.WARNING
        
        return {
            "overall_health": overall_health.value,
            "active_alerts": active_alert_count,
            "critical_alerts": critical_alerts,
            "component_health": {k: v.value for k, v in self.component_health.items()},
            "adaptive_thresholds": {k: v.get_status() for k, v in self.adaptive_thresholds.items()},
            "is_monitoring": self.is_monitoring,
            "auto_remediation_enabled": self.enable_auto_remediation,
            "metrics_history_size": len(self.metrics_history),
            "last_check": time.time()
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get all active alerts."""
        return [
            {
                "id": alert.id,
                "component": alert.component,
                "metric": alert.metric,
                "severity": alert.severity.value,
                "value": alert.value,
                "threshold": alert.threshold,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "context": alert.context,
                "recommendations": alert.recommendations,
                "acknowledged": alert.acknowledged,
                "auto_remediation": alert.auto_remediation
            }
            for alert in self.active_alerts.values()
        ]
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            self.active_alerts[alert_id].acknowledged = True
            logger.info(f"Alert {alert_id} acknowledged by {acknowledged_by}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Manually resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolution_time = time.time()
            self.resolved_alerts.append(alert)
            del self.active_alerts[alert_id]
            logger.info(f"Alert {alert_id} resolved by {resolved_by}")
            return True
        return False


# Factory function for easy instantiation
def create_adaptive_health_monitor(config: Optional[Dict[str, Any]] = None) -> AdaptiveHealthMonitor:
    """Create and configure an adaptive health monitor."""
    config = config or {}
    
    return AdaptiveHealthMonitor(
        check_interval=config.get('check_interval', 30),
        base_thresholds=config.get('thresholds'),
        enable_auto_remediation=config.get('auto_remediation', True)
    )