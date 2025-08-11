"""
Enterprise-grade health monitoring system for BCI bridge components.
Includes predictive alerts, trend analysis, and comprehensive health tracking.
"""

import asyncio
import time
import logging
import threading
import statistics
import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import json
import uuid


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    CRITICAL = "critical"


class TrendDirection(Enum):
    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    VOLATILE = "volatile"


@dataclass
class HealthMetric:
    """Individual health metric with trend analysis."""
    name: str
    value: float
    timestamp: float
    status: HealthStatus
    trend_direction: TrendDirection
    trend_confidence: float  # 0.0 to 1.0
    prediction: Optional[float] = None
    prediction_confidence: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    message: str
    last_check: float
    duration_ms: float
    details: Dict[str, Any] = field(default_factory=dict)
    metrics: List[HealthMetric] = field(default_factory=list)
    trend_analysis: Dict[str, Any] = field(default_factory=dict)
    prediction: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert health check to dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "message": self.message,
            "last_check": self.last_check,
            "duration_ms": self.duration_ms,
            "details": self.details,
            "metrics": [
                {
                    "name": m.name,
                    "value": m.value,
                    "status": m.status.value,
                    "trend": m.trend_direction.value,
                    "trend_confidence": m.trend_confidence,
                    "prediction": m.prediction,
                    "prediction_confidence": m.prediction_confidence,
                    "tags": m.tags
                } for m in self.metrics
            ],
            "trend_analysis": self.trend_analysis,
            "prediction": self.prediction
        }


@dataclass
class PredictiveAlert:
    """Predictive alert based on health trends."""
    id: str
    component: str
    metric_name: str
    predicted_status: HealthStatus
    confidence: float
    predicted_time: float  # When the issue is predicted to occur
    current_trend: TrendDirection
    recommendation: str
    created_at: float = field(default_factory=time.time)


class HealthTrendAnalyzer:
    """Analyzes health trends and makes predictions."""
    
    def __init__(self, history_size: int = 100, min_samples: int = 10):
        self.history_size = history_size
        self.min_samples = min_samples
        
    def analyze_trend(self, values: List[float], timestamps: List[float]) -> Tuple[TrendDirection, float]:
        """Analyze trend direction and confidence."""
        if len(values) < self.min_samples:
            return TrendDirection.STABLE, 0.0
            
        try:
            # Calculate linear regression
            x = np.array(timestamps)
            y = np.array(values)
            
            # Normalize timestamps to avoid numerical issues
            x = x - x[0]
            
            # Calculate slope
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            
            # Calculate correlation coefficient for confidence
            r = np.corrcoef(x, y)[0, 1] if len(x) > 1 else 0.0
            confidence = abs(r)
            
            # Calculate volatility
            volatility = np.std(y) / (np.mean(y) + 1e-8)  # Avoid division by zero
            
            # Determine trend direction
            if volatility > 0.5:  # High volatility threshold
                return TrendDirection.VOLATILE, confidence
            elif abs(slope) < 0.001:  # Stable threshold
                return TrendDirection.STABLE, confidence
            elif slope > 0:
                return TrendDirection.IMPROVING, confidence
            else:
                return TrendDirection.DEGRADING, confidence
                
        except Exception as e:
            logging.getLogger(__name__).warning(f"Trend analysis error: {e}")
            return TrendDirection.STABLE, 0.0
    
    def predict_future_value(self, values: List[float], timestamps: List[float], 
                           future_time: float) -> Tuple[Optional[float], Optional[float]]:
        """Predict future value and confidence."""
        if len(values) < self.min_samples:
            return None, None
            
        try:
            x = np.array(timestamps)
            y = np.array(values)
            
            # Normalize timestamps
            x = x - x[0]
            future_x = future_time - timestamps[0]
            
            # Linear regression prediction
            n = len(x)
            slope = (n * np.sum(x * y) - np.sum(x) * np.sum(y)) / (n * np.sum(x**2) - np.sum(x)**2)
            intercept = np.mean(y) - slope * np.mean(x)
            
            predicted_value = slope * future_x + intercept
            
            # Calculate prediction confidence based on R²
            y_pred = slope * x + intercept
            ss_res = np.sum((y - y_pred)**2)
            ss_tot = np.sum((y - np.mean(y))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
            
            return predicted_value, max(0.0, min(1.0, r_squared))
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Prediction error: {e}")
            return None, None


class HealthMonitor:
    """
    Enterprise-grade health monitor with predictive analytics and trend analysis.
    """
    
    def __init__(self, check_interval: float = 30.0, enable_predictions: bool = True):
        self.check_interval = check_interval
        self.enable_predictions = enable_predictions
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.health_checks: Dict[str, Callable[[], HealthCheck]] = {}
        self.last_results: Dict[str, HealthCheck] = {}
        self.recovery_strategies: Dict[str, Callable] = {}
        
        # Historical data for trend analysis
        self.history_size = 1000
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_size))
        self.metric_history: Dict[str, Dict[str, deque]] = defaultdict(lambda: defaultdict(lambda: deque(maxlen=self.history_size)))
        
        # Predictive analytics
        self.trend_analyzer = HealthTrendAnalyzer()
        self.predictive_alerts: List[PredictiveAlert] = []
        self.prediction_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Monitoring control
        self.monitoring_active = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._prediction_task: Optional[asyncio.Task] = None
        
        # Performance metrics
        self.metrics = {
            'total_checks': 0,
            'failed_checks': 0,
            'predictions_made': 0,
            'predictions_accurate': 0,
            'trend_changes_detected': 0,
            'recovery_attempts': 0,
            'recovery_successes': 0
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Setup default prediction thresholds
        self._setup_default_thresholds()

    def _setup_default_thresholds(self):
        """Setup default prediction thresholds."""
        self.prediction_thresholds = {
            'signal_quality': {
                'degraded': 0.5,
                'unhealthy': 0.3,
                'critical': 0.1
            },
            'cpu_usage': {
                'degraded': 70.0,
                'unhealthy': 85.0,
                'critical': 95.0
            },
            'memory_usage': {
                'degraded': 1000.0,  # MB
                'unhealthy': 1500.0,
                'critical': 2000.0
            },
            'response_time': {
                'degraded': 1000.0,  # ms
                'unhealthy': 2000.0,
                'critical': 5000.0
            }
        }

    def register_health_check(self, name: str, check_func: Callable[[], HealthCheck]) -> None:
        """Register a health check function."""
        self.health_checks[name] = check_func
        self.logger.info(f"Registered health check: {name}")

    def register_recovery_strategy(self, component_name: str, recovery_func: Callable) -> None:
        """Register a recovery strategy for a component."""
        self.recovery_strategies[component_name] = recovery_func
        self.logger.info(f"Registered recovery strategy for: {component_name}")

    def set_prediction_threshold(self, metric_name: str, thresholds: Dict[str, float]) -> None:
        """Set prediction thresholds for a metric."""
        self.prediction_thresholds[metric_name] = thresholds
        self.logger.info(f"Set prediction thresholds for {metric_name}: {thresholds}")

    async def start_monitoring(self) -> None:
        """Start comprehensive health monitoring with predictions."""
        if self.monitoring_active:
            self.logger.warning("Health monitoring already active")
            return
        
        self.monitoring_active = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        
        if self.enable_predictions:
            self._prediction_task = asyncio.create_task(self._prediction_loop())
        
        self.logger.info("Enhanced health monitoring started")

    async def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self.monitoring_active = False
        
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        if self._prediction_task:
            self._prediction_task.cancel()
            try:
                await self._prediction_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("Health monitoring stopped")

    async def _monitoring_loop(self) -> None:
        """Enhanced monitoring loop with trend analysis."""
        while self.monitoring_active:
            try:
                await self.run_all_checks()
                await self.handle_unhealthy_components()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}", exc_info=True)
                await asyncio.sleep(5.0)

    async def _prediction_loop(self) -> None:
        """Prediction loop for generating predictive alerts."""
        while self.monitoring_active:
            try:
                await self._generate_predictions()
                await asyncio.sleep(60)  # Generate predictions every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Prediction loop error: {e}", exc_info=True)
                await asyncio.sleep(60)

    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks with enhanced analysis."""
        results = {}
        current_time = time.time()
        
        for name, check_func in self.health_checks.items():
            try:
                start_time = time.time()
                result = await asyncio.to_thread(check_func)
                result.duration_ms = (time.time() - start_time) * 1000
                result.last_check = current_time
                
                # Enhance result with trend analysis
                await self._enhance_health_check(name, result)
                
                results[name] = result
                self.last_results[name] = result
                
                # Store in history
                with self._lock:
                    self.health_history[name].append({
                        'timestamp': current_time,
                        'status': result.status.value,
                        'duration_ms': result.duration_ms,
                        'metrics': [m.__dict__ for m in result.metrics]
                    })
                
                self.metrics['total_checks'] += 1
                
            except Exception as e:
                self.logger.error(f"Health check '{name}' failed: {e}", exc_info=True)
                
                # Create failure result
                failure_result = HealthCheck(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check failed: {str(e)}",
                    last_check=current_time,
                    duration_ms=0.0,
                    details={"error": str(e)}
                )
                
                results[name] = failure_result
                self.last_results[name] = failure_result
                self.metrics['failed_checks'] += 1
        
        return results

    async def _enhance_health_check(self, check_name: str, health_check: HealthCheck):
        """Enhance health check with trend analysis and predictions."""
        current_time = time.time()
        
        # Analyze trends for each metric
        for metric in health_check.metrics:
            metric_key = f"{check_name}_{metric.name}"
            
            # Store metric history
            with self._lock:
                self.metric_history[check_name][metric.name].append({
                    'timestamp': current_time,
                    'value': metric.value,
                    'status': metric.status.value
                })
            
            # Analyze trend if we have enough history
            history = list(self.metric_history[check_name][metric.name])
            if len(history) >= 10:
                values = [h['value'] for h in history]
                timestamps = [h['timestamp'] for h in history]
                
                trend_direction, trend_confidence = self.trend_analyzer.analyze_trend(values, timestamps)
                metric.trend_direction = trend_direction
                metric.trend_confidence = trend_confidence
                
                # Generate prediction
                if self.enable_predictions:
                    future_time = current_time + 300  # 5 minutes ahead
                    predicted_value, prediction_confidence = self.trend_analyzer.predict_future_value(
                        values, timestamps, future_time
                    )
                    
                    if predicted_value is not None:
                        metric.prediction = predicted_value
                        metric.prediction_confidence = prediction_confidence
        
        # Generate overall trend analysis for the health check
        health_check.trend_analysis = self._generate_trend_analysis(check_name)
        
        # Generate prediction for overall health
        if self.enable_predictions:
            health_check.prediction = await self._generate_health_prediction(check_name)

    def _generate_trend_analysis(self, check_name: str) -> Dict[str, Any]:
        """Generate trend analysis summary for a health check."""
        history = list(self.health_history[check_name])
        if len(history) < 5:
            return {"status": "insufficient_data"}
        
        # Analyze status changes
        status_changes = 0
        last_status = None
        healthy_periods = []
        current_period_start = None
        
        for entry in history:
            if last_status and entry['status'] != last_status:
                status_changes += 1
                
                # Track healthy periods
                if last_status == HealthStatus.HEALTHY.value and current_period_start:
                    healthy_periods.append(entry['timestamp'] - current_period_start)
                
                if entry['status'] == HealthStatus.HEALTHY.value:
                    current_period_start = entry['timestamp']
            
            last_status = entry['status']
        
        # Calculate stability metrics
        stability_score = 1.0 - (status_changes / len(history))
        avg_healthy_period = statistics.mean(healthy_periods) if healthy_periods else 0
        
        return {
            "status": "analyzed",
            "stability_score": stability_score,
            "status_changes": status_changes,
            "avg_healthy_period_seconds": avg_healthy_period,
            "total_samples": len(history)
        }

    async def _generate_health_prediction(self, check_name: str) -> Dict[str, Any]:
        """Generate health prediction for a component."""
        history = list(self.health_history[check_name])
        if len(history) < 20:
            return {"status": "insufficient_data"}
        
        try:
            # Convert status to numeric values for trend analysis
            status_values = []
            timestamps = []
            
            status_map = {
                HealthStatus.HEALTHY.value: 4,
                HealthStatus.DEGRADED.value: 3,
                HealthStatus.UNHEALTHY.value: 2,
                HealthStatus.CRITICAL.value: 1,
                HealthStatus.UNKNOWN.value: 0
            }
            
            for entry in history[-50:]:  # Use last 50 entries
                status_values.append(status_map.get(entry['status'], 0))
                timestamps.append(entry['timestamp'])
            
            # Predict status in 5, 15, and 30 minutes
            predictions = {}
            current_time = time.time()
            
            for minutes in [5, 15, 30]:
                future_time = current_time + (minutes * 60)
                predicted_value, confidence = self.trend_analyzer.predict_future_value(
                    status_values, timestamps, future_time
                )
                
                if predicted_value is not None:
                    # Convert back to status
                    predicted_status = HealthStatus.UNKNOWN
                    for status, value in status_map.items():
                        if abs(predicted_value - value) < 0.5:
                            predicted_status = HealthStatus(status)
                            break
                    
                    predictions[f"{minutes}_minutes"] = {
                        "predicted_status": predicted_status.value,
                        "confidence": confidence,
                        "predicted_value": predicted_value
                    }
            
            return {
                "status": "predicted",
                "predictions": predictions,
                "generated_at": current_time
            }
            
        except Exception as e:
            self.logger.warning(f"Health prediction error for {check_name}: {e}")
            return {"status": "error", "error": str(e)}

    async def _generate_predictions(self):
        """Generate predictive alerts based on trends."""
        current_time = time.time()
        new_alerts = []
        
        for check_name in self.health_checks.keys():
            try:
                # Analyze each metric for potential future issues
                for metric_name, metric_history in self.metric_history[check_name].items():
                    history = list(metric_history)
                    if len(history) < 20:
                        continue
                    
                    values = [h['value'] for h in history[-30:]]  # Last 30 values
                    timestamps = [h['timestamp'] for h in history[-30:]]
                    
                    # Predict values for next 10, 30, and 60 minutes
                    for minutes in [10, 30, 60]:
                        future_time = current_time + (minutes * 60)
                        predicted_value, confidence = self.trend_analyzer.predict_future_value(
                            values, timestamps, future_time
                        )
                        
                        if predicted_value is None or confidence < 0.7:
                            continue
                        
                        # Check if predicted value crosses thresholds
                        thresholds = self.prediction_thresholds.get(metric_name, {})
                        predicted_status = self._determine_status_from_value(
                            metric_name, predicted_value, thresholds
                        )
                        
                        # Only create alert if prediction is for degraded or worse status
                        if predicted_status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                            # Check if we already have a similar alert
                            existing_alert = any(
                                alert.component == check_name and 
                                alert.metric_name == metric_name and
                                alert.predicted_status == predicted_status and
                                (current_time - alert.created_at) < 1800  # 30 minutes
                                for alert in self.predictive_alerts
                            )
                            
                            if not existing_alert:
                                # Analyze current trend
                                trend_direction, _ = self.trend_analyzer.analyze_trend(values, timestamps)
                                
                                alert = PredictiveAlert(
                                    id=str(uuid.uuid4()),
                                    component=check_name,
                                    metric_name=metric_name,
                                    predicted_status=predicted_status,
                                    confidence=confidence,
                                    predicted_time=future_time,
                                    current_trend=trend_direction,
                                    recommendation=self._generate_recommendation(
                                        check_name, metric_name, predicted_status, trend_direction
                                    )
                                )
                                
                                new_alerts.append(alert)
                                self.metrics['predictions_made'] += 1
            
            except Exception as e:
                self.logger.warning(f"Prediction generation error for {check_name}: {e}")
        
        # Add new alerts
        self.predictive_alerts.extend(new_alerts)
        
        # Clean up old alerts (older than 4 hours)
        cutoff_time = current_time - 14400
        self.predictive_alerts = [
            alert for alert in self.predictive_alerts 
            if alert.created_at > cutoff_time
        ]
        
        if new_alerts:
            self.logger.info(f"Generated {len(new_alerts)} new predictive alerts")
            for alert in new_alerts:
                self.logger.warning(
                    f"PREDICTIVE ALERT: {alert.component}.{alert.metric_name} "
                    f"predicted to be {alert.predicted_status.value} "
                    f"in {(alert.predicted_time - current_time) / 60:.1f} minutes "
                    f"(confidence: {alert.confidence:.2f})"
                )

    def _determine_status_from_value(self, metric_name: str, value: float, 
                                   thresholds: Dict[str, float]) -> HealthStatus:
        """Determine health status from a metric value and thresholds."""
        if not thresholds:
            return HealthStatus.HEALTHY
        
        if 'critical' in thresholds and self._threshold_exceeded(metric_name, value, thresholds['critical']):
            return HealthStatus.CRITICAL
        elif 'unhealthy' in thresholds and self._threshold_exceeded(metric_name, value, thresholds['unhealthy']):
            return HealthStatus.UNHEALTHY
        elif 'degraded' in thresholds and self._threshold_exceeded(metric_name, value, thresholds['degraded']):
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

    def _threshold_exceeded(self, metric_name: str, value: float, threshold: float) -> bool:
        """Check if a threshold is exceeded based on metric type."""
        # For quality metrics (0-1 scale), lower values are worse
        if 'quality' in metric_name.lower() or 'confidence' in metric_name.lower():
            return value < threshold
        # For most other metrics, higher values are worse
        else:
            return value > threshold

    def _generate_recommendation(self, component: str, metric: str, 
                               predicted_status: HealthStatus, trend: TrendDirection) -> str:
        """Generate recommendation for predictive alert."""
        recommendations = {
            'signal_quality': {
                HealthStatus.DEGRADED: "Check electrode connections and signal amplifiers",
                HealthStatus.UNHEALTHY: "Inspect BCI hardware and recalibrate if necessary",
                HealthStatus.CRITICAL: "Immediate hardware inspection required - consider backup system"
            },
            'cpu_usage': {
                HealthStatus.DEGRADED: "Monitor CPU-intensive processes and consider optimization",
                HealthStatus.UNHEALTHY: "Scale up resources or optimize algorithms",
                HealthStatus.CRITICAL: "Immediate resource scaling required to prevent service disruption"
            },
            'memory_usage': {
                HealthStatus.DEGRADED: "Monitor memory leaks and optimize buffer usage",
                HealthStatus.UNHEALTHY: "Increase available memory or optimize data structures",
                HealthStatus.CRITICAL: "Immediate memory optimization required to prevent crashes"
            },
            'response_time': {
                HealthStatus.DEGRADED: "Optimize processing algorithms and check network latency",
                HealthStatus.UNHEALTHY: "Review system architecture and consider performance tuning",
                HealthStatus.CRITICAL: "Immediate performance optimization required for patient safety"
            }
        }
        
        base_recommendation = recommendations.get(metric, {}).get(
            predicted_status, 
            f"Monitor {component} {metric} closely and prepare intervention"
        )
        
        trend_suffix = {
            TrendDirection.DEGRADING: " - Trend is worsening rapidly",
            TrendDirection.VOLATILE: " - Metric is highly volatile, investigate root cause",
            TrendDirection.STABLE: " - Current trend is stable but threshold may be crossed",
            TrendDirection.IMPROVING: " - Trend is improving but monitoring recommended"
        }.get(trend, "")
        
        return base_recommendation + trend_suffix

    async def handle_unhealthy_components(self) -> None:
        """Enhanced component recovery with predictive considerations."""
        recovery_attempts = []
        
        for name, result in self.last_results.items():
            if result.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                self.logger.warning(f"Component '{name}' is {result.status.value}: {result.message}")
                
                # Check if we have predictive alerts for this component
                related_predictions = [
                    alert for alert in self.predictive_alerts 
                    if alert.component == name and alert.predicted_time > time.time()
                ]
                
                if related_predictions:
                    self.logger.info(f"Found {len(related_predictions)} predictive alerts for {name}")
                
                # Attempt recovery if strategy is available
                if name in self.recovery_strategies:
                    try:
                        self.logger.info(f"Attempting recovery for '{name}'")
                        recovery_result = await asyncio.to_thread(self.recovery_strategies[name])
                        
                        recovery_attempts.append({
                            'component': name,
                            'result': recovery_result,
                            'timestamp': time.time()
                        })
                        
                        self.metrics['recovery_attempts'] += 1
                        if recovery_result:
                            self.metrics['recovery_successes'] += 1
                        
                        self.logger.info(f"Recovery {'succeeded' if recovery_result else 'failed'} for '{name}'")
                        
                    except Exception as e:
                        self.logger.error(f"Recovery failed for '{name}': {e}", exc_info=True)

    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status with enhanced logic."""
        if not self.last_results:
            return HealthStatus.UNKNOWN
        
        statuses = [result.status for result in self.last_results.values()]
        
        # Count statuses
        status_counts = {status: statuses.count(status) for status in HealthStatus}
        
        # Weighted health calculation
        if status_counts[HealthStatus.CRITICAL] > 0:
            return HealthStatus.CRITICAL
        elif status_counts[HealthStatus.UNHEALTHY] > 0:
            return HealthStatus.UNHEALTHY
        elif status_counts[HealthStatus.DEGRADED] > len(self.last_results) * 0.3:  # 30% threshold
            return HealthStatus.DEGRADED
        elif status_counts[HealthStatus.UNKNOWN] > len(self.last_results) * 0.5:  # 50% threshold
            return HealthStatus.UNKNOWN
        else:
            return HealthStatus.HEALTHY

    def get_comprehensive_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary with trends and predictions."""
        overall_status = self.get_overall_health()
        current_time = time.time()
        
        # Component summaries with trends
        components = {}
        for name, result in self.last_results.items():
            components[name] = {
                "status": result.status.value,
                "message": result.message,
                "last_check": result.last_check,
                "duration_ms": result.duration_ms,
                "metrics": [m.__dict__ for m in result.metrics],
                "trend_analysis": result.trend_analysis,
                "prediction": result.prediction,
                "details": result.details
            }
        
        # Predictive alerts summary
        active_predictions = [
            alert for alert in self.predictive_alerts 
            if alert.predicted_time > current_time
        ]
        
        prediction_summary = {
            "total_active": len(active_predictions),
            "by_severity": {
                status.value: len([a for a in active_predictions if a.predicted_status == status])
                for status in HealthStatus
            },
            "by_component": defaultdict(int)
        }
        
        for alert in active_predictions:
            prediction_summary["by_component"][alert.component] += 1
        
        # System-wide trends
        trend_summary = self._calculate_system_trends()
        
        return {
            "overall_health": {
                "status": overall_status.value,
                "monitoring_active": self.monitoring_active,
                "check_interval": self.check_interval,
                "predictions_enabled": self.enable_predictions
            },
            "components": components,
            "summary": {
                "total_components": len(self.last_results),
                "healthy": sum(1 for r in self.last_results.values() if r.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for r in self.last_results.values() if r.status == HealthStatus.DEGRADED),
                "unhealthy": sum(1 for r in self.last_results.values() if r.status == HealthStatus.UNHEALTHY),
                "critical": sum(1 for r in self.last_results.values() if r.status == HealthStatus.CRITICAL),
                "unknown": sum(1 for r in self.last_results.values() if r.status == HealthStatus.UNKNOWN)
            },
            "predictions": {
                "summary": dict(prediction_summary),
                "alerts": [
                    {
                        "id": alert.id,
                        "component": alert.component,
                        "metric": alert.metric_name,
                        "predicted_status": alert.predicted_status.value,
                        "confidence": alert.confidence,
                        "minutes_until": (alert.predicted_time - current_time) / 60,
                        "trend": alert.current_trend.value,
                        "recommendation": alert.recommendation
                    } for alert in active_predictions
                ]
            },
            "trends": trend_summary,
            "performance": {
                **self.metrics,
                "prediction_accuracy": (
                    self.metrics['predictions_accurate'] / max(self.metrics['predictions_made'], 1)
                ) * 100,
                "recovery_success_rate": (
                    self.metrics['recovery_successes'] / max(self.metrics['recovery_attempts'], 1)
                ) * 100
            }
        }

    def _calculate_system_trends(self) -> Dict[str, Any]:
        """Calculate system-wide health trends."""
        if not self.health_history:
            return {"status": "no_data"}
        
        try:
            # Aggregate health scores across all components
            current_time = time.time()
            time_window = 3600  # 1 hour
            cutoff_time = current_time - time_window
            
            system_scores = []
            timestamps = []
            
            # Sample system health every minute for the past hour
            for t in range(int(cutoff_time), int(current_time), 60):
                # Calculate system health score at time t
                component_scores = []
                
                for component_name, history in self.health_history.items():
                    # Find the closest health check to time t
                    closest_entry = None
                    min_time_diff = float('inf')
                    
                    for entry in history:
                        time_diff = abs(entry['timestamp'] - t)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_entry = entry
                    
                    if closest_entry and min_time_diff < 300:  # Within 5 minutes
                        status_score = {
                            'healthy': 100,
                            'degraded': 75,
                            'unhealthy': 50,
                            'critical': 25,
                            'unknown': 0
                        }.get(closest_entry['status'], 0)
                        
                        component_scores.append(status_score)
                
                if component_scores:
                    system_scores.append(statistics.mean(component_scores))
                    timestamps.append(t)
            
            if len(system_scores) < 10:
                return {"status": "insufficient_data"}
            
            # Calculate trend
            trend_direction, trend_confidence = self.trend_analyzer.analyze_trend(
                system_scores, timestamps
            )
            
            # Calculate volatility
            volatility = statistics.stdev(system_scores) if len(system_scores) > 1 else 0
            
            return {
                "status": "calculated",
                "current_score": system_scores[-1] if system_scores else 0,
                "trend_direction": trend_direction.value,
                "trend_confidence": trend_confidence,
                "volatility": volatility,
                "score_range": {
                    "min": min(system_scores) if system_scores else 0,
                    "max": max(system_scores) if system_scores else 0,
                    "avg": statistics.mean(system_scores) if system_scores else 0
                },
                "samples": len(system_scores)
            }
            
        except Exception as e:
            self.logger.warning(f"System trend calculation error: {e}")
            return {"status": "error", "error": str(e)}

    def get_predictive_alerts(self, component: Optional[str] = None) -> List[PredictiveAlert]:
        """Get predictive alerts, optionally filtered by component."""
        current_time = time.time()
        
        # Filter active alerts
        active_alerts = [
            alert for alert in self.predictive_alerts 
            if alert.predicted_time > current_time
        ]
        
        if component:
            active_alerts = [alert for alert in active_alerts if alert.component == component]
        
        return sorted(active_alerts, key=lambda a: a.predicted_time)

    async def cleanup(self):
        """Cleanup monitoring resources."""
        await self.stop_monitoring()
        self.logger.info("Health monitor cleanup complete")


# Enhanced health check factory functions
def create_enhanced_bci_health_checks(bci_bridge) -> Dict[str, Callable]:
    """Create enhanced health checks for BCI components with detailed metrics."""
    
    def check_bci_device_enhanced() -> HealthCheck:
        """Enhanced BCI device connectivity check."""
        try:
            device_info = bci_bridge.get_device_info()
            current_time = time.time()
            
            # Create metrics
            metrics = []
            
            # Connection status
            connection_metric = HealthMetric(
                name="connection_status",
                value=1.0 if device_info.get('connected', False) else 0.0,
                timestamp=current_time,
                status=HealthStatus.HEALTHY if device_info.get('connected', False) else HealthStatus.CRITICAL,
                trend_direction=TrendDirection.STABLE,
                trend_confidence=0.0,
                tags={"type": "connectivity"}
            )
            metrics.append(connection_metric)
            
            # Signal strength
            signal_strength = device_info.get('signal_strength', 0.0)
            signal_metric = HealthMetric(
                name="signal_strength",
                value=signal_strength,
                timestamp=current_time,
                status=HealthStatus.HEALTHY if signal_strength > 0.7 else 
                       HealthStatus.DEGRADED if signal_strength > 0.5 else HealthStatus.UNHEALTHY,
                trend_direction=TrendDirection.STABLE,
                trend_confidence=0.0,
                tags={"type": "signal", "unit": "ratio"}
            )
            metrics.append(signal_metric)
            
            # Determine overall status
            if device_info.get('connected', False) and signal_strength > 0.5:
                overall_status = HealthStatus.HEALTHY
                message = "BCI device connected with good signal"
            elif device_info.get('connected', False):
                overall_status = HealthStatus.DEGRADED
                message = f"BCI device connected but signal strength low: {signal_strength:.2f}"
            else:
                overall_status = HealthStatus.CRITICAL
                message = "BCI device not connected"
            
            return HealthCheck(
                name="bci_device",
                status=overall_status,
                message=message,
                last_check=current_time,
                duration_ms=0.0,
                details=device_info,
                metrics=metrics
            )
            
        except Exception as e:
            return HealthCheck(
                name="bci_device",
                status=HealthStatus.UNKNOWN,
                message=f"Device check failed: {str(e)}",
                last_check=time.time(),
                duration_ms=0.0,
                details={"error": str(e)}
            )

    def check_signal_quality_enhanced() -> HealthCheck:
        """Enhanced neural signal quality check."""
        try:
            current_time = time.time()
            metrics = []
            
            buffer_size = len(bci_bridge.data_buffer)
            
            # Buffer size metric
            buffer_metric = HealthMetric(
                name="buffer_size",
                value=float(buffer_size),
                timestamp=current_time,
                status=HealthStatus.HEALTHY if buffer_size > 0 else HealthStatus.UNHEALTHY,
                trend_direction=TrendDirection.STABLE,
                trend_confidence=0.0,
                tags={"type": "buffer", "unit": "samples"}
            )
            metrics.append(buffer_metric)
            
            if buffer_size > 0:
                recent_data = bci_bridge.get_buffer(min(100, buffer_size))
                
                if recent_data.size > 0:
                    # Signal quality metrics
                    noise_level = float(recent_data.std())
                    signal_range = float(recent_data.max() - recent_data.min())
                    signal_quality = min(1.0, signal_range / (noise_level + 1e-8))
                    
                    # Noise level metric
                    noise_metric = HealthMetric(
                        name="noise_level",
                        value=noise_level,
                        timestamp=current_time,
                        status=HealthStatus.HEALTHY if noise_level < 50 else
                               HealthStatus.DEGRADED if noise_level < 100 else HealthStatus.UNHEALTHY,
                        trend_direction=TrendDirection.STABLE,
                        trend_confidence=0.0,
                        tags={"type": "noise", "unit": "microvolts"}
                    )
                    metrics.append(noise_metric)
                    
                    # Signal quality metric
                    quality_metric = HealthMetric(
                        name="signal_quality",
                        value=signal_quality,
                        timestamp=current_time,
                        status=HealthStatus.HEALTHY if signal_quality > 0.7 else
                               HealthStatus.DEGRADED if signal_quality > 0.4 else HealthStatus.UNHEALTHY,
                        trend_direction=TrendDirection.STABLE,
                        trend_confidence=0.0,
                        tags={"type": "quality", "unit": "ratio"}
                    )
                    metrics.append(quality_metric)
                    
                    # Determine overall status
                    if signal_quality > 0.7 and noise_level < 50:
                        status = HealthStatus.HEALTHY
                        message = f"Signal quality is good (quality: {signal_quality:.2f})"
                    elif signal_quality > 0.4:
                        status = HealthStatus.DEGRADED
                        message = f"Signal quality is acceptable (quality: {signal_quality:.2f})"
                    else:
                        status = HealthStatus.UNHEALTHY
                        message = f"Signal quality is poor (quality: {signal_quality:.2f})"
                    
                    return HealthCheck(
                        name="signal_quality",
                        status=status,
                        message=message,
                        last_check=current_time,
                        duration_ms=0.0,
                        details={
                            "noise_level": noise_level,
                            "signal_range": signal_range,
                            "signal_quality": signal_quality,
                            "buffer_size": buffer_size
                        },
                        metrics=metrics
                    )
                else:
                    return HealthCheck(
                        name="signal_quality",
                        status=HealthStatus.UNHEALTHY,
                        message="No signal data available in buffer",
                        last_check=current_time,
                        duration_ms=0.0,
                        metrics=metrics
                    )
            else:
                return HealthCheck(
                    name="signal_quality",
                    status=HealthStatus.DEGRADED,
                    message="Buffer is empty - no recent data",
                    last_check=current_time,
                    duration_ms=0.0,
                    metrics=metrics
                )
                
        except Exception as e:
            return HealthCheck(
                name="signal_quality",
                status=HealthStatus.UNKNOWN,
                message=f"Signal quality check failed: {str(e)}",
                last_check=time.time(),
                duration_ms=0.0,
                details={"error": str(e)}
            )

    def check_decoder_status_enhanced() -> HealthCheck:
        """Enhanced neural decoder status check."""
        try:
            current_time = time.time()
            metrics = []
            
            if bci_bridge.decoder is None:
                return HealthCheck(
                    name="decoder_status",
                    status=HealthStatus.CRITICAL,
                    message="No decoder initialized - critical system component missing",
                    last_check=current_time,
                    duration_ms=0.0
                )
            
            decoder_info = bci_bridge.decoder.get_decoder_info()
            
            # Calibration status metric
            calibration_metric = HealthMetric(
                name="calibration_status",
                value=1.0 if decoder_info.get('calibrated', False) else 0.0,
                timestamp=current_time,
                status=HealthStatus.HEALTHY if decoder_info.get('calibrated', False) else HealthStatus.DEGRADED,
                trend_direction=TrendDirection.STABLE,
                trend_confidence=0.0,
                tags={"type": "calibration"}
            )
            metrics.append(calibration_metric)
            
            # Confidence metric
            confidence = decoder_info.get('last_confidence', 0.0)
            confidence_metric = HealthMetric(
                name="decoder_confidence",
                value=confidence,
                timestamp=current_time,
                status=HealthStatus.HEALTHY if confidence >= 0.8 else
                       HealthStatus.DEGRADED if confidence >= 0.6 else HealthStatus.UNHEALTHY,
                trend_direction=TrendDirection.STABLE,
                trend_confidence=0.0,
                tags={"type": "confidence", "unit": "ratio"}
            )
            metrics.append(confidence_metric)
            
            # Processing latency metric
            latency = decoder_info.get('processing_latency_ms', 0.0)
            latency_metric = HealthMetric(
                name="processing_latency",
                value=latency,
                timestamp=current_time,
                status=HealthStatus.HEALTHY if latency < 50 else
                       HealthStatus.DEGRADED if latency < 100 else HealthStatus.UNHEALTHY,
                trend_direction=TrendDirection.STABLE,
                trend_confidence=0.0,
                tags={"type": "latency", "unit": "milliseconds"}
            )
            metrics.append(latency_metric)
            
            # Determine overall status
            if decoder_info.get('calibrated', False):
                if confidence >= 0.8 and latency < 50:
                    status = HealthStatus.HEALTHY
                    message = f"Decoder operating optimally (confidence: {confidence:.2f})"
                elif confidence >= 0.6:
                    status = HealthStatus.DEGRADED
                    message = f"Decoder operating with acceptable confidence (confidence: {confidence:.2f})"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f"Decoder confidence is low (confidence: {confidence:.2f})"
            else:
                status = HealthStatus.DEGRADED
                message = "Decoder not calibrated - performance may be suboptimal"
            
            return HealthCheck(
                name="decoder_status",
                status=status,
                message=message,
                last_check=current_time,
                duration_ms=0.0,
                details=decoder_info,
                metrics=metrics
            )
            
        except Exception as e:
            return HealthCheck(
                name="decoder_status",
                status=HealthStatus.UNKNOWN,
                message=f"Decoder check failed: {str(e)}",
                last_check=time.time(),
                duration_ms=0.0,
                details={"error": str(e)}
            )

    return {
        "bci_device": check_bci_device_enhanced,
        "signal_quality": check_signal_quality_enhanced,
        "decoder_status": check_decoder_status_enhanced
    }


def create_enhanced_claude_health_checks(claude_adapter) -> Dict[str, Callable]:
    """Create enhanced health checks for Claude adapter."""
    
    def check_claude_connectivity_enhanced() -> HealthCheck:
        """Enhanced Claude API connectivity check."""
        try:
            current_time = time.time()
            metrics = []
            
            # Configuration check metric
            config_status = 1.0 if (hasattr(claude_adapter, 'client') and claude_adapter.client) else 0.0
            config_metric = HealthMetric(
                name="configuration_status",
                value=config_status,
                timestamp=current_time,
                status=HealthStatus.HEALTHY if config_status > 0 else HealthStatus.CRITICAL,
                trend_direction=TrendDirection.STABLE,
                trend_confidence=0.0,
                tags={"type": "configuration"}
            )
            metrics.append(config_metric)
            
            if hasattr(claude_adapter, 'client') and claude_adapter.client:
                # API response time metric (simulated - would need actual API call)
                response_time = getattr(claude_adapter, 'last_response_time', 1000.0)
                response_metric = HealthMetric(
                    name="api_response_time",
                    value=response_time,
                    timestamp=current_time,
                    status=HealthStatus.HEALTHY if response_time < 2000 else
                           HealthStatus.DEGRADED if response_time < 5000 else HealthStatus.UNHEALTHY,
                    trend_direction=TrendDirection.STABLE,
                    trend_confidence=0.0,
                    tags={"type": "performance", "unit": "milliseconds"}
                )
                metrics.append(response_metric)
                
                # Error rate metric
                error_rate = getattr(claude_adapter, 'error_rate', 0.0)
                error_metric = HealthMetric(
                    name="api_error_rate",
                    value=error_rate,
                    timestamp=current_time,
                    status=HealthStatus.HEALTHY if error_rate < 0.05 else
                           HealthStatus.DEGRADED if error_rate < 0.15 else HealthStatus.UNHEALTHY,
                    trend_direction=TrendDirection.STABLE,
                    trend_confidence=0.0,
                    tags={"type": "error_rate", "unit": "ratio"}
                )
                metrics.append(error_metric)
                
                # Determine overall status
                if response_time < 2000 and error_rate < 0.05:
                    status = HealthStatus.HEALTHY
                    message = "Claude adapter is operating optimally"
                elif response_time < 5000 and error_rate < 0.15:
                    status = HealthStatus.DEGRADED
                    message = f"Claude adapter performance degraded (response: {response_time}ms, errors: {error_rate:.1%})"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f"Claude adapter has performance issues (response: {response_time}ms, errors: {error_rate:.1%})"
                
                return HealthCheck(
                    name="claude_connectivity",
                    status=status,
                    message=message,
                    last_check=current_time,
                    duration_ms=0.0,
                    details={
                        "model": getattr(claude_adapter, 'model', 'unknown'),
                        "safety_mode": getattr(claude_adapter, 'safety_mode', 'unknown'),
                        "response_time": response_time,
                        "error_rate": error_rate
                    },
                    metrics=metrics
                )
            else:
                return HealthCheck(
                    name="claude_connectivity",
                    status=HealthStatus.CRITICAL,
                    message="Claude adapter not properly configured - client not initialized",
                    last_check=current_time,
                    duration_ms=0.0,
                    metrics=metrics
                )
                
        except Exception as e:
            return HealthCheck(
                name="claude_connectivity",
                status=HealthStatus.UNKNOWN,
                message=f"Claude connectivity check failed: {str(e)}",
                last_check=time.time(),
                duration_ms=0.0,
                details={"error": str(e)}
            )

    return {
        "claude_connectivity": check_claude_connectivity_enhanced
    }


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import random
    import numpy as np
    
    # Mock BCI bridge for testing
    class MockBCIBridge:
        def __init__(self):
            self.data_buffer = np.random.randn(1000)
            
        def get_device_info(self):
            return {
                'connected': random.choice([True, True, True, False]),  # 75% connected
                'signal_strength': random.uniform(0.3, 1.0)
            }
            
        def get_buffer(self, size):
            return self.data_buffer[:size]
    
    class MockDecoder:
        def get_decoder_info(self):
            return {
                'calibrated': random.choice([True, True, False]),  # 67% calibrated
                'last_confidence': random.uniform(0.4, 1.0),
                'processing_latency_ms': random.uniform(20, 150)
            }
    
    class MockClaudeAdapter:
        def __init__(self):
            self.client = True
            self.model = "claude-3-sonnet"
            self.safety_mode = "strict"
            self.last_response_time = random.uniform(500, 3000)
            self.error_rate = random.uniform(0.0, 0.2)
    
    async def demo_enhanced_health_monitor():
        # Create mock components
        bci_bridge = MockBCIBridge()
        bci_bridge.decoder = MockDecoder()
        claude_adapter = MockClaudeAdapter()
        
        # Create enhanced health monitor
        health_monitor = HealthMonitor(check_interval=10.0, enable_predictions=True)
        
        # Register enhanced health checks
        bci_checks = create_enhanced_bci_health_checks(bci_bridge)
        claude_checks = create_enhanced_claude_health_checks(claude_adapter)
        
        for name, check_func in {**bci_checks, **claude_checks}.items():
            health_monitor.register_health_check(name, check_func)
        
        print("Enhanced BCI Health Monitor Demo")
        print("=" * 50)
        
        # Start monitoring
        await health_monitor.start_monitoring()
        
        # Let it run for a while to collect data
        print("Collecting health data...")
        await asyncio.sleep(20)
        
        # Generate some trend data by running multiple checks
        for i in range(10):
            await health_monitor.run_all_checks()
            await asyncio.sleep(2)
        
        # Show comprehensive summary
        print("\n--- Comprehensive Health Summary ---")
        summary = health_monitor.get_comprehensive_health_summary()
        print(json.dumps(summary, indent=2, default=str))
        
        # Show predictive alerts
        print("\n--- Predictive Alerts ---")
        predictions = health_monitor.get_predictive_alerts()
        for alert in predictions:
            print(f"🔮 PREDICTIVE ALERT: {alert.component}.{alert.metric_name}")
            print(f"   Predicted Status: {alert.predicted_status.value}")
            print(f"   Confidence: {alert.confidence:.2f}")
            print(f"   Time to Issue: {(alert.predicted_time - time.time()) / 60:.1f} minutes")
            print(f"   Recommendation: {alert.recommendation}")
            print()
        
        # Cleanup
        await health_monitor.cleanup()
    
    asyncio.run(demo_enhanced_health_monitor())