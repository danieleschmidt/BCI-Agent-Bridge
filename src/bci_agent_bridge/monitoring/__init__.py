"""
Enterprise-grade monitoring and health check components for BCI Agent Bridge.

This module provides comprehensive monitoring capabilities including:
- Enhanced alert management with escalation rules
- Predictive health monitoring with trend analysis  
- Time-series metrics collection with anomaly detection
- Real-time dashboard feeds with WebSocket support
- SLA monitoring and violation tracking
- Adaptive circuit breakers with recovery strategies
- Distributed tracing with correlation
- Structured logging with audit trails
"""

# Core monitoring components (enhanced versions)
from .alert_manager import (
    AlertManager, Alert, AlertSeverity, AlertStatus, AlertRule, 
    EscalationPolicy, create_enhanced_bci_alert_rules, 
    create_default_notification_handlers
)

from .health_monitor import (
    HealthMonitor, HealthStatus, HealthCheck, HealthMetric, TrendDirection,
    PredictiveAlert, HealthTrendAnalyzer, create_enhanced_bci_health_checks,
    create_enhanced_claude_health_checks
)

from .metrics_collector import (
    MetricsCollector, BCIMetricsCollector, Metric, MetricSummary, MetricType,
    AnomalyDetection, AnomalyType, TimeSeriesStorage, AnomalyDetector
)

# Advanced monitoring features
from .dashboard_feeds import (
    DashboardFeedManager, DashboardMessage, FeedType, MessageType,
    ConnectedClient
)

from .sla_monitor import (
    SLAMonitor, SLAThreshold, SLAViolation, SLAReport, SLAStatus,
    SLAMetricType
)

from .adaptive_circuit_breaker import (
    AdaptiveCircuitBreaker, AdaptiveCircuitBreakerManager, AdaptiveConfig,
    CircuitState, CircuitMetrics, RecoveryStrategy, FailureCategory,
    CircuitBreakerOpenException, create_bci_adaptive_circuit_breakers
)

from .distributed_tracing import (
    DistributedTracer, Span, SpanKind, SpanStatus, TraceContext,
    initialize_global_tracer, get_global_tracer, trace_function,
    trace_bci_operation, inject_trace_context, extract_trace_context,
    ConsoleTraceExporter, JSONFileTraceExporter
)

from .structured_logging import (
    StructuredLogger, LogLevel, LogCategory, LogContext, StructuredLogRecord,
    LoggingContext, get_logger, configure_logging, log_function_calls,
    ConsoleLogHandler, FileLogHandler, AuditLogHandler, SecurityLogHandler
)


# Convenience factory functions for easy setup
def create_comprehensive_monitoring_system(
    service_name: str = "bci-agent-bridge",
    enable_dashboard: bool = True,
    enable_tracing: bool = True,
    enable_structured_logging: bool = True,
    dashboard_port: int = 8765,
    log_level = None  # Will import LogLevel when needed
) -> dict:
    """
    Create a comprehensive monitoring system with all components configured.
    
    Args:
        service_name: Name of the service being monitored
        enable_dashboard: Whether to enable real-time dashboard feeds
        enable_tracing: Whether to enable distributed tracing
        enable_structured_logging: Whether to enable structured logging
        dashboard_port: Port for dashboard WebSocket server
        log_level: Logging level for structured logger
    
    Returns:
        Dictionary containing all initialized monitoring components
    """
    if log_level is None:
        log_level = LogLevel.INFO
        
    components = {}
    
    # Core monitoring components
    components['alert_manager'] = AlertManager()
    components['health_monitor'] = HealthMonitor(enable_predictions=True)
    components['metrics_collector'] = BCIMetricsCollector(
        enable_time_series=True, 
        enable_anomaly_detection=True
    )
    components['sla_monitor'] = SLAMonitor()
    components['circuit_breaker_manager'] = create_bci_adaptive_circuit_breakers()
    
    # Dashboard feeds
    if enable_dashboard:
        components['dashboard_manager'] = DashboardFeedManager(
            alert_manager=components['alert_manager'],
            health_monitor=components['health_monitor'],
            metrics_collector=components['metrics_collector'],
            port=dashboard_port
        )
    
    # Distributed tracing
    if enable_tracing:
        components['tracer'] = initialize_global_tracer(
            service_name=service_name,
            sampling_rate=1.0  # Full sampling for BCI medical systems
        )
    
    # Structured logging
    if enable_structured_logging:
        components['logger'] = configure_logging(
            level=log_level,
            enable_security=True,
            name=service_name
        )
    
    return components


async def start_monitoring_system(components: dict) -> None:
    """
    Start all monitoring system components.
    
    Args:
        components: Dictionary of monitoring components from create_comprehensive_monitoring_system
    """
    start_tasks = []
    
    # Start health monitoring
    if 'health_monitor' in components:
        start_tasks.append(components['health_monitor'].start_monitoring())
    
    # Start SLA monitoring  
    if 'sla_monitor' in components:
        start_tasks.append(components['sla_monitor'].start_monitoring())
    
    # Start circuit breaker monitoring
    if 'circuit_breaker_manager' in components:
        start_tasks.append(components['circuit_breaker_manager'].start_all_monitoring())
    
    # Start dashboard server
    if 'dashboard_manager' in components:
        start_tasks.append(components['dashboard_manager'].start_server())
    
    # Start distributed tracing cleanup
    if 'tracer' in components:
        start_tasks.append(components['tracer'].start_cleanup_task())
    
    # Execute all start tasks
    if start_tasks:
        import asyncio
        await asyncio.gather(*start_tasks)


async def stop_monitoring_system(components: dict) -> None:
    """
    Stop all monitoring system components gracefully.
    
    Args:
        components: Dictionary of monitoring components from create_comprehensive_monitoring_system
    """
    stop_tasks = []
    
    # Stop health monitoring
    if 'health_monitor' in components:
        stop_tasks.append(components['health_monitor'].stop_monitoring())
    
    # Stop SLA monitoring
    if 'sla_monitor' in components:
        stop_tasks.append(components['sla_monitor'].stop_monitoring())
    
    # Stop circuit breaker monitoring
    if 'circuit_breaker_manager' in components:
        stop_tasks.append(components['circuit_breaker_manager'].stop_all_monitoring())
    
    # Stop dashboard server
    if 'dashboard_manager' in components:
        stop_tasks.append(components['dashboard_manager'].stop_server())
    
    # Stop distributed tracing
    if 'tracer' in components:
        stop_tasks.append(components['tracer'].stop_cleanup_task())
    
    # Cleanup metrics collector
    if 'metrics_collector' in components:
        stop_tasks.append(components['metrics_collector'].cleanup())
    
    # Execute all stop tasks
    if stop_tasks:
        import asyncio
        await asyncio.gather(*stop_tasks, return_exceptions=True)


# Enhanced BCI monitoring factory
def create_bci_monitoring_suite(
    neural_session_id: str = None,
    patient_id: str = None,
    device_id: str = None,
    paradigm: str = None
) -> dict:
    """
    Create a BCI-specific monitoring suite with pre-configured rules and thresholds.
    
    Args:
        neural_session_id: Neural session identifier
        patient_id: Patient identifier (will be hashed for privacy)
        device_id: BCI device identifier
        paradigm: BCI paradigm (P300, SSVEP, Motor Imagery, etc.)
    
    Returns:
        Dictionary containing BCI-specific monitoring components
    """
    # Create base monitoring system
    components = create_comprehensive_monitoring_system(
        service_name="bci-agent-bridge",
        enable_dashboard=True,
        enable_tracing=True,
        enable_structured_logging=True
    )
    
    # Register BCI-specific alert rules
    bci_alert_rules = create_enhanced_bci_alert_rules()
    for rule in bci_alert_rules:
        components['alert_manager'].register_alert_rule(rule)
    
    # Setup BCI context for logging and tracing
    if neural_session_id or patient_id or device_id or paradigm:
        if 'logger' in components:
            components['logger'] = components['logger'].with_neural_session(
                session_id=neural_session_id,
                patient_id=patient_id,
                device_id=device_id,
                paradigm=paradigm
            )
    
    return components


# All available exports
__all__ = [
    # Core monitoring components
    "AlertManager", "Alert", "AlertSeverity", "AlertStatus", "AlertRule", 
    "EscalationPolicy", "create_enhanced_bci_alert_rules", "create_default_notification_handlers",
    
    "HealthMonitor", "HealthStatus", "HealthCheck", "HealthMetric", "TrendDirection",
    "PredictiveAlert", "HealthTrendAnalyzer", "create_enhanced_bci_health_checks",
    "create_enhanced_claude_health_checks",
    
    "MetricsCollector", "BCIMetricsCollector", "Metric", "MetricSummary", "MetricType",
    "AnomalyDetection", "AnomalyType", "TimeSeriesStorage", "AnomalyDetector",
    
    # Advanced monitoring features
    "DashboardFeedManager", "DashboardMessage", "FeedType", "MessageType", "ConnectedClient",
    
    "SLAMonitor", "SLAThreshold", "SLAViolation", "SLAReport", "SLAStatus", "SLAMetricType",
    
    "AdaptiveCircuitBreaker", "AdaptiveCircuitBreakerManager", "AdaptiveConfig",
    "CircuitState", "CircuitMetrics", "RecoveryStrategy", "FailureCategory",
    "CircuitBreakerOpenException", "create_bci_adaptive_circuit_breakers",
    
    "DistributedTracer", "Span", "SpanKind", "SpanStatus", "TraceContext",
    "initialize_global_tracer", "get_global_tracer", "trace_function", "trace_bci_operation",
    "inject_trace_context", "extract_trace_context", "ConsoleTraceExporter", "JSONFileTraceExporter",
    
    "StructuredLogger", "LogLevel", "LogCategory", "LogContext", "StructuredLogRecord",
    "LoggingContext", "get_logger", "configure_logging", "log_function_calls",
    "ConsoleLogHandler", "FileLogHandler", "AuditLogHandler", "SecurityLogHandler",
    
    # Factory and utility functions
    "create_comprehensive_monitoring_system", "start_monitoring_system", "stop_monitoring_system",
    "create_bci_monitoring_suite"
]