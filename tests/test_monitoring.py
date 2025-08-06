"""
Tests for monitoring and health check components.
"""

import pytest
import asyncio
import time
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from bci_agent_bridge.monitoring.health_monitor import (
    HealthMonitor, HealthStatus, HealthCheck, 
    create_bci_health_checks, create_claude_health_checks
)
from bci_agent_bridge.monitoring.metrics_collector import (
    MetricsCollector, BCIMetricsCollector, Metric
)
from bci_agent_bridge.monitoring.alert_manager import (
    AlertManager, Alert, AlertRule, AlertSeverity, AlertStatus,
    create_bci_alert_rules, create_default_notification_handlers
)


class TestHealthMonitor:
    """Test suite for HealthMonitor."""
    
    def test_health_monitor_initialization(self):
        """Test health monitor initialization."""
        monitor = HealthMonitor(check_interval=10.0)
        
        assert monitor.check_interval == 10.0
        assert not monitor.monitoring_active
        assert len(monitor.health_checks) == 0
        assert len(monitor.recovery_strategies) == 0
    
    def test_register_health_check(self):
        """Test registering health checks."""
        monitor = HealthMonitor()
        
        def test_check():
            return HealthCheck(
                name="test",
                status=HealthStatus.HEALTHY,
                message="Test check",
                last_check=time.time(),
                duration_ms=1.0
            )
        
        monitor.register_health_check("test_check", test_check)
        assert "test_check" in monitor.health_checks
    
    def test_register_recovery_strategy(self):
        """Test registering recovery strategies."""
        monitor = HealthMonitor()
        
        def test_recovery():
            return True
        
        monitor.register_recovery_strategy("test_component", test_recovery)
        assert "test_component" in monitor.recovery_strategies
    
    @pytest.mark.asyncio
    async def test_run_health_checks(self):
        """Test running health checks."""
        monitor = HealthMonitor()
        
        def healthy_check():
            return HealthCheck(
                name="healthy",
                status=HealthStatus.HEALTHY,
                message="All good",
                last_check=time.time(),
                duration_ms=1.0
            )
        
        def unhealthy_check():
            return HealthCheck(
                name="unhealthy", 
                status=HealthStatus.UNHEALTHY,
                message="Something wrong",
                last_check=time.time(),
                duration_ms=2.0
            )
        
        monitor.register_health_check("healthy", healthy_check)
        monitor.register_health_check("unhealthy", unhealthy_check)
        
        results = await monitor.run_all_checks()
        
        assert len(results) == 2
        assert results["healthy"].status == HealthStatus.HEALTHY
        assert results["unhealthy"].status == HealthStatus.UNHEALTHY
    
    def test_overall_health_calculation(self):
        """Test overall health status calculation."""
        monitor = HealthMonitor()
        
        # No results - should be UNKNOWN
        assert monitor.get_overall_health() == HealthStatus.UNKNOWN
        
        # Add healthy result
        monitor.last_results["test1"] = HealthCheck(
            name="test1",
            status=HealthStatus.HEALTHY,
            message="Good",
            last_check=time.time(),
            duration_ms=1.0
        )
        assert monitor.get_overall_health() == HealthStatus.HEALTHY
        
        # Add unhealthy result
        monitor.last_results["test2"] = HealthCheck(
            name="test2",
            status=HealthStatus.UNHEALTHY,
            message="Bad",
            last_check=time.time(),
            duration_ms=1.0
        )
        assert monitor.get_overall_health() == HealthStatus.UNHEALTHY
    
    def test_health_summary(self):
        """Test health summary generation."""
        monitor = HealthMonitor()
        
        monitor.last_results["test1"] = HealthCheck(
            name="test1",
            status=HealthStatus.HEALTHY,
            message="Good",
            last_check=time.time(),
            duration_ms=1.0
        )
        
        monitor.last_results["test2"] = HealthCheck(
            name="test2",
            status=HealthStatus.DEGRADED,
            message="Okay",
            last_check=time.time(),
            duration_ms=2.0
        )
        
        summary = monitor.get_health_summary()
        
        assert summary["overall_status"] == "degraded"
        assert summary["summary"]["total_components"] == 2
        assert summary["summary"]["healthy"] == 1
        assert summary["summary"]["degraded"] == 1
        assert summary["summary"]["unhealthy"] == 0


class TestMetricsCollector:
    """Test suite for MetricsCollector."""
    
    def test_metrics_collector_initialization(self):
        """Test metrics collector initialization."""
        collector = MetricsCollector(
            retention_period=1800.0,
            max_metrics_per_name=500
        )
        
        assert collector.retention_period == 1800.0
        assert collector.max_metrics_per_name == 500
        assert len(collector.metrics) == 0
    
    def test_record_metric(self):
        """Test recording metrics."""
        collector = MetricsCollector()
        
        collector.record_metric("test_metric", 42.0, {"tag": "value"}, "units")
        
        assert "test_metric" in collector.metrics
        assert len(collector.metrics["test_metric"]) == 1
        
        metric = collector.metrics["test_metric"][0]
        assert metric.name == "test_metric"
        assert metric.value == 42.0
        assert metric.tags == {"tag": "value"}
        assert metric.unit == "units"
    
    def test_increment_counter(self):
        """Test counter increment."""
        collector = MetricsCollector()
        
        collector.increment_counter("test_counter", 5)
        collector.increment_counter("test_counter", 3)
        
        assert collector.counters["test_counter"] == 8
        assert "test_counter_total" in collector.metrics
    
    def test_set_gauge(self):
        """Test gauge setting."""
        collector = MetricsCollector()
        
        collector.set_gauge("test_gauge", 100.0, unit="percent")
        
        assert collector.gauges["test_gauge"] == 100.0
        assert "test_gauge" in collector.metrics
    
    def test_record_histogram(self):
        """Test histogram recording."""
        collector = MetricsCollector()
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            collector.record_histogram("test_histogram", value)
        
        assert len(collector.histograms["test_histogram"]) == 5
        assert collector.histograms["test_histogram"] == values
    
    def test_metric_summary(self):
        """Test metric summary generation."""
        collector = MetricsCollector()
        
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        for value in values:
            collector.record_metric("test_metric", value)
        
        summary = collector.get_metric_summary("test_metric")
        
        assert summary is not None
        assert summary.count == 5
        assert summary.min_value == 1.0
        assert summary.max_value == 5.0
        assert summary.avg_value == 3.0
        assert summary.last_value == 5.0
    
    def test_export_json(self):
        """Test JSON export."""
        collector = MetricsCollector()
        
        collector.record_metric("test", 42.0)
        collector.increment_counter("counter", 5)
        collector.set_gauge("gauge", 100.0)
        
        json_export = collector.export_metrics("json")
        
        assert "metrics" in json_export
        assert "counters" in json_export
        assert "gauges" in json_export
        assert isinstance(json_export, str)
    
    def test_export_prometheus(self):
        """Test Prometheus export."""
        collector = MetricsCollector()
        
        collector.record_metric("test_metric", 42.0, {"env": "test"})
        collector.increment_counter("test_counter", 5)
        
        prom_export = collector.export_metrics("prometheus")
        
        assert "test_metric" in prom_export
        assert "test_counter" in prom_export
        assert isinstance(prom_export, str)


class TestBCIMetricsCollector:
    """Test suite for BCIMetricsCollector."""
    
    def test_bci_metrics_initialization(self):
        """Test BCI metrics collector initialization."""
        collector = BCIMetricsCollector()
        
        assert len(collector.processors) > 0  # Should have BCI-specific processors
    
    def test_record_neural_sample(self):
        """Test neural sample recording."""
        collector = BCIMetricsCollector()
        
        collector.record_neural_sample(
            channels=8,
            sampling_rate=250,
            quality_score=0.8
        )
        
        assert collector.counters["neural_samples"] == 1
        assert collector.gauges["neural_channels"] == 8
        assert "neural_data_rate" in collector.metrics
        assert "signal_quality" in collector.histograms
    
    def test_record_decoding_event(self):
        """Test decoding event recording."""
        collector = BCIMetricsCollector()
        
        collector.record_decoding_event(
            paradigm="P300",
            confidence=0.85,
            latency_ms=50.0
        )
        
        assert collector.counters["decoding_attempts"] == 1
        assert collector.counters["high_confidence_decodings"] == 1
        assert "decoding_confidence" in collector.histograms
        assert "decoding_latency" in collector.histograms
    
    def test_record_claude_interaction(self):
        """Test Claude interaction recording."""
        collector = BCIMetricsCollector()
        
        collector.record_claude_interaction(
            safety_mode="medical",
            tokens_used=150,
            response_time_ms=1200.0,
            safety_flags=["medical_urgency"]
        )
        
        assert collector.counters["claude_requests"] == 1
        assert collector.counters["safety_flags"] == 1
        assert "claude_response_time" in collector.histograms
        assert "claude_tokens_used" in collector.histograms
    
    def test_bci_performance_summary(self):
        """Test BCI performance summary."""
        collector = BCIMetricsCollector()
        
        # Record some test data
        collector.record_neural_sample(8, 250, 0.8)
        collector.record_decoding_event("P300", 0.9, 45.0)
        collector.record_claude_interaction("medical", 100, 800.0)
        
        summary = collector.get_bci_performance_summary()
        
        assert "neural_processing" in summary
        assert "decoding_performance" in summary
        assert "claude_integration" in summary
        assert "system_health" in summary


class TestAlertManager:
    """Test suite for AlertManager."""
    
    def test_alert_manager_initialization(self):
        """Test alert manager initialization."""
        manager = AlertManager()
        
        assert len(manager.active_alerts) == 0
        assert len(manager.alert_history) == 0
        assert len(manager.alert_rules) == 0
    
    def test_register_alert_rule(self):
        """Test registering alert rules."""
        manager = AlertManager()
        
        rule = AlertRule(
            name="test_rule",
            condition=lambda ctx: ctx.get("value", 0) > 100,
            severity=AlertSeverity.WARNING,
            message_template="Value too high: {value}"
        )
        
        manager.register_alert_rule(rule)
        assert "test_rule" in manager.alert_rules
    
    def test_register_notification_handler(self):
        """Test registering notification handlers."""
        manager = AlertManager()
        
        def test_handler(alert):
            pass
        
        manager.register_notification_handler("test", test_handler)
        assert "test" in manager.notification_handlers
    
    @pytest.mark.asyncio
    async def test_evaluate_rules_trigger_alert(self):
        """Test rule evaluation and alert triggering."""
        manager = AlertManager()
        
        # Mock handler to capture alerts
        triggered_alerts = []
        def mock_handler(alert):
            triggered_alerts.append(alert)
        
        manager.register_notification_handler("mock", mock_handler)
        manager.escalation_policies[AlertSeverity.WARNING] = ["mock"]
        
        # Register rule
        rule = AlertRule(
            name="high_value",
            condition=lambda ctx: ctx.get("value", 0) > 100,
            severity=AlertSeverity.WARNING,
            message_template="Value is {value}"
        )
        manager.register_alert_rule(rule)
        
        # Trigger rule
        context = {"value": 150}
        alerts = await manager.evaluate_rules(context)
        
        assert len(alerts) == 1
        assert alerts[0].name == "high_value"
        assert alerts[0].severity == AlertSeverity.WARNING
        assert "150" in alerts[0].message
    
    @pytest.mark.asyncio
    async def test_evaluate_rules_no_trigger(self):
        """Test rule evaluation without triggering."""
        manager = AlertManager()
        
        rule = AlertRule(
            name="high_value",
            condition=lambda ctx: ctx.get("value", 0) > 100,
            severity=AlertSeverity.WARNING,
            message_template="Value is {value}"
        )
        manager.register_alert_rule(rule)
        
        # Don't trigger rule
        context = {"value": 50}
        alerts = await manager.evaluate_rules(context)
        
        assert len(alerts) == 0
    
    def test_acknowledge_alert(self):
        """Test alert acknowledgment."""
        manager = AlertManager()
        
        alert = Alert(
            id="test_alert",
            name="test",
            severity=AlertSeverity.INFO,
            status=AlertStatus.ACTIVE,
            message="Test alert",
            details={},
            created_at=time.time()
        )
        
        manager.active_alerts["test_alert"] = alert
        
        success = manager.acknowledge_alert("test_alert", "test_user")
        
        assert success
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_at is not None
    
    def test_resolve_alert(self):
        """Test alert resolution."""
        manager = AlertManager()
        
        alert = Alert(
            id="test_alert",
            name="test",
            severity=AlertSeverity.INFO,
            status=AlertStatus.ACTIVE,
            message="Test alert",
            details={},
            created_at=time.time()
        )
        
        manager.active_alerts["test_alert"] = alert
        
        success = manager.resolve_alert("test_alert", "test_user")
        
        assert success
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None
        assert "test_alert" not in manager.active_alerts
    
    def test_alert_summary(self):
        """Test alert summary generation."""
        manager = AlertManager()
        
        # Add some test alerts
        alert1 = Alert(
            id="alert1",
            name="test1",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            message="Warning alert",
            details={},
            created_at=time.time()
        )
        
        alert2 = Alert(
            id="alert2", 
            name="test2",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.ACTIVE,
            message="Critical alert",
            details={},
            created_at=time.time()
        )
        
        manager.active_alerts["alert1"] = alert1
        manager.active_alerts["alert2"] = alert2
        
        summary = manager.get_alert_summary()
        
        assert summary["total_active"] == 2
        assert summary["active_by_severity"]["warning"] == 1
        assert summary["active_by_severity"]["critical"] == 1


class TestBCIHealthChecks:
    """Test BCI-specific health check creation."""
    
    def test_create_bci_health_checks(self):
        """Test creation of BCI health checks."""
        # Mock BCI bridge
        mock_bridge = Mock()
        mock_bridge.get_device_info.return_value = {
            'connected': True,
            'device': 'Simulation',
            'channels': 8
        }
        mock_bridge.data_buffer = [Mock(), Mock(), Mock()]
        mock_bridge.get_buffer.return_value = np.random.randn(8, 100)
        mock_bridge.decoder = Mock()
        mock_bridge.decoder.get_decoder_info.return_value = {
            'calibrated': True,
            'last_confidence': 0.8
        }
        
        health_checks = create_bci_health_checks(mock_bridge)
        
        assert "bci_device" in health_checks
        assert "signal_quality" in health_checks 
        assert "decoder_status" in health_checks
        
        # Test device health check
        device_check = health_checks["bci_device"]()
        assert device_check.status == HealthStatus.HEALTHY
        
        # Test signal quality check
        quality_check = health_checks["signal_quality"]()
        assert quality_check.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        
        # Test decoder check
        decoder_check = health_checks["decoder_status"]()
        assert decoder_check.status == HealthStatus.HEALTHY


class TestBCIAlertRules:
    """Test BCI-specific alert rules."""
    
    def test_create_bci_alert_rules(self):
        """Test creation of BCI alert rules."""
        rules = create_bci_alert_rules()
        
        assert len(rules) > 0
        
        rule_names = [rule.name for rule in rules]
        assert "low_signal_quality" in rule_names
        assert "no_neural_data" in rule_names
        assert "low_decoding_confidence" in rule_names
        assert "high_cpu_usage" in rule_names
    
    def test_low_signal_quality_rule(self):
        """Test low signal quality alert rule."""
        rules = create_bci_alert_rules()
        
        rule = next(r for r in rules if r.name == "low_signal_quality")
        
        # Should trigger
        assert rule.condition({"signal_quality": 0.2})
        
        # Should not trigger
        assert not rule.condition({"signal_quality": 0.8})
    
    def test_emergency_signal_rule(self):
        """Test emergency signal detection rule."""
        rules = create_bci_alert_rules()
        
        rule = next(r for r in rules if r.name == "emergency_signal_detected")
        
        # Should trigger
        assert rule.condition({"emergency_detected": True})
        assert rule.severity == AlertSeverity.EMERGENCY
        
        # Should not trigger
        assert not rule.condition({"emergency_detected": False})


class TestDefaultNotificationHandlers:
    """Test default notification handlers."""
    
    def test_create_default_handlers(self):
        """Test creation of default notification handlers."""
        handlers = create_default_notification_handlers()
        
        expected_handlers = ["log", "console", "email", "sms", "pager"]
        for handler_name in expected_handlers:
            assert handler_name in handlers
    
    def test_log_handler(self, caplog):
        """Test log notification handler."""
        handlers = create_default_notification_handlers()
        log_handler = handlers["log"]
        
        alert = Alert(
            id="test",
            name="test_alert", 
            severity=AlertSeverity.WARNING,
            status=AlertStatus.ACTIVE,
            message="Test alert message",
            details={},
            created_at=time.time()
        )
        
        log_handler(alert)
        
        assert "ALERT [WARNING] test_alert: Test alert message" in caplog.text
    
    @patch('builtins.print')
    def test_console_handler(self, mock_print):
        """Test console notification handler."""
        handlers = create_default_notification_handlers()
        console_handler = handlers["console"]
        
        alert = Alert(
            id="test",
            name="test_alert",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.ACTIVE, 
            message="Critical test alert",
            details={},
            created_at=time.time()
        )
        
        console_handler(alert)
        
        mock_print.assert_called_once()
        args = mock_print.call_args[0]
        assert "🚨 ALERT: Critical test alert" in args[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])