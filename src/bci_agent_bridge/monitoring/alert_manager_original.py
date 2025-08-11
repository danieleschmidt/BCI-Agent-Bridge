"""
Alert management system for critical BCI events.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass
from enum import Enum
import json


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Alert:
    id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    details: Dict[str, Any]
    created_at: float
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    tags: Dict[str, str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "created_at": self.created_at,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
            "tags": self.tags or {}
        }


class AlertRule:
    """Defines conditions for triggering alerts."""
    
    def __init__(self, name: str, condition: Callable[[Dict[str, Any]], bool], 
                 severity: AlertSeverity, message_template: str):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
        self.last_triggered = 0.0
        self.cooldown_period = 300.0  # 5 minutes default


class AlertManager:
    """
    Manages alerts for BCI bridge components with escalation and notification.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        self.notification_handlers: Dict[str, Callable] = {}
        self.escalation_policies: Dict[AlertSeverity, List[str]] = {}
        self.suppression_rules: Set[str] = set()
        
        # Default escalation policies
        self.escalation_policies = {
            AlertSeverity.INFO: ["log"],
            AlertSeverity.WARNING: ["log", "console"],
            AlertSeverity.CRITICAL: ["log", "console", "email"],
            AlertSeverity.EMERGENCY: ["log", "console", "email", "sms", "pager"]
        }
    
    def register_alert_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Registered alert rule: {rule.name}")
    
    def register_notification_handler(self, name: str, handler: Callable[[Alert], None]) -> None:
        """Register a notification handler."""
        self.notification_handlers[name] = handler
        self.logger.info(f"Registered notification handler: {name}")
    
    def set_escalation_policy(self, severity: AlertSeverity, handlers: List[str]) -> None:
        """Set escalation policy for a severity level."""
        self.escalation_policies[severity] = handlers
        self.logger.info(f"Set escalation policy for {severity.value}: {handlers}")
    
    def add_suppression_rule(self, alert_name: str) -> None:
        """Add alert suppression rule."""
        self.suppression_rules.add(alert_name)
        self.logger.info(f"Added suppression rule for: {alert_name}")
    
    def remove_suppression_rule(self, alert_name: str) -> None:
        """Remove alert suppression rule."""
        self.suppression_rules.discard(alert_name)
        self.logger.info(f"Removed suppression rule for: {alert_name}")
    
    async def evaluate_rules(self, context: Dict[str, Any]) -> List[Alert]:
        """Evaluate all alert rules against current context."""
        triggered_alerts = []
        current_time = time.time()
        
        for rule_name, rule in self.alert_rules.items():
            # Check if rule is suppressed
            if rule_name in self.suppression_rules:
                continue
            
            # Check cooldown period
            if current_time - rule.last_triggered < rule.cooldown_period:
                continue
            
            try:
                if rule.condition(context):
                    # Generate alert
                    alert_id = f"{rule_name}_{int(current_time)}"
                    message = rule.message_template.format(**context)
                    
                    alert = Alert(
                        id=alert_id,
                        name=rule_name,
                        severity=rule.severity,
                        status=AlertStatus.ACTIVE,
                        message=message,
                        details=context.copy(),
                        created_at=current_time,
                        tags={"rule": rule_name}
                    )
                    
                    await self.trigger_alert(alert)
                    triggered_alerts.append(alert)
                    rule.last_triggered = current_time
                    
            except Exception as e:
                self.logger.error(f"Error evaluating rule '{rule_name}': {e}")
        
        return triggered_alerts
    
    async def trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert and handle notifications."""
        self.logger.info(f"Triggering alert: {alert.name} - {alert.message}")
        
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Handle notifications based on escalation policy
        handlers = self.escalation_policies.get(alert.severity, ["log"])
        
        for handler_name in handlers:
            if handler_name in self.notification_handlers:
                try:
                    await asyncio.to_thread(self.notification_handlers[handler_name], alert)
                except Exception as e:
                    self.logger.error(f"Notification handler '{handler_name}' failed: {e}")
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system") -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = time.time()
            alert.details["acknowledged_by"] = acknowledged_by
            
            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}")
            return True
        return False
    
    def resolve_alert(self, alert_id: str, resolved_by: str = "system") -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = time.time()
            alert.details["resolved_by"] = resolved_by
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert resolved: {alert_id} by {resolved_by}")
            return True
        return False
    
    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Alert]:
        """Get active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status."""
        active_by_severity = {}
        for severity in AlertSeverity:
            active_by_severity[severity.value] = len([
                a for a in self.active_alerts.values() if a.severity == severity
            ])
        
        return {
            "total_active": len(self.active_alerts),
            "active_by_severity": active_by_severity,
            "total_history": len(self.alert_history),
            "suppression_rules": list(self.suppression_rules),
            "registered_rules": len(self.alert_rules),
            "notification_handlers": list(self.notification_handlers.keys())
        }
    
    def export_alerts(self, format: str = "json", include_history: bool = False) -> str:
        """Export alerts in specified format."""
        alerts_to_export = list(self.active_alerts.values())
        
        if include_history:
            alerts_to_export.extend(self.alert_history)
        
        if format.lower() == "json":
            return json.dumps([alert.to_dict() for alert in alerts_to_export], indent=2)
        else:
            raise ValueError(f"Unsupported export format: {format}")


def create_bci_alert_rules() -> List[AlertRule]:
    """Create standard alert rules for BCI components."""
    
    rules = []
    
    # Neural signal quality alerts
    rules.append(AlertRule(
        name="low_signal_quality",
        condition=lambda ctx: ctx.get("signal_quality", 1.0) < 0.3,
        severity=AlertSeverity.WARNING,
        message_template="Neural signal quality is low: {signal_quality:.2f}"
    ))
    
    rules.append(AlertRule(
        name="no_neural_data",
        condition=lambda ctx: ctx.get("data_rate", 250) < 50,
        severity=AlertSeverity.CRITICAL,
        message_template="Very low neural data rate: {data_rate} Hz"
    ))
    
    # Decoding performance alerts
    rules.append(AlertRule(
        name="low_decoding_confidence",
        condition=lambda ctx: ctx.get("avg_confidence", 1.0) < 0.5,
        severity=AlertSeverity.WARNING,
        message_template="Average decoding confidence is low: {avg_confidence:.2f}"
    ))
    
    rules.append(AlertRule(
        name="high_decoding_latency",
        condition=lambda ctx: ctx.get("decoding_latency", 0) > 200,
        severity=AlertSeverity.WARNING,
        message_template="High decoding latency detected: {decoding_latency}ms"
    ))
    
    # System resource alerts
    rules.append(AlertRule(
        name="high_cpu_usage",
        condition=lambda ctx: ctx.get("cpu_usage", 0) > 85,
        severity=AlertSeverity.WARNING,
        message_template="High CPU usage: {cpu_usage}%"
    ))
    
    rules.append(AlertRule(
        name="high_memory_usage",
        condition=lambda ctx: ctx.get("memory_usage", 0) > 1000,  # 1GB
        severity=AlertSeverity.WARNING,
        message_template="High memory usage: {memory_usage}MB"
    ))
    
    # Claude integration alerts
    rules.append(AlertRule(
        name="claude_api_errors",
        condition=lambda ctx: ctx.get("claude_error_rate", 0) > 0.1,
        severity=AlertSeverity.CRITICAL,
        message_template="High Claude API error rate: {claude_error_rate:.1%}"
    ))
    
    rules.append(AlertRule(
        name="slow_claude_responses",
        condition=lambda ctx: ctx.get("claude_response_time", 0) > 5000,
        severity=AlertSeverity.WARNING,
        message_template="Slow Claude responses: {claude_response_time}ms"
    ))
    
    # Medical safety alerts
    rules.append(AlertRule(
        name="emergency_signal_detected",
        condition=lambda ctx: ctx.get("emergency_detected", False),
        severity=AlertSeverity.EMERGENCY,
        message_template="Emergency signal detected in neural data"
    ))
    
    rules.append(AlertRule(
        name="safety_flag_triggered",
        condition=lambda ctx: len(ctx.get("safety_flags", [])) > 0,
        severity=AlertSeverity.CRITICAL,
        message_template="Safety flags triggered: {safety_flags}"
    ))
    
    return rules


def create_default_notification_handlers() -> Dict[str, Callable]:
    """Create default notification handlers."""
    
    def log_handler(alert: Alert) -> None:
        """Log alert to system logger."""
        logger = logging.getLogger("alerts")
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
            AlertSeverity.EMERGENCY: logger.critical
        }.get(alert.severity, logger.info)
        
        log_func(f"ALERT [{alert.severity.value.upper()}] {alert.name}: {alert.message}")
    
    def console_handler(alert: Alert) -> None:
        """Print alert to console."""
        severity_symbols = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🆘"
        }
        
        symbol = severity_symbols.get(alert.severity, "❓")
        print(f"{symbol} ALERT: {alert.message}")
    
    def email_handler(alert: Alert) -> None:
        """Email alert handler (placeholder)."""
        # In production, this would send actual emails
        logger = logging.getLogger("alerts.email")
        logger.info(f"EMAIL ALERT: {alert.name} - {alert.message}")
    
    def sms_handler(alert: Alert) -> None:
        """SMS alert handler (placeholder)."""
        # In production, this would send SMS messages
        logger = logging.getLogger("alerts.sms")
        logger.info(f"SMS ALERT: {alert.name} - {alert.message}")
    
    def pager_handler(alert: Alert) -> None:
        """Pager alert handler (placeholder)."""
        # In production, this would trigger pager systems
        logger = logging.getLogger("alerts.pager")
        logger.critical(f"PAGER ALERT: {alert.name} - {alert.message}")
    
    return {
        "log": log_handler,
        "console": console_handler,
        "email": email_handler,
        "sms": sms_handler,
        "pager": pager_handler
    }