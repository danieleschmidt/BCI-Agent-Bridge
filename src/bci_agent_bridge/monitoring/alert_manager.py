"""
Enterprise-grade alert management system for critical BCI events.
Includes advanced escalation rules, notification throttling, and comprehensive alert lifecycle management.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import smtplib
import requests
from email.mime.text import MIMEText
from collections import defaultdict
import uuid
import statistics
from concurrent.futures import ThreadPoolExecutor


class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"


@dataclass
class EscalationPolicy:
    """Defines escalation policy for alerts."""
    levels: List[Dict[str, Any]]
    time_intervals: List[int]  # Escalation time intervals in seconds
    max_escalations: int = 3
    notification_throttle: int = 300  # Throttle repeat notifications (seconds)
    

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
    tags: Dict[str, str] = field(default_factory=dict)
    escalation_level: int = 0
    last_escalation_at: Optional[float] = None
    escalation_count: int = 0
    suppressed_until: Optional[float] = None
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    source_system: str = "bci-bridge"
    runbook_url: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "correlation_id": self.correlation_id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "details": self.details,
            "created_at": self.created_at,
            "acknowledged_at": self.acknowledged_at,
            "resolved_at": self.resolved_at,
            "tags": self.tags,
            "escalation_level": self.escalation_level,
            "escalation_count": self.escalation_count,
            "last_escalation_at": self.last_escalation_at,
            "suppressed_until": self.suppressed_until,
            "source_system": self.source_system,
            "runbook_url": self.runbook_url
        }
    
    def is_sla_violated(self, thresholds: Dict[str, float]) -> Dict[str, bool]:
        """Check if alert violates SLA thresholds."""
        current_time = time.time()
        violations = {}
        
        # Check acknowledgement SLA
        ack_threshold = thresholds.get('acknowledge', float('inf'))
        if self.status == AlertStatus.ACTIVE and self.acknowledged_at is None:
            violations['acknowledge'] = (current_time - self.created_at) > ack_threshold
        else:
            violations['acknowledge'] = False
            
        # Check resolution SLA
        resolve_threshold = thresholds.get('resolve', float('inf'))
        if self.status != AlertStatus.RESOLVED:
            violations['resolve'] = (current_time - self.created_at) > resolve_threshold
        else:
            violations['resolve'] = False
            
        return violations


class AlertRule:
    """Enhanced alert rule with escalation and auto-resolution capabilities."""
    
    def __init__(self, name: str, condition: Callable[[Dict[str, Any]], bool], 
                 severity: AlertSeverity, message_template: str,
                 escalation_policy: Optional[EscalationPolicy] = None,
                 auto_resolve_condition: Optional[Callable[[Dict[str, Any]], bool]] = None,
                 suppression_conditions: Optional[List[Callable[[Dict[str, Any]], bool]]] = None,
                 runbook_url: Optional[str] = None,
                 tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.condition = condition
        self.severity = severity
        self.message_template = message_template
        self.escalation_policy = escalation_policy
        self.auto_resolve_condition = auto_resolve_condition
        self.suppression_conditions = suppression_conditions or []
        self.runbook_url = runbook_url
        self.tags = tags or {}
        
        # Tracking
        self.last_triggered = 0.0
        self.cooldown_period = 300.0  # 5 minutes default
        self.trigger_count = 0
        self.resolved_count = 0
        self.last_evaluation_time = 0.0
        self.evaluation_count = 0
        
    def should_suppress(self, context: Dict[str, Any]) -> bool:
        """Check if alert should be suppressed based on conditions."""
        for condition in self.suppression_conditions:
            try:
                if condition(context):
                    return True
            except Exception:
                continue
        return False


class AlertManager:
    """
    Enterprise-grade alert manager for BCI bridge components with advanced escalation,
    notification throttling, and comprehensive alert lifecycle management.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Core alert storage
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.alert_rules: Dict[str, AlertRule] = {}
        
        # Notification system
        self.notification_handlers: Dict[str, Callable] = {}
        self.escalation_policies: Dict[AlertSeverity, List[str]] = {}
        self.notification_history: Dict[str, List[float]] = defaultdict(list)
        self.notification_failures: Dict[str, int] = defaultdict(int)
        
        # Advanced features
        self.suppression_rules: Set[str] = set()
        self.maintenance_windows: List[Dict[str, Any]] = []
        self.alert_correlations: Dict[str, List[str]] = defaultdict(list)
        self.sla_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Thread pool for async notifications
        self.notification_executor = ThreadPoolExecutor(max_workers=10, thread_name_prefix="alert-notifier")
        
        # Performance tracking
        self.metrics = {
            'total_alerts_triggered': 0,
            'alerts_by_severity': defaultdict(int),
            'mean_time_to_acknowledge': 0.0,
            'mean_time_to_resolve': 0.0,
            'escalation_rate': 0.0,
            'notification_failure_rate': 0.0,
            'alert_correlations_found': 0,
            'sla_violations': defaultdict(int)
        }
        
        # Default escalation policies with time-based escalation
        self._setup_default_escalation_policies()
        
        # Default SLA thresholds (in seconds)
        self._setup_default_sla_thresholds()
        
        # Initialize notification handlers
        self.notification_handlers.update(create_default_notification_handlers())
        
        # Escalation background task
        self._escalation_task: Optional[asyncio.Task] = None
        self._setup_escalation_scheduler()

    def _setup_default_escalation_policies(self):
        """Setup default escalation policies."""
        self.escalation_policies = {
            AlertSeverity.INFO: ["log"],
            AlertSeverity.WARNING: ["log", "console"],
            AlertSeverity.CRITICAL: ["log", "console", "email"],
            AlertSeverity.EMERGENCY: ["log", "console", "email", "sms", "pager"]
        }

    def _setup_default_sla_thresholds(self):
        """Setup default SLA thresholds."""
        self.sla_thresholds = {
            AlertSeverity.EMERGENCY.value: {'acknowledge': 60, 'resolve': 300},  # 1min ack, 5min resolve
            AlertSeverity.CRITICAL.value: {'acknowledge': 300, 'resolve': 1800},  # 5min ack, 30min resolve
            AlertSeverity.WARNING.value: {'acknowledge': 1800, 'resolve': 7200},  # 30min ack, 2hr resolve
            AlertSeverity.INFO.value: {'acknowledge': 3600, 'resolve': 86400}  # 1hr ack, 1day resolve
        }

    def _setup_escalation_scheduler(self):
        """Setup the escalation scheduler."""
        if not self._escalation_task or self._escalation_task.done():
            self._escalation_task = asyncio.create_task(self._escalation_scheduler_loop())

    async def _escalation_scheduler_loop(self):
        """Background task to handle alert escalations."""
        while True:
            try:
                await self._process_escalations()
                await asyncio.sleep(30)  # Check every 30 seconds
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Escalation scheduler error: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait longer on error

    async def _process_escalations(self):
        """Process pending alert escalations."""
        current_time = time.time()
        
        for alert in list(self.active_alerts.values()):
            if alert.status not in [AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED]:
                continue
                
            rule = self.alert_rules.get(alert.name)
            if not rule or not rule.escalation_policy:
                continue
                
            policy = rule.escalation_policy
            
            # Check if escalation is due
            time_since_created = current_time - alert.created_at
            time_since_last_escalation = current_time - (alert.last_escalation_at or alert.created_at)
            
            # Find appropriate escalation level
            escalation_due = False
            target_level = alert.escalation_level
            
            for i, interval in enumerate(policy.time_intervals):
                if time_since_created >= interval and i >= alert.escalation_level:
                    escalation_due = True
                    target_level = i + 1
                    break
            
            if escalation_due and alert.escalation_count < policy.max_escalations:
                await self._escalate_alert(alert, target_level)

    async def _escalate_alert(self, alert: Alert, escalation_level: int):
        """Escalate an alert to the next level."""
        current_time = time.time()
        
        alert.escalation_level = escalation_level
        alert.escalation_count += 1
        alert.last_escalation_at = current_time
        
        self.logger.warning(
            f"Escalating alert '{alert.name}' to level {escalation_level}",
            extra={'correlation_id': alert.correlation_id}
        )
        
        # Send escalated notifications
        await self._send_notifications(alert, escalation_level)
        
        # Update metrics
        self.metrics['escalation_rate'] = len([a for a in self.active_alerts.values() 
                                             if a.escalation_level > 0]) / max(len(self.active_alerts), 1)

    def register_alert_rule(self, rule: AlertRule) -> None:
        """Register an enhanced alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Registered alert rule: {rule.name}")

    def register_notification_handler(self, name: str, handler: Callable[[Alert, int], None]) -> None:
        """Register a notification handler."""
        self.notification_handlers[name] = handler
        self.logger.info(f"Registered notification handler: {name}")

    def set_escalation_policy(self, severity: AlertSeverity, handlers: List[str]) -> None:
        """Set escalation policy for a severity level."""
        self.escalation_policies[severity] = handlers
        self.logger.info(f"Set escalation policy for {severity.value}: {handlers}")

    def add_maintenance_window(self, start_time: float, end_time: float, 
                              description: str, affected_systems: List[str] = None) -> str:
        """Add a maintenance window to suppress alerts."""
        window_id = str(uuid.uuid4())
        maintenance_window = {
            'id': window_id,
            'start': start_time,
            'end': end_time,
            'description': description,
            'affected_systems': affected_systems or [],
            'created_at': time.time()
        }
        
        self.maintenance_windows.append(maintenance_window)
        self.logger.info(f"Added maintenance window: {description} ({window_id})")
        return window_id

    def _is_in_maintenance_window(self, current_time: float) -> bool:
        """Check if current time is in a maintenance window."""
        for window in self.maintenance_windows:
            if window['start'] <= current_time <= window['end']:
                return True
        return False

    def _is_suppressed(self, rule: AlertRule, context: Dict[str, Any], current_time: float) -> bool:
        """Check if alert rule is suppressed."""
        # Check global suppression
        if rule.name in self.suppression_rules:
            return True
            
        # Check rule-specific suppression conditions
        if rule.should_suppress(context):
            return True
            
        # Check cooldown period
        if current_time - rule.last_triggered < rule.cooldown_period:
            return True
            
        return False

    async def _create_alert(self, rule: AlertRule, context: Dict[str, Any], current_time: float) -> Alert:
        """Create a new alert from rule and context."""
        alert_id = f"{rule.name}_{int(current_time)}_{uuid.uuid4().hex[:8]}"
        
        try:
            message = rule.message_template.format(**context)
        except KeyError as e:
            self.logger.warning(f"Template formatting error for rule '{rule.name}': {e}")
            message = f"{rule.message_template} (context: {context})"
        
        # Merge rule tags with context tags
        tags = rule.tags.copy()
        tags.update(context.get('tags', {}))
        tags['rule'] = rule.name
        
        alert = Alert(
            id=alert_id,
            name=rule.name,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            message=message,
            details=context.copy(),
            created_at=current_time,
            tags=tags,
            runbook_url=rule.runbook_url
        )
        
        await self.trigger_alert(alert)
        rule.last_triggered = current_time
        
        return alert

    async def _auto_resolve_alert(self, alert_id: str) -> bool:
        """Auto-resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = time.time()
            alert.details["auto_resolved"] = True
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Auto-resolved alert: {alert_id}", 
                           extra={'correlation_id': alert.correlation_id})
            return True
        return False

    async def _correlate_alert(self, alert: Alert):
        """Find and store alert correlations."""
        correlations = []
        
        # Look for similar active alerts
        for existing_alert in self.active_alerts.values():
            if existing_alert.id == alert.id:
                continue
                
            # Simple correlation based on name similarity or tags
            correlation_score = 0
            
            # Same alert name
            if existing_alert.name == alert.name:
                correlation_score += 0.5
                
            # Similar tags
            common_tags = set(existing_alert.tags.keys()) & set(alert.tags.keys())
            if common_tags:
                correlation_score += len(common_tags) * 0.1
                
            # Same severity
            if existing_alert.severity == alert.severity:
                correlation_score += 0.2
                
            # Time proximity (within 5 minutes)
            if abs(existing_alert.created_at - alert.created_at) < 300:
                correlation_score += 0.3
                
            if correlation_score >= 0.5:  # Correlation threshold
                correlations.append(existing_alert.id)
        
        if correlations:
            self.alert_correlations[alert.id] = correlations
            self.metrics['alert_correlations_found'] += 1
            self.logger.info(f"Found {len(correlations)} correlations for alert {alert.id}")

    async def _send_notifications(self, alert: Alert, escalation_level: int = 0):
        """Send notifications with throttling and failure handling."""
        handlers = self.escalation_policies.get(alert.severity, ["log"])
        current_time = time.time()
        
        for handler_name in handlers:
            if handler_name not in self.notification_handlers:
                self.logger.warning(f"Handler '{handler_name}' not registered")
                continue
                
            # Check notification throttling
            handler_key = f"{alert.correlation_id}_{handler_name}"
            last_notifications = self.notification_history[handler_key]
            
            # Remove old notification times (outside throttle window)
            throttle_window = 300  # 5 minutes default
            last_notifications[:] = [t for t in last_notifications 
                                   if current_time - t < throttle_window]
            
            # Check if we should throttle
            if len(last_notifications) >= 3:  # Max 3 notifications per throttle window
                self.logger.debug(f"Throttling notifications for {handler_key}")
                continue
            
            # Send notification
            try:
                handler = self.notification_handlers[handler_name]
                await asyncio.get_event_loop().run_in_executor(
                    self.notification_executor, handler, alert, escalation_level
                )
                
                # Record successful notification
                last_notifications.append(current_time)
                self.notification_failures[handler_name] = 0
                
            except Exception as e:
                self.logger.error(f"Notification handler '{handler_name}' failed: {e}", exc_info=True)
                self.notification_failures[handler_name] += 1
                
                # Update failure rate metric
                total_attempts = sum(len(hist) for hist in self.notification_history.values())
                total_failures = sum(self.notification_failures.values())
                self.metrics['notification_failure_rate'] = total_failures / max(total_attempts, 1)

    def _check_sla_compliance(self, alert: Alert):
        """Check and record SLA compliance for alert."""
        thresholds = self.sla_thresholds.get(alert.severity.value, {})
        violations = alert.is_sla_violated(thresholds)
        
        for sla_type, is_violated in violations.items():
            if is_violated:
                self.metrics['sla_violations'][f"{alert.severity.value}_{sla_type}"] += 1
                self.logger.warning(
                    f"SLA violation: {sla_type} threshold exceeded for alert {alert.id}",
                    extra={'correlation_id': alert.correlation_id}
                )

    async def evaluate_rules(self, context: Dict[str, Any]) -> List[Alert]:
        """Evaluate all alert rules against current context with advanced features."""
        triggered_alerts = []
        resolved_alerts = []
        current_time = time.time()
        
        # Check for maintenance windows
        if self._is_in_maintenance_window(current_time):
            self.logger.debug("Skipping rule evaluation - in maintenance window")
            return triggered_alerts
        
        for rule_name, rule in self.alert_rules.items():
            rule.evaluation_count += 1
            rule.last_evaluation_time = current_time
            
            try:
                # Check suppression conditions
                if self._is_suppressed(rule, context, current_time):
                    continue
                
                # Check if condition is met
                condition_met = rule.condition(context)
                
                # Handle auto-resolution for existing alerts
                if rule.auto_resolve_condition and not condition_met:
                    existing_alerts = [a for a in self.active_alerts.values() 
                                     if a.name == rule_name and a.status == AlertStatus.ACTIVE]
                    
                    for alert in existing_alerts:
                        if rule.auto_resolve_condition(context):
                            await self._auto_resolve_alert(alert.id)
                            resolved_alerts.append(alert)
                            rule.resolved_count += 1
                
                # Trigger new alert if condition is met
                if condition_met:
                    # Check for duplicate active alerts
                    existing_active = [a for a in self.active_alerts.values() 
                                     if a.name == rule_name and a.status == AlertStatus.ACTIVE]
                    
                    if not existing_active:  # Only create if no active alerts exist
                        alert = await self._create_alert(rule, context, current_time)
                        triggered_alerts.append(alert)
                        rule.trigger_count += 1
                        self.metrics['total_alerts_triggered'] += 1
                        self.metrics['alerts_by_severity'][rule.severity] += 1
                    
            except Exception as e:
                self.logger.error(f"Error evaluating rule '{rule_name}': {e}", exc_info=True)
        
        # Log summary
        if triggered_alerts or resolved_alerts:
            self.logger.info(f"Rule evaluation complete: {len(triggered_alerts)} triggered, {len(resolved_alerts)} resolved")
        
        return triggered_alerts

    async def trigger_alert(self, alert: Alert) -> None:
        """Trigger an alert with enhanced notification and correlation."""
        self.logger.info(f"Triggering alert: {alert.name} - {alert.message}", 
                        extra={'correlation_id': alert.correlation_id})
        
        # Store alert
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Check for alert correlations
        await self._correlate_alert(alert)
        
        # Send initial notifications
        await self._send_notifications(alert, escalation_level=0)
        
        # Check SLA thresholds
        self._check_sla_compliance(alert)

    def acknowledge_alert(self, alert_id: str, acknowledged_by: str = "system", 
                         note: str = "") -> bool:
        """Acknowledge an active alert with enhanced tracking."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = time.time()
            alert.details["acknowledged_by"] = acknowledged_by
            if note:
                alert.details["acknowledgment_note"] = note
            
            self.logger.info(f"Alert acknowledged: {alert_id} by {acknowledged_by}", 
                           extra={'correlation_id': alert.correlation_id})
            return True
        return False

    def resolve_alert(self, alert_id: str, resolved_by: str = "system", 
                     resolution_note: str = "") -> bool:
        """Resolve an active alert with enhanced tracking."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = time.time()
            alert.details["resolved_by"] = resolved_by
            if resolution_note:
                alert.details["resolution_note"] = resolution_note
            
            # Remove from active alerts
            del self.active_alerts[alert_id]
            
            self.logger.info(f"Alert resolved: {alert_id} by {resolved_by}", 
                           extra={'correlation_id': alert.correlation_id})
            return True
        return False

    def suppress_alert(self, alert_id: str, suppress_until: float, 
                      reason: str = "") -> bool:
        """Suppress an alert until specified time."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.status = AlertStatus.SUPPRESSED
            alert.suppressed_until = suppress_until
            alert.details["suppression_reason"] = reason
            
            self.logger.info(f"Alert suppressed: {alert_id} until {suppress_until}", 
                           extra={'correlation_id': alert.correlation_id})
            return True
        return False

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None,
                         include_suppressed: bool = False) -> List[Alert]:
        """Get active alerts with filtering options."""
        current_time = time.time()
        alerts = []
        
        for alert in self.active_alerts.values():
            # Check if suppression has expired
            if (alert.status == AlertStatus.SUPPRESSED and 
                alert.suppressed_until and alert.suppressed_until <= current_time):
                alert.status = AlertStatus.ACTIVE
                alert.suppressed_until = None
            
            # Apply filters
            if not include_suppressed and alert.status == AlertStatus.SUPPRESSED:
                continue
                
            if severity and alert.severity != severity:
                continue
                
            alerts.append(alert)
        
        return sorted(alerts, key=lambda a: a.created_at, reverse=True)

    def _calculate_sla_violations(self) -> Dict[str, int]:
        """Calculate current SLA violations."""
        violations = defaultdict(int)
        current_time = time.time()
        
        for alert in self.active_alerts.values():
            thresholds = self.sla_thresholds.get(alert.severity.value, {})
            alert_violations = alert.is_sla_violated(thresholds)
            
            for sla_type, is_violated in alert_violations.items():
                if is_violated:
                    violations[f"{alert.severity.value}_{sla_type}"] += 1
        
        return dict(violations)

    def _update_performance_metrics(self):
        """Update performance metrics."""
        if not self.alert_history:
            return
            
        # Calculate mean time to acknowledge
        ack_times = [alert.acknowledged_at - alert.created_at 
                    for alert in self.alert_history[-100:]  # Last 100 alerts
                    if alert.acknowledged_at]
        
        if ack_times:
            self.metrics['mean_time_to_acknowledge'] = statistics.mean(ack_times)
        
        # Calculate mean time to resolve
        resolve_times = [alert.resolved_at - alert.created_at 
                        for alert in self.alert_history[-100:]
                        if alert.resolved_at]
        
        if resolve_times:
            self.metrics['mean_time_to_resolve'] = statistics.mean(resolve_times)

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of alert status and metrics."""
        active_by_severity = {}
        for severity in AlertSeverity:
            active_by_severity[severity.value] = len([
                a for a in self.active_alerts.values() if a.severity == severity
            ])
        
        # Calculate SLA compliance
        sla_violations = self._calculate_sla_violations()
        
        # Calculate mean times
        self._update_performance_metrics()
        
        return {
            "alerts": {
                "total_active": len(self.active_alerts),
                "active_by_severity": active_by_severity,
                "total_history": len(self.alert_history),
                "escalated_alerts": len([a for a in self.active_alerts.values() if a.escalation_level > 0]),
                "suppressed_alerts": len([a for a in self.active_alerts.values() 
                                         if a.status == AlertStatus.SUPPRESSED]),
                "correlated_alerts": len(self.alert_correlations)
            },
            "rules": {
                "registered_rules": len(self.alert_rules),
                "suppression_rules": list(self.suppression_rules),
                "rules_with_escalation": len([r for r in self.alert_rules.values() if r.escalation_policy])
            },
            "notifications": {
                "handlers": list(self.notification_handlers.keys()),
                "failure_rate": self.metrics['notification_failure_rate'],
                "total_failures": sum(self.notification_failures.values())
            },
            "sla": {
                "violations": sla_violations,
                "mean_time_to_acknowledge": self.metrics['mean_time_to_acknowledge'],
                "mean_time_to_resolve": self.metrics['mean_time_to_resolve']
            },
            "performance": {
                "total_alerts_triggered": self.metrics['total_alerts_triggered'],
                "escalation_rate": self.metrics['escalation_rate'],
                "alerts_by_severity": dict(self.metrics['alerts_by_severity']),
                "correlations_found": self.metrics['alert_correlations_found']
            },
            "maintenance": {
                "active_windows": len([w for w in self.maintenance_windows 
                                      if w['start'] <= time.time() <= w['end']]),
                "total_windows": len(self.maintenance_windows)
            }
        }

    def export_alerts(self, format: str = "json", include_history: bool = False,
                     include_correlations: bool = True) -> str:
        """Export alerts with enhanced formatting options."""
        alerts_to_export = list(self.active_alerts.values())
        
        if include_history:
            alerts_to_export.extend(self.alert_history)
        
        if format.lower() == "json":
            export_data = {
                "timestamp": time.time(),
                "alerts": [alert.to_dict() for alert in alerts_to_export],
                "summary": self.get_alert_summary()
            }
            
            if include_correlations:
                export_data["correlations"] = dict(self.alert_correlations)
            
            return json.dumps(export_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported export format: {format}")

    async def cleanup(self):
        """Cleanup resources."""
        if self._escalation_task:
            self._escalation_task.cancel()
            try:
                await self._escalation_task
            except asyncio.CancelledError:
                pass
                
        self.notification_executor.shutdown(wait=True)
        self.logger.info("Alert manager cleanup complete")


def create_default_notification_handlers() -> Dict[str, Callable]:
    """Create enhanced default notification handlers with throttling and formatting."""
    
    def log_handler(alert: Alert, escalation_level: int = 0) -> None:
        """Enhanced log alert handler with structured logging."""
        logger = logging.getLogger("alerts")
        log_func = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.error,
            AlertSeverity.EMERGENCY: logger.critical
        }.get(alert.severity, logger.info)
        
        escalation_info = f" [ESCALATION-L{escalation_level}]" if escalation_level > 0 else ""
        
        log_func(
            f"ALERT{escalation_info} [{alert.severity.value.upper()}] {alert.name}: {alert.message}",
            extra={
                'alert_id': alert.id,
                'correlation_id': alert.correlation_id,
                'severity': alert.severity.value,
                'escalation_level': escalation_level,
                'tags': alert.tags,
                'source_system': alert.source_system
            }
        )

    def console_handler(alert: Alert, escalation_level: int = 0) -> None:
        """Enhanced console alert handler with formatting."""
        severity_symbols = {
            AlertSeverity.INFO: "ℹ️",
            AlertSeverity.WARNING: "⚠️",
            AlertSeverity.CRITICAL: "🚨",
            AlertSeverity.EMERGENCY: "🆘"
        }
        
        symbol = severity_symbols.get(alert.severity, "❓")
        escalation_info = f" [ESC-L{escalation_level}]" if escalation_level > 0 else ""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(alert.created_at))
        
        print(f"\n{symbol} ALERT{escalation_info} [{timestamp}] {alert.name}")
        print(f"   Severity: {alert.severity.value.upper()}")
        print(f"   Message: {alert.message}")
        print(f"   Correlation ID: {alert.correlation_id}")
        if alert.runbook_url:
            print(f"   Runbook: {alert.runbook_url}")
        if alert.tags:
            print(f"   Tags: {', '.join(f'{k}={v}' for k, v in alert.tags.items())}")
        print()

    def email_handler(alert: Alert, escalation_level: int = 0) -> None:
        """Enhanced email alert handler with HTML formatting."""
        logger = logging.getLogger("alerts.email")
        
        try:
            escalation_info = f" [ESCALATION LEVEL {escalation_level}]" if escalation_level > 0 else ""
            subject = f"BCI Alert{escalation_info}: {alert.severity.value.upper()} - {alert.name}"
            
            # HTML email body
            body = f"""
            <html><body>
            <h2>BCI Agent Bridge Alert{escalation_info}</h2>
            <table border="1" cellpadding="5" style="border-collapse: collapse;">
            <tr><td><b>Alert Name:</b></td><td>{alert.name}</td></tr>
            <tr><td><b>Severity:</b></td><td style="color: {'red' if alert.severity in [AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY] else 'orange' if alert.severity == AlertSeverity.WARNING else 'blue'};">{alert.severity.value.upper()}</td></tr>
            <tr><td><b>Status:</b></td><td>{alert.status.value}</td></tr>
            <tr><td><b>Message:</b></td><td>{alert.message}</td></tr>
            <tr><td><b>Created:</b></td><td>{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(alert.created_at))}</td></tr>
            <tr><td><b>Correlation ID:</b></td><td>{alert.correlation_id}</td></tr>
            {"<tr><td><b>Runbook:</b></td><td><a href='" + alert.runbook_url + "'>View Runbook</a></td></tr>" if alert.runbook_url else ""}
            </table>
            
            <h3>Details:</h3>
            <pre style="background-color: #f5f5f5; padding: 10px;">{json.dumps(alert.details, indent=2)}</pre>
            
            <h3>Tags:</h3>
            <ul>
            {''.join(f'<li><b>{k}:</b> {v}</li>' for k, v in alert.tags.items())}
            </ul>
            
            <p style="color: #666; font-size: 12px;">
            Generated by BCI Agent Bridge Alert Manager<br>
            Alert ID: {alert.id}
            </p>
            </body></html>
            """
            
            # For now, just log (implement actual SMTP in production)
            logger.info(f"EMAIL ALERT{escalation_info}: {alert.name} - {alert.message}")
            logger.debug(f"Email would be sent with subject: {subject}")
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def sms_handler(alert: Alert, escalation_level: int = 0) -> None:
        """Enhanced SMS alert handler with rate limiting."""
        logger = logging.getLogger("alerts.sms")
        
        try:
            escalation_info = f" [ESC-L{escalation_level}]" if escalation_level > 0 else ""
            message = f"BCI ALERT{escalation_info}: {alert.severity.value.upper()} - {alert.name}: {alert.message[:100]}..."
            
            # For now, just log (implement actual SMS API in production)
            logger.info(f"SMS ALERT{escalation_info}: {alert.name} - {alert.message}")
            logger.debug(f"SMS would be sent: {message}")
            
        except Exception as e:
            logger.error(f"Failed to send SMS alert: {e}")

    def pager_handler(alert: Alert, escalation_level: int = 0) -> None:
        """Enhanced pager alert handler for critical incidents."""
        logger = logging.getLogger("alerts.pager")
        
        try:
            escalation_info = f" [ESC-L{escalation_level}]" if escalation_level > 0 else ""
            
            # For now, just log (implement actual pager API in production)
            logger.critical(f"PAGER ALERT{escalation_info}: {alert.name} - {alert.message}")
            logger.debug(f"Pager incident would be created for correlation ID: {alert.correlation_id}")
            
        except Exception as e:
            logger.error(f"Failed to trigger pager alert: {e}")
    
    def webhook_handler(alert: Alert, escalation_level: int = 0) -> None:
        """Webhook handler for integration with external systems."""
        logger = logging.getLogger("alerts.webhook")
        
        try:
            escalation_info = f" [ESC-L{escalation_level}]" if escalation_level > 0 else ""
            
            payload = {
                'alert_id': alert.id,
                'correlation_id': alert.correlation_id,
                'name': alert.name,
                'severity': alert.severity.value,
                'status': alert.status.value,
                'message': alert.message,
                'escalation_level': escalation_level,
                'created_at': alert.created_at,
                'tags': alert.tags,
                'details': alert.details,
                'runbook_url': alert.runbook_url
            }
            
            # For now, just log (implement actual webhook call in production)
            logger.info(f"WEBHOOK ALERT{escalation_info}: {alert.name} - {alert.message}")
            logger.debug(f"Webhook payload: {json.dumps(payload, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")

    return {
        "log": log_handler,
        "console": console_handler,
        "email": email_handler,
        "sms": sms_handler,
        "pager": pager_handler,
        "webhook": webhook_handler
    }


def create_enhanced_bci_alert_rules() -> List[AlertRule]:
    """Create enhanced alert rules for BCI components with escalation policies."""
    
    rules = []
    
    # Create escalation policy for critical BCI alerts
    critical_escalation = EscalationPolicy(
        levels=[
            {"handlers": ["log", "console"]},
            {"handlers": ["log", "console", "email"]},
            {"handlers": ["log", "console", "email", "sms"]}
        ],
        time_intervals=[0, 300, 900],  # 0, 5min, 15min
        max_escalations=3,
        notification_throttle=300
    )
    
    # Emergency escalation policy
    emergency_escalation = EscalationPolicy(
        levels=[
            {"handlers": ["log", "console", "email", "sms"]},
            {"handlers": ["log", "console", "email", "sms", "pager"]},
            {"handlers": ["log", "console", "email", "sms", "pager", "webhook"]}
        ],
        time_intervals=[0, 60, 180],  # 0, 1min, 3min
        max_escalations=3,
        notification_throttle=60
    )
    
    # Neural signal quality alerts with auto-resolution
    rules.append(AlertRule(
        name="low_signal_quality",
        condition=lambda ctx: ctx.get("signal_quality", 1.0) < 0.3,
        auto_resolve_condition=lambda ctx: ctx.get("signal_quality", 1.0) >= 0.5,
        severity=AlertSeverity.WARNING,
        message_template="Neural signal quality is low: {signal_quality:.2f}",
        escalation_policy=critical_escalation,
        runbook_url="https://docs.bci-bridge.local/runbooks/signal-quality",
        tags={"component": "neural_interface", "type": "quality"}
    ))
    
    rules.append(AlertRule(
        name="no_neural_data",
        condition=lambda ctx: ctx.get("data_rate", 250) < 50,
        auto_resolve_condition=lambda ctx: ctx.get("data_rate", 250) >= 100,
        severity=AlertSeverity.CRITICAL,
        message_template="Very low neural data rate: {data_rate} Hz",
        escalation_policy=critical_escalation,
        runbook_url="https://docs.bci-bridge.local/runbooks/neural-data",
        tags={"component": "neural_interface", "type": "connectivity"}
    ))
    
    # Decoding performance alerts
    rules.append(AlertRule(
        name="low_decoding_confidence",
        condition=lambda ctx: ctx.get("avg_confidence", 1.0) < 0.5,
        auto_resolve_condition=lambda ctx: ctx.get("avg_confidence", 1.0) >= 0.7,
        severity=AlertSeverity.WARNING,
        message_template="Average decoding confidence is low: {avg_confidence:.2f}",
        runbook_url="https://docs.bci-bridge.local/runbooks/decoding-confidence",
        tags={"component": "decoder", "type": "performance"}
    ))
    
    rules.append(AlertRule(
        name="high_decoding_latency",
        condition=lambda ctx: ctx.get("decoding_latency", 0) > 200,
        auto_resolve_condition=lambda ctx: ctx.get("decoding_latency", 0) <= 150,
        severity=AlertSeverity.WARNING,
        message_template="High decoding latency detected: {decoding_latency}ms",
        escalation_policy=critical_escalation,
        runbook_url="https://docs.bci-bridge.local/runbooks/decoding-latency",
        tags={"component": "decoder", "type": "latency"}
    ))
    
    # System resource alerts
    rules.append(AlertRule(
        name="high_cpu_usage",
        condition=lambda ctx: ctx.get("cpu_usage", 0) > 85,
        auto_resolve_condition=lambda ctx: ctx.get("cpu_usage", 0) <= 70,
        severity=AlertSeverity.WARNING,
        message_template="High CPU usage: {cpu_usage}%",
        suppression_conditions=[lambda ctx: ctx.get("suppress_resource_alerts", False)],
        runbook_url="https://docs.bci-bridge.local/runbooks/high-cpu",
        tags={"component": "system", "type": "resource"}
    ))
    
    rules.append(AlertRule(
        name="high_memory_usage",
        condition=lambda ctx: ctx.get("memory_usage", 0) > 1000,  # 1GB
        auto_resolve_condition=lambda ctx: ctx.get("memory_usage", 0) <= 800,
        severity=AlertSeverity.WARNING,
        message_template="High memory usage: {memory_usage}MB",
        suppression_conditions=[lambda ctx: ctx.get("suppress_resource_alerts", False)],
        runbook_url="https://docs.bci-bridge.local/runbooks/high-memory",
        tags={"component": "system", "type": "resource"}
    ))
    
    # Claude integration alerts
    rules.append(AlertRule(
        name="claude_api_errors",
        condition=lambda ctx: ctx.get("claude_error_rate", 0) > 0.1,
        auto_resolve_condition=lambda ctx: ctx.get("claude_error_rate", 0) <= 0.05,
        severity=AlertSeverity.CRITICAL,
        message_template="High Claude API error rate: {claude_error_rate:.1%}",
        escalation_policy=critical_escalation,
        runbook_url="https://docs.bci-bridge.local/runbooks/claude-api-errors",
        tags={"component": "claude_adapter", "type": "api_error"}
    ))
    
    rules.append(AlertRule(
        name="slow_claude_responses",
        condition=lambda ctx: ctx.get("claude_response_time", 0) > 5000,
        auto_resolve_condition=lambda ctx: ctx.get("claude_response_time", 0) <= 3000,
        severity=AlertSeverity.WARNING,
        message_template="Slow Claude responses: {claude_response_time}ms",
        runbook_url="https://docs.bci-bridge.local/runbooks/claude-performance",
        tags={"component": "claude_adapter", "type": "performance"}
    ))
    
    # Medical safety alerts - highest priority
    rules.append(AlertRule(
        name="emergency_signal_detected",
        condition=lambda ctx: ctx.get("emergency_detected", False),
        severity=AlertSeverity.EMERGENCY,
        message_template="EMERGENCY: Emergency signal detected in neural data - immediate medical attention required",
        escalation_policy=emergency_escalation,
        runbook_url="https://docs.bci-bridge.local/runbooks/emergency-protocol",
        tags={"component": "safety_system", "type": "emergency", "medical": "true"}
    ))
    
    rules.append(AlertRule(
        name="safety_flag_triggered",
        condition=lambda ctx: len(ctx.get("safety_flags", [])) > 0,
        auto_resolve_condition=lambda ctx: len(ctx.get("safety_flags", [])) == 0,
        severity=AlertSeverity.CRITICAL,
        message_template="Safety flags triggered: {safety_flags} - review required",
        escalation_policy=critical_escalation,
        runbook_url="https://docs.bci-bridge.local/runbooks/safety-flags",
        tags={"component": "safety_system", "type": "safety_flag", "medical": "true"}
    ))
    
    # Device connectivity alerts
    rules.append(AlertRule(
        name="device_disconnected",
        condition=lambda ctx: not ctx.get("device_connected", True),
        auto_resolve_condition=lambda ctx: ctx.get("device_connected", True),
        severity=AlertSeverity.CRITICAL,
        message_template="BCI device disconnected - patient monitoring interrupted",
        escalation_policy=critical_escalation,
        runbook_url="https://docs.bci-bridge.local/runbooks/device-disconnect",
        tags={"component": "device", "type": "connectivity", "medical": "true"}
    ))
    
    return rules


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import random
    
    async def demo_enhanced_alert_manager():
        # Create enhanced alert manager
        alert_manager = AlertManager()
        
        # Register enhanced BCI alert rules
        rules = create_enhanced_bci_alert_rules()
        for rule in rules:
            alert_manager.register_alert_rule(rule)
        
        print("Enhanced BCI Alert Manager Demo")
        print("=" * 50)
        
        # Simulate various alert conditions
        test_contexts = [
            {"signal_quality": 0.2, "data_rate": 250, "cpu_usage": 45},
            {"signal_quality": 0.8, "data_rate": 30, "claude_error_rate": 0.15},
            {"emergency_detected": True, "safety_flags": ["neural_anomaly", "high_voltage"]},
            {"cpu_usage": 90, "memory_usage": 1200, "device_connected": False},
            {"signal_quality": 0.6, "data_rate": 180, "claude_response_time": 6000}
        ]
        
        # Trigger alerts
        for i, context in enumerate(test_contexts):
            print(f"\n--- Test Context {i+1} ---")
            print(f"Context: {context}")
            
            alerts = await alert_manager.evaluate_rules(context)
            if alerts:
                print(f"Triggered {len(alerts)} alerts:")
                for alert in alerts:
                    print(f"  - {alert.name}: {alert.message}")
            else:
                print("No alerts triggered")
        
        # Wait for escalations
        print("\nWaiting for escalations...")
        await asyncio.sleep(2)
        
        # Show summary
        print("\n--- Alert Summary ---")
        summary = alert_manager.get_alert_summary()
        print(json.dumps(summary, indent=2, default=str))
        
        # Cleanup
        await alert_manager.cleanup()
    
    asyncio.run(demo_enhanced_alert_manager())