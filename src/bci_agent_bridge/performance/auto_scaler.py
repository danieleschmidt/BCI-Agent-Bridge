"""
Auto-scaling system for dynamic BCI processing capacity management.
"""

import asyncio
import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import statistics
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .load_balancer import LoadBalancer, Worker, WorkerState


class ScalingDirection(Enum):
    UP = "up"
    DOWN = "down"
    STABLE = "stable"


class ScalingTrigger(Enum):
    CPU_UTILIZATION = "cpu"
    MEMORY_UTILIZATION = "memory"
    RESPONSE_TIME = "response_time"
    QUEUE_DEPTH = "queue_depth"
    REQUEST_RATE = "request_rate"
    ERROR_RATE = "error_rate"


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_workers: int = 1
    max_workers: int = 10
    target_cpu_utilization: float = 70.0  # Target CPU %
    target_response_time: float = 100.0   # Target response time in ms
    scale_up_threshold: float = 80.0      # Scale up when metric > threshold
    scale_down_threshold: float = 40.0    # Scale down when metric < threshold
    scale_up_cooldown: float = 300.0      # Cooldown in seconds after scaling up
    scale_down_cooldown: float = 600.0    # Cooldown in seconds after scaling down
    evaluation_period: float = 60.0       # Evaluation period in seconds
    consecutive_periods: int = 2          # Periods before scaling action


@dataclass
class ScalingEvent:
    """Record of scaling events."""
    timestamp: float
    direction: ScalingDirection
    trigger: ScalingTrigger
    from_count: int
    to_count: int
    metric_value: float
    reason: str


class NeuralProcessingAutoScaler:
    """
    Advanced auto-scaler for BCI neural processing workloads.
    
    Features:
    - Predictive scaling based on neural processing patterns
    - Multi-metric scaling decisions  
    - Workload-aware scaling policies
    - Cost optimization with performance guarantees
    """
    
    def __init__(self,
                 load_balancer: LoadBalancer,
                 scaling_policy: Optional[ScalingPolicy] = None,
                 worker_factory: Optional[Callable[[], Worker]] = None):
        self.load_balancer = load_balancer
        self.policy = scaling_policy or ScalingPolicy()
        self.worker_factory = worker_factory
        
        self.logger = logging.getLogger(__name__)
        self.is_running = False
        self._lock = threading.RLock()
        
        # Scaling state tracking
        self.last_scale_up_time = 0.0
        self.last_scale_down_time = 0.0
        self.scaling_history: List[ScalingEvent] = []
        self.consecutive_scale_signals = {}  # trigger -> count
        
        # Metrics tracking for decision making
        self.metric_history: Dict[str, List[float]] = {
            'cpu_utilization': [],
            'memory_utilization': [],
            'response_time': [],
            'queue_depth': [],
            'request_rate': [],
            'error_rate': []
        }
        
        # Predictive scaling
        self.workload_patterns = {}  # time_of_day -> typical_load
        self.seasonal_adjustments = {}
        
        # Performance tracking
        self._evaluation_thread = None
        self._prediction_thread = None
    
    def start(self) -> None:
        """Start the auto-scaling system."""
        with self._lock:
            if self.is_running:
                return
            
            self.is_running = True
            
            # Start evaluation thread
            self._evaluation_thread = threading.Thread(
                target=self._evaluation_loop,
                name="AutoScaler-Evaluation",
                daemon=True
            )
            self._evaluation_thread.start()
            
            # Start predictive scaling thread
            self._prediction_thread = threading.Thread(
                target=self._prediction_loop,
                name="AutoScaler-Prediction",
                daemon=True
            )
            self._prediction_thread.start()
            
            self.logger.info("Neural processing auto-scaler started")
    
    def stop(self) -> None:
        """Stop the auto-scaling system."""
        with self._lock:
            self.is_running = False
            self.logger.info("Auto-scaler stopped")
    
    def _evaluation_loop(self) -> None:
        """Main evaluation loop for scaling decisions."""
        while self.is_running:
            try:
                self._evaluate_and_scale()
                time.sleep(self.policy.evaluation_period)
            except Exception as e:
                self.logger.error(f"Error in auto-scaler evaluation: {e}")
                time.sleep(30)  # Back off on error
    
    def _prediction_loop(self) -> None:
        """Predictive scaling loop based on historical patterns."""
        while self.is_running:
            try:
                self._update_workload_patterns()
                self._predictive_scaling_check()
                time.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Error in predictive scaling: {e}")
                time.sleep(300)
    
    def _evaluate_and_scale(self) -> None:
        """Evaluate current metrics and make scaling decisions."""
        current_metrics = self._collect_current_metrics()
        
        # Update metric history
        for metric_name, value in current_metrics.items():
            history = self.metric_history.get(metric_name, [])
            history.append(value)
            if len(history) > 100:  # Keep last 100 readings
                history.pop(0)
            self.metric_history[metric_name] = history
        
        # Evaluate scaling triggers
        scaling_signals = self._evaluate_scaling_triggers(current_metrics)
        
        # Make scaling decision
        scaling_decision = self._make_scaling_decision(scaling_signals)
        
        if scaling_decision != ScalingDirection.STABLE:
            self._execute_scaling_action(scaling_decision, scaling_signals)
    
    def _collect_current_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        workers = self.load_balancer.get_healthy_workers()
        
        if not workers:
            return {
                'cpu_utilization': 100.0,
                'memory_utilization': 100.0,
                'response_time': 1000.0,
                'queue_depth': 0.0,
                'request_rate': 0.0,
                'error_rate': 100.0
            }
        
        # Aggregate metrics across workers
        total_cpu = sum(w.metrics.cpu_usage for w in workers)
        total_memory = sum(w.metrics.memory_usage for w in workers)
        total_queue = sum(w.metrics.queue_depth for w in workers)
        avg_response_time = statistics.mean([w.metrics.avg_response_time for w in workers])
        
        # Calculate rates
        total_requests = sum(w.metrics.total_requests for w in workers)
        total_failures = sum(w.metrics.failed_requests for w in workers)
        error_rate = (total_failures / max(total_requests, 1)) * 100
        
        # Request rate (requests per second)
        request_rate = total_requests / max(self.policy.evaluation_period, 1)
        
        return {
            'cpu_utilization': total_cpu / len(workers),
            'memory_utilization': total_memory / len(workers),
            'response_time': avg_response_time,
            'queue_depth': total_queue / len(workers),
            'request_rate': request_rate,
            'error_rate': error_rate
        }
    
    def _evaluate_scaling_triggers(self, metrics: Dict[str, float]) -> Dict[ScalingTrigger, ScalingDirection]:
        """Evaluate individual scaling triggers."""
        signals = {}
        
        # CPU utilization trigger
        cpu = metrics.get('cpu_utilization', 0)
        if cpu > self.policy.scale_up_threshold:
            signals[ScalingTrigger.CPU_UTILIZATION] = ScalingDirection.UP
        elif cpu < self.policy.scale_down_threshold:
            signals[ScalingTrigger.CPU_UTILIZATION] = ScalingDirection.DOWN
        
        # Response time trigger
        response_time = metrics.get('response_time', 0)
        if response_time > self.policy.target_response_time * 1.5:  # 50% above target
            signals[ScalingTrigger.RESPONSE_TIME] = ScalingDirection.UP
        elif response_time < self.policy.target_response_time * 0.5:  # 50% below target
            signals[ScalingTrigger.RESPONSE_TIME] = ScalingDirection.DOWN
        
        # Queue depth trigger (neural processing specific)
        queue_depth = metrics.get('queue_depth', 0)
        if queue_depth > 10:  # More than 10 pending neural processing tasks
            signals[ScalingTrigger.QUEUE_DEPTH] = ScalingDirection.UP
        elif queue_depth < 2:
            signals[ScalingTrigger.QUEUE_DEPTH] = ScalingDirection.DOWN
        
        # Error rate trigger
        error_rate = metrics.get('error_rate', 0)
        if error_rate > 5.0:  # > 5% error rate indicates overload
            signals[ScalingTrigger.ERROR_RATE] = ScalingDirection.UP
        
        return signals
    
    def _make_scaling_decision(self, signals: Dict[ScalingTrigger, ScalingDirection]) -> ScalingDirection:
        """Make final scaling decision based on all signals."""
        current_time = time.time()
        current_worker_count = len(self.load_balancer.workers)
        
        # Check cooldown periods
        if (current_time - self.last_scale_up_time < self.policy.scale_up_cooldown and
            any(direction == ScalingDirection.UP for direction in signals.values())):
            return ScalingDirection.STABLE
        
        if (current_time - self.last_scale_down_time < self.policy.scale_down_cooldown and
            any(direction == ScalingDirection.DOWN for direction in signals.values())):
            return ScalingDirection.STABLE
        
        # Check worker limits
        if current_worker_count >= self.policy.max_workers:
            return ScalingDirection.STABLE
        
        if current_worker_count <= self.policy.min_workers:
            return ScalingDirection.STABLE
        
        # Count consecutive signals
        up_signals = sum(1 for direction in signals.values() if direction == ScalingDirection.UP)
        down_signals = sum(1 for direction in signals.values() if direction == ScalingDirection.DOWN)
        
        # Prioritize scale-up for performance
        if up_signals >= 2:  # At least 2 metrics indicate scale up needed
            return ScalingDirection.UP
        elif down_signals >= 3 and up_signals == 0:  # Strong signal to scale down
            return ScalingDirection.DOWN
        
        return ScalingDirection.STABLE
    
    def _execute_scaling_action(self, direction: ScalingDirection, 
                              signals: Dict[ScalingTrigger, ScalingDirection]) -> None:
        """Execute the scaling action."""
        current_worker_count = len(self.load_balancer.workers)
        current_time = time.time()
        
        if direction == ScalingDirection.UP:
            new_worker_count = min(current_worker_count + 1, self.policy.max_workers)
            if new_worker_count > current_worker_count and self.worker_factory:
                # Add new worker
                new_worker = self.worker_factory()
                if self.load_balancer.add_worker(new_worker):
                    self.last_scale_up_time = current_time
                    
                    # Record scaling event
                    primary_trigger = self._get_primary_trigger(signals, ScalingDirection.UP)
                    self._record_scaling_event(
                        direction=ScalingDirection.UP,
                        trigger=primary_trigger,
                        from_count=current_worker_count,
                        to_count=new_worker_count,
                        reason=f"Scale up triggered by {primary_trigger.value}"
                    )
                    
                    self.logger.info(f"Scaled up: {current_worker_count} -> {new_worker_count} workers")
        
        elif direction == ScalingDirection.DOWN:
            new_worker_count = max(current_worker_count - 1, self.policy.min_workers)
            if new_worker_count < current_worker_count:
                # Remove a worker (preferably the least utilized)
                worker_to_remove = self._select_worker_for_removal()
                if worker_to_remove and self.load_balancer.remove_worker(worker_to_remove.worker_id):
                    self.last_scale_down_time = current_time
                    
                    # Record scaling event
                    primary_trigger = self._get_primary_trigger(signals, ScalingDirection.DOWN)
                    self._record_scaling_event(
                        direction=ScalingDirection.DOWN,
                        trigger=primary_trigger,
                        from_count=current_worker_count,
                        to_count=new_worker_count,
                        reason=f"Scale down triggered by {primary_trigger.value}"
                    )
                    
                    self.logger.info(f"Scaled down: {current_worker_count} -> {new_worker_count} workers")
    
    def _get_primary_trigger(self, signals: Dict[ScalingTrigger, ScalingDirection], 
                           direction: ScalingDirection) -> ScalingTrigger:
        """Get the primary trigger for scaling decision."""
        triggers = [trigger for trigger, signal_dir in signals.items() if signal_dir == direction]
        
        # Priority order for triggers
        priority_order = [
            ScalingTrigger.ERROR_RATE,
            ScalingTrigger.RESPONSE_TIME,
            ScalingTrigger.CPU_UTILIZATION,
            ScalingTrigger.QUEUE_DEPTH,
            ScalingTrigger.MEMORY_UTILIZATION,
            ScalingTrigger.REQUEST_RATE
        ]
        
        for trigger in priority_order:
            if trigger in triggers:
                return trigger
        
        return triggers[0] if triggers else ScalingTrigger.CPU_UTILIZATION
    
    def _select_worker_for_removal(self) -> Optional[Worker]:
        """Select the best worker to remove when scaling down."""
        workers = self.load_balancer.get_healthy_workers()
        
        if not workers:
            return None
        
        # Select worker with lowest utilization and no active connections
        candidates = [w for w in workers if w.metrics.active_connections == 0]
        
        if not candidates:
            candidates = workers
        
        # Select worker with lowest CPU usage
        return min(candidates, key=lambda w: w.metrics.cpu_usage)
    
    def _record_scaling_event(self, direction: ScalingDirection, trigger: ScalingTrigger,
                             from_count: int, to_count: int, reason: str) -> None:
        """Record scaling event for analysis."""
        event = ScalingEvent(
            timestamp=time.time(),
            direction=direction,
            trigger=trigger,
            from_count=from_count,
            to_count=to_count,
            metric_value=0.0,  # TODO: Add specific metric value
            reason=reason
        )
        
        self.scaling_history.append(event)
        
        # Keep only recent events (last 1000)
        if len(self.scaling_history) > 1000:
            self.scaling_history.pop(0)
    
    def _update_workload_patterns(self) -> None:
        """Update workload patterns for predictive scaling."""
        current_metrics = self._collect_current_metrics()
        current_hour = time.localtime().tm_hour
        
        # Update hourly patterns
        if current_hour not in self.workload_patterns:
            self.workload_patterns[current_hour] = []
        
        self.workload_patterns[current_hour].append(current_metrics['request_rate'])
        
        # Keep only recent data (last 30 days worth)
        if len(self.workload_patterns[current_hour]) > 30:
            self.workload_patterns[current_hour].pop(0)
    
    def _predictive_scaling_check(self) -> None:
        """Check if predictive scaling is needed."""
        current_hour = time.localtime().tm_hour
        next_hour = (current_hour + 1) % 24
        
        # Predict load for next hour
        if next_hour in self.workload_patterns and self.workload_patterns[next_hour]:
            predicted_load = statistics.mean(self.workload_patterns[next_hour])
            current_load = self._collect_current_metrics()['request_rate']
            
            # If predicted load is significantly higher, pre-scale
            if predicted_load > current_load * 1.5:
                self.logger.info(f"Predictive scaling: preparing for increased load at hour {next_hour}")
                # Could trigger pre-scaling here
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current auto-scaling status."""
        current_metrics = self._collect_current_metrics()
        
        return {
            "is_running": self.is_running,
            "current_workers": len(self.load_balancer.workers),
            "policy": {
                "min_workers": self.policy.min_workers,
                "max_workers": self.policy.max_workers,
                "target_cpu": self.policy.target_cpu_utilization,
                "scale_up_threshold": self.policy.scale_up_threshold,
                "scale_down_threshold": self.policy.scale_down_threshold
            },
            "current_metrics": current_metrics,
            "last_scale_up": self.last_scale_up_time,
            "last_scale_down": self.last_scale_down_time,
            "total_scaling_events": len(self.scaling_history),
            "recent_scaling_events": len([e for e in self.scaling_history 
                                        if time.time() - e.timestamp < 3600])
        }
    
    def get_scaling_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent scaling history."""
        recent_events = sorted(self.scaling_history, key=lambda e: e.timestamp, reverse=True)[:limit]
        
        return [{
            "timestamp": event.timestamp,
            "direction": event.direction.value,
            "trigger": event.trigger.value,
            "from_count": event.from_count,
            "to_count": event.to_count,
            "reason": event.reason
        } for event in recent_events]