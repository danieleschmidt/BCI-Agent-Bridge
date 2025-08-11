"""
Adaptive circuit breaker patterns with dynamic thresholds and intelligent recovery strategies.
Enhanced for medical-grade BCI systems requiring high reliability and fault tolerance.
"""

import time
import threading
import logging
import asyncio
import functools
import statistics
import numpy as np
from typing import Any, Callable, Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import uuid


class CircuitState(Enum):
    CLOSED = "closed"           # Normal operation
    OPEN = "open"              # Circuit is open, blocking calls
    HALF_OPEN = "half_open"    # Testing if service has recovered
    ADAPTIVE = "adaptive"       # Learning optimal thresholds


class FailureCategory(Enum):
    TIMEOUT = "timeout"
    CONNECTION_ERROR = "connection_error"
    API_ERROR = "api_error"
    VALIDATION_ERROR = "validation_error"
    RESOURCE_ERROR = "resource_error"
    MEDICAL_SAFETY = "medical_safety"
    UNKNOWN = "unknown"


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive circuit breaker behavior."""
    base_failure_threshold: int = 5
    base_success_threshold: int = 3
    base_timeout: float = 60.0
    
    # Adaptive learning parameters
    learning_enabled: bool = True
    learning_rate: float = 0.1
    confidence_threshold: float = 0.7
    adaptation_window: int = 100  # Number of operations for learning
    
    # Performance tracking
    latency_threshold_ms: float = 5000.0
    success_rate_threshold: float = 0.8
    
    # Medical safety parameters (for BCI systems)
    medical_safety_mode: bool = True
    medical_failure_threshold: int = 2  # Lower threshold for medical safety
    medical_recovery_confirmation: int = 5  # Require more confirmations for medical systems
    
    # Advanced features
    jitter_factor: float = 0.1  # Add randomness to prevent thundering herd
    backoff_multiplier: float = 1.5
    max_timeout: float = 300.0


@dataclass
class OperationResult:
    """Result of a circuit breaker protected operation."""
    success: bool
    duration_ms: float
    error_category: FailureCategory
    error_details: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None


@dataclass
class CircuitMetrics:
    """Enhanced metrics for adaptive circuit breaker."""
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    
    # Time-based metrics
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0
    
    # Performance metrics
    average_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    current_success_rate: float = 1.0
    
    # Failure categorization
    failure_by_category: Dict[FailureCategory, int] = field(default_factory=lambda: defaultdict(int))
    
    # Adaptive learning metrics
    threshold_adaptations: int = 0
    learning_confidence: float = 0.0
    optimal_thresholds: Dict[str, float] = field(default_factory=dict)


@dataclass
class RecoveryStrategy:
    """Defines recovery strategy for a circuit breaker."""
    name: str
    test_function: Callable[[], bool]
    recovery_action: Optional[Callable[[], bool]] = None
    validation_steps: int = 3
    validation_interval: float = 10.0
    description: str = ""


class AdaptiveCircuitBreaker:
    """
    Advanced circuit breaker with adaptive thresholds and intelligent recovery.
    Designed for medical-grade BCI systems requiring high reliability.
    """
    
    def __init__(self, name: str, config: AdaptiveConfig = None):
        self.name = name
        self.config = config or AdaptiveConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Operation history for adaptive learning
        self.operation_history: deque = deque(maxlen=self.config.adaptation_window)
        self.latency_history: deque = deque(maxlen=100)
        
        # Adaptive thresholds (start with base values)
        self.current_failure_threshold = self.config.base_failure_threshold
        self.current_success_threshold = self.config.base_success_threshold
        self.current_timeout = self.config.base_timeout
        
        # Recovery strategies
        self.recovery_strategies: List[RecoveryStrategy] = []
        self.current_recovery_strategy: Optional[RecoveryStrategy] = None
        
        # State change callbacks
        self.on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
        self.on_failure: Optional[Callable[[OperationResult], None]] = None
        self.on_recovery: Optional[Callable[[], None]] = None
        
        # Background tasks
        self._adaptation_task: Optional[asyncio.Task] = None
        self._recovery_task: Optional[asyncio.Task] = None
        
        # Medical safety features
        if self.config.medical_safety_mode:
            self._setup_medical_safety_features()

    def _setup_medical_safety_features(self):
        """Setup medical safety specific features."""
        # Override thresholds for medical safety
        if self.name in ["neural_acquisition", "signal_processing", "medical_device"]:
            self.current_failure_threshold = self.config.medical_failure_threshold
            self.current_success_threshold = self.config.medical_recovery_confirmation
            
            self.logger.info(f"Medical safety mode enabled for {self.name}")

    async def start_adaptive_monitoring(self):
        """Start background adaptive monitoring tasks."""
        if self.config.learning_enabled:
            self._adaptation_task = asyncio.create_task(self._adaptive_learning_loop())
        
        self._recovery_task = asyncio.create_task(self._recovery_monitoring_loop())
        
        self.logger.info(f"Adaptive monitoring started for circuit breaker: {self.name}")

    async def stop_adaptive_monitoring(self):
        """Stop background adaptive monitoring tasks."""
        if self._adaptation_task:
            self._adaptation_task.cancel()
            try:
                await self._adaptation_task
            except asyncio.CancelledError:
                pass
        
        if self._recovery_task:
            self._recovery_task.cancel()
            try:
                await self._recovery_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info(f"Adaptive monitoring stopped for circuit breaker: {self.name}")

    def add_recovery_strategy(self, strategy: RecoveryStrategy):
        """Add a recovery strategy to the circuit breaker."""
        self.recovery_strategies.append(strategy)
        self.logger.info(f"Added recovery strategy '{strategy.name}' to {self.name}")

    def _should_attempt_reset(self) -> bool:
        """Enhanced logic for determining if circuit should attempt to reset."""
        if self.state != CircuitState.OPEN:
            return False
        
        if self.metrics.last_failure_time is None:
            return True
        
        # Calculate adaptive timeout with jitter
        jitter = self.current_timeout * self.config.jitter_factor * (0.5 - np.random.random())
        adaptive_timeout = self.current_timeout + jitter
        
        time_since_failure = time.time() - self.metrics.last_failure_time
        
        # Consider recent performance trends
        if self._has_improving_trend():
            adaptive_timeout *= 0.8  # Reduce timeout if trend is improving
        elif self._has_degrading_trend():
            adaptive_timeout *= 1.2  # Increase timeout if trend is degrading
        
        return time_since_failure >= adaptive_timeout

    def _has_improving_trend(self) -> bool:
        """Check if recent performance shows improving trend."""
        if len(self.latency_history) < 10:
            return False
        
        recent = list(self.latency_history)[-10:]
        older = list(self.latency_history)[-20:-10] if len(self.latency_history) >= 20 else recent
        
        try:
            return statistics.mean(recent) < statistics.mean(older) * 0.9
        except statistics.StatisticsError:
            return False

    def _has_degrading_trend(self) -> bool:
        """Check if recent performance shows degrading trend."""
        if len(self.latency_history) < 10:
            return False
        
        recent = list(self.latency_history)[-10:]
        older = list(self.latency_history)[-20:-10] if len(self.latency_history) >= 20 else recent
        
        try:
            return statistics.mean(recent) > statistics.mean(older) * 1.1
        except statistics.StatisticsError:
            return False

    def _categorize_failure(self, exception: Exception) -> FailureCategory:
        """Categorize failure type for adaptive learning."""
        error_type = type(exception).__name__.lower()
        error_msg = str(exception).lower()
        
        if "timeout" in error_msg or "timeout" in error_type:
            return FailureCategory.TIMEOUT
        elif "connection" in error_msg or "network" in error_msg:
            return FailureCategory.CONNECTION_ERROR
        elif "api" in error_msg or "http" in error_msg:
            return FailureCategory.API_ERROR
        elif "validation" in error_msg or "invalid" in error_msg:
            return FailureCategory.VALIDATION_ERROR
        elif "memory" in error_msg or "resource" in error_msg:
            return FailureCategory.RESOURCE_ERROR
        elif any(term in error_msg for term in ["safety", "medical", "critical", "emergency"]):
            return FailureCategory.MEDICAL_SAFETY
        else:
            return FailureCategory.UNKNOWN

    def _record_success(self, duration_ms: float, correlation_id: Optional[str] = None) -> None:
        """Record a successful call with enhanced metrics."""
        with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()
            
            # Update latency metrics
            self.latency_history.append(duration_ms)
            self._update_latency_metrics()
            
            # Update success rate
            self._update_success_rate()
            
            # Record operation for adaptive learning
            result = OperationResult(
                success=True,
                duration_ms=duration_ms,
                error_category=FailureCategory.UNKNOWN,  # No error
                correlation_id=correlation_id
            )
            self.operation_history.append(result)
            
            # State transitions
            if (self.state == CircuitState.HALF_OPEN and 
                self.metrics.consecutive_successes >= self.current_success_threshold):
                self._change_state(CircuitState.CLOSED)

    def _record_failure(self, exception: Exception, duration_ms: float = 0.0,
                       correlation_id: Optional[str] = None) -> None:
        """Record a failed call with enhanced categorization."""
        error_category = self._categorize_failure(exception)
        
        # Medical safety: immediate circuit opening for safety-related failures
        if (error_category == FailureCategory.MEDICAL_SAFETY and 
            self.config.medical_safety_mode):
            self._emergency_circuit_open(f"Medical safety failure: {exception}")
            return
        
        with self._lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = time.time()
            
            # Update failure categorization
            self.metrics.failure_by_category[error_category] += 1
            
            # Record latency if available
            if duration_ms > 0:
                self.latency_history.append(duration_ms)
                self._update_latency_metrics()
            
            # Update success rate
            self._update_success_rate()
            
            # Record operation for adaptive learning
            result = OperationResult(
                success=False,
                duration_ms=duration_ms,
                error_category=error_category,
                error_details=str(exception),
                correlation_id=correlation_id
            )
            self.operation_history.append(result)
            
            # State transitions with adaptive thresholds
            failure_threshold = self._get_adaptive_failure_threshold(error_category)
            
            if (self.state == CircuitState.CLOSED and 
                self.metrics.consecutive_failures >= failure_threshold):
                self._change_state(CircuitState.OPEN)
                self._select_recovery_strategy(error_category)
            elif self.state == CircuitState.HALF_OPEN:
                self._change_state(CircuitState.OPEN)
                self._select_recovery_strategy(error_category)
        
        # Notify failure callback
        if self.on_failure:
            try:
                self.on_failure(result)
            except Exception as e:
                self.logger.error(f"Failure callback error: {e}")

    def _get_adaptive_failure_threshold(self, error_category: FailureCategory) -> int:
        """Get adaptive failure threshold based on error category and learning."""
        base_threshold = self.current_failure_threshold
        
        # Adjust based on error category
        if error_category == FailureCategory.MEDICAL_SAFETY:
            return 1  # Immediate failure for medical safety
        elif error_category == FailureCategory.TIMEOUT:
            return max(1, base_threshold - 1)  # More sensitive to timeouts
        elif error_category == FailureCategory.RESOURCE_ERROR:
            return base_threshold + 2  # Less sensitive to resource errors
        else:
            return base_threshold

    def _emergency_circuit_open(self, reason: str):
        """Emergency circuit opening for critical failures."""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.OPEN
            self.metrics.last_failure_time = time.time()
            self.current_timeout = min(self.config.max_timeout, self.current_timeout * 2)
            
            self.logger.critical(f"EMERGENCY: Circuit breaker '{self.name}' opened: {reason}")
            
            if self.on_state_change and old_state != CircuitState.OPEN:
                try:
                    self.on_state_change(old_state, CircuitState.OPEN)
                except Exception as e:
                    self.logger.error(f"State change callback error: {e}")

    def _update_latency_metrics(self):
        """Update latency-based metrics."""
        if not self.latency_history:
            return
        
        latencies = list(self.latency_history)
        self.metrics.average_latency_ms = statistics.mean(latencies)
        
        if len(latencies) >= 20:
            self.metrics.p95_latency_ms = np.percentile(latencies, 95)

    def _update_success_rate(self):
        """Update current success rate."""
        if self.metrics.total_calls > 0:
            self.metrics.current_success_rate = (
                self.metrics.successful_calls / self.metrics.total_calls
            )

    def _select_recovery_strategy(self, error_category: FailureCategory):
        """Select appropriate recovery strategy based on failure type."""
        if not self.recovery_strategies:
            return
        
        # Simple strategy selection (can be enhanced with ML)
        category_preferences = {
            FailureCategory.TIMEOUT: ["restart_service", "scale_resources"],
            FailureCategory.CONNECTION_ERROR: ["reconnect", "failover"],
            FailureCategory.API_ERROR: ["retry_with_backoff", "circuit_reset"],
            FailureCategory.RESOURCE_ERROR: ["cleanup_resources", "scale_resources"],
            FailureCategory.MEDICAL_SAFETY: ["emergency_shutdown", "manual_intervention"]
        }
        
        preferred_strategies = category_preferences.get(error_category, [])
        
        for strategy in self.recovery_strategies:
            if strategy.name in preferred_strategies:
                self.current_recovery_strategy = strategy
                self.logger.info(f"Selected recovery strategy '{strategy.name}' for {error_category.value}")
                return
        
        # Default to first available strategy
        if self.recovery_strategies:
            self.current_recovery_strategy = self.recovery_strategies[0]

    async def _adaptive_learning_loop(self):
        """Background task for adaptive threshold learning."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._perform_adaptive_learning()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Adaptive learning error: {e}")

    async def _perform_adaptive_learning(self):
        """Perform adaptive learning to optimize thresholds."""
        if len(self.operation_history) < 20:
            return  # Need more data
        
        operations = list(self.operation_history)
        
        # Analyze success patterns
        success_ops = [op for op in operations if op.success]
        failure_ops = [op for op in operations if not op.success]
        
        if not failure_ops:
            return  # No failures to learn from
        
        # Calculate optimal failure threshold
        failure_durations = [op.duration_ms for op in failure_ops if op.duration_ms > 0]
        success_durations = [op.duration_ms for op in success_ops if op.duration_ms > 0]
        
        if failure_durations and success_durations:
            # Use statistical analysis to find optimal thresholds
            failure_mean = statistics.mean(failure_durations)
            success_mean = statistics.mean(success_durations)
            
            # Adapt timeout based on latency patterns
            if failure_mean > success_mean * 2:
                new_timeout = min(
                    self.config.max_timeout,
                    success_mean * 3  # 3x success latency as timeout
                )
                
                if abs(new_timeout - self.current_timeout) > self.current_timeout * 0.1:
                    self.current_timeout = new_timeout
                    self.metrics.threshold_adaptations += 1
                    
                    self.logger.info(f"Adapted timeout for {self.name}: {self.current_timeout:.1f}ms")
        
        # Calculate learning confidence
        total_ops = len(operations)
        if total_ops >= self.config.adaptation_window:
            success_rate = len(success_ops) / total_ops
            self.metrics.learning_confidence = min(1.0, success_rate + (total_ops / self.config.adaptation_window) * 0.1)

    async def _recovery_monitoring_loop(self):
        """Background task for monitoring recovery strategies."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                if self.state == CircuitState.OPEN and self.current_recovery_strategy:
                    await self._attempt_recovery()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Recovery monitoring error: {e}")

    async def _attempt_recovery(self):
        """Attempt recovery using the selected strategy."""
        if not self.current_recovery_strategy:
            return
        
        strategy = self.current_recovery_strategy
        
        try:
            self.logger.info(f"Attempting recovery with strategy: {strategy.name}")
            
            # Execute recovery action if available
            if strategy.recovery_action:
                recovery_success = await asyncio.to_thread(strategy.recovery_action)
                if not recovery_success:
                    self.logger.warning(f"Recovery action failed for strategy: {strategy.name}")
                    return
            
            # Validate recovery
            validation_successful = True
            for i in range(strategy.validation_steps):
                await asyncio.sleep(strategy.validation_interval)
                
                try:
                    test_result = await asyncio.to_thread(strategy.test_function)
                    if not test_result:
                        validation_successful = False
                        break
                except Exception as e:
                    self.logger.warning(f"Recovery validation failed: {e}")
                    validation_successful = False
                    break
            
            if validation_successful:
                # Successful recovery
                with self._lock:
                    self._change_state(CircuitState.HALF_OPEN)
                    self.current_recovery_strategy = None
                
                self.logger.info(f"Recovery successful with strategy: {strategy.name}")
                
                if self.on_recovery:
                    try:
                        self.on_recovery()
                    except Exception as e:
                        self.logger.error(f"Recovery callback error: {e}")
            else:
                self.logger.warning(f"Recovery validation failed for strategy: {strategy.name}")
                
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")

    def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit state with enhanced logging and callbacks."""
        old_state = self.state
        self.state = new_state
        self.metrics.state_changes += 1
        
        # Adjust timeouts based on state changes
        if new_state == CircuitState.OPEN:
            # Increase timeout with backoff
            self.current_timeout = min(
                self.config.max_timeout,
                self.current_timeout * self.config.backoff_multiplier
            )
        elif new_state == CircuitState.CLOSED:
            # Reset timeout to base value
            self.current_timeout = self.config.base_timeout
        
        self.logger.info(
            f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value} "
            f"(timeout: {self.current_timeout:.1f}s)"
        )
        
        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except Exception as e:
                self.logger.error(f"State change callback error: {e}")

    def _can_execute(self) -> bool:
        """Enhanced execution permission logic."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                with self._lock:
                    if self.state == CircuitState.OPEN:
                        self._change_state(CircuitState.HALF_OPEN)
                        return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False

    async def call_async(self, func: Callable, *args, correlation_id: Optional[str] = None, **kwargs) -> Any:
        """Execute async function with enhanced circuit breaker protection."""
        if not self._can_execute():
            raise CircuitBreakerOpenException(
                f"Circuit breaker '{self.name}' is open. "
                f"State: {self.state.value}, "
                f"Last failure: {self.metrics.last_failure_time}, "
                f"Recovery strategy: {self.current_recovery_strategy.name if self.current_recovery_strategy else 'None'}"
            )
        
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            self._record_success(duration_ms, correlation_id)
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_failure(e, duration_ms, correlation_id)
            raise

    def call_sync(self, func: Callable, *args, correlation_id: Optional[str] = None, **kwargs) -> Any:
        """Execute sync function with enhanced circuit breaker protection."""
        if not self._can_execute():
            raise CircuitBreakerOpenException(
                f"Circuit breaker '{self.name}' is open. "
                f"State: {self.state.value}, "
                f"Last failure: {self.metrics.last_failure_time}"
            )
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            self._record_success(duration_ms, correlation_id)
            return result
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._record_failure(e, duration_ms, correlation_id)
            raise

    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get comprehensive circuit breaker metrics."""
        with self._lock:
            failure_rate = (self.metrics.failed_calls / max(1, self.metrics.total_calls)) * 100
            uptime = time.time() - (self.metrics.last_failure_time or time.time())
            
            return {
                "name": self.name,
                "state": self.state.value,
                "basic_metrics": {
                    "total_calls": self.metrics.total_calls,
                    "successful_calls": self.metrics.successful_calls,
                    "failed_calls": self.metrics.failed_calls,
                    "failure_rate_pct": round(failure_rate, 2),
                    "consecutive_failures": self.metrics.consecutive_failures,
                    "consecutive_successes": self.metrics.consecutive_successes,
                    "uptime_seconds": round(uptime, 2),
                    "state_changes": self.metrics.state_changes
                },
                "performance_metrics": {
                    "average_latency_ms": round(self.metrics.average_latency_ms, 2),
                    "p95_latency_ms": round(self.metrics.p95_latency_ms, 2),
                    "current_success_rate": round(self.metrics.current_success_rate, 3)
                },
                "adaptive_metrics": {
                    "current_failure_threshold": self.current_failure_threshold,
                    "current_success_threshold": self.current_success_threshold,
                    "current_timeout": round(self.current_timeout, 1),
                    "threshold_adaptations": self.metrics.threshold_adaptations,
                    "learning_confidence": round(self.metrics.learning_confidence, 3)
                },
                "failure_analysis": {
                    category.value: count 
                    for category, count in self.metrics.failure_by_category.items()
                },
                "recovery_info": {
                    "available_strategies": len(self.recovery_strategies),
                    "current_strategy": self.current_recovery_strategy.name if self.current_recovery_strategy else None,
                    "strategy_descriptions": [s.description for s in self.recovery_strategies]
                },
                "configuration": {
                    "medical_safety_mode": self.config.medical_safety_mode,
                    "learning_enabled": self.config.learning_enabled,
                    "base_failure_threshold": self.config.base_failure_threshold,
                    "base_success_threshold": self.config.base_success_threshold,
                    "base_timeout": self.config.base_timeout
                }
            }

    def reset(self) -> None:
        """Enhanced reset with adaptive threshold preservation."""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.metrics.consecutive_failures = 0
            self.metrics.consecutive_successes = 0
            
            # Don't reset adaptive thresholds unless explicitly requested
            # This preserves learned behavior
            
            self.current_recovery_strategy = None
            
            if old_state != CircuitState.CLOSED:
                self.logger.info(f"Circuit breaker '{self.name}' manually reset to closed")
                if self.on_state_change:
                    self.on_state_change(old_state, CircuitState.CLOSED)

    def force_learning_reset(self) -> None:
        """Reset adaptive learning and return to base thresholds."""
        with self._lock:
            self.current_failure_threshold = self.config.base_failure_threshold
            self.current_success_threshold = self.config.base_success_threshold
            self.current_timeout = self.config.base_timeout
            
            self.metrics.threshold_adaptations = 0
            self.metrics.learning_confidence = 0.0
            self.metrics.optimal_thresholds.clear()
            
            self.operation_history.clear()
            self.latency_history.clear()
            
            self.logger.info(f"Adaptive learning reset for circuit breaker: {self.name}")


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Enhanced circuit breaker manager with medical safety features
class AdaptiveCircuitBreakerManager:
    """
    Manages multiple adaptive circuit breakers with coordinated recovery strategies.
    """
    
    def __init__(self, enable_coordination: bool = True):
        self.breakers: Dict[str, AdaptiveCircuitBreaker] = {}
        self.enable_coordination = enable_coordination
        self.logger = logging.getLogger(__name__)
        
        # Global recovery coordination
        self.recovery_coordination_task: Optional[asyncio.Task] = None
        self.system_health_threshold = 0.7  # 70% of breakers should be healthy

    def create_breaker(self, name: str, config: AdaptiveConfig = None) -> AdaptiveCircuitBreaker:
        """Create and register a new adaptive circuit breaker."""
        if name in self.breakers:
            self.logger.warning(f"Adaptive circuit breaker '{name}' already exists")
            return self.breakers[name]
        
        breaker = AdaptiveCircuitBreaker(name, config)
        
        # Add coordinated recovery monitoring
        if self.enable_coordination:
            def coordinated_state_change(old_state: CircuitState, new_state: CircuitState):
                asyncio.create_task(self._handle_coordinated_state_change(name, old_state, new_state))
            
            breaker.on_state_change = coordinated_state_change
        
        self.breakers[name] = breaker
        self.logger.info(f"Created adaptive circuit breaker: {name}")
        
        return breaker

    async def start_all_monitoring(self):
        """Start monitoring for all circuit breakers."""
        tasks = []
        for breaker in self.breakers.values():
            tasks.append(breaker.start_adaptive_monitoring())
        
        if tasks:
            await asyncio.gather(*tasks)
        
        if self.enable_coordination:
            self.recovery_coordination_task = asyncio.create_task(self._coordination_loop())
        
        self.logger.info("All adaptive circuit breakers monitoring started")

    async def stop_all_monitoring(self):
        """Stop monitoring for all circuit breakers."""
        tasks = []
        for breaker in self.breakers.values():
            tasks.append(breaker.stop_adaptive_monitoring())
        
        if tasks:
            await asyncio.gather(*tasks)
        
        if self.recovery_coordination_task:
            self.recovery_coordination_task.cancel()
            try:
                await self.recovery_coordination_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("All adaptive circuit breakers monitoring stopped")

    async def _handle_coordinated_state_change(self, breaker_name: str, 
                                              old_state: CircuitState, new_state: CircuitState):
        """Handle coordinated response to state changes."""
        if new_state == CircuitState.OPEN:
            # Check if system health is degraded
            healthy_breakers = self._count_healthy_breakers()
            total_breakers = len(self.breakers)
            
            if healthy_breakers / total_breakers < self.system_health_threshold:
                self.logger.critical(
                    f"System health degraded: {healthy_breakers}/{total_breakers} breakers healthy. "
                    f"Breaker '{breaker_name}' opened."
                )
                
                # Trigger system-wide recovery coordination
                await self._trigger_system_recovery()

    def _count_healthy_breakers(self) -> int:
        """Count number of healthy (closed) circuit breakers."""
        return sum(1 for breaker in self.breakers.values() 
                  if breaker.state == CircuitState.CLOSED)

    async def _trigger_system_recovery(self):
        """Trigger coordinated system-wide recovery."""
        self.logger.info("Triggering coordinated system recovery")
        
        # Prioritize critical services for recovery
        critical_services = ["neural_acquisition", "signal_processing", "medical_device"]
        
        for service_name in critical_services:
            if service_name in self.breakers:
                breaker = self.breakers[service_name]
                if breaker.state == CircuitState.OPEN:
                    # Force faster recovery for critical services
                    breaker.current_timeout = min(30.0, breaker.current_timeout)
                    self.logger.info(f"Accelerated recovery for critical service: {service_name}")

    async def _coordination_loop(self):
        """Background coordination monitoring."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._perform_health_coordination()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Coordination loop error: {e}")

    async def _perform_health_coordination(self):
        """Perform coordinated health monitoring and optimization."""
        # Analyze cross-breaker patterns
        open_breakers = [name for name, breaker in self.breakers.items() 
                        if breaker.state == CircuitState.OPEN]
        
        if len(open_breakers) > len(self.breakers) * 0.5:
            self.logger.warning(f"Many breakers open: {open_breakers}")
            
            # Check for system-wide issues
            common_failure_categories = self._analyze_common_failures()
            if common_failure_categories:
                self.logger.info(f"Common failure patterns detected: {common_failure_categories}")

    def _analyze_common_failures(self) -> List[FailureCategory]:
        """Analyze common failure patterns across breakers."""
        category_counts = defaultdict(int)
        
        for breaker in self.breakers.values():
            for category, count in breaker.metrics.failure_by_category.items():
                category_counts[category] += count
        
        # Return categories that appear in multiple breakers
        common_categories = [
            category for category, count in category_counts.items()
            if count >= len(self.breakers) * 0.3  # 30% threshold
        ]
        
        return common_categories

    def get_system_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive system health summary."""
        total_breakers = len(self.breakers)
        healthy_breakers = self._count_healthy_breakers()
        
        breaker_states = defaultdict(int)
        total_operations = 0
        total_failures = 0
        
        for breaker in self.breakers.values():
            breaker_states[breaker.state.value] += 1
            total_operations += breaker.metrics.total_calls
            total_failures += breaker.metrics.failed_calls
        
        system_success_rate = 1.0
        if total_operations > 0:
            system_success_rate = (total_operations - total_failures) / total_operations
        
        return {
            "timestamp": time.time(),
            "system_health": {
                "overall_status": "healthy" if healthy_breakers / total_breakers >= self.system_health_threshold else "degraded",
                "healthy_breakers": healthy_breakers,
                "total_breakers": total_breakers,
                "health_percentage": (healthy_breakers / total_breakers) * 100 if total_breakers > 0 else 0
            },
            "breaker_states": dict(breaker_states),
            "system_performance": {
                "total_operations": total_operations,
                "total_failures": total_failures,
                "system_success_rate": round(system_success_rate, 3)
            },
            "coordination": {
                "enabled": self.enable_coordination,
                "health_threshold": self.system_health_threshold
            },
            "individual_breakers": {
                name: breaker.get_enhanced_metrics()
                for name, breaker in self.breakers.items()
            }
        }


def create_bci_adaptive_circuit_breakers() -> AdaptiveCircuitBreakerManager:
    """Create adaptive circuit breakers for BCI system components."""
    manager = AdaptiveCircuitBreakerManager(enable_coordination=True)
    
    # Neural data acquisition circuit breaker (critical medical device)
    neural_config = AdaptiveConfig(
        base_failure_threshold=2,
        base_success_threshold=3,
        base_timeout=10.0,
        medical_safety_mode=True,
        learning_enabled=True,
        latency_threshold_ms=100.0
    )
    neural_breaker = manager.create_breaker("neural_acquisition", neural_config)
    
    # Add recovery strategies for neural acquisition
    def test_neural_connection():
        # Test function would check actual neural device
        return True  # Placeholder
    
    def restart_neural_service():
        # Recovery action would restart neural acquisition
        return True  # Placeholder
    
    neural_breaker.add_recovery_strategy(RecoveryStrategy(
        name="restart_neural_service",
        test_function=test_neural_connection,
        recovery_action=restart_neural_service,
        validation_steps=5,  # More validation for medical devices
        description="Restart neural acquisition service and validate connection"
    ))
    
    # Claude API circuit breaker
    claude_config = AdaptiveConfig(
        base_failure_threshold=5,
        base_success_threshold=3,
        base_timeout=30.0,
        learning_enabled=True,
        latency_threshold_ms=5000.0,
        backoff_multiplier=1.5
    )
    claude_breaker = manager.create_breaker("claude_api", claude_config)
    
    def test_claude_connection():
        return True  # Would test Claude API connectivity
    
    claude_breaker.add_recovery_strategy(RecoveryStrategy(
        name="retry_with_backoff",
        test_function=test_claude_connection,
        validation_steps=2,
        description="Retry Claude API with exponential backoff"
    ))
    
    # Signal processing circuit breaker
    signal_config = AdaptiveConfig(
        base_failure_threshold=3,
        base_success_threshold=2,
        base_timeout=15.0,
        medical_safety_mode=True,
        learning_enabled=True
    )
    signal_breaker = manager.create_breaker("signal_processing", signal_config)
    
    def test_signal_processing():
        return True  # Would test signal processing pipeline
    
    def restart_processing_pipeline():
        return True  # Would restart signal processing
    
    signal_breaker.add_recovery_strategy(RecoveryStrategy(
        name="restart_processing_pipeline",
        test_function=test_signal_processing,
        recovery_action=restart_processing_pipeline,
        validation_steps=3,
        description="Restart signal processing pipeline with validation"
    ))
    
    return manager


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    import random
    
    async def demo_adaptive_circuit_breaker():
        print("Adaptive Circuit Breaker Demo")
        print("=" * 50)
        
        # Create BCI circuit breaker manager
        manager = create_bci_adaptive_circuit_breakers()
        
        # Start monitoring
        await manager.start_all_monitoring()
        
        print("Adaptive circuit breaker system started")
        
        # Get neural acquisition breaker for testing
        neural_breaker = manager.breakers["neural_acquisition"]
        
        # Simulate operations with varying success rates
        async def simulate_neural_operation(success_rate: float = 0.8) -> str:
            await asyncio.sleep(random.uniform(0.05, 0.2))  # Simulate processing time
            
            if random.random() < success_rate:
                return "Neural data acquired successfully"
            else:
                # Simulate different types of failures
                failure_types = [
                    ConnectionError("Neural device disconnected"),
                    TimeoutError("Neural acquisition timeout"),
                    ValueError("Invalid neural signal detected"),
                    RuntimeError("Medical safety threshold exceeded")
                ]
                raise random.choice(failure_types)
        
        # Test adaptive behavior
        print("\nTesting adaptive circuit breaker behavior...")
        
        success_count = 0
        failure_count = 0
        circuit_open_count = 0
        
        # Phase 1: Normal operation
        print("\nPhase 1: Normal operation (80% success rate)")
        for i in range(20):
            try:
                result = await neural_breaker.call_async(
                    simulate_neural_operation, 
                    success_rate=0.8,
                    correlation_id=f"op_{i}"
                )
                success_count += 1
                print(f"✓ Operation {i+1}: {result}")
            except Exception as e:
                if "Circuit breaker" in str(e):
                    circuit_open_count += 1
                    print(f"⭕ Operation {i+1}: Circuit open")
                else:
                    failure_count += 1
                    print(f"✗ Operation {i+1}: {type(e).__name__}: {e}")
            
            await asyncio.sleep(0.1)
        
        # Phase 2: High failure rate
        print("\nPhase 2: High failure rate (20% success rate)")
        for i in range(20, 40):
            try:
                result = await neural_breaker.call_async(
                    simulate_neural_operation, 
                    success_rate=0.2,
                    correlation_id=f"op_{i}"
                )
                success_count += 1
                print(f"✓ Operation {i+1}: {result}")
            except Exception as e:
                if "Circuit breaker" in str(e):
                    circuit_open_count += 1
                    print(f"⭕ Operation {i+1}: Circuit open")
                else:
                    failure_count += 1
                    print(f"✗ Operation {i+1}: {type(e).__name__}: {e}")
            
            await asyncio.sleep(0.1)
        
        # Phase 3: Recovery
        print("\nPhase 3: Recovery (90% success rate)")
        await asyncio.sleep(5)  # Wait for potential recovery
        
        for i in range(40, 50):
            try:
                result = await neural_breaker.call_async(
                    simulate_neural_operation, 
                    success_rate=0.9,
                    correlation_id=f"op_{i}"
                )
                success_count += 1
                print(f"✓ Operation {i+1}: {result}")
            except Exception as e:
                if "Circuit breaker" in str(e):
                    circuit_open_count += 1
                    print(f"⭕ Operation {i+1}: Circuit open")
                else:
                    failure_count += 1
                    print(f"✗ Operation {i+1}: {type(e).__name__}: {e}")
            
            await asyncio.sleep(0.1)
        
        # Show results
        print(f"\n--- Test Results ---")
        print(f"Successful operations: {success_count}")
        print(f"Failed operations: {failure_count}")
        print(f"Circuit open blocks: {circuit_open_count}")
        
        # Show metrics
        print(f"\n--- Circuit Breaker Metrics ---")
        metrics = neural_breaker.get_enhanced_metrics()
        print(f"State: {metrics['state']}")
        print(f"Total calls: {metrics['basic_metrics']['total_calls']}")
        print(f"Success rate: {100 - metrics['basic_metrics']['failure_rate_pct']:.1f}%")
        print(f"Average latency: {metrics['performance_metrics']['average_latency_ms']:.1f}ms")
        print(f"Adaptations made: {metrics['adaptive_metrics']['threshold_adaptations']}")
        print(f"Learning confidence: {metrics['adaptive_metrics']['learning_confidence']:.2f}")
        
        # Show system health
        print(f"\n--- System Health Summary ---")
        health = manager.get_system_health_summary()
        print(f"Overall status: {health['system_health']['overall_status']}")
        print(f"Healthy breakers: {health['system_health']['healthy_breakers']}/{health['system_health']['total_breakers']}")
        print(f"System success rate: {health['system_performance']['system_success_rate']:.3f}")
        
        # Stop monitoring
        await manager.stop_all_monitoring()
        print("\nAdaptive circuit breaker demo completed")
    
    asyncio.run(demo_adaptive_circuit_breaker())