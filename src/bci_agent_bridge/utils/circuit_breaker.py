"""
Circuit breaker pattern implementation for BCI system resilience.
"""

import time
import threading
import logging
from typing import Any, Callable, Optional, Dict, List
from dataclasses import dataclass
from enum import Enum
import asyncio
import functools


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit is open, blocking calls
    HALF_OPEN = "half_open"  # Testing if service has recovered


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5      # Number of failures to open circuit
    success_threshold: int = 3      # Number of successes to close circuit
    timeout: float = 60.0          # Time to wait before trying half-open
    expected_exception: tuple = (Exception,)
    failure_condition: Optional[Callable[[Exception], bool]] = None


@dataclass
class CircuitMetrics:
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    consecutive_failures: int = 0
    consecutive_successes: int = 0
    last_failure_time: Optional[float] = None
    last_success_time: Optional[float] = None
    state_changes: int = 0


class CircuitBreakerOpenException(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class CircuitBreaker:
    """
    Circuit breaker implementation for protecting BCI services from cascading failures.
    """
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.metrics = CircuitMetrics()
        self.logger = logging.getLogger(__name__)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # State change callbacks
        self.on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset from open to half-open."""
        if self.state != CircuitState.OPEN:
            return False
        
        if self.metrics.last_failure_time is None:
            return True
        
        time_since_failure = time.time() - self.metrics.last_failure_time
        return time_since_failure >= self.config.timeout
    
    def _record_success(self) -> None:
        """Record a successful call."""
        with self._lock:
            self.metrics.total_calls += 1
            self.metrics.successful_calls += 1
            self.metrics.consecutive_successes += 1
            self.metrics.consecutive_failures = 0
            self.metrics.last_success_time = time.time()
            
            # Transition from half-open to closed after enough successes
            if (self.state == CircuitState.HALF_OPEN and 
                self.metrics.consecutive_successes >= self.config.success_threshold):
                self._change_state(CircuitState.CLOSED)
    
    def _record_failure(self, exception: Exception) -> None:
        """Record a failed call."""
        # Check if this failure should count
        if not isinstance(exception, self.config.expected_exception):
            return
        
        if (self.config.failure_condition and 
            not self.config.failure_condition(exception)):
            return
        
        with self._lock:
            self.metrics.total_calls += 1
            self.metrics.failed_calls += 1
            self.metrics.consecutive_failures += 1
            self.metrics.consecutive_successes = 0
            self.metrics.last_failure_time = time.time()
            
            # Transition to open after enough failures
            if (self.state == CircuitState.CLOSED and 
                self.metrics.consecutive_failures >= self.config.failure_threshold):
                self._change_state(CircuitState.OPEN)
            elif self.state == CircuitState.HALF_OPEN:
                self._change_state(CircuitState.OPEN)
    
    def _change_state(self, new_state: CircuitState) -> None:
        """Change circuit state and notify callbacks."""
        old_state = self.state
        self.state = new_state
        self.metrics.state_changes += 1
        
        self.logger.info(f"Circuit breaker '{self.name}' state changed: {old_state.value} -> {new_state.value}")
        
        if self.on_state_change:
            try:
                self.on_state_change(old_state, new_state)
            except Exception as e:
                self.logger.error(f"State change callback failed: {e}")
    
    def _can_execute(self) -> bool:
        """Check if execution is allowed based on current state."""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                with self._lock:
                    if self.state == CircuitState.OPEN:  # Double-check after acquiring lock
                        self._change_state(CircuitState.HALF_OPEN)
                        return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        
        return False
    
    async def call_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with circuit breaker protection."""
        if not self._can_execute():
            raise CircuitBreakerOpenException(
                f"Circuit breaker '{self.name}' is open. "
                f"Last failure: {self.metrics.last_failure_time}"
            )
        
        try:
            result = await func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    def call_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with circuit breaker protection."""
        if not self._can_execute():
            raise CircuitBreakerOpenException(
                f"Circuit breaker '{self.name}' is open. "
                f"Last failure: {self.metrics.last_failure_time}"
            )
        
        try:
            result = func(*args, **kwargs)
            self._record_success()
            return result
        except Exception as e:
            self._record_failure(e)
            raise
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get circuit breaker metrics."""
        with self._lock:
            failure_rate = (self.metrics.failed_calls / max(1, self.metrics.total_calls)) * 100
            uptime = time.time() - (self.metrics.last_failure_time or time.time())
            
            return {
                "name": self.name,
                "state": self.state.value,
                "total_calls": self.metrics.total_calls,
                "successful_calls": self.metrics.successful_calls,
                "failed_calls": self.metrics.failed_calls,
                "failure_rate_pct": round(failure_rate, 2),
                "consecutive_failures": self.metrics.consecutive_failures,
                "consecutive_successes": self.metrics.consecutive_successes,
                "last_failure_time": self.metrics.last_failure_time,
                "last_success_time": self.metrics.last_success_time,
                "uptime_seconds": round(uptime, 2),
                "state_changes": self.metrics.state_changes,
                "config": {
                    "failure_threshold": self.config.failure_threshold,
                    "success_threshold": self.config.success_threshold,
                    "timeout": self.config.timeout
                }
            }
    
    def reset(self) -> None:
        """Manually reset circuit breaker to closed state."""
        with self._lock:
            old_state = self.state
            self.state = CircuitState.CLOSED
            self.metrics.consecutive_failures = 0
            self.metrics.consecutive_successes = 0
            
            if old_state != CircuitState.CLOSED:
                self.logger.info(f"Circuit breaker '{self.name}' manually reset to closed")
                if self.on_state_change:
                    self.on_state_change(old_state, CircuitState.CLOSED)
    
    def force_open(self) -> None:
        """Manually force circuit breaker to open state."""
        with self._lock:
            old_state = self.state
            self._change_state(CircuitState.OPEN)
            self.metrics.last_failure_time = time.time()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator interface for circuit breaker."""
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.call_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.call_sync(func, *args, **kwargs)
            return sync_wrapper


class CircuitBreakerManager:
    """
    Manages multiple circuit breakers for different BCI services.
    """
    
    def __init__(self):
        self.breakers: Dict[str, CircuitBreaker] = {}
        self.logger = logging.getLogger(__name__)
    
    def create_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Create and register a new circuit breaker."""
        if name in self.breakers:
            self.logger.warning(f"Circuit breaker '{name}' already exists")
            return self.breakers[name]
        
        breaker = CircuitBreaker(name, config)
        
        # Add state change logging
        def log_state_change(old_state: CircuitState, new_state: CircuitState):
            self.logger.info(f"Circuit breaker '{name}': {old_state.value} -> {new_state.value}")
        
        breaker.on_state_change = log_state_change
        self.breakers[name] = breaker
        
        self.logger.info(f"Created circuit breaker: {name}")
        return breaker
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get circuit breaker by name."""
        return self.breakers.get(name)
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all circuit breakers."""
        return {name: breaker.get_metrics() for name, breaker in self.breakers.items()}
    
    def reset_all(self) -> None:
        """Reset all circuit breakers."""
        for breaker in self.breakers.values():
            breaker.reset()
        self.logger.info("All circuit breakers reset")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get overall health summary."""
        total_breakers = len(self.breakers)
        open_breakers = sum(1 for b in self.breakers.values() if b.state == CircuitState.OPEN)
        half_open_breakers = sum(1 for b in self.breakers.values() if b.state == CircuitState.HALF_OPEN)
        
        return {
            "total_breakers": total_breakers,
            "healthy_breakers": total_breakers - open_breakers - half_open_breakers,
            "open_breakers": open_breakers,
            "half_open_breakers": half_open_breakers,
            "overall_health": "healthy" if open_breakers == 0 else "degraded" if open_breakers < total_breakers else "unhealthy"
        }


def create_bci_circuit_breakers() -> CircuitBreakerManager:
    """Create circuit breakers for common BCI services."""
    manager = CircuitBreakerManager()
    
    # Neural data acquisition circuit breaker
    manager.create_breaker(
        "neural_acquisition",
        CircuitBreakerConfig(
            failure_threshold=3,
            success_threshold=2,
            timeout=10.0,
            expected_exception=(IOError, OSError, ConnectionError)
        )
    )
    
    # Claude API circuit breaker
    manager.create_breaker(
        "claude_api",
        CircuitBreakerConfig(
            failure_threshold=5,
            success_threshold=3,
            timeout=30.0,
            expected_exception=(ConnectionError, TimeoutError),
            failure_condition=lambda e: "rate limit" not in str(e).lower()  # Don't break on rate limits
        )
    )
    
    # Device connection circuit breaker
    manager.create_breaker(
        "device_connection",
        CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=1,
            timeout=60.0,
            expected_exception=(ConnectionError, OSError, TimeoutError)
        )
    )
    
    # Signal processing circuit breaker
    manager.create_breaker(
        "signal_processing",
        CircuitBreakerConfig(
            failure_threshold=10,  # Higher threshold for processing errors
            success_threshold=5,
            timeout=5.0,
            expected_exception=(ValueError, RuntimeError, MemoryError)
        )
    )
    
    return manager


# Example usage
if __name__ == "__main__":
    import random
    
    # Create circuit breaker
    breaker = CircuitBreaker("test_service", CircuitBreakerConfig(
        failure_threshold=3,
        success_threshold=2,
        timeout=5.0
    ))
    
    async def unreliable_service(success_rate: float = 0.5) -> str:
        if random.random() < success_rate:
            return "Success"
        else:
            raise ConnectionError("Service unavailable")
    
    @breaker
    async def protected_service() -> str:
        return await unreliable_service(0.3)  # 30% success rate
    
    async def test_circuit_breaker():
        for i in range(20):
            try:
                result = await protected_service()
                print(f"Call {i+1}: {result}")
            except CircuitBreakerOpenException as e:
                print(f"Call {i+1}: Circuit breaker open - {e}")
            except Exception as e:
                print(f"Call {i+1}: Failed - {e}")
            
            await asyncio.sleep(1)
            
            if i == 10:  # Print metrics midway
                print("\nCircuit Breaker Metrics:")
                metrics = breaker.get_metrics()
                for key, value in metrics.items():
                    print(f"  {key}: {value}")
                print()
    
    asyncio.run(test_circuit_breaker())