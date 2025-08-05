"""
Retry mechanisms with exponential backoff for robust BCI operations.
"""

import asyncio
import time
import random
import logging
from typing import Any, Callable, Optional, Type, Union, List
from dataclasses import dataclass
from enum import Enum
import functools


class RetryStrategy(Enum):
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIBONACCI_BACKOFF = "fibonacci_backoff"


@dataclass
class RetryConfig:
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    exceptions: tuple = (Exception,)
    retry_condition: Optional[Callable[[Exception], bool]] = None


@dataclass
class RetryState:
    attempt: int
    total_delay: float
    last_exception: Optional[Exception]
    start_time: float


class RetryManager:
    """
    Manages retry logic with various backoff strategies for BCI operations.
    """
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
    
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if an exception should trigger a retry."""
        # Check max attempts
        if attempt >= self.config.max_attempts:
            return False
        
        # Check exception type
        if not isinstance(exception, self.config.exceptions):
            return False
        
        # Check custom retry condition
        if self.config.retry_condition:
            return self.config.retry_condition(exception)
        
        return True
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for the given attempt number."""
        if self.config.strategy == RetryStrategy.FIXED_DELAY:
            delay = self.config.base_delay
        
        elif self.config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = self.config.base_delay * (2 ** (attempt - 1))
        
        elif self.config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = self.config.base_delay * attempt
        
        elif self.config.strategy == RetryStrategy.FIBONACCI_BACKOFF:
            # Generate fibonacci sequence up to attempt
            fib = [1, 1]
            for i in range(2, attempt + 1):
                fib.append(fib[i-1] + fib[i-2])
            delay = self.config.base_delay * fib[min(attempt, len(fib) - 1)]
        
        else:
            delay = self.config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, self.config.max_delay)
        
        # Add jitter if enabled
        if self.config.jitter:
            jitter_amount = delay * 0.1 * random.random()
            delay += jitter_amount
        
        return delay
    
    async def retry_async(self, func: Callable, *args, **kwargs) -> Any:
        """Execute async function with retry logic."""
        state = RetryState(
            attempt=0,
            total_delay=0.0,
            last_exception=None,
            start_time=time.time()
        )
        
        while state.attempt < self.config.max_attempts:
            state.attempt += 1
            
            try:
                self.logger.debug(f"Attempting {func.__name__} (attempt {state.attempt})")
                result = await func(*args, **kwargs)
                
                if state.attempt > 1:
                    self.logger.info(f"{func.__name__} succeeded after {state.attempt} attempts")
                
                return result
            
            except Exception as e:
                state.last_exception = e
                
                if not self.should_retry(e, state.attempt):
                    self.logger.error(f"{func.__name__} failed after {state.attempt} attempts: {e}")
                    raise e
                
                delay = self.calculate_delay(state.attempt)
                state.total_delay += delay
                
                self.logger.warning(f"{func.__name__} attempt {state.attempt} failed: {e}. "
                                  f"Retrying in {delay:.2f}s")
                
                await asyncio.sleep(delay)
        
        # All attempts exhausted
        self.logger.error(f"{func.__name__} failed after {state.attempt} attempts. "
                         f"Total time: {time.time() - state.start_time:.2f}s")
        raise state.last_exception
    
    def retry_sync(self, func: Callable, *args, **kwargs) -> Any:
        """Execute sync function with retry logic."""
        state = RetryState(
            attempt=0,
            total_delay=0.0,
            last_exception=None,
            start_time=time.time()
        )
        
        while state.attempt < self.config.max_attempts:
            state.attempt += 1
            
            try:
                self.logger.debug(f"Attempting {func.__name__} (attempt {state.attempt})")
                result = func(*args, **kwargs)
                
                if state.attempt > 1:
                    self.logger.info(f"{func.__name__} succeeded after {state.attempt} attempts")
                
                return result
            
            except Exception as e:
                state.last_exception = e
                
                if not self.should_retry(e, state.attempt):
                    self.logger.error(f"{func.__name__} failed after {state.attempt} attempts: {e}")
                    raise e
                
                delay = self.calculate_delay(state.attempt)
                state.total_delay += delay
                
                self.logger.warning(f"{func.__name__} attempt {state.attempt} failed: {e}. "
                                  f"Retrying in {delay:.2f}s")
                
                time.sleep(delay)
        
        # All attempts exhausted
        self.logger.error(f"{func.__name__} failed after {state.attempt} attempts. "
                         f"Total time: {time.time() - state.start_time:.2f}s")
        raise state.last_exception


class ExponentialBackoff:
    """
    Specialized exponential backoff implementation for common BCI operations.
    """
    
    def __init__(self, base_delay: float = 1.0, max_delay: float = 60.0, 
                 max_attempts: int = 5, jitter: bool = True):
        self.config = RetryConfig(
            max_attempts=max_attempts,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=base_delay,
            max_delay=max_delay,
            jitter=jitter
        )
        self.retry_manager = RetryManager(self.config)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator for applying exponential backoff."""
        if asyncio.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.retry_manager.retry_async(func, *args, **kwargs)
            return async_wrapper
        else:
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return self.retry_manager.retry_sync(func, *args, **kwargs)
            return sync_wrapper


def create_bci_retry_configs() -> dict:
    """Create retry configurations for different BCI operations."""
    
    configs = {
        # Neural data acquisition - quick retries
        "neural_acquisition": RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay=0.1,
            max_delay=1.0,
            exceptions=(IOError, OSError, ConnectionError)
        ),
        
        # Claude API calls - exponential backoff with longer delays
        "claude_api": RetryConfig(
            max_attempts=5,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=1.0,
            max_delay=30.0,
            exceptions=(ConnectionError, TimeoutError, Exception),
            retry_condition=lambda e: "rate limit" in str(e).lower() or 
                                    "timeout" in str(e).lower() or
                                    "connection" in str(e).lower()
        ),
        
        # Device connection - linear backoff
        "device_connection": RetryConfig(
            max_attempts=4,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            base_delay=2.0,
            max_delay=10.0,
            exceptions=(ConnectionError, OSError, TimeoutError)
        ),
        
        # Signal processing - minimal retries
        "signal_processing": RetryConfig(
            max_attempts=2,
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay=0.5,
            max_delay=2.0,
            exceptions=(ValueError, RuntimeError)
        ),
        
        # Clinical data operations - careful retries
        "clinical_data": RetryConfig(
            max_attempts=3,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            base_delay=2.0,
            max_delay=20.0,
            jitter=False,  # Deterministic for clinical operations
            exceptions=(IOError, PermissionError, OSError)
        )
    }
    
    return configs


# Convenient decorators for common BCI operations
def retry_neural_acquisition(func: Callable) -> Callable:
    """Decorator for neural data acquisition with appropriate retry."""
    config = create_bci_retry_configs()["neural_acquisition"]
    retry_manager = RetryManager(config)
    
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await retry_manager.retry_async(func, *args, **kwargs)
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return retry_manager.retry_sync(func, *args, **kwargs)
        return sync_wrapper


def retry_claude_api(func: Callable) -> Callable:
    """Decorator for Claude API calls with appropriate retry."""
    config = create_bci_retry_configs()["claude_api"]
    retry_manager = RetryManager(config)
    
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await retry_manager.retry_async(func, *args, **kwargs)
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return retry_manager.retry_sync(func, *args, **kwargs)
        return sync_wrapper


def retry_device_connection(func: Callable) -> Callable:
    """Decorator for device connection with appropriate retry."""
    config = create_bci_retry_configs()["device_connection"]
    retry_manager = RetryManager(config)
    
    if asyncio.iscoroutinefunction(func):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await retry_manager.retry_async(func, *args, **kwargs)
        return async_wrapper
    else:
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return retry_manager.retry_sync(func, *args, **kwargs)
        return sync_wrapper


# Example usage functions for testing
async def unreliable_api_call(success_rate: float = 0.3) -> str:
    """Simulate an unreliable API call for testing."""
    if random.random() < success_rate:
        return "API call successful"
    else:
        raise ConnectionError("Simulated API failure")


@retry_claude_api
async def stable_claude_call() -> str:
    """Example of Claude API call with retry decorator."""
    return await unreliable_api_call(0.4)  # 40% success rate


if __name__ == "__main__":
    # Example usage
    async def test_retry():
        try:
            result = await stable_claude_call()
            print(f"Success: {result}")
        except Exception as e:
            print(f"Final failure: {e}")
    
    asyncio.run(test_retry())