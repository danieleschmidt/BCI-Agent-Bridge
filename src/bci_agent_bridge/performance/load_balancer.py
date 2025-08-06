"""
Load balancing system for distributing BCI processing across multiple workers.
"""

import asyncio
import time
import random
import logging
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
import statistics
import numpy as np
from abc import ABC, abstractmethod


class LoadBalancingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESPONSE_TIME = "response_time"
    RESOURCE_BASED = "resource_based"
    ADAPTIVE = "adaptive"


class WorkerState(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class WorkerMetrics:
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    active_connections: int = 0
    avg_response_time: float = 0.0
    last_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    queue_depth: int = 0
    last_health_check: float = 0.0
    consecutive_failures: int = 0


@dataclass
class Worker:
    id: str
    endpoint: str
    weight: float = 1.0
    max_connections: int = 100
    state: WorkerState = WorkerState.HEALTHY
    metrics: WorkerMetrics = field(default_factory=WorkerMetrics)
    processor_func: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        total = self.metrics.total_requests
        if total == 0:
            return 1.0
        return self.metrics.successful_requests / total
    
    @property
    def is_available(self) -> bool:
        return (self.state in [WorkerState.HEALTHY, WorkerState.DEGRADED] and
                self.metrics.active_connections < self.max_connections)
    
    @property
    def load_score(self) -> float:
        """Calculate load score for balancing decisions."""
        if not self.is_available:
            return float('inf')
        
        # Combine multiple factors
        connection_load = self.metrics.active_connections / max(self.max_connections, 1)
        response_time_factor = min(self.metrics.avg_response_time / 1000, 2.0)  # Cap at 2 seconds
        failure_penalty = self.metrics.consecutive_failures * 0.1
        
        return connection_load + response_time_factor + failure_penalty


class HealthChecker:
    """Health checker for workers."""
    
    def __init__(self, check_interval: float = 30.0, timeout: float = 5.0):
        self.check_interval = check_interval
        self.timeout = timeout
        self.logger = logging.getLogger(__name__)
        
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self, workers: List[Worker], 
                   health_check_func: Optional[Callable[[Worker], bool]] = None) -> None:
        """Start health checking."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(
            self._health_check_loop(workers, health_check_func)
        )
    
    async def stop(self) -> None:
        """Stop health checking."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _health_check_loop(self, workers: List[Worker], 
                                health_check_func: Optional[Callable]) -> None:
        """Main health check loop."""
        while self._running:
            try:
                await self._check_all_workers(workers, health_check_func)
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Health check error: {e}")
                await asyncio.sleep(min(self.check_interval, 10.0))
    
    async def _check_all_workers(self, workers: List[Worker], 
                               health_check_func: Optional[Callable]) -> None:
        """Check health of all workers."""
        health_tasks = []
        
        for worker in workers:
            task = asyncio.create_task(
                self._check_worker_health(worker, health_check_func)
            )
            health_tasks.append(task)
        
        if health_tasks:
            await asyncio.gather(*health_tasks, return_exceptions=True)
    
    async def _check_worker_health(self, worker: Worker, 
                                 health_check_func: Optional[Callable]) -> None:
        """Check health of individual worker."""
        try:
            if health_check_func:
                is_healthy = await asyncio.wait_for(
                    asyncio.to_thread(health_check_func, worker),
                    timeout=self.timeout
                )
            else:
                is_healthy = await self._default_health_check(worker)
            
            # Update worker state based on health check
            if is_healthy:
                if worker.state == WorkerState.UNHEALTHY:
                    worker.state = WorkerState.HEALTHY
                    self.logger.info(f"Worker {worker.id} recovered")
                worker.metrics.consecutive_failures = 0
            else:
                worker.metrics.consecutive_failures += 1
                
                if worker.metrics.consecutive_failures >= 3:
                    worker.state = WorkerState.UNHEALTHY
                    self.logger.warning(f"Worker {worker.id} marked unhealthy")
                elif worker.state == WorkerState.HEALTHY:
                    worker.state = WorkerState.DEGRADED
            
            worker.metrics.last_health_check = time.time()
            
        except Exception as e:
            self.logger.error(f"Health check failed for worker {worker.id}: {e}")
            worker.metrics.consecutive_failures += 1
            worker.state = WorkerState.UNHEALTHY
    
    async def _default_health_check(self, worker: Worker) -> bool:
        """Default health check implementation."""
        # Simple check - just verify worker is not in failed state
        return worker.state != WorkerState.OFFLINE


class LoadBalancer:
    """
    Load balancer for distributing requests across multiple workers.
    """
    
    def __init__(self, strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN,
                 health_check_interval: float = 30.0):
        self.strategy = strategy
        self.workers: List[Worker] = []
        self.logger = logging.getLogger(__name__)
        
        # Strategy state
        self._round_robin_index = 0
        self._lock = threading.Lock()
        
        # Health checking
        self.health_checker = HealthChecker(check_interval=health_check_interval)
        
        # Metrics
        self.total_requests = 0
        self.failed_requests = 0
        self.avg_response_time = 0.0
    
    def add_worker(self, worker: Worker) -> None:
        """Add a worker to the load balancer."""
        with self._lock:
            if worker.id not in [w.id for w in self.workers]:
                self.workers.append(worker)
                self.logger.info(f"Added worker {worker.id} at {worker.endpoint}")
    
    def remove_worker(self, worker_id: str) -> bool:
        """Remove a worker from the load balancer."""
        with self._lock:
            for i, worker in enumerate(self.workers):
                if worker.id == worker_id:
                    removed_worker = self.workers.pop(i)
                    self.logger.info(f"Removed worker {worker_id}")
                    return True
        return False
    
    def get_worker(self, request_context: Dict[str, Any] = None) -> Optional[Worker]:
        """Get next worker based on load balancing strategy."""
        with self._lock:
            available_workers = [w for w in self.workers if w.is_available]
            
            if not available_workers:
                self.logger.warning("No available workers")
                return None
            
            if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.RESPONSE_TIME:
                return self._response_time_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                return self._resource_based_selection(available_workers)
            elif self.strategy == LoadBalancingStrategy.ADAPTIVE:
                return self._adaptive_selection(available_workers, request_context)
            else:
                return available_workers[0]
    
    def _round_robin_selection(self, workers: List[Worker]) -> Worker:
        """Round-robin worker selection."""
        worker = workers[self._round_robin_index % len(workers)]
        self._round_robin_index += 1
        return worker
    
    def _least_connections_selection(self, workers: List[Worker]) -> Worker:
        """Select worker with least active connections."""
        return min(workers, key=lambda w: w.metrics.active_connections)
    
    def _weighted_round_robin_selection(self, workers: List[Worker]) -> Worker:
        """Weighted round-robin selection."""
        total_weight = sum(w.weight for w in workers)
        if total_weight == 0:
            return workers[0]
        
        # Create weighted list
        weighted_workers = []
        for worker in workers:
            count = max(1, int(worker.weight * 10))  # Scale weights
            weighted_workers.extend([worker] * count)
        
        if weighted_workers:
            return weighted_workers[self._round_robin_index % len(weighted_workers)]
        return workers[0]
    
    def _response_time_selection(self, workers: List[Worker]) -> Worker:
        """Select worker with best response time."""
        return min(workers, key=lambda w: w.metrics.avg_response_time or float('inf'))
    
    def _resource_based_selection(self, workers: List[Worker]) -> Worker:
        """Select worker based on resource utilization."""
        def resource_score(worker: Worker) -> float:
            cpu_factor = worker.metrics.cpu_usage / 100.0
            memory_factor = worker.metrics.memory_usage / 100.0
            queue_factor = worker.metrics.queue_depth / 100.0
            return cpu_factor + memory_factor + queue_factor
        
        return min(workers, key=resource_score)
    
    def _adaptive_selection(self, workers: List[Worker], 
                          request_context: Dict[str, Any]) -> Worker:
        """Adaptive worker selection based on multiple factors."""
        # Score workers based on multiple criteria
        scored_workers = []
        
        for worker in workers:
            score = worker.load_score
            
            # Adjust score based on request context
            if request_context:
                # Priority requests go to best workers
                if request_context.get('priority', 0) > 5:
                    score *= (1.0 - worker.success_rate)
                
                # Large requests avoid heavily loaded workers
                if request_context.get('size', 0) > 1000:
                    score *= (1 + worker.metrics.active_connections / worker.max_connections)
            
            scored_workers.append((score, worker))
        
        # Select worker with lowest score
        scored_workers.sort(key=lambda x: x[0])
        return scored_workers[0][1]
    
    async def process_request(self, request_data: Any, 
                            request_context: Dict[str, Any] = None) -> Any:
        """Process request through load balancer."""
        start_time = time.time()
        worker = self.get_worker(request_context)
        
        if not worker:
            raise RuntimeError("No available workers")
        
        try:
            # Update worker metrics
            worker.metrics.active_connections += 1
            worker.metrics.total_requests += 1
            
            # Process request
            if worker.processor_func:
                result = await asyncio.to_thread(worker.processor_func, request_data)
            else:
                # Default processing
                result = request_data
            
            # Update success metrics
            response_time = time.time() - start_time
            worker.metrics.successful_requests += 1
            worker.metrics.last_response_time = response_time
            
            # Update average response time
            total_successful = worker.metrics.successful_requests
            if total_successful == 1:
                worker.metrics.avg_response_time = response_time
            else:
                worker.metrics.avg_response_time = (
                    (worker.metrics.avg_response_time * (total_successful - 1) + response_time) / 
                    total_successful
                )
            
            # Update global metrics
            self._update_global_metrics(response_time, success=True)
            
            return result
            
        except Exception as e:
            # Update failure metrics
            worker.metrics.failed_requests += 1
            worker.metrics.consecutive_failures += 1
            
            self._update_global_metrics(time.time() - start_time, success=False)
            
            self.logger.error(f"Request failed on worker {worker.id}: {e}")
            raise
        
        finally:
            worker.metrics.active_connections = max(0, worker.metrics.active_connections - 1)
    
    def _update_global_metrics(self, response_time: float, success: bool) -> None:
        """Update global load balancer metrics."""
        self.total_requests += 1
        
        if not success:
            self.failed_requests += 1
        
        # Update average response time
        if self.total_requests == 1:
            self.avg_response_time = response_time
        else:
            self.avg_response_time = (
                (self.avg_response_time * (self.total_requests - 1) + response_time) / 
                self.total_requests
            )
    
    async def start_health_checking(self, 
                                  health_check_func: Optional[Callable] = None) -> None:
        """Start health checking for all workers."""
        await self.health_checker.start(self.workers, health_check_func)
    
    async def stop_health_checking(self) -> None:
        """Stop health checking."""
        await self.health_checker.stop()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self._lock:
            worker_stats = []
            for worker in self.workers:
                worker_stats.append({
                    'id': worker.id,
                    'endpoint': worker.endpoint,
                    'state': worker.state.value,
                    'weight': worker.weight,
                    'active_connections': worker.metrics.active_connections,
                    'total_requests': worker.metrics.total_requests,
                    'success_rate': round(worker.success_rate * 100, 2),
                    'avg_response_time_ms': round(worker.metrics.avg_response_time * 1000, 2),
                    'load_score': round(worker.load_score, 3)
                })
            
            return {
                'strategy': self.strategy.value,
                'total_workers': len(self.workers),
                'healthy_workers': len([w for w in self.workers if w.state == WorkerState.HEALTHY]),
                'available_workers': len([w for w in self.workers if w.is_available]),
                'total_requests': self.total_requests,
                'failed_requests': self.failed_requests,
                'success_rate_pct': round((1 - self.failed_requests / max(1, self.total_requests)) * 100, 2),
                'avg_response_time_ms': round(self.avg_response_time * 1000, 2),
                'workers': worker_stats
            }


class AdaptiveLoadBalancer(LoadBalancer):
    """
    Advanced load balancer that adapts strategy based on performance.
    """
    
    def __init__(self, **kwargs):
        super().__init__(strategy=LoadBalancingStrategy.ADAPTIVE, **kwargs)
        
        # Adaptive parameters
        self.strategy_performance: Dict[LoadBalancingStrategy, List[float]] = {
            strategy: [] for strategy in LoadBalancingStrategy
        }
        self.current_strategy = LoadBalancingStrategy.ROUND_ROBIN
        self.strategy_evaluation_window = 100  # requests
        self.strategy_switch_threshold = 0.1   # 10% improvement needed
        
        # Learning parameters
        self.learning_enabled = True
        self.exploration_rate = 0.1  # 10% of requests for exploration
    
    async def process_request(self, request_data: Any, 
                            request_context: Dict[str, Any] = None) -> Any:
        """Process request with adaptive strategy selection."""
        # Occasionally explore different strategies
        if (self.learning_enabled and 
            random.random() < self.exploration_rate and 
            len(self.workers) > 1):
            
            original_strategy = self.strategy
            exploration_strategy = random.choice(list(LoadBalancingStrategy))
            self.strategy = exploration_strategy
            
            try:
                start_time = time.time()
                result = await super().process_request(request_data, request_context)
                performance = time.time() - start_time
                
                # Record performance for exploration strategy
                if exploration_strategy not in self.strategy_performance:
                    self.strategy_performance[exploration_strategy] = []
                self.strategy_performance[exploration_strategy].append(performance)
                
                return result
            finally:
                self.strategy = original_strategy
        else:
            # Normal processing with current strategy
            start_time = time.time()
            result = await super().process_request(request_data, request_context)
            performance = time.time() - start_time
            
            # Record performance for current strategy
            if self.current_strategy not in self.strategy_performance:
                self.strategy_performance[self.current_strategy] = []
            self.strategy_performance[self.current_strategy].append(performance)
            
            # Evaluate strategy performance periodically
            if (len(self.strategy_performance[self.current_strategy]) >= 
                self.strategy_evaluation_window):
                self._evaluate_and_adapt_strategy()
            
            return result
    
    def _evaluate_and_adapt_strategy(self) -> None:
        """Evaluate strategy performance and adapt if needed."""
        if not self.learning_enabled:
            return
        
        current_performance = self.strategy_performance[self.current_strategy]
        current_avg = statistics.mean(current_performance[-self.strategy_evaluation_window:])
        
        # Find best performing strategy
        best_strategy = self.current_strategy
        best_performance = current_avg
        
        for strategy, performances in self.strategy_performance.items():
            if len(performances) < 10:  # Need sufficient data
                continue
            
            avg_performance = statistics.mean(performances[-50:])  # Recent performance
            
            if avg_performance < best_performance * (1 - self.strategy_switch_threshold):
                best_strategy = strategy
                best_performance = avg_performance
        
        # Switch strategy if significant improvement found
        if best_strategy != self.current_strategy:
            self.logger.info(f"Switching load balancing strategy from {self.current_strategy.value} "
                           f"to {best_strategy.value} (improvement: "
                           f"{((current_avg - best_performance) / current_avg) * 100:.1f}%)")
            
            self.current_strategy = best_strategy
            self.strategy = best_strategy
            
            # Clear old performance data to allow re-adaptation
            for strategy_perfs in self.strategy_performance.values():
                if len(strategy_perfs) > 200:
                    strategy_perfs[:] = strategy_perfs[-100:]  # Keep recent 100
    
    def get_adaptive_stats(self) -> Dict[str, Any]:
        """Get adaptive load balancer specific statistics."""
        base_stats = self.get_stats()
        
        strategy_stats = {}
        for strategy, performances in self.strategy_performance.items():
            if performances:
                strategy_stats[strategy.value] = {
                    'sample_count': len(performances),
                    'avg_response_time_ms': round(statistics.mean(performances) * 1000, 2),
                    'median_response_time_ms': round(statistics.median(performances) * 1000, 2),
                    'p95_response_time_ms': round(np.percentile(performances, 95) * 1000, 2) if len(performances) > 5 else 0
                }
        
        adaptive_stats = {
            'current_strategy': self.current_strategy.value,
            'learning_enabled': self.learning_enabled,
            'exploration_rate': self.exploration_rate,
            'strategy_performance': strategy_stats
        }
        
        base_stats.update(adaptive_stats)
        return base_stats


# Factory functions
def create_load_balancer(balancer_type: str = "basic", **kwargs) -> LoadBalancer:
    """Create load balancer of specified type."""
    
    if balancer_type == "basic":
        return LoadBalancer(**kwargs)
    elif balancer_type == "adaptive":
        return AdaptiveLoadBalancer(**kwargs)
    else:
        raise ValueError(f"Unknown load balancer type: {balancer_type}")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_load_balancer():
        # Create load balancer
        lb = AdaptiveLoadBalancer(strategy=LoadBalancingStrategy.ADAPTIVE)
        
        # Create test workers
        def worker_func_1(data):
            time.sleep(random.uniform(0.01, 0.05))  # Fast worker
            return f"Worker 1 processed: {data}"
        
        def worker_func_2(data):
            time.sleep(random.uniform(0.02, 0.08))  # Medium worker
            return f"Worker 2 processed: {data}"
        
        def worker_func_3(data):
            time.sleep(random.uniform(0.05, 0.1))   # Slow worker
            return f"Worker 3 processed: {data}"
        
        # Add workers
        lb.add_worker(Worker("worker1", "endpoint1", weight=2.0, processor_func=worker_func_1))
        lb.add_worker(Worker("worker2", "endpoint2", weight=1.5, processor_func=worker_func_2))
        lb.add_worker(Worker("worker3", "endpoint3", weight=1.0, processor_func=worker_func_3))
        
        # Start health checking
        await lb.start_health_checking()
        
        try:
            # Process test requests
            tasks = []
            for i in range(50):
                task = asyncio.create_task(
                    lb.process_request(f"request_{i}", {"priority": random.randint(1, 10)})
                )
                tasks.append(task)
            
            # Wait for all requests to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Print results
            successful = sum(1 for r in results if not isinstance(r, Exception))
            print(f"Processed {successful}/{len(tasks)} requests successfully")
            
            # Print statistics
            stats = lb.get_adaptive_stats()
            print("\nLoad Balancer Statistics:")
            for key, value in stats.items():
                if key != 'workers' and key != 'strategy_performance':
                    print(f"  {key}: {value}")
            
            print("\nWorker Statistics:")
            for worker_stat in stats['workers']:
                print(f"  Worker {worker_stat['id']}: "
                      f"{worker_stat['total_requests']} requests, "
                      f"{worker_stat['success_rate']}% success rate, "
                      f"{worker_stat['avg_response_time_ms']}ms avg response")
        
        finally:
            await lb.stop_health_checking()
    
    asyncio.run(test_load_balancer())